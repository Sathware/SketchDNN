import torch
import math
from typing import Dict, List, Tuple, Any
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from config import NODE_FEATURE_DIMENSION, EDGE_FEATURE_DIMENSION, MAX_NUM_PRIMITIVES
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from dataset1 import SketchDataset
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint
from functools import partial
from IPython import display
import statistics
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def normalize_probs(tensor : Tensor) -> Tensor:
    tensor = torch.clamp(tensor, 1e-24, 1.0) # stave off floating point error nonsense
    return tensor / tensor.sum(dim = -1, keepdim = True)

class CosineNoiseScheduler(nn.Module):
  def __init__(self, max_timestep : int, device : torch.device):
    super().__init__()
    self.device = device
    self.max_timestep = max_timestep
    self.softmax_scale = 4.0
    self.offset = .008 # Fixed offset to improve noise prediction at early timesteps

    # --- Variance Schedule --- #
    # Cosine Beta Schedule Formula: https://arxiv.org/abs/2102.09672     1.00015543316 is 1/a(0), for offset = .008
    self.cumulative_precisions = torch.cos((torch.linspace(0, 1, self.max_timestep).to(self.device) + self.offset) * 0.5 * math.pi / (1 + self.offset)) ** 2 * 1.00015543316
    self.cumulative_variances = 1 - self.cumulative_precisions
    self.variances = torch.cat([torch.Tensor([0]).to(self.device), 1 - (self.cumulative_precisions[1:] / self.cumulative_precisions[:-1])]).clamp(.0001, .9999)
    self.precisions = 1 - self.variances
    self.sqrt_cumulative_precisions = torch.sqrt(self.cumulative_precisions)
    self.sqrt_cumulative_variances = torch.sqrt(self.cumulative_variances)
    self.sqrt_precisions = torch.sqrt(self.precisions)
    self.sqrt_variances = torch.sqrt(self.variances)
    self.sqrt_posterior_variances = torch.cat([torch.Tensor([0]).to(self.device), torch.sqrt(self.variances[1:] * self.cumulative_variances[:-1] / self.cumulative_variances[1:])])

    # --- Probability Distributions --- #
    self.gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor([0.0]), torch.tensor([1.0]))
    self.normal_dist = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

  def forward(self, nodes : Tensor, edges : Tensor, timestep : Tensor):
    ''' Apply noise to graph '''
    noisy_nodes = torch.zeros(size = nodes.size(), device = self.device)
    noisy_edges = torch.zeros(size = edges.size(), device = self.device)
    batch_size, num_nodes, node_dim = nodes.size()
    true_node_noise = torch.zeros(size = (batch_size, num_nodes, node_dim + 1), device = self.device)
    true_edge_noise = torch.zeros(size = edges.size(), device = self.device)
    # nodes = batch_size x num_nodes x NODE_FEATURE_DIMENSION ; edges = batch_size x num_nodes x num_nodes x EDGE_FEATURE_DIMENSION
    bernoulli_is_constructible = nodes[:,:,0] # batch_size x num_nodes x 1
    categorical_primitive_types = nodes[:,:,1:6] # batch_size x num_nodes x 5
    gaussian_primitive_parameters = nodes[:,:,6:] # batch_size x num_nodes x 14
    # subnode just means if the constraint applies to the start, center, or end of a primitive
    categorical_subnode_a_types = edges[:,:,:,0:4] # batch_size x num_nodes x 4
    categorical_subnode_b_types = edges[:,:,:,4:8] # batch_size x num_nodes x 4
    categorical_constraint_types = edges[:,:,:,8:] # batch_size x num_nodes x 9
    # IsConstructible noise
    b, n = bernoulli_is_constructible.size()
    is_construct_noise = self.gumbel_dist.sample((b, n, 2)).to(self.device).squeeze(-1) # standard gumbel noise; (b, n, 2)
    true_node_noise[...,0:2] = is_construct_noise
    noisy_nodes[:,:,0] = self.apply_binary_noise(bernoulli_is_constructible, is_construct_noise, timestep)
    # Primitive Types noise
    prim_type_noise = self.gumbel_dist.sample(categorical_primitive_types.size()).to(self.device).squeeze(-1) # standard gumbel noise; (b, n, 5)
    true_node_noise[...,2:7] = prim_type_noise
    noisy_nodes[:,:,1:6] = self.apply_discrete_noise(categorical_primitive_types, prim_type_noise, timestep) # noised_primitive_types
    # Primitive parameters noise
    parameter_noise = self.normal_dist.sample(gaussian_primitive_parameters.size()).to(self.device).squeeze(-1) # standard gaussian noise; (b, n, 14)
    true_node_noise[:,:,7:] = parameter_noise
    noisy_nodes[:,:,6:] = self.apply_gaussian_noise(gaussian_primitive_parameters, timestep, parameter_noise)
    # Subnode A noise
    suba_type_noise = self.gumbel_dist.sample(categorical_subnode_a_types.size()).to(self.device).squeeze(-1) # standard gumbel noise; (b, n, 4)
    true_edge_noise[...,0:4] = suba_type_noise
    noisy_edges[:,:,:,0:4] = self.apply_discrete_noise(categorical_subnode_a_types, suba_type_noise, timestep) # noised_subnode_a_types
    # Subnode B noise
    subb_type_noise = self.gumbel_dist.sample(categorical_subnode_b_types.size()).to(self.device).squeeze(-1) # standard gumbel noise; (b, n, 4)
    true_edge_noise[...,4:8] = subb_type_noise
    noisy_edges[:,:,:,4:8] = self.apply_discrete_noise(categorical_subnode_b_types, subb_type_noise, timestep) # noised_subnode_a_types
    # Constraint Types noise
    constraint_type_noise = self.gumbel_dist.sample(categorical_constraint_types.size()).to(self.device).squeeze(-1) # standard gumbel noise; (b, n, 9)
    true_edge_noise[...,8:] = constraint_type_noise
    noisy_edges[:,:,:,8:] = self.apply_discrete_noise(categorical_constraint_types, constraint_type_noise, timestep) # noised_constraint_types

    return noisy_nodes, noisy_edges, true_node_noise, true_edge_noise
  
  def get_transition_noise(self, parameters : Tensor, timestep : int, gaussian_noise : Tensor = None):
    if gaussian_noise is None:
      gaussian_noise = torch.randn_like(parameters) # standard gaussian noise
    return self.sqrt_precisions[timestep] * parameters + self.sqrt_variances[timestep] * gaussian_noise
  
  def apply_gaussian_noise(self, parameters : Tensor, timestep : Tensor | int, gaussian_noise : Tensor):
    if type(timestep) is int: timestep = [timestep]
    # parameters shape is batch_size x num_nodes x num_params
    # gaussian_noise shape is batch_size x num_nodes x num_params
    batched_sqrt_precisions = self.sqrt_cumulative_precisions[timestep,None,None] # (b,1,1) or (1,1,1)
    batched_sqrt_variances = self.sqrt_cumulative_variances[timestep,None,None]   # (b,1,1) or (1,1,1)
    return batched_sqrt_precisions * parameters + batched_sqrt_variances * gaussian_noise
  
  def apply_gaussian_posterior_step(self, curr_params : Tensor, pred_noise : Tensor, timestep : int):
    # sqrt_prev_cumul_prec = self.sqrt_cumulative_precisions[timestep - 1]
    var = self.variances[timestep]
    sqrt_prec = self.sqrt_precisions[timestep]
    sqrt_cumul_var = self.sqrt_cumulative_variances[timestep]
    # prev_cumul_var = self.cumulative_variances[timestep - 1]
    # cumul_var = self.cumulative_variances[timestep]
    sqrt_cumul_prec = self.sqrt_cumulative_precisions[timestep]
    
    denoised_mean = (curr_params - pred_noise * var / sqrt_cumul_var) / sqrt_prec
    if timestep > 1:
      # denoised_mean = (sqrt_prev_cumul_prec * var * pred_params + sqrt_prec * prev_cumul_var * curr_params) / cumul_var
      pred_true_params = (curr_params - pred_noise * sqrt_cumul_var) / sqrt_cumul_prec

      gaussian_noise = torch.randn_like(curr_params)
      return denoised_mean + gaussian_noise * self.sqrt_posterior_variances[timestep], pred_true_params
    else:
      return denoised_mean, denoised_mean # denoised_mean
    
  def get_transition_matrix(self, dimension : int, timestep : int | Tensor):
    if type(timestep) is int: assert timestep > 0; timestep = [timestep]
    batched_precisions = self.precisions[timestep,None,None] # (batch_size, 1, 1) or (1, 1, 1)
    if dimension == 5:
      return batched_precisions * torch.eye(dimension, dtype = torch.float32, device = self.device) + (1 - batched_precisions) * torch.tensor([[1.0,1.0,1.0,1.0,1.0]], device = self.device).T @ torch.tensor([[0.3245, 0.0299, 0.0297, 0.0393, 0.5766]], device = self.device) # for nodes use marginal probabilities as stationary distribution
    else:
      return batched_precisions * torch.eye(dimension, dtype = torch.float32, device = self.device) + (1 - batched_precisions) / dimension # (batch_size, d, d) or (1, d, d)
  
  def get_cumulative_transition_matrix(self, dimension : int, timestep : int | Tensor):
    if type(timestep) is int: assert timestep > 0; timestep = [timestep]
    batched_precisions = self.cumulative_precisions[timestep,None,None] # (batch_size, 1, 1) or (1, 1, 1)
    if dimension == 5:
      return batched_precisions * torch.eye(dimension, dtype = torch.float32, device = self.device) + (1 - batched_precisions) * torch.tensor([[1.0,1.0,1.0,1.0,1.0]], device = self.device).T @ torch.tensor([[0.3245, 0.0299, 0.0297, 0.0393, 0.5766]], device = self.device) # for nodes use marginal probabilities as stationary distribution
    else:
      return batched_precisions * torch.eye(dimension, dtype = torch.float32, device = self.device) + (1 - batched_precisions) / dimension # (batch_size, d, d) or (1, d, d)
  
  def get_inverse_cumulative_transition_matrix(self, dimension : int, timestep : int | Tensor):
    if type(timestep) is int: assert timestep > 0; timestep = [timestep]
    batched_inv_precisions = 1.0 / self.cumulative_precisions[timestep,None,None] # (batch_size, 1, 1) or (1, 1, 1)
    if dimension == 5:
      out = batched_inv_precisions * torch.eye(dimension, dtype = torch.float32, device = self.device) + (1 - batched_inv_precisions) * torch.tensor([[1.0,1.0,1.0,1.0,1.0]], device = self.device).T @ torch.tensor([[0.3245, 0.0299, 0.0297, 0.0393, 0.5766]], device = self.device) # for nodes use marginal probabilities as stationary distribution
    else:
      out = batched_inv_precisions * torch.eye(dimension, dtype = torch.float32, device = self.device) + (1 - batched_inv_precisions) / dimension # (batch_size, d, d) or (1, d, d)
    return torch.clamp(out, 0.0, 1.0) # Prevent floating point error nonsense

  def get_posterior_transition_matrix(self, pred_true_probs : Tensor, timestep : int) -> torch.Tensor:
    x_size, pred_true_probs = self.flatten_middle(pred_true_probs) # (b, n, d) or (b, n * n, d), for convenience let m = n or n * n
    d = x_size[-1]
    qt = self.get_transition_matrix(d, timestep) # element at [i, j] = p(x_t = j | x_t-1 = i); (1, d, d)
    qt_bar = self.get_cumulative_transition_matrix(d, timestep) # element at [i, j] = p(x_t = j | x_0 = i); (1, d, d)
    qt_1bar = self.get_cumulative_transition_matrix(d, timestep - 1) # element at [i, j] = p(x_t-1 = j | x_0 = i); (1, d, d)

    cond_xt_x0_probs = (qt.permute(0, 2, 1).unsqueeze(1) * qt_1bar.unsqueeze(2) / qt_bar.unsqueeze(3)).squeeze(0) # (d, d, d) where element at [i, j, k] = p(x_t-1 = k | x_t = j, x_0 = i)
    cond_xt_probs = torch.einsum('bij,jkt->bikt', pred_true_probs, cond_xt_x0_probs) # (b, m, d, d) where element at [i, j] = posterior transition matrix to get x_t-1 given x_t

    # qt = xt @ self.get_transition_matrix(d, timestep).permute(0, 2, 1) # (b, m, d), since xt is onehot we are plucking out rows corresponding to p(x_t = class | x_(t-1))
    # qt_bar = xt @ self.get_cumulative_transition_matrix(d, timestep).permute(0, 2, 1) # (b, m, d), since xt is onehot we are plucking out rows corresponding to p(x_t = class | x_0)
    # q = qt.unsqueeze(2) / qt_bar.unsqueeze(3) # (b, m, d, d), perform an outer product so element at (b, m, i, j) = p(x_t = class | x_(t-1) = j) / p(x_t = class | x_0 = i)
    # q = q * self.get_cumulative_transition_matrix(d, timestep - 1).unsqueeze(1) # (b, m, d, d), broadcast multiply so element at (b, m, i, j) = p(x_t = class | x_(t-1) = j) * p(x_(t-1) = j | x_0 = i) / p(x_t = class | x_0 = i)

    return cond_xt_probs.view(size = x_size + (d,)) # reshape into (b, n, d, d) or (b, n, n, d, d)
  
  def apply_discrete_noise(self, x_one_hot : Tensor, gumbel_noise : Tensor, timestep : Tensor | int):
    size, x = self.flatten_middle(x_one_hot)
    q = self.get_cumulative_transition_matrix(size[-1], timestep) # (b, d, d) or (1, d, d)
    distribution = x @ q # (b, n, d) or (b, n * n, d)
    distribution = distribution.view(size) # (b, n, d) or (b, n, n, d)
    return self.sample_discrete_distribution(distribution, gumbel_noise)
  
  def apply_multinomial_posterior_step(self, class_probs : Tensor, pred_noise : Tensor, timestep : int):
    # class_probs and pred_noise = (b, n, d) or (b, n, n, d)
    # m = (n) or (n, n)
    d = class_probs.size(-1)
    class_probs = normalize_probs(class_probs) # Avoid floating point error nonsense

    log_probs = class_probs.log()

    pred_true_probs = torch.exp((log_probs - log_probs[...,-1,None]) / self.softmax_scale - pred_noise + pred_noise[...,-1,None]) # b, m, d
    pred_true_probs = pred_true_probs / pred_true_probs.sum(-1, keepdim = True)
    pred_true_probs = pred_true_probs @ self.get_inverse_cumulative_transition_matrix(d, timestep)

    pred_true_probs = normalize_probs(pred_true_probs)

    if timestep > 1:
      q = self.get_posterior_transition_matrix(pred_true_probs, timestep) # (b, n, d, d) or (b, n, n, d, d)
      class_probs = class_probs.unsqueeze(-2) # (b, n, 1, d) or (b, n, n, 1, d), make probs into row vector
      posterior_distribution = class_probs @ q # (b, n, 1, d) or (b, n, n, 1, d), batched vector-matrix multiply
      posterior_distribution = posterior_distribution.squeeze(-2) # (b, n, d) or (b, n, n, d)
      new_noise = self.gumbel_dist.sample(pred_noise.size()).to(self.device).squeeze(-1)
      return self.sample_discrete_distribution(posterior_distribution, new_noise), pred_true_probs
    else:
      return pred_true_probs, pred_true_probs
    
  def apply_binary_noise(self, boolean_flag : Tensor, gumbel_noise : Tensor, timestep : int | Tensor):
    boolean_flag = boolean_flag.unsqueeze(-1)
    one_hot = torch.cat([1 - boolean_flag, boolean_flag], dim = -1) # (b, n, 2)
    noised_one_hot = self.apply_discrete_noise(one_hot, gumbel_noise, timestep) # (b, n, 2)
    return noised_one_hot[...,1] # (b, n)
  
  def apply_bernoulli_posterior_step(self, boolean_prob : Tensor, pred_noise : Tensor, timestep : int):
    boolean_prob = boolean_prob.unsqueeze(-1) # b, n, 1
    class_probs = torch.cat([1 - boolean_prob, boolean_prob], dim = -1) # (b, n, 2)
    # pred_noise is shape (b, n, 2)
    new_probs, pred_true_probs = self.apply_multinomial_posterior_step(class_probs, pred_noise, timestep) # (b, n, 2)
    return new_probs[...,1], pred_true_probs[...,1] # (b, n)
  
    # if timestep > 1:
    #   boolean_prob = boolean_prob.unsqueeze(-1) # b, n, 1
    #   class_probs = torch.cat([1 - boolean_prob, boolean_prob], dim = -1) # (b, n, 2)
    #   # pred_noise is shape (b, n, 2)
    #   new_probs = self.apply_multinomial_posterior_step(class_probs, pred_noise, timestep) # (b, n, 2)
    #   return new_probs[...,1] # (b, n)
    # else:
    #   return pred_boolean_prob
  
  def sample_discrete_distribution(self, class_probs : Tensor, gumbel_vals : Tensor):
    '''Performs Gumbel-Softmax'''
    # (b, n, d) or (b, n, n, d) is shape of class_probs and gumbel_vals
    class_probs = normalize_probs(class_probs)
    log_probs = class_probs.log()
    
    temp = log_probs + gumbel_vals - (log_probs[...,-1,None] + gumbel_vals[...,-1,None])
    pseudo_one_hot = F.softmax(self.softmax_scale * temp, -1)
    out = normalize_probs(pseudo_one_hot)
    return out
    # size = tensor.size()
    # num_classes = size[-1]
    # return F.one_hot(tensor.reshape(-1, num_classes).multinomial(1), num_classes).reshape(size)
  
  def flatten_middle(self, x : Tensor):
    prev_size = x.size() # shape of x_one_hot is (b, n, d) or (b, n, n, d)
    return prev_size, x.view(prev_size[0], -1, prev_size[-1]) # (b, n, d) or (b, n * n, d)
  
  def get_pred_probs(self, class_probs : Tensor, pred_noise : Tensor, timestep : Tensor):
    d = class_probs.size(-1)
    class_probs = normalize_probs(class_probs) # Avoid floating point error nonsense

    log_probs = class_probs.log()

    pred_true_probs = torch.exp((log_probs - log_probs[...,-1,None]) / self.softmax_scale - pred_noise + pred_noise[...,-1,None]) # b, m, d
    pred_true_probs = pred_true_probs / pred_true_probs.sum(-1, keepdim = True) # b, m, d
    pred_true_probs = pred_true_probs @ self.get_inverse_cumulative_transition_matrix(d, timestep) # (b, m, d) @ (b, d, d) = (b, m, d)

    # pred_true_probs = normalize_probs(pred_true_probs)
    return pred_true_probs

class TimeEmbedder(nn.Module):
  def __init__(self, max_timestep : int, embedding_dimension : int, device : torch.device):
    super().__init__()
    self.device = device
    self.embed_dim = embedding_dimension
    self.max_steps = max_timestep + 1
    self.max_timestep = max_timestep
      
    timesteps = torch.arange(self.max_steps, device = self.device).unsqueeze(1) # num_timesteps x 1
    scales = torch.exp(torch.arange(0, self.embed_dim, 2, device = self.device) * (-math.log(10000.0) / self.embed_dim)).unsqueeze(0) # 1 x (embedding_dimension // 2)
    self.time_embs = torch.zeros(self.max_steps, self.embed_dim, device = self.device) # num_timesteps x embedding_dimension
    self.time_embs[:, 0::2] = torch.sin(timesteps * scales) # fill even columns with sin(timestep * 1000^-(2*i/embedding_dimension))
    self.time_embs[:, 1::2] = torch.cos(timesteps * scales) # fill odd columns with cos(timestep * 1000^-(2*i/embedding_dimension))
      
  def forward(self, timestep : Tensor):
    return self.time_embs[timestep] # batch_size x embedding_dimension

'''Layer Norm and Additive Skip Connection'''
class SkipLayerNorm(nn.Module):
  def __init__(self, dim : int, device : torch.device):
    super().__init__()

    # self.layer_norm = nn.LayerNorm(normalized_shape = dim, device = device, elementwise_affine = False, bias = False)
    self.layer_norm = nn.LayerNorm(normalized_shape = dim, device = device)

  def forward(self, A, B):
    return F.leaky_relu(self.layer_norm(A + B), 0.1)

class FiLM2(nn.Module):
    def __init__(self, dim_a : int, dim_b : int, device : torch.device):
        super().__init__()

        self.lin_mul = nn.Linear(in_features = dim_b, out_features = dim_a, device = device)
        self.lin_add = nn.Linear(in_features = dim_b, out_features = dim_a, device = device)

        self.mlp_out = nn.Sequential(
            nn.Linear(in_features = dim_a, out_features = dim_a, device = device),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features = dim_a, out_features = dim_a, device = device),
            nn.LeakyReLU(0.1)
        )
    
    def forward(self, A, B):
        # A is shape (b, ..., dim_a)
        # B is shape (b, *, dim_b)
        mul = self.lin_mul(B) # (b, *, dim_a)
        add = self.lin_add(B) # (b, *, dim_a)

        return self.mlp_out(A * mul + add + A)
    

class SoftAttention2(nn.Module):
    def __init__(self, in_dim : int, out_dim, device : torch.device):
        super().__init__()

        self.lin_weights = nn.Sequential(
            nn.Linear(in_features = in_dim, out_features = in_dim, device = device),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features = in_dim, out_features = 1, device = device)
        )
        self.lin_values = nn.Sequential(
            nn.Linear(in_features = in_dim, out_features = out_dim, device = device),
            # nn.LeakyReLU(0.1),
            # nn.Linear(in_features = out_dim, out_features = out_dim, device = device)
        )
        self.lin_out = nn.Sequential(
            nn.Linear(in_features = out_dim, out_features = out_dim, device = device),
            nn.LeakyReLU(0.1),
            # nn.Linear(in_features = out_dim, out_features = out_dim, device = device),
            # nn.LeakyReLU(0.1)
        )
    
    def forward(self, M):
        # M is shape (b, *, dim)
        weights = self.lin_weights(M) # (b, *, 1)
        weights = F.softmax(input = weights.squeeze(-1), dim = -1).unsqueeze(-1)
        values = self.lin_values(M) # (b, *, dim)

        # The output will have one less dimension
        # batched matrix multiply results in (b, ..., 1, out_dim), then squeeze makes it (b, ..., out_dim)
        out = (weights.transpose(-2, -1) @ values).squeeze(-2)
        return self.lin_out(out) # (b,...,out_dim)

class AttentionBlock2(nn.Module):
    def __init__(self, class_dim : int, param_dim : int, edge_dim : int, time_dim : int, num_heads : int, device : torch.device):
        super().__init__()
        self.num_heads = num_heads
        self.class_dim = class_dim
        self.param_dim = param_dim

        self.mlp_time_in = nn.Sequential(
            nn.Linear(in_features = time_dim, out_features = time_dim, device = device),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features = time_dim, out_features = time_dim, device = device),
            nn.LeakyReLU(0.1)
        )

        self.film_class_time = FiLM2(dim_a = class_dim, dim_b = time_dim, device = device)
        self.film_param_time = FiLM2(dim_a = param_dim, dim_b = time_dim, device = device)
        self.film_edges_time = FiLM2(dim_a = edge_dim, dim_b = time_dim, device = device)
        
        self.edge_soft_attn = SoftAttention2(in_dim = edge_dim, out_dim = edge_dim, device = device)

        self.film_class_edge = FiLM2(dim_a = class_dim, dim_b = edge_dim, device = device)
        self.film_param_edge = FiLM2(dim_a = param_dim, dim_b = edge_dim, device = device)

        self.lin_class_qkv = nn.Sequential(
            nn.Linear(in_features = class_dim, out_features = 3 * class_dim, device = device),
        )
        self.lin_param_qkv = nn.Sequential(
            nn.Linear(in_features = param_dim, out_features = 3 * param_dim, device = device),
        )
        self.film_param_class = FiLM2(dim_a = 3 * param_dim, dim_b = 3 * class_dim, device = device)

        self.film_edge_node = FiLM2(dim_a = edge_dim, dim_b = 2 * class_dim, device = device)

        self.film_class_param = FiLM2(dim_a = class_dim, dim_b = param_dim, device = device)

        self.film_class_time1 = FiLM2(dim_a = class_dim, dim_b = time_dim, device = device)
        self.film_param_time1 = FiLM2(dim_a = param_dim, dim_b = time_dim, device = device)

        self.lin_class_out = nn.Sequential(
            nn.Linear(in_features = class_dim, out_features = class_dim, device = device),
            nn.LeakyReLU(0.1),
            # nn.Linear(in_features = class_dim, out_features = class_dim, device = device),
            # nn.LeakyReLU(0.1),
        )
        self.lin_param_out = nn.Sequential(
            nn.Linear(in_features = param_dim, out_features = param_dim, device = device),
            nn.LeakyReLU(0.1),
            # nn.Linear(in_features = param_dim, out_features = param_dim, device = device),
            # nn.LeakyReLU(0.1),
        )
        self.lin_edges_out = nn.Sequential(
            nn.Linear(in_features = edge_dim, out_features = edge_dim, device = device),
            nn.LeakyReLU(0.1),
            # nn.Linear(in_features = edge_dim, out_features = edge_dim, device = device),
            # nn.LeakyReLU(0.01),
        )

        self.skip_class = SkipLayerNorm(dim = class_dim, device = device)
        self.skip_param = SkipLayerNorm(dim = param_dim, device = device)
        self.skip_edges = SkipLayerNorm(dim = edge_dim, device = device)
        self.skip_times = SkipLayerNorm(dim = time_dim, device = device)

    def forward(self, classes : Tensor, params : Tensor, edges : Tensor, times : Tensor):
        old_classes = classes
        old_params = params
        old_edges = edges
        old_times = times

        times = self.mlp_time_in(times)
        
        classes = self.film_class_time(classes, times[:,None,:])      # batch_size, num_nodes, node_dim
        params = self.film_param_time(params, times[:,None,:])      # batch_size, num_nodes, node_dim
        edges = self.film_edges_time(edges, times[:,None,None,:]) # batch_size, num_nodes, num_nodes, edge_dim

        edge_aggrs = self.edge_soft_attn(edges)

        classes = self.film_class_edge(classes, edge_aggrs)
        params = self.film_param_edge(params, edge_aggrs)

        class_qkv = self.lin_class_qkv(classes) # batch_size, num_nodes, attn_dim * (2 * key_query_dim + value_dim)
        param_qkv = self.lin_param_qkv(params) # batch_size, num_nodes, attn_dim * (2 * key_query_dim + value_dim)

        param_qkv = self.film_param_class(param_qkv, class_qkv)

        # New params
        b, n, _ = params.size() # get batchsize and number of nodes
        queries, keys, values = param_qkv.split(split_size = (self.param_dim, self.param_dim, self.param_dim), dim = -1)
        queries = queries.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3) # batch_size, num_heads, num_nodes, attn_dim
        keys = keys.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)       # batch_size, num_heads, num_nodes, attn_dim
        values = values.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)   # batch_size, num_heads, num_nodes, attn_val_dim

        attn_vals = F.scaled_dot_product_attention(query = queries, key = keys, value = values)
        attn_vals = attn_vals.permute(0, 2, 1, 3).flatten(start_dim = 2)  # batch_size, num_nodes, num_heads * attn_val_dim
        temp_param_vals = attn_vals

        new_params = self.film_param_time1(attn_vals, times[:,None,:]) # batch_size, num_nodes, num_heads * attn_val_dim
        new_params = self.lin_param_out(new_params)          # batch_size, num_nodes, node_dim
        # new_params = self.lin_param_out(attn_vals)
        new_params = self.skip_param(new_params, old_params)                     # batch_size, num_nodes, node_dim

        # New classes
        b, n, _ = classes.size() # get batchsize and number of nodes
        queries, keys, values = class_qkv.split(split_size = (self.class_dim, self.class_dim, self.class_dim), dim = -1)
        queries = queries.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3) # batch_size, num_heads, num_nodes, attn_dim
        keys = keys.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)       # batch_size, num_heads, num_nodes, attn_dim
        values = values.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)   # batch_size, num_heads, num_nodes, attn_val_dim

        attn_vals = F.scaled_dot_product_attention(query = queries, key = keys, value = values)
        attn_vals = attn_vals.permute(0, 2, 1, 3).flatten(start_dim = 2)  # batch_size, num_nodes, num_heads * attn_val_dim

        new_classes = self.film_class_param(attn_vals, temp_param_vals)
        new_classes = self.film_class_time1(new_classes, times[:,None,:])
        new_classes = self.lin_class_out(new_classes)
        # new_classes = self.lin_class_out(attn_vals)
        new_classes = self.skip_class(new_classes, old_classes)

        # New edges
        # edge_summary = self.film_edge_node(attn_vals, edge_aggrs)
        # hstack = edge_summary[:,:,None,:].expand(-1, -1, n, -1)    # batch_size, num_nodes, num_nodes, num_heads * attn_val_dim
        # vstack = hstack.transpose(2, 1)                         # batch_size, num_nodes, num_nodes, num_heads * attn_val_dim
        # outer_concat = torch.cat([hstack, vstack], dim = -1)    # batch_size, num_nodes, num_nodes, 2 * num_heads * attn_val_dim
        outer_concat = torch.cat([attn_vals.unsqueeze(1).expand(-1, n, -1, -1), attn_vals.unsqueeze(2).expand(-1, -1, n, -1)], dim = -1)
        new_edges = self.film_edge_node(edges, outer_concat)
        new_edges = self.lin_edges_out(new_edges)             # batch_size, num_nodes, num_nodes, edge_dim
        new_edges = self.skip_edges(new_edges, old_edges)     # batch_size, num_nodes, num_nodes, edge_dim

        # New times
        new_times = self.skip_times(times, old_times)

        return new_classes, new_params, new_edges, new_times


class AttentionBlock4(nn.Module):
  def __init__(self, node_dim : int, edge_dim : int, time_dim : int, num_heads : int, device : torch.device):
    super().__init__()
    self.num_heads = num_heads
    self.node_dim = node_dim

    self.lin_time_in = nn.Sequential(
        nn.Linear(in_features = time_dim, out_features = time_dim, device = device),
        nn.LeakyReLU(0.1),
        # nn.Linear(in_features = time_dim, out_features = time_dim, device = device),
        # nn.LeakyReLU(0.1)
    )

    self.film_nodes_time = FiLM2(dim_a = node_dim, dim_b = time_dim, device = device)
    self.film_edges_time = FiLM2(dim_a = edge_dim, dim_b = time_dim, device = device)

    self.lin_node_qkv = nn.Sequential(
        nn.Linear(in_features = node_dim, out_features = 3 * node_dim, device = device),
    )
    self.film_kq_edges = FiLM2(dim_a = node_dim, dim_b = edge_dim, device = device) 
    self.lin_edge_attn = nn.Sequential(
        nn.Linear(in_features = node_dim, out_features = edge_dim, device = device),
    )

    # self.lin_time_out = nn.Sequential(
    #     nn.Linear(in_features = time_dim, out_features = time_dim, device = device),
    #     nn.LeakyReLU(0.1),
    #     # nn.Linear(in_features = time_dim, out_features = time_dim, device = device),
    #     # nn.LeakyReLU(0.1)
    # )
    self.film_node_time1 = FiLM2(dim_a = node_dim, dim_b = time_dim, device = device)
    self.film_edge_time1 = FiLM2(dim_a = edge_dim, dim_b = time_dim, device = device)

    self.lin_nodes_out = nn.Sequential(
        nn.Linear(in_features = node_dim, out_features = node_dim, device = device),
        nn.LeakyReLU(0.1),
        # nn.Linear(in_features = class_dim, out_features = class_dim, device = device),
        # nn.LeakyReLU(0.1),
    )
  
    self.lin_edges_out = nn.Sequential(
        nn.Linear(in_features = edge_dim, out_features = edge_dim, device = device),
        nn.LeakyReLU(0.1),
        # nn.Linear(in_features = edge_dim, out_features = edge_dim, device = device),
        # nn.LeakyReLU(0.01),
    )

    self.skip_nodes = SkipLayerNorm(dim = node_dim, device = device)
    self.skip_edges = SkipLayerNorm(dim = edge_dim, device = device)
    self.skip_times = SkipLayerNorm(dim = time_dim, device = device)

  def forward(self, nodes : Tensor, edges : Tensor, times : Tensor):
    old_nodes = nodes
    old_edges = edges
    old_times = times

    times = self.lin_time_in(times)
        
    nodes = self.film_nodes_time(nodes, times[:,None,:])      # batch_size, num_nodes, node_dim
    edges = self.film_edges_time(edges, times[:,None,None,:]) # batch_size, num_nodes, num_nodes, edge_dim

    # Node Self Attention
    b, n, _ = nodes.size() # get batchsize and number of nodes

    node_qkv = self.lin_node_qkv(nodes) # batch_size, num_nodes, attn_dim * (2 * key_query_dim + value_dim)
    queries, keys, values = node_qkv.split(split_size = (self.node_dim, self.node_dim, self.node_dim), dim = -1)

    weights = self.film_kq_edges(keys.unsqueeze(2) * queries.unsqueeze(1), edges)
    weights = weights.reshape(b, self.num_heads, n, n, -1)
    values = values.reshape(b, self.num_heads, n, -1)  # batch_size, num_heads, num_nodes, attn_val_dim

    attn_vals = torch.softmax(weights.sum(dim = -1), dim = -1) @ values
    attn_vals = attn_vals.permute(0, 2, 1, 3).flatten(start_dim = 2)  # batch_size, num_nodes, num_heads * attn_val_dim

    # New times
    # times = self.lin_time_out(times)
    new_times = self.skip_times(times, old_times)

    # New Nodes
    new_nodes = self.film_node_time1(attn_vals, times[:,None,:])
    new_nodes = self.lin_nodes_out(new_nodes)
    new_nodes = self.skip_nodes(new_nodes, old_nodes)

    # New edges
    new_edges = self.lin_edge_attn(weights.permute(0, 2, 3, 1, 4).flatten(start_dim = 3))
    new_edges = self.film_edge_time1(new_edges, times[:,None,None,:])
    new_edges = self.lin_edges_out(new_edges)             # batch_size, num_nodes, num_nodes, edge_dim
    new_edges = self.skip_edges(new_edges, old_edges)     # batch_size, num_nodes, num_nodes, edge_dim

    return new_nodes, new_edges, new_times

class TimeConditioningBlock(nn.Module):
  def __init__(self, class_dim : int, param_dim : int, edge_dim : int, time_dim : int, device : torch.device):
    super().__init__()
    self.film_class_time = FiLM2(dim_a = class_dim, dim_b = time_dim, device = device)
    self.film_param_time = FiLM2(dim_a = param_dim, dim_b = time_dim, device = device)
    self.film_edge_time = FiLM2(dim_a = edge_dim, dim_b = time_dim, device = device)
  
  def forward(self, classes : Tensor, params : Tensor, edges : Tensor, times : Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    return self.film_class_time(classes, times[:,None,:]), self.film_param_time(params, times[:,None,:]), self.film_edge_time(edges, times[:,None,None,:])

class SkipNormBlock(nn.Module):
  def __init__(self, class_dim : int, param_dim : int, edge_dim : int, device : torch.device):
    super().__init__()
    self.skip_class = SkipLayerNorm(dim = class_dim, device = device)
    self.skip_param = SkipLayerNorm(dim = param_dim, device = device)
    self.skip_edges = SkipLayerNorm(dim = edge_dim, device = device)
  
  def forward(self, classes : Tensor, params : Tensor, edges : Tensor, old_classes : Tensor, old_params : Tensor, old_edges : Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    return self.skip_class(classes, old_classes), self.skip_param(params, old_params), self.skip_edges(edges, old_edges)

class CrossAttentionBlock(nn.Module):
  def __init__(self, class_dim : int, param_dim : int, edge_dim : int, num_heads : int, device : torch.device):
    super().__init__()
    self.num_heads = num_heads
    self.class_dim = class_dim
    self.param_dim = param_dim

    self.lin_class_qkv = nn.Sequential(
      nn.Linear(in_features = class_dim, out_features = 3 * class_dim, device = device),
    )
    self.lin_param_qkv = nn.Sequential(
      nn.Linear(in_features = param_dim, out_features = 3 * param_dim, device = device),
    )

    self.lin_edge_class_mul = nn.Linear(in_features = edge_dim, out_features = class_dim, device = device)
    self.lin_edge_class_add = nn.Linear(in_features = edge_dim, out_features = class_dim, device = device)
    self.lin_edge_param_mul = nn.Linear(in_features = edge_dim, out_features = param_dim, device = device)
    self.lin_edge_param_add = nn.Linear(in_features = edge_dim, out_features = param_dim, device = device)
    # self.film_class_edge = FiLM2(dim_a = class_dim, dim_b = edge_dim, device = device)
    # self.film_param_edge = FiLM2(dim_a = param_dim, dim_b = edge_dim, device = device)

    self.lin_class_out = nn.Sequential(
      nn.Linear(in_features = class_dim, out_features = class_dim, device = device),
    )
    self.lin_param_out = nn.Sequential(
      nn.Linear(in_features = param_dim, out_features = param_dim, device = device),
    )
    self.lin_edge_out = nn.Sequential(
      nn.Linear(in_features = class_dim + param_dim, out_features = edge_dim, device = device),
    )

  def forward(self, classes : Tensor, params : Tensor, edges : Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    b, n, _ = params.size() # get batchsize and number of nodes, same size for classes as well

    Qc, Kc, Vc = self.lin_class_qkv(classes).chunk(chunks = 3, dim = -1)
    Qp, Kp, Vp = self.lin_param_qkv(params).chunk(chunks = 3, dim = -1)

    # Outer Product Attention -------
    Qc = Qc.view(b, n, self.num_heads, -1) # batch_size x num_nodes x num_heads x attn_dim
    Kc = Kc.view(b, n, self.num_heads, -1) # batch_size x num_nodes x num_heads x attn_dim
    Qp = Qp.view(b, n, self.num_heads, -1) # batch_size x num_nodes x num_heads x attn_dim
    Kp = Kp.view(b, n, self.num_heads, -1) # batch_size x num_nodes x num_heads x attn_dim

    attn_class = Qc.unsqueeze(2) * Kc.unsqueeze(1) / math.sqrt(self.class_dim) # batch_size x num_nodes x num_nodes x num_heads x attn_dim
    del Qc
    del Kc
    attn_param = Qp.unsqueeze(2) * Kp.unsqueeze(1) / math.sqrt(self.param_dim) # batch_size x num_nodes x num_nodes x num_heads x attn_dim
    del Qp
    del Kp

    # Condition attention based on edge features
    attn_class = attn_class * self.lin_edge_class_mul(edges).reshape(b, n, n, self.num_heads, -1) + attn_class + self.lin_edge_class_add(edges).reshape(b, n, n, self.num_heads, -1)
    attn_param = attn_param * self.lin_edge_param_mul(edges).reshape(b, n, n, self.num_heads, -1) + attn_param + self.lin_edge_param_add(edges).reshape(b, n, n, self.num_heads, -1)
    #attn_class = self.film_class_edge(attn_class, edges) # batch_size x num_nodes x num_nodes x num_heads x attn_dim
    #attn_param = self.film_param_edge(attn_param, edges) # batch_size x num_nodes x num_nodes x num_heads x attn_dim

    new_edges = torch.cat((attn_class, attn_param), dim = -1) # batch_size x num_nodes x num_nodes x num_heads x attn_dim
        
    # Normalize attention
    attn_class = torch.softmax(input = attn_class.sum(dim = 4), dim = 2) # batch_size x num_nodes x num_nodes x num_heads (Finish dot product & softmax)
    attn_param = torch.softmax(input = attn_param.sum(dim = 4), dim = 2) # batch_size x num_nodes x num_nodes x num_heads (Finish dot product & softmax)

    # Cross Attention ; Weight node representations and sum --------
    Vc = Vc.view(b, n, self.num_heads, -1) # batch_size x num_nodes x num_heads x attn_dim
    del classes
    Vp = Vp.view(b, n, self.num_heads, -1) # batch_size x num_nodes x num_heads x attn_dim
    del params

                                                                                                        # batch_size x num_nodes x num_heads x attn_dim
    weighted_classes = (attn_param.unsqueeze(4) * Vc.unsqueeze(1)).sum(dim = 2).flatten(start_dim = 2)  # batch_size x num_nodes x node_dim
    del Vc
                                                                                                      # batch_size x num_nodes x num_heads x attn_dim
    weighted_params = (attn_class.unsqueeze(4) * Vp.unsqueeze(1)).sum(dim = 2).flatten(start_dim = 2) # batch_size x num_nodes x node_dim
    del Vp

    # Flatten attention heads
    new_edges = new_edges.flatten(start_dim = 3)
        
    # Combine attention heads
    return self.lin_class_out(weighted_classes), self.lin_param_out(weighted_params), self.lin_edge_out(new_edges)
  
class CrossTransformerLayer(nn.Module):
    def __init__(self, class_dim : int, param_dim : int, edge_dim : int, time_dim : int, num_heads : int, device : torch.device):
        super().__init__()

        self.lin_time_in = nn.Sequential(
            nn.Linear(in_features = time_dim, out_features = time_dim, device = device),
            nn.LeakyReLU(0.1),
        )

        self.time_cond1 = TimeConditioningBlock(class_dim = class_dim, param_dim = param_dim, edge_dim = edge_dim, time_dim = time_dim, device = device)
        
        self.skip1 = SkipNormBlock(class_dim = class_dim, param_dim = param_dim, edge_dim = edge_dim, device = device)

        self.cross_attn = CrossAttentionBlock(class_dim = class_dim, param_dim = param_dim, edge_dim = edge_dim, num_heads = num_heads, device = device)

        self.skip2 = SkipNormBlock(class_dim = class_dim, param_dim = param_dim, edge_dim = edge_dim, device = device)

        self.lin_time_out = nn.Sequential(
            nn.Linear(in_features = time_dim, out_features = time_dim, device = device),
            nn.LeakyReLU(0.1),
        )

        self.time_cond2 = TimeConditioningBlock(class_dim = class_dim, param_dim = param_dim, edge_dim = edge_dim, time_dim = time_dim, device = device)

        self.skip3 = SkipNormBlock(class_dim = class_dim, param_dim = param_dim, edge_dim = edge_dim, device = device)

        self.mlp_class_out = nn.Sequential(
            nn.Linear(in_features = class_dim, out_features = class_dim, device = device),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features = class_dim, out_features = class_dim, device = device),
            nn.LeakyReLU(0.1),
        )
        self.mlp_param_out = nn.Sequential(
            nn.Linear(in_features = param_dim, out_features = param_dim, device = device),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features = param_dim, out_features = param_dim, device = device),
            nn.LeakyReLU(0.1),
        )
        self.mlp_edges_out = nn.Sequential(
            nn.Linear(in_features = edge_dim, out_features = edge_dim, device = device),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features = edge_dim, out_features = edge_dim, device = device),
            nn.LeakyReLU(0.01),
        )

        self.skip4 = SkipNormBlock(class_dim = class_dim, param_dim = param_dim, edge_dim = edge_dim, device = device)
        self.skip_times = nn.LayerNorm(normalized_shape = time_dim, device = device, elementwise_affine = False, bias = False)

    def forward(self, classes : Tensor, params : Tensor, edges : Tensor, times : Tensor):
      old_classes = classes
      old_params = params
      old_edges = edges
      old_times = times

      # Time Conditioning
      times = self.lin_time_in(times)
      classes, params, edges = self.time_cond1(classes, params, edges, times)

      # Skip Connection
      classes, params, edges = self.skip1(classes, params, edges, old_classes, old_params, old_edges)
      old_classes = classes
      old_params = params
      old_edges = edges

      # Cross Attention
      classes, params, edges = self.cross_attn(classes, params, edges)

      # Skip Connection
      classes, params, edges = self.skip2(classes, params, edges, old_classes, old_params, old_edges)
      old_classes = classes
      old_params = params
      old_edges = edges

      # Time Conditioning
      times = self.lin_time_out(times)
      classes, params, edges = self.time_cond2(classes, params, edges, times)

      # Skip Connection
      classes, params, edges = self.skip3(classes, params, edges, old_classes, old_params, old_edges)
      old_classes = classes
      old_params = params
      old_edges = edges

      classes = self.mlp_class_out(classes)
      params = self.mlp_param_out(params)
      edges = self.mlp_edges_out(edges)

      # Skip Connection
      new_classes, new_params, new_edges = self.skip4(classes, params, edges, old_classes, old_params, old_edges)
      new_times = self.skip_times(times + old_times)

      return new_classes, new_params, new_edges, new_times

class GD3PM(nn.Module):
  def __init__(self, device : torch.device):
    super().__init__()
    self.device = device
    self.node_dim = NODE_FEATURE_DIMENSION
    self.edge_dim = EDGE_FEATURE_DIMENSION
    self.node_hidden_dim = 256# hidden_node
    self.edge_hidden_dim = 256# hidden_edge
    self.time_hidden_dim = 256# hidden_time
    self.num_tf_layers = 24# num_layers
    self.num_checkpoints = 22
    self.num_heads = 16
    self.max_timestep = 500
    self.noise_scheduler = CosineNoiseScheduler(self.max_timestep, self.device)
    self.time_embedder = TimeEmbedder(self.max_timestep, self.time_hidden_dim, self.device)

    self.class_dim = self.node_hidden_dim
    self.param_dim = self.node_hidden_dim
    self.mlp_in_classes = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.node_hidden_dim, device = device),
                                      nn.LeakyReLU(0.1),
                                      nn.Linear(in_features = self.node_hidden_dim, out_features = self.class_dim, device = device),
                                      nn.LeakyReLU(0.1),
                                     )
    self.mlp_in_params = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.node_hidden_dim, device = device),
                                      nn.LeakyReLU(0.1),
                                      nn.Linear(in_features = self.node_hidden_dim, out_features = self.param_dim, device = device),
                                      nn.LeakyReLU(0.1),
                                     )
    # self.mlp_in_nodes = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.node_hidden_dim, device = device),
    #                                   nn.LeakyReLU(0.1),
    #                                   nn.Linear(in_features = self.node_hidden_dim, out_features = self.node_hidden_dim, device = device),
    #                                   nn.LeakyReLU(0.1),
    #                                  )
    self.mlp_in_edges = nn.Sequential(nn.Linear(in_features = self.edge_dim, out_features = self.edge_hidden_dim, device = device),
                                      nn.LeakyReLU(0.1),
                                      nn.Linear(in_features = self.edge_hidden_dim, out_features = self.edge_hidden_dim, device = device),
                                      nn.LeakyReLU(0.1),
                                     )
    
    self.block_layers = nn.ModuleList([CrossTransformerLayer(class_dim = self.class_dim,
                                                       param_dim = self.param_dim, 
                                                       edge_dim = self.edge_hidden_dim, 
                                                       time_dim = self.time_hidden_dim, 
                                                       num_heads = self.num_heads, 
                                                       device = self.device)
                                      for i in range(self.num_tf_layers)])
    
    self.mlp_out_nodes = nn.Sequential(nn.Linear(in_features = self.class_dim + self.param_dim, out_features = self.class_dim + self.param_dim, device = device),
                                       nn.LeakyReLU(0.1),
                                       nn.Linear(in_features = self.class_dim + self.param_dim, out_features = self.node_dim + 1, device = device) # The extra 1 is there since we need 2 gumbel noise values for isconstructible, which means output size is one bigger
                                      )
    self.mlp_out_edges = nn.Sequential(nn.Linear(in_features = self.edge_hidden_dim, out_features = self.edge_hidden_dim, device = device),
                                       nn.LeakyReLU(0.1),
                                       nn.Linear(in_features = self.edge_hidden_dim, out_features = self.edge_dim, device = device)
                                      )

  def forward(self, nodes : Tensor, edges : Tensor, timestep : Tensor):
    # embed timestep
    time_encs = self.time_embedder(timestep) # batch_size x hidden_dim
    classes = self.mlp_in_classes(nodes) # batch_size x num_nodes x hidden_dim
    params = self.mlp_in_params(nodes) # batch_size x num_nodes x hidden_dim
    # nodes = self.mlp_in_nodes(nodes)
    edges = self.mlp_in_edges(edges) # batch_size x num_nodes x num_nodes x hidden_dim

    num_checkpoint = self.num_checkpoints
    for layer in self.block_layers:      
      # layer = self.block_layers[idx]
      # nodes, edges, time_encs = checkpoint(layer, nodes, edges, time_encs, use_reentrant = False) if num_checkpoint > 0 else layer(nodes, edges, time_encs)
      classes, params, edges, time_encs = checkpoint(layer, classes, params, edges, time_encs, use_reentrant = False) if num_checkpoint > 0 else layer(classes, params, edges, time_encs)
      
      num_checkpoint = num_checkpoint - 1
    
    nodes = torch.cat([classes, params], dim = -1)
    nodes = self.mlp_out_nodes(nodes)
    edges = self.mlp_out_edges(edges)
    return nodes, edges
  
  @torch.no_grad()
  def sample(self, batch_size : int):
    # Sample Noise
    num_nodes = MAX_NUM_PRIMITIVES
    num_node_features = NODE_FEATURE_DIMENSION
    num_edge_features = EDGE_FEATURE_DIMENSION
    nodes = torch.zeros(batch_size, num_nodes, num_node_features)
    edges = torch.zeros(batch_size, num_nodes, num_nodes, num_edge_features)
    # binary noise (isConstructible)
    nodes[:,:,0] = torch.ones(size = (batch_size * num_nodes, 2)).multinomial(1)\
                        .reshape(batch_size, num_nodes).float()
    # categorical noise (primitive type)
    nodes[:,:,1:6] = F.one_hot(torch.ones(size = (batch_size * num_nodes, 5)).multinomial(1), 5)\
                      .reshape(batch_size, num_nodes, -1).float()
    # gaussian noise (primitive parameters)
    nodes[:,:,6:] = torch.randn(size = (batch_size, num_nodes, 14))
    # categorical noise (subnode a type)
    edges[:,:,:,0:4] = F.one_hot(torch.ones(size = (batch_size * num_nodes * num_nodes, 4)).multinomial(1), 4)\
                      .reshape(batch_size, num_nodes, num_nodes, -1).float()
    # categorical noise (subnode b type)
    edges[:,:,:,4:8] = F.one_hot(torch.ones(size = (batch_size * num_nodes * num_nodes, 4)).multinomial(1), 4)\
                      .reshape(batch_size, num_nodes, num_nodes, -1).float()
    # categorical noise (subnode a type)
    edges[:,:,:,8:] = F.one_hot(torch.ones(size = (batch_size * num_nodes * num_nodes, 9)).multinomial(1), 9)\
                     .reshape(batch_size, num_nodes, num_nodes, -1).float()
    
    nodes = nodes.to(self.device)
    edges = edges.to(self.device)
    return self.denoise(nodes, edges)

  @torch.no_grad()
  def denoise(self, nodes, edges):
    for t in reversed(range(1, self.max_timestep)):
      # model expects a timestep for each batch
      batch_size = nodes.size(0)
      time = torch.Tensor([t]).expand(batch_size).int()
      pred_node_noise, pred_edge_noise = self.forward(nodes, edges, time)
      nodes, edges = self.reverse_step(nodes, edges, pred_node_noise, pred_edge_noise, t)
    return nodes, edges
  
  @torch.no_grad()
  def noise(self, nodes, edges):
    nodes, edges, _, _ = self.noise_scheduler(nodes, edges, self.max_timestep - 1)
    return nodes, edges
  
  @torch.no_grad()
  def reverse_step(self, curr_nodes : Tensor, curr_edges : Tensor, pred_nodes : Tensor, pred_edges : Tensor, timestep : int):
    new_nodes = torch.zeros_like(curr_nodes)
    new_edges = torch.zeros_like(curr_edges)
    # IsConstructible denoising
    new_nodes[:,:,0], _ = self.noise_scheduler.apply_bernoulli_posterior_step(curr_nodes[:,:,0], pred_nodes[:,:,0:2], timestep)
    # Primitive Types denoising
    new_nodes[:,:,1:6], _ = self.noise_scheduler.apply_multinomial_posterior_step(curr_nodes[:,:,1:6], pred_nodes[:,:,2:7], timestep)
    # Primitive parameters denoising
    new_nodes[:,:,6:], _ = self.noise_scheduler.apply_gaussian_posterior_step(curr_nodes[:,:,6:], pred_nodes[:,:,7:], timestep)
    # Subnode A denoising
    new_edges[:,:,:,0:4], _ = self.noise_scheduler.apply_multinomial_posterior_step(curr_edges[:,:,:,0:4], pred_edges[:,:,:,0:4], timestep)
    # Subnode B denoising
    new_edges[:,:,:,4:8], _ = self.noise_scheduler.apply_multinomial_posterior_step(curr_edges[:,:,:,4:8], pred_edges[:,:,:,4:8], timestep)
    # Constraint Types denoising
    new_edges[:,:,:,8:], _ = self.noise_scheduler.apply_multinomial_posterior_step(curr_edges[:,:,:,8:], pred_edges[:,:,:,8:], timestep)
    return new_nodes, new_edges
  

# num_loss_samples = 100
# loss_samples = []
def diffusion_loss(pred_node_noise : Tensor, pred_edge_noise : Tensor, true_node_noise : Tensor, true_edge_noise : Tensor, params_mask : Tensor, none_node_mask : Tensor, none_edge_mask : Tensor, 
                   true_nodes : Tensor, noised_nodes : Tensor, true_edges : Tensor, noised_edges : Tensor, noise_scheduler : CosineNoiseScheduler, timestep : Tensor, loss_dict : dict) -> Tensor:
    '''Edge Loss'''
    # Only apply subnode loss to constraints that are not none -------
    masked_pred_suba = torch.masked_select(pred_edge_noise[...,0:4], none_edge_mask)
    masked_true_suba = torch.masked_select(true_edge_noise[...,0:4], none_edge_mask)
    suba_loss = ((masked_pred_suba - masked_true_suba) ** 2).mean()

    masked_pred_subb = torch.masked_select(pred_edge_noise[...,4:8], none_edge_mask)
    masked_true_subb = torch.masked_select(true_edge_noise[...,4:8], none_edge_mask)
    subb_loss = ((masked_pred_subb - masked_true_subb) ** 2).mean()

    # Apply constraint loss
    # constraint_loss = ((pred_edge_noise[...,8:] - true_edge_noise[...,8:]) ** 2).mean()
    constraint_type_labels = torch.argmax(true_edges[...,8:], dim = 3)
    constraint_type_logits = noise_scheduler.get_pred_probs(noised_nodes[...,8:], pred_node_noise[...,8:], timestep).log()
    
    constraint_loss = F.nll_loss(
        input = constraint_type_logits.reshape(-1, 9), 
        target = constraint_type_labels.flatten(),
        # weight = weight, 
        reduction = 'mean')

    edge_loss = suba_loss + subb_loss + constraint_loss

    '''Node Loss'''
    # Only apply isconstruct loss to primitives that are not none type
    masked_pred_construct = torch.masked_select(pred_node_noise[...,0:2], none_node_mask)
    masked_true_construct = torch.masked_select(true_node_noise[...,0:2], none_node_mask)
    isconstruct_loss = ((masked_pred_construct - masked_true_construct) ** 2).mean()

    # primitive_type_loss = ((pred_node_noise[...,2:7] - true_node_noise[...,2:7]) ** 2).mean()

    primitive_type_labels = torch.argmax(true_nodes[:,:,1:6], dim = 2) # batch_size x num_nodes (class index for each node)
    primitive_type_logits = noise_scheduler.get_pred_probs(noised_nodes[:,:,1:6], pred_node_noise[...,2:7], timestep).log() # batch_size x num_primitive_types x num_nodes
    
    primitive_type_loss = F.cross_entropy(
        input = primitive_type_logits.reshape(-1, 5), 
        target = primitive_type_labels.flatten(),
        # weight = weight, 
        reduction = 'mean')

    # Only apply parameter loss to relevant primitive parameters
    # masked_pred_param = torch.masked_select(pred_node_noise[...,7:], none_node_mask)
    # masked_true_param = torch.masked_select(true_node_noise[...,7:], none_node_mask)
    sqrt_cumul_var = noise_scheduler.sqrt_cumulative_variances[timestep][:,None,None] # b, 1, 1
    sqrt_cumul_prec = noise_scheduler.sqrt_cumulative_precisions[timestep][:,None,None] # b, 1, 1
    pred_true_params = (noised_nodes[...,6:] - pred_node_noise[...,7:] * sqrt_cumul_var) / sqrt_cumul_prec
    parameter_loss = ((pred_true_params - true_nodes[...,6:]) ** 2 * params_mask).sum() / params_mask.sum()

    node_loss = 1.0 * isconstruct_loss + 2.0 * primitive_type_loss + 2.0 * parameter_loss

    total_loss = node_loss + 1.0 * edge_loss

    loss_dict["edge loss"] = suba_loss.item() + subb_loss.item() + constraint_loss.item()
    loss_dict["edge sub_a loss"] = suba_loss.item()
    loss_dict["edge sub_b loss"] = subb_loss.item()
    loss_dict["edge type loss"] = constraint_loss.item()
    loss_dict["node loss"] = isconstruct_loss.item() + primitive_type_loss.item() + parameter_loss.item()
    loss_dict["node isconstruct loss"] = isconstruct_loss.item()
    loss_dict["node type loss"] = primitive_type_loss.item()
    loss_dict["node parameter loss"] = parameter_loss.item()
    loss_dict["total loss"] = isconstruct_loss.item() + primitive_type_loss.item() + parameter_loss.item() + suba_loss.item() + subb_loss.item() + constraint_loss.item()

    # global loss_samples
    # loss_samples.append(total_loss.item())
    # if len(loss_samples) >= num_loss_samples:
    #     loss_samples = loss_samples[1:] 
    # mu = statistics.fmean(loss_samples)
    # # sig = statistics.stdev(loss_samples, mu)

    # total_loss = (total_loss - mu) / 8 + mu

    return total_loss

# # %%
# import torch
# import math
# from typing import Dict, List, Tuple, Any
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor
# from torch.utils.tensorboard.writer import SummaryWriter
# from torch.utils.checkpoint import checkpoint
# from config import NODE_FEATURE_DIMENSION, EDGE_FEATURE_DIMENSION, MAX_NUM_PRIMITIVES

# def normalize_probs(tensor : Tensor) -> Tensor:
#     tensor = torch.clamp(tensor, 0.0, 1.0) # stave off floating point error nonsense
#     return tensor / tensor.sum(dim = -1, keepdim = True)

# # %% [markdown]
# # ### MODULES

# # %%
# class CosineNoiseScheduler(nn.Module):
#   def __init__(self, max_timestep : int, device : torch.device):
#     super().__init__()
#     self.device = device
#     self.max_timestep = max_timestep
#     self.softmax_scale = 4.0
#     self.offset = .008 # Fixed offset to improve noise prediction at early timesteps

#     # --- Variance Schedule --- #
#     # Cosine Beta Schedule Formula: https://arxiv.org/abs/2102.09672     1.00015543316 is 1/a(0), for offset = .008
#     self.cumulative_precisions = torch.cos((torch.linspace(0, 1, self.max_timestep).to(self.device) + self.offset) * 0.5 * math.pi / (1 + self.offset)) ** 2 * 1.00015543316
#     self.cumulative_variances = 1 - self.cumulative_precisions
#     self.variances = torch.cat([torch.Tensor([0]).to(self.device), 1 - (self.cumulative_precisions[1:] / self.cumulative_precisions[:-1])]).clamp(.0001, .9999)
#     self.precisions = 1 - self.variances
#     self.sqrt_cumulative_precisions = torch.sqrt(self.cumulative_precisions)
#     self.sqrt_cumulative_variances = torch.sqrt(self.cumulative_variances)
#     self.sqrt_precisions = torch.sqrt(self.precisions)
#     self.sqrt_variances = torch.sqrt(self.variances)
#     self.sqrt_posterior_variances = torch.cat([torch.Tensor([0]).to(self.device), torch.sqrt(self.variances[1:] * self.cumulative_variances[:-1] / self.cumulative_variances[1:])])

#     # --- Probability Distributions --- #
#     self.gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor([0.0]), torch.tensor([1.0]))
#     self.normal_dist = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

#   def forward(self, nodes : Tensor, edges : Tensor, timestep : Tensor):
#     ''' Apply noise to graph '''
#     noisy_nodes = torch.zeros(size = nodes.size(), device = self.device)
#     noisy_edges = torch.zeros(size = edges.size(), device = self.device)
#     batch_size, num_nodes, node_dim = nodes.size()
#     true_node_noise = torch.zeros(size = (batch_size, num_nodes, node_dim + 1), device = self.device)
#     true_edge_noise = torch.zeros(size = edges.size(), device = self.device)
#     # nodes = batch_size x num_nodes x NODE_FEATURE_DIMENSION ; edges = batch_size x num_nodes x num_nodes x EDGE_FEATURE_DIMENSION
#     bernoulli_is_constructible = nodes[:,:,0] # batch_size x num_nodes x 1
#     categorical_primitive_types = nodes[:,:,1:6] # batch_size x num_nodes x 5
#     gaussian_primitive_parameters = nodes[:,:,6:] # batch_size x num_nodes x 14
#     # subnode just means if the constraint applies to the start, center, or end of a primitive
#     categorical_subnode_a_types = edges[:,:,:,0:4] # batch_size x num_nodes x 4
#     categorical_subnode_b_types = edges[:,:,:,4:8] # batch_size x num_nodes x 4
#     categorical_constraint_types = edges[:,:,:,8:] # batch_size x num_nodes x 9
#     # IsConstructible noise
#     b, n = bernoulli_is_constructible.size()
#     is_construct_noise = self.gumbel_dist.sample((b, n, 2)).to(self.device).squeeze(-1) # standard gumbel noise; (b, n, 2)
#     true_node_noise[...,0:2] = is_construct_noise
#     noisy_nodes[:,:,0] = self.apply_binary_noise(bernoulli_is_constructible, is_construct_noise, timestep)
#     # Primitive Types noise
#     prim_type_noise = self.gumbel_dist.sample(categorical_primitive_types.size()).to(self.device).squeeze(-1) # standard gumbel noise; (b, n, 5)
#     true_node_noise[...,2:7] = prim_type_noise
#     noisy_nodes[:,:,1:6] = self.apply_discrete_noise(categorical_primitive_types, prim_type_noise, timestep) # noised_primitive_types
#     # Primitive parameters noise
#     parameter_noise = self.normal_dist.sample(gaussian_primitive_parameters.size()).to(self.device).squeeze(-1) # standard gaussian noise; (b, n, 14)
#     true_node_noise[:,:,7:] = parameter_noise
#     noisy_nodes[:,:,6:] = self.apply_gaussian_noise(gaussian_primitive_parameters, timestep, parameter_noise)
#     # Subnode A noise
#     suba_type_noise = self.gumbel_dist.sample(categorical_subnode_a_types.size()).to(self.device).squeeze(-1) # standard gumbel noise; (b, n, 4)
#     true_edge_noise[...,0:4] = suba_type_noise
#     noisy_edges[:,:,:,0:4] = self.apply_discrete_noise(categorical_subnode_a_types, suba_type_noise, timestep) # noised_subnode_a_types
#     # Subnode B noise
#     subb_type_noise = self.gumbel_dist.sample(categorical_subnode_b_types.size()).to(self.device).squeeze(-1) # standard gumbel noise; (b, n, 4)
#     true_edge_noise[...,4:8] = subb_type_noise
#     noisy_edges[:,:,:,4:8] = self.apply_discrete_noise(categorical_subnode_b_types, subb_type_noise, timestep) # noised_subnode_a_types
#     # Constraint Types noise
#     constraint_type_noise = self.gumbel_dist.sample(categorical_constraint_types.size()).to(self.device).squeeze(-1) # standard gumbel noise; (b, n, 9)
#     true_edge_noise[...,8:] = constraint_type_noise
#     noisy_edges[:,:,:,8:] = self.apply_discrete_noise(categorical_constraint_types, constraint_type_noise, timestep) # noised_constraint_types

#     return noisy_nodes, noisy_edges, true_node_noise, true_edge_noise
  
#   def get_transition_noise(self, parameters : Tensor, timestep : int, gaussian_noise : Tensor = None):
#     if gaussian_noise is None:
#       gaussian_noise = torch.randn_like(parameters) # standard gaussian noise
#     return self.sqrt_precisions[timestep] * parameters + self.sqrt_variances[timestep] * gaussian_noise
  
#   def apply_gaussian_noise(self, parameters : Tensor, timestep : Tensor | int, gaussian_noise : Tensor):
#     if type(timestep) is int: timestep = [timestep]
#     # parameters shape is batch_size x num_nodes x num_params
#     # gaussian_noise shape is batch_size x num_nodes x num_params
#     batched_sqrt_precisions = self.sqrt_cumulative_precisions[timestep,None,None] # (b,1,1) or (1,1,1)
#     batched_sqrt_variances = self.sqrt_cumulative_variances[timestep,None,None]   # (b,1,1) or (1,1,1)
#     return batched_sqrt_precisions * parameters + batched_sqrt_variances * gaussian_noise
  
#   def apply_gaussian_posterior_step(self, curr_params : Tensor, pred_noise : Tensor, timestep : int):
#     # sqrt_prev_cumul_prec = self.sqrt_cumulative_precisions[timestep - 1]
#     var = self.variances[timestep]
#     sqrt_prec = self.sqrt_precisions[timestep]
#     sqrt_cumul_var = self.sqrt_cumulative_variances[timestep]
#     # prev_cumul_var = self.cumulative_variances[timestep - 1]
#     # cumul_var = self.cumulative_variances[timestep]
#     sqrt_cumul_prec = self.sqrt_cumulative_precisions[timestep]
    
#     denoised_mean = (curr_params - pred_noise * var / sqrt_cumul_var) / sqrt_prec
#     if timestep > 1:
#       # denoised_mean = (sqrt_prev_cumul_prec * var * pred_params + sqrt_prec * prev_cumul_var * curr_params) / cumul_var
#       pred_true_params = (curr_params - pred_noise * sqrt_cumul_var) / sqrt_cumul_prec

#       gaussian_noise = torch.randn_like(curr_params)
#       return denoised_mean + gaussian_noise * self.sqrt_posterior_variances[timestep], pred_true_params
#     else:
#       return denoised_mean, denoised_mean
    
#   def get_transition_matrix(self, dimension : int, timestep : int | Tensor):
#     if type(timestep) is int: assert timestep > 0; timestep = [timestep]
#     batched_precisions = self.precisions[timestep,None,None] # (batch_size, 1, 1) or (1, 1, 1)
#     return batched_precisions * torch.eye(dimension, dtype = torch.float32, device = self.device) + (1 - batched_precisions) / dimension # (batch_size, d, d) or (1, d, d)
  
#   def get_cumulative_transition_matrix(self, dimension : int, timestep : int | Tensor):
#     if type(timestep) is int: assert timestep > 0; timestep = [timestep]
#     batched_precisions = self.cumulative_precisions[timestep,None,None] # (batch_size, 1, 1) or (1, 1, 1)
#     return batched_precisions * torch.eye(dimension, dtype = torch.float32, device = self.device) + (1 - batched_precisions) / dimension # (batch_size, d, d) or (1, d, d)
  
#   def get_inverse_cumulative_transition_matrix(self, dimension : int, timestep : int | Tensor):
#     if type(timestep) is int: assert timestep > 0; timestep = [timestep]
#     batched_inv_precisions = 1.0 / self.cumulative_precisions[timestep,None,None] # (batch_size, 1, 1) or (1, 1, 1)
#     out = batched_inv_precisions * torch.eye(dimension, dtype = torch.float32, device = self.device) + (1 - batched_inv_precisions) / dimension # (batch_size, d, d) or (1, d, d)
#     return torch.clamp(out, 0.0, 1.0) # Prevent floating point error nonsense
  
#   def get_posterior_transition_matrix(self, pred_true_probs : Tensor, timestep : int) -> torch.Tensor:
#     x_size, pred_true_probs = self.flatten_middle(pred_true_probs) # (b, n, d) or (b, n * n, d), for convenience let m = n or n * n
#     d = x_size[-1]
#     qt = self.get_transition_matrix(d, timestep) # element at [i, j] = p(x_t = j | x_t-1 = i); (1, d, d)
#     qt_bar = self.get_cumulative_transition_matrix(d, timestep) # element at [i, j] = p(x_t = j | x_0 = i); (1, d, d)
#     qt_1bar = self.get_cumulative_transition_matrix(d, timestep - 1) # element at [i, j] = p(x_t-1 = j | x_0 = i); (1, d, d)

#     cond_xt_x0_probs = (qt.permute(0, 2, 1).unsqueeze(1) * qt_1bar.unsqueeze(2) / qt_bar.unsqueeze(3)).squeeze(0) # (d, d, d) where element at [i, j, k] = p(x_t-1 = k | x_t = j, x_0 = i)
#     cond_xt_probs = torch.einsum('bij,jkt->bikt', pred_true_probs, cond_xt_x0_probs) # (b, m, d, d) where element at [i, j] = posterior transition matrix to get x_t-1 given x_t

#     # qt = xt @ self.get_transition_matrix(d, timestep).permute(0, 2, 1) # (b, m, d), since xt is onehot we are plucking out rows corresponding to p(x_t = class | x_(t-1))
#     # qt_bar = xt @ self.get_cumulative_transition_matrix(d, timestep).permute(0, 2, 1) # (b, m, d), since xt is onehot we are plucking out rows corresponding to p(x_t = class | x_0)
#     # q = qt.unsqueeze(2) / qt_bar.unsqueeze(3) # (b, m, d, d), perform an outer product so element at (b, m, i, j) = p(x_t = class | x_(t-1) = j) / p(x_t = class | x_0 = i)
#     # q = q * self.get_cumulative_transition_matrix(d, timestep - 1).unsqueeze(1) # (b, m, d, d), broadcast multiply so element at (b, m, i, j) = p(x_t = class | x_(t-1) = j) * p(x_(t-1) = j | x_0 = i) / p(x_t = class | x_0 = i)

#     return cond_xt_probs.view(size = x_size + (d,)) # reshape into (b, n, d, d) or (b, n, n, d, d)
  
#   def apply_discrete_noise(self, x_one_hot : Tensor, gumbel_noise : Tensor, timestep : Tensor | int):
#     size, x = self.flatten_middle(x_one_hot)
#     q = self.get_cumulative_transition_matrix(size[-1], timestep) # (b, d, d) or (1, d, d)
#     distribution = x @ q # (b, n, d) or (b, n * n, d)
#     distribution = distribution.view(size) # (b, n, d) or (b, n, n, d)
#     return self.sample_discrete_distribution(distribution, gumbel_noise)
  
#   def apply_multinomial_posterior_step(self, class_probs : Tensor, pred_noise : Tensor, timestep : int):
#     # class_probs and pred_noise = (b, n, d) or (b, n, n, d)
#     # m = (n) or (n, n)
#     d = class_probs.size(-1)
#     class_probs = normalize_probs(class_probs) # Avoid floating point error nonsense

#     log_probs = class_probs.log()

#     pred_true_probs = torch.exp((log_probs - log_probs[...,-1,None]) / self.softmax_scale - pred_noise + pred_noise[...,-1,None]) # b, m, d
#     pred_true_probs = pred_true_probs / pred_true_probs.sum(-1, keepdim = True)
#     pred_true_probs = pred_true_probs @ self.get_inverse_cumulative_transition_matrix(d, timestep)

#     pred_true_probs = normalize_probs(pred_true_probs)
    
#     if timestep > 1:
#       q = self.get_posterior_transition_matrix(pred_true_probs, timestep) # (b, n, d, d) or (b, n, n, d, d)
#       class_probs = class_probs.unsqueeze(-2) # (b, n, 1, d) or (b, n, n, 1, d), make probs into row vector
#       posterior_distribution = class_probs @ q # (b, n, 1, d) or (b, n, n, 1, d), batched vector-matrix multiply
#       posterior_distribution = posterior_distribution.squeeze(-2) # (b, n, d) or (b, n, n, d)
#       new_noise = self.gumbel_dist.sample(pred_noise.size()).to(self.device).squeeze(-1)
#       return self.sample_discrete_distribution(posterior_distribution, new_noise), pred_true_probs
#     else:
#       return pred_true_probs, pred_true_probs
    
#   def apply_binary_noise(self, boolean_flag : Tensor, gumbel_noise : Tensor, timestep : int | Tensor):
#     boolean_flag = boolean_flag.unsqueeze(-1)
#     one_hot = torch.cat([1 - boolean_flag, boolean_flag], dim = -1) # (b, n, 2)
#     noised_one_hot = self.apply_discrete_noise(one_hot, gumbel_noise, timestep) # (b, n, 2)
#     return noised_one_hot[...,1] # (b, n)
  
#   def apply_bernoulli_posterior_step(self, boolean_prob : Tensor, pred_noise : Tensor, timestep : int):
#     boolean_prob = boolean_prob.unsqueeze(-1) # b, n, 1
#     class_probs = torch.cat([1 - boolean_prob, boolean_prob], dim = -1) # (b, n, 2)
#     # pred_noise is shape (b, n, 2)
#     new_probs, pred_true_probs = self.apply_multinomial_posterior_step(class_probs, pred_noise, timestep) # (b, n, 2)
#     return new_probs[...,1], pred_true_probs[...,1] # (b, n)
  
#     # if timestep > 1:
#     #   boolean_prob = boolean_prob.unsqueeze(-1) # b, n, 1
#     #   class_probs = torch.cat([1 - boolean_prob, boolean_prob], dim = -1) # (b, n, 2)
#     #   # pred_noise is shape (b, n, 2)
#     #   new_probs = self.apply_multinomial_posterior_step(class_probs, pred_noise, timestep) # (b, n, 2)
#     #   return new_probs[...,1] # (b, n)
#     # else:
#     #   return pred_boolean_prob
  
#   def sample_discrete_distribution(self, class_probs : Tensor, gumbel_vals : Tensor):
#     '''Performs Gumbel-Softmax'''
#     # (b, n, d) or (b, n, n, d) is shape of class_probs and gumbel_vals
#     class_probs = normalize_probs(class_probs)
#     log_probs = class_probs.log()
#     temp = log_probs + gumbel_vals - (log_probs[...,-1,None] + gumbel_vals[...,-1,None])
#     pseudo_one_hot = F.softmax(self.softmax_scale * temp, -1)
#     out = normalize_probs(pseudo_one_hot)
#     return out
#     # size = tensor.size()
#     # num_classes = size[-1]
#     # return F.one_hot(tensor.reshape(-1, num_classes).multinomial(1), num_classes).reshape(size)
  
#   def flatten_middle(self, x : Tensor):
#     prev_size = x.size() # shape of x_one_hot is (b, n, d) or (b, n, n, d)
#     return prev_size, x.view(prev_size[0], -1, prev_size[-1]) # (b, n, d) or (b, n * n, d)

# # %%


# # %%
# class TimeEmbedder(nn.Module):
#   def __init__(self, max_timestep : int, embedding_dimension : int, device : torch.device):
#     super().__init__()
#     self.device = device
#     self.embed_dim = embedding_dimension
#     self.max_steps = max_timestep + 1
#     self.max_timestep = max_timestep
      
#     timesteps = torch.arange(self.max_steps, device = self.device).unsqueeze(1) # num_timesteps x 1
#     scales = torch.exp(torch.arange(0, self.embed_dim, 2, device = self.device) * (-math.log(10000.0) / self.embed_dim)).unsqueeze(0) # 1 x (embedding_dimension // 2)
#     self.time_embs = torch.zeros(self.max_steps, self.embed_dim, device = self.device) # num_timesteps x embedding_dimension
#     self.time_embs[:, 0::2] = torch.sin(timesteps * scales) # fill even columns with sin(timestep * 1000^-(2*i/embedding_dimension))
#     self.time_embs[:, 1::2] = torch.cos(timesteps * scales) # fill odd columns with cos(timestep * 1000^-(2*i/embedding_dimension))
      
#   def forward(self, timestep : Tensor):
#     return self.time_embs[timestep] # batch_size x embedding_dimension

# # %%
# class MPSiLU(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.silu = nn.SiLU()
    
#     def forward(self, x):
#         return self.silu(x) / 0.596

# # %%
# class TransformerLayer(nn.Module):
#     def __init__(self, num_heads : int, node_dim : int, edge_dim : int, device : torch.device):
#         super().__init__()
#         self.num_heads = num_heads
#         self.node_dim = node_dim
#         self.edge_dim = edge_dim

#         self.lin_node_add_embs = nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device)
#         self.lin_node_mul_embs = nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device)

#         self.lin_edge_add_embs = nn.Linear(in_features = self.node_dim, out_features = self.edge_dim, device = device)
#         self.lin_edge_mul_embs = nn.Linear(in_features = self.node_dim, out_features = self.edge_dim, device = device)

#         self.attention_heads = MultiHeadAttention(node_dim = self.node_dim, edge_dim = self.edge_dim, num_heads = self.num_heads, device = device)

#         self.layer_norm_nodes = nn.Sequential(
#             nn.LayerNorm(normalized_shape = self.node_dim, device = device, elementwise_affine = False, bias = False),
#             # nn.LeakyReLU(.01),
#             )

#         self.layer_norm_edges = nn.Sequential(
#             nn.LayerNorm(normalized_shape = self.edge_dim, device = device, elementwise_affine = False, bias = False),
#             # nn.LeakyReLU(.01),
#             )

#         self.mlp_nodes = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device),
#                                        nn.LeakyReLU(0.01),
#                                     #    nn.Dropout(p = 0.1),
#                                        nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device),
#                                       )
        
#         self.mlp_edges = nn.Sequential(nn.Linear(in_features = self.edge_dim, out_features = self.edge_dim, device = device),
#                                        nn.LeakyReLU(0.01),
#                                     #    nn.Dropout(p = 0.1),
#                                        nn.Linear(in_features = self.edge_dim, out_features = self.edge_dim, device = device),
#                                       )
        
#         self.layer_norm_nodes2 = nn.Sequential(
#             nn.LayerNorm(normalized_shape = self.node_dim, device = device, elementwise_affine = False, bias = False),
#             # nn.LeakyReLU(.01),
#             )

#         self.layer_norm_edges2 = nn.Sequential(
#             nn.LayerNorm(normalized_shape = self.edge_dim, device = device, elementwise_affine = False, bias = False),
#             # nn.LeakyReLU(.01),
#             )
    
#     def forward(self, nodes : Tensor, edges : Tensor, time_emb : Tensor) -> Tuple[Tensor, Tensor]:
#       # Inject timestep information
#       time_node_add = self.lin_node_add_embs(time_emb).unsqueeze(1)
#       time_node_mul = self.lin_node_mul_embs(time_emb).unsqueeze(1)
#       nodes = nodes * time_node_mul + nodes + time_node_add

#       time_edge_add = self.lin_edge_add_embs(time_emb).unsqueeze(1).unsqueeze(1)
#       time_edge_mul = self.lin_edge_mul_embs(time_emb).unsqueeze(1).unsqueeze(1)
#       edges = edges * time_edge_mul + edges + time_edge_add

#       # Perform multi head attention
#       attn_nodes, attn_edges = self.attention_heads(nodes, edges) # batch_size x num_nodes x node_dim ; batch_size x num_nodes x num_nodes x edge_dim

#       # Layer normalization with a skip connection
#       attn_nodes = self.layer_norm_nodes(attn_nodes + nodes) # batch_size x num_nodes x node_dim
#       attn_edges = self.layer_norm_edges(attn_edges + edges) # batch_size x num_nodes x num_nodes x edge_dim

#       del nodes
#       del edges

#       # MLP out
#       new_nodes = self.mlp_nodes(attn_nodes) # batch_size x num_nodes x node_dim
#       new_edges = self.mlp_edges(attn_edges) # batch_size x num_nodes x num_nodes x edge_dim

#       # Second layer normalization with a skip connection
#       new_nodes = self.layer_norm_nodes2(new_nodes + attn_nodes) # batch_size x num_nodes x node_dim
#       new_edges = self.layer_norm_edges2(new_edges + attn_edges) # batch_size x num_nodes x num_nodes x edge_dim
#       del attn_nodes
#       del attn_edges

#       return new_nodes, new_edges
    
# # Outer Product Attention Head
# class MultiHeadAttention(nn.Module):
#     def __init__(self, node_dim : int, edge_dim : int, num_heads : int, device : torch.device):
#         super().__init__()
#         self.node_dim = node_dim
#         self.edge_dim = edge_dim
#         self.num_heads = num_heads
#         self.attn_dim = node_dim // num_heads

#         self.lin_query = nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device)
#         self.lin_key = nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device)
#         self.lin_value = nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device)

#         self.lin_mul = nn.Linear(in_features = self.edge_dim, out_features = self.node_dim, device = device)
#         self.lin_add = nn.Linear(in_features = self.edge_dim, out_features = self.node_dim, device = device)

#         self.lin_nodes_out = nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device)
#         self.lin_edges_out = nn.Sequential(
#                                            nn.Linear(in_features = self.node_dim, out_features = self.edge_dim, device = device),
#                                           )

#     def forward(self, nodes : Tensor, edges : Tensor):
#         batch_size, num_nodes, _ = nodes.size()
        
#         # Outer Product Attention -------
#         queries = self.lin_query(nodes).view(batch_size, num_nodes, self.num_heads, -1) # batch_size x num_nodes x num_heads x attn_dim
#         keys = self.lin_key(nodes).view(batch_size, num_nodes, self.num_heads, -1)      # batch_size x num_nodes x num_heads x attn_dim

#         attention = queries.unsqueeze(2) * keys.unsqueeze(1) / math.sqrt(self.node_dim) # batch_size x num_nodes x num_nodes x num_heads x attn_dim
#         del queries
#         del keys

#         # Condition attention based on edge features
#         edges_mul = self.lin_mul(edges).view(batch_size, num_nodes, num_nodes, self.num_heads, -1) # batch_size x num_nodes x num_nodes x num_heads x attn_dim
#         edges_add = self.lin_add(edges).view(batch_size, num_nodes, num_nodes, self.num_heads, -1) # batch_size x num_nodes x num_nodes x num_heads x attn_dim
#         del edges
#         new_edges = attention * edges_mul + attention + edges_add # batch_size x num_nodes x num_nodes x num_heads x attn_dim
#         del edges_add
#         del edges_mul
        
#         # Normalize attention
#                                                                            # batch_size x num_nodes x num_nodes x num_heads (Finish dot product)
#         attention = torch.softmax(input = new_edges.sum(dim = 4), dim = 2) # batch_size x num_nodes x num_nodes x num_heads (softmax) 

#         # Weight node representations and sum
#         values = self.lin_value(nodes).view(batch_size, num_nodes, self.num_heads, -1)  # batch_size x num_nodes x num_heads x attn_dim
#         del nodes
#                                                                                                              # batch_size x num_nodes x num_heads x attn_dim
#         weighted_values = (attention.unsqueeze(4) * values.unsqueeze(1)).sum(dim = 2).flatten(start_dim = 2) # batch_size x num_nodes x node_dim
#         del values

#         # Flatten attention heads
#         new_edges = new_edges.flatten(start_dim = 3)
        
#         # Combine attention heads
#         new_nodes = self.lin_nodes_out(weighted_values)
#         new_edges = self.lin_edges_out(new_edges)

#         return new_nodes, new_edges

# # %%
# class SoftAttentionLayer(nn.Module):
#   def __init__(self, node_dim : int, edge_dim : int, num_heads : int, device : torch.device):
#     super().__init__()
#     self.device = device
#     self.node_dim = node_dim
#     self.edge_dim = edge_dim
#     self.num_heads = num_heads
#     self.attn_dim = node_dim // num_heads
#     self.num_nodes = MAX_NUM_PRIMITIVES

#     concat_dim = 2 * self.node_dim + self.edge_dim

#     self.lin_add_embs = nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device)
#     self.lin_mul_embs = nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device)
    
#     self.mlp_haggr_weights = nn.Sequential(nn.Linear(in_features = concat_dim, out_features = concat_dim, device = self.device),
#                                            nn.LeakyReLU(0.01),
#                                            nn.Linear(in_features = concat_dim, out_features = 1, device = self.device),
#                                            nn.Softmax(dim = 2)
#                                           )
    
#     self.mlp_haggr_values = nn.Sequential(nn.Linear(in_features = concat_dim, out_features = concat_dim, device = self.device),
#                                            nn.LeakyReLU(0.01),
#                                            nn.Linear(in_features = concat_dim, out_features = self.node_dim, device = self.device),
#                                          )

#     self.query_key_value_mlp = nn.Linear(in_features = self.node_dim, out_features = 3 * self.node_dim, device = self.device)

#     self.layer_norm_embs = nn.Sequential(nn.LayerNorm(normalized_shape = self.node_dim, device = self.device), nn.LeakyReLU(0.01),)

#     self.time_weight_mlp = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = self.device),
#                                          nn.LeakyReLU(0.01),
#                                          nn.Linear(in_features = self.node_dim, out_features = 1, device = self.device),
#                                          nn.Softmax(dim = 2)
#                                         )
#     self.time_value_mlp = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = self.device),
#                                          nn.LeakyReLU(0.01),
#                                          nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = self.device),
#                                          nn.Softmax(dim = 2)
#                                         )

#     self.node_mlp = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = self.device),
#                                   nn.LeakyReLU(0.01),
#                                   nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = self.device),
#                                  )
#     self.edge_mlp = nn.Sequential(nn.Linear(in_features = 2 * self.node_dim, out_features = self.edge_dim, device = self.device),
#                                   nn.LeakyReLU(0.01),
#                                   nn.Linear(in_features = self.edge_dim, out_features = self.edge_dim , device = self.device),
#                                  )
    
#     self.layer_norm_out_nodes = nn.Sequential(nn.LayerNorm(normalized_shape = self.node_dim, device = self.device), nn.LeakyReLU(0.01),)
    
#     self.layer_norm_out_edges = nn.Sequential(nn.LayerNorm(normalized_shape = self.edge_dim, device = self.device), nn.LeakyReLU(0.01),)
    
#   def forward(self, nodes : Tensor, edges : Tensor, time_emb : Tensor) -> Tuple[Tensor, Tensor]:
#     # Inject timestep information
#     time_add = self.lin_add_embs(time_emb).unsqueeze(1)
#     time_mul = self.lin_mul_embs(time_emb).unsqueeze(1)
#     nodes = nodes * time_mul + nodes + time_add
#     # edges = edges * time_mul.unsqueeze(1).unsqueeze(1) + edges + time_add.unsqueeze(1).unsqueeze(1)

#     # Outer Product Concatenation
#     hstack = nodes.unsqueeze(2).expand(-1, -1, self.num_nodes, -1) # (b, n, n, d)
#     vstack = hstack.permute(0, 2, 1, 3) # (b, n, n, d)
#     graph_features = torch.cat(tensors = (hstack, vstack, edges), dim = 3) # (b, n, n, 3 * d)

#     # Soft Attentional Encoder
#     haggr_weights = self.mlp_haggr_weights(graph_features).permute(0, 1, 3, 2) # (b, n, 1, n)
#     haggr_values = self.mlp_haggr_values(graph_features) # (b, n, n, d)
#     graph_embs = (haggr_weights @ haggr_values).squeeze(2) # (b, n, d)

#     # Low Dimensional Attention
#     b, n, d = graph_embs.size()
#     query_key_value = self.query_key_value_mlp(graph_embs).view(b, n, self.num_heads, 3 * self.attn_dim).permute(0, 2, 1, 3) # (b, h, n, 3 * attn_dim)
#     queries, keys, values = query_key_value.reshape(b * self.num_heads, n, 3 * self.attn_dim).chunk(3, dim = 2) # (b * h, n, attn_dim) is shape for the three tensors
#     attn_embs = F.scaled_dot_product_attention(queries, keys, values).view(b, self.num_heads, n, self.attn_dim) # (b * h, n, attn_dim)
#     attn_embs = attn_embs.permute(0, 2, 1, 3).reshape(b, n, self.node_dim) # (b, n, d)

#     # Residual Connection and LayerNorm
#     attn_embs = self.layer_norm_embs(attn_embs + graph_embs)
    
#     # Incorporate global graph info into time emb
#     time_emb = time_mul * attn_embs + time_add + attn_embs
#     time_weights = self.time_weight_mlp(time_emb) # b, n, 1
#     time_values = self.time_value_mlp(time_emb) # b, n, d
#     time_emb = (time_weights.permute(0, 2, 1) @ time_values).squeeze(1) # b, d

#     # Outer Product Decoder
#     emb_hstack = attn_embs.unsqueeze(2).expand(-1, -1, self.num_nodes, -1) # (b, n, n, d)
#     emb_vstack = emb_hstack.permute(0, 2, 1, 3) # (b, n, n, d)
#     emb_edges = torch.cat(tensors = (emb_hstack, emb_vstack), dim = 3) # (b, n, n, 2 * d)

#     new_edges = self.edge_mlp(emb_edges) # (b, n, n, d)
#     new_nodes = self.node_mlp(attn_embs) # (b, n, d)

#     # Residual Connection and LayerNorm
#     new_edges = self.layer_norm_out_edges(new_edges + edges)
#     new_nodes = self.layer_norm_out_nodes(new_nodes + nodes)

#     return new_nodes, new_edges, time_emb

# # %%
# '''A + W1 @ B + A * W2 @ B'''
# class FiLM(nn.Module):
#     def __init__(self, dim_a : int, dim_b : int, device : torch.device):
#         super().__init__()

#         self.lin_mul = nn.Linear(in_features = dim_b, out_features = dim_a, device = device)
#         self.lin_add = nn.Linear(in_features = dim_b, out_features = dim_a, device = device)
    
#     def forward(self, A, B):
#         # A is shape (b, ..., dim_a)
#         # B is shape (b, *, dim_b)
#         mul = self.lin_mul(B) # (b, *, dim_a)
#         add = self.lin_add(B) # (b, *, dim_a)

#         return A + add + mul * A

# '''Soft Attention Aggregation'''
# class SoftAttention(nn.Module):
#     def __init__(self, dim : int, device : torch.device):
#         super().__init__()

#         self.lin_weights = nn.Linear(in_features = dim, out_features = 1, device = device)
#         self.lin_values = nn.Linear(in_features = dim, out_features = dim, device = device)
    
#     def forward(self, M):
#         # M is shape (b, *, dim)
#         weights = self.lin_weights(M) # (b, *, 1)
#         values = self.lin_values(M) # (b, *, dim)

#         # The output will have one less dimension
#         # batched matrix multiply results in (b, ..., 1, d), then squeeze makes it (b, ..., d)
#         return (weights.transpose(-2, -1) @ values).squeeze(-2)
    
# '''Layer Norm and Additive Skip Connection'''
# class SkipLayerNorm(nn.Module):
#     def __init__(self, dim : int, device : torch.device):
#         super().__init__()

#         self.layer_norm = nn.LayerNorm(normalized_shape = dim, device = device)

#     def forward(self, A, B):
#         return self.layer_norm(A + B)

# # %%
# class AttentionBlock(nn.Module):
#     def __init__(self, node_dim : int, edge_dim : int, time_dim : int, num_heads : int, device : torch.device, bool_last = False):
#         super().__init__()
#         self.attn_dim = node_dim // num_heads
#         self.bool_last = bool_last

#         self.lin_node_in = nn.Linear(in_features = node_dim, out_features = node_dim, device = device)
#         self.lin_edge_in = nn.Linear(in_features = edge_dim, out_features = edge_dim, device = device)
#         self.lin_time_in = nn.Linear(in_features = time_dim, out_features = time_dim, device = device)

#         self.film_node_time = FiLM(dim_a = node_dim, dim_b = time_dim, device = device)
#         self.film_edge_time = FiLM(dim_a = edge_dim, dim_b = time_dim, device = device)

#         self.edge_soft_attn = SoftAttention(dim = edge_dim, device = device)
#         self.lin_node_qkv = nn.Linear(in_features = node_dim, out_features = 3 * node_dim, device = device)
#         self.film_node_edge = FiLM(dim_a = 3 * node_dim, dim_b = edge_dim, device = device)

#         self.lin_attn = nn.Linear(in_features = node_dim, out_features = node_dim, device = device)

#         if bool_last == False:
#           self.node_soft_attn = SoftAttention(dim = node_dim, device = device)
#           self.film_time_node = FiLM(dim_a = time_dim, dim_b = node_dim, device = device)

#         self.mlp_node_out = nn.Sequential(
#             nn.Linear(in_features = node_dim, out_features = node_dim, device = device),
#             nn.LeakyReLU(0.01),
#             nn.Linear(in_features = node_dim, out_features = node_dim, device = device),
#             nn.LeakyReLU(0.01),
#         )
#         self.mlp_edge_out = nn.Sequential(
#             nn.Linear(in_features = 2 * node_dim, out_features = edge_dim, device = device),
#             nn.LeakyReLU(0.01),
#             nn.Linear(in_features = edge_dim, out_features = edge_dim, device = device),
#             nn.LeakyReLU(0.01),
#         )
#         if bool_last == False:
#           self.mlp_time_out = nn.Sequential(
#               nn.Linear(in_features = time_dim, out_features = time_dim, device = device),
#               nn.LeakyReLU(0.01),
#               nn.Linear(in_features = time_dim, out_features = time_dim, device = device),
#               nn.LeakyReLU(0.01),
#           )

#         self.skip_nodes = SkipLayerNorm(dim = node_dim, device = device)
#         self.skip_edges = SkipLayerNorm(dim = edge_dim, device = device)
#         if bool_last == False:
#           self.skip_times = SkipLayerNorm(dim = time_dim, device = device)

#     def forward(self, nodes : Tensor, edges : Tensor, times : Tensor):
#         old_nodes = nodes
#         old_edges = edges
#         old_times = times

#         nodes = self.lin_node_in(nodes) # batch_size, num_nodes, node_dim
#         edges = self.lin_edge_in(edges) # batch_size, num_nodes, num_nodes, edge_dim
#         times = self.lin_time_in(times) # batch_size, time_dim
        
#         # times = F.leaky_relu(times, 0.01)
#         nodes = self.film_node_time(nodes, F.leaky_relu(times, 0.01)[:,None,:])      # batch_size, num_nodes, node_dim
#         edges = self.film_edge_time(edges, F.leaky_relu(times, 0.01)[:,None,None,:]) # batch_size, num_nodes, num_nodes, edge_dim
#         nodes = F.leaky_relu(nodes, 0.01)
#         edges = F.leaky_relu(edges, 0.01)

#         nodes = self.lin_node_qkv(nodes)   # batch_size, num_nodes, 3 * node_dim

#         edge_aggrs = self.edge_soft_attn(edges) # batch_size, num_nodes, edge_dim
#         edge_aggrs = F.leaky_relu(edge_aggrs, 0.01)
#         nodes = self.film_node_edge(nodes, edge_aggrs)

#         b, n, _ = nodes.size() # get batchsize and number of nodes
#         queries, keys, values = nodes.chunk(chunks = 3, dim = -1)
#         queries = queries.reshape(b, n, -1, self.attn_dim) # batch_size, num_nodes, num_heads, attn_dim
#         keys = keys.reshape(b, n, -1, self.attn_dim)       # batch_size, num_nodes, num_heads, attn_dim
#         values = values.reshape(b, n, -1, self.attn_dim)   # batch_size, num_nodes, num_heads, attn_dim

#         attn_vals = F.scaled_dot_product_attention(query = queries, key = keys, value = values)
#         attn_vals = attn_vals.flatten(start_dim = 2)
#         attn_vals = self.lin_attn(attn_vals)
#         attn_vals = F.leaky_relu(attn_vals, 0.01)

#         # New nodes
#         new_nodes = self.mlp_node_out(attn_vals)
#         new_nodes = self.skip_nodes(new_nodes, old_nodes)

#         # New edges
#         hstack = attn_vals[:,:,None,:].expand(-1, -1, n, -1)
#         vstack = hstack.transpose(2, 1)
#         outer_concat = torch.cat([hstack, vstack], dim = -1)
#         new_edges = self.mlp_edge_out(outer_concat)
#         new_edges = self.skip_edges(new_edges, old_edges)

#         # New times
#         if self.bool_last == False:
#           node_aggrs = self.node_soft_attn(attn_vals)
#           node_aggrs = F.leaky_relu(node_aggrs, 0.01)
#           new_times = self.film_time_node(times, node_aggrs)
#           new_times = F.leaky_relu(new_times, 0.01)
#           new_times = self.mlp_time_out(new_times)
#           new_times = self.skip_times(new_times, old_times)
        
#         if self.bool_last == False:
#           return new_nodes, new_edges, new_times
#         else:
#            return new_nodes, new_edges

# # %%
# class Block(nn.Module):
#     def __init__(self, node_dim : int, edge_dim : int, num_heads : int, device : torch.device):
#         super().__init__()

#         # self.emb_mlp1 = nn.Sequential(nn.Linear(in_features = node_dim, out_features = node_dim, device = device),
#         #                               nn.LeakyReLU(0.01),
#         #                               nn.Linear(in_features = node_dim, out_features = node_dim, device = device),
#         #                               nn.LeakyReLU(0.01)
#         #                              )
        
#         self.tf_layer = AttentionBlock(num_heads = num_heads, node_dim = node_dim, edge_dim = edge_dim, time_dim = node_dim, device = device)

#     def forward(self, nodes, edges, time_emb):

#         # time_emb = self.emb_mlp1(time_emb)
#         nodes, edges, time_emb = self.tf_layer(nodes, edges, time_emb)

#         # time_emb = F.leaky_relu(input = time_emb, negative_slope = 0.01)

#         return nodes, edges, time_emb

# # %%
# class TransformerEncoder(nn.Module):
#   def __init__(self, node_dim : int, edge_dim : int, graph_emb_dim : int, num_tf_layers: int, num_heads : int, device : torch.device): # perm_emb_dim: int,
#     super().__init__()
#     self.node_dim = node_dim # Number of features per node
#     self.edge_dim = edge_dim # Number of features per edge
#     self.graph_emb_dim = graph_emb_dim # Size of graph embedding vector
#     self.num_nodes = MAX_NUM_PRIMITIVES # Number of nodes in each graph
#     self.num_edges = self.num_nodes * self.num_nodes # Number of edges in each graph
    
#     self.hidden_dim = 256
#     self.num_tf_layers = num_tf_layers
#     self.num_heads = num_heads
#     self.mlp_in_nodes = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.hidden_dim, device = device),
#                                       nn.LeakyReLU(.1),
#                                     #   nn.Dropout(p = 0.1),
#                                       nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
#                                       nn.LeakyReLU(.1),
#                                     #   nn.Dropout(p = 0.1)
#                                      )
#     self.mlp_in_edges = nn.Sequential(nn.Linear(in_features = self.edge_dim, out_features = self.hidden_dim, device = device),
#                                       nn.LeakyReLU(.1),
#                                     #   nn.Dropout(p = 0.1),
#                                       nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
#                                       nn.LeakyReLU(.1),
#                                     #   nn.Dropout(p = 0.1)
#                                      )
    
#     self.tf_layers = nn.ModuleList([TransformerLayer(num_heads = self.num_heads, 
#                                                      node_dim = self.hidden_dim,
#                                                      edge_dim = self.hidden_dim, 
#                                                      device = device
#                                                     ) 
#                                     for _ in range(self.num_tf_layers)])
    
#     self.time_emb_mlps = nn.ModuleList([nn.Sequential(nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
#                                                       nn.LeakyReLU(0.1),
#                                                       nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
#                                                       nn.LeakyReLU(0.1)
#                                                      )
#                                         for _ in range(self.num_tf_layers)])
    
#     self.mlp_haggr_weights = nn.Sequential(nn.Linear(in_features = 3 * self.hidden_dim, out_features = 3 * self.hidden_dim, device = device),
#                                           nn.LeakyReLU(.1),
#                                         #   nn.Dropout(p = 0.1),
#                                           nn.Linear(in_features = 3 * self.hidden_dim, out_features = 1, device = device),
#                                           nn.Softmax(dim = 2),
#                                         #   nn.Dropout(p = 0.1)
#                                          )
#     self.mlp_haggr_values = nn.Sequential(nn.Linear(in_features = 3 * self.hidden_dim, out_features = 3 * self.hidden_dim, device = device),
#                                           nn.LeakyReLU(.1),
#                                         #   nn.Dropout(p = 0.1),
#                                           nn.Linear(in_features = 3 * self.hidden_dim, out_features = self.hidden_dim, device = device)
#                                          )
    
#     self.mlp_out = nn.Sequential(nn.Linear(in_features = self.num_nodes * self.hidden_dim, out_features = self.graph_emb_dim, device = device),
#                                   nn.LeakyReLU(.1),
#                                 #   nn.Dropout(p = 0.1),
#                                   nn.Linear(in_features = self.graph_emb_dim, out_features = self.graph_emb_dim, device = device)
#                                  )
    
#   def forward(self, nodes : Tensor, edges : Tensor, time_embs : Tensor):
#     nodes = self.mlp_in_nodes(nodes) # batch_size x num_nodes x mlp_node_hidden_dim
#     edges = self.mlp_in_edges(edges) # batch_size x num_nodes x num_nodes x mlp_edge_hidden_dim

#     for idx in range(self.num_tf_layers):
#       time_embs = self.time_emb_mlps[idx](time_embs)

#       layer = self.tf_layers[idx]
#       nodes, edges = layer(nodes, edges, time_embs)
#       # nodes, edges = layer(nodes, edges) # batch_size x num_nodes x tf_node_hidden_dim ; batch_size x num_nodes x num_nodes x tf_edge_hidden_dim
    
#     # nodes = nodes.flatten(start_dim = 1) # batch_size x (num_nodes * node_dim)
#     # edges = edges.flatten(start_dim = 1) # batch_size x (num_nodes * num_nodes * edge_dim)
#     # graph_embs = torch.cat((nodes, edges), 1) # batch_size x (num_nodes * node_dim + num_nodes * num_nodes * edge_dim)
    
#     # Soft attentional aggregation
#     hstack = nodes.unsqueeze(3).expand(-1, -1, -1, self.num_nodes).permute(0, 1, 3, 2) # batch_size x num_nodes x num_nodes x hidden_dim
#     vstack = hstack.permute(0, 2, 1, 3) # batch_size x num_nodes x num_nodes x hidden_dim
#     graph_features = torch.cat(tensors = (hstack, vstack, edges), dim = 3) # batch_size x num_nodes x num_nodes x (3 * hidden_dim)

#     del nodes
#     del edges
#     del hstack
#     del vstack

#     haggr_weights = self.mlp_haggr_weights(graph_features) # batch_size x num_nodes x num_nodes x 1
#     graph_features = self.mlp_haggr_values(graph_features) # batch_size x num_nodes x num_nodes x hidden_dim
#     graph_embs = (haggr_weights.permute(0, 1, 3, 2) @ graph_features).squeeze().flatten(start_dim = 1) # batch_size x (num_nodes * hidden_dim)

#     del haggr_weights
#     del graph_features

#     out_embs = self.mlp_out(graph_embs)     # batch_size x graph_emb_dim

#     del graph_embs
#     return out_embs, time_embs
  
# class TransformerDecoder(nn.Module):
#   def __init__(self, node_dim : int, edge_dim : int, graph_emb_dim : int, num_tf_layers : int, num_heads : int, device : torch.device):
#     super().__init__()
#     self.node_dim = node_dim
#     self.edge_dim = edge_dim
#     self.graph_emb_dim = graph_emb_dim
#     self.num_nodes = MAX_NUM_PRIMITIVES # Number of nodes in each graph
#     self.num_edges = self.num_nodes * self.num_nodes # Number of edges in each graph
#     self.temp_node_dim = 128
#     self.hidden_dim = 256
#     self.num_tf_layers = num_tf_layers
#     self.num_heads = num_heads
#     self.mlp_create_nodes = nn.Sequential(nn.Linear(in_features = self.graph_emb_dim, out_features = self.num_nodes * self.temp_node_dim, device = device),
#                                           nn.LeakyReLU(.1),
#                                         #   nn.Dropout(p = 0.1),
#                                           nn.Linear(in_features = self.num_nodes * self.temp_node_dim, out_features = self.num_nodes * self.temp_node_dim, device = device),
#                                           nn.LeakyReLU(.1),
#                                         #   nn.Dropout(p = 0.1)
#                                          )
#     self.mlp_create_edges = nn.Sequential(nn.Linear(in_features = self.graph_emb_dim, out_features = self.num_edges * self.edge_dim, device = device),
#                                           nn.LeakyReLU(.1),
#                                         #   nn.Dropout(p = 0.1),
#                                           nn.Linear(in_features = self.num_edges * self.edge_dim, out_features = self.num_edges * self.edge_dim, device = device),
#                                           nn.LeakyReLU(.1),
#                                         #   nn.Dropout(p = 0.1)
#                                          )
    
#     self.lin_node_transform = nn.Sequential(nn.Linear(in_features = self.temp_node_dim, out_features = self.hidden_dim, device = device),
#                                             nn.LeakyReLU(.1),
#                                             # nn.Dropout(p = 0.1),
#                                             nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
#                                             nn.LeakyReLU(.1),
#                                             # nn.Dropout(p = 0.1)
#                                            )
#     self.lin_edge_transform = nn.Sequential(nn.Linear(in_features = self.edge_dim, out_features = self.hidden_dim, device = device),
#                                             nn.LeakyReLU(.1),
#                                             # nn.Dropout(p = 0.1),
#                                             nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
#                                             nn.LeakyReLU(.1),
#                                             # nn.Dropout(p = 0.1)
#                                            )
    
#     self.tf_layers = nn.ModuleList([TransformerLayer(num_heads = self.num_heads, 
#                                                      node_dim = self.hidden_dim,
#                                                      edge_dim = self.hidden_dim,
#                                                      device = device
#                                                     ) 
#                                     for _ in range(self.num_tf_layers)])
#     self.time_emb_mlps = nn.ModuleList([nn.Sequential(nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
#                                                       nn.LeakyReLU(0.1),
#                                                       nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
#                                                       nn.LeakyReLU(0.1)
#                                                      )
#                                         for _ in range(self.num_tf_layers)])
    
#     self.mlp_out_nodes = nn.Sequential(nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
#                                        nn.LeakyReLU(.1),
#                                        nn.Linear(in_features = self.hidden_dim, out_features = self.node_dim, device = device)
#                                       )
#     self.mlp_out_edges = nn.Sequential(nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
#                                        nn.LeakyReLU(.1),
#                                     #    nn.Dropout(p = 0.1),
#                                        nn.Linear(in_features = self.hidden_dim, out_features = self.edge_dim, device = device)
#                                       )

#   def forward(self, latents, time_embs):
#     nodes = torch.reshape(input = self.mlp_create_nodes(latents), shape = (-1, self.num_nodes, self.temp_node_dim))            # batch_size x num_nodes x mlp_node_out_dim
#     edges = torch.reshape(input = self.mlp_create_edges(latents), shape = (-1, self.num_nodes, self.num_nodes, self.edge_dim)) # batch_size x num_nodes x num_nodes x mlp_edge_out_dim
#     nodes = self.lin_node_transform(nodes) # batch_size x num_nodes x hidden_dim
#     edges = self.lin_edge_transform(edges) # batch_size x num_nodes x num_nodes x hidden_dim
#     for idx in range(self.num_tf_layers):
#       time_embs = self.time_emb_mlps[idx](time_embs)

#       layer = self.tf_layers[idx]
#       nodes, edges = layer(nodes, edges, time_embs)
#       # nodes, edges = layer(nodes, edges) # batch_size x num_nodes x tf_node_hidden_dim ; # batch_size x num_nodes x num_nodes x tf_edge_hidden_dim
    
#     nodes = self.mlp_out_nodes(nodes) # batch_size x num_nodes x node_dim
#     edges = self.mlp_out_edges(edges) # batch_size x num_nodes x num_nodes x edge_dim
#     # # sigmoid and softmax for nodes
#     # nodes[:,:,0] = F.sigmoid(nodes[:,:,0])              # Sigmoid for isConstructible
#     # nodes[:,:,1:6] = F.softmax(nodes[:,:,1:6], dim = 2) # Softmax for primitive classes (i.e. line, circle, arc, point, none)
    
#     # # softmax for constraints; Conceptual map => n1 (out) -> n2 (in) i.e. out_node, edge, in_node
#     # edges[:,:,:,0:4] = F.softmax(edges[:,:,:,0:4], dim = 3) # Softmax for out_node subnode type
#     # edges[:,:,:,4:8] = F.softmax(edges[:,:,:,4:8], dim = 3) # Softmax for in_node subnode type
#     # edges[:,:,:,8: ] = F.softmax(edges[:,:,:,8: ], dim = 3) # Softmax for edge (aka constraint) type (i.e horizontal, vertical, etc...)
#     return nodes, edges

# # %%
# class GD3PM(nn.Module):
#   def __init__(self, device : torch.device):
#     super().__init__()
#     self.device = device
#     self.node_dim = NODE_FEATURE_DIMENSION
#     self.edge_dim = EDGE_FEATURE_DIMENSION
#     self.node_hidden_dim = 512
#     self.edge_hidden_dim = 64
#     self.time_hidden_dim = 512
#     self.num_tf_layers = 24
#     self.num_checkpoints = 12
#     self.num_heads = 16
#     self.max_timestep = 500
#     self.noise_scheduler = CosineNoiseScheduler(self.max_timestep, self.device)
#     self.time_embedder = TimeEmbedder(self.max_timestep, self.time_hidden_dim, self.device)

#     self.mlp_in_nodes = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.node_hidden_dim, device = device),
#                                       nn.LeakyReLU(0.01),
#                                       nn.Linear(in_features = self.node_hidden_dim, out_features = self.node_hidden_dim, device = device),
#                                       nn.LeakyReLU(0.01),
#                                      )
#     self.mlp_in_edges = nn.Sequential(nn.Linear(in_features = self.edge_dim, out_features = self.edge_hidden_dim, device = device),
#                                       nn.LeakyReLU(0.01),
#                                       nn.Linear(in_features = self.edge_hidden_dim, out_features = self.edge_hidden_dim, device = device),
#                                       nn.LeakyReLU(0.01),
#                                      )
    
#     self.block_layers = nn.ModuleList([AttentionBlock(node_dim = self.node_hidden_dim, edge_dim = self.edge_hidden_dim, time_dim = self.time_hidden_dim, num_heads = self.num_heads, device = self.device)
#                                       for _ in range(self.num_tf_layers - 1)])
    
#     self.last_block = AttentionBlock(node_dim = self.node_hidden_dim, edge_dim = self.edge_hidden_dim, time_dim = self.time_hidden_dim, num_heads = self.num_heads, device = self.device, bool_last=True)
#     self.mlp_out_nodes = nn.Sequential(nn.Linear(in_features = self.node_hidden_dim, out_features = self.node_hidden_dim, device = device),
#                                        nn.LeakyReLU(0.01),
#                                        nn.Linear(in_features = self.node_hidden_dim, out_features = self.node_dim + 1, device = device) # The extra 1 is there since we need 2 gumbel noise values for isconstructible, which means output size is one bigger
#                                       )
#     self.mlp_out_edges = nn.Sequential(nn.Linear(in_features = self.edge_hidden_dim, out_features = self.edge_hidden_dim, device = device),
#                                        nn.LeakyReLU(0.01),
#                                        nn.Linear(in_features = self.edge_hidden_dim, out_features = self.edge_dim, device = device)
#                                       )

#   def forward(self, nodes : Tensor, edges : Tensor, timestep : Tensor):
#     # embed timestep
#     time_encs = self.time_embedder(timestep) # batch_size x hidden_dim
#     nodes = self.mlp_in_nodes(nodes) # batch_size x num_nodes x hidden_dim
#     edges = self.mlp_in_edges(edges) # batch_size x num_nodes x num_nodes x hidden_dim

#     remaining_checkpoints = self.num_checkpoints
#     for idx in range(self.num_tf_layers - 1):      
#       layer = self.block_layers[idx]
#       nodes, edges, time_encs = checkpoint(layer, nodes, edges, time_encs, use_reentrant = False) if remaining_checkpoints > 0 else layer(nodes, edges, time_encs)
      
#       remaining_checkpoints = remaining_checkpoints - 1
    
#     nodes, edges = self.last_block(nodes, edges, time_encs)
#     nodes = self.mlp_out_nodes(nodes)
#     edges = self.mlp_out_edges(edges)
#     return nodes, edges
  
#   @torch.no_grad()
#   def sample(self, batch_size : int):
#     # Sample Noise
#     num_nodes = MAX_NUM_PRIMITIVES
#     num_node_features = NODE_FEATURE_DIMENSION
#     num_edge_features = EDGE_FEATURE_DIMENSION
#     nodes = torch.zeros(batch_size, num_nodes, num_node_features)
#     edges = torch.zeros(batch_size, num_nodes, num_nodes, num_edge_features)
#     # binary noise (isConstructible)
#     nodes[:,:,0] = torch.ones(size = (batch_size * num_nodes, 2)).multinomial(1)\
#                         .reshape(batch_size, num_nodes).float()
#     # categorical noise (primitive type)
#     nodes[:,:,1:6] = F.one_hot(torch.ones(size = (batch_size * num_nodes, 5)).multinomial(1), 5)\
#                       .reshape(batch_size, num_nodes, -1).float()
#     # gaussian noise (primitive parameters)
#     nodes[:,:,6:] = torch.randn(size = (batch_size, num_nodes, 14))
#     # categorical noise (subnode a type)
#     edges[:,:,:,0:4] = F.one_hot(torch.ones(size = (batch_size * num_nodes * num_nodes, 4)).multinomial(1), 4)\
#                       .reshape(batch_size, num_nodes, num_nodes, -1).float()
#     # categorical noise (subnode b type)
#     edges[:,:,:,4:8] = F.one_hot(torch.ones(size = (batch_size * num_nodes * num_nodes, 4)).multinomial(1), 4)\
#                       .reshape(batch_size, num_nodes, num_nodes, -1).float()
#     # categorical noise (subnode a type)
#     edges[:,:,:,8:] = F.one_hot(torch.ones(size = (batch_size * num_nodes * num_nodes, 9)).multinomial(1), 9)\
#                      .reshape(batch_size, num_nodes, num_nodes, -1).float()
    
#     nodes = nodes.to(self.device)
#     edges = edges.to(self.device)
#     return self.denoise(nodes, edges)

#   @torch.no_grad()
#   def denoise(self, nodes, edges):
#     for t in reversed(range(1, self.max_timestep)):
#       # model expects a timestep for each batch
#       batch_size = nodes.size(0)
#       time = torch.Tensor([t]).expand(batch_size).int()
#       pred_node_noise, pred_edge_noise = self.forward(nodes, edges, time)
#       nodes, edges = self.reverse_step(nodes, edges, pred_node_noise, pred_edge_noise, t)
#     return nodes, edges
  
#   @torch.no_grad()
#   def noise(self, nodes, edges):
#     nodes, edges, _, _ = self.noise_scheduler(nodes, edges, self.max_timestep - 1)
#     return nodes, edges
  
#   @torch.no_grad()
#   def reverse_step(self, curr_nodes : Tensor, curr_edges : Tensor, pred_nodes : Tensor, pred_edges : Tensor, timestep : int):
#     new_nodes = torch.zeros_like(curr_nodes)
#     new_edges = torch.zeros_like(curr_edges)
#     # IsConstructible denoising
#     new_nodes[:,:,0], _ = self.noise_scheduler.apply_bernoulli_posterior_step(curr_nodes[:,:,0], pred_nodes[:,:,0:2], timestep)
#     # Primitive Types denoising
#     new_nodes[:,:,1:6], _ = self.noise_scheduler.apply_multinomial_posterior_step(curr_nodes[:,:,1:6], pred_nodes[:,:,2:7], timestep)
#     # Primitive parameters denoising
#     new_nodes[:,:,6:], _ = self.noise_scheduler.apply_gaussian_posterior_step(curr_nodes[:,:,6:], pred_nodes[:,:,7:], timestep)
#     # Subnode A denoising
#     new_edges[:,:,:,0:4], _ = self.noise_scheduler.apply_multinomial_posterior_step(curr_edges[:,:,:,0:4], pred_edges[:,:,:,0:4], timestep)
#     # Subnode B denoising
#     new_edges[:,:,:,4:8], _ = self.noise_scheduler.apply_multinomial_posterior_step(curr_edges[:,:,:,4:8], pred_edges[:,:,:,4:8], timestep)
#     # Constraint Types denoising
#     new_edges[:,:,:,8:], _ = self.noise_scheduler.apply_multinomial_posterior_step(curr_edges[:,:,:,8:], pred_edges[:,:,:,8:], timestep)
#     return new_nodes, new_edges

# # %%


# # %% [markdown]
# # ### Loss

# # %%
# def diffusion_loss(pred_node_noise : Tensor, pred_edge_noise : Tensor, true_node_noise : Tensor, true_edge_noise : Tensor, params_mask : Tensor, loss_dict : dict) -> Tensor:
#     '''Edge Loss'''
#     # Only apply subnode loss to constraints that are not none -------
#     suba_loss = F.mse_loss(input = pred_edge_noise[...,0:4], target = true_edge_noise[...,0:4], reduction = 'mean') * 0.1
#     subb_loss = F.mse_loss(input = pred_edge_noise[...,4:8], target = true_edge_noise[...,4:8], reduction = 'mean') * 0.1
#     constraint_loss = suba_loss = F.mse_loss(input = pred_edge_noise[...,8:], target = true_edge_noise[...,8:], reduction = 'mean') * 0.1

#     edge_loss = suba_loss + subb_loss + constraint_loss

#     '''Node Loss'''
#     isconstruct_loss = F.mse_loss(input = pred_node_noise[...,0:2], target = true_node_noise[...,0:2], reduction = 'mean')
#     primitive_type_loss = F.mse_loss(input = pred_node_noise[...,2:7], target = true_node_noise[...,2:7], reduction = 'mean')
#     parameter_loss = F.mse_loss(input = pred_node_noise[...,7:] * params_mask, target = true_node_noise[...,7:], reduction = 'mean')

#     node_loss = isconstruct_loss + primitive_type_loss + parameter_loss

#     total_loss = node_loss + edge_loss

#     loss_dict["edge loss"] = edge_loss.item()
#     loss_dict["edge sub_a loss"] = suba_loss.item()
#     loss_dict["edge sub_b loss"] = subb_loss.item()
#     loss_dict["edge type loss"] = constraint_loss.item()
#     loss_dict["node loss"] = node_loss.item()
#     loss_dict["node isconstruct loss"] = isconstruct_loss.item()
#     loss_dict["node type loss"] = primitive_type_loss.item()
#     loss_dict["node parameter loss"] = parameter_loss.item()
#     loss_dict["total loss"] = total_loss.item()

#     return total_loss

# # %%
# def plot_loss(writer : SummaryWriter, loss_dict : dict, step : int):
#     writer.add_scalar("Training/Total_Loss", loss_dict["total loss"],            step)
#     writer.add_scalar("Training/Node_Loss",  loss_dict["node loss"],             step)
#     writer.add_scalar("Training/Node_BCE",   loss_dict["node isconstruct loss"], step)
#     writer.add_scalar("Training/Node_Cross", loss_dict["node type loss"],        step)
#     writer.add_scalar("Training/Node_MAE",   loss_dict["node parameter loss"],   step)
#     writer.add_scalar("Training/Edge_Loss",  loss_dict["edge loss"],             step)
#     writer.add_scalar("Training/Edge_sub_a", loss_dict["edge sub_a loss"],       step)
#     writer.add_scalar("Training/Edge_sub_b", loss_dict["edge sub_b loss"],       step)
#     writer.add_scalar("Training/Edge_Cross", loss_dict["edge type loss"],        step)

# # # %% [markdown]
# # # ### Train Loop

# # # %%
# # num_train_iters = 50000
# # t_max = 499
# # lr = 2e-4
# # experiment_string = f"gd3pm_gumbeldiff_Adam_24attnblocks16heads512node256edge"
# # writer = SummaryWriter(f'runs3/{experiment_string}')
# # gpu_id = 2
# # render_freq = 100

# # tensor_dict = torch.load('temp_dataset.pth')

# # nodes = tensor_dict["nodes"].to(gpu_id)
# # edges = tensor_dict["edges"].to(gpu_id)
# # params_mask = tensor_dict["params_mask"].to(gpu_id)

# # # %%
# # model = GD3PM(gpu_id)
# # optimizer = torch.optim.Adam(model.parameters(), lr = lr)
# # timestep_distribution = torch.distributions.beta.Beta(torch.tensor([1.1]), torch.tensor([3.9]))

# # # %%
# # model.train()
# # pbar = tqdm(range(num_train_iters))
# # for step in pbar:
# #     optimizer.zero_grad()

# #     t = torch.randint(low = 1, high = model.max_timestep, size = (16,))
        
# #     noised_nodes, noised_edges, true_node_noise, true_edge_noise = model.noise_scheduler(nodes, edges, t)

# #     pred_node_noise, pred_edge_noise = model(noised_nodes, noised_edges, t)

# #     loss_dict = {}
# #     loss = diffusion_loss(pred_node_noise, pred_edge_noise, true_node_noise, true_edge_noise, params_mask, loss_dict)

# #     plot_loss(writer, loss_dict, step)
# #     pbar.set_description(f"Iter Loss: {loss.item()}")

# #     loss.backward()
# #     # torch.nn.utils.clip_grad_norm_(model.parameters(), 5e-1)
# #     optimizer.step()

# # # %% [markdown]
# # # ### Validate Loop

# # # %%
# # model.eval()
# # graph_idx = 2
# # frames = 20
# # fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(16, 4))
# # fig.suptitle("True - Noised - Next")

# # def animation_fun(frame : int, curr_nodes : Tensor, curr_edges : Tensor, axes):
# #     timestep = frames - frame + 1

# #     pred_node_noise, pred_edge_noise = model(curr_nodes, curr_edges, torch.Tensor([timestep]).int())

# #     # Reverse Step
# #     new_nodes, new_edges = model.reverse_step(curr_nodes, curr_edges, pred_node_noise, pred_edge_noise, timestep)
# #     # curr_nodes, curr_edges = model.reverse_step(curr_nodes, curr_edges, nodes[graph_idx].unsqueeze(0), edges[graph_idx].unsqueeze(0), timestep)
    
# #     for ax in axes:
# #         ax.cla()
    
# #     SketchDataset.render_graph(nodes[graph_idx].cpu().squeeze(0), edges[graph_idx].cpu().squeeze(0), axes[0])
# #     SketchDataset.render_graph(curr_nodes.cpu().squeeze(0), curr_edges.cpu().squeeze(0), axes[1])
# #     # SketchDataset.render_graph(pred_nodes.cpu().squeeze(0), pred_edges.cpu().squeeze(0), axes[2])
# #     SketchDataset.render_graph(new_nodes.cpu().squeeze(0), new_edges.cpu().squeeze(0), axes[2])

# #     curr_nodes[...] = new_nodes
# #     curr_edges[...] = new_edges

# # curr_nodes, curr_edges, _, _ = model.noise_scheduler(nodes[graph_idx].unsqueeze(0), edges[graph_idx].unsqueeze(0), frames + 1)
# # animation = FuncAnimation(fig = fig, func = partial(animation_fun, curr_nodes = curr_nodes, curr_edges = curr_edges, axes = axes), frames = frames, interval = 200, repeat_delay = 1000)

# # video = animation.to_html5_video()
# # display.display(display.HTML(video))
# # plt.close()



