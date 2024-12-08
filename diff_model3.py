import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, device
from torch.utils.checkpoint import checkpoint
from config import NODE_FEATURE_DIMENSION, EDGE_FEATURE_DIMENSION, MAX_NUM_PRIMITIVES
from dataset1 import SketchDataset
from matplotlib import pyplot as plt

class GD3PM(nn.Module):
  def __init__(self, device : torch.device):
    super().__init__()
    self.device = device
    self.node_dim = NODE_FEATURE_DIMENSION
    self.edge_dim = EDGE_FEATURE_DIMENSION
    self.node_hidden_dim = 256
    self.edge_hidden_dim = 256
    self.cond_hidden_dim = 256
    self.num_tf_layers = 48
    self.num_checkpoints = 35
    self.num_heads = 8
    self.max_timestep = 1000
    self.noise_scheduler = CosineNoiseScheduler(self.max_timestep, self.device)
    self.architecture = DiffusionModel(node_dim = self.node_dim, 
                                       edge_dim = self.edge_dim, 
                                       node_hidden_dim = self.node_hidden_dim,
                                       edge_hidden_dim = self.edge_hidden_dim,
                                       cond_hidden_dim = self.cond_hidden_dim,
                                       num_heads = self.num_heads,
                                       num_tf_layers = self.num_tf_layers,
                                       num_checkpoints = self.num_checkpoints,
                                       max_timestep = self.max_timestep,
                                       device = self.device)

  def forward(self, nodes : Tensor, edges : Tensor, timestep : Tensor):
    return self.architecture(nodes, edges, timestep)
  
  @torch.no_grad()
  def sample(self, batch_size : int):
    # Sample Noise
    nodes, edges = self.noise_scheduler.sample_latent(batch_size)
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
      # Normalize output into probabilities
      pred_node_noise[:,:,0] = F.sigmoid(input = pred_node_noise[:,:,0])
      pred_node_noise[:,:,1:6] = F.softmax(input = pred_node_noise[:,:,1:6], dim = 2)
      pred_edge_noise[:,:,:,0:4] = F.softmax(input = pred_edge_noise[:,:,:,0:4], dim = 3)
      pred_edge_noise[:,:,:,4:8] = F.softmax(input = pred_edge_noise[:,:,:,4:8], dim = 3)
      pred_edge_noise[:,:,:,8:] = F.softmax(input = pred_edge_noise[:,:,:,8:], dim = 3)
      nodes, edges = self.reverse_step(nodes, edges, pred_node_noise, pred_edge_noise, t)

    return nodes, edges
  
  @torch.no_grad()
  def noise(self, nodes, edges):
    nodes, edges, _ = self.noise_scheduler(nodes, edges, self.max_timestep)
    return nodes, edges
  
  @torch.no_grad()
  def reverse_step(self, curr_nodes : Tensor, curr_edges : Tensor, pred_nodes : Tensor, pred_edges : Tensor, timestep : int):
    new_nodes = torch.zeros_like(curr_nodes)
    new_edges = torch.zeros_like(curr_edges)
    # IsConstructible denoising
    new_nodes[:,:,0] = self.noise_scheduler.apply_bernoulli_posterior_step(curr_nodes[:,:,0], pred_nodes[:,:,0], timestep)
    # Primitive Types denoising
    new_nodes[:,:,1:6] = self.noise_scheduler.apply_multinomial_posterior_step(curr_nodes[:,:,1:6], pred_nodes[:,:,1:6], timestep)
    # Primitive parameters denoising
    new_nodes[:,:,6:] = self.noise_scheduler.apply_gaussian_posterior_step(curr_nodes[:,:,6:], pred_nodes[:,:,6:], timestep)
    # Subnode A denoising
    new_edges[:,:,:,0:4] = self.noise_scheduler.apply_multinomial_posterior_step(curr_edges[:,:,:,0:4], pred_edges[:,:,:,0:4], timestep)
    # Subnode B denoising
    new_edges[:,:,:,4:8] = self.noise_scheduler.apply_multinomial_posterior_step(curr_edges[:,:,:,4:8], pred_edges[:,:,:,4:8], timestep)
    # Constraint Types denoising
    new_edges[:,:,:,8:] = self.noise_scheduler.apply_multinomial_posterior_step(curr_edges[:,:,:,8:], pred_edges[:,:,:,8:], timestep)
    return new_nodes, new_edges

class DiffusionModel(nn.Module):
    def __init__(self, node_dim, edge_dim, node_hidden_dim, edge_hidden_dim, cond_hidden_dim, num_heads, num_tf_layers, num_checkpoints, max_timestep, device: device):
        super().__init__()
        self.device = device
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.cond_hidden_dim = cond_hidden_dim
        self.num_heads = num_heads
        self.num_tf_layers = num_tf_layers
        self.num_checkpoints = num_checkpoints
        self.max_timestep = max_timestep

        self.time_embedder = TimeEmbedder(self.max_timestep, self.cond_hidden_dim, self.device)

        # Input MLP layers
        self.mlp_in_nodes = nn.Sequential(
            nn.Linear(in_features = self.node_dim, out_features = self.node_hidden_dim, device = device),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features = self.node_hidden_dim, out_features = self.node_hidden_dim, device = device),
            nn.LeakyReLU(0.01)
        )
        
        self.mlp_in_edges = nn.Sequential(
            nn.Linear(in_features = self.edge_dim, out_features = self.edge_hidden_dim, device = device),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features = self.edge_hidden_dim, out_features = self.edge_hidden_dim, device = device),
            nn.LeakyReLU(0.01)
        )

        self.mlp_in_conds = nn.Sequential(
            nn.Linear(in_features = self.cond_hidden_dim, out_features = self.cond_hidden_dim, device = device),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features = self.cond_hidden_dim, out_features = self.cond_hidden_dim, device = device),
            nn.LeakyReLU(0.01)
        )

        # Transformer Layers with Graph Attention Network
        self.block_layers = nn.ModuleList([
            TransformerLayer(
                node_dim = self.node_hidden_dim,
                edge_dim = self.edge_hidden_dim,
                cond_dim = self.cond_hidden_dim,
                num_heads = self.num_heads,
                device = self.device
            ) for _ in range(self.num_tf_layers)
        ])
        
        # Output MLP layers
        self.mlp_out_node_types = nn.Sequential(
            nn.Linear(in_features = self.node_hidden_dim, out_features = self.node_hidden_dim, device = device),
            nn.LeakyReLU(0.01),
            # nn.Linear(in_features = self.node_hidden_dim, out_features = self.node_hidden_dim, device = device),
            # nn.LeakyReLU(0.1),
            nn.Linear(in_features = self.node_hidden_dim, out_features = 6, device = device)
        )
        self.mlp_out_node_params = nn.Sequential(
            nn.Linear(in_features = self.node_hidden_dim, out_features = self.node_hidden_dim, device = device),
            nn.LeakyReLU(0.01),
            # nn.Linear(in_features = self.node_hidden_dim, out_features = self.node_hidden_dim, device = device),
            # nn.LeakyReLU(0.1),
            nn.Linear(in_features = self.node_hidden_dim, out_features = 14, device = device)
        )
        
        self.mlp_out_edges = nn.Sequential(
            nn.Linear(in_features = self.edge_hidden_dim, out_features = self.edge_hidden_dim, device = device),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features = self.edge_hidden_dim, out_features = self.edge_dim, device = device)
        )

    def forward(self, nodes : Tensor, edges : Tensor, timestep : Tensor):
        nodes = self.mlp_in_nodes(nodes)     # shape: (batch_size, num_nodes, node_hidden_dim)
        edges = self.mlp_in_edges(edges)     # shape: (batch_size, num_nodes, num_nodes, edge_hidden_dim)
        conds = self.time_embedder(timestep) # shape: (batch_size, cond_hidden_dim)
        conds = self.mlp_in_conds(conds)     # shape: (batch_size, cond_hidden_dim)

        # nodes = nodes + conds.unsqueeze(1)
        # edges = edges + conds.unsqueeze(1).unsqueeze(1)
        checkpoints = self.num_checkpoints
        for layer in self.block_layers:
            nodes, edges = checkpoint(layer, nodes, edges, conds, use_reentrant = False) if checkpoints > 0 else layer(nodes, edges, conds) # shape: (batch_size, num_nodes, node_hidden_dim) ; shape: (batch_size, num_nodes, num_nodes, edge_hidden_dim)
            checkpoints = checkpoints - 1

        nodes = torch.cat([self.mlp_out_node_types(nodes), self.mlp_out_node_params(nodes)], dim = -1) # shape: (batch_size, num_nodes, node_dim)
        edges = self.mlp_out_edges(edges) # shape: (batch_size, num_nodes, num_nodes, edge_dim)

        return nodes, edges

class TransformerLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, cond_dim: int, num_heads: int, device: device):
        super().__init__()
        num_nodes = MAX_NUM_PRIMITIVES

        # Attention Layer
        self.attention_heads = MHA2(
            node_dim = node_dim,
            edge_dim = edge_dim,
            cond_dim = cond_dim,
            num_heads = num_heads,
            device = device
        )

        # Normalization and Residual Connections
        self.norm_attn_nodes = nn.InstanceNorm1d(num_features = node_dim, affine = True, device = device)
        self.norm_attn_edges = nn.InstanceNorm2d(num_features = edge_dim, affine = True, device = device)

        # Node and edge MLPs with residual connections
        self.mlp_nodes = nn.Sequential(
            nn.Linear(in_features = node_dim, out_features = node_dim, device = device),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features = node_dim, out_features = node_dim, device = device),
            nn.LeakyReLU(0.01)
        )

        self.mlp_edges = nn.Sequential(
            nn.Linear(in_features = edge_dim, out_features = edge_dim, device = device),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features = edge_dim, out_features = edge_dim, device = device),
            nn.LeakyReLU(0.01)
        )

        self.norm_nodes = nn.InstanceNorm1d(num_features = node_dim, affine = True, device = device)
        self.norm_edges = nn.InstanceNorm2d(num_features = edge_dim, affine = True, device = device)

    def forward(self, nodes : Tensor, edges : Tensor, conds : Tensor) -> tuple[Tensor, Tensor]:
        attn_nodes, attn_edges = self.attention_heads(nodes, edges, conds)

        nodes = self.norm_attn_nodes((nodes + attn_nodes).permute(0, 2, 1)).permute(0, 2, 1)
        edges = self.norm_attn_edges((edges + attn_edges).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # nodes = self.norm_attn_nodes(nodes + attn_nodes)
        # edges = self.norm_attn_edges(edges + attn_edges)

        # MLP
        new_nodes = self.mlp_nodes(nodes)
        new_edges = self.mlp_edges(edges)

        new_nodes = self.norm_nodes((nodes + new_nodes).permute(0, 2, 1)).permute(0, 2, 1)
        new_edges = self.norm_edges((edges + new_edges).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # new_nodes = self.norm_nodes(nodes + new_nodes)
        # new_edges = self.norm_edges(edges + new_edges)

        return new_nodes, new_edges

class MHA2(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, cond_dim: int, num_heads: int, device: torch.device):
        super().__init__()
        self.num_heads = num_heads
        feature_dim = node_dim
        value_dim = node_dim
        self.weight_dim = node_dim

        # Conditioning
        self.lin_features = nn.Sequential(
            nn.Linear(in_features = edge_dim + 2 * node_dim, out_features = feature_dim, device = device)
        )
        self.lin_cond_add_features = nn.Sequential(
            nn.Linear(in_features = cond_dim, out_features = feature_dim, device = device)
        )
        self.lin_cond_mul_features = nn.Sequential(
            nn.Linear(in_features = cond_dim, out_features = feature_dim, device = device)
        )

        # New Edges
        self.lin_edges = nn.Sequential(
            nn.Linear(in_features = feature_dim, out_features = edge_dim, device = device)
        )

        # New Nodes
        self.lin_values = nn.Sequential(
            nn.Linear(in_features = feature_dim, out_features = value_dim, device = device),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features = value_dim, out_features = value_dim, device = device)
        )
        self.lin_weights = nn.Sequential(
            nn.Linear(in_features = feature_dim, out_features = self.weight_dim, device = device),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features = self.weight_dim, out_features = self.weight_dim, device = device)
        )
        
        self.lin_nodes = nn.Sequential(
            nn.Linear(in_features = value_dim, out_features = node_dim, device = device)
        )

    def forward(self, nodes : Tensor, edges : Tensor, cond : Tensor) -> tuple[Tensor, Tensor]:
        b, n, d = nodes.size()
        
        cond = cond.unsqueeze(1).unsqueeze(1) # (b, 1, 1, d)

        # Create Features ----
        graph_features = torch.cat([edges, nodes.unsqueeze(1).expand(b, n, n, d), nodes.unsqueeze(2).expand(b, n, n, d)], dim = -1) # (b, n, n, de + 2 * dn)

        # Condition on timestep
        graph_features = F.leaky_relu(self.lin_features(graph_features) * self.lin_cond_mul_features(cond) + self.lin_cond_add_features(cond), 0.01) # (b, n, n, de + 2 * dn)

        # New Edges ----
        new_edges = F.leaky_relu(self.lin_edges(graph_features), 0.01) # (b, n, n, de)
        
        # New Nodes ----
        h = self.num_heads

        # Project Values
        values = self.lin_values(graph_features).reshape(b, n, n, h, -1).permute(0, 3, 1, 2, 4) # (b, h, n, n, dv/h )

        # Calculate Attention Weights
        weights = self.lin_weights(graph_features).reshape(b, n, n, h, -1).permute(0, 3, 1, 2, 4) # (b, h, n, n, dw/h)
        weights = (weights ** 2 / self.weight_dim).sum(dim = -1, keepdim = True).sqrt().softmax(dim = -2) # (b, h, n, n, 1)
        weights = weights.permute(0, 1, 2, 4, 3) # (b, h, n, 1, n)

        # Aggregate Information Across Nodes
        weighted_values = (weights @ values).permute(0, 2, 1, 3, 4).flatten(2) # (b, n, dv)

        # Project Values Back
        new_nodes = F.leaky_relu(self.lin_nodes(weighted_values), 0.01) # (b, n, dn)

        return new_nodes, new_edges

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
  
class CosineNoiseScheduler(nn.Module):
  def __init__(self, max_timestep : int, device : torch.device):
    super().__init__()
    self.device = device
    self.max_timestep = max_timestep
    self.offset = .008 # Fixed offset to improve noise prediction at early timesteps
    # Cosine Beta Schedule Formula: https://arxiv.org/abs/2102.09672     1.00015543316 is 1/a(0), for offset = .008
    self.cumulative_precisions = torch.cos((torch.linspace(0, 1, self.max_timestep + 1).to(self.device) + self.offset) * 0.5 * math.pi / (1 + self.offset)) ** 2 * 1.00015543316
    self.cumulative_variances = 1 - self.cumulative_precisions
    self.variances = torch.cat([torch.Tensor([0]).to(self.device), 1 - (self.cumulative_precisions[1:] / self.cumulative_precisions[:-1])]).clamp(.0001, .9999)
    self.precisions = 1 - self.variances
    self.sqrt_cumulative_precisions = torch.sqrt(self.cumulative_precisions)
    self.sqrt_cumulative_variances = torch.sqrt(self.cumulative_variances)
    self.sqrt_precisions = torch.sqrt(self.precisions)
    self.sqrt_variances = torch.sqrt(self.variances)
    self.sqrt_posterior_variances = torch.cat([torch.Tensor([0]).to(self.device), torch.sqrt(self.variances[1:] * self.cumulative_variances[:-1] / self.cumulative_variances[1:])])

    # Power Schedule
    self.disc_cumulative_precisions = 1 - torch.linspace(0, 1, self.max_timestep + 1).to(self.device) ** (1/16)
    self.disc_cumulative_variances = 1 - self.disc_cumulative_precisions
    self.disc_variances = torch.cat([torch.Tensor([0]).to(self.device), 1 - (self.disc_cumulative_precisions[1:] / self.disc_cumulative_precisions[:-1])]).clamp(.0001, .9999)
    self.disc_precisions = 1 - self.disc_variances
    self.sqrt_disc_cumulative_precisions = torch.sqrt(self.disc_cumulative_precisions)
    self.sqrt_disc_cumulative_variances = torch.sqrt(self.disc_cumulative_variances)
    self.sqrt_disc_precisions = torch.sqrt(self.disc_precisions)
    self.sqrt_disc_variances = torch.sqrt(self.disc_variances)
    self.sqrt_disc_posterior_variances = torch.cat([torch.Tensor([0]).to(self.device), torch.sqrt(self.disc_variances[1:] * self.disc_cumulative_variances[:-1] / self.disc_cumulative_variances[1:])])
  
    self.clip_value = -10
  def forward(self, nodes : Tensor, edges : Tensor, timestep : Tensor):
    ''' Apply noise to graph '''
    noisy_nodes = torch.zeros(size = nodes.size(), device = nodes.device)
    noisy_edges = torch.zeros(size = edges.size(), device = edges.device)
    # nodes = batch_size x num_nodes x NODE_FEATURE_DIMENSION ; edges = batch_size x num_nodes x num_nodes x EDGE_FEATURE_DIMENSION
    bernoulli_is_constructible = nodes[:,:,0] # batch_size x num_nodes x 1
    categorical_primitive_types = nodes[:,:,1:6] # batch_size x num_nodes x 5
    gaussian_primitive_parameters = nodes[:,:,6:] # batch_size x num_nodes x 14
    # subnode just means if the constraint applies to the start, center, or end of a primitive
    categorical_subnode_a_types = edges[:,:,:,0:4] # batch_size x num_nodes x 4
    categorical_subnode_b_types = edges[:,:,:,4:8] # batch_size x num_nodes x 4
    categorical_constraint_types = edges[:,:,:,8:] # batch_size x num_nodes x 9
    # IsConstructible noise
    noisy_nodes[:,:,0] = self.apply_binary_noise(bernoulli_is_constructible, timestep)
    # Primitive Types noise
    noisy_nodes[:,:,1:6] = self.apply_discrete_noise(categorical_primitive_types, timestep) # noised_primitive_types
    # Primitive parameters noise
    gaussian_noise = torch.randn_like(gaussian_primitive_parameters) # standard gaussian noise
    noisy_nodes[:,:,6:] = self.apply_gaussian_noise(gaussian_primitive_parameters, timestep, gaussian_noise)
    # Subnode A noise
    noisy_edges[:,:,:,0:4] = self.apply_discrete_noise(categorical_subnode_a_types, timestep) # noised_subnode_a_types
    # Subnode B noise
    noisy_edges[:,:,:,4:8] = self.apply_discrete_noise(categorical_subnode_b_types, timestep) # noised_subnode_a_types
    # Constraint Types noise
    noisy_edges[:,:,:,8:] = self.apply_discrete_noise(categorical_constraint_types, timestep) # noised_constraint_types
    return noisy_nodes, noisy_edges, gaussian_noise
  
  def get_transition_noise(self, parameters : Tensor, timestep : int, gaussian_noise : Tensor = None):
    return self.sqrt_precisions[timestep] * parameters + self.sqrt_variances[timestep] * gaussian_noise
  
  def apply_gaussian_noise(self, parameters : Tensor, timestep : Tensor | int, gaussian_noise : Tensor = None):
    if type(timestep) is int: timestep = [timestep]
    # parameters shape is batch_size x num_nodes x num_params
    # gaussian_noise shape is batch_size x num_nodes x num_params
    batched_precisions = self.sqrt_cumulative_precisions[timestep,None,None] # (b,1,1) or (1,1,1)
    batched_variances = self.sqrt_cumulative_variances[timestep,None,None]   # (b,1,1) or (1,1,1)
    return batched_precisions * parameters + batched_variances * gaussian_noise
  
  def apply_gaussian_posterior_step(self, curr_params : Tensor, pred_params : Tensor, timestep : int):
    if timestep > 1:
      sqrt_prev_cumul_prec = self.sqrt_cumulative_precisions[timestep - 1]
      var = self.variances[timestep]
      sqrt_prec = self.sqrt_precisions[timestep]
      prev_cumul_var = self.cumulative_variances[timestep - 1]
      cumul_var = self.cumulative_variances[timestep]
      
      denoised_mean = (sqrt_prev_cumul_prec * var * pred_params + sqrt_prec * prev_cumul_var * curr_params) / cumul_var

      gaussian_noise = torch.randn_like(curr_params)
      return denoised_mean + gaussian_noise * self.sqrt_posterior_variances[timestep]
    else:
      return pred_params # denoised_mean
    
  def get_transition_matrix(self, dimension : int, timestep : int | Tensor):
    if type(timestep) is int: 
      assert timestep > 0; 
      assert timestep < self.max_timestep; 
      timestep = [timestep]
    
    batched_precisions = self.precisions[timestep,None,None] # (batch_size, 1, 1) or (1, 1, 1)
    return batched_precisions * torch.eye(dimension, dtype = torch.float32, device = self.device) + (1 - batched_precisions) / dimension # (batch_size, d, d) or (1, d, d)
  
  def get_cumulative_transition_matrix(self, dimension : int, timestep : int | Tensor):
    if type(timestep) is int: 
      assert timestep > 0; 
      assert timestep < self.max_timestep; 
      timestep = [timestep]

    batched_precisions = self.cumulative_precisions[timestep,None,None] # (batch_size, 1, 1) or (1, 1, 1)
    return batched_precisions * torch.eye(dimension, dtype = torch.float32, device = self.device) + (1 - batched_precisions) / dimension # (batch_size, d, d) or (1, d, d)
  
  def get_posterior_transition_matrix(self, xt : Tensor, timestep : Tensor | int) -> torch.Tensor:
    xt_size, xt = self.flatten_middle(xt) # (b, n, d) or (b, n * n, d), for convenience let m = n or n * n
    d = xt_size[-1]
    qt = xt @ self.get_transition_matrix(d, timestep).permute(0, 2, 1) # (b, m, d), since xt is onehot we are plucking out rows corresponding to p(x_t = class | x_(t-1))
    qt_bar = xt @ self.get_cumulative_transition_matrix(d, timestep).permute(0, 2, 1) # (b, m, d), since xt is onehot we are plucking out rows corresponding to p(x_t = class | x_0)
    q = qt.unsqueeze(2) / qt_bar.unsqueeze(3) # (b, m, d, d), perform an outer product so element at (b, m, i, j) = p(x_t = class | x_(t-1) = j) / p(x_t = class | x_0 = i)
    q = q * self.get_cumulative_transition_matrix(d, timestep - 1).unsqueeze(1) # (b, m, d, d), broadcast multiply so element at (b, m, i, j) = p(x_t = class | x_(t-1) = j) * p(x_(t-1) = j | x_0 = i) / p(x_t = class | x_0 = i)
    return q.view(size = xt_size + (d,)) # reshape into (b, n, d, d) or (b, n, n, d, d)
  
  def apply_discrete_noise(self, x_one_hot : Tensor, timestep : Tensor | int):
    # size, x = self.flatten_middle(x_one_hot)
    # q = self.get_cumulative_transition_matrix(size[-1], timestep) # (b, d, d) or (1, d, d)
    # distribution = x @ q # (b, n, d) or (b, n * n, d)
    # distribution = distribution.view(size) # (b, n, d) or (b, n, n, d)
    # return self.sample_discrete_distribution(distribution).float()
    return self.apply_disc_noise(x_one_hot, timestep)
  
  def apply_disc_noise(self, x : Tensor, timestep : Tensor | int):
    size, x = self.flatten_middle(x)
    logits = x.log().clamp(self.clip_value)
    noise = torch.randn_like(logits)

    batched_precisions = self.sqrt_disc_cumulative_precisions[timestep,None,None] # (b,1,1) or (1,1,1)
    batched_variances = self.sqrt_disc_cumulative_variances[timestep,None,None]   # (b,1,1) or (1,1,1)

    return torch.softmax(batched_precisions * logits + batched_variances * noise, dim = -1).reshape(size)

  def apply_disc_posterior_step(self, curr_class_probs : Tensor, pred_class_probs : Tensor, timestep : int):
    if timestep > 1:
      sqrt_prev_cumul_prec = self.sqrt_disc_cumulative_precisions[timestep - 1]
      var = self.disc_variances[timestep]
      sqrt_prec = self.sqrt_disc_precisions[timestep]
      prev_cumul_var = self.disc_cumulative_variances[timestep - 1]
      cumul_var = self.disc_cumulative_variances[timestep]

      # class_probs_size, class_probs = self.flatten_middle(curr_class_probs)
      # pred_probs_size, pred_probs = self.flatten_middle(pred_class_probs)

      curr_logits = curr_class_probs.log().clamp(self.clip_value)
      pred_logits = pred_class_probs.log().clamp(self.clip_value)
      
      denoised_mean = (sqrt_prev_cumul_prec * var * pred_logits + sqrt_prec * prev_cumul_var * curr_logits) / cumul_var

      gaussian_noise = torch.randn_like(curr_class_probs)
      return torch.softmax(denoised_mean + gaussian_noise * self.sqrt_disc_posterior_variances[timestep], dim = -1) # .reshape(pred_probs_size)
    else:
      return pred_class_probs # denoised_mean
  
  def apply_multinomial_posterior_step(self, classes_one_hot : Tensor, pred_class_probs : Tensor, timestep : int):
    # # classes_one_hot = (b, n, d) or (b, n, n, d)
    # # pred_class_probs = (b, n, d) or (b, n, n, d)
    # if timestep > 1:
    #   q = self.get_posterior_transition_matrix(classes_one_hot, timestep) # (b, n, d, d) or (b, n, n, d, d)
    #   pred_class_probs = pred_class_probs.unsqueeze(-2) # (b, n, 1, d) or (b, n, n, 1, d), make probs into row vector
    #   posterior_distribution = pred_class_probs @ q # (b, n, 1, d) or (b, n, n, 1, d), batched vector-matrix multiply
    #   posterior_distribution = posterior_distribution.squeeze(-2) # (b, n, d) or (b, n, n, d)
    #   return self.sample_discrete_distribution(posterior_distribution).float()
    # else:
    #   return pred_class_probs

    return self.apply_disc_posterior_step(classes_one_hot, pred_class_probs, timestep)
    
  def apply_binary_noise(self, boolean_flag : Tensor, timestep : int | Tensor):
    boolean_flag = boolean_flag.unsqueeze(-1)
    one_hot = torch.cat([1 - boolean_flag, boolean_flag], dim = -1) # (b, n, 2)
    noised_one_hot = self.apply_discrete_noise(one_hot, timestep) # (b, n, 2)
    return noised_one_hot[...,1] # (b, n)
  
  def apply_bernoulli_posterior_step(self, boolean_flag : Tensor, pred_boolean_prob : Tensor, timestep : int):
    if timestep > 1:
      boolean_flag = boolean_flag.unsqueeze(-1) # b, n, 1
      pred_boolean_prob = pred_boolean_prob.unsqueeze(-1) # b, n, 1
      one_hot_xt = torch.cat([1 - boolean_flag, boolean_flag], dim = -1) # (b, n, 2)
      probs = torch.cat([1 - pred_boolean_prob, pred_boolean_prob], dim = -1) # (b, n, 2)
      noised_one_hot = self.apply_multinomial_posterior_step(one_hot_xt, probs, timestep) # (b, n, 2)
      return noised_one_hot[...,1] # (b, n)
    else:
      return pred_boolean_prob
    
  def sample_discrete_distribution(self, tensor : Tensor):
    # size = tensor.size()
    # num_classes = size[-1]
    # return F.one_hot(tensor.reshape(-1, num_classes).multinomial(1), num_classes).reshape(size)

    noise = -torch.log(-torch.log(torch.rand_like(tensor))) # self.gumbel_dist.sample(tensor.shape).squeeze(-1).to(self.device)
    return torch.softmax(tensor.log() + noise, dim = -1)
  
  def flatten_middle(self, x : Tensor):
    prev_size = x.size() # shape of x_one_hot is (b, n, d) or (b, n, n, d)
    return prev_size, x.view(prev_size[0], -1, prev_size[-1]) # (b, n, d) or (b, n * n, d)
  
  def sample_latent(self, batch_size : int) -> tuple[Tensor, Tensor] :
    nodes = torch.zeros(size = (batch_size, MAX_NUM_PRIMITIVES, NODE_FEATURE_DIMENSION), device = self.device)
    edges = torch.zeros(size = (batch_size, MAX_NUM_PRIMITIVES, MAX_NUM_PRIMITIVES, EDGE_FEATURE_DIMENSION), device = self.device)

    # Boolean Variable for IsConstructible
    nodes[...,:0] = torch.randn(size = (batch_size, MAX_NUM_PRIMITIVES, 1), device = self.device).sigmoid()
    # Multinomial for Node type
    nodes[...,1:6] = torch.randn(size = (batch_size, MAX_NUM_PRIMITIVES, 5), device = self.device).softmax(dim = -1)
    # Gaussian for Node parameters
    nodes[...,6:] = torch.randn(size = (batch_size, MAX_NUM_PRIMITIVES, 14), device = self.device)

    # Multinomial for Subnode A type
    edges[...,0:4] = torch.randn(size = (batch_size, MAX_NUM_PRIMITIVES, MAX_NUM_PRIMITIVES, 4), device = self.device).softmax(dim = -1)
    # Multinomial for Subnode B type
    edges[...,4:8] = torch.randn(size = (batch_size, MAX_NUM_PRIMITIVES, MAX_NUM_PRIMITIVES, 4), device = self.device).softmax(dim = -1)
    # Multinomial for Edge type
    edges[...,8:] = torch.randn(size = (batch_size, MAX_NUM_PRIMITIVES, MAX_NUM_PRIMITIVES, 9), device = self.device).softmax(dim = -1)

    return nodes, edges

# class GD3PM(nn.Module):
#   def __init__(self, device : device):
#     super().__init__()
#     self.device = device
#     self.node_dim = NODE_FEATURE_DIMENSION
#     self.edge_dim = EDGE_FEATURE_DIMENSION
#     self.node_hidden_dim = 256
#     self.edge_hidden_dim = 128
#     self.cond_hidden_dim = 128
#     self.num_tf_layers = 32
#     self.num_checkpoints = 0
#     self.num_heads = 8
#     self.max_timestep = 1000
#     self.max_steps = self.max_timestep + 1
#     self.noise_scheduler = CosineNoiseScheduler(self.max_steps, self.device)
#     self.architecture = DiffusionModel(node_dim = self.node_dim, 
#                                        edge_dim = self.edge_dim, 
#                                        node_hidden_dim = self.node_hidden_dim,
#                                        edge_hidden_dim = self.edge_hidden_dim,
#                                        cond_hidden_dim = self.cond_hidden_dim,
#                                        num_heads = self.num_heads,
#                                        num_tf_layers = self.num_tf_layers,
#                                        max_timestep = self.max_timestep,
#                                        device = self.device)

#   def forward(self, nodes : Tensor, edges : Tensor, timestep : Tensor):
#     return self.architecture(nodes, edges, timestep)
  
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
#     for t in reversed(range(1, self.max_steps)):
#       # model expects a timestep for each batch
#       batch_size = nodes.size(0)
#       time = torch.Tensor([t]).expand(batch_size).int()
#       pred_node_noise, pred_edge_noise = self.forward(nodes, edges, time)
#       # Normalize output into probabilities
#       pred_node_noise[:,:,0] = F.sigmoid(input = pred_node_noise[:,:,0])
#       pred_node_noise[:,:,1:6] = F.softmax(input = pred_node_noise[:,:,1:6], dim = 2)
#       pred_edge_noise[:,:,:,0:4] = F.softmax(input = pred_edge_noise[:,:,:,0:4], dim = 3)
#       pred_edge_noise[:,:,:,4:8] = F.softmax(input = pred_edge_noise[:,:,:,4:8], dim = 3)
#       pred_edge_noise[:,:,:,8:] = F.softmax(input = pred_edge_noise[:,:,:,8:], dim = 3)
#       nodes, edges = self.reverse_step(nodes, edges, pred_node_noise, pred_edge_noise, t)

#     return nodes, edges
  
#   @torch.no_grad()
#   def noise(self, nodes, edges):
#     nodes, edges, _ = self.noise_scheduler(nodes, edges, self.max_timestep)
#     return nodes, edges
  
#   @torch.no_grad()
#   def reverse_step(self, curr_nodes : Tensor, curr_edges : Tensor, pred_nodes : Tensor, pred_edges : Tensor, timestep : int):
#     new_nodes = torch.zeros_like(curr_nodes)
#     new_edges = torch.zeros_like(curr_edges)
#     # IsConstructible denoising
#     new_nodes[:,:,0] = self.noise_scheduler.apply_bernoulli_posterior_step(curr_nodes[:,:,0], pred_nodes[:,:,0], timestep)
#     # Primitive Types denoising
#     new_nodes[:,:,1:6] = self.noise_scheduler.apply_multinomial_posterior_step(curr_nodes[:,:,1:6], pred_nodes[:,:,1:6], timestep)
#     # Primitive parameters denoising
#     new_nodes[:,:,6:] = self.noise_scheduler.apply_gaussian_posterior_step(curr_nodes[:,:,6:], pred_nodes[:,:,6:], timestep)
#     # Subnode A denoising
#     new_edges[:,:,:,0:4] = self.noise_scheduler.apply_multinomial_posterior_step(curr_edges[:,:,:,0:4], pred_edges[:,:,:,0:4], timestep)
#     # Subnode B denoising
#     new_edges[:,:,:,4:8] = self.noise_scheduler.apply_multinomial_posterior_step(curr_edges[:,:,:,4:8], pred_edges[:,:,:,4:8], timestep)
#     # Constraint Types denoising
#     new_edges[:,:,:,8:] = self.noise_scheduler.apply_multinomial_posterior_step(curr_edges[:,:,:,8:], pred_edges[:,:,:,8:], timestep)
#     return new_nodes, new_edges

# class DiffusionModel(nn.Module):
#     def __init__(self, node_dim, edge_dim, node_hidden_dim, edge_hidden_dim, cond_hidden_dim, num_heads, num_tf_layers, max_timestep, device: device):
#         super().__init__()
#         self.device = device
#         self.node_dim = node_dim
#         self.edge_dim = edge_dim
#         self.node_hidden_dim = node_hidden_dim
#         self.edge_hidden_dim = edge_hidden_dim
#         self.cond_hidden_dim = cond_hidden_dim
#         self.num_heads = num_heads
#         self.num_tf_layers = num_tf_layers
#         self.max_timestep = max_timestep

#         self.time_embedder = TimeEmbedder(self.max_timestep, self.cond_hidden_dim, self.device)

#         # Input MLP layers
#         self.mlp_in_nodes = nn.Sequential(
#             nn.Linear(in_features = self.node_dim, out_features = self.node_hidden_dim, device = device),
#             nn.ReLU(),
#             nn.Linear(in_features = self.node_hidden_dim, out_features = self.node_hidden_dim, device = device),
#             nn.ReLU()
#         )
        
#         self.mlp_in_edges = nn.Sequential(
#             nn.Linear(in_features = self.edge_dim, out_features = self.edge_hidden_dim, device = device),
#             nn.ReLU(),
#             nn.Linear(in_features = self.edge_hidden_dim, out_features = self.edge_hidden_dim, device = device),
#             nn.ReLU()
#         )

#         self.mlp_in_conds = nn.Sequential(
#             nn.Linear(in_features = self.cond_hidden_dim, out_features = self.cond_hidden_dim, device = device),
#             nn.ReLU(),
#             nn.Linear(in_features = self.cond_hidden_dim, out_features = self.cond_hidden_dim, device = device),
#             nn.ReLU()
#         )

#         # Transformer Layers with Graph Attention Network
#         self.block_layers = nn.ModuleList([
#             TransformerLayer(
#                 node_dim = self.node_hidden_dim,
#                 edge_dim = self.edge_hidden_dim,
#                 cond_dim = self.cond_hidden_dim,
#                 num_heads = self.num_heads,
#                 device = self.device
#             ) for _ in range(self.num_tf_layers)
#         ])
        
#         # Output MLP layers
#         self.mlp_out_nodes = nn.Sequential(
#             nn.Linear(in_features = self.node_hidden_dim, out_features = self.node_hidden_dim, device = device),
#             nn.ReLU(),
#             nn.Linear(in_features = self.node_hidden_dim, out_features = self.node_dim, device = device)
#         )
        
#         self.mlp_out_edges = nn.Sequential(
#             nn.Linear(in_features = self.edge_hidden_dim, out_features = self.edge_hidden_dim, device = device),
#             nn.ReLU(),
#             nn.Linear(in_features = self.edge_hidden_dim, out_features = self.edge_dim, device = device)
#         )

#     def forward(self, nodes : Tensor, edges : Tensor, timestep : Tensor):
#         nodes = self.mlp_in_nodes(nodes)     # shape: (batch_size, num_nodes, node_hidden_dim)
#         edges = self.mlp_in_edges(edges)     # shape: (batch_size, num_nodes, num_nodes, edge_hidden_dim)
#         conds = self.time_embedder(timestep) # shape: (batch_size, cond_hidden_dim)
#         conds = self.mlp_in_conds(conds)     # shape: (batch_size, cond_hidden_dim)

#         for layer in self.block_layers:
#             nodes, edges = layer(nodes, edges, conds) # shape: (batch_size, num_nodes, node_hidden_dim) ; shape: (batch_size, num_nodes, num_nodes, edge_hidden_dim)

#         nodes = self.mlp_out_nodes(nodes) # shape: (batch_size, num_nodes, node_dim)
#         edges = self.mlp_out_edges(edges) # shape: (batch_size, num_nodes, num_nodes, edge_dim)
#         return nodes, edges

# class TransformerLayer(nn.Module):
#     def __init__(self, node_dim: int, edge_dim: int, cond_dim: int, num_heads: int, device: device):
#         super().__init__()
#         num_nodes = MAX_NUM_PRIMITIVES

#         # Attention Layer
#         self.attention_heads = MultiHeadAttention(
#             node_dim = node_dim,
#             edge_dim = edge_dim,
#             cond_dim = cond_dim,
#             num_heads = num_heads,
#             device = device
#         )

#         # Custom normalization and residual connections
#         self.norm_nodes1 = NormalizationLayer(dim = node_dim, device = device)

#         # Node and edge MLPs with residual connections
#         self.mlp_nodes = nn.Sequential(
#             nn.Linear(in_features = node_dim, out_features = node_dim, device = device),
#             nn.ReLU(),
#             nn.Linear(in_features = node_dim, out_features = node_dim, device = device),
#             nn.ReLU()
#         )

#         self.norm_nodes2 = NormalizationLayer(dim = node_dim, device = device)
#         self.norm_edges = NormalizationLayer(dim = edge_dim, device = device)

#     def forward(self, nodes : Tensor, edges : Tensor, conds : Tensor) -> tuple[Tensor, Tensor]:
#         attn_nodes, attn_edges = self.attention_heads(nodes, edges, conds)

#         # Add & Norm layers with residual connections
#         nodes = self.norm_nodes1(nodes + attn_nodes)

#         new_nodes = self.mlp_nodes(nodes)

#         new_nodes = self.norm_nodes2(nodes + new_nodes)
#         new_edges = self.norm_edges(edges + attn_edges)

#         return new_nodes, new_edges

# class MultiHeadAttention(nn.Module):
#     def __init__(self, node_dim: int, edge_dim: int, cond_dim: int, num_heads: int, device: torch.device):
#         super().__init__()
#         self.num_heads = num_heads
#         feature_dim = edge_dim + 2 * node_dim
#         value_dim = node_dim

#         # Conditioning
#         self.lin_features = nn.Linear(in_features = feature_dim, out_features = feature_dim, device = device)
#         self.lin_cond_add_features = nn.Linear(in_features = cond_dim, out_features = feature_dim, device = device)
#         self.lin_cond_add_node = nn.Linear(in_features = cond_dim, out_features = value_dim, device = device)

#         # New Edges
#         self.mlp_edges = nn.Sequential(
#             nn.Linear(in_features = feature_dim, out_features = node_dim, device = device),
#             nn.ReLU(),
#             nn.Linear(in_features = node_dim, out_features = edge_dim, device = device),
#             nn.ReLU()
#         )

#         # New Nodes
#         self.lin_values1 = nn.Linear(in_features = node_dim, out_features = value_dim, device = device)
#         self.lin_values2 = nn.Linear(in_features = value_dim, out_features = value_dim, device = device)
        
#         self.mlp_weights = nn.Sequential(
#             nn.Linear(in_features = feature_dim, out_features = node_dim, device = device),
#             nn.ReLU(),
#             nn.Linear(in_features = node_dim, out_features = num_heads, device = device),
#         )
#         self.lin_nodes = nn.Sequential(
#             nn.Linear(in_features = value_dim, out_features = node_dim, device = device),
#             nn.ReLU(),
#         )

#     def forward(self, nodes : Tensor, edges : Tensor, cond : Tensor) -> tuple[Tensor, Tensor]:
#         b, n, d = nodes.size()
        
#         cond = cond.unsqueeze(1).unsqueeze(1) # (b, 1, 1, d)

#         # Create Features ----
#         graph_features = torch.cat([edges, nodes.unsqueeze(1).expand(b, n, n, d), nodes.unsqueeze(2).expand(b, n, n, d)], dim = -1) # (b, n, n, de + 2 * dn)

#         # Condition on timestep
#         graph_features = F.relu(self.lin_features(graph_features) + self.lin_cond_add_features(cond)) # (b, n, n, de + 2 * dn)

#         # New Edges ----
#         new_edges = self.mlp_edges(graph_features) # (b, n, n, de)
        
#         # New Nodes ----
#         h = self.num_heads

#         # Project Values
#         values = F.relu(self.lin_values1(nodes) + self.lin_cond_add_node(cond.squeeze(1)))
#         values = self.lin_values2(values).reshape(b, n, h, -1).permute(0, 2, 1, 3) # (b, h, n, dv / h)

#         # Calculate Attention Weights
#         weights = self.mlp_weights(graph_features).permute(0, 3, 1, 2).softmax(dim = -1) # (b, h, n, n)

#         # Aggregate Information Across Nodes
#         weighted_values = (weights @ values).permute(0, 2, 1, 3).flatten(2) # (b, n, dv)

#         # Project Values Back
#         new_nodes = self.lin_nodes(weighted_values) # (b, n, dn)

#         return new_nodes, new_edges

# class NormalizationLayer(nn.Module):
#     def __init__(self, dim: int, device: torch.device):
#         super().__init__()

#         # Hadamard
#         self.scales = nn.Parameter(torch.ones(dim, device=device))
#         # Bias
#         self.shifts = nn.Parameter(torch.zeros(dim, device=device))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Omit batch dimension and feature dimension
#         dims = list(range(x.dim()))[1:-1]

#         # Calculate mean of each feature channel
#         mean = x.mean(dim=dims, keepdim=True)

#         # Calculate standard deviation of each feature channel
#         var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
        
#         # Normalize feature channel
#         epsilon = 1e-5
#         x_norm = (x - mean) / torch.sqrt(var + epsilon)

#         # Learned rescaling and shift with clamping
#         scales = torch.clamp(self.scales, min=0.01, max=100)
#         shifts = torch.clamp(self.shifts, min=-100, max=100)

#         return x_norm * scales + shifts

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
  
# class CosineNoiseScheduler(nn.Module):
#   def __init__(self, max_timestep : int, device : torch.device):
#     super().__init__()
#     self.device = device
#     self.max_timestep = max_timestep
#     self.offset = .008 # Fixed offset to improve noise prediction at early timesteps
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
  
#     self.gumbel_dist = torch.distributions.Gumbel(torch.tensor([0.0]), torch.tensor([1.0]))
#   def forward(self, nodes : Tensor, edges : Tensor, timestep : Tensor):
#     ''' Apply noise to graph '''
#     noisy_nodes = torch.zeros(size = nodes.size(), device = nodes.device)
#     noisy_edges = torch.zeros(size = edges.size(), device = edges.device)
#     # nodes = batch_size x num_nodes x NODE_FEATURE_DIMENSION ; edges = batch_size x num_nodes x num_nodes x EDGE_FEATURE_DIMENSION
#     bernoulli_is_constructible = nodes[:,:,0] # batch_size x num_nodes x 1
#     categorical_primitive_types = nodes[:,:,1:6] # batch_size x num_nodes x 5
#     gaussian_primitive_parameters = nodes[:,:,6:] # batch_size x num_nodes x 14
#     # subnode just means if the constraint applies to the start, center, or end of a primitive
#     categorical_subnode_a_types = edges[:,:,:,0:4] # batch_size x num_nodes x 4
#     categorical_subnode_b_types = edges[:,:,:,4:8] # batch_size x num_nodes x 4
#     categorical_constraint_types = edges[:,:,:,8:] # batch_size x num_nodes x 9
#     # IsConstructible noise
#     noisy_nodes[:,:,0] = self.apply_binary_noise(bernoulli_is_constructible, timestep)
#     # Primitive Types noise
#     noisy_nodes[:,:,1:6] = self.apply_discrete_noise(categorical_primitive_types, timestep) # noised_primitive_types
#     # Primitive parameters noise
#     gaussian_noise = torch.randn_like(gaussian_primitive_parameters) # standard gaussian noise
#     noisy_nodes[:,:,6:] = self.apply_gaussian_noise(gaussian_primitive_parameters, timestep, gaussian_noise)
#     # Subnode A noise
#     noisy_edges[:,:,:,0:4] = self.apply_discrete_noise(categorical_subnode_a_types, timestep) # noised_subnode_a_types
#     # Subnode B noise
#     noisy_edges[:,:,:,4:8] = self.apply_discrete_noise(categorical_subnode_b_types, timestep) # noised_subnode_a_types
#     # Constraint Types noise
#     noisy_edges[:,:,:,8:] = self.apply_discrete_noise(categorical_constraint_types, timestep) # noised_constraint_types
#     return noisy_nodes, noisy_edges, gaussian_noise
  
#   def get_transition_noise(self, parameters : Tensor, timestep : int, gaussian_noise : Tensor = None):
#     if gaussian_noise is None:
#       gaussian_noise = torch.randn_like(parameters) # standard gaussian noise
#     return self.sqrt_precisions[timestep] * parameters + self.sqrt_variances[timestep] * gaussian_noise
  
#   def apply_gaussian_noise(self, parameters : Tensor, timestep : Tensor | int, gaussian_noise : Tensor = None):
#     if gaussian_noise is None:
#       gaussian_noise = torch.randn_like(parameters) # standard gaussian noise
    
#     if type(timestep) is int: timestep = [timestep]
#     # parameters shape is batch_size x num_nodes x num_params
#     # gaussian_noise shape is batch_size x num_nodes x num_params
#     batched_precisions = self.sqrt_cumulative_precisions[timestep,None,None] # (b,1,1) or (1,1,1)
#     batched_variances = self.sqrt_cumulative_variances[timestep,None,None]   # (b,1,1) or (1,1,1)
#     return batched_precisions * parameters + batched_variances * gaussian_noise
  
#   def apply_gaussian_posterior_step(self, curr_params : Tensor, pred_params : Tensor, timestep : int):
#     if timestep > 1:
#       sqrt_prev_cumul_prec = self.sqrt_cumulative_precisions[timestep - 1]
#       var = self.variances[timestep]
#       sqrt_prec = self.sqrt_precisions[timestep]
#       prev_cumul_var = self.cumulative_variances[timestep - 1]
#       cumul_var = self.cumulative_variances[timestep]
      
#       denoised_mean = (sqrt_prev_cumul_prec * var * pred_params + sqrt_prec * prev_cumul_var * curr_params) / cumul_var

#       gaussian_noise = torch.randn_like(curr_params)
#       return denoised_mean + gaussian_noise * self.sqrt_posterior_variances[timestep]
#     else:
#       return pred_params # denoised_mean
    
#   def get_transition_matrix(self, dimension : int, timestep : int | Tensor):
#     if type(timestep) is int: 
#       assert timestep > 0; 
#       assert timestep < self.max_timestep; 
#       timestep = [timestep]
    
#     batched_precisions = self.precisions[timestep,None,None] # (batch_size, 1, 1) or (1, 1, 1)
#     return batched_precisions * torch.eye(dimension, dtype = torch.float32, device = self.device) + (1 - batched_precisions) / dimension # (batch_size, d, d) or (1, d, d)
  
#   def get_cumulative_transition_matrix(self, dimension : int, timestep : int | Tensor):
#     if type(timestep) is int: 
#       assert timestep > 0; 
#       assert timestep < self.max_timestep; 
#       timestep = [timestep]

#     batched_precisions = self.cumulative_precisions[timestep,None,None] # (batch_size, 1, 1) or (1, 1, 1)
#     return batched_precisions * torch.eye(dimension, dtype = torch.float32, device = self.device) + (1 - batched_precisions) / dimension # (batch_size, d, d) or (1, d, d)
  
#   def get_posterior_transition_matrix(self, xt : Tensor, timestep : Tensor | int) -> torch.Tensor:
#     xt_size, xt = self.flatten_middle(xt) # (b, n, d) or (b, n * n, d), for convenience let m = n or n * n
#     d = xt_size[-1]
#     qt = xt @ self.get_transition_matrix(d, timestep).permute(0, 2, 1) # (b, m, d), since xt is onehot we are plucking out rows corresponding to p(x_t = class | x_(t-1))
#     qt_bar = xt @ self.get_cumulative_transition_matrix(d, timestep).permute(0, 2, 1) # (b, m, d), since xt is onehot we are plucking out rows corresponding to p(x_t = class | x_0)
#     q = qt.unsqueeze(2) / qt_bar.unsqueeze(3) # (b, m, d, d), perform an outer product so element at (b, m, i, j) = p(x_t = class | x_(t-1) = j) / p(x_t = class | x_0 = i)
#     q = q * self.get_cumulative_transition_matrix(d, timestep - 1).unsqueeze(1) # (b, m, d, d), broadcast multiply so element at (b, m, i, j) = p(x_t = class | x_(t-1) = j) * p(x_(t-1) = j | x_0 = i) / p(x_t = class | x_0 = i)
#     return q.view(size = xt_size + (d,)) # reshape into (b, n, d, d) or (b, n, n, d, d)
  
#   def apply_discrete_noise(self, x_one_hot : Tensor, timestep : Tensor | int):
#     size, x = self.flatten_middle(x_one_hot)
#     q = self.get_cumulative_transition_matrix(size[-1], timestep) # (b, d, d) or (1, d, d)
#     distribution = x @ q # (b, n, d) or (b, n * n, d)
#     distribution = distribution.view(size) # (b, n, d) or (b, n, n, d)
#     return self.sample_discrete_distribution(distribution).float()
  
#   def apply_multinomial_posterior_step(self, classes_one_hot : Tensor, pred_class_probs : Tensor, timestep : int):
#     # classes_one_hot = (b, n, d) or (b, n, n, d)
#     # pred_class_probs = (b, n, d) or (b, n, n, d)
#     if timestep > 1:
#       q = self.get_posterior_transition_matrix(classes_one_hot, timestep) # (b, n, d, d) or (b, n, n, d, d)
#       pred_class_probs = pred_class_probs.unsqueeze(-2) # (b, n, 1, d) or (b, n, n, 1, d), make probs into row vector
#       posterior_distribution = pred_class_probs @ q # (b, n, 1, d) or (b, n, n, 1, d), batched vector-matrix multiply
#       posterior_distribution = posterior_distribution.squeeze(-2) # (b, n, d) or (b, n, n, d)
#       return self.sample_discrete_distribution(posterior_distribution).float()
#     else:
#       return pred_class_probs
    
#   def apply_binary_noise(self, boolean_flag : Tensor, timestep : int | Tensor):
#     boolean_flag = boolean_flag.unsqueeze(-1)
#     one_hot = torch.cat([1 - boolean_flag, boolean_flag], dim = -1) # (b, n, 2)
#     noised_one_hot = self.apply_discrete_noise(one_hot, timestep) # (b, n, 2)
#     return noised_one_hot[...,1] # (b, n)
  
#   def apply_bernoulli_posterior_step(self, boolean_flag : Tensor, pred_boolean_prob : Tensor, timestep : int):
#     if timestep > 1:
#       boolean_flag = boolean_flag.unsqueeze(-1) # b, n, 1
#       pred_boolean_prob = pred_boolean_prob.unsqueeze(-1) # b, n, 1
#       one_hot_xt = torch.cat([1 - boolean_flag, boolean_flag], dim = -1) # (b, n, 2)
#       probs = torch.cat([1 - pred_boolean_prob, pred_boolean_prob], dim = -1) # (b, n, 2)
#       noised_one_hot = self.apply_multinomial_posterior_step(one_hot_xt, probs, timestep) # (b, n, 2)
#       return noised_one_hot[...,1] # (b, n)
#     else:
#       return pred_boolean_prob
    
#   def sample_discrete_distribution(self, tensor : Tensor):
#     size = tensor.size()
#     num_classes = size[-1]
#     return F.one_hot(tensor.reshape(-1, num_classes).multinomial(1), num_classes).reshape(size)
#     # noise = self.gumbel_dist.sample(tensor.shape).squeeze(-1).to(self.device)
#     # return torch.softmax(tensor.log() + noise, dim = -1)
  
#   def flatten_middle(self, x : Tensor):
#     prev_size = x.size() # shape of x_one_hot is (b, n, d) or (b, n, n, d)
#     return prev_size, x.view(prev_size[0], -1, prev_size[-1]) # (b, n, d) or (b, n * n, d)