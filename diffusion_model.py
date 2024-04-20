# %%
import torch
import math
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from config import NODE_FEATURE_DIMENSION, EDGE_FEATURE_DIMENSION, MAX_NUM_PRIMITIVES

# %%
class GD3PM(nn.Module):
  def __init__(self, device : torch.device):
    super().__init__()
    self.device = device
    self.node_dim = NODE_FEATURE_DIMENSION
    self.edge_dim = EDGE_FEATURE_DIMENSION
    self.hidden_dim = 256
    self.num_tf_layers = 32
    self.num_heads = 12
    self.max_timestep = 500
    self.max_steps = self.max_timestep + 1
    self.noise_scheduler = CosineNoiseScheduler(self.max_timestep, self.device)
    self.time_embedder = TimeEmbedder(self.max_timestep, self.hidden_dim, self.device)
    self.mlp_in_nodes = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.hidden_dim, device = device),
                                            nn.LeakyReLU(.1),
                                            # nn.Dropout(p = 0.1),
                                            nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
                                            nn.LeakyReLU(.1),
                                            # nn.Dropout(p = 0.1)
                                           )
    self.mlp_in_edges = nn.Sequential(nn.Linear(in_features = self.edge_dim, out_features = self.hidden_dim, device = device),
                                            nn.LeakyReLU(.1),
                                            # nn.Dropout(p = 0.1),
                                            nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
                                            nn.LeakyReLU(.1),
                                            # nn.Dropout(p = 0.1)
                                           )
    
    self.time_emb_lins = nn.ModuleList([nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = self.device)
                                        for _ in range(self.num_tf_layers)])
    
    # self.tf_layers = nn.ModuleList([TransformerLayer(num_heads = self.num_heads,
    #                                                  node_dim = self.hidden_dim,
    #                                                  edge_dim = self.hidden_dim,
    #                                                  device = device
    #                                                 )
    #                                 for _ in range(self.num_tf_layers)])
    self.soft_attn_layers = nn.ModuleList([SoftAttentionLayer(self.hidden_dim, self.num_heads, device = self.device)
                                           for _ in range(self.num_tf_layers)])
    
    self.mlp_out_nodes = nn.Sequential(nn.Linear(in_features = self.hidden_dim, out_features = self.node_dim, device = device),
                                       nn.LeakyReLU(.1),
                                    #    nn.Dropout(p = 0.1),
                                       nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device)
                                      )
    self.mlp_out_edges = nn.Sequential(nn.Linear(in_features = self.hidden_dim, out_features = self.edge_dim, device = device),
                                       nn.LeakyReLU(.1),
                                    #    nn.Dropout(p = 0.1),
                                       nn.Linear(in_features = self.edge_dim, out_features = self.edge_dim, device = device)
                                      )

  def forward(self, nodes : Tensor, edges : Tensor, timestep : Tensor):
    # embed timestep
    time_embs = self.time_embedder(timestep) # batch_size x hidden_dim
    nodes = self.mlp_in_nodes(nodes) # batch_size x num_nodes x hidden_dim
    edges = self.mlp_in_edges(edges) # batch_size x num_nodes x num_nodes x hidden_dim

    for idx in range(self.num_tf_layers):
      #nodes, edges, time_embs = layer(nodes, edges, time_embs) if idx % 8 == 0 else checkpoint(layer, nodes, edges, time_embs, use_reentrant = False)
      time_embs = self.time_emb_lins[idx](time_embs)
      nodes = nodes + time_embs.unsqueeze(1)
      edges = edges + time_embs.unsqueeze(1).unsqueeze(1)

      nodes, edges = self.soft_attn_layers(nodes, edges)

      time_embs = F.leaky_relu(time_embs, .1)
    
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
    for t in reversed(range(1, self.max_steps)):
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
  def reverse_step(self, nodes, edges, pred_node_noise, pred_edge_noise, timestep):
    # IsConstructible denoising
    nodes[:,:,0] = self.noise_scheduler.apply_bernoulli_posterior_step(nodes[:,:,0], pred_node_noise[:,:,0], timestep)
    # Primitive Types denoising
    nodes[:,:,1:6] = self.noise_scheduler.apply_multinomial_posterior_step(nodes[:,:,1:6], pred_node_noise[:,:,1:6], timestep)
    # Primitive parameters denoising
    nodes[:,:,6:] = self.noise_scheduler.apply_gaussian_posterior_step(nodes[:,:,6:], pred_node_noise[:,:,6:], timestep)
    # Subnode A denoising
    edges[:,:,:,0:4] = self.noise_scheduler.apply_multinomial_posterior_step(edges[:,:,:,0:4], pred_edge_noise[:,:,:,0:4], timestep)
    # Subnode B denoising
    edges[:,:,:,4:8] = self.noise_scheduler.apply_multinomial_posterior_step(edges[:,:,:,4:8], pred_edge_noise[:,:,:,4:8], timestep)
    # Constraint Types denoising
    edges[:,:,:,8:] = self.noise_scheduler.apply_multinomial_posterior_step(edges[:,:,:,8:], pred_edge_noise[:,:,:,8:], timestep)
    return nodes, edges
  
  # @torch.no_grad()
  # def forward_step(self, nodes, edges, timestep):
  #   bool_class_probs = torch.cat(((1 - nodes[:,:,0]).unsqueeze(-1), nodes[:,:,0].unsqueeze(-1)), dim = -1) # batch_size x num_nodes x 2 i.e. [p(fail), p(success)]
  #   nodes[:,:,0] = (bool_class_probs @ self.noise_scheduler.get_transition_matrix(2, timestep)).reshape(-1, 2).multinomial(1).reshape(nodes[:,:,0].size()).float()
  #   # Primitive Types noising
  #   nodes[:,:,1:6] = self.noise_scheduler.sample_discrete(nodes[:,:,1:6] @ self.noise_scheduler.get_transition_matrix(5, timestep)).float()
  #   # Primitive parameters noising
  #   nodes[:,:,6:] = self.noise_scheduler.get_transition_noise(nodes[:,:,6:], timestep)
  #   # Subnode A noising
  #   edges[:,:,:,0:4] = self.noise_scheduler.sample_discrete(edges[:,:,:,0:4] @ self.noise_scheduler.get_transition_matrix(4, timestep)).float()
  #   # Subnode B noising
  #   edges[:,:,:,4:8] = self.noise_scheduler.sample_discrete(edges[:,:,:,4:8] @ self.noise_scheduler.get_transition_matrix(4, timestep)).float()
  #   # Constraint Types noising
  #   edges[:,:,:,8:] = self.noise_scheduler.sample_discrete(edges[:,:,:,8:] @ self.noise_scheduler.get_transition_matrix(9, timestep)).float()
  #   return nodes, edges

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
        self.max_steps = max_timestep + 1
        self.max_timestep = max_timestep
        self.offset = .008 # Fixed offset to improve noise prediction at early timesteps
        # Cosine Beta Schedule Formula: https://arxiv.org/abs/2102.09672     1.00015543316 is 1/a(0), for offset = .008
        self.cumulative_precisions = torch.cos((torch.linspace(0, 1, self.max_steps).to(self.device) + self.offset) * 0.5 * math.pi / (1 + self.offset)) ** 2 * 1.00015543316
        self.cumulative_variances = 1 - self.cumulative_precisions
        self.variances = torch.cat([torch.Tensor([0]).to(self.device), 1 - (self.cumulative_precisions[1:] / self.cumulative_precisions[:-1])]).clamp(.0001, .9999)
        self.precisions = 1 - self.variances
        self.sqrt_cumulative_precisions = torch.sqrt(self.cumulative_precisions)
        self.sqrt_cumulative_variances = torch.sqrt(self.cumulative_variances)
        self.sqrt_precisions = torch.sqrt(self.precisions)
        self.sqrt_variances = torch.sqrt(self.variances)
        self.sqrt_posterior_variances = torch.cat([torch.Tensor([0]).to(self.device), torch.sqrt(self.variances[1:] * self.cumulative_variances[:-1] / self.cumulative_variances[1:])])

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
      if gaussian_noise is None:
        gaussian_noise = torch.randn_like(parameters) # standard gaussian noise
      return self.sqrt_precisions[timestep] * parameters + self.sqrt_variances[timestep] * gaussian_noise

    def apply_gaussian_noise(self, parameters : Tensor, timestep : Tensor | int, gaussian_noise : Tensor = None):
      if gaussian_noise is None:
        gaussian_noise = torch.randn_like(parameters) # standard gaussian noise
      
      if type(timestep) is int: timestep = [timestep]
      # parameters shape is batch_size x num_nodes x num_params
      # gaussian_noise shape is batch_size x num_nodes x num_params
      batched_precisions = self.sqrt_cumulative_precisions[timestep,None,None] # (b,1,1) or (1,1,1)
      batched_variances = self.sqrt_cumulative_variances[timestep,None,None]   # (b,1,1) or (1,1,1)
      return batched_precisions * parameters + batched_variances * gaussian_noise

    def apply_gaussian_posterior_step(self, parameters : Tensor, pred_noise : Tensor, timestep : int):
      var = self.variances[timestep]
      sqrt_cumulative_var = self.sqrt_cumulative_variances[timestep]
      sqrt_precision = self.sqrt_precisions[timestep]
      
      denoised_mean = (parameters - pred_noise * var / sqrt_cumulative_var) / sqrt_precision
      if timestep > 1:
        gaussian_noise = torch.randn_like(parameters)
        return denoised_mean + gaussian_noise * self.sqrt_posterior_variances[timestep]
      else:
        return denoised_mean

    def get_transition_matrix(self, dimension : int, timestep : int | Tensor):
      if type(timestep) is int: assert timestep > 0; timestep = [timestep]
      batched_precisions = self.sqrt_precisions[timestep,None,None] # (batch_size, 1, 1) or (1, 1, 1)
      return batched_precisions * torch.eye(dimension, dtype = torch.float32, device = self.device) + (1 - batched_precisions) / dimension # (batch_size, d, d) or (1, d, d)

    def get_cumulative_transition_matrix(self, dimension : int, timestep : int | Tensor):
      if type(timestep) is int: assert timestep > 0; timestep = [timestep]
      batched_precisions = self.sqrt_cumulative_precisions[timestep,None,None] # (batch_size, 1, 1) or (1, 1, 1)
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
      size, x = self.flatten_middle(x_one_hot)
      q = self.get_cumulative_transition_matrix(size[-1], timestep) # (b, d, d) or (1, d, d)
      distribution = x @ q # (b, n, d) or (b, n * n, d)
      distribution = distribution.view(size) # (b, n, d) or (b, n, n, d)
      return self.sample_discrete_distribution(distribution).float()
    
    def apply_multinomial_posterior_step(self, classes_one_hot : Tensor, pred_class_probs : Tensor, timestep : int):
      # classes_one_hot = (b, n, d) or (b, n, n, d)
      # pred_class_probs = (b, n, d) or (b, n, n, d)
      if timestep > 1:
        q = self.get_posterior_transition_matrix(classes_one_hot, timestep) # (b, n, d, d) or (b, n, n, d, d)
        pred_class_probs = pred_class_probs.unsqueeze(-2) # (b, n, 1, d) or (b, n, n, 1, d), make probs into row vector
        posterior_distribution = pred_class_probs @ q # (b, n, 1, d) or (b, n, n, 1, d), batched vector-matrix multiply
        posterior_distribution = posterior_distribution.squeeze(-2) # (b, n, d) or (b, n, n, d)
        return self.sample_discrete_distribution(posterior_distribution).float()
      else:
        return pred_class_probs
      
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
       size = tensor.size()
       num_classes = size[-1]
       return F.one_hot(tensor.reshape(-1, num_classes).multinomial(1), num_classes).reshape(size)
    
    def flatten_middle(self, x : Tensor):
      prev_size = x.size() # shape of x_one_hot is (b, n, d) or (b, n, n, d)
      return prev_size, x.view(prev_size[0], -1, prev_size[-1]) # (b, n, d) or (b, n * n, d)

class SoftAttentionLayer(nn.Module):
  def __init__(self, dim : int, num_heads : int, device : torch.device):
    super().__init__()
    self.device = device
    self.node_dim = dim
    self.edge_dim = dim
    self.num_heads = num_heads
    self.attn_dim = dim // num_heads
    self.num_nodes = MAX_NUM_PRIMITIVES

    concat_dim = 2 * self.node_dim + self.edge_dim
    self.mlp_haggr_weights = nn.Sequential(nn.Linear(in_features = concat_dim, out_features = concat_dim, device = self.device),
                                           nn.LeakyReLU(.1),
                                           nn.Linear(in_features = concat_dim, out_features = 1, device = self.device),
                                           nn.Softmax(dim = 2)
                                          )
    
    self.mlp_haggr_values = nn.Sequential(nn.Linear(in_features = concat_dim, out_features = concat_dim, device = self.device),
                                           nn.LeakyReLU(.1),
                                           nn.Linear(in_features = concat_dim, out_features = self.node_dim, device = self.device),
                                         )

    self.query_key_value_mlp = nn.Linear(in_features = self.node_dim, out_features = 3 * self.node_dim, device = self.device)

    self.layer_norm_embs = nn.Sequential(nn.LayerNorm(normalized_shape = self.node_dim, device = self.device),
                                         nn.LeakyReLU(.1),
                                        )

    self.node_mlp = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = self.device),
                                  nn.LeakyReLU(.1),
                                  nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = self.device),
                                 )
    self.edge_mlp = nn.Sequential(nn.Linear(in_features = 2 * self.node_dim, out_features = 2 * self.node_dim, device = self.device),
                                  nn.LeakyReLU(.1),
                                  nn.Linear(in_features = 2 * self.node_dim, out_features = self.node_dim, device = self.device),
                                 )
    
    self.layer_norm_out_nodes = nn.Sequential(nn.LayerNorm(normalized_shape = self.node_dim, device = self.device),
                                              nn.LeakyReLU(.1),
                                             )
    
    self.layer_norm_out_edges = nn.Sequential(nn.LayerNorm(normalized_shape = self.node_dim, device = self.device),
                                              nn.LeakyReLU(.1),
                                             )
    
  def forward(self, nodes : Tensor, edges : Tensor) -> Tuple[Tensor, Tensor]:
    # Outer Product Concatenation
    hstack = nodes.unsqueeze(2).expand(-1, -1, self.num_nodes, -1) # (b, n, n, d)
    vstack = hstack.permute(0, 2, 1, 3) # (b, n, n, d)
    graph_features = torch.cat(tensors = (hstack, vstack, edges), dim = 3) # (b, n, n, 3 * d)

    # Soft Attentional Encoder
    haggr_weights = self.mlp_haggr_weights(graph_features).permute(0, 1, 3, 2) # (b, n, 1, n)
    haggr_values = self.mlp_haggr_values(graph_features) # (b, n, n, d)
    graph_embs = (haggr_weights @ haggr_values).squeeze(2) # (b, n, d)

    # Low Dimensional Attention
    b, n, d = graph_embs.size()
    query_key_value = self.query_key_value_mlp(graph_embs).view(b, n, self.num_heads, 3 * self.attn_dim).permute(0, 2, 1, 3) # (b, h, n, 3 * attn_dim)
    queries, keys, values = query_key_value.reshape(b * self.num_heads, n, 3 * self.attn_dim).chunk(3, dim = 2) # (b * h, n, attn_dim) is shape for the three tensors
    attn_embs = F.scaled_dot_product_attention(queries, keys, values).view(b, self.num_heads, n, self.attn_dim) # (b * h, n, attn_dim)
    attn_embs = attn_embs.permute(0, 2, 1, 3).reshape(b, n, self.node_dim) # (b, n, d)

    # Residual Connection and LayerNorm
    self.layer_norm_embs(attn_embs + graph_embs)

    # Outer Product Decoder
    emb_hstack = attn_embs.unsqueeze(2).expand(-1, -1, self.num_nodes, -1) # (b, n, n, d)
    emb_vstack = emb_hstack.permute(0, 2, 1, 3) # (b, n, n, d)
    emb_edges = torch.cat(tensors = (emb_hstack, emb_vstack), dim = 3) # (b, n, n, 2 * d)

    new_edges = self.edge_mlp(emb_edges) # (b, n, n, d)
    new_nodes = self.node_mlp(attn_embs) # (b, n, d)

    # Residual Connection and LayerNorm
    new_edges = self.layer_norm_out_nodes(new_edges + new_edges)
    new_nodes = self.layer_norm_out_nodes(new_nodes + nodes)

    return new_nodes, new_edges