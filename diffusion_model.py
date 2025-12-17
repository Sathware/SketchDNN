import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, device
import config
from dataset import SketchDataset

class GD3PM(nn.Module):
  def __init__(self, device : torch.device):
    super().__init__()
    self.device = device
    self.node_dim = config.NODE_FEATURE_DIMENSION
    self.node_hidden_dim = config.NODE_HIDDEN_DIM
    self.cond_hidden_dim = config.COND_HIDDEN_DIM
    self.num_tf_layers = config.NUM_TRANSFORMER_LAYERS
    self.num_heads = config.NUM_ATTENTION_HEADS
    self.max_timestep = config.MAX_DIFFUSION_TIMESTEP
    self.noise_scheduler = CosineNoiseScheduler(self.max_timestep, self.device)
    self.architecture = DiffusionModel(node_dim = self.node_dim,
                                  node_hidden_dim = self.node_hidden_dim,
                                  cond_hidden_dim = self.cond_hidden_dim,
                                  num_heads = self.num_heads,
                                  num_tf_layers = self.num_tf_layers,
                                  max_timestep = self.max_timestep,
                                  device = self.device)
    
  def forward(self, nodes : Tensor, timestep : Tensor):
    nodes = self.architecture(nodes, timestep)
    nodes[...,config.NODE_BOOL_SLICE] = nodes[...,config.NODE_BOOL_SLICE].softmax(dim = -1)
    nodes[...,config.NODE_TYPE_SLICE] = nodes[...,config.NODE_TYPE_SLICE].softmax(dim = -1)
    return nodes

  @torch.no_grad()
  def sample(self, batch_size : int, axes = None):
    # Sample Noise
    nodes = self.noise_scheduler.sample_latent(batch_size)
    nodes = nodes.to(self.device)
    return self.denoise(nodes, axes)

  @torch.no_grad()
  def denoise(self, nodes, axes = None):
    num_images = config.NUM_TRAJECTORY_VISUALIZATION_IMAGES
    j = 0
    stepsize = int(self.max_timestep/num_images)
    
    for t in reversed(range(1, self.max_timestep)):
      # model expects a timestep for each batch
      batch_size = nodes.size(0)
      time = torch.ones(size = (batch_size,), dtype = torch.int32, device = self.device) * t
      denoised_nodes = self.forward(nodes, time)
      nodes = self.reverse_step(denoised_nodes, nodes, t)

      if (axes is not None) and (t % stepsize == 0):
        SketchDataset.render_graph(nodes = nodes[0], ax = axes[0, j])
        SketchDataset.render_graph(nodes = denoised_nodes[0], ax = axes[1, j])
        j = j + 1

    if (axes is not None):
        SketchDataset.render_graph(nodes = nodes[0], ax = axes[0, j])
        SketchDataset.render_graph(nodes = denoised_nodes[0], ax = axes[1, j])

    return nodes
  
  @torch.no_grad()
  def reverse_step(self, pred_nodes, curr_nodes, timestep):
    denoised_nodes = torch.zeros_like(pred_nodes)
    denoised_nodes[...,config.NODE_BOOL_SLICE] = self.noise_scheduler.discrete_posterior_step(pred_nodes[...,config.NODE_BOOL_SLICE], curr_nodes[...,config.NODE_BOOL_SLICE], timestep)
    denoised_nodes[...,config.NODE_TYPE_SLICE] = self.noise_scheduler.discrete_posterior_step(pred_nodes[...,config.NODE_TYPE_SLICE], curr_nodes[...,config.NODE_TYPE_SLICE], timestep)
    
    weights = torch.cat(
      [pred_nodes[...,2,None].expand(-1, -1, 4), # Line
       pred_nodes[...,3,None].expand(-1, -1, 3), # Circle
       pred_nodes[...,4,None].expand(-1, -1, 5), # Arc
       pred_nodes[...,5,None].expand(-1, -1, 2)  # Point
      ], dim = -1)
    
    # Weight parameters by corresponding primitive types
    vals, _ = torch.max(pred_nodes[...,config.NODE_TYPE_SLICE], dim = -1, keepdim = True)
    denoised_nodes[...,config.NODE_PARM_SLICE] = self.noise_scheduler.continuous_posterior_step(weights / vals * pred_nodes[...,config.NODE_PARM_SLICE], curr_nodes[...,config.NODE_PARM_SLICE], timestep)
    return denoised_nodes

class DiffusionModel(nn.Module):
  def __init__(self, node_dim, node_hidden_dim, cond_hidden_dim, num_heads, num_tf_layers, max_timestep, device: device):
    super().__init__()

    # self.pos_embedder = SinuisodalEncoding(max_length = 24, embedding_dimension = cond_hidden_dim // 4, device = device)
    self.time_embedder = nn.Embedding(num_embeddings = max_timestep, embedding_dim = cond_hidden_dim, device = device)

    # Input MLP layers
    self.mlp_in_nodes = nn.Sequential(
      nn.Linear(in_features = node_dim, out_features = 4 * node_hidden_dim, device = device),
      nn.SiLU(),
      nn.Linear(in_features = 4 * node_hidden_dim, out_features = node_hidden_dim, device = device),
    )

    # Transformer Layers with Graph Attention Network
    self.block_layers = nn.ModuleList([
      TransformerLayer(
        node_dim = node_hidden_dim,
        cond_dim = cond_hidden_dim,
        num_heads = num_heads,
        device = device
      ) for _ in range(num_tf_layers)
      ])
        
    # Normalization
    self.norm = Normalization(node_dim = node_hidden_dim, device = device)
    self.lin_cond = nn.Linear(in_features = cond_hidden_dim, out_features = 2 * node_hidden_dim, device = device)
    nn.init.zeros_(self.lin_cond.weight)
    nn.init.zeros_(self.lin_cond.bias)
        
    self.mlp_out = nn.Sequential(
      nn.Linear(in_features = node_hidden_dim, out_features = 4 * node_hidden_dim, device = device),
      nn.SiLU(),
      nn.Linear(in_features = 4 * node_hidden_dim, out_features = node_dim, device = device)
    )

  def forward(self, nodes : Tensor, timestep : Tensor):
    nodes = self.mlp_in_nodes(nodes)     # shape: (batch_size, num_nodes, node_hidden_dim)
    conds = F.silu(self.time_embedder(timestep)).unsqueeze(1) # shape: (batch_size, num_nodes, cond_hidden_dim)

    for layer in self.block_layers:
      nodes = layer(nodes, conds) # shape: (batch_size, num_nodes, node_hidden_dim)

    shift, scale = self.lin_cond(conds).chunk(chunks = 2, dim = -1)
    nodes = self.norm(nodes) * (1 + scale) + shift
      
    nodes = self.mlp_out(nodes) # shape: (batch_size, num_nodes, node_dim)

    return nodes

class TransformerLayer(nn.Module):
  def __init__(self, node_dim: int, cond_dim: int, num_heads: int, device: device):
    super().__init__()
    self.node_dim = node_dim

    # Normalization
    self.norm_in = Normalization(node_dim = node_dim, device = device)
    # Attention Layer
    self.attention_heads = MultiHeadDotAttention(node_dim = node_dim, num_heads = num_heads, device = device)
    # Normalization
    self.norm_attn = Normalization(node_dim = node_dim, device = device)
    # Node and edge MLPs
    self.mlp_nodes = nn.Sequential(
      nn.Linear(in_features = node_dim, out_features = 4 * node_dim, device = device),
      nn.SiLU(),
      nn.Linear(in_features = 4 * node_dim, out_features = node_dim, device = device)
    )
    # Conditioning
    self.lin_cond = nn.Linear(in_features = cond_dim, out_features = 4 * node_dim, device = device)
    nn.init.zeros_(self.lin_cond.weight)
    nn.init.zeros_(self.lin_cond.bias)

  def forward(self, nodes : Tensor, conds : Tensor) -> Tensor:
    mul_in, add_in, mul_attn, add_attn = self.lin_cond(conds).chunk(chunks = 4, dim = -1)
    # Attention
    nodes = self.attention_heads(self.norm_in(nodes) * (1 + mul_in) + add_in) + nodes
    # MLP
    nodes = self.mlp_nodes(self.norm_attn(nodes) * (1 + mul_attn) + add_attn) + nodes
    return nodes
    
class MultiHeadDotAttention(nn.Module):
  def __init__(self, node_dim : int, num_heads : int, device : torch.device):
    super().__init__()
    self.node_dim = node_dim
    self.num_heads = num_heads

    self.lin_qkv = nn.Linear(in_features = node_dim, out_features = 3 * node_dim, device = device)
    self.lin_nodes_out = nn.Linear(in_features = node_dim, out_features = node_dim, device = device)

  def forward(self, nodes : Tensor) -> Tensor:
    batch_size, num_nodes, _ = nodes.size()
        
    queries, keys, values = self.lin_qkv(nodes).chunk(chunks = 3, dim = -1)

    queries = queries.reshape(batch_size, num_nodes, self.num_heads, -1).permute(0, 2, 1, 3) # batch_size x num_heads x num_nodes x attn_dim
    keys = keys.reshape(batch_size, num_nodes, self.num_heads, -1).permute(0, 2, 1, 3)       # batch_size x num_heads x num_nodes x attn_dim
    values = values.reshape(batch_size, num_nodes, self.num_heads, -1).permute(0, 2, 1, 3)   # batch_size x num_heads x num_nodes x attn_dim

    weighted_values = F.scaled_dot_product_attention(query = queries, key = keys, value = values).permute(0, 2, 1, 3).flatten(start_dim = 2)

    return self.lin_nodes_out(weighted_values)

class Normalization(nn.Module):
  def __init__(self, node_dim: int, device: device):
    super().__init__()
    self.norm_nodes = nn.LayerNorm(normalized_shape = node_dim, elementwise_affine = False, device = device)

  def forward(self, nodes : Tensor) -> Tensor:
    return self.norm_nodes(nodes)

class SinuisodalEncoding(nn.Module):
  def __init__(self, max_length : int, embedding_dimension : int, device : torch.device):
    super().__init__()
    self.device = device
    self.embed_dim = embedding_dimension
    
    steps = torch.arange(max_length, device = self.device).unsqueeze(1) # num_timesteps x 1
    scales = torch.exp(torch.arange(0, self.embed_dim, 2, device = self.device) * (-math.log(10000.0) / self.embed_dim)).unsqueeze(0) # 1 x (embedding_dimension // 2)
    self.embs = torch.zeros(max_length, self.embed_dim, device = self.device) # num_timesteps x embedding_dimension
    self.embs[:, 0::2] = torch.sin(steps * scales) # fill even columns with sin(timestep * 1000^-(2*i/embedding_dimension))
    self.embs[:, 1::2] = torch.cos(steps * scales) # fill odd columns with cos(timestep * 1000^-(2*i/embedding_dimension))
      
  def forward(self, step : Tensor):
    return self.embs[step] # batch_size x embedding_dimension

'''----- Soft Gauss -----'''
class CosineNoiseScheduler(nn.Module):
  def __init__(self, max_timestep : int, device : torch.device):
    super().__init__()
    self.device = device
    self.max_timestep = max_timestep

    t = torch.linspace(0, 1, self.max_timestep + 1, device = device)

    # Cosine Beta Schedule Formula: https://arxiv.org/abs/2102.09672
    self.a_bar = torch.cos(t * 0.5 * math.pi) ** 2
    self.a_bar = self.a_bar.clamp(min = 0.0001, max = 0.9999)
    self.min_k = self.a_bar[1] # Minimum Noise to make onehot vectors into near onehot
  
  def forward(self, nodes : Tensor, timestep : Tensor):
    ''' Apply noise to graph '''
    noisy_nodes = torch.zeros(size = nodes.size(), device = nodes.device)
    
    # IsConstructible noise
    noisy_nodes[...,config.NODE_BOOL_SLICE] = self.apply_discrete_noise(nodes[...,config.NODE_BOOL_SLICE], timestep)
    # Primitive Types noise
    noisy_nodes[...,config.NODE_TYPE_SLICE] = self.apply_discrete_noise(nodes[...,config.NODE_TYPE_SLICE], timestep)
    # Primitive parameters noise
    noisy_nodes[...,config.NODE_PARM_SLICE] = self.apply_continuous_noise(nodes[...,config.NODE_PARM_SLICE], timestep)
    
    return noisy_nodes
  
  def sample_latent(self, batch_size : int) -> Tensor:
    noisy_nodes = torch.zeros(size = (batch_size, config.MAX_NUM_PRIMITIVES, config.NODE_FEATURE_DIMENSION), device = self.device)

    # IsConstructible noise
    noisy_nodes[...,config.NODE_BOOL_SLICE] = torch.randn_like(noisy_nodes[...,config.NODE_BOOL_SLICE]).softmax(dim = -1)
    # Primitive Types noise
    noisy_nodes[...,config.NODE_TYPE_SLICE] = torch.randn_like(noisy_nodes[...,config.NODE_TYPE_SLICE]).softmax(dim = -1)
    # Primitive parameters noise
    noisy_nodes[...,config.NODE_PARM_SLICE] = torch.randn_like(noisy_nodes[...,config.NODE_PARM_SLICE])
    
    return noisy_nodes
  
  def apply_continuous_noise(self, params : Tensor, timestep : Tensor | int) -> Tensor:
    if type(timestep) is int:
      if timestep == 0: 
        return params, 0 
      assert timestep > 0 
      assert timestep < self.max_timestep 
      timestep = [timestep]

    a = torch.sqrt(self.a_bar[timestep, None, None])
    b = torch.sqrt(1 - self.a_bar[timestep, None, None])

    noise = torch.randn_like(params)
    return a * params + b * noise #, noise
  
  def continuous_posterior_step(self, pred_params : Tensor, curr_params : Tensor, timestep : Tensor | int) -> Tensor:
    if type(timestep) is int:
      if timestep == 0: 
        return pred_params
      assert timestep > 0 
      assert timestep < self.max_timestep 
      timestep = torch.tensor(data = [timestep], device = pred_params.device)

    assert timestep > 0, "Timestep is 0 for continuous posterior step!"

    curr_a = self.a_bar[timestep] / self.a_bar[timestep - 1]
    curr_b = 1 - curr_a
    curr_a_bar = self.a_bar[timestep]
    curr_b_bar = 1 - curr_a_bar
    prev_a_bar = self.a_bar[timestep - 1]
    prev_b_bar = 1 - prev_a_bar

    if timestep > 1:
      mean = (prev_a_bar.sqrt() * curr_b * pred_params + curr_a.sqrt() * prev_b_bar * curr_params) / curr_b_bar
      noise = torch.randn_like(pred_params)
      return mean + torch.sqrt(prev_b_bar / curr_b_bar * curr_b) * noise #, noise
    else:
      return pred_params
  
  def apply_discrete_noise(self, params : Tensor, timestep : Tensor | int) -> Tensor:
    if type(timestep) is int:
      if timestep == 0: 
        return params, 0 
      assert timestep > 0
      assert timestep < self.max_timestep
      timestep = [timestep]
    
    da_bar = self.continous_variance_to_discrete_variance(self.a_bar[timestep, None, None].sqrt(), params.size(-1))
    a = torch.sqrt(da_bar)
    b = torch.sqrt(1 - da_bar)

    params = self.min_k * params + (1 - self.min_k) / params.size(-1)

    noise = torch.randn_like(params)
    return torch.softmax(a * params.log() + b * noise, dim = -1) #, noise
  
  def discrete_posterior_step(self, pred_params : Tensor, curr_params : Tensor, timestep : Tensor | int) -> Tensor:
    if type(timestep) is int:
      if timestep == 0: 
        return pred_params
      assert timestep > 0
      assert timestep < self.max_timestep
      timestep = torch.tensor(data = [timestep], device = pred_params.device)

    curr_a_bar = self.continous_variance_to_discrete_variance(self.a_bar[timestep].sqrt(), pred_params.size(-1))
    prev_a_bar = self.continous_variance_to_discrete_variance(self.a_bar[timestep - 1].sqrt(), pred_params.size(-1))
    curr_a = curr_a_bar / prev_a_bar
    curr_b = 1 - curr_a
    curr_b_bar = 1 - curr_a_bar
    prev_b_bar = 1 - prev_a_bar

    if timestep > 1:
      log_pred = pred_params.log() * 2
      log_curr = curr_params.log()
      mean = (prev_a_bar.sqrt() * curr_b * log_pred + curr_a.sqrt() * prev_b_bar * log_curr) / curr_b_bar
      noise = torch.randn_like(pred_params)
      return torch.softmax(mean + torch.sqrt(prev_b_bar / curr_b_bar * curr_b) * noise, dim = -1) #, noise
    else:
      return pred_params
    
  # Variance Schedule Augmentation
  def continous_variance_to_discrete_variance(self, a : Tensor, D : int):
    n = torch.log((1 - a) / ((D - 1) * a + 1)) ** 2
    m = torch.log((1 - self.min_k) / ((D - 1) * self.min_k + 1)) ** 2

    return n / (n + m)