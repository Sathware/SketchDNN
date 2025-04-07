import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, device
from torch.utils.checkpoint import checkpoint
from config import NODE_FEATURE_DIMENSION, EDGE_FEATURE_DIMENSION, MAX_NUM_PRIMITIVES
from dataset1 import SketchDataset
from matplotlib import pyplot as plt
from utils import ToNaive

class ConstraintModel(nn.Module):
    def __init__(self, device: device):
        super().__init__()
        self.device = device
        dim = 2 * NODE_FEATURE_DIMENSION + 2
        hidden_dim = 256
        num_tf_layers = 24
        num_checkpoints = 0
        num_heads = 8
        self.max_timestep = 1000
        self.noise_scheduler = CosineNoiseScheduler(self.max_timestep, self.device)
        self.num_checkpoints = num_checkpoints

        # Sinusoidal Positional Encoding
        self.pos_embedder = SinuisodalEncoding(max_length = MAX_NUM_PRIMITIVES, embedding_dimension = hidden_dim // 2, device = device)

        # Input MLP layers
        self.mlp_in_edges = nn.Sequential(
            nn.Linear(in_features = dim, out_features = hidden_dim, device = device),
            nn.SiLU(),
            nn.Linear(in_features = hidden_dim, out_features = hidden_dim, device = device),
            # nn.SiLU(),
        )

        # Transformer Layers
        self.block_layers = nn.ModuleList([
            TransformerLayer(
                dim = hidden_dim,
                num_heads = num_heads,
                device = device
            ) for _ in range(num_tf_layers)
        ])
        
        # Normalization
        self.norm = nn.LayerNorm(normalized_shape = hidden_dim, device = device)
        # Output MLP layers
        self.mlp_out = nn.Sequential(
            nn.Linear(in_features = hidden_dim, out_features = hidden_dim, device = device),
            nn.SiLU(),
            nn.Linear(in_features = hidden_dim, out_features = dim, device = device)
        )

    def forward(self, nodes : Tensor):
        pos = self.pos_embedder.embs
        conds = torch.cat([pos.unsqueeze(0).expand(nodes.size(1), -1, -1), pos.unsqueeze(1).expand(-1, nodes.size(1), -1)], dim = -1) # 2D positional encodings
        edges = torch.cat([nodes.unsqueeze(1).expand(-1, nodes.size(1), -1, -1), nodes.unsqueeze(2).expand(-1, -1, nodes.size(1), -1)], dim = -1)
        edges = self.mlp_in_edges(edges) + conds.unsqueeze(0) # shape: (batch_size, num_nodes, node_hidden_dim)

        checkpoints = self.num_checkpoints
        for layer in self.block_layers:
            edges = checkpoint(layer, edges, use_reentrant = False) if checkpoints > 1 else layer(edges) # shape: (batch_size, num_nodes, node_hidden_dim) ; shape: (batch_size, num_nodes, num_nodes, edge_hidden_dim)
            checkpoints = checkpoints - 1

        edges = self.norm(edges)
        edges = self.mlp_out(edges) # shape: (batch_size, num_nodes, node_dim)

        return edges

class TransformerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, device: device):
        super().__init__()
        self.node_dim = dim

        # Normalization
        self.norm_in = nn.LayerNorm(normalized_shape = dim, device = device)

        # Attention Layer
        self.attention_heads = MultiHeadDotAttention(dim = dim, num_heads = num_heads, device = device)

        # Normalization
        self.norm_attn = nn.LayerNorm(normalized_shape = dim, device = device)

        # Node and edge MLPs
        self.mlp = nn.Sequential(
            nn.Linear(in_features = dim, out_features = dim, device = device),
            nn.SiLU(),
            nn.Linear(in_features = dim, out_features = dim, device = device),
            # nn.SiLU()
        )

    def forward(self, edges : Tensor) -> Tensor:
        # Attention
        edges = self.attention_heads(self.norm_in(edges)) + edges
        # MLP
        edges = self.mlp(self.norm_attn(edges)) + edges

        return edges
    
class MultiHeadDotAttention(nn.Module):
    def __init__(self, dim : int, num_heads : int, device : torch.device):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.lin_qkv = nn.Linear(in_features = dim, out_features = 3 * dim, device = device)
        self.lin_out = nn.Linear(in_features = dim, out_features = dim, device = device)           

    def forward(self, nodes : Tensor) -> Tensor:
        batch_size, num_nodes, _, _ = nodes.size()
        
        queries, keys, values = self.lin_qkv(nodes).chunk(chunks = 3, dim = -1) # .split([self.node_dim, self.node_dim, 2 * self.node_dim], dim = -1)

        queries = queries.reshape(batch_size, num_nodes, self.num_heads, -1).permute(0, 3, 1, 2, 4) # batch_size x num_heads x num_nodes x num_nodes x attn_dim
        keys = keys.reshape(batch_size, num_nodes, self.num_heads, -1).permute(0, 3, 1, 2, 4)       # batch_size x num_heads x num_nodes x num_nodes x attn_dim
        values = values.reshape(batch_size, num_nodes, self.num_heads, -1).permute(0, 2, 1, 2, 4)   # batch_size x num_heads x num_nodes x num_nodes x attn_dim
        # attn_mask = ~torch.eye(queries.size(-2), dtype = torch.bool, device = queries.device)

        weighted_values = F.scaled_dot_product_attention(query = queries, key = keys, value = values).permute(0, 2, 1, 3).flatten(start_dim = 2)

        return self.lin_out(weighted_values)

class SinuisodalEncoding(nn.Module):
  def __init__(self, max_length : int, embedding_dimension : int, device : torch.device):
    super().__init__()
    self.device = device
    self.embed_dim = embedding_dimension
    
    # self.time_embs = nn.Embedding(num_embeddings = max_timestep, embedding_dim = embedding_dimension, device = device)
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
    self.offset = .008 # Fixed offset to improve noise prediction at early timesteps

    t = torch.linspace(0, 1, self.max_timestep + 1, device = device)

    # Cosine Beta Schedule Formula: https://arxiv.org/abs/2102.09672
    # self.a_bar = torch.cos((torch.linspace(0, 1, self.max_timestep + 1).to(self.device) + self.offset) * 0.5 * math.pi / (1 + self.offset)) ** 2
    # self.a_bar = self.a_bar / self.a_bar[0]
    # self.a_bar = self.a_bar.clamp(min = 0.0001, max = 0.9999)
    self.a_bar = torch.cos(t * 0.5 * math.pi) ** 2 # 1 - torch.linspace(0, 1, self.max_timestep + 1, device = device) ** 2
    self.a_bar = self.a_bar.clamp(min = 0.0001, max = 0.9999)
    self.min_k = self.a_bar[9] # Minimum Noise to make onehot vectors into near onehot
  
  def forward(self, nodes : Tensor, timestep : Tensor):
    ''' Apply noise to graph '''
    noisy_nodes = torch.zeros(size = nodes.size(), device = nodes.device)
    
    # IsConstructible noise
    noisy_nodes[...,0:2] = self.apply_discrete_noise(nodes[...,0:2], timestep)
    # Primitive Types noise
    noisy_nodes[...,2:7] = self.apply_discrete_noise(nodes[...,2:7], timestep)
    # Primitive parameters noise
    noisy_nodes[...,7:] = self.apply_continuous_noise(nodes[...,7:], timestep)
    
    return noisy_nodes
  
  def sample_latent(self, batch_size : int) -> Tensor:
    noisy_nodes = torch.zeros(size = (batch_size, MAX_NUM_PRIMITIVES, NODE_FEATURE_DIMENSION + 1), device = self.device)

    # IsConstructible noise
    noisy_nodes[...,0:2] = torch.randn_like(noisy_nodes[...,0:2]).softmax(dim = -1)
    # Primitive Types noise
    noisy_nodes[...,2:7] = torch.randn_like(noisy_nodes[...,2:7]).softmax(dim = -1)
    # Primitive parameters noise
    noisy_nodes[...,7:] = torch.randn_like(noisy_nodes[...,7:])
    
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
    
    da_bar = self.continous_variance_to_discrete_variance(self.a_bar[timestep, None, None], params.size(-1))
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

    curr_a_bar = self.continous_variance_to_discrete_variance(self.a_bar[timestep], pred_params.size(-1))
    prev_a_bar = self.continous_variance_to_discrete_variance(self.a_bar[timestep - 1], pred_params.size(-1))
    curr_a = curr_a_bar / prev_a_bar
    curr_b = 1 - curr_a
    curr_b_bar = 1 - curr_a_bar
    prev_b_bar = 1 - prev_a_bar

    if timestep > 1:
      log_pred = pred_params.log() # * 1.5
      log_curr = curr_params.log()
      mean = (prev_a_bar.sqrt() * curr_b * log_pred + curr_a.sqrt() * prev_b_bar * log_curr) / curr_b_bar
      noise = torch.randn_like(pred_params)
      return torch.softmax(mean + torch.sqrt(prev_b_bar / curr_b_bar * curr_b) * noise, dim = -1) #, noise
    else:
      return pred_params
    
  def continous_variance_to_discrete_variance(self, a : Tensor, D : int):
    n = torch.log((1 - a) / ((D - 1) * a + 1)) ** 2
    m = torch.log((1 - self.min_k) / ((D - 1) * self.min_k + 1)) ** 2

    return n / (n + m)