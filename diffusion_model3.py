import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, device
from torch.utils.checkpoint import checkpoint
from config import NODE_FEATURE_DIMENSION, EDGE_FEATURE_DIMENSION, MAX_NUM_PRIMITIVES
from dataset1 import SketchDataset
from matplotlib import pyplot as plt

BINARY_CONSTRUCT_ISOCELES_NODE_DIM = NODE_FEATURE_DIMENSION + 1
class GD3PM(nn.Module):
  def __init__(self, device : torch.device):
    super().__init__()
    self.device = device
    self.node_dim = BINARY_CONSTRUCT_ISOCELES_NODE_DIM
    self.edge_dim = EDGE_FEATURE_DIMENSION
    self.node_hidden_dim = 512
    self.edge_hidden_dim = 256
    self.cond_hidden_dim = 256
    self.num_tf_layers = 24
    self.num_checkpoints = 8
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
    nodes, edges = self.architecture(nodes, edges, timestep)
    nodes[...,0:2] = nodes[...,0:2].softmax(dim = -1)
    nodes[...,2:7] = nodes[...,2:7].softmax(dim = -1)
    edges[...,0:4] = edges[...,0:4].softmax(dim = -1)
    edges[...,4:8] = edges[...,4:8].softmax(dim = -1)
    edges[...,8:] = edges[...,8:].softmax(dim = -1)

    return nodes, edges

  @torch.no_grad()
  def sample(self, batch_size : int):
    # Sample Noise
    nodes, edges = self.noise_scheduler.sample_latent(batch_size)
    nodes = nodes.to(self.device)
    edges = edges.to(self.device)
    return self.denoise(nodes, edges)

  @torch.no_grad()
  def denoise(self, nodes, edges, axes = None):
    num_images = 10
    j = num_images - 1
    if axes is None:
      fig, axes = plt.subplots(nrows = 2, ncols = num_images, figsize=(40, 8))
    stepsize = int(self.max_timestep/num_images)
    
    for t in reversed(range(1, self.max_timestep)):
      # model expects a timestep for each batch
      batch_size = nodes.size(0)
      time = torch.ones(size = (batch_size,), dtype = torch.int32, device = self.device) * t
      denoised_nodes, denoised_edges = self.forward(nodes, edges, time)
      nodes, edges = self.reverse(denoised_nodes, nodes, denoised_edges, edges, t)

      if t % stepsize == 0:
        SketchDataset.render_graph(nodes[0,...,1:].cpu(), edges[0].cpu(), axes[0, j])
        SketchDataset.render_graph(denoised_nodes[0,...,1:].cpu(), denoised_edges[0].cpu(), axes[1, j])
        j = j - 1
    
    SketchDataset.render_graph(nodes[0,...,1:].cpu(), edges[0].cpu(), axes[0, 0])
    SketchDataset.render_graph(denoised_nodes[0,...,1:].cpu(), denoised_edges[0].cpu(), axes[1, 0])

    return nodes, edges
  
  @torch.no_grad()
  def reverse(self, pred_nodes, curr_nodes, pred_edges, curr_edges, timestep):
    denoised_nodes = torch.zeros_like(pred_nodes)
    denoised_edges = torch.zeros_like(pred_edges)

    denoised_nodes[...,0:2] = self.noise_scheduler.discrete_posterior_step(pred_nodes[...,0:2], curr_nodes[...,0:2], timestep)
    denoised_nodes[...,2:7] = self.noise_scheduler.discrete_posterior_step(pred_nodes[...,2:7], curr_nodes[...,2:7], timestep)
    
    weights = torch.cat(
      [pred_nodes[...,2,None].expand(-1, -1, 4), 
       pred_nodes[...,3,None].expand(-1, -1, 3), 
       pred_nodes[...,4,None].expand(-1, -1, 5),
       pred_nodes[...,5,None].expand(-1, -1, 2)], dim = -1)
    
    vals, _ = torch.max(pred_nodes[...,2:7], dim = -1, keepdim = True)
    denoised_nodes[...,7:] = self.noise_scheduler.continuous_posterior_step(weights / vals * pred_nodes[...,7:], curr_nodes[...,7:], timestep)

    edges_shape = pred_edges.shape
    denoised_edges = denoised_edges.view(edges_shape[0], -1, edges_shape[-1]) # Flatten middle
    pred_edges = pred_edges.view(edges_shape[0], -1, edges_shape[-1])
    curr_edges = curr_edges.view(edges_shape[0], -1, edges_shape[-1])

    denoised_edges[...,0:4] = self.noise_scheduler.discrete_posterior_step(pred_edges[...,0:4], curr_edges[...,0:4], timestep)
    denoised_edges[...,4:8] = self.noise_scheduler.discrete_posterior_step(pred_edges[...,4:8], curr_edges[...,4:8], timestep)
    denoised_edges[...,8:] = self.noise_scheduler.discrete_posterior_step(pred_edges[...,8:], curr_edges[...,8:], timestep)
    
    return denoised_nodes, denoised_edges.view(edges_shape) # Reshape to edge shape

class DiffusionModel(nn.Module):
  def __init__(self, node_dim, edge_dim, node_hidden_dim, edge_hidden_dim, cond_hidden_dim, num_heads, num_tf_layers, num_checkpoints, max_timestep, device: device):
    super().__init__()
    self.num_checkpoints = num_checkpoints
    self.nhd = node_hidden_dim
    self.ehd = edge_hidden_dim

    # Learned Time and Positional Embeddings
    self.time_embedder = SinuisodalEncoding(max_length = max_timestep, embedding_dimension = cond_hidden_dim, device = device)
    self.pos_embedder  = SinuisodalEncoding(max_length = MAX_NUM_PRIMITIVES, embedding_dimension = node_hidden_dim, device = device)
    self.mlp_in_conds = nn.Sequential(
      nn.Linear(in_features = cond_hidden_dim, out_features = cond_hidden_dim, device = device),
      nn.LeakyReLU(0.1),
      nn.Linear(in_features = cond_hidden_dim, out_features = cond_hidden_dim, device = device),
      nn.LeakyReLU(0.1)
    )

    # Input MLP layers
    self.mlp_in_nodes = nn.Sequential(
      nn.Linear(in_features = node_dim, out_features = 4 * node_hidden_dim, device = device),
      nn.LeakyReLU(0.1),
      nn.Linear(in_features = 4 * node_hidden_dim, out_features = node_hidden_dim, device = device)
    )
    self.mlp_in_edges = nn.Sequential(
      nn.Linear(in_features = edge_dim, out_features = edge_hidden_dim, device = device),
      nn.LeakyReLU(0.1),
      nn.Linear(in_features = edge_hidden_dim, out_features = edge_hidden_dim, device = device)
    )

    # Transformer Layers with Graph Attention Network
    self.block_layers = nn.ModuleList([
        TransformerLayer(
            node_dim = node_hidden_dim,
            edge_dim = edge_hidden_dim,
            cond_dim = cond_hidden_dim,
            num_heads = num_heads,
            device = device
        ) for _ in range(num_tf_layers)
    ])

    # Normalization
    self.node_in_norm = nn.LayerNorm(normalized_shape = node_hidden_dim, elementwise_affine = False, device = device)
    self.edge_in_norm = nn.LayerNorm(normalized_shape = edge_hidden_dim, elementwise_affine = False, device = device)
    # Conditioning
    # self.lin_cond = nn.Linear(in_features = cond_hidden_dim, out_features = 2 * (node_hidden_dim + edge_hidden_dim), device = device)
    self.lin_cond = nn.Linear(in_features = cond_hidden_dim, out_features = 2 * node_hidden_dim, device = device)
    # nn.init.zeros_(self.lin_cond.weight)
    # nn.init.zeros_(self.lin_cond.bias)
        
    # Output MLP layers
    self.mlp_out_params = nn.Sequential(
      nn.Linear(in_features = node_hidden_dim, out_features = 4 * node_hidden_dim, device = device),
      nn.LeakyReLU(0.1),
      nn.Linear(in_features = 4 * node_hidden_dim, out_features = 14, device = device)
    )
    self.mlp_out_types = nn.Sequential(
      nn.Linear(in_features = node_hidden_dim, out_features = 4 * node_hidden_dim, device = device),
      nn.LeakyReLU(0.1),
      nn.Linear(in_features = 4 * node_hidden_dim, out_features = node_dim - 14 - 1, device = device)
    )
    self.mlp_out_edges = nn.Sequential(
      nn.Linear(in_features = edge_hidden_dim, out_features = edge_hidden_dim, device = device),
      nn.LeakyReLU(0.1),
      nn.Linear(in_features = edge_hidden_dim, out_features = edge_dim, device = device)
    )

  def forward(self, nodes : Tensor, edges : Tensor, timestep : Tensor):
    nodes = self.mlp_in_nodes(nodes) + self.pos_embedder.embs # shape: (batch_size, num_nodes, node_hidden_dim)
    edges = self.mlp_in_edges(edges) # shape: (batch_size, num_nodes, num_nodes, edge_hidden_dim)
    # conds = torch.cat([self.pos_embedder.embs.unsqueeze(0).expand(nodes.size(0), -1, -1), self.time_embedder(timestep).unsqueeze(1).expand(-1, MAX_NUM_PRIMITIVES, -1)], dim = -1) # shape: (batch_size, num_nodes, cond_hidden_dim)
    conds = self.mlp_in_conds(self.time_embedder(timestep)).unsqueeze(1)

    checkpoints = self.num_checkpoints
    for layer in self.block_layers:
        nodes, edges = checkpoint(layer, nodes, edges, conds, use_reentrant = False) if checkpoints > 1 else layer(nodes, edges, conds) # shape: (batch_size, num_nodes, node_hidden_dim) ; shape: (batch_size, num_nodes, num_nodes, edge_hidden_dim)
        checkpoints = checkpoints - 1

    # node_shift, node_scale, edge_shift, edge_scale = self.lin_cond(conds).split(2 * [self.nhd] + 2 * [self.ehd], dim = -1)
    # nodes = F.leaky_relu(self.node_in_norm(nodes) * (1 + node_scale) + node_shift, 0.1)
    # edges = F.leaky_relu(self.edge_in_norm(edges) * (1 + edge_scale.unsqueeze(1)) + edge_shift.unsqueeze(1), 0.1)
    node_shift, node_scale = self.lin_cond(conds).chunk(chunks = 2, dim = -1)
    nodes = F.leaky_relu(self.node_in_norm(nodes) * (1 + node_scale) + node_shift, 0.1)
    edges = F.leaky_relu(self.edge_in_norm(edges), 0.1)
    
    nodes = torch.cat([torch.zeros_like(nodes[...,[0]]), self.mlp_out_types(nodes), self.mlp_out_params(nodes)], dim = -1) # shape: (batch_size, num_nodes, node_dim)
    edges = self.mlp_out_edges(edges)

    return nodes, edges

class TransformerLayer(nn.Module):
  def __init__(self, node_dim: int, edge_dim : int, cond_dim: int, num_heads: int, device: device):
    super().__init__()
    self.node_dim = node_dim
    self.edge_dim = edge_dim

    # Conditioning
    # self.lin_cond = nn.Linear(in_features = cond_dim, out_features = 4 * (node_dim + edge_dim), device = device)
    self.lin_cond = nn.Linear(in_features = cond_dim, out_features = 4 * node_dim, device = device)
    # nn.init.zeros_(self.lin_cond.weight)
    # nn.init.zeros_(self.lin_cond.bias)

    # Attention Layer
    self.attention_heads = MultiHeadAttention(node_dim = node_dim, edge_dim = edge_dim, num_heads = num_heads, device = device)

    # Normalization
    self.node_in_norm = nn.LayerNorm(normalized_shape = node_dim, elementwise_affine = False, device = device)
    self.edge_in_norm = nn.LayerNorm(normalized_shape = edge_dim, elementwise_affine = False, device = device)

    # Node and edge MLPs
    self.mlp_nodes = nn.Sequential(
      nn.Linear(in_features = node_dim, out_features = 4 * node_dim, device = device),
      nn.LeakyReLU(0.1),
      nn.Linear(in_features = 4 * node_dim, out_features = node_dim, device = device),
    )
    self.mlp_edges = nn.Sequential(
      nn.Linear(in_features = edge_dim, out_features = edge_dim, device = device),
      nn.LeakyReLU(0.1),
      nn.Linear(in_features = edge_dim, out_features = edge_dim, device = device),
    )

    # Normalization
    self.node_out_norm = nn.LayerNorm(normalized_shape = node_dim, elementwise_affine = False, device = device)
    self.edge_out_norm = nn.LayerNorm(normalized_shape = edge_dim, elementwise_affine = False, device = device)

  def forward(self, nodes : Tensor, edges : Tensor, conds : Tensor) -> Tensor:
    # Conditioning
    mul_in_node, add_in_node, mul_out_node, add_out_node = self.lin_cond(conds).chunk(chunks = 4, dim = -1)
    # mul_in_node, add_in_node, mul_out_node, add_out_node, mul_in_edge, add_in_edge, mul_out_edge, add_out_edge = self.lin_cond(conds).split(4 * [self.node_dim] + 4 * [self.edge_dim], dim = -1)
      
    # Attention
    delta_nodes, delta_edges = self.attention_heads(
      F.leaky_relu(self.node_in_norm(nodes) * (1 + mul_in_node) + add_in_node, 0.1), 
      F.leaky_relu(self.edge_in_norm(edges), 0.1) 
      # F.leaky_relu(self.edge_in_norm(edges) * (1 + mul_in_edge).unsqueeze(1) + add_in_edge.unsqueeze(1), 0.1)
    )
    nodes = nodes + delta_nodes
    edges = edges + delta_edges

    # MLP
    nodes = nodes + self.mlp_nodes(F.leaky_relu(self.node_out_norm(nodes) * (1 + mul_out_node) + add_out_node, 0.1))
    edges = edges + self.mlp_edges(F.leaky_relu(self.edge_out_norm(edges), 0.1)) 
    # edges = edges + self.mlp_edges(F.leaky_relu(self.edge_out_norm(edges) * (1 + mul_out_edge).unsqueeze(1) + add_out_edge.unsqueeze(1), 0.1))

    return nodes, edges
  
# Outer Product Attention Head
class MultiHeadAttention(nn.Module):
  def __init__(self, node_dim : int, edge_dim : int, num_heads : int, device : torch.device):
    super().__init__()
    self.node_dim = node_dim
    self.edge_dim = edge_dim
    self.num_heads = num_heads
    self.attn_dim = node_dim // num_heads

    self.lin_qkv = nn.Linear(in_features = node_dim, out_features = 3 * node_dim, device = device)

    self.lin_shift_scale = nn.Linear(in_features = edge_dim, out_features = 2 * node_dim, device = device)

    self.lin_nodes_out = nn.Sequential(
      # nn.LeakyReLU(0.1),
      nn.Linear(in_features = node_dim, out_features = node_dim, device = device)
    )
    self.lin_edges_out = nn.Sequential(
      nn.LeakyReLU(0.1),
      nn.Linear(in_features = node_dim, out_features = edge_dim, device = device),
    )

  def forward(self, nodes : Tensor, edges : Tensor):
    batch_size, num_nodes, _ = nodes.size()
        
    # Outer Product Attention -------
    queries, keys, values = self.lin_qkv(nodes).chunk(chunks = 3, dim = -1)   # batch_size x num_nodes x node_dim
    del nodes
    queries = queries.view(batch_size, num_nodes, self.num_heads, -1).transpose(1, 2) # batch_size x num_heads x num_nodes x (node_dim // num_heads)
    keys = keys.view(batch_size, num_nodes, self.num_heads, -1).transpose(1, 2)       # batch_size x num_heads x num_nodes x (node_dim // num_heads)
    values = values.view(batch_size, num_nodes, self.num_heads, -1).transpose(1, 2)   # batch_size x num_heads x num_nodes x (node_dim // num_heads)

    # queries = queries.unsqueeze(3)                            # batch_size x num_heads x num_nodes x 1 x attn_dim 
    # keys = keys.unsqueeze(2)                                  # batch_size x num_heads x 1 x num_nodes x attn_dim 
    attention = queries.unsqueeze(3) * keys.unsqueeze(2) # batch_size x num_heads x num_nodes x num_nodes x attn_dim
    del queries
    del keys

    # Condition attention based on edge features
    shift, scale = self.lin_shift_scale(edges).chunk(chunks = 2, dim = -1) # batch_size x num_nodes x num_nodes x node_dim
    del edges
    shift = shift.view(batch_size, num_nodes, num_nodes, self.num_heads, -1).permute(0, 3, 1, 2, 4) # batch_size x num_heads x num_nodes x num_nodes x attn_dim
    scale = scale.view(batch_size, num_nodes, num_nodes, self.num_heads, -1).permute(0, 3, 1, 2, 4) # batch_size x num_heads x num_nodes x num_nodes x attn_dim

    new_edges = attention * scale + attention + shift # batch_size x num_heads x num_nodes x num_nodes x attn_dim
    del shift
    del scale
        
    # Normalize attention
    # batch_size x num_heads x num_nodes x num_nodes (Finish dot product)
    attention = torch.softmax(input = new_edges.sum(dim = 4) / math.sqrt(self.node_dim // self.num_heads), dim = 2) # batch_size x num_heads x num_nodes x num_nodes (softmax) 

    # Weight node representations and sum
    # batch_size x num_heads x num_nodes x attn_dim
    weighted_values = (attention @ values.transpose(1, 2)).flatten(start_dim = 2) # batch_size x num_nodes x node_dim
    del values
    # Flatten attention heads
    new_edges = new_edges.permute(0, 2, 3, 1, 4).flatten(start_dim = 3) # batch_size x num_nodes x num_nodes x node_dim
        
    # Combine attention heads
    new_nodes = self.lin_nodes_out(weighted_values) # batch_size x num_nodes x node_dim
    new_edges = self.lin_edges_out(new_edges)       # batch_size x num_nodes x num_nodes x edge_dim

    return new_nodes, new_edges

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

    # Discrete variance schedule
    self.clamp_min = -10
    s = 16
    self.da_bar = (1-t) * (1 - t ** (1/s)) + (t) * (torch.exp(t * -s))
  
  def forward(self, nodes : Tensor, edges : Tensor, timestep : Tensor):
    ''' Apply noise to graph '''
    noisy_nodes = torch.zeros(size = nodes.size(), device = nodes.device)
    noisy_edges = torch.zeros(size = edges.size(), device = edges.device)
    
    # IsConstructible noise
    noisy_nodes[...,0:2] = self.apply_discrete_noise(nodes[...,0:2], timestep)
    # Primitive Types noise
    noisy_nodes[...,2:7] = self.apply_discrete_noise(nodes[...,2:7], timestep)
    # Primitive parameters noise
    noisy_nodes[...,7:] = self.apply_continuous_noise(nodes[...,7:], timestep)
    
    edges_shape = edges.shape
    noisy_edges = noisy_edges.view(edges_shape[0], -1, edges_shape[-1])
    edges = edges.view(edges_shape[0], -1, edges_shape[-1])

    # Sub A noise
    noisy_edges[...,0:4] = self.apply_discrete_noise(edges[...,0:4], timestep)
    # Sub B noise
    noisy_edges[...,4:8] = self.apply_discrete_noise(edges[...,4:8], timestep)
    # Constraint noise
    noisy_edges[...,8:] = self.apply_discrete_noise(edges[...,8:], timestep)
    
    return noisy_nodes, noisy_edges.view(edges_shape)
  
  def sample_latent(self, batch_size : int) -> Tensor:
    noisy_nodes = torch.zeros(size = (batch_size, MAX_NUM_PRIMITIVES, BINARY_CONSTRUCT_ISOCELES_NODE_DIM), device = self.device)
    noisy_edges = torch.zeros(size = (batch_size, MAX_NUM_PRIMITIVES, MAX_NUM_PRIMITIVES, EDGE_FEATURE_DIMENSION), device = self.device)

    # IsConstructible noise
    noisy_nodes[...,0:2] = torch.randn_like(noisy_nodes[...,0:2]).softmax(dim = -1)
    # Primitive Types noise
    noisy_nodes[...,2:7] = torch.randn_like(noisy_nodes[...,2:7]).softmax(dim = -1)
    # Primitive parameters noise
    noisy_nodes[...,7:] = torch.randn_like(noisy_nodes[...,7:])
    # Sub A Noise
    noisy_edges[...,0:4] = torch.randn_like(noisy_edges[...,0:4]).softmax(dim = -1)
    # Sub B Noise
    noisy_edges[...,4:8] = torch.randn_like(noisy_edges[...,4:8]).softmax(dim = -1)
    # Constraint Noise
    noisy_edges[...,8:] = torch.randn_like(noisy_edges[...,8:]).softmax(dim = -1)
    
    return noisy_nodes, noisy_edges
  
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
      
    a = torch.sqrt(self.da_bar[timestep, None, None])
    b = torch.sqrt(1 - self.da_bar[timestep, None, None])

    noise = torch.randn_like(params)
    return torch.softmax(a * params.log().clamp(self.clamp_min) + b * noise, dim = -1) #, noise
  
  def discrete_posterior_step(self, pred_params : Tensor, curr_params : Tensor, timestep : Tensor | int) -> Tensor:
    if type(timestep) is int:
      if timestep == 0: 
        return pred_params
      assert timestep > 0
      assert timestep < self.max_timestep
      timestep = torch.tensor(data = [timestep], device = pred_params.device)

    curr_a = self.da_bar[timestep] / self.da_bar[timestep - 1]
    curr_b = 1 - curr_a
    curr_a_bar = self.da_bar[timestep]
    curr_b_bar = 1 - curr_a_bar
    prev_a_bar = self.da_bar[timestep - 1]
    prev_b_bar = 1 - prev_a_bar

    if timestep > 1:
      log_pred = 2 * pred_params.log()
      log_curr = curr_params.log().clamp(self.clamp_min)
      mean = (prev_a_bar.sqrt() * curr_b * log_pred + curr_a.sqrt() * prev_b_bar * log_curr) / curr_b_bar
      noise = torch.randn_like(pred_params)
      return torch.softmax(mean + torch.sqrt(prev_b_bar / curr_b_bar * curr_b) * noise, dim = -1) #, noise
    else:
      return pred_params

'''----- Soft Gumb -----'''
# class CosineNoiseScheduler(nn.Module):
#   def __init__(self, max_timestep : int, device : torch.device):
#     super().__init__()
#     self.device = device
#     self.max_timestep = max_timestep
#     self.offset = .008 # Fixed offset to improve noise prediction at early timesteps

#     # Cosine Beta Schedule Formula: https://arxiv.org/abs/2102.09672
#     # self.a_bar = torch.cos((torch.linspace(0, 1, self.max_timestep + 1).to(self.device) + self.offset) * 0.5 * math.pi / (1 + self.offset)) ** 2
#     # self.a_bar = self.a_bar / self.a_bar[0]
#     # self.a_bar = self.a_bar.clamp(min = 0.0001, max = 0.9999)
#     self.a_bar = torch.cos(torch.linspace(0, 1, self.max_timestep + 1, device = device) * 0.5 * math.pi) ** 2 # 1 - torch.linspace(0, 1, self.max_timestep + 1, device = device) ** 2
#     self.a_bar = self.a_bar.clamp(min = 0.0001, max = 0.9999)
  
#   def forward(self, nodes : Tensor, edges : Tensor, timestep : Tensor):
#     ''' Apply noise to graph '''
#     noisy_nodes = torch.zeros(size = nodes.size(), device = nodes.device)
#     noisy_edges = torch.zeros(size = edges.size(), device = nodes.device)

#     # added_noise = torch.zeros(size = nodes.size(), device = nodes.device)
    
#     # IsConstructible noise
#     noisy_nodes[...,0:2] = self.apply_discrete_noise(nodes[...,0:2], timestep)
#     # Primitive Types noise
#     noisy_nodes[...,2:7] = self.apply_discrete_noise(nodes[...,2:7], timestep)
#     # Primitive parameters noise
#     noisy_nodes[...,7:] = self.apply_continuous_noise(nodes[...,7:], timestep)

#     edge_shape = edges.shape
#     noisy_edges = noisy_edges.view(edge_shape[0], -1, edge_shape[-1])
#     edges = edges.view(edge_shape[0], -1, edge_shape[-1])

#     # Sub A noise
#     noisy_edges[...,0:4] = self.apply_discrete_noise(edges[...,0:4], timestep)
#     noisy_edges[...,4:8] = self.apply_discrete_noise(edges[...,4:8], timestep)
#     noisy_edges[...,8:] = self.apply_discrete_noise(edges[...,8:], timestep)
    
#     return noisy_nodes, noisy_edges.view(edge_shape)
  
#   def sample_latent(self, batch_size : int) -> Tensor:
#     noisy_nodes = torch.zeros(size = (batch_size, 24, 21), device = self.device)
#     noisy_edges = torch.zeros(size = (batch_size, 24, 24, 17), device = self.device)

#     # IsConstructible noise
#     uniform_noise = torch.rand_like(noisy_nodes[...,0:2].clamp(min = 1e-10, max = 1 - 1e-10))
#     gumbel_noise = -torch.log(-torch.log(uniform_noise))
#     noisy_nodes[...,0:2] = gumbel_noise.softmax(dim = -1)
#     # Primitive Types noise
#     uniform_noise = torch.rand_like(noisy_nodes[...,2:7].clamp(min = 1e-10, max = 1 - 1e-10))
#     gumbel_noise = -torch.log(-torch.log(uniform_noise))
#     noisy_nodes[...,2:7] = gumbel_noise.softmax(dim = -1)
#     # Primitive parameters noise
#     gaussian_noise = torch.randn_like(noisy_nodes[...,7:].clamp(min = 1e-10, max = 1 - 1e-10))
#     noisy_nodes[...,7:] = gaussian_noise
#     # SubA noise
#     uniform_noise = torch.rand_like(noisy_edges[...,0:4].clamp(min = 1e-10, max = 1 - 1e-10))
#     gumbel_noise = -torch.log(-torch.log(uniform_noise))
#     noisy_edges[...,0:4] = gumbel_noise.softmax(dim = -1)
#     # SubB noise
#     uniform_noise = torch.rand_like(noisy_edges[...,4:8].clamp(min = 1e-10, max = 1 - 1e-10))
#     gumbel_noise = -torch.log(-torch.log(uniform_noise))
#     noisy_edges[...,4:8] = gumbel_noise.softmax(dim = -1)
#     # SubA noise
#     uniform_noise = torch.rand_like(noisy_edges[...,8:].clamp(min = 1e-10, max = 1 - 1e-10))
#     gumbel_noise = -torch.log(-torch.log(uniform_noise))
#     noisy_edges[...,8:] = gumbel_noise.softmax(dim = -1)
    
#     return noisy_nodes, noisy_edges
  
#   def apply_continuous_noise(self, params : Tensor, timestep : Tensor | int) -> Tensor:
#     if type(timestep) is int:
#       if timestep == 0: 
#         return params, 0 
#       assert timestep > 0 
#       assert timestep < self.max_timestep 
#       timestep = [timestep]

#     a = torch.sqrt(self.a_bar[timestep, None, None])
#     b = torch.sqrt(1 - self.a_bar[timestep, None, None])

#     noise = torch.randn_like(params)
#     return a * params + b * noise #, noise
  
#   def continuous_posterior_step(self, pred_params : Tensor, curr_params : Tensor, timestep : Tensor | int) -> Tensor:
#     if type(timestep) is int:
#       if timestep == 0: 
#         return pred_params
#       assert timestep > 0 
#       assert timestep < self.max_timestep 
#       timestep = torch.tensor(data = [timestep], device = pred_params.device)

#     assert timestep > 0, "Timestep is 0 for continuous posterior step!"

#     curr_a = self.a_bar[timestep] / self.a_bar[timestep - 1]
#     curr_b = 1 - curr_a
#     curr_a_bar = self.a_bar[timestep]
#     curr_b_bar = 1 - curr_a_bar
#     prev_a_bar = self.a_bar[timestep - 1]
#     prev_b_bar = 1 - prev_a_bar

#     if timestep > 1:
#       mean = (prev_a_bar.sqrt() * curr_b * pred_params + curr_a.sqrt() * prev_b_bar * curr_params) / curr_b_bar
#       noise = torch.randn_like(pred_params)
#       return mean + torch.sqrt(prev_b_bar / curr_b_bar * curr_b) * noise #, noise
#     else:
#       return pred_params
  
#   def apply_discrete_noise(self, params : Tensor, timestep : Tensor | int) -> Tensor:
#     if type(timestep) is int:
#       if timestep == 0: 
#         return params, 0 
#       assert timestep > 0
#       assert timestep < self.max_timestep
#       timestep = [timestep]
      
#     a = self.a_bar[timestep, None, None].sqrt()

#     D = params.size(-1)
#     noise = torch.log(-torch.log(torch.rand_like(params).clamp(min = 1e-10, max = 1 - 1e-10))) # Gumbel Noise
#     return torch.softmax(torch.log(a * params + (1 - a) / D) + noise, dim = -1) #, noise
  
#   def discrete_posterior_step(self, pred_params : Tensor, curr_params : Tensor, timestep : Tensor | int) -> Tensor:
#     if type(timestep) is int:
#       if timestep == 0: 
#         return pred_params
#       assert timestep > 0
#       assert timestep < self.max_timestep
#       timestep = torch.tensor(data = [timestep], device = pred_params.device)
      
#     D = pred_params.size(-1)
#     a_bar = self.a_bar[timestep, None, None]
#     prev_a_bar = self.a_bar[timestep - 1, None, None]
#     curr_a = self.a_bar[timestep, None, None] / self.a_bar[timestep - 1, None, None]
#     Q_bar = a_bar * torch.eye(D, device = pred_params.device) + (1 - a_bar) / D
#     prev_Q_bar = prev_a_bar * torch.eye(D, device = pred_params.device) + (1 - prev_a_bar) / D
#     curr_Q = curr_a * torch.eye(D, device = pred_params.device) + (1 - curr_a) / D

#     xt = curr_params # F.one_hot(torch.argmax(curr_params, dim = -1), D).to(pred_params.device).float()
#     qt = xt @ curr_Q.permute(0, 2, 1) # (b, m, d), since xt is onehot we are plucking out rows corresponding to p(x_t = class | x_(t-1))
#     qt_bar = xt @ Q_bar.permute(0, 2, 1) # (b, m, d), since xt is onehot we are plucking out rows corresponding to p(x_t = class | x_0)
#     q = qt.unsqueeze(2) / qt_bar.unsqueeze(3) # (b, m, d, d), perform an outer product so element at (b, m, i, j) = p(x_t = class | x_(t-1) = j) / p(x_t = class | x_0 = i)
#     q = q * prev_Q_bar.unsqueeze(1) # (b, m, d, d), broadcast multiply so element at (b, m, i, j) = p(x_t = class | x_(t-1) = j) * p(x_(t-1) = j | x_0 = i) / p(x_t = class | x_0 = i)
#     pred_class_probs = pred_params.unsqueeze(-2) # (b, n, 1, d), make probs into row vector
#     posterior_distribution = pred_class_probs @ q # (b, n, 1, d), batched vector-matrix multiply
#     posterior_distribution = posterior_distribution.squeeze(-2) # (b, n, d)

#     noise = torch.log(-torch.log(torch.rand_like(pred_params).clamp(min = 1e-10, max = 1 - 1e-10))) # Gumbel Noise
#     return torch.softmax(torch.log(posterior_distribution) + noise, dim = -1) #, noise