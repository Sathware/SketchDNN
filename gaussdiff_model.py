# %%
import torch
import math
from typing import Dict, List, Tuple, Any
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from config import NODE_FEATURE_DIMENSION, EDGE_FEATURE_DIMENSION, MAX_NUM_PRIMITIVES #, NodeBool, NodeType, NodeParams, EdgeSubA, EdgeSubB, EdgeType
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from dataset1 import SketchDataset
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint
from functools import partial
from IPython import display
import os

# %% [markdown]
# ### MODULES

# %%
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


# %%
class SkipLayerNorm(nn.Module):
  def __init__(self, dim : int, device : torch.device):
    super().__init__()

    # self.layer_norm = nn.LayerNorm(normalized_shape = dim, device = device, elementwise_affine = False, bias = False)
    self.layer_norm = nn.LayerNorm(normalized_shape = dim, device = device)

  def forward(self, A, B):
    return self.layer_norm(A + B)


# %%
class FiLM2(nn.Module):
  def __init__(self, dim_a : int, dim_b : int, device : torch.device):
    super().__init__()

    self.lin_mul = nn.Linear(in_features = dim_b, out_features = dim_a, device = device)
    self.lin_add = nn.Linear(in_features = dim_b, out_features = dim_a, device = device)

    self.mlp_out = nn.Sequential(
      nn.Linear(in_features = dim_a, out_features = dim_a, device = device),
      nn.LeakyReLU(0.01),
      nn.Linear(in_features = dim_a, out_features = dim_a, device = device),
      nn.LeakyReLU(0.01)
    )
    
  def forward(self, A, B):
    # A is shape (b, ..., dim_a)
    # B is shape (b, *, dim_b)
    mul = self.lin_mul(B) # (b, *, dim_a)
    add = self.lin_add(B) # (b, *, dim_a)

    return self.mlp_out(A * mul + add + A)


# %%
class SoftAttention2(nn.Module):
  def __init__(self, in_dim : int, out_dim, device : torch.device):
    super().__init__()

    self.lin_weights = nn.Sequential(
      nn.Linear(in_features = in_dim, out_features = in_dim, device = device),
      nn.LeakyReLU(0.01),
      nn.Linear(in_features = in_dim, out_features = 1, device = device)
    )
    self.lin_values = nn.Sequential(
      nn.Linear(in_features = in_dim, out_features = out_dim, device = device),
      # nn.LeakyReLU(0.1),
      # nn.Linear(in_features = out_dim, out_features = out_dim, device = device)
    )
    self.lin_out = nn.Sequential(
      nn.Linear(in_features = out_dim, out_features = out_dim, device = device),
      nn.LeakyReLU(0.01),
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


# %%
class TimeConditioningBlock(nn.Module):
  def __init__(self, class_dim : int, param_dim : int, edge_dim : int, time_dim : int, device : torch.device):
    super().__init__()
    self.film_class_time = FiLM2(dim_a = class_dim, dim_b = time_dim, device = device)
    self.film_param_time = FiLM2(dim_a = param_dim, dim_b = time_dim, device = device)
    self.film_edge_time = FiLM2(dim_a = edge_dim, dim_b = time_dim, device = device)
  
  def forward(self, classes : Tensor, params : Tensor, edges : Tensor, times : Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    return self.film_class_time(classes, times[:,None,:]), self.film_param_time(params, times[:,None,:]), self.film_edge_time(edges, times[:,None,None,:])


# %%
class SkipNormBlock(nn.Module):
  def __init__(self, class_dim : int, param_dim : int, edge_dim : int, device : torch.device):
    super().__init__()
    self.skip_class = SkipLayerNorm(dim = class_dim, device = device)
    self.skip_param = SkipLayerNorm(dim = param_dim, device = device)
    self.skip_edges = SkipLayerNorm(dim = edge_dim, device = device)
  
  def forward(self, classes : Tensor, params : Tensor, edges : Tensor, old_classes : Tensor, old_params : Tensor, old_edges : Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    return self.skip_class(classes, old_classes), F.leaky_relu(self.skip_param(params, old_params), 0.01), self.skip_edges(edges, old_edges)


# %%
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

    # self.lin_edge_class_mul = nn.Linear(in_features = edge_dim, out_features = class_dim, device = device)
    # self.lin_edge_class_add = nn.Linear(in_features = edge_dim, out_features = class_dim, device = device)
    # self.lin_edge_param_mul = nn.Linear(in_features = edge_dim, out_features = param_dim, device = device)
    # self.lin_edge_param_add = nn.Linear(in_features = edge_dim, out_features = param_dim, device = device)
    self.film_class_edge = FiLM2(dim_a = class_dim, dim_b = edge_dim, device = device)
    self.film_param_edge = FiLM2(dim_a = param_dim, dim_b = edge_dim, device = device)

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
    # attn_class = attn_class * self.lin_edge_class_mul(edges).reshape(b, n, n, self.num_heads, -1) + attn_class + self.lin_edge_class_add(edges).reshape(b, n, n, self.num_heads, -1)
    # attn_param = attn_param * self.lin_edge_param_mul(edges).reshape(b, n, n, self.num_heads, -1) + attn_param + self.lin_edge_param_add(edges).reshape(b, n, n, self.num_heads, -1)
    attn_class = self.film_class_edge(attn_class.reshape(b, n, n, -1), edges).reshape(b, n, n, self.num_heads, -1) # batch_size x num_nodes x num_nodes x num_heads x attn_dim
    attn_param = self.film_param_edge(attn_param.reshape(b, n, n, -1), edges).reshape(b, n, n, self.num_heads, -1) # batch_size x num_nodes x num_nodes x num_heads x attn_dim

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


# %%
class CrossTransformerLayer(nn.Module):
  def __init__(self, class_dim : int, param_dim : int, edge_dim : int, time_dim : int, num_heads : int, device : torch.device):
    super().__init__()

    self.lin_time_in = nn.Sequential(
      nn.Linear(in_features = time_dim, out_features = time_dim, device = device),
      nn.LeakyReLU(0.01),
    )

    self.time_cond1 = TimeConditioningBlock(class_dim = class_dim, param_dim = param_dim, edge_dim = edge_dim, time_dim = time_dim, device = device)
        
    self.skip1 = SkipNormBlock(class_dim = class_dim, param_dim = param_dim, edge_dim = edge_dim, device = device)

    self.cross_attn = CrossAttentionBlock(class_dim = class_dim, param_dim = param_dim, edge_dim = edge_dim, num_heads = num_heads, device = device)

    self.skip2 = SkipNormBlock(class_dim = class_dim, param_dim = param_dim, edge_dim = edge_dim, device = device)

    self.lin_time_out = nn.Sequential(
        nn.Linear(in_features = time_dim, out_features = time_dim, device = device),
        nn.LeakyReLU(0.01),
    )

    self.time_cond2 = TimeConditioningBlock(class_dim = class_dim, param_dim = param_dim, edge_dim = edge_dim, time_dim = time_dim, device = device)

    self.skip3 = SkipNormBlock(class_dim = class_dim, param_dim = param_dim, edge_dim = edge_dim, device = device)

    self.mlp_class_out = nn.Sequential(
        nn.Linear(in_features = class_dim, out_features = class_dim, device = device),
        nn.LeakyReLU(0.01),
        nn.Linear(in_features = class_dim, out_features = class_dim, device = device),
        nn.LeakyReLU(0.01),
    )
    self.mlp_param_out = nn.Sequential(
        nn.Linear(in_features = param_dim, out_features = param_dim, device = device),
        nn.LeakyReLU(0.01),
        nn.Linear(in_features = param_dim, out_features = param_dim, device = device),
        nn.LeakyReLU(0.01),
    )
    self.mlp_edges_out = nn.Sequential(
        nn.Linear(in_features = edge_dim, out_features = edge_dim, device = device),
        nn.LeakyReLU(0.01),
        nn.Linear(in_features = edge_dim, out_features = edge_dim, device = device),
        nn.LeakyReLU(0.01),
    )

    self.skip4 = SkipNormBlock(class_dim = class_dim, param_dim = param_dim, edge_dim = edge_dim, device = device)
    self.skip_times = nn.LayerNorm(normalized_shape = time_dim, device = device, elementwise_affine = False, bias = False)

  def forward(self, classes : Tensor, params : Tensor, edges : Tensor, times : Tensor):
    old_classes, old_params, old_edges = (classes, params, edges)
    old_times = times

    # Time Conditioning
    times = self.lin_time_in(times)
    classes, params, edges = self.time_cond1(classes, params, edges, times)

    # Skip Connection
    classes, params, edges = self.skip1(classes, params, edges, old_classes, old_params, old_edges)
    old_classes, old_params, old_edges = (classes, params, edges)

    # Cross Attention
    classes, params, edges = self.cross_attn(classes, params, edges)

    # Skip Connection
    classes, params, edges = self.skip2(classes, params, edges, old_classes, old_params, old_edges)
    old_classes, old_params, old_edges = (classes, params, edges)

    # Time Conditioning
    times = self.lin_time_out(times)
    classes, params, edges = self.time_cond2(classes, params, edges, times)

    # Skip Connection
    classes, params, edges = self.skip3(classes, params, edges, old_classes, old_params, old_edges)
    old_classes, old_params, old_edges = (classes, params, edges)

    # MLP
    classes, params, edges = (self.mlp_class_out(classes), self.mlp_param_out(params), self.mlp_edges_out(edges))

    # Skip Connection
    new_classes, new_params, new_edges = self.skip4(classes, params, edges, old_classes, old_params, old_edges)
    new_times = self.skip_times(times + old_times)

    return new_classes, new_params, new_edges, new_times


# %%
class GD3PM(nn.Module):
  def __init__(self, device : torch.device):
    super().__init__()
    self.device = device
    self.node_dim = NODE_FEATURE_DIMENSION
    self.edge_dim = EDGE_FEATURE_DIMENSION
    self.node_hidden_dim = 576 # hidden_node
    self.edge_hidden_dim = 64 # hidden_edge
    self.time_hidden_dim = 128 # hidden_time
    self.num_tf_layers = 32 # num_layers
    self.num_checkpoints = 28
    self.num_heads = 12
    self.max_timestep = 1000
    self.time_embedder = TimeEmbedder(self.max_timestep, self.time_hidden_dim, self.device)

    self.class_dim = self.node_hidden_dim
    self.param_dim = self.node_hidden_dim
    self.mlp_in_classes = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.node_hidden_dim, device = device),
                                      nn.LeakyReLU(0.01),
                                      nn.Linear(in_features = self.node_hidden_dim, out_features = self.class_dim, device = device),
                                      nn.LeakyReLU(0.01),
                                     )
    self.mlp_in_params = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.node_hidden_dim, device = device),
                                      nn.LeakyReLU(0.01),
                                      nn.Linear(in_features = self.node_hidden_dim, out_features = self.param_dim, device = device),
                                      nn.LeakyReLU(0.01),
                                     )
    # self.mlp_in_nodes = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.node_hidden_dim, device = device),
    #                                   nn.LeakyReLU(0.1),
    #                                   nn.Linear(in_features = self.node_hidden_dim, out_features = self.node_hidden_dim, device = device),
    #                                   nn.LeakyReLU(0.1),
    #                                  )
    self.mlp_in_edges = nn.Sequential(nn.Linear(in_features = self.edge_dim, out_features = self.edge_hidden_dim, device = device),
                                      nn.LeakyReLU(0.01),
                                      nn.Linear(in_features = self.edge_hidden_dim, out_features = self.edge_hidden_dim, device = device),
                                      nn.LeakyReLU(0.01),
                                     )
    
    self.block_layers = nn.ModuleList([CrossTransformerLayer(class_dim = self.class_dim,
                                                       param_dim = self.param_dim, 
                                                       edge_dim = self.edge_hidden_dim, 
                                                       time_dim = self.time_hidden_dim, 
                                                       num_heads = self.num_heads, 
                                                       device = self.device)
                                      for i in range(self.num_tf_layers)])
    
    self.mlp_out_nodes = nn.Sequential(nn.Linear(in_features = self.class_dim + self.param_dim, out_features = self.class_dim + self.param_dim, device = device),
                                       nn.LeakyReLU(0.01),
                                       nn.Linear(in_features = self.class_dim + self.param_dim, out_features = self.node_dim, device = device)
                                      )
    self.mlp_out_edges = nn.Sequential(nn.Linear(in_features = self.edge_hidden_dim, out_features = self.edge_hidden_dim, device = device),
                                       nn.LeakyReLU(0.01),
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
  
  # @torch.no_grad()
  # def sample(self, batch_size : int):
  #   # Sample Noise
  #   num_nodes = MAX_NUM_PRIMITIVES
  #   num_node_features = NODE_FEATURE_DIMENSION
  #   num_edge_features = EDGE_FEATURE_DIMENSION
  #   nodes = torch.zeros(batch_size, num_nodes, num_node_features)
  #   edges = torch.zeros(batch_size, num_nodes, num_nodes, num_edge_features)
  #   # binary noise (isConstructible)
  #   nodes[:,:,0] = torch.ones(size = (batch_size * num_nodes, 2)).multinomial(1)\
  #                       .reshape(batch_size, num_nodes).float()
  #   # categorical noise (primitive type)
  #   nodes[:,:,1:6] = F.one_hot(torch.ones(size = (batch_size * num_nodes, 5)).multinomial(1), 5)\
  #                     .reshape(batch_size, num_nodes, -1).float()
  #   # gaussian noise (primitive parameters)
  #   nodes[:,:,6:] = torch.randn(size = (batch_size, num_nodes, 14))
  #   # categorical noise (subnode a type)
  #   edges[:,:,:,0:4] = F.one_hot(torch.ones(size = (batch_size * num_nodes * num_nodes, 4)).multinomial(1), 4)\
  #                     .reshape(batch_size, num_nodes, num_nodes, -1).float()
  #   # categorical noise (subnode b type)
  #   edges[:,:,:,4:8] = F.one_hot(torch.ones(size = (batch_size * num_nodes * num_nodes, 4)).multinomial(1), 4)\
  #                     .reshape(batch_size, num_nodes, num_nodes, -1).float()
  #   # categorical noise (subnode a type)
  #   edges[:,:,:,8:] = F.one_hot(torch.ones(size = (batch_size * num_nodes * num_nodes, 9)).multinomial(1), 9)\
  #                    .reshape(batch_size, num_nodes, num_nodes, -1).float()
    
  #   nodes = nodes.to(self.device)
  #   edges = edges.to(self.device)
  #   return self.denoise(nodes, edges)

  # @torch.no_grad()
  # def denoise(self, nodes, edges):
  #   for t in reversed(range(1, self.max_timestep)):
  #     # model expects a timestep for each batch
  #     batch_size = nodes.size(0)
  #     time = torch.Tensor([t]).expand(batch_size).int()
  #     pred_node_noise, pred_edge_noise = self.forward(nodes, edges, time)
  #     nodes, edges = self.reverse_step(nodes, edges, pred_node_noise, pred_edge_noise, t)
  #   return nodes, edges
  
  # @torch.no_grad()
  # def noise(self, nodes, edges):
  #   nodes, edges, _, _ = self.noise_scheduler(nodes, edges, self.max_timestep - 1)
  #   return nodes, edges
  
  # @torch.no_grad()
  # def reverse_step(self, curr_nodes : Tensor, pred_nodes_noise : Tensor, curr_edges : Tensor, pred_edges_noise : Tensor, timestep : int):
  #   # Denoise one timestep
  #   new_nodes = torch.zeros_like(curr_nodes)
  #   new_edges = torch.zeros_like(curr_edges)
  #   pred_nodes = torch.zeros_like(curr_nodes)
  #   pred_edges = torch.zeros_like(curr_edges)
  #   # new_edges = torch.zeros_like(curr_edges)
  #   # What the model thinks the true graph is
  #   # pred_edges = torch.zeros_like(curr_edges)

  #   # # IsConstructible denoising
  #   # new_nodes[...,0], pred_nodes[...,0] =     self.noise_scheduler.apply_bernoulli_posterior_step(  curr_nodes[...,[0]], pred_node_noise[...,[0]], timestep)
  #   # # Primitive Types denoising
  #   # new_nodes[...,1:6], pred_nodes[...,1:6] = self.noise_scheduler.apply_multinomial_posterior_step(curr_nodes[...,1:6], pred_node_noise[...,1:6], timestep)
  #   # # Primitive parameters denoising
  #   # new_nodes[...,6:], pred_nodes[...,6:] =   self.noise_scheduler.apply_gaussian_posterior_step(   curr_nodes[...,6:],  pred_node_noise[...,6:],  timestep)
  #   # # Subnode A denoising
  #   # new_edges[...,0:4], pred_edges[...,0:4] = self.noise_scheduler.apply_multinomial_posterior_step(curr_edges[...,0:4], pred_edge_noise[...,0:4], timestep)
  #   # # Subnode B denoising
  #   # new_edges[...,4:8], pred_edges[...,4:8] = self.noise_scheduler.apply_multinomial_posterior_step(curr_edges[...,4:8], pred_edge_noise[...,4:8], timestep)
  #   # # Constraint Types denoising
  #   # new_edges[...,8:], pred_edges[...,8:] =   self.noise_scheduler.apply_multinomial_posterior_step(curr_edges[...,8:],  pred_edge_noise[...,8:],  timestep)

  #   # Node Denoising
  #   new_nodes[...], pred_nodes[...] = self.noise_scheduler.apply_gaussian_posterior_step(curr_nodes, pred_nodes_noise, timestep)
  #   new_edges[...], pred_edges[...] = self.noise_scheduler.apply_gaussian_posterior_step(curr_edges, pred_edges_noise, timestep)
  #   # Edge Denoising
  #   # new_edges[...], pred_edges[...] = self.noise_scheduler.apply_gaussian_posterior_step(curr_edges, pred_edge_noise, timestep)

  #   return new_nodes, new_edges, pred_nodes, pred_edges

# %% [markdown]
# ### Loss

# %%
# def diffusion_loss(pred_nodes : Tensor, true_nodes : Tensor, pred_edges : Tensor, true_edges : Tensor, params_mask : Tensor, loss_dict : dict) -> Tensor:
#     loss_scales = (2 * self.sqrt_a_bar[t,None,None]).clamp(1)
#     node_type_loss = ((nodes[...,:6] - denoised_nodes[...,:6]) ** 2).mean()
#     node_param_loss = (( loss_scales * (nodes[...,6:] - denoised_nodes[...,6:]) ) ** 2 * params_mask).mean()
#     edge_loss = ((edges - denoised_edges) ** 2).mean()
#     loss = node_type_loss + node_param_loss + 0.1 * edge_loss

#     loss_dict["node type"] = node_type_loss.item()
#     loss_dict["node param"] = node_param_loss.item()
#     loss_dict["node loss"] = node_type_loss.item() + node_param_loss.item()
#     loss_dict["edge loss"] = edge_loss.item()
#     loss_dict["total loss"] = node_type_loss.item() + node_param_loss.item() + edge_loss.item()


#     return total_loss
