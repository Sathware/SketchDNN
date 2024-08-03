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
      nn.LeakyReLU(0.05),
      nn.Linear(in_features = dim_a, out_features = dim_a, device = device),
      nn.LeakyReLU(0.05)
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
      nn.LeakyReLU(0.05),
      nn.Linear(in_features = in_dim, out_features = 1, device = device)
    )
    self.lin_values = nn.Sequential(
      nn.Linear(in_features = in_dim, out_features = out_dim, device = device),
      # nn.LeakyReLU(0.05),
      # nn.Linear(in_features = out_dim, out_features = out_dim, device = device)
    )
    self.lin_out = nn.Sequential(
      nn.Linear(in_features = out_dim, out_features = out_dim, device = device),
      nn.LeakyReLU(0.05),
      # nn.Linear(in_features = out_dim, out_features = out_dim, device = device),
      # nn.LeakyReLU(0.05)
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
    return self.skip_class(classes, old_classes), F.leaky_relu(self.skip_param(params, old_params), 0.05), self.skip_edges(edges, old_edges)


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
      nn.LeakyReLU(0.05),
    )

    self.time_cond1 = TimeConditioningBlock(class_dim = class_dim, param_dim = param_dim, edge_dim = edge_dim, time_dim = time_dim, device = device)
        
    self.skip1 = SkipNormBlock(class_dim = class_dim, param_dim = param_dim, edge_dim = edge_dim, device = device)

    self.cross_attn = CrossAttentionBlock(class_dim = class_dim, param_dim = param_dim, edge_dim = edge_dim, num_heads = num_heads, device = device)

    self.skip2 = SkipNormBlock(class_dim = class_dim, param_dim = param_dim, edge_dim = edge_dim, device = device)

    self.lin_time_out = nn.Sequential(
        nn.Linear(in_features = time_dim, out_features = time_dim, device = device),
        nn.LeakyReLU(0.05),
    )

    self.time_cond2 = TimeConditioningBlock(class_dim = class_dim, param_dim = param_dim, edge_dim = edge_dim, time_dim = time_dim, device = device)

    self.skip3 = SkipNormBlock(class_dim = class_dim, param_dim = param_dim, edge_dim = edge_dim, device = device)

    self.mlp_class_out = nn.Sequential(
        nn.Linear(in_features = class_dim, out_features = class_dim, device = device),
        nn.LeakyReLU(0.05),
        nn.Linear(in_features = class_dim, out_features = class_dim, device = device),
        nn.LeakyReLU(0.05),
    )
    self.mlp_param_out = nn.Sequential(
        nn.Linear(in_features = param_dim, out_features = param_dim, device = device),
        nn.LeakyReLU(0.05),
        nn.Linear(in_features = param_dim, out_features = param_dim, device = device),
        nn.LeakyReLU(0.05),
    )
    self.mlp_edges_out = nn.Sequential(
        nn.Linear(in_features = edge_dim, out_features = edge_dim, device = device),
        nn.LeakyReLU(0.05),
        nn.Linear(in_features = edge_dim, out_features = edge_dim, device = device),
        nn.LeakyReLU(0.05),
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
class DenoiseBlock(nn.Module):
  def __init__(self, device : torch.device, num_layers : int):
    super().__init__()
    self.device = device
    self.node_dim = NODE_FEATURE_DIMENSION
    self.edge_dim = EDGE_FEATURE_DIMENSION
    self.node_hidden_dim = 512 # hidden_node
    self.edge_hidden_dim = 128 # hidden_edge
    self.time_hidden_dim = 128 # hidden_time
    self.num_tf_layers = num_layers # num_layers
    self.num_checkpoints = 0
    self.num_heads = 8
    self.max_timestep = 1000
    self.time_embedder = TimeEmbedder(self.max_timestep, self.time_hidden_dim, self.device)

    self.class_dim = self.node_hidden_dim
    self.param_dim = self.node_hidden_dim
    self.mlp_in_classes = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.node_hidden_dim, device = device),
                                      nn.LeakyReLU(0.05),
                                      nn.Linear(in_features = self.node_hidden_dim, out_features = self.class_dim, device = device),
                                      nn.LeakyReLU(0.05),
                                     )
    self.mlp_in_params = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.node_hidden_dim, device = device),
                                      nn.LeakyReLU(0.05),
                                      nn.Linear(in_features = self.node_hidden_dim, out_features = self.param_dim, device = device),
                                      nn.LeakyReLU(0.05),
                                     )
    
    self.mlp_in_edges = nn.Sequential(nn.Linear(in_features = self.edge_dim, out_features = self.edge_hidden_dim, device = device),
                                      nn.LeakyReLU(0.05),
                                      nn.Linear(in_features = self.edge_hidden_dim, out_features = self.edge_hidden_dim, device = device),
                                      nn.LeakyReLU(0.05),
                                     )
    
    self.block_layers = nn.ModuleList([CrossTransformerLayer(class_dim = self.class_dim,
                                                       param_dim = self.param_dim, 
                                                       edge_dim = self.edge_hidden_dim, 
                                                       time_dim = self.time_hidden_dim, 
                                                       num_heads = self.num_heads, 
                                                       device = self.device)
                                      for i in range(self.num_tf_layers)])
    
    self.mlp_out_nodes = nn.Sequential(nn.Linear(in_features = self.class_dim + self.param_dim, out_features = self.class_dim + self.param_dim, device = device),
                                       nn.LeakyReLU(0.05),
                                       nn.Linear(in_features = self.class_dim + self.param_dim, out_features = self.node_dim, device = device)
                                      )
    self.mlp_out_edges = nn.Sequential(nn.Linear(in_features = self.edge_hidden_dim, out_features = self.edge_hidden_dim, device = device),
                                       nn.LeakyReLU(0.05),
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

# Graph Transformer Layer outlined by DiGress Graph Diffusion
class TransformerLayer(nn.Module):
    def __init__(self, num_heads : int, node_dim : int, edge_dim : int, time_dim : int, device : torch.device):
        super().__init__()
        self.num_heads = num_heads
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim

        self.attention_heads = MultiHeadAttention(node_dim = self.node_dim, edge_dim = self.edge_dim, time_dim = self.time_dim, num_heads = self.num_heads, device = device)

        self.layer_norm_nodes = nn.Sequential(nn.LayerNorm(normalized_shape = self.node_dim, device = device),
                                              nn.LeakyReLU(0.05)
                                             )
        self.layer_norm_edges = nn.Sequential(nn.LayerNorm(normalized_shape = self.edge_dim, device = device),
                                              # nn.Dropout(p = 0.05)
                                             )

        self.mlp_nodes = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device),
                                       nn.LeakyReLU(0.05),
                                    #    nn.Dropout(p = 0.05),
                                       nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device),
                                      )
        
        self.mlp_edges = nn.Sequential(nn.Linear(in_features = self.edge_dim, out_features = self.edge_dim, device = device),
                                       nn.LeakyReLU(0.05),
                                    #    nn.Dropout(p = 0.05),
                                       nn.Linear(in_features = self.edge_dim, out_features = self.edge_dim, device = device),
                                      )

        # self.mlp_times = nn.Sequential(nn.Linear(in_features = self.time_dim, out_features = self.time_dim, device = device),
        #                                nn.LeakyReLU(0.05),
        #                             #    nn.Dropout(p = 0.05),
        #                                nn.Linear(in_features = self.time_dim, out_features = self.time_dim, device = device),)
        
        self.layer_norm_nodes2 = nn.Sequential(nn.LayerNorm(normalized_shape = self.node_dim, device = device),
                                               nn.LeakyReLU(0.05)
                                              )
        self.layer_norm_edges2 = nn.Sequential(nn.LayerNorm(normalized_shape = self.edge_dim, device = device),
                                              #  nn.Dropout(p = 0.05)
                                              )
    
    def forward(self, nodes : Tensor, edges : Tensor, times : Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Perform multi head attention
        # attn_nodes, attn_edges = checkpoint(self.attention_heads, nodes, edges, use_reentrant = False) # batch_size x num_nodes x node_dim ; batch_size x num_nodes x num_nodes x edge_dim
        
        attn_nodes, attn_edges = self.attention_heads(nodes, edges, times)

        # Layer normalization with a skip connection
        attn_nodes = self.layer_norm_nodes(attn_nodes + nodes) # batch_size x num_nodes x node_dim
        attn_edges = self.layer_norm_edges(attn_edges + edges) # batch_size x num_nodes x num_nodes x edge_dim
        del nodes
        del edges

        # MLP out
        new_nodes = self.mlp_nodes(attn_nodes) # batch_size x num_nodes x node_dim
        new_edges = self.mlp_edges(attn_edges) # batch_size x num_nodes x num_nodes x edge_dim
        # new_times = self.mlp_times(times)

        # Second layer normalization with a skip connection
        new_nodes = self.layer_norm_nodes2(new_nodes + attn_nodes) # batch_size x num_nodes x node_dim
        new_edges = self.layer_norm_edges2(new_edges + attn_edges) # batch_size x num_nodes x num_nodes x edge_dim
        del attn_nodes
        del attn_edges

        return new_nodes, new_edges

# Outer Product Attention Head
class MultiHeadAttention(nn.Module):
    def __init__(self, node_dim : int, edge_dim : int, time_dim : int, num_heads : int, device : torch.device):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.num_heads = num_heads
        self.attn_dim = node_dim // num_heads

        self.lin_query = nn.Linear(in_features = self.node_dim + self.time_dim, out_features = self.node_dim, device = device)
        self.lin_key = nn.Linear(in_features = self.node_dim + self.time_dim, out_features = self.node_dim, device = device)
        self.lin_value = nn.Sequential(nn.Linear(in_features = self.node_dim + self.time_dim, out_features = self.node_dim, device = device),
                                      #  nn.LeakyReLU(0.05),
                                      #  nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device)
                                      )

        self.lin_mul = nn.Sequential(nn.Linear(in_features = self.edge_dim, out_features = self.node_dim, device = device),
                                    #  nn.GELU(approximate='tanh'),
                                    #  nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device)
                                    )
        self.lin_add = nn.Sequential(nn.Linear(in_features = self.edge_dim, out_features = self.node_dim, device = device),
                                    #  nn.GELU(approximate='tanh'),
                                    #  nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device)
                                    )
        #self.edge_film = FiLM(self.edge_dim, self.node_dim, device = device)

        self.lin_nodes_out = nn.Sequential(
                                           nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device),
                                          #  nn.LeakyReLU(0.05),
                                          #  nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device)
                                          )
        self.lin_edges_out = nn.Sequential(nn.LeakyReLU(0.05),
                                           nn.Linear(in_features = self.node_dim, out_features = self.edge_dim, device = device),
                                          #  nn.LeakyReLU(0.05),
                                          #  nn.Linear(in_features = self.node_dim, out_features = self.edge_dim, device = device)
                                          )

    def forward(self, nodes : Tensor, edges : Tensor, times : Tensor):
        batch_size, num_nodes, _ = nodes.size()
        nodes = torch.cat([nodes, times.unsqueeze(1).expand(-1, num_nodes, -1)], dim = -1)
        
        # Outer Product Attention -------
        queries = self.lin_query(nodes).view(batch_size, num_nodes, self.num_heads, -1) # batch_size x num_nodes x num_heads x attn_dim
        keys = self.lin_key(nodes).view(batch_size, num_nodes, self.num_heads, -1)      # batch_size x num_nodes x num_heads x attn_dim
        # queries = queries.unsqueeze(2)                            # batch_size x num_nodes x 1 x num_heads x attn_dim 
        # keys = keys.unsqueeze(1)                                  # batch_size x 1 x num_nodes x num_heads x attn_dim 
        attention = queries.unsqueeze(2) * keys.unsqueeze(1) / math.sqrt(self.node_dim) # batch_size x num_nodes x num_nodes x num_heads x attn_dim
        del queries
        del keys

        # Condition attention based on edge features
        edges_mul = self.lin_mul(edges).view(batch_size, num_nodes, num_nodes, self.num_heads, -1) # batch_size x num_nodes x num_nodes x num_heads x attn_dim
        edges_add = self.lin_add(edges).view(batch_size, num_nodes, num_nodes, self.num_heads, -1) # batch_size x num_nodes x num_nodes x num_heads x attn_dim
        del edges
        new_edges = attention * edges_mul + attention + edges_add # batch_size x num_nodes x num_nodes x num_heads x attn_dim
        del edges_add
        del edges_mul
        
        # Normalize attention
                                                                           # batch_size x num_nodes x num_nodes x num_heads (Finish dot product)
        attention = torch.softmax(input = new_edges.sum(dim = 4), dim = 2) # batch_size x num_nodes x num_nodes x num_heads (softmax) 

        # Weight node representations and sum
        values = self.lin_value(nodes).view(batch_size, num_nodes, self.num_heads, -1)  # batch_size x num_nodes x num_heads x attn_dim
        del nodes
                                                                                                             # batch_size x num_nodes x num_heads x attn_dim
        weighted_values = (attention.unsqueeze(4) * values.unsqueeze(1)).sum(dim = 2).flatten(start_dim = 2) # batch_size x num_nodes x node_dim
        del values
        # Flatten attention heads
        new_edges = new_edges.flatten(start_dim = 3)
        # weighted_values = weighted_values.flatten(start_dim = 2)
        
        # Combine attention heads
        new_nodes = self.lin_nodes_out(weighted_values)
        new_edges = self.lin_edges_out(new_edges)

        return new_nodes, new_edges

# Feature-Wise Linear Modulation Layer
class FiLM(nn.Module):
    def __init__(self, a_feature_dim : int, b_feature_dim : int, device : torch.device):
        super().__init__()
        
        self.lin1 = nn.Linear(in_features = a_feature_dim, out_features = b_feature_dim, device = device)
        self.lin2 = nn.Linear(in_features = a_feature_dim, out_features = b_feature_dim, device = device)
    
    def forward(self, a : Tensor, b : Tensor):
        mul = self.lin1(a)
        add = self.lin2(a)

        # For vanilla FiLM you are only supposed to do mul * b + add
        # I assume digress put the additional '+ b' as a skip connection
        return mul * b + add + b

# Principal Neighbourhood Aggregation Layer
class PNA(nn.Module):
    def __init__(self):
        super().__init__()
        # self.mean_aggr = MeanAggregation()
        # self.max_aggr = MaxAggregation()
        # self.min_aggr = MinAggregation()
        # self.std_aggr = StdAggregation()
    
    def forward(self, input : Tensor, dim):
        # find the biggest element in dim
        mean = torch.mean(input, dim)
        max = torch.amax(input, dim)
        min = torch.amin(input, dim)
        stdev = torch.std(input, dim) # Standard Deviation

        return torch.cat(tensors = (mean, max, min, stdev), dim = 1)

class TimeEmbedder1(nn.Module):
  def __init__(self, max_timestep : int, embedding_dimension : int, device : torch.device):
    super().__init__()
    self.device = device
    self.embed_dim = embedding_dimension
    self.max_timestep = max_timestep
    
    self.embedder = nn.Linear(in_features = max_timestep, out_features = self.embed_dim, device = self.device)
      
  def forward(self, timestep : Tensor):
    return self.embedder(F.one_hot(timestep, self.max_timestep).float())

class GD3PM(nn.Module):
  def __init__(self, device : torch.device):
    super().__init__()
    self.device = device
    self.node_dim = NODE_FEATURE_DIMENSION
    self.edge_dim = EDGE_FEATURE_DIMENSION
    self.node_hidden_dim = 576 # hidden_node
    self.edge_hidden_dim = 128 # hidden_edge
    self.time_hidden_dim = 128 # hidden_time
    self.num_heads = 12
    self.num_tf_layers = 48
    self.num_checkpoints = 28
    self.max_timestep = 1000
    # self.block1 = DenoiseBlock(device = self.device, num_layers = 16)
    # self.block2 = DenoiseBlock(device = self.device, num_layers = 16)
    self.time_embedder = TimeEmbedder1(self.max_timestep, self.time_hidden_dim, self.device)
    self.mlp_in_nodes = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.node_hidden_dim, device = device),
                                      nn.LeakyReLU(0.05),
                                      nn.Linear(in_features = self.node_hidden_dim, out_features = self.node_hidden_dim, device = device),
                                      nn.LeakyReLU(0.05),
                                     )
    
    self.mlp_in_edges = nn.Sequential(nn.Linear(in_features = self.edge_dim, out_features = self.edge_hidden_dim, device = device),
                                      nn.LeakyReLU(0.05),
                                      nn.Linear(in_features = self.edge_hidden_dim, out_features = self.edge_hidden_dim, device = device),
                                      nn.LeakyReLU(0.05),
                                     )
    
    self.block_layers = nn.ModuleList([TransformerLayer(node_dim = self.node_hidden_dim,
                                                       edge_dim = self.edge_hidden_dim, 
                                                       time_dim = self.time_hidden_dim,
                                                       num_heads = self.num_heads, 
                                                       device = self.device)
                                      for i in range(self.num_tf_layers)])
    
    self.mlp_out_nodes = nn.Sequential(nn.Linear(in_features = self.node_hidden_dim, out_features = self.node_hidden_dim, device = device),
                                       nn.LeakyReLU(0.05),
                                       nn.Linear(in_features = self.node_hidden_dim, out_features = self.node_dim, device = device)
                                      )
    self.mlp_out_edges = nn.Sequential(nn.Linear(in_features = self.edge_hidden_dim, out_features = self.edge_hidden_dim, device = device),
                                       nn.LeakyReLU(0.05),
                                       nn.Linear(in_features = self.edge_hidden_dim, out_features = self.edge_dim, device = device)
                                      )

  def forward(self, nodes : Tensor, edges : Tensor, timestep : Tensor):
    time_encs = self.time_embedder(timestep) # batch_size x hidden_dim
    nodes = self.mlp_in_nodes(nodes) # batch_size x num_nodes x hidden_dim
    edges = self.mlp_in_edges(edges) # batch_size x num_nodes x num_nodes x hidden_dim

    num_checkpoint = self.num_checkpoints
    for layer in self.block_layers:      
      nodes, edges = checkpoint(layer, nodes, edges, time_encs, use_reentrant = False) if num_checkpoint > 0 else layer(nodes, edges, time_encs)
      num_checkpoint = num_checkpoint - 1
    
    nodes = self.mlp_out_nodes(nodes)
    edges = self.mlp_out_edges(edges)
    return nodes, edges
