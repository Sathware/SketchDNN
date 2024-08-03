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

# Graph Transformer Layer outlined by DiGress Graph Diffusion
class TransformerLayer(nn.Module):
    def __init__(self, num_heads : int, node_dim : int, edge_dim : int, device : torch.device):
        super().__init__()
        self.num_heads = num_heads
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.attention_heads = MultiHeadAttention(node_dim = self.node_dim, edge_dim = self.edge_dim, num_heads = self.num_heads, device = device)

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
        
        self.layer_norm_nodes2 = nn.Sequential(nn.LayerNorm(normalized_shape = self.node_dim, device = device),
                                               nn.LeakyReLU(0.05)
                                              )
        self.layer_norm_edges2 = nn.Sequential(nn.LayerNorm(normalized_shape = self.edge_dim, device = device),
                                              #  nn.Dropout(p = 0.05)
                                              )
    
    def forward(self, nodes : Tensor, edges : Tensor) -> Tuple[Tensor, Tensor]:
        # Perform multi head attention
        # attn_nodes, attn_edges = checkpoint(self.attention_heads, nodes, edges, use_reentrant = False) # batch_size x num_nodes x node_dim ; batch_size x num_nodes x num_nodes x edge_dim
        
        attn_nodes, attn_edges = self.attention_heads(nodes, edges)

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
    def __init__(self, node_dim : int, edge_dim : int, num_heads : int, device : torch.device):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.attn_dim = node_dim // num_heads

        self.lin_query = nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device)
        self.lin_key = nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device)
        self.lin_value = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device),
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

    def forward(self, nodes : Tensor, edges : Tensor):
        batch_size, num_nodes, _ = nodes.size()
        
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

class Sharpener(nn.Module):
  def __init__(self, device : torch.device):
    super().__init__()
    self.device = device
    self.node_dim = NODE_FEATURE_DIMENSION
    self.edge_dim = EDGE_FEATURE_DIMENSION
    self.node_hidden_dim = 256 # hidden_node
    self.edge_hidden_dim = 128 # hidden_edge
    self.time_hidden_dim = 128 # hidden_time
    self.num_heads = 8
    self.num_tf_layers = 12
    self.num_checkpoints = 0
    # self.block1 = DenoiseBlock(device = self.device, num_layers = 16)
    # self.block2 = DenoiseBlock(device = self.device, num_layers = 16)
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

  def forward(self, nodes : Tensor, edges : Tensor):
    nodes = self.mlp_in_nodes(nodes) # batch_size x num_nodes x hidden_dim
    edges = self.mlp_in_edges(edges) # batch_size x num_nodes x num_nodes x hidden_dim

    num_checkpoint = self.num_checkpoints
    for layer in self.block_layers:      
      nodes, edges = checkpoint(layer, nodes, edges, use_reentrant = False) if num_checkpoint > 0 else layer(nodes, edges)
      num_checkpoint = num_checkpoint - 1
    
    nodes = self.mlp_out_nodes(nodes)
    edges = self.mlp_out_edges(edges)
    return nodes, edges