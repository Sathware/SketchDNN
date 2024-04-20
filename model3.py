# %%
import torch
import math
from typing import Dict, List, Tuple, Any
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from config import NODE_FEATURE_DIMENSION, EDGE_FEATURE_DIMENSION, MAX_NUM_PRIMITIVES, GRAPH_EMBEDDING_SIZE, NUM_PRIMITIVE_TYPES
# %%
class GVAE(nn.Module):
    def __init__(self, device : torch.device):
        super().__init__()
        self.device = device
        self.categorical_node_dim = NUM_PRIMITIVE_TYPES + 1 # 6
        self.parameter_node_dim = NODE_FEATURE_DIMENSION - self.categorical_node_dim # 14
        self.node_dim = NODE_FEATURE_DIMENSION # 20
        self.edge_dim = EDGE_FEATURE_DIMENSION # 17
        self.graph_emb_dim = 512

        self.encoder = TransformerEncoder(node_dim = self.node_dim, 
                                          edge_dim = self.edge_dim, 
                                          graph_emb_dim = self.graph_emb_dim, 
                                          num_tf_layers = 12, 
                                          num_heads = 8,
                                          device = self.device
                                         )
        
        self.categorical_decoder = TransformerDecoder(node_dim = self.categorical_node_dim, 
                                          edge_dim = self.edge_dim, 
                                          graph_emb_dim = self.graph_emb_dim,
                                          num_tf_layers = 16,
                                          num_heads = 8, 
                                          device = self.device
                                         )

        self.parameter_decoder = ParameterDecoder(parameter_node_dim = self.parameter_node_dim,
                                                  categorical_node_dim = self.categorical_node_dim,
                                                  edge_dim = self.edge_dim,
                                                  graph_emb_dim = self.graph_emb_dim,
                                                  num_layers = 24,
                                                  device = self.device
                                                 )
    
    def forward(self, nodes : Tensor, edges : Tensor):
        means, logvars = self.encoder(nodes, edges)
        latents = self.sample_latent(means, logvars)
        pred_node_types, pred_edge_types = self.categorical_decoder(latents)
        pred_params = self.parameter_decoder(latents, 
                                             pred_node_types.detach().clone().to(self.device), 
                                             pred_edge_types.detach().clone().to(self.device)
                                            )
        
        pred_nodes = torch.cat(tensors = (pred_node_types, pred_params), dim = 2)
        pred_edges = pred_edge_types
        return pred_nodes, pred_edges, means, logvars
    
    def sample_latent(self, mean : Tensor, log_variance : Tensor):
        return mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
    
    @torch.no_grad()
    def sample_graph(self) -> Tuple[Tensor, Tensor]:
        latent = torch.randn(size = (self.graph_emb_dim,), device = self.device)
        nodes, edges = self.decoder(latent)

        nodes[:,:,1:6] = torch.exp(nodes[:,:,1:6], dim = 2) # Softmax for primitive classes
        
        # softmax for constraints; Conceptual map => n1 (out) -> n2 (in) i.e. out_node, edge, in_node
        edges[:,:,:,0:4] = torch.exp(edges[:,:,:,0:4], dim = 3) # Softmax for out_node subnode type
        edges[:,:,:,4:8] = torch.exp(edges[:,:,:,4:8], dim = 3) # Softmax for in_node subnode type
        edges[:,:,:,8: ] = torch.exp(edges[:,:,:,8: ], dim = 3) # Softmax for edge type
        return torch.squeeze(nodes), torch.squeeze(edges)

class TransformerEncoder(nn.Module):
    def __init__(self, node_dim : int, edge_dim : int, graph_emb_dim : int, num_tf_layers: int, num_heads : int, device : torch.device): # perm_emb_dim: int,
        super().__init__()
        self.node_dim = node_dim # Number of features per node
        self.edge_dim = edge_dim # Number of features per edge
        self.graph_emb_dim = graph_emb_dim # Size of graph embedding vector
        self.num_nodes = MAX_NUM_PRIMITIVES # Number of nodes in each graph
        self.num_edges = self.num_nodes * self.num_nodes # Number of edges in each graph
        
        self.hidden_dim = 192
        self.num_tf_layers = num_tf_layers
        self.num_heads = num_heads

        self.mlp_in_nodes = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.hidden_dim, device = device),
                                          nn.ReLU(),
                                        #   nn.Dropout(p = 0.1),
                                          nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
                                          nn.ReLU(),
                                        #   nn.Dropout(p = 0.1)
                                         )

        self.mlp_in_edges = nn.Sequential(nn.Linear(in_features = self.edge_dim, out_features = self.hidden_dim, device = device),
                                          nn.ReLU(),
                                        #   nn.Dropout(p = 0.1),
                                          nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
                                          nn.ReLU(),
                                        #   nn.Dropout(p = 0.1)
                                         )
        
        self.tf_layers = nn.ModuleList([TransformerLayer(num_heads = self.num_heads, 
                                                         node_dim = self.hidden_dim,
                                                         edge_dim = self.hidden_dim, 
                                                         device = device
                                                        ) 
                                        for _ in range(self.num_tf_layers)])

        self.mlp_haggr_weights = nn.Sequential(nn.Linear(in_features = 3 * self.hidden_dim, out_features = 3 * self.hidden_dim, device = device),
                                              nn.ReLU(),
                                            #   nn.Dropout(p = 0.1),
                                              nn.Linear(in_features = 3 * self.hidden_dim, out_features = 1, device = device),
                                              nn.Softmax(dim = 2),
                                            #   nn.Dropout(p = 0.1)
                                             )
        self.mlp_haggr_values = nn.Sequential(nn.Linear(in_features = 3 * self.hidden_dim, out_features = 3 * self.hidden_dim, device = device),
                                              nn.ReLU(),
                                            #   nn.Dropout(p = 0.1),
                                              nn.Linear(in_features = 3 * self.hidden_dim, out_features = self.hidden_dim, device = device)
                                             )
        
        self.mlp_mean = nn.Sequential(nn.Linear(in_features = self.num_nodes * self.hidden_dim, out_features = self.graph_emb_dim, device = device),
                                      nn.ReLU(),
                                    #   nn.Dropout(p = 0.1),
                                      nn.Linear(in_features = self.graph_emb_dim, out_features = self.graph_emb_dim, device = device)
                                     )
        self.mlp_logvar = nn.Sequential(nn.Linear(in_features = self.num_nodes * self.hidden_dim, out_features = self.graph_emb_dim, device = device),
                                        nn.ReLU(),
                                        # nn.Dropout(p = 0.1),
                                        nn.Linear(in_features = self.graph_emb_dim, out_features = self.graph_emb_dim, device = device)
                                       )

    def forward(self, nodes : Tensor, edges : Tensor):
        nodes = self.mlp_in_nodes(nodes) # batch_size x num_nodes x mlp_node_hidden_dim
        edges = self.mlp_in_edges(edges) # batch_size x num_nodes x num_nodes x mlp_edge_hidden_dim

        for idx, layer in enumerate(self.tf_layers):
            nodes, edges = layer(nodes, edges) if idx % 6 == 0 else checkpoint(layer, nodes, edges, use_reentrant = False)
            # nodes, edges = layer(nodes, edges) # batch_size x num_nodes x tf_node_hidden_dim ; batch_size x num_nodes x num_nodes x tf_edge_hidden_dim
        
        # nodes = nodes.flatten(start_dim = 1) # batch_size x (num_nodes * node_dim)
        # edges = edges.flatten(start_dim = 1) # batch_size x (num_nodes * num_nodes * edge_dim)
        # graph_embs = torch.cat((nodes, edges), 1) # batch_size x (num_nodes * node_dim + num_nodes * num_nodes * edge_dim)
        
        # Soft attentional aggregation
        hstack = nodes.unsqueeze(3).expand(-1, -1, -1, self.num_nodes).permute(0, 1, 3, 2) # batch_size x num_nodes x num_nodes x hidden_dim
        vstack = hstack.permute(0, 2, 1, 3) # batch_size x num_nodes x num_nodes x hidden_dim
        graph_features = torch.cat(tensors = (hstack, vstack, edges), dim = 3) # batch_size x num_nodes x num_nodes x (3 * hidden_dim)
        del nodes
        del edges
        del hstack
        del vstack

        haggr_weights = self.mlp_haggr_weights(graph_features) # batch_size x num_nodes x num_nodes x 1
        graph_features = self.mlp_haggr_values(graph_features) # batch_size x num_nodes x num_nodes x hidden_dim

        graph_embs = (haggr_weights.permute(0, 1, 3, 2) @ graph_features).squeeze().flatten(start_dim = 1) # batch_size x (num_nodes * hidden_dim)
        del haggr_weights
        del graph_features

        means = self.mlp_mean(graph_embs)     # batch_size x graph_emb_dim
        logvars = self.mlp_logvar(graph_embs) # batch_size x graph_emb_dim
        del graph_embs

        return means, logvars     

class TransformerDecoder(nn.Module):
    def __init__(self, node_dim : int, edge_dim : int, graph_emb_dim : int, num_tf_layers : int, num_heads : int, device : torch.device):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.graph_emb_dim = graph_emb_dim
        self.num_nodes = MAX_NUM_PRIMITIVES # Number of nodes in each graph
        self.num_edges = self.num_nodes * self.num_nodes # Number of edges in each graph

        self.hidden_dim = 192
        self.num_tf_layers = num_tf_layers
        self.num_heads = num_heads

        self.mlp_create_nodes = nn.Sequential(nn.Linear(in_features = self.graph_emb_dim, out_features = self.num_nodes * self.node_dim, device = device),
                                              nn.ReLU(),
                                            #   nn.Dropout(p = 0.1),
                                              nn.Linear(in_features = self.num_nodes * self.node_dim, out_features = self.num_nodes * self.node_dim, device = device),
                                              nn.ReLU(),
                                            #   nn.Dropout(p = 0.1)
                                             )
        self.mlp_create_edges = nn.Sequential(nn.Linear(in_features = self.graph_emb_dim, out_features = self.graph_emb_dim, device = device),
                                              nn.ReLU(),
                                            #   nn.Dropout(p = 0.1),
                                              nn.Linear(in_features = self.graph_emb_dim, out_features = self.num_edges * self.edge_dim, device = device),
                                              nn.ReLU(),
                                            #   nn.Dropout(p = 0.1)
                                             )
        
        self.lin_node_transform = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.hidden_dim, device = device),
                                                nn.ReLU(),
                                                # nn.Dropout(p = 0.1),
                                                nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
                                                nn.ReLU(),
                                                # nn.Dropout(p = 0.1)
                                               )
        self.lin_edge_transform = nn.Sequential(nn.Linear(in_features = self.edge_dim, out_features = self.hidden_dim, device = device),
                                                nn.ReLU(),
                                                # nn.Dropout(p = 0.1),
                                                nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
                                                nn.ReLU(),
                                                # nn.Dropout(p = 0.1)
                                               )
        
        self.tf_layers = nn.ModuleList([TransformerLayer(num_heads = self.num_heads, 
                                                         node_dim = self.hidden_dim,
                                                         edge_dim = self.hidden_dim,
                                                         device = device
                                                        ) 
                                        for _ in range(self.num_tf_layers)])

        self.mlp_out_nodes = nn.Sequential(nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
                                           nn.ReLU(),
                                        #    nn.Dropout(p = 0.1),
                                           nn.Linear(in_features = self.hidden_dim, out_features = node_dim, device = device)
                                          )
        self.mlp_out_edges = nn.Sequential(nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
                                           nn.ReLU(),
                                        #    nn.Dropout(p = 0.1),
                                           nn.Linear(in_features = self.hidden_dim, out_features = self.edge_dim, device = device)
                                          )

    def forward(self, latents):
        nodes = torch.reshape(input = self.mlp_create_nodes(latents), shape = (-1, self.num_nodes, self.node_dim))            # batch_size x num_nodes x mlp_node_out_dim
        edges = torch.reshape(input = self.mlp_create_edges(latents), shape = (-1, self.num_nodes, self.num_nodes, self.edge_dim)) # batch_size x num_nodes x num_nodes x mlp_edge_out_dim

        nodes = self.lin_node_transform(nodes) # batch_size x num_nodes x hidden_dim
        edges = self.lin_edge_transform(edges) # batch_size x num_nodes x num_nodes x hidden_dim

        for idx, layer in enumerate(self.tf_layers):
            # nodes, edges = checkpoint(layer, nodes, edges, use_reentrant = False)
            nodes, edges = layer(nodes, edges) if idx % 8 == 0 else checkpoint(layer, nodes, edges, use_reentrant = False)
            # nodes, edges = layer(nodes, edges) # batch_size x num_nodes x tf_node_hidden_dim ; # batch_size x num_nodes x num_nodes x tf_edge_hidden_dim
        
        nodes = self.mlp_out_nodes(nodes) # batch_size x num_nodes x node_dim
        edges = self.mlp_out_edges(edges) # batch_size x num_nodes x num_nodes x edge_dim

        # # sigmoid and softmax for nodes
        # nodes[:,:,0] = F.sigmoid(nodes[:,:,0])              # Sigmoid for isConstructible
        # nodes[:,:,1:6] = F.softmax(nodes[:,:,1:6], dim = 2) # Softmax for primitive classes (i.e. line, circle, arc, point, none)
        
        # # softmax for constraints; Conceptual map => n1 (out) -> n2 (in) i.e. out_node, edge, in_node
        # edges[:,:,:,0:4] = F.softmax(edges[:,:,:,0:4], dim = 3) # Softmax for out_node subnode type
        # edges[:,:,:,4:8] = F.softmax(edges[:,:,:,4:8], dim = 3) # Softmax for in_node subnode type
        # edges[:,:,:,8: ] = F.softmax(edges[:,:,:,8: ], dim = 3) # Softmax for edge (aka constraint) type (i.e horizontal, vertical, etc...)

        return nodes, edges

class ParameterDecoder(nn.Module):
    def __init__(self, parameter_node_dim : int, categorical_node_dim : int, edge_dim : int, graph_emb_dim : int, num_layers : int, device : torch.device):
      super().__init__()
      self.device = device
      self.parameter_node_dim = parameter_node_dim
      self.categorical_node_dim = categorical_node_dim
      self.edge_dim = edge_dim
      self.graph_emb_dim = graph_emb_dim
      self.num_nodes = MAX_NUM_PRIMITIVES # Number of nodes in each graph
      self.num_edges = self.num_nodes * self.num_nodes # Number of edges in each graph

      self.hidden_dim = 192
      self.num_layers = num_layers

      self.mlp_create_params = nn.Sequential(nn.Linear(in_features = self.graph_emb_dim, out_features = self.num_nodes * self.hidden_dim, device = device),
                                              nn.ReLU(),
                                            #   nn.Dropout(p = 0.1),
                                              nn.Linear(in_features = self.num_nodes * self.hidden_dim, out_features = self.num_nodes * self.hidden_dim, device = device),
                                              nn.ReLU(),
                                            #   nn.Dropout(p = 0.1)
                                           )
      
      self.lin_embed_nodes = nn.Sequential(nn.Linear(in_features = self.categorical_node_dim, out_features = self.hidden_dim, device = self.device),
                                           nn.ReLU(),
                                           )
      
      self.lin_embed_edges = nn.Sequential(nn.Linear(in_features = self.edge_dim, out_features = self.hidden_dim, device = self.device),
                                           nn.ReLU(),
                                           )
      
      self.decoder_layers = nn.ModuleList([DecoderBlock(hidden_dim = self.hidden_dim, device = self.device) for _ in range(self.num_layers)])

      self.lin_params = nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = self.device)
      self.film_mul = SoftAttentionalLayer(node_dim = self.hidden_dim, edge_dim = self.hidden_dim, out_dim = self.hidden_dim, device = self.device)
      self.film_add = SoftAttentionalLayer(node_dim = self.hidden_dim, edge_dim = self.hidden_dim, out_dim = self.hidden_dim, device = self.device)

      self.lin_out = nn.Sequential(nn.ReLU(),
                                   nn.Linear(in_features = self.hidden_dim, out_features = self.parameter_node_dim, device = self.device))
      
    def forward(self, graph_embs, node_types, edge_types):
      node_params = torch.reshape(input = self.mlp_create_params(graph_embs), shape = (-1, self.num_nodes, self.hidden_dim))

      # sigmoid and softmax for nodes
      node_types[:,:,0] = F.sigmoid(node_types[:,:,0])              # Sigmoid for isConstructible
      node_types[:,:,1:6] = F.softmax(node_types[:,:,1:6], dim = 2) # Softmax for primitive classes (i.e. line, circle, arc, point, none)
      # softmax for constraints; Conceptual map => n1 (out) -> n2 (in) i.e. out_node, edge, in_node
      edge_types[:,:,:,0:4] = F.softmax(edge_types[:,:,:,0:4], dim = 3) # Softmax for out_node subnode type
      edge_types[:,:,:,4:8] = F.softmax(edge_types[:,:,:,4:8], dim = 3) # Softmax for in_node subnode type
      edge_types[:,:,:,8: ] = F.softmax(edge_types[:,:,:,8: ], dim = 3) # Softmax for edge (aka constraint) type (i.e horizontal, vertical, etc...)
      
      node_info = self.lin_embed_nodes(node_types)
      edge_info = self.lin_embed_edges(edge_types)

      for idx, block in enumerate(self.decoder_layers):
        node_params, node_info, edge_info = block(node_params, node_info, edge_info) if idx % 8 == 0 else checkpoint(block, node_params, node_info, edge_info, use_reentrant = False)
        # node_params, node_info, edge_info = block(node_params, node_info, edge_info)

      node_params = self.lin_params(node_params)
      node_params = node_params * self.film_mul(node_info, edge_info) + node_params + self.film_add(node_info, edge_info)
      node_params = self.lin_out(node_params)
      
      return node_params
          
class DecoderBlock(nn.Module):
    def __init__(self, hidden_dim : int, device : torch.device):
      super().__init__()
      self.device = device
      self.hidden_dim = hidden_dim

      self.lin_param_values = nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = self.device)

      self.film_mul = SoftAttentionalLayer(node_dim = self.hidden_dim, edge_dim = self.hidden_dim, out_dim = self.hidden_dim, device = self.device)
      self.film_add = SoftAttentionalLayer(node_dim = self.hidden_dim, edge_dim = self.hidden_dim, out_dim = self.hidden_dim, device = self.device)

      self.lin_node = nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = self.device)    
      self.lin_edge = nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = self.device)
      
      self.params_layer_norm = nn.LayerNorm(normalized_shape = self.hidden_dim, device = self.device)
      self.node_layer_norm = nn.LayerNorm(normalized_shape = self.hidden_dim, device = self.device)
      self.edge_layer_norm = nn.LayerNorm(normalized_shape = self.hidden_dim, device = self.device)

    def forward(self, params, node_info, edge_info):
        new_params = self.lin_param_values(params)
        new_params = new_params * self.film_mul(node_info, edge_info) + new_params + self.film_add(node_info, edge_info)

        new_node_info = self.lin_node(node_info)
        new_edge_info = self.lin_edge(edge_info)

        params = self.params_layer_norm(new_params + params)
        node_info = self.node_layer_norm(new_node_info + node_info)
        edge_info = self.edge_layer_norm(new_edge_info + edge_info)
        
        return params, node_info, edge_info
    
class SoftAttentionalLayer(nn.Module):
    def __init__(self, node_dim : int, edge_dim : int, out_dim : int, device : torch.device):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_nodes = MAX_NUM_PRIMITIVES # Number of nodes in each graph
        self.num_edges = self.num_nodes * self.num_nodes # Number of edges in each graph
        self.out_dim = out_dim

        self.mlp_haggr_weights = nn.Sequential(nn.Linear(in_features = 2 * self.node_dim + self.edge_dim, out_features = 1, device = device),
                                               nn.Softmax(dim = 2),
                                              )
        self.mlp_haggr_values = nn.Linear(in_features = 2 * self.node_dim + self.edge_dim, out_features = self.out_dim, device = device)
    
    def forward(self, nodes : Tensor, edges : Tensor) -> Tensor:
        hstack = nodes.unsqueeze(3).expand(-1, -1, -1, self.num_nodes).permute(0, 1, 3, 2) # batch_size x num_nodes x num_nodes x hidden_dim
        vstack = hstack.permute(0, 2, 1, 3) # batch_size x num_nodes x num_nodes x hidden_dim
        graph_features = torch.cat(tensors = (hstack, vstack, edges), dim = 3) # batch_size x num_nodes x num_nodes x (3 * hidden_dim)
        del nodes
        del edges
        del hstack
        del vstack

        haggr_weights = self.mlp_haggr_weights(graph_features) # batch_size x num_nodes x num_nodes x 1
        graph_features = self.mlp_haggr_values(graph_features) # batch_size x num_nodes x num_nodes x out_dim

        graph_embs = (haggr_weights.permute(0, 1, 3, 2) @ graph_features).squeeze() # batch_size x num_nodes x out_dim
        return graph_embs

# Graph Transformer Layer outlined by DiGress Graph Diffusion
class TransformerLayer(nn.Module):
    def __init__(self, num_heads : int, node_dim : int, edge_dim : int, device : torch.device):
        super().__init__()
        self.num_heads = num_heads
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.attention_heads = MultiHeadAttention(node_dim = self.node_dim, edge_dim = self.edge_dim, num_heads = self.num_heads, device = device)

        self.layer_norm_nodes = nn.LayerNorm(normalized_shape = self.node_dim, device = device)
        self.layer_norm_edges = nn.LayerNorm(normalized_shape = self.edge_dim, device = device)

        self.mlp_nodes = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device),
                                       nn.ReLU(),
                                    #    nn.Dropout(p = 0.1),
                                       nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device),
                                      )
        
        self.mlp_edges = nn.Sequential(nn.Linear(in_features = self.edge_dim, out_features = self.edge_dim, device = device),
                                       nn.ReLU(),
                                    #    nn.Dropout(p = 0.1),
                                       nn.Linear(in_features = self.edge_dim, out_features = self.edge_dim, device = device),
                                      )
        
        self.layer_norm_nodes2 = nn.LayerNorm(normalized_shape = self.node_dim, device = device)
        self.layer_norm_edges2 = nn.LayerNorm(normalized_shape = self.edge_dim, device = device)
    
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
        self.lin_value = nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device)

        self.lin_mul = nn.Linear(in_features = self.edge_dim, out_features = self.node_dim, device = device)
        self.lin_add = nn.Linear(in_features = self.edge_dim, out_features = self.node_dim, device = device)
        #self.edge_film = FiLM(self.edge_dim, self.node_dim, device = device)

        self.lin_nodes_out = nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device)
        self.lin_edges_out = nn.Sequential(nn.ReLU(),
                                           nn.Linear(in_features = self.node_dim, out_features = self.edge_dim, device = device)
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