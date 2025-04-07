# %%
import torch
import math
from typing import Dict, List, Tuple, Any
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from config import NODE_FEATURE_DIMENSION, EDGE_FEATURE_DIMENSION, GRAPH_EMBEDDING_SIZE #, MAX_NUM_PRIMITIVES
MAX_NUM_PRIMITIVES = 24
# %%
class GVAE(nn.Module):
    def __init__(self, device : torch.device):
        super().__init__()
        self.device = device
        self.node_dim = NODE_FEATURE_DIMENSION
        self.edge_dim = EDGE_FEATURE_DIMENSION
        self.graph_emb_dim = GRAPH_EMBEDDING_SIZE

        self.encoder = TransformerEncoder(node_dim = self.node_dim, 
                                          edge_dim = self.edge_dim, 
                                          graph_emb_dim = self.graph_emb_dim, 
                                          num_tf_layers = 16,
                                          num_heads = 16,
                                          device = self.device
                                         )
        
        self.decoder = TransformerDecoder(node_dim = self.node_dim, 
                                          edge_dim = self.edge_dim, 
                                          graph_emb_dim = self.graph_emb_dim,
                                          num_tf_layers = 16,
                                          num_heads = 16, 
                                          device = self.device
                                         )
    
    def forward(self, nodes : Tensor, edges : Tensor):
        means, logvars = self.encoder(nodes, edges)
        pred_nodes, pred_edges = self.decoder( self.sample_latent(means, torch.exp(0.5 * logvars)) )

        return pred_nodes, pred_edges, means, logvars
    
    def sample_latent(self, mean : Tensor, standard_deviation : Tensor):
        return mean + standard_deviation * torch.randn(size=mean.size(), device = self.device)
    
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
        
        self.hidden_dim = 256
        self.num_tf_layers = num_tf_layers
        self.num_heads = num_heads

        self.mlp_in_nodes = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.hidden_dim, device = device),
                                          nn.LeakyReLU(.1),
                                        #   nn.Dropout(p = 0.1),
                                          nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
                                          nn.LeakyReLU(.1),
                                        #   nn.Dropout(p = 0.1)
                                         )

        self.mlp_in_edges = nn.Sequential(nn.Linear(in_features = self.edge_dim, out_features = self.hidden_dim, device = device),
                                          nn.LeakyReLU(.1),
                                        #   nn.Dropout(p = 0.1),
                                          nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
                                          nn.LeakyReLU(.1),
                                        #   nn.Dropout(p = 0.1)
                                         )
        
        self.tf_layers = nn.ModuleList([TransformerLayer(num_heads = self.num_heads, 
                                                         node_dim = self.hidden_dim,
                                                         edge_dim = self.hidden_dim, 
                                                         device = device
                                                        ) 
                                        for _ in range(self.num_tf_layers)])

        self.mlp_haggr_weights = nn.Sequential(nn.Linear(in_features = 3 * self.hidden_dim, out_features = 3 * self.hidden_dim, device = device),
                                              nn.LeakyReLU(.1),
                                            #   nn.Dropout(p = 0.1),
                                              nn.Linear(in_features = 3 * self.hidden_dim, out_features = 1, device = device),
                                              nn.Softmax(dim = 2),
                                            #   nn.Dropout(p = 0.1)
                                             )
        self.mlp_haggr_values = nn.Sequential(nn.Linear(in_features = 3 * self.hidden_dim, out_features = 3 * self.hidden_dim, device = device),
                                              nn.LeakyReLU(.1),
                                            #   nn.Dropout(p = 0.1),
                                              nn.Linear(in_features = 3 * self.hidden_dim, out_features = self.hidden_dim, device = device)
                                             )
        
        self.mlp_mean = nn.Sequential(nn.Linear(in_features = self.num_nodes * self.hidden_dim, out_features = self.graph_emb_dim, device = device),
                                      nn.LeakyReLU(.1),
                                    #   nn.Dropout(p = 0.1),
                                      nn.Linear(in_features = self.graph_emb_dim, out_features = self.graph_emb_dim, device = device)
                                     )
        self.mlp_logvar = nn.Sequential(nn.Linear(in_features = self.num_nodes * self.hidden_dim, out_features = self.graph_emb_dim, device = device),
                                        nn.LeakyReLU(.1),
                                        # nn.Dropout(p = 0.1),
                                        nn.Linear(in_features = self.graph_emb_dim, out_features = self.graph_emb_dim, device = device)
                                       )

    def forward(self, nodes : Tensor, edges : Tensor):
        nodes = self.mlp_in_nodes(nodes) # batch_size x num_nodes x mlp_node_hidden_dim
        edges = self.mlp_in_edges(edges) # batch_size x num_nodes x num_nodes x mlp_edge_hidden_dim

        for idx, layer in enumerate(self.tf_layers):
            nodes, edges = layer(nodes, edges) if idx % 8 == -1 else checkpoint(layer, nodes, edges, use_reentrant = False)
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

        graph_embs = (haggr_weights.permute(0, 1, 3, 2) @ graph_features).squeeze(2).flatten(start_dim = 1) # batch_size x (num_nodes * hidden_dim)
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

        self.temp_node_dim = 128
        self.hidden_dim = 256
        self.num_tf_layers = num_tf_layers
        self.num_heads = num_heads

        self.mlp_create_nodes = nn.Sequential(nn.Linear(in_features = self.graph_emb_dim, out_features = self.num_nodes * self.temp_node_dim, device = device),
                                              nn.LeakyReLU(.1),
                                            #   nn.Dropout(p = 0.1),
                                              nn.Linear(in_features = self.num_nodes * self.temp_node_dim, out_features = self.num_nodes * self.temp_node_dim, device = device),
                                              nn.LeakyReLU(.1),
                                            #   nn.Dropout(p = 0.1)
                                             )
        self.mlp_create_edges = nn.Sequential(nn.Linear(in_features = self.graph_emb_dim, out_features = self.num_edges * self.edge_dim, device = device),
                                              nn.LeakyReLU(.1),
                                            #   nn.Dropout(p = 0.1),
                                              nn.Linear(in_features = self.num_edges * self.edge_dim, out_features = self.num_edges * self.edge_dim, device = device),
                                              nn.LeakyReLU(.1),
                                            #   nn.Dropout(p = 0.1)
                                             )
        
        self.lin_node_transform = nn.Sequential(nn.Linear(in_features = self.temp_node_dim, out_features = self.hidden_dim, device = device),
                                                nn.LeakyReLU(.1),
                                                # nn.Dropout(p = 0.1),
                                                nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
                                                nn.LeakyReLU(.1),
                                                # nn.Dropout(p = 0.1)
                                               )
        self.lin_edge_transform = nn.Sequential(nn.Linear(in_features = self.edge_dim, out_features = self.hidden_dim, device = device),
                                                nn.LeakyReLU(.1),
                                                # nn.Dropout(p = 0.1),
                                                nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
                                                nn.LeakyReLU(.1),
                                                # nn.Dropout(p = 0.1)
                                               )
        
        self.tf_layers = nn.ModuleList([TransformerLayer(num_heads = self.num_heads, 
                                                         node_dim = self.hidden_dim,
                                                         edge_dim = self.hidden_dim,
                                                         device = device
                                                        ) 
                                        for _ in range(self.num_tf_layers)])

        self.mlp_out_nodes = nn.Sequential(nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
                                           nn.LeakyReLU(.1),
                                        #    nn.Dropout(p = 0.1),
                                          #  nn.Linear(in_features = self.hidden_dim, out_features = node_dim, device = device),
                                          #  nn.LeakyReLU(.1),
                                          #  nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device),
                                          #  nn.LeakyReLU(.1),
                                          #  nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device),
                                          #  nn.LeakyReLU(.1),
                                           nn.Linear(in_features = self.hidden_dim, out_features = self.node_dim, device = device)
                                          )
        self.mlp_out_edges = nn.Sequential(nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
                                           nn.LeakyReLU(.1),
                                        #    nn.Dropout(p = 0.1),
                                           nn.Linear(in_features = self.hidden_dim, out_features = self.edge_dim, device = device)
                                          )

    def forward(self, latents):
        nodes = torch.reshape(input = self.mlp_create_nodes(latents), shape = (-1, self.num_nodes, self.temp_node_dim))            # batch_size x num_nodes x mlp_node_out_dim
        edges = torch.reshape(input = self.mlp_create_edges(latents), shape = (-1, self.num_nodes, self.num_nodes, self.edge_dim)) # batch_size x num_nodes x num_nodes x mlp_edge_out_dim

        nodes = self.lin_node_transform(nodes) # batch_size x num_nodes x hidden_dim
        edges = self.lin_edge_transform(edges) # batch_size x num_nodes x num_nodes x hidden_dim

        for idx, layer in enumerate(self.tf_layers):
            # nodes, edges = checkpoint(layer, nodes, edges, use_reentrant = False)
            nodes, edges = layer(nodes, edges) if idx % 10 == -1 else checkpoint(layer, nodes, edges, use_reentrant = False)
            # nodes, edges = layer(nodes, edges) # batch_size x num_nodes x tf_node_hidden_dim ; # batch_size x num_nodes x num_nodes x tf_edge_hidden_dim
        
        nodes = self.mlp_out_nodes(nodes) # batch_size x num_nodes x node_dim
        edges = self.mlp_out_edges(edges) # batch_size x num_nodes x num_nodes x edge_dim

        # sigmoid and softmax for nodes
        nodes[:,:,0] = F.sigmoid(nodes[:,:,0])              # Sigmoid for isConstructible
        nodes[:,:,1:6] = F.softmax(nodes[:,:,1:6], dim = 2) # Softmax for primitive classes (i.e. line, circle, arc, point, none)
        
        # softmax for constraints; Conceptual map => n1 (out) -> n2 (in) i.e. out_node, edge, in_node
        edges[:,:,:,0:4] = F.softmax(edges[:,:,:,0:4], dim = 3) # Softmax for out_node subnode type
        edges[:,:,:,4:8] = F.softmax(edges[:,:,:,4:8], dim = 3) # Softmax for in_node subnode type
        edges[:,:,:,8: ] = F.softmax(edges[:,:,:,8: ], dim = 3) # Softmax for edge (aka constraint) type (i.e horizontal, vertical, etc...)

        return nodes, edges

# Graph Transformer Layer outlined by DiGress Graph Diffusion
class TransformerLayer(nn.Module):
    def __init__(self, num_heads : int, node_dim : int, edge_dim : int, device : torch.device):
        super().__init__()
        self.num_heads = num_heads
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.attention_heads = MultiHeadAttention(node_dim = self.node_dim, edge_dim = self.edge_dim, num_heads = self.num_heads, device = device)

        self.layer_norm_nodes = nn.Sequential(nn.LayerNorm(normalized_shape = self.node_dim, device = device),
                                              nn.LeakyReLU(.1)
                                             )
        self.layer_norm_edges = nn.Sequential(nn.LayerNorm(normalized_shape = self.edge_dim, device = device),
                                              # nn.Dropout(p = 0.1)
                                             )

        self.mlp_nodes = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device),
                                       nn.LeakyReLU(.1),
                                    #    nn.Dropout(p = 0.1),
                                       nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device),
                                      )
        
        self.mlp_edges = nn.Sequential(nn.Linear(in_features = self.edge_dim, out_features = self.edge_dim, device = device),
                                       nn.LeakyReLU(.1),
                                    #    nn.Dropout(p = 0.1),
                                       nn.Linear(in_features = self.edge_dim, out_features = self.edge_dim, device = device),
                                      )
        
        self.layer_norm_nodes2 = nn.Sequential(nn.LayerNorm(normalized_shape = self.node_dim, device = device),
                                               nn.LeakyReLU(.1)
                                              )
        self.layer_norm_edges2 = nn.Sequential(nn.LayerNorm(normalized_shape = self.edge_dim, device = device),
                                              #  nn.Dropout(p = 0.1)
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
                                      #  nn.LeakyReLU(.1),
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
                                          #  nn.LeakyReLU(.1),
                                          #  nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device)
                                          )
        self.lin_edges_out = nn.Sequential(nn.LeakyReLU(.1),
                                           nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device),
                                          #  nn.LeakyReLU(.1),
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



# # %%
# import torch
# import math
# from typing import Dict, List, Tuple, Any
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor
# from config import NODE_FEATURE_DIMENSION, EDGE_FEATURE_DIMENSION, MAX_NUM_PRIMITIVES

# # %%
# class GVAE(nn.Module):
#     def __init__(self, device : torch.device):
#         super().__init__()
#         self.device = device
#         self.node_dim = NODE_FEATURE_DIMENSION
#         self.edge_dim = EDGE_FEATURE_DIMENSION
#         self.graph_emb_dim = 1024

#         self.encoder = TransformerEncoder(node_dim = self.node_dim, 
#                                           edge_dim = self.edge_dim, 
#                                           graph_emb_dim = self.graph_emb_dim, 
#                                           num_tf_layers = 24, 
#                                           num_heads = 12,
#                                           device = self.device
#                                          )
        
#         self.decoder = TransformerDecoder(node_dim = self.node_dim, 
#                                           edge_dim = self.edge_dim, 
#                                           graph_emb_dim = self.graph_emb_dim,
#                                           num_tf_layers = 24,
#                                           num_heads = 12, 
#                                           device = self.device
#                                          )
    
#     def forward(self, nodes : Tensor, edges : Tensor):
#         means, logvars = self.encoder(nodes, edges)
#         latents = self.sample_latent(means, torch.exp(0.5 * logvars))
#         pred_nodes, pred_edges = self.decoder(latents)

#         return pred_nodes, pred_edges, means, logvars
    
#     def sample_latent(self, mean : Tensor, standard_deviation : Tensor):
#         epsilon = torch.randn(size=mean.size(), device = self.device)
#         latents = mean + standard_deviation * epsilon
#         return latents
    
#     @torch.no_grad()
#     def sample_graph(self) -> Tuple[Tensor, Tensor]:
#         latent = torch.randn(size = (self.graph_emb_dim,), device = self.device)
#         nodes, edges = self.decoder(latent)

#         nodes[:,:,1:6] = torch.exp(nodes[:,:,1:6], dim = 2) # Softmax for primitive classes
        
#         # softmax for constraints; Conceptual map => n1 (out) -> n2 (in) i.e. out_node, edge, in_node
#         edges[:,:,:,0:4] = torch.exp(edges[:,:,:,0:4], dim = 3) # Softmax for out_node subnode type
#         edges[:,:,:,4:8] = torch.exp(edges[:,:,:,4:8], dim = 3) # Softmax for in_node subnode type
#         edges[:,:,:,8: ] = torch.exp(edges[:,:,:,8: ], dim = 3) # Softmax for edge type
#         return torch.squeeze(nodes), torch.squeeze(edges)

# class TransformerEncoder(nn.Module):
#     def __init__(self, node_dim : int, edge_dim : int, graph_emb_dim : int, num_tf_layers: int, num_heads : int, device : torch.device): # perm_emb_dim: int,
#         super().__init__()
#         self.node_dim = node_dim # Number of features per node
#         self.edge_dim = edge_dim # Number of features per edge
#         self.graph_emb_dim = graph_emb_dim # Size of graph embedding vector
#         self.num_nodes = MAX_NUM_PRIMITIVES # Number of nodes in each graph
#         self.num_edges = self.num_nodes * self.num_nodes # Number of edges in each graph
        
#         self.hidden_dim = 128
#         self.num_tf_layers = num_tf_layers
#         self.num_heads = num_heads

#         self.mlp_in_nodes = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.hidden_dim, device = device),
#                                           nn.LeakyReLU(.1),
#                                           nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
#                                           nn.LeakyReLU(.1)
#                                          )

#         self.mlp_in_edges = nn.Sequential(nn.Linear(in_features = self.edge_dim, out_features = self.hidden_dim, device = device),
#                                           nn.LeakyReLU(.1),
#                                           nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim, device = device),
#                                           nn.LeakyReLU(.1)
#                                          )
        
#         self.tf_layers = nn.ModuleList([TransformerLayer(num_heads = self.num_heads, 
#                                                          node_dim = self.hidden_dim,
#                                                          edge_dim = self.hidden_dim, 
#                                                          device = device
#                                                         ) 
#                                         for _ in range(self.num_tf_layers)])

#         self.mlp_mean = nn.Sequential(nn.Linear(in_features = self.num_nodes * self.hidden_dim + self.num_edges * self.hidden_dim, out_features = self.graph_emb_dim, device = device),
#                                       nn.LeakyReLU(.1),
#                                       nn.Linear(in_features = self.graph_emb_dim, out_features = self.graph_emb_dim, device = device),
#                                       nn.LeakyReLU(.1)
#                                      )
        
#         self.mlp_logvar = nn.Sequential(nn.Linear(in_features = self.num_nodes * self.hidden_dim + self.num_edges * self.hidden_dim, out_features = self.graph_emb_dim, device = device),
#                                         nn.LeakyReLU(.1),
#                                         nn.Linear(in_features = self.graph_emb_dim, out_features = self.graph_emb_dim, device = device),
#                                         nn.LeakyReLU(.1)
#                                        )

#     def forward(self, nodes : Tensor, edges : Tensor):
#         nodes = self.mlp_in_nodes(nodes) # batch_size x num_nodes x mlp_node_hidden_dim
#         edges = self.mlp_in_edges(edges) # batch_size x num_nodes x num_nodes x mlp_edge_hidden_dim

#         for layer in self.tf_layers:
#             nodes, edges = layer(nodes, edges) # batch_size x num_nodes x tf_node_hidden_dim ; batch_size x num_nodes x num_nodes x tf_edge_hidden_dim
        
#         nodes = nodes.flatten(start_dim = 1) # batch_size x (num_nodes * node_dim)
#         edges = edges.flatten(start_dim = 1) # batch_size x (num_nodes * num_nodes * edge_dim)
#         graph_embs = torch.cat((nodes, edges), 1) # batch_size x (num_nodes * node_dim + num_nodes * num_nodes * edge_dim)

#         means = self.mlp_mean(graph_embs)     # batch_size x graph_emb_dim
#         logvars = self.mlp_logvar(graph_embs) # batch_size x graph_emb_dim

#         return means, logvars     

# class TransformerDecoder(nn.Module):
#     def __init__(self, node_dim : int, edge_dim : int, graph_emb_dim : int, num_tf_layers : int, num_heads : int, device : torch.device):
#         super().__init__()
#         self.node_dim = node_dim
#         self.edge_dim = edge_dim
#         self.graph_emb_dim = graph_emb_dim
#         self.num_nodes = MAX_NUM_PRIMITIVES # Number of nodes in each graph
#         self.num_edges = self.num_nodes * self.num_nodes # Number of edges in each graph

#         self.hidden_dim = 128
#         self.num_tf_layers = num_tf_layers
#         self.num_heads = num_heads

#         self.mlp_create_nodes = nn.Sequential(nn.Linear(in_features = self.graph_emb_dim, out_features = self.num_nodes * self.hidden_dim, device = device),
#                                               nn.LeakyReLU(.1),
#                                               nn.Linear(in_features = self.num_nodes * self.hidden_dim, out_features = self.num_nodes * self.hidden_dim, device = device),
#                                               nn.LeakyReLU(.1)
#                                              )
        
#         self.mlp_create_edges = nn.Sequential(nn.Linear(in_features = self.graph_emb_dim, out_features = self.num_edges * self.hidden_dim, device = device),
#                                               nn.LeakyReLU(.1),
#                                               nn.Linear(in_features = self.num_edges * self.hidden_dim, out_features = self.num_edges * self.hidden_dim, device = device),
#                                               nn.LeakyReLU(.1)
#                                              )

#         self.tf_layers = nn.ModuleList([TransformerLayer(num_heads = self.num_heads, 
#                                                          node_dim = self.hidden_dim,
#                                                          edge_dim = self.hidden_dim,
#                                                          device = device
#                                                         ) 
#                                         for _ in range(self.num_tf_layers)])

#         self.mlp_out_nodes = nn.Sequential(nn.Linear(in_features = self.hidden_dim, out_features = self.node_dim, device = device),
#                                            nn.LeakyReLU(.1),
#                                            nn.Linear(in_features = self.node_dim, out_features = node_dim, device = device)
#                                           )

#         self.mlp_out_edges = nn.Sequential(nn.Linear(in_features = self.hidden_dim, out_features = self.edge_dim, device = device),
#                                            nn.LeakyReLU(.1),
#                                            nn.Linear(in_features = self.edge_dim, out_features = self.edge_dim, device = device)
#                                           )

#     def forward(self, latents):
#         nodes = torch.reshape(input = self.mlp_create_nodes(latents), shape = (-1, self.num_nodes, 32))                 # batch_size x num_nodes x mlp_node_out_dim
#         edges = torch.reshape(input = self.mlp_create_edges(latents), shape = (-1, self.num_nodes, self.num_nodes, 32)) # batch_size x num_nodes x num_nodes x mlp_edge_out_dim

#         for layer in self.tf_layers:
#             nodes, edges = layer(nodes, edges) # batch_size x num_nodes x tf_node_hidden_dim ; # batch_size x num_nodes x num_nodes x tf_edge_hidden_dim
        
#         nodes = self.mlp_out_nodes(nodes) # batch_size x num_nodes x node_dim
#         edges = self.mlp_out_edges(edges) # batch_size x num_nodes x num_nodes x edge_dim

#         # sigmoid and logsoftmax for nodes
#         nodes[:,:,0] = F.sigmoid(nodes[:,:,0])              # Sigmoid for isConstructible
#         nodes[:,:,1:6] = F.softmax(nodes[:,:,1:6], dim = 2) # Softmax for primitive classes (i.e. line, circle, arc, point, none)
        
#         # logsoftmax for constraints; Conceptual map => n1 (out) -> n2 (in) i.e. out_node, edge, in_node
#         edges[:,:,:,0:4] = F.softmax(edges[:,:,:,0:4], dim = 3) # Softmax for out_node subnode type
#         edges[:,:,:,4:8] = F.softmax(edges[:,:,:,4:8], dim = 3) # Softmax for in_node subnode type
#         edges[:,:,:,8: ] = F.softmax(edges[:,:,:,8: ], dim = 3) # Softmax for edge (aka constraint) type (i.e horizontal, vertical, etc...)

#         return nodes, edges

# # Graph Transformer Layer outlined by DiGress Graph Diffusion
# class TransformerLayer(nn.Module):
#     def __init__(self, num_heads : int, node_dim : int, edge_dim : int, device : torch.device):
#         super().__init__()
#         self.num_heads = num_heads
#         self.node_dim = node_dim
#         self.edge_dim = edge_dim

#         self.attention_heads = MultiHeadAttention(node_dim = self.node_dim, edge_dim = self.edge_dim, num_heads = self.num_heads, device = device)

#         self.layer_norm_nodes = nn.LayerNorm(normalized_shape = [MAX_NUM_PRIMITIVES, self.node_dim], device = device)
#         self.layer_norm_edges = nn.LayerNorm(normalized_shape = [MAX_NUM_PRIMITIVES, MAX_NUM_PRIMITIVES, self.edge_dim], device = device)

#         self.mlp_nodes = nn.Sequential(nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device),
#                                        nn.Dropout(p = 0.1),
#                                        nn.LeakyReLU(.1),
#                                        nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device),
#                                        nn.Dropout(p = 0.1),
#                                        nn.LeakyReLU(.1)
#                                       )
        
#         self.mlp_edges = nn.Sequential(nn.Linear(in_features = self.edge_dim, out_features = self.edge_dim, device = device),
#                                        nn.Dropout(p = 0.1),
#                                        nn.LeakyReLU(.1),
#                                        nn.Linear(in_features = self.edge_dim, out_features = self.edge_dim, device = device),
#                                        nn.Dropout(p = 0.1),
#                                        nn.LeakyReLU(.1)
#                                       )
        
#         self.layer_norm_nodes2 = nn.LayerNorm(normalized_shape = [MAX_NUM_PRIMITIVES, self.node_dim], device = device)
#         self.layer_norm_edges2 = nn.LayerNorm(normalized_shape = [MAX_NUM_PRIMITIVES, MAX_NUM_PRIMITIVES, self.edge_dim], device = device)
    
#     def forward(self, nodes : Tensor, edges : Tensor) -> Tuple[Tensor, Tensor]:
#         # Perform multi head attention
#         attn_nodes, attn_edges = self.attention_heads(nodes, edges) # batch_size x num_nodes x node_dim ; batch_size x num_nodes x num_nodes x edge_dim

#         # Layer normalization with a skip connection
#         attn_nodes = self.layer_norm_nodes(attn_nodes + nodes) # batch_size x num_nodes x node_dim
#         attn_edges = self.layer_norm_edges(attn_edges + edges) # batch_size x num_nodes x num_nodes x edge_dim

#         # MLP out
#         new_nodes = self.mlp_nodes(attn_nodes) # batch_size x num_nodes x node_dim
#         new_edges = self.mlp_edges(attn_edges) # batch_size x num_nodes x num_nodes x edge_dim

#         # Second layer normalization with a skip connection
#         new_nodes = self.layer_norm_nodes2(new_nodes + attn_nodes) # batch_size x num_nodes x node_dim
#         new_edges = self.layer_norm_edges2(new_edges + attn_edges) # batch_size x num_nodes x num_nodes x edge_dim

#         return new_nodes, new_edges

# # Outer Product Attention Head
# class MultiHeadAttention(nn.Module):
#     def __init__(self, node_dim : int, edge_dim : int, num_heads : int, device : torch.device):
#         super().__init__()
#         self.node_dim = node_dim
#         self.edge_dim = edge_dim
#         self.num_heads = num_heads

#         self.lin_query = nn.Linear(in_features = self.node_dim, out_features = self.num_heads * self.node_dim, device = device)
#         self.lin_key = nn.Linear(in_features = self.node_dim, out_features = self.num_heads * self.node_dim, device = device)
#         self.lin_value = nn.Linear(in_features = self.node_dim, out_features = self.num_heads * self.node_dim, device = device)

#         self.lin_mul = nn.Linear(in_features = self.edge_dim, out_features = self.num_heads * self.node_dim, device = device)
#         self.lin_add = nn.Linear(in_features = self.edge_dim, out_features = self.num_heads * self.node_dim, device = device)
#         #self.edge_film = FiLM(self.edge_dim, self.node_dim, device = device)

#         self.lin_nodes_out = nn.Linear(in_features = self.num_heads * self.node_dim, out_features = self.node_dim, device = device)
#         self.lin_edges_out = nn.Linear(in_features = self.num_heads * self.node_dim, out_features = self.edge_dim, device = device)

#     def forward(self, nodes : Tensor, edges : Tensor):
#         queries = self.lin_query(nodes) # batch_size x num_nodes x (num_heads * out_node_dim)
#         keys = self.lin_key(nodes)      # batch_size x num_nodes x (num_heads * out_node_dim)
#         values = self.lin_value(nodes)  # batch_size x num_nodes x (num_heads * out_node_dim)
#         edges_mul = self.lin_mul(edges) # batch_size x num_nodes x num_nodes x (num_heads * out_node_dim)
#         edges_add = self.lin_add(edges) # batch_size x num_nodes x num_nodes x (num_heads * out_node_dim)

#         # Reshape to accomodate independent attention heads
#         batch_size, num_nodes, _ = queries.size()
#         queries = queries.reshape(batch_size, num_nodes, self.num_heads, self.node_dim)\
#                          .permute(0, 2, 1, 3)\
#                          .reshape(batch_size * self.num_heads, num_nodes, self.node_dim)
#         keys = keys.reshape(batch_size, num_nodes, self.num_heads, self.node_dim)\
#                          .permute(0, 2, 1, 3)\
#                          .reshape(batch_size * self.num_heads, num_nodes, self.node_dim)
#         values = values.reshape(batch_size, num_nodes, self.num_heads, self.node_dim)\
#                          .permute(0, 2, 1, 3)\
#                          .reshape(batch_size * self.num_heads, num_nodes, self.node_dim)
#         edges_mul = edges_mul.reshape(batch_size, num_nodes, num_nodes, self.num_heads, self.node_dim)\
#                          .permute(0, 3, 1, 2, 4)\
#                          .reshape(batch_size * self.num_heads, num_nodes, num_nodes, self.node_dim)
#         edges_add = edges_add.reshape(batch_size, num_nodes, num_nodes, self.num_heads, self.node_dim)\
#                          .permute(0, 3, 1, 2, 4)\
#                          .reshape(batch_size * self.num_heads, num_nodes, num_nodes, self.node_dim)
        

#         # Outer Product Attention -------
#         queries = queries.unsqueeze(2)                            # (batch_size * num_heads) x num_nodes x 1 x out_node_dim (each item in batch is a column tensor where each element is a row? vector)
#         keys = keys.unsqueeze(1)                                  # (batch_size * num_heads) x 1 x num_nodes x out_node_dim (each item in batch is a row tensor where each element is a row? vector)
#         attention = queries * keys / math.sqrt(self.node_dim) # (batch_size * num_heads) x num_nodes x num_nodes x out_node_dim (each element in batch is a 2d tensor where each element is a row? vector)

#         # Condition attention based on edge features
#         new_edges = edges_mul * attention + edges_add + attention # (batch_size * num_heads) x num_nodes x num_nodes x out_node_dim (each element in batch is a 2d tensor, where each element in 2d tensor is a row? vector)
        
#         # Normalize attention
#         attention = new_edges.sum(dim = 3)                               # (batch_size * num_heads) x num_nodes x num_nodes (Finish dot product)
#         normalized_attention = torch.softmax(input = attention, dim = 2) # (batch_size * num_heads) x num_nodes x num_nodes (softmax row wise)

#         # Weight node representations and sum
#         weighted_values = normalized_attention @ values # (batch_size * num_heads) x num_nodes x node_dim
        
#         # Reshape to flatten attention heads
#         weighted_values = weighted_values.reshape(batch_size, self.num_heads, num_nodes, self.node_dim)\
#                                          .permute(0, 2, 1, 3)\
#                                          .reshape(batch_size, num_nodes, self.num_heads * self.node_dim)
#         new_edges = new_edges.reshape(batch_size, self.num_heads, num_nodes, num_nodes, self.node_dim)\
#                              .permute(0, 2, 3, 1, 4)\
#                              .reshape(batch_size, num_nodes, num_nodes, self.num_heads * self.node_dim)
        
#         # Combine attention heads
#         new_nodes = self.lin_nodes_out(weighted_values)
#         new_edges = self.lin_edges_out(new_edges)

#         return new_nodes, new_edges

# # Feature-Wise Linear Modulation Layer
# class FiLM(nn.Module):
#     def __init__(self, a_feature_dim : int, b_feature_dim : int, device : torch.device):
#         super().__init__()
        
#         self.lin1 = nn.Linear(in_features = a_feature_dim, out_features = b_feature_dim, device = device)
#         self.lin2 = nn.Linear(in_features = a_feature_dim, out_features = b_feature_dim, device = device)
    
#     def forward(self, a : Tensor, b : Tensor):
#         mul = self.lin1(a)
#         add = self.lin2(a)

#         # For vanilla FiLM you are only supposed to do mul * b + add
#         # I assume digress put the additional '+ b' as a skip connection
#         return mul * b + add + b

# # Principal Neighbourhood Aggregation Layer
# class PNA(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.mean_aggr = MeanAggregation()
#         # self.max_aggr = MaxAggregation()
#         # self.min_aggr = MinAggregation()
#         # self.std_aggr = StdAggregation()
    
#     def forward(self, input : Tensor, dim):
#         # find the biggest element in dim
#         mean = torch.mean(input, dim)
#         max = torch.amax(input, dim)
#         min = torch.amin(input, dim)
#         stdev = torch.std(input, dim) # Standard Deviation

#         return torch.cat(tensors = (mean, max, min, stdev), dim = 1)
    
