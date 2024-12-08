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
    self.node_dim = NODE_FEATURE_DIMENSION + 1
    self.node_hidden_dim = 1536
    self.cond_hidden_dim = 512
    self.num_tf_layers = 32
    self.num_checkpoints = 19
    self.num_heads = 16
    self.max_timestep = 1000
    self.noise_scheduler = CosineNoiseScheduler(self.max_timestep, self.device)
    self.architecture = DiffusionModel(node_dim = self.node_dim,
                                  node_hidden_dim = self.node_hidden_dim,
                                  cond_hidden_dim = self.cond_hidden_dim,
                                  num_heads = self.num_heads,
                                  num_tf_layers = self.num_tf_layers,
                                  num_checkpoints = self.num_checkpoints,
                                  max_timestep = self.max_timestep,
                                  device = self.device)
    # decay = 0.9999
    # self.ema_model = torch.optim.swa_utils.AveragedModel(self.architecture, device = device, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay))
    
    # self.fine_model = DiffusionModel(node_dim = self.node_dim, 
    #                               node_hidden_dim = self.node_hidden_dim,
    #                               cond_hidden_dim = self.cond_hidden_dim,
    #                               num_heads = self.num_heads,
    #                               num_tf_layers = self.num_tf_layers,
    #                               num_checkpoints = self.num_checkpoints,
    #                               max_timestep = self.max_timestep,
    #                               device = self.device)
    # self.coarse_model = DiffusionModel(node_dim = self.node_dim, 
    #                               node_hidden_dim = self.node_hidden_dim,
    #                               cond_hidden_dim = self.cond_hidden_dim,
    #                               num_heads = self.num_heads,
    #                               num_tf_layers = self.num_tf_layers,
    #                               num_checkpoints = self.num_checkpoints,
    #                               max_timestep = self.max_timestep,
    #                               device = self.device)
    # self.step_cutoff = self.max_timestep // 2
    # num_partitions = 5
    # self.partitions = [(1, 250), (251, 500), (501, 1000)] # [(i * self.max_timestep // num_partitions, (i + 1) * self.max_timestep // num_partitions) for i in range(num_partitions)]
    # self.architectures = nn.ModuleList([DiffusionModel(node_dim = self.node_dim, 
    #                                                    node_hidden_dim = self.node_hidden_dim,
    #                                                    cond_hidden_dim = self.cond_hidden_dim,
    #                                                    num_heads = self.num_heads,
    #                                                    num_tf_layers = self.num_tf_layers,
    #                                                    num_checkpoints = self.num_checkpoints,
    #                                                    max_timestep = self.max_timestep,
    #                                                    device = self.device) 
    #                                     for _ in range(len(self.partitions))])

  def forward(self, nodes : Tensor, timestep : Tensor):
    # Output Buffer
    # out_nodes = torch.zeros_like(nodes)

    # for idx, (min, max) in enumerate(self.partitions):
    #   indices = torch.nonzero((timestep >= min) & (timestep <= max)).squeeze(-1)
    #   if indices.nelement() != 0: 
    #     out_nodes[indices] = self.architectures[idx](nodes[indices], timestep[indices])

    # out_nodes[...,0:2] = out_nodes[...,0:2].softmax(dim = -1)
    # out_nodes[...,2:7] = out_nodes[...,2:7].softmax(dim = -1)
    # return out_nodes


    # # Split small perturbations from large perturbations
    # s_idx = torch.nonzero(timestep < self.step_cutoff).squeeze(-1)
    # l_idx = torch.nonzero(timestep >= self.step_cutoff).squeeze(-1)

    # # Fine model refines small perturbations
    # if s_idx.nelement() != 0: 
    #   out_nodes[s_idx] = self.fine_model(nodes[s_idx], timestep[s_idx])
    # # Coarse model refines large perturbations
    # if l_idx.nelement() != 0: 
    #   out_nodes[l_idx] = self.coarse_model(nodes[l_idx], timestep[l_idx])

    # out_nodes[...,0:2] = out_nodes[...,0:2].softmax(dim = -1)
    # out_nodes[...,2:7] = out_nodes[...,2:7].softmax(dim = -1)
    # return out_nodes

    # nodes = self.architecture(nodes, self.noise_scheduler.sqrt_b_bar[timestep])
    # Normalize to Probabilities
    # nodes[...,0:2] = nodes[...,0:2].softmax(dim = -1)
    # nodes[...,2:7] = nodes[...,2:7].softmax(dim = -1)

    # if self.training:
    #   nodes = self.architecture(nodes, timestep)
    #   nodes[...,0:2] = nodes[...,0:2].softmax(dim = -1)
    #   nodes[...,2:7] = nodes[...,2:7].softmax(dim = -1)
    #   return nodes
    # else:
    #   nodes = self.ema_model(nodes, timestep)
    #   nodes[...,0:2] = nodes[...,0:2].softmax(dim = -1)
    #   nodes[...,2:7] = nodes[...,2:7].softmax(dim = -1)
    #   return nodes

    nodes = self.architecture(nodes, timestep)
    nodes[...,0:2] = nodes[...,0:2].softmax(dim = -1)
    nodes[...,2:7] = nodes[...,2:7].softmax(dim = -1)
    return nodes
  
  # @torch.no_grad()
  # def update_ema(self):
  #   self.ema_model.update_parameters(self.architecture)

  @torch.no_grad()
  def sample(self, batch_size : int):
    # Sample Noise
    nodes = self.noise_scheduler.sample_latent(batch_size)
    nodes = nodes.to(self.device)
    return self.denoise(nodes)

  @torch.no_grad()
  def denoise(self, nodes, axes = None):
    num_images = 10
    j = num_images - 1
    if axes is None:
      fig, axes = plt.subplots(nrows = 2, ncols = num_images, figsize=(40, 8))
    stepsize = int(self.max_timestep/num_images)
    
    for t in reversed(range(1, self.max_timestep)):
      # model expects a timestep for each batch
      batch_size = nodes.size(0)
      time = torch.ones(size = (batch_size,), dtype = torch.int32, device = self.device) * t
      denoised_nodes = self.forward(nodes, time)
      nodes = self.reverse(denoised_nodes, nodes, t)

      if t % stepsize == 0:
        SketchDataset.render_graph(nodes[0,...,1:].cpu(), torch.zeros(size = (24, 24, 17)).cpu(), axes[0, j])
        SketchDataset.render_graph(denoised_nodes[0,...,1:].cpu(), torch.zeros(size = (24, 24, 17)).cpu(), axes[1, j])
        j = j - 1
    
    SketchDataset.render_graph(nodes[0,...,1:].cpu(), torch.zeros(size = (24, 24, 17)).cpu(), axes[0, 0])
    SketchDataset.render_graph(denoised_nodes[0,...,1:].cpu(), torch.zeros(size = (24, 24, 17)).cpu(), axes[1, 0])

    return nodes
  
  @torch.no_grad()
  def reverse(self, pred_nodes, curr_nodes, timestep):
    denoised_nodes = torch.zeros_like(pred_nodes)
    denoised_nodes[...,0:2] = self.noise_scheduler.discrete_posterior_step(pred_nodes[...,0:2], curr_nodes[...,0:2], timestep)
    denoised_nodes[...,2:7] = self.noise_scheduler.discrete_posterior_step(pred_nodes[...,2:7], curr_nodes[...,2:7], timestep)
    # denoised_nodes[...,7:] = self.noise_scheduler.continuous_posterior_step(pred_nodes[...,7:], curr_nodes[...,7:], timestep)
    # Weight the superposition of parameters
    # weights = F.one_hot(torch.argmax(pred_nodes[...,2:7], dim = -1), pred_nodes[...,2:7].size(-1)).float()
    # weights[weights == 0] += (1 - self.noise_scheduler.a_bar[timestep].sqrt())
    # weights = torch.cat(
    #   [weights[...,0,None].expand(-1, -1, 4), 
    #    weights[...,1,None].expand(-1, -1, 3), 
    #    weights[...,2,None].expand(-1, -1, 5),
    #    weights[...,3,None].expand(-1, -1, 2)], dim = -1)
    # denoised_nodes[...,7:] = self.noise_scheduler.continuous_posterior_step(weights * pred_nodes[...,7:], curr_nodes[...,7:], timestep)
    weights = torch.cat(
      [pred_nodes[...,2,None].expand(-1, -1, 4), 
       pred_nodes[...,3,None].expand(-1, -1, 3), 
       pred_nodes[...,4,None].expand(-1, -1, 5),
       pred_nodes[...,5,None].expand(-1, -1, 2)], dim = -1)
    # denoised_nodes[...,7:] = self.noise_scheduler.continuous_posterior_step(weights * pred_nodes[...,7:], curr_nodes[...,7:], timestep)
    vals, _ = torch.max(pred_nodes[...,2:7], dim = -1, keepdim = True)
    denoised_nodes[...,7:] = self.noise_scheduler.continuous_posterior_step(weights / vals * pred_nodes[...,7:], curr_nodes[...,7:], timestep)
    return denoised_nodes

class DiffusionModel(nn.Module):
    def __init__(self, node_dim, node_hidden_dim, cond_hidden_dim, num_heads, num_tf_layers, num_checkpoints, max_timestep, device: device):
        super().__init__()
        self.num_checkpoints = num_checkpoints

        self.time_embedder = SinuisodalEncoding(max_length = max_timestep, embedding_dimension = 3 * cond_hidden_dim // 4, device = device)
        # self.time_embedder = nn.Embedding(num_embeddings = max_timestep, embedding_dim = cond_hidden_dim, device = device)
        self.pos_embedder = SinuisodalEncoding(max_length = 24, embedding_dimension = cond_hidden_dim // 4, device = device)

        # Input MLP layers
        self.mlp_in_nodes = nn.Sequential(
            nn.Linear(in_features = node_dim, out_features = node_hidden_dim, device = device),
            # nn.SiLU(),
            # nn.Linear(in_features = node_hidden_dim, out_features = node_hidden_dim, device = device),
            # nn.SiLU(),
        )

        self.mlp_in_conds = nn.Sequential(
            nn.Linear(in_features = cond_hidden_dim, out_features = cond_hidden_dim, device = device),
            nn.SiLU(),
            # nn.Linear(in_features = cond_hidden_dim, out_features = cond_hidden_dim, device = device),
            # nn.SiLU(),
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
        # Output MLP layers
        self.lin_out_params = nn.Sequential(
            # nn.Linear(in_features = node_hidden_dim, out_features = node_hidden_dim, device = device),
            # nn.SiLU(),
            nn.Linear(in_features = node_hidden_dim, out_features = 14, device = device)
        )
        self.lin_out_types = nn.Sequential(
            # nn.Linear(in_features = node_hidden_dim, out_features = node_hidden_dim, device = device),
            # nn.SiLU(),
            nn.Linear(in_features = node_hidden_dim, out_features = 5, device = device)
        )
        self.lin_out_bool = nn.Sequential(
            # nn.Linear(in_features = node_hidden_dim, out_features = node_hidden_dim, device = device),
            # nn.SiLU(),
            nn.Linear(in_features = node_hidden_dim, out_features = 1, device = device)
        )

    def forward(self, nodes : Tensor, timestep : Tensor):
        nodes = self.mlp_in_nodes(nodes)     # shape: (batch_size, num_nodes, node_hidden_dim)
        # nodes = nodes + self.pos_embedder.embs # [torch.randperm(24)]
        conds = torch.cat([self.pos_embedder.embs.unsqueeze(0).expand(nodes.size(0), -1, -1), self.time_embedder(timestep).unsqueeze(1).expand(-1, MAX_NUM_PRIMITIVES, -1)], dim = -1) # F.silu(self.time_embedder(timestep)) # shape: (batch_size, cond_hidden_dim)
        conds = self.mlp_in_conds(conds)     # shape: (batch_size, num_nodes, cond_hidden_dim)

        checkpoints = self.num_checkpoints
        for layer in self.block_layers:
            nodes = checkpoint(layer, nodes, conds, use_reentrant = False) if checkpoints > 1 else layer(nodes, conds) # shape: (batch_size, num_nodes, node_hidden_dim) ; shape: (batch_size, num_nodes, num_nodes, edge_hidden_dim)
            checkpoints = checkpoints - 1

        shift, scale = self.lin_cond(conds).chunk(chunks = 2, dim = -1)
        nodes = self.norm(nodes) * (1 + scale) + shift
        nodes = torch.cat([torch.zeros_like(nodes[...,[0]]), self.lin_out_bool(nodes), self.lin_out_types(nodes), self.lin_out_params(nodes)], dim = -1) # shape: (batch_size, num_nodes, node_dim)

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
            nn.Linear(in_features = node_dim, out_features = node_dim, device = device),
            nn.SiLU(),
            nn.Linear(in_features = node_dim, out_features = node_dim, device = device),
            # nn.SiLU()
        )
        # self.w1 = nn.Parameter(data = torch.randn(MAX_NUM_PRIMITIVES, node_dim, node_dim, device = device) / node_dim)
        # self.b1 = nn.Parameter(data = torch.randn(MAX_NUM_PRIMITIVES, 1, node_dim, device = device) / node_dim)
        # self.w2 = nn.Parameter(data = torch.randn(MAX_NUM_PRIMITIVES, node_dim, node_dim, device = device) / node_dim)
        # self.b2 = nn.Parameter(data = torch.randn(MAX_NUM_PRIMITIVES, 1, node_dim, device = device) / node_dim)

        # Conditioning
        self.lin_cond = nn.Linear(in_features = cond_dim, out_features = 6 * node_dim, device = device)
        nn.init.zeros_(self.lin_cond.weight)
        nn.init.zeros_(self.lin_cond.bias)

    def forward(self, nodes : Tensor, conds : Tensor) -> Tensor:
        mul_in, add_in, gate_in, mul_attn, add_attn, gate_attn = self.lin_cond(conds).chunk(chunks = 6, dim = -1)
        # Attention
        # nodes = self.norm_in(nodes)
        nodes = self.attention_heads(self.norm_in(nodes) * (1 + mul_in) + add_in) * gate_in + nodes
        # MLP
        # nodes = self.norm_attn(nodes)
        # nodes = F.leaky_relu(F.leaky_relu((nodes * mul_attn + add_attn).unsqueeze(dim = 2) @ self.w1 + self.b1, 0.1) @ self.w2 + self.b2, 0.1).squeeze(dim = 2) + nodes
        nodes = self.mlp_nodes(self.norm_attn(nodes) * (1 + mul_attn) + add_attn) * gate_attn + nodes

        return nodes
    
class MultiHeadDotAttention(nn.Module):
    def __init__(self, node_dim : int, num_heads : int, device : torch.device):
        super().__init__()
        self.node_dim = node_dim
        self.num_heads = num_heads
        attn_dim = node_dim // num_heads # 64

        self.lin_qkv = nn.Linear(in_features = self.node_dim, out_features = 3 * attn_dim * num_heads, device = device)

        self.lin_nodes_out = nn.Sequential(
           nn.Linear(in_features = attn_dim * num_heads, out_features = self.node_dim, device = device),
          #  nn.SiLU()
        )              

    def forward(self, nodes : Tensor) -> Tensor:
        batch_size, num_nodes, _ = nodes.size()
        
        queries, keys, values = self.lin_qkv(nodes).chunk(chunks = 3, dim = -1)

        queries = queries.reshape(batch_size, num_nodes, self.num_heads, -1).permute(0, 2, 1, 3) # batch_size x num_heads x num_nodes x attn_dim
        keys = keys.reshape(batch_size, num_nodes, self.num_heads, -1).permute(0, 2, 1, 3)       # batch_size x num_heads x num_nodes x attn_dim
        values = values.reshape(batch_size, num_nodes, self.num_heads, -1).permute(0, 2, 1, 3)   # batch_size x num_heads x num_nodes x attn_dim
        # attn_mask = ~torch.eye(queries.size(-2), dtype = torch.bool, device = queries.device)

        weighted_values = F.scaled_dot_product_attention(query = queries, key = keys, value = values).permute(0, 2, 1, 3).flatten(start_dim = 2)

        return self.lin_nodes_out(weighted_values)

class Normalization(nn.Module):
    def __init__(self, node_dim: int, device: device):
        super().__init__()
        
        # self.norm_nodes = nn.InstanceNorm1d(num_features = node_dim, device = device)
        # self.norm_nodes = nn.BatchNorm1d(num_features = node_dim, affine = False, device = device)
        self.norm_nodes = nn.LayerNorm(normalized_shape = node_dim, elementwise_affine = False, device = device)

    def forward(self, nodes : Tensor) -> Tensor:
        # return self.norm_nodes(nodes.permute(0, 2, 1)).permute(0, 2, 1)
        return self.norm_nodes(nodes)

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

'''----- Soft Gumb -----'''
class CosineNoiseScheduler(nn.Module):
  def __init__(self, max_timestep : int, device : torch.device):
    super().__init__()
    self.device = device
    self.max_timestep = max_timestep
    self.offset = .008 # Fixed offset to improve noise prediction at early timesteps

    # Cosine Beta Schedule Formula: https://arxiv.org/abs/2102.09672
    # self.a_bar = torch.cos((torch.linspace(0, 1, self.max_timestep + 1).to(self.device) + self.offset) * 0.5 * math.pi / (1 + self.offset)) ** 2
    # self.a_bar = self.a_bar / self.a_bar[0]
    # self.a_bar = self.a_bar.clamp(min = 0.0001, max = 0.9999)
    self.a_bar = torch.cos(torch.linspace(0, 1, self.max_timestep + 1, device = device) * 0.5 * math.pi) ** 2 # 1 - torch.linspace(0, 1, self.max_timestep + 1, device = device) ** 2
    self.a_bar = self.a_bar.clamp(min = 0.0001, max = 0.9999)
  
  def forward(self, nodes : Tensor, timestep : Tensor):
    ''' Apply noise to graph '''
    noisy_nodes = torch.zeros(size = nodes.size(), device = nodes.device)
    added_noise = torch.zeros(size = nodes.size(), device = nodes.device)
    
    # IsConstructible noise
    noisy_nodes[...,0:2], added_noise[...,0:2] = self.apply_discrete_noise(nodes[...,0:2], timestep)
    # Primitive Types noise
    noisy_nodes[...,2:7], added_noise[...,2:7] = self.apply_discrete_noise(nodes[...,2:7], timestep)
    # Primitive parameters noise
    noisy_nodes[...,7:], added_noise[...,7:] = self.apply_continuous_noise(nodes[...,7:], timestep)
    
    return noisy_nodes, added_noise
  
  def sample_latent(self, batch_size : int) -> Tensor:
    noisy_nodes = torch.zeros(size = (batch_size, 24, 21), device = self.device)

    # IsConstructible noise
    uniform_noise = torch.rand_like(noisy_nodes[...,0:2].clamp(min = 1e-10, max = 1 - 1e-10))
    gumbel_noise = -torch.log(-torch.log(uniform_noise))
    noisy_nodes[...,0:2] = gumbel_noise.softmax(dim = -1)
    # Primitive Types noise
    uniform_noise = torch.rand_like(noisy_nodes[...,2:7].clamp(min = 1e-10, max = 1 - 1e-10))
    gumbel_noise = -torch.log(-torch.log(uniform_noise))
    noisy_nodes[...,2:7] = gumbel_noise.softmax(dim = -1)
    # Primitive parameters noise
    gaussian_noise = torch.randn_like(noisy_nodes[...,7:].clamp(min = 1e-10, max = 1 - 1e-10))
    noisy_nodes[...,7:] = gaussian_noise
    
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
    return a * params + b * noise, noise
  
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
      
    a = self.a_bar[timestep, None, None].sqrt()

    D = params.size(-1)
    noise = torch.log(-torch.log(torch.rand_like(params).clamp(min = 1e-10, max = 1 - 1e-10))) # Gumbel Noise
    return torch.softmax(torch.log(a * params + (1 - a) / D) + noise, dim = -1), noise
  
  def discrete_posterior_step(self, pred_params : Tensor, curr_params : Tensor, timestep : Tensor | int) -> Tensor:
    if type(timestep) is int:
      if timestep == 0: 
        return pred_params
      assert timestep > 0
      assert timestep < self.max_timestep
      timestep = torch.tensor(data = [timestep], device = pred_params.device)
      
    D = pred_params.size(-1)
    a_bar = self.a_bar[timestep, None, None]
    prev_a_bar = self.a_bar[timestep - 1, None, None]
    curr_a = self.a_bar[timestep, None, None] / self.a_bar[timestep - 1, None, None]
    Q_bar = a_bar * torch.eye(D, device = pred_params.device) + (1 - a_bar) / D
    prev_Q_bar = prev_a_bar * torch.eye(D, device = pred_params.device) + (1 - prev_a_bar) / D
    curr_Q = curr_a * torch.eye(D, device = pred_params.device) + (1 - curr_a) / D

    xt = curr_params # F.one_hot(torch.argmax(curr_params, dim = -1), D).to(pred_params.device).float()
    qt = xt @ curr_Q.permute(0, 2, 1) # (b, m, d), since xt is onehot we are plucking out rows corresponding to p(x_t = class | x_(t-1))
    qt_bar = xt @ Q_bar.permute(0, 2, 1) # (b, m, d), since xt is onehot we are plucking out rows corresponding to p(x_t = class | x_0)
    q = qt.unsqueeze(2) / qt_bar.unsqueeze(3) # (b, m, d, d), perform an outer product so element at (b, m, i, j) = p(x_t = class | x_(t-1) = j) / p(x_t = class | x_0 = i)
    q = q * prev_Q_bar.unsqueeze(1) # (b, m, d, d), broadcast multiply so element at (b, m, i, j) = p(x_t = class | x_(t-1) = j) * p(x_(t-1) = j | x_0 = i) / p(x_t = class | x_0 = i)
    pred_class_probs = pred_params.unsqueeze(-2) # (b, n, 1, d), make probs into row vector
    posterior_distribution = pred_class_probs @ q # (b, n, 1, d), batched vector-matrix multiply
    posterior_distribution = posterior_distribution.squeeze(-2) # (b, n, d)

    # posterior_distribution = (curr_a * xt + (1 - curr_a) / D) * (prev_a_bar * pred_params + (1 - prev_a_bar) / D) # Unnormalized posterior distribution

    # a_bar = self.a_bar[timestep]
    # prev_a_bar = self.a_bar[timestep - 1]
    # curr_a = a_bar / prev_a_bar
    # D = pred_params.size(-1)
    # m = torch.ones(D, device = self.device) / D
    # u = (1 - prev_a_bar) / (1 - a_bar)

    # xt = F.one_hot(torch.argmax(curr_params, dim = -1), D).to(pred_params.device).float()
    # mxt = (xt * m).sum(dim = -1, keepdim = True)
    # lmbda = (1 - prev_a_bar) * (1 - curr_a) * mxt / (a_bar + (1 - a_bar) * mxt)
    # gmma = (u - lmbda - u * curr_a) * (xt * pred_params).sum(dim = -1, keepdim = True)

    # posterior_distribution = (1 - u) * pred_params + (u * curr_a + gmma) * xt + (u * (1 - curr_a) - gmma) * m

    # assert posterior_distribution.isfinite().all(), timestep
    noise = torch.log(-torch.log(torch.rand_like(pred_params).clamp(min = 1e-10, max = 1 - 1e-10))) # Gumbel Noise
    return torch.softmax(torch.log(posterior_distribution) + noise, dim = -1) #, noise

'''----- Soft Gauss -----'''
# class CosineNoiseScheduler(nn.Module):
#   def __init__(self, max_timestep : int, device : torch.device):
#     super().__init__()
#     self.device = device
#     self.max_timestep = max_timestep
#     self.offset = .008 # Fixed offset to improve noise prediction at early timesteps

#     t = torch.linspace(0, 1, self.max_timestep + 1, device = device)

#     # Cosine Beta Schedule Formula: https://arxiv.org/abs/2102.09672
#     # self.a_bar = torch.cos((torch.linspace(0, 1, self.max_timestep + 1).to(self.device) + self.offset) * 0.5 * math.pi / (1 + self.offset)) ** 2
#     # self.a_bar = self.a_bar / self.a_bar[0]
#     # self.a_bar = self.a_bar.clamp(min = 0.0001, max = 0.9999)
#     self.a_bar = torch.cos(t * 0.5 * math.pi) ** 2 # 1 - torch.linspace(0, 1, self.max_timestep + 1, device = device) ** 2
#     self.a_bar = self.a_bar.clamp(min = 0.0001, max = 0.9999)

#     # Discrete variance schedule
#     self.clamp_min = -10
#     s = 16
#     self.da_bar = (1-t) * (1 - t ** (1/s)) + (t) * (torch.exp(t * -s))

  
#   def forward(self, nodes : Tensor, timestep : Tensor):
#     ''' Apply noise to graph '''
#     noisy_nodes = torch.zeros(size = nodes.size(), device = nodes.device)
#     added_noise = torch.zeros(size = nodes.size(), device = nodes.device)
    
#     # IsConstructible noise
#     noisy_nodes[...,0:2], added_noise[...,0:2] = self.apply_discrete_noise(nodes[...,0:2], timestep)
#     # Primitive Types noise
#     noisy_nodes[...,2:7], added_noise[...,2:7] = self.apply_discrete_noise(nodes[...,2:7], timestep)
#     # Primitive parameters noise
#     noisy_nodes[...,7:], added_noise[...,7:] = self.apply_continuous_noise(nodes[...,7:], timestep)
    
#     return noisy_nodes, added_noise
  
#   def sample_latent(self, batch_size : int) -> Tensor:
#     noisy_nodes = torch.zeros(size = (batch_size, 24, 21), device = self.device)

#     # IsConstructible noise
#     noisy_nodes[...,0:2] = torch.randn_like(noisy_nodes[...,0:2]).softmax(dim = -1)
#     # Primitive Types noise
#     noisy_nodes[...,2:7] = torch.randn_like(noisy_nodes[...,2:7]).softmax(dim = -1)
#     # Primitive parameters noise
#     noisy_nodes[...,7:] = torch.randn_like(noisy_nodes[...,7:])
    
#     return noisy_nodes
  
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
#     return a * params + b * noise, noise
  
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
      
#     a = torch.sqrt(self.a_bar[timestep, None, None])
#     b = torch.sqrt(1 - self.a_bar[timestep, None, None])

#     noise = torch.randn_like(params)
#     return torch.softmax(a * params.log().clamp(self.clamp_min) + b * noise, dim = -1), noise
  
#   def discrete_posterior_step(self, pred_params : Tensor, curr_params : Tensor, timestep : Tensor | int) -> Tensor:
#     if type(timestep) is int:
#       if timestep == 0: 
#         return pred_params
#       assert timestep > 0
#       assert timestep < self.max_timestep
#       timestep = torch.tensor(data = [timestep], device = pred_params.device)

#     curr_a = self.da_bar[timestep] / self.da_bar[timestep - 1]
#     curr_b = 1 - curr_a
#     curr_a_bar = self.da_bar[timestep]
#     curr_b_bar = 1 - curr_a_bar
#     prev_a_bar = self.da_bar[timestep - 1]
#     prev_b_bar = 1 - prev_a_bar

#     if timestep > 1:
#       log_pred = pred_params.log().clamp(self.clamp_min)
#       log_curr = curr_params.log().clamp(self.clamp_min)
#       mean = (prev_a_bar.sqrt() * curr_b * log_pred + curr_a.sqrt() * prev_b_bar * log_curr) / curr_b_bar
#       noise = torch.randn_like(pred_params)
#       return torch.softmax(mean + torch.sqrt(prev_b_bar / curr_b_bar * curr_b) * noise, -1) #, noise
#     else:
#       return pred_params





# class GD3PM(nn.Module):
#   def __init__(self, device : torch.device):
#     super().__init__()
#     self.device = device
#     self.node_dim = NODE_FEATURE_DIMENSION
#     self.node_hidden_dim = 1024
#     self.cond_hidden_dim = 1024
#     self.num_tf_layers = 32
#     self.num_heads = 16
#     self.max_timestep = 1000
#     self.noise_scheduler = CosineNoiseScheduler(self.max_timestep, self.device)
#     self.architecture = DiffusionModel(node_dim = self.node_dim, 
#                                        node_hidden_dim = self.node_hidden_dim,
#                                        cond_hidden_dim = self.cond_hidden_dim,
#                                        num_heads = self.num_heads,
#                                        num_tf_layers = self.num_tf_layers,
#                                        max_timestep = self.max_timestep,
#                                        device = self.device)

#   def forward(self, nodes : Tensor, timestep : Tensor):
#     nodes = self.architecture(nodes, timestep)
#     # Normalize to Probabilities
#     nodes[...,0] = nodes[...,0].sigmoid()
#     nodes[...,1:6] = nodes[...,1:6].softmax(dim = -1)

#     return nodes

#   @torch.no_grad()
#   def sample(self, batch_size : int):
#     # Sample Noise
#     nodes = self.noise_scheduler.sample_latent(batch_size)
#     nodes = nodes.to(self.device)
#     return self.denoise(nodes)

#   @torch.no_grad()
#   def denoise(self, nodes):  
#     for t in reversed(range(1, self.max_timestep)):
#       # model expects a timestep for each batch
#       batch_size = nodes.size(0)
#       time = torch.Tensor([t]).expand(batch_size).int().to(self.device)
#       nodes, _ = self.noise_scheduler(self.forward(nodes, time), t - 1)

#     return nodes

# class DiffusionModel(nn.Module):
#     def __init__(self, node_dim, node_hidden_dim, cond_hidden_dim, num_heads, num_tf_layers, max_timestep, device: device):
#         super().__init__()
#         self.time_embedder = TimeEmbedder(max_timestep, cond_hidden_dim, device)

#         # Input MLP layers
#         self.mlp_in_nodes = nn.Sequential(
#             nn.Linear(in_features = node_dim, out_features = node_hidden_dim, device = device),
#             nn.LeakyReLU(0.1),
#             nn.Linear(in_features = node_hidden_dim, out_features = node_hidden_dim, device = device),
#             # nn.SiLU()
#         )

#         self.mlp_in_conds = nn.Sequential(
#             nn.Linear(in_features = cond_hidden_dim, out_features = cond_hidden_dim, device = device),
#             nn.LeakyReLU(0.1),
#             nn.Linear(in_features = cond_hidden_dim, out_features = cond_hidden_dim, device = device),
#             # nn.SiLU()
#         )

#         # Transformer Layers with Graph Attention Network
#         self.block_layers = nn.ModuleList([
#             TransformerLayer(
#                 node_dim = node_hidden_dim,
#                 cond_dim = cond_hidden_dim,
#                 num_heads = num_heads,
#                 device = device
#             ) for _ in range(num_tf_layers)
#         ])
        
#         self.norm = nn.Sequential(
#             # nn.LeakyReLU(0.1),
#             Normalization(node_dim = node_hidden_dim, device = device),
#         )
#         # Output MLP layers
#         self.mlp_out_node_types = nn.Sequential(
#             nn.Linear(in_features = node_hidden_dim, out_features = node_hidden_dim, device = device),
#             nn.LeakyReLU(0.1),
#             # nn.Linear(in_features = self.node_hidden_dim, out_features = self.node_hidden_dim, device = device),
#             # nn.LeakyReLU(0.1),
#             nn.Linear(in_features = node_hidden_dim, out_features = 6, device = device)
#         )
#         self.mlp_out_node_params = nn.Sequential(
#             nn.Linear(in_features = node_hidden_dim, out_features = node_hidden_dim, device = device),
#             nn.LeakyReLU(0.1),
#             # nn.Linear(in_features = self.node_hidden_dim, out_features = self.node_hidden_dim, device = device),
#             # nn.LeakyReLU(0.1),
#             nn.Linear(in_features = node_hidden_dim, out_features = 14, device = device)
#         )

#     def forward(self, nodes : Tensor, timestep : Tensor):
#         nodes = self.mlp_in_nodes(nodes)     # shape: (batch_size, num_nodes, node_hidden_dim)
#         conds = self.time_embedder(timestep) # shape: (batch_size, cond_hidden_dim)
#         conds = self.mlp_in_conds(conds)     # shape: (batch_size, cond_hidden_dim)

#         for layer in self.block_layers:
#             nodes = layer(nodes, conds) # shape: (batch_size, num_nodes, node_hidden_dim) ; shape: (batch_size, num_nodes, num_nodes, edge_hidden_dim)

#         nodes = self.norm(nodes)
#         nodes = torch.cat([self.mlp_out_node_types(nodes), self.mlp_out_node_params(nodes)], dim = -1) # shape: (batch_size, num_nodes, node_dim)

#         return nodes

# class TransformerLayer(nn.Module):
#     def __init__(self, node_dim: int, cond_dim: int, num_heads: int, device: device):
#         super().__init__()
#         self.node_dim = node_dim

#         # Normalization
#         self.norm_in = nn.Sequential(
#             # nn.LeakyReLU(0.1),
#             Normalization(node_dim = node_dim, device = device),
#         )

#         # Attention Layer
#         self.attention_heads = MultiHeadDotAttention(
#             node_dim = node_dim,
#             num_heads = num_heads,
#             device = device
#         )

#         # Normalization
#         self.norm_attn = nn.Sequential(
#             # nn.LeakyReLU(0.1),
#             Normalization(node_dim = node_dim, device = device),
#         )

#         # Node and edge MLPs
#         self.mlp_nodes = nn.Sequential(
#             nn.Linear(in_features = node_dim, out_features = node_dim, device = device),
#             nn.LeakyReLU(0.1),
#             nn.Linear(in_features = node_dim, out_features = node_dim, device = device),
#             # nn.SiLU()
#         )

#         # Conditioning
#         self.film_in = FiLM(node_dim = node_dim, cond_dim = cond_dim, device = device)
#         self.film_attn = FiLM(node_dim = node_dim, cond_dim = cond_dim, device = device)
#         # self.mul = nn.Linear(in_features = cond_dim, out_features = 2 * (node_dim + edge_dim), device = device)

#     def forward(self, nodes : Tensor, conds : Tensor) -> Tensor:
#         # Attention
#         nodes = self.attention_heads(self.film_in(self.norm_in(nodes), conds)) + nodes
#         # MLP
#         nodes = self.mlp_nodes(self.film_attn(self.norm_attn(nodes), conds)) + nodes

#         return nodes

# class MultiHeadMLPAttention(nn.Module):
#     def __init__(self, node_dim : int, num_heads : int, device : torch.device):
#         super().__init__()
#         self.num_heads = num_heads

#         self.lin_v = nn.Linear(in_features = 2 * node_dim, out_features = node_dim, device = device)
#         self.mlp_w = nn.Sequential(
#             nn.Linear(in_features = 2 * node_dim, out_features = 2 * node_dim, device = device),
#             nn.LeakyReLU(0.1),
#             nn.Linear(in_features = 2 * node_dim, out_features = num_heads, device = device),
#         )

#         self.lin_nodes_out = nn.Linear(in_features = node_dim, out_features = node_dim, device = device)                

#     def forward(self, nodes : Tensor) -> Tensor:
#         b, n, d = nodes.size()
#         h = self.num_heads
        
#         features = torch.cat([nodes.unsqueeze(1).expand(-1, n, -1, -1), nodes.unsqueeze(2).expand(-1, -1, n, -1)], dim = -1)
#         v = self.lin_v(features).reshape(b, n, n, h, -1)
#         w = self.mlp_w(features).reshape(b, n, n, h, -1).softmax(dim = 2)
#         weighted_values = (v * w).sum(dim = 2).flatten(start_dim = 2)

#         return self.lin_nodes_out(weighted_values)
    
# class MultiHeadDotAttention(nn.Module):
#     def __init__(self, node_dim : int, num_heads : int, device : torch.device):
#         super().__init__()
#         self.node_dim = node_dim
#         self.num_heads = num_heads
#         self.attn_dim = node_dim // num_heads

#         self.lin_qkv = nn.Linear(in_features = self.node_dim, out_features = 3 * self.node_dim, device = device)

#         self.lin_nodes_out = nn.Linear(in_features = self.node_dim, out_features = self.node_dim, device = device)                     

#     def forward(self, nodes : Tensor) -> Tensor:
#         batch_size, num_nodes, _ = nodes.size()
        
#         queries, keys, values = self.lin_qkv(nodes).chunk(chunks = 3, dim = -1)

#         queries = queries.reshape(batch_size, num_nodes, self.num_heads, -1).permute(0, 2, 1, 3) # batch_size x num_heads x num_nodes x attn_dim
#         keys = keys.reshape(batch_size, num_nodes, self.num_heads, -1).permute(0, 2, 1, 3)       # batch_size x num_heads x num_nodes x attn_dim
#         values = values.reshape(batch_size, num_nodes, self.num_heads, -1).permute(0, 2, 1, 3)   # batch_size x num_heads x num_nodes x attn_dim

#         weighted_values = F.scaled_dot_product_attention(query = queries, key = keys, value = values).permute(0, 2, 1, 3).flatten(start_dim = 2)

#         return self.lin_nodes_out(weighted_values)

# class Normalization(nn.Module):
#     def __init__(self, node_dim: int, device: torch.device):
#         super().__init__()
        
#         # self.norm_nodes = nn.InstanceNorm1d(num_features = node_dim, device = device)
#         self.norm_nodes = nn.LayerNorm(normalized_shape = node_dim, elementwise_affine = False, device = device)

#     def forward(self, nodes : Tensor) -> Tensor:
#         # return self.norm_nodes(nodes.permute(0, 2, 1)).permute(0, 2, 1)
#         return self.norm_nodes(nodes)

# class FiLM(nn.Module):
#     def __init__(self, node_dim : int, cond_dim : int, device : device):
#         super().__init__()
        
#         self.lin_node = nn.Linear(in_features = cond_dim, out_features = 2 * node_dim, device = device)
    
#     def forward(self, node : Tensor, cond : Tensor) -> Tensor:
#         node_mul, node_add = self.lin_node(cond).unsqueeze(1).chunk(chunks = 2, dim = -1)

#         return node_mul * node + node_add + node


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

#     # Cosine Beta Schedule Formula: https://arxiv.org/abs/2102.09672
#     self.a_bar = torch.cos((torch.linspace(0, 1, self.max_timestep + 1).to(self.device) + self.offset) * 0.5 * math.pi / (1 + self.offset)) ** 2
#     self.a_bar = self.a_bar / self.a_bar[0]
#     self.a_bar = self.a_bar.clamp(min = 0.001, max = 0.999)
#     self.sqrt_a_bar = torch.sqrt(self.a_bar).clamp(min = 0.001, max = 0.999)
#     self.sqrt_b_bar = torch.sqrt(1 - self.a_bar).clamp(min = 0.001, max = 0.999)
  
#   def forward(self, nodes : Tensor, timestep : Tensor):
#     ''' Apply noise to graph '''
#     noisy_nodes = torch.zeros(size = nodes.size(), device = nodes.device)
#     added_noise = torch.zeros(size = nodes.size(), device = nodes.device)
    
#     # IsConstructible noise
#     noisy_nodes[...,0], added_noise[...,0] = self.apply_binary_noise(nodes[...,0], timestep)
#     # Primitive Types noise
#     noisy_nodes[...,1:6], added_noise[...,1:6] = self.apply_discrete_noise(nodes[...,1:6], timestep)
#     # Primitive parameters noise
#     noisy_nodes[...,6:], added_noise[...,6:] = self.apply_continuous_noise(nodes[...,6:], timestep)
    
#     return noisy_nodes, added_noise
  
#   def sample_latent(self, batch_size : int) -> Tensor:
#     noisy_nodes = torch.zeros(size = (batch_size, 24, 20), device = self.device)

#     # IsConstructible noise
#     uniform_noise = torch.rand_like(noisy_nodes[...,0])
#     logistic_noise = torch.log(uniform_noise) - torch.log(1 - uniform_noise)
#     noisy_nodes[...,0] = logistic_noise.sigmoid()
#     # Primitive Types noise
#     uniform_noise = torch.rand_like(noisy_nodes[...,1:6])
#     gumbel_noise = -torch.log(-torch.log(1 - uniform_noise))
#     noisy_nodes[...,1:6] = gumbel_noise.softmax(dim = -1)
#     # Primitive parameters noise
#     gaussian_noise = torch.randn_like(noisy_nodes[...,6:])
#     noisy_nodes[...,6:] = gaussian_noise
    
#     return noisy_nodes
  
#   def apply_continuous_noise(self, params : Tensor, timestep : Tensor | int) -> Tensor:
#     if type(timestep) is int:
#       if timestep == 0: 
#         return params, 0 
#       assert timestep > 0 
#       assert timestep < self.max_timestep 
#       timestep = [timestep]

#     a = self.sqrt_a_bar[timestep, None, None]
#     b = self.sqrt_b_bar[timestep, None, None]

#     noise = torch.randn_like(params)
#     return a * params + b * noise, noise
  
#   def apply_discrete_noise(self, params : Tensor, timestep : Tensor | int) -> Tensor:
#     if type(timestep) is int:
#       if timestep == 0: 
#         return params, 0 
#       assert timestep > 0
#       assert timestep < self.max_timestep
#       timestep = [timestep]
      
#     a = self.a_bar[timestep, None, None]
#     s = 1 - self.a_bar[timestep, None, None]

#     D = params.size(-1)
#     noise = torch.log(-torch.log(torch.rand_like(params).clamp(min = 1e-10, max = 1 - 1e-10))) # Gumbel Noise
#     return torch.softmax((torch.log(a * params + (1 - a) / D) + noise) / s, dim = -1), noise
  
#   def apply_binary_noise(self, params : Tensor, timestep : Tensor | int) -> Tensor:
#     if type(timestep) is int:
#       if timestep == 0: 
#         return params, 0 
#       assert timestep > 0
#       assert timestep < self.max_timestep
#       timestep = [timestep]
    
#     a = self.a_bar[timestep, None]
#     s = 1 - self.a_bar[timestep, None]

#     noise = torch.rand_like(params).clamp(min = 1e-10, max = 1 - 1e-10)
#     noise = torch.log(noise) - torch.log(1 - noise) # Logistic Noise
#     return torch.sigmoid(( (a * params + (1 - a) / 2).log() - (a * (1 - params) + (1 - a) / 2).log() + noise) / s), noise