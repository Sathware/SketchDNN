# # %%
# import os
# from typing import Sequence
# from tqdm import tqdm
# from matplotlib import pyplot as plt
# from torch.utils.tensorboard.writer import SummaryWriter
# # from torch.profiler import profile, record_function, ProfilerActivity
# import torch
# import torch.nn.functional as F
# from torch import Tensor
# from nodediff_model import GD3PM
# # from loss import diffusion_loss
# from dataset1 import SketchDataset
# from torch.utils.data import DataLoader, Subset, random_split
# os.chdir('SketchGraphs/')
# import sketchgraphs.data as datalib
# os.chdir('../')
# from utils import ToNaive

# # %%
# import torch.multiprocessing as mp
# from torch.utils.data.distributed import DistributedSampler
# from torch.distributed.optim import ZeroRedundancyOptimizer
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group

# torch.set_float32_matmul_precision('high')

# # %%
# class MultiGPUTrainer:
#     def __init__(
#             self,
#             model: GD3PM,
#             train_set: Subset,
#             validate_set: Subset,
#             # learning_rate: float,
#             gpu_id: int,
#             num_epochs: int,
#             experiment_string: str,
#             # batch_size: int
#             ):
#         model.device = gpu_id
#         self.model = DDP(model.to(gpu_id), device_ids=[gpu_id])
#         # decay = 0.9999
#         # self.ema_model = torch.optim.swa_utils.AveragedModel(self.model.module, device = gpu_id, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay))
#         self.batch_size = 512
#         # self.num_grad_accum_steps = 4

#         self.train_sampler = DistributedSampler(train_set)
#         self.train_loader = DataLoader(dataset = train_set, 
#                                        batch_size = self.batch_size, 
#                                        shuffle = False, 
#                                        pin_memory = True, 
#                                        sampler = self.train_sampler
#                                       )
#         self.validate_sampler = DistributedSampler(validate_set)
#         self.validate_loader = DataLoader(dataset = validate_set, 
#                                           batch_size = self.batch_size, 
#                                           shuffle = False, 
#                                           pin_memory = True, 
#                                           sampler = self.validate_sampler
#                                          )
        
#         self.learning_rate = 1e-4
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
#         # sched = lambda epoch : 1 # if epoch >= 1 else 1e-2
#         # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda = sched)

#         # if os.path.exists(f"checkpoint_nodesoftgaussdiff_ddp_adam_32layers_512nodedim_512condim_16heads.pth"):
#         #     map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_id}
#         #     checkpoint = torch.load(f"checkpoint_nodesoftgaussdiff_ddp_adam_32layers_512nodedim_512condim_16heads.pth", map_location = map_location)
#         #     self.model.load_state_dict(checkpoint["model"])
#         #     self.optimizer.load_state_dict(checkpoint["optimizer"])

#         # self.optimizer = ZeroRedundancyOptimizer(self.model.parameters(),
#         #                                          optimizer_class = torch.optim.Adam,
#         #                                          lr = learning_rate
#         #                                         )
#         # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0 = 4, T_mult = 2, eta_min = 1e-5)
        
#         self.gpu_id = gpu_id
#         self.experiment_string = experiment_string
#         self.writer = SummaryWriter(f'runs8/{self.experiment_string}')
#         self.num_epochs = num_epochs

#         self.global_step = 0
#         self.curr_epoch = 0
#         self.min_validation_loss = float('inf')

#         self.record_freq = len(self.train_loader) // 5
#         if self.record_freq == 0: self.record_freq = 1

#         # self.timestep_distribution = torch.distributions.beta.Beta(torch.tensor([1.6]), torch.tensor([2.0]))
#         self.timestep_distribution = torch.where(torch.linspace(0, 1, self.model.module.max_timestep - 1) <= 0.5, 0.75, 0.25)
    
#     def render_nodes(self, nodes, ax = None):
#         SketchDataset.render_graph(ToNaive(nodes[...,1:]).cpu(), torch.zeros(size = (16, 16, 17)).cpu(), ax)
    
#     def train_batch(self, nodes : Tensor, params_mask : Tensor, edges : Tensor) -> float:
#         self.optimizer.zero_grad()

#         batch_size = nodes.size(0)
#         nodes = nodes.to(self.gpu_id) # .repeat(len(self.model.module.partitions), 1, 1) #.reshape(5 * batch_size, nodes.size(1), nodes.size(2))
#         params_mask = params_mask.to(self.gpu_id) # .repeat(len(self.model.module.partitions), 1, 1) #.reshape(5 * batch_size, params_mask.size(1), params_mask.size(2))

#         # t = torch.cat([torch.randint(low = low_step, high = high_step, size = (batch_size,), device = self.gpu_id) for (low_step, high_step) in self.model.module.partitions])
#         # t1 = torch.randint(low = 1, high = self.model.module.max_timestep // 4, size = (batch_size // 3,)).to(self.gpu_id)
#         # t2 = torch.randint(low = self.model.module.max_timestep // 4, high = self.model.module.max_timestep // 2, size = (batch_size // 3,)).to(self.gpu_id)
#         # t3 = torch.randint(low = self.model.module.max_timestep // 2, high = self.model.module.max_timestep, size = (batch_size // 3,)).to(self.gpu_id)
#         # t = torch.cat([t1,t2,t3])
#         t = torch.multinomial(self.timestep_distribution, batch_size, replacement = True).to(self.gpu_id) + 1
#         # t = torch.randint(low = 1, high = self.model.module.max_timestep, size = (batch_size,), device = self.gpu_id)
#         # noised_nodes, _ = self.model.module.noise_scheduler(nodes, t)
#         noised_nodes = self.model.module.noise_scheduler(nodes, t)

#         pred_nodes = self.model(noised_nodes, t)
#         assert pred_nodes.isfinite().all(), "NaN was generated by model!"

#         loss_dict = {} # dictionary to record loss values
#         a_bar_t = self.model.module.noise_scheduler.a_bar[t]
#         scales = torch.clamp(a_bar_t / (1 - a_bar_t), max = 32, min = .1) # .unsqueeze(1).unsqueeze(1)
#         # scales = torch.where(t <= (self.model.module.max_timestep / 5), 4, 1)
#         # scales = scales.unsqueeze(1).unsqueeze(1)
#         loss = self.diffusion_loss(pred_nodes, nodes, params_mask, loss_dict, edges, scales)
#         assert loss.isfinite().all(), "NaN was generated!"

#         loss.backward()
#         # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
#         self.optimizer.step()
#         # self.ema_model.update_parameters(self.model.module)

#         if self.global_step % 250 == 249 and self.gpu_id == 0:
#             p = 0 # batch_size * torch.randint(low = 0, high = len(self.model.module.partitions), size = (1,), device = 'cpu').item()
#             fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(12, 4))
#             fig.suptitle(f"Noised - Pred - True at time {t[p].item()}")
#             self.render_nodes(noised_nodes[p].squeeze(0), axes[0])
#             self.render_nodes(pred_nodes[p].squeeze(0), axes[1])
#             self.render_nodes(nodes[p].squeeze(0), axes[2])
#             self.writer.add_figure("Training/Visual", fig, self.global_step)
#             plt.close()

#         return loss_dict
    
#     def train_epoch(self):
#         self.train_sampler.set_epoch(self.curr_epoch)
#         pbar = tqdm(self.train_loader) if self.gpu_id == 0 else self.train_loader
#         for idx, targets in enumerate(pbar):
#             nodes, params_mask, edges = targets
            
#             iter_loss_dict = self.train_batch(nodes, params_mask, edges)

#             self.global_step += 1
#             # self.scheduler.step(self.curr_epoch + idx / iters)
#             # if self.gpu_id == 0: self.writer.add_scalar("Learning Rate", self.scheduler.get_last_lr()[0], self.global_step)
            
#             if self.gpu_id == 0: self.plot_loss(iter_loss_dict) # self.writer.add_scalar("Training Loss", iter_loss, self.global_step)

#             iter_loss = iter_loss_dict["node loss"]
#             if self.gpu_id == 0: pbar.set_description(f"Training Epoch {self.curr_epoch} Iter Loss: {iter_loss}")
    
#     # def extract_params(self, node_type : int, pred_node : Tensor, sub_type : int):
#     #     params = None
#     #     match node_type:
#     #         case 0: # Line
#     #             params = pred_node[7:11]
#     #         case 1: # Circle
#     #             params = pred_node[11:14]
#     #         case 2: # Arc
#     #             params = pred_node[14:19]
#     #         case 3: # Point
#     #             params = pred_node[19:21]
        
#     #     match sub_type:
#     #         case 0: # Start
#     #             if node_type == 0: return 3, params[0:2] # Line Startpoint
#     #             if node_type == 2: return 3, params[0:2] # Arc Startpoint
#     #         case 1: # Midpoint
#     #             if node_type == 0: return 3, 0.5 * (params[0:2] + params[2:4]) # Line Midpoint
#     #             if node_type == 2: # Arc Midpoint
#     #                 m = torch.tensor([[0.0, -1.0], [1.0, 0.0]], device = pred_node.device)
#     #                 a = params[0:2]
#     #                 b = params[2:4]
#     #                 r_s = params[4]

#     #                 d = a - b
#     #                 w = 0.5 * (d ** 2).sum().sqrt()
#     #                 w = w.clamp(max = r_s.abs())
#     #                 h = torch.sqrt(r_s ** 2 - w ** 2)

#     #                 # Center point
#     #                 c = b + d / 2 + h / w * d @ m * r_s.sign()

#     #                 mid = 0.5 * (a + b) - c
#     #                 midpoint = c + mid / torch.sum(mid ** 2).sqrt()
#     #                 # assert midpoint.isfinite().all(), "Arc Midpoint calculation produced Nan!"
#     #                 return 3, midpoint
#     #         case 2: # End
#     #             if node_type == 0: return 3, params[2:4] # Line Endpoint
#     #             if node_type == 2: return 3, params[2:4] # Arc Endpoint
        
#     #     return node_type, params

#     # def constraint_enforcement_loss(self, a_type, a_params, b_type, b_params, edge_type):
#     #     # if a_type == 4 or b_type == 4:
#     #     #     return 0
#     #     # if a_type > b_type:
#     #     #     a_type, b_type = b_type, a_type
#     #     #     a_params, b_params = b_params, a_params
        
#     #     match edge_type:
#     #         case 0: # Coincident
#     #             if (a_type == 3 and b_type == 3): # Two Points
#     #                 los = ((a_params - b_params) ** 2).sum()
#     #                 # assert los.isfinite().all(), "Loss generated in Point Coincident loss!"
#     #                 return los
#     #             # if (a_type == 0 and b_type == 3): # Line and Point
#     #             #     return 1 - torch.abs(torch.sum(F.normalize(b_params - a_params[0:2]) * F.normalize(a_params[2:4] - a_params[0:2])))
#     #             # if (a_type == 1 and b_type == 3): # Circle and Point
#     #             #     return torch.abs(a_params[2] - torch.linalg.vector_norm(a_params[0:2] - b_params))
#     #     return None

#     def extract_parameters(self, types : Tensor, params : Tensor, start : Tensor, center : Tensor, end : Tensor, none : Tensor):
#         out = torch.zeros(params.size(0), 2, device = params.device)

#         if start[0].numel() > 0:
#                 line_start_points = params[start][...,0:2]
#                 arc_start_points = params[start][...,7:9]
#                 out[start] = torch.where((types == 0)[start].unsqueeze(1), line_start_points, arc_start_points)
#                 assert out[start].isfinite().all(), "Constraint enforcement loss has Nan!"

#         if end[0].numel() > 0:
#                 line_end_points = params[end][...,2:4]
#                 arc_end_points = params[end][...,9:11]
#                 out[end] = torch.where((types == 0)[end].unsqueeze(1), line_end_points, arc_end_points)
#                 assert out[end].isfinite().all(), "Constraint enforcement loss has Nan!"

#         if center[0].numel() > 0:
#                 center_params = params[center]
#                 line_center_points = 0.5 * (center_params[...,0:2] + center_params[...,2:4])
#                 m = torch.tensor([[0.0, -1.0], [1.0, 0.0]], device = params.device)
#                 a = center_params[...,7:9]
#                 b = center_params[...,9:11]
#                 r_s = center_params[...,11,None]

#                 d = a - b
#                 w = (d ** 2).sum(dim = -1, keepdim=True).sqrt()
#                 w = w.clamp(max = 2 * r_s.abs())
#                 h = torch.sqrt(r_s ** 2 - w ** 2 / 4)

#                 # Center point
#                 arc_center_points = (a + b) / 2 + h / w.clamp(min = 1e-8) * d @ m * r_s.sign()

#                 # mid = 0.5 * (a + b) - c
#                 # arc_mid_points = c + mid / torch.sum(mid ** 2, dim = -1, keepdim=True).sqrt().clamp(min = 1e-8)

#                 out[center] = torch.where((types == 0)[center].unsqueeze(1), line_center_points, torch.where((types == 2)[center].unsqueeze(1), arc_center_points, params[center][...,4:6]))
#                 assert out[center].isfinite().all(), "Constraint enforcement loss has Nan!"

#         if none[0].numel() > 0:
#                 out[none] = torch.where((types == 3)[none].unsqueeze(1), params[none][...,12:14], float('nan'))

#         return out

#     def diffusion_loss(self, pred_nodes : Tensor, true_nodes : Tensor, params_mask : Tensor, loss_dict : dict, edges : Tensor, scales : Tensor | int = 1) -> Tensor:
#         '''Node Loss'''
#         primitive_type_labels = true_nodes[:,:,2:7]    # batch_size x num_nodes (class index for each node)
#         primitive_type_logits = pred_nodes[:,:,2:7] # .log()#.permute(0,2,1).contiguous() # batch_size x num_primitive_types x num_nodes
        
#         node_cross = ((-primitive_type_logits.log() * primitive_type_labels)).mean() # (-primitive_type_labels * primitive_type_logits).mean()
        
#         construct_type_labels = true_nodes[:,:,0:2]    # batch_size x num_nodes (class index for each node)
#         construct_type_logits = pred_nodes[:,:,0:2]# .log()#.permute(0,2,1).contiguous() # batch_size x num_primitive_types x num_nodes
        
#         bce = ((-construct_type_logits.log() * construct_type_labels)).mean() # (-construct_type_labels * construct_type_logits).mean()
        
#         pred_params = pred_nodes[:,:,7:]
#         target_params = true_nodes[:,:,7:]
#         mse = ((pred_params - target_params) ** 2 * params_mask).sum() / params_mask.sum()

#         node_loss = bce + node_cross + mse

#         '''Coincident Constraint Loss'''
#         # Get indices of coincident constraints i.e. where the argmax is 0
#         coincident_indices = torch.nonzero(edges[..., 8:].argmax(dim=-1) == 0, as_tuple=True) # Tuple of (batch_indices, node_i_indices, node_j_indices)
#         params = pred_nodes[...,7:] * params_mask
#         node_a = params[coincident_indices[0], coincident_indices[1]]
#         node_a_type = true_nodes[coincident_indices[0], coincident_indices[1]][...,2:7].argmax(dim = -1)
#         node_b = params[coincident_indices[0], coincident_indices[2]]
#         node_b_type = true_nodes[coincident_indices[0], coincident_indices[2]][...,2:7].argmax(dim = -1)

#         suba_types = edges[coincident_indices][...,0:4].argmax(dim = -1)
#         subb_types = edges[coincident_indices][...,4:8].argmax(dim = -1)

#         suba_start_indices =  torch.nonzero(suba_types == 0, as_tuple = True)
#         suba_center_indices = torch.nonzero(suba_types == 1, as_tuple = True)
#         suba_end_indices =    torch.nonzero(suba_types == 2, as_tuple = True)
#         suba_none_indices =   torch.nonzero(suba_types == 3, as_tuple = True)
#         i_points = self.extract_parameters(node_a_type, node_a, suba_start_indices, suba_center_indices, suba_end_indices, suba_none_indices)
        
#         subb_start_indices =  torch.nonzero(subb_types == 0, as_tuple = True)
#         subb_center_indices = torch.nonzero(subb_types == 1, as_tuple = True)
#         subb_end_indices =    torch.nonzero(subb_types == 2, as_tuple = True)
#         subb_none_indices =   torch.nonzero(subb_types == 3, as_tuple = True)
#         j_points = self.extract_parameters(node_b_type, node_b, subb_start_indices, subb_center_indices, subb_end_indices, subb_none_indices)

#         const_enf_loss = torch.nanmean((i_points - j_points) ** 2)
#         assert const_enf_loss.isfinite().all(), "Constraint enforcement loss has Nan!"

#         # prim_types = true_nodes[:,:,2:7].argmax(dim = -1)
#         # line_indices =   torch.nonzero(prim_types == 0, as_tuple = True)
#         # circle_indices = torch.nonzero(prim_types == 1, as_tuple = True)
#         # arc_indices =    torch.nonzero(prim_types == 2, as_tuple = True)
#         # point_indices =  torch.nonzero(prim_types == 3, as_tuple = True)

#         # const_enf_loss = 0
#         # num = 0
#         # for b in range(edges.size(0)):
#         #     for i in range(edges.size(1)):
#         #         for j in range(edges.size(2)):
#         #             a_type, a_params = self.extract_params(true_nodes[b,i,2:7].argmax().item(), pred_nodes[b,i], edges[b,i,j,0:4].argmax().item())
#         #             b_type, b_params = self.extract_params(true_nodes[b,j,2:7].argmax().item(), pred_nodes[b,j], edges[b,i,j,4:8].argmax().item())

#         #             temp = self.constraint_enforcement_loss(a_type, a_params, b_type, b_params, edges[b,i,j,8:].argmax().item())
#         #             if temp is not None:
#         #                 num = num + 1
#         #                 const_enf_loss = const_enf_loss + temp

#         node_loss = node_loss + 0.00001 * const_enf_loss # (const_enf_loss / num)

#         loss_dict["node loss"] = bce.item() + node_cross.item() + mse.item()
#         loss_dict["node construct"] = bce.item()
#         loss_dict["node type"] = node_cross.item()
#         loss_dict["node param"] = mse.item()
#         loss_dict["constraint enforcement"] = const_enf_loss.item()

#         return node_loss

#     def plot_loss(self, loss_dict : dict):
#         self.writer.add_scalar("Training/Node_Loss",      loss_dict["node loss"],      self.global_step)
#         self.writer.add_scalar("Training/Node_Construct", loss_dict["node construct"], self.global_step)
#         self.writer.add_scalar("Training/Node_Type",      loss_dict["node type"],      self.global_step)
#         self.writer.add_scalar("Training/Node_Param",     loss_dict["node param"],     self.global_step)
#         self.writer.add_scalar("Training/Constrain_Enforcment", loss_dict["constraint enforcement"], self.global_step)


#     @torch.no_grad()
#     def validate(self):
#         # self.validate_sampler.set_epoch(self.curr_epoch)
#         pbar = tqdm(self.validate_loader) if self.gpu_id == 0 else self.validate_loader
#         total_loss = 0
#         for nodes, params_mask, edges in pbar:
#             batch_size = nodes.size(0)
#             nodes = nodes.to(self.gpu_id)
#             params_mask = params_mask.to(self.gpu_id)

#             t = torch.randint(low = 1, high = self.model.module.max_timestep, size = (batch_size,)).to(self.gpu_id)
#             # noised_nodes, _ = self.model.module.noise_scheduler(nodes, t)
#             noised_nodes = self.model.module.noise_scheduler(nodes, t)

#             pred_nodes = self.model(noised_nodes, t)

#             loss_dict = {} # dictionary to record loss values 
#             loss = self.diffusion_loss(pred_nodes, nodes, params_mask, loss_dict, edges)

#             total_loss += loss
#             # assert loss.isfinite().all(), "Loss is non finite value"

#             if self.gpu_id == 0: pbar.set_description(f"Validating Epoch {self.curr_epoch}  ")
        
#         avg_loss = total_loss / len(pbar)
#         if self.gpu_id == 0:
#             self.save_checkpoint()
#             print("---Saved Model Checkpoint---")
        
        
#         if self.gpu_id == 0: self.writer.add_scalar("Validation/Loss", avg_loss, self.curr_epoch)

#         if self.gpu_id == 0:
#             fig, axes = plt.subplots(nrows = 2, ncols = 10, figsize=(40, 8))
#             seed = self.model.module.noise_scheduler.sample_latent(1)
#             sample = self.model.module.denoise(seed, axes)
#             self.writer.add_figure("Validation/Visualization", fig, self.curr_epoch)
#             plt.close(fig)

#     def train(self):
#         self.global_step = 0# 44944
#         self.curr_epoch = 0# 101

#         while (self.curr_epoch < self.num_epochs):
#             self.model.train()
#             self.train_epoch()
#             if self.curr_epoch % 5 == 0:
#                 self.model.eval()
#                 self.validate()
#             # self.model.eval()
#             # self.validate()
#             self.curr_epoch += 1
#             # self.scheduler.step()
#             # if self.gpu_id == 0: self.writer.add_scalar("LearningRate", self.scheduler.get_last_lr()[0])
#             # print("Learning Rate: ", self.scheduler.get_last_lr())
    
#     def save_checkpoint(self):
#         checkpoint = self.model.state_dict()
#         # ema_checkpoint = self.ema_model.state_dict()
#         optimizer_state = self.optimizer.state_dict()
#         epoch = self.curr_epoch
#         torch.save(
#             {
#                 "model": checkpoint,
#                 # "ema": ema_checkpoint,
#                 "optimizer": optimizer_state,
#                 "epoch": epoch
#             }, "checkpoint_"+self.experiment_string+".pth")

#     @staticmethod
#     def ddp_setup(rank, world_size):
#         '''
#         Args:
#             rank: Unique identifier of each process
#             world_size: The number of gpus (1 process per gpu)
#         '''
#         os.environ["MASTER_ADDR"] = "localhost"
#         os.environ["MASTER_PORT"] = "44444"
#         init_process_group(backend="nccl", rank=rank, world_size=world_size)
#         torch.cuda.set_device(rank)

# # %%
# def train_on_multiple_gpus(rank: int, 
#                            world_size: int,
#                            train_set: Subset,
#                            validate_set: Subset,
#                         #    learning_rate: float,
#                         #    batch_size: int,
#                            num_epochs: int, 
#                            experiment_string: str
#                           ):
#     MultiGPUTrainer.ddp_setup(rank, world_size)

#     #dataset = SketchDataset(root = "data/")
#     #train_set = Subset(dataset = dataset, indices = train_indices)
#     #validate_set = Subset(dataset = dataset, indices = validate_indices)

#     model = GD3PM(rank)
#     # if os.path.exists(f"best_model_checkpoint.pth"):
#     #     model.load_state_dict(torch.load(f"best_model_checkpoint.pth"))

#     trainer = MultiGPUTrainer(
#         model = model,
#         train_set = train_set,
#         validate_set = validate_set,
#         # learning_rate = learning_rate,
#         gpu_id = rank,
#         num_epochs = num_epochs,
#         experiment_string = experiment_string,
#         # batch_size = batch_size
#     )

#     trainer.train()
    
#     destroy_process_group()

# %%
import os
from typing import Sequence
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
# from torch.profiler import profile, record_function, ProfilerActivity
import torch
import torch.nn.functional as F
from torch import Tensor
from nodediff_model import GD3PM
# from loss import diffusion_loss
from dataset1 import SketchDataset
from torch.utils.data import DataLoader, Subset, random_split
os.chdir('SketchGraphs/')
import sketchgraphs.data as datalib
os.chdir('../')
from utils import ToNaive

# %%
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

torch.set_float32_matmul_precision('high')

# %%
class MultiGPUTrainer:
    def __init__(
            self,
            model: GD3PM,
            train_set: Subset,
            validate_set: Subset,
            # learning_rate: float,
            gpu_id: int,
            num_epochs: int,
            experiment_string: str,
            # batch_size: int
            ):
        model.device = gpu_id
        self.model = DDP(model.to(gpu_id), device_ids=[gpu_id])
        # decay = 0.9999
        # self.ema_model = torch.optim.swa_utils.AveragedModel(self.model.module, device = gpu_id, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay))
        self.batch_size = 256
        # self.num_grad_accum_steps = 4

        self.train_sampler = DistributedSampler(train_set)
        self.train_loader = DataLoader(dataset = train_set, 
                                       batch_size = self.batch_size, 
                                       shuffle = False, 
                                       pin_memory = True, 
                                       sampler = self.train_sampler
                                      )
        self.validate_sampler = DistributedSampler(validate_set)
        self.validate_loader = DataLoader(dataset = validate_set, 
                                          batch_size = self.batch_size, 
                                          shuffle = False, 
                                          pin_memory = True, 
                                          sampler = self.validate_sampler
                                         )
        
        self.learning_rate = 1e-4
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        # sched = lambda epoch : 1 if epoch >= 20 else 1e-2
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda = sched)

        # if os.path.exists(f"checkpoint_softgaussdiff_ddp_adam_dropout_48layers_512nodedim_256condim_8heads_best.pth"):
        #     map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_id}
        #     checkpoint = torch.load(f"checkpoint_softgaussdiff_ddp_adam_dropout_48layers_512nodedim_256condim_8heads_best.pth", map_location = map_location)
        #     self.model.load_state_dict(checkpoint["model"])
        #     self.optimizer.load_state_dict(checkpoint["optimizer"])

        # self.optimizer = ZeroRedundancyOptimizer(self.model.parameters(),
        #                                          optimizer_class = torch.optim.Adam,
        #                                          lr = learning_rate
        #                                         )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0 = 4, T_mult = 2, eta_min = 1e-5)
        
        self.gpu_id = gpu_id
        self.experiment_string = experiment_string
        self.writer = SummaryWriter(f'runs8/{self.experiment_string}')
        self.num_epochs = num_epochs

        self.global_step = 0
        self.curr_epoch = 0
        self.min_validation_loss = float('inf')

        self.record_freq = len(self.train_loader) // 5
        if self.record_freq == 0: self.record_freq = 1

        # self.timestep_distribution = torch.distributions.beta.Beta(torch.tensor([1.6]), torch.tensor([2.0]))
        # self.timestep_distribution = torch.where(torch.linspace(0, 1, self.model.module.max_timestep - 1) <= 0.5, 0.75, 0.25)
    
    def render_nodes(self, nodes, ax = None):
        SketchDataset.render_graph(ToNaive(nodes[...,1:]).cpu(), torch.zeros(size = (16, 16, 17)).cpu(), ax)
    
    def train_batch(self, nodes : Tensor, params_mask : Tensor) -> float:
        self.optimizer.zero_grad()

        batch_size = nodes.size(0)
        nodes = nodes.to(self.gpu_id) # .repeat(len(self.model.module.partitions), 1, 1) #.reshape(5 * batch_size, nodes.size(1), nodes.size(2))
        params_mask = params_mask.to(self.gpu_id) # .repeat(len(self.model.module.partitions), 1, 1) #.reshape(5 * batch_size, params_mask.size(1), params_mask.size(2))

        # t = torch.cat([torch.randint(low = low_step, high = high_step, size = (batch_size,), device = self.gpu_id) for (low_step, high_step) in self.model.module.partitions])
        # t1 = torch.randint(low = 1, high = self.model.module.max_timestep // 4, size = (batch_size // 3,)).to(self.gpu_id)
        # t2 = torch.randint(low = self.model.module.max_timestep // 4, high = self.model.module.max_timestep // 2, size = (batch_size // 3,)).to(self.gpu_id)
        # t3 = torch.randint(low = self.model.module.max_timestep // 2, high = self.model.module.max_timestep, size = (batch_size // 3,)).to(self.gpu_id)
        # t = torch.cat([t1,t2,t3])
        # t = torch.multinomial(self.timestep_distribution, batch_size, replacement = True).to(self.gpu_id) + 1
        t = torch.randint(low = 1, high = self.model.module.max_timestep, size = (batch_size,), device = self.gpu_id)
        # noised_nodes, _ = self.model.module.noise_scheduler(nodes, t)
        noised_nodes = self.model.module.noise_scheduler(nodes, t)

        pred_nodes = self.model(noised_nodes, t)

        loss_dict = {} # dictionary to record loss values
        # a_bar_t = self.model.module.noise_scheduler.a_bar[t]
        # scales = torch.clamp(a_bar_t / (1 - a_bar_t), max = 32).unsqueeze(1).unsqueeze(1)
        scales = torch.where(t <= 150, 32, 1)
        scales = scales.unsqueeze(1).unsqueeze(1)
        loss = self.diffusion_loss(pred_nodes, nodes, params_mask, loss_dict, scales)
        assert loss.isfinite().all(), "NaN was generated!"

        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
        self.optimizer.step()
        # self.ema_model.update_parameters(self.model.module)

        # if self.global_step % 250 == 249 and self.gpu_id == 0:
        #     p = 0 # batch_size * torch.randint(low = 0, high = len(self.model.module.partitions), size = (1,), device = 'cpu').item()
        #     fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(12, 4))
        #     fig.suptitle(f"Noised - Pred - True at time {t[p].item()}")
        #     self.render_nodes(noised_nodes[p].squeeze(0), axes[0])
        #     self.render_nodes(pred_nodes[p].squeeze(0), axes[1])
        #     self.render_nodes(nodes[p].squeeze(0), axes[2])
        #     self.writer.add_figure("Training/Visual", fig, self.global_step)
        #     plt.close()

        return loss_dict
    
    def train_epoch(self):
        self.train_sampler.set_epoch(self.curr_epoch)
        pbar = tqdm(self.train_loader) if self.gpu_id == 0 else self.train_loader
        for idx, targets in enumerate(pbar):
            nodes, params_mask = targets
            
            iter_loss_dict = self.train_batch(nodes, params_mask)

            self.global_step += 1
            # self.scheduler.step(self.curr_epoch + idx / iters)
            # if self.gpu_id == 0: self.writer.add_scalar("Learning Rate", self.scheduler.get_last_lr()[0], self.global_step)
            
            if self.gpu_id == 0: self.plot_loss(iter_loss_dict) # self.writer.add_scalar("Training Loss", iter_loss, self.global_step)

            iter_loss = iter_loss_dict["node loss"]
            if self.gpu_id == 0: pbar.set_description(f"Training Epoch {self.curr_epoch} Iter Loss: {iter_loss}")
    
    def diffusion_loss(self, pred_nodes : Tensor, true_nodes : Tensor, params_mask : Tensor, loss_dict : dict, scales : Tensor | int = 1) -> Tensor:
        '''Node Loss'''
        primitive_type_labels = true_nodes[:,:,2:7]    # batch_size x num_nodes (class index for each node)
        primitive_type_logits = pred_nodes[:,:,2:7] # .log()#.permute(0,2,1).contiguous() # batch_size x num_primitive_types x num_nodes
        
        node_cross = ((-primitive_type_logits.log() * primitive_type_labels)).mean() # (-primitive_type_labels * primitive_type_logits).mean()
        
        construct_type_labels = true_nodes[:,:,0:2]    # batch_size x num_nodes (class index for each node)
        construct_type_logits = pred_nodes[:,:,0:2]# .log()#.permute(0,2,1).contiguous() # batch_size x num_primitive_types x num_nodes
        
        bce = ((-construct_type_logits.log() * construct_type_labels)).mean() # (-construct_type_labels * construct_type_logits).mean()
        
        pred_params = pred_nodes[:,:,7:]
        target_params = true_nodes[:,:,7:]
        mse = (scales * params_mask * (pred_params - target_params) ** 2).sum() / params_mask.sum()

        node_loss = bce + node_cross + mse

        loss_dict["node loss"] = bce.item() + node_cross.item() + mse.item()
        loss_dict["node construct"] = bce.item()
        loss_dict["node type"] = node_cross.item()
        loss_dict["node param"] = mse.item()

        return node_loss

    def plot_loss(self, loss_dict : dict):
        self.writer.add_scalar("Training/Node_Loss",      loss_dict["node loss"],      self.global_step)
        self.writer.add_scalar("Training/Node_Construct", loss_dict["node construct"], self.global_step)
        self.writer.add_scalar("Training/Node_Type",      loss_dict["node type"],      self.global_step)
        self.writer.add_scalar("Training/Node_Param",     loss_dict["node param"],     self.global_step)

    @torch.no_grad()
    def validate(self):
        # self.validate_sampler.set_epoch(self.curr_epoch)
        pbar = tqdm(self.validate_loader) if self.gpu_id == 0 else self.validate_loader
        total_loss = 0
        for nodes, params_mask in pbar:
            batch_size = nodes.size(0)
            nodes = nodes.to(self.gpu_id)
            params_mask = params_mask.to(self.gpu_id)

            t = torch.randint(low = 1, high = self.model.module.max_timestep, size = (batch_size,)).to(self.gpu_id)
            # noised_nodes, _ = self.model.module.noise_scheduler(nodes, t)
            noised_nodes = self.model.module.noise_scheduler(nodes, t)

            pred_nodes = self.model(noised_nodes, t)

            loss_dict = {} # dictionary to record loss values 
            a_bar_t = self.model.module.noise_scheduler.a_bar[t]
            scales = torch.clamp(a_bar_t / (1 - a_bar_t), max = 16).unsqueeze(1).unsqueeze(1)
            loss = self.diffusion_loss(pred_nodes, nodes, params_mask, loss_dict, scales)

            total_loss += loss
            # assert loss.isfinite().all(), "Loss is non finite value"

            if self.gpu_id == 0: pbar.set_description(f"Validating Epoch {self.curr_epoch}  ")
        
        avg_loss = total_loss / len(pbar)
        if self.gpu_id == 0:
            self.save_checkpoint()
            print("---Saved Model Checkpoint---")
        
        
        if self.gpu_id == 0: self.writer.add_scalar("Validation/Loss", avg_loss, self.curr_epoch)

        if self.gpu_id == 0:
            fig, axes = plt.subplots(nrows = 2, ncols = 10, figsize=(40, 8))
            seed = self.model.module.noise_scheduler.sample_latent(1)
            sample = self.model.module.denoise(seed, axes)
            self.writer.add_figure("Validation/Visualization", fig, self.curr_epoch)
            plt.close(fig)

    def train(self):
        self.global_step = 0# 44944
        self.curr_epoch = 0# 101

        while (self.curr_epoch < self.num_epochs):
            self.model.train()
            self.train_epoch()
            # if self.curr_epoch % 5 == 0:
            self.model.eval()
            self.validate()
            # self.model.eval()
            # self.validate()
            self.curr_epoch += 1
            # self.scheduler.step()
            # if self.gpu_id == 0: self.writer.add_scalar("LearningRate", self.scheduler.get_last_lr()[0])
            # print("Learning Rate: ", self.scheduler.get_last_lr())
    
    def save_checkpoint(self):
        checkpoint = self.model.state_dict()
        # ema_checkpoint = self.ema_model.state_dict()
        optimizer_state = self.optimizer.state_dict()
        epoch = self.curr_epoch
        torch.save(
            {
                "model": checkpoint,
                # "ema": ema_checkpoint,
                "optimizer": optimizer_state,
                "epoch": epoch
            }, "checkpoint_"+self.experiment_string+".pth")

    @staticmethod
    def ddp_setup(rank, world_size):
        '''
        Args:
            rank: Unique identifier of each process
            world_size: The number of gpus (1 process per gpu)
        '''
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "44445"
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

# %%
def train_on_multiple_gpus(rank: int, 
                           world_size: int,
                           train_set: Subset,
                           validate_set: Subset,
                        #    learning_rate: float,
                        #    batch_size: int,
                           num_epochs: int, 
                           experiment_string: str
                          ):
    MultiGPUTrainer.ddp_setup(rank, world_size)

    #dataset = SketchDataset(root = "data/")
    #train_set = Subset(dataset = dataset, indices = train_indices)
    #validate_set = Subset(dataset = dataset, indices = validate_indices)

    model = GD3PM(rank)
    # if os.path.exists(f"best_model_checkpoint.pth"):
    #     model.load_state_dict(torch.load(f"best_model_checkpoint.pth"))

    trainer = MultiGPUTrainer(
        model = model,
        train_set = train_set,
        validate_set = validate_set,
        # learning_rate = learning_rate,
        gpu_id = rank,
        num_epochs = num_epochs,
        experiment_string = experiment_string,
        # batch_size = batch_size
    )

    trainer.train()
    
    destroy_process_group()


