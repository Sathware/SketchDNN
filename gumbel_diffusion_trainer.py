# %%
import enum
import os
import math
from typing import Sequence
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
# from torch.profiler import profile, record_function, ProfilerActivity
import torch
import torch.nn.functional as F
from torch import Tensor
from diffusion_model3 import GD3PM
# from loss import diffusion_loss
from dataset1 import SketchDataset
from torch.utils.data import DataLoader, Subset, random_split
from utils import ToNaive

os.chdir('SketchGraphs/')
import sketchgraphs.data as datalib
os.chdir('../')

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
        self.learning_rate = 1e-4
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        # if os.path.exists(f"checkpoint_softgaussdiff_ddp_adam_32layers_512nodedim_512edgedim_512condim.pth"):
        #     map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_id}
        #     checkpoint = torch.load(f"checkpoint_softgaussdiff_ddp_adam_32layers_512nodedim_512edgedim_512condim.pth", map_location = map_location)
        #     self.model.load_state_dict(checkpoint["model"], strict=False)
        #     self.optimizer.load_state_dict(checkpoint["optimizer"])

        #     # for name, p in model.named_parameters():
        #     #     if "edge" not in name:
        #     #         p.requires_grad = False

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
        
        # sched = lambda step : 1 if step >= 1_000 else 1e-2
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda = sched)

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

        self.timestep_distribution = torch.where(torch.linspace(0, 1, self.model.module.max_timestep - 1) <= 0.5, 0.8, 0.2)
    
    def render_nodes(self, nodes, ax = None):
        SketchDataset.render_graph(ToNaive(nodes[...,1:]).cpu(), torch.zeros(size = (24, 24, 17)).cpu(), ax)
    
    def train_batch(self, nodes : Tensor, edges : Tensor, params_mask : Tensor) -> float:
        self.optimizer.zero_grad()

        batch_size = nodes.size(0)
        nodes = nodes.to(self.gpu_id) # .repeat(len(self.model.module.partitions), 1, 1) #.reshape(5 * batch_size, nodes.size(1), nodes.size(2))
        edges = edges.to(self.gpu_id)
        params_mask = params_mask.to(self.gpu_id) # .repeat(len(self.model.module.partitions), 1, 1) #.reshape(5 * batch_size, params_mask.size(1), params_mask.size(2))

        # t = torch.cat([torch.randint(low = low_step, high = high_step, size = (batch_size,), device = self.gpu_id) for (low_step, high_step) in self.model.module.partitions])
        # t1 = torch.randint(low = 1, high = self.model.module.max_timestep // 4, size = (batch_size // 3,)).to(self.gpu_id)
        # t2 = torch.randint(low = self.model.module.max_timestep // 4, high = self.model.module.max_timestep // 2, size = (batch_size // 3,)).to(self.gpu_id)
        # t3 = torch.randint(low = self.model.module.max_timestep // 2, high = self.model.module.max_timestep, size = (batch_size // 3,)).to(self.gpu_id)
        # t = torch.cat([t1,t2,t3])
        t = torch.multinomial(self.timestep_distribution, batch_size, replacement = True).to(self.gpu_id) + 1
        noised_nodes, noised_edges = self.model.module.noise_scheduler(nodes, edges, t)

        pred_nodes, pred_edges = self.model(noised_nodes, noised_edges, t)
        # assert pred_edges.isfinite().all(), "NaN was generated by model!"

        loss_dict = {} # dictionary to record loss values
        a_bar_t = self.model.module.noise_scheduler.a_bar[t]
        scales = torch.clamp(a_bar_t / (1 - a_bar_t), max = 16).unsqueeze(1).unsqueeze(1)
        # scales = torch.where(t <= (self.model.module.max_timestep / 5), 4, 1)
        # scales = scales.unsqueeze(1).unsqueeze(1)
        loss = self.diffusion_loss(pred_nodes, nodes, pred_edges, edges, params_mask, loss_dict)
        assert loss.isfinite().all(), "NaN was generated in loss calculation!"

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        # self.scheduler.step()
        # self.ema_model.update_parameters(self.model.module)

        if self.global_step % 250 == 249 and self.gpu_id == 0:
            p = 0 # batch_size * torch.randint(low = 0, high = len(self.model.module.partitions), size = (1,), device = 'cpu').item()
            fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(12, 4))
            fig.suptitle(f"Noised - Pred - True at time {t[p].item()}")
            self.render_nodes(noised_nodes[p].squeeze(0), axes[0])
            self.render_nodes(pred_nodes[p].squeeze(0), axes[1])
            self.render_nodes(nodes[p].squeeze(0), axes[2])
            self.writer.add_figure("Training/Visual", fig, self.global_step)
            plt.close()

        return loss_dict
    
    def train_epoch(self):
        self.train_sampler.set_epoch(self.curr_epoch)
        pbar = tqdm(self.train_loader) if self.gpu_id == 0 else self.train_loader
        for idx, targets in enumerate(pbar):
            nodes, edges, params_mask = targets
            
            iter_loss_dict = self.train_batch(nodes, edges, params_mask)

            self.global_step += 1
            # self.scheduler.step(self.curr_epoch + idx / iters)
            # if self.gpu_id == 0: self.writer.add_scalar("Learning Rate", self.scheduler.get_last_lr()[0], self.global_step)
            
            if self.gpu_id == 0: self.plot_loss(iter_loss_dict) # self.writer.add_scalar("Training Loss", iter_loss, self.global_step)

            iter_loss = iter_loss_dict["node loss"]
            if self.gpu_id == 0: pbar.set_description(f"Training Epoch {self.curr_epoch} Iter Loss: {iter_loss}")

    # def constraint_loss(self, prim_a, prim_b, constraint):
    #     class PrimT(enum.IntEnum):
    #         Line = 0
    #         Circle = 1
    #         Arc = 2
    #         Point = 3
        
    #     class SubT(enum.IntEnum):
    #         Start = 0
    #         Center = 1
    #         End = 2

    #     class ConstrT(enum.IntEnum):
    #         Coincident = 0
    #         Horizontal = 1
    #         Vertical = 2
    #         Parallel = 3
    #         Perpendicular = 4
    #         Tangent = 5
    #         Midpoint = 6
    #         Equal = 7

    #     def extract_subnode(prim, prim_type, subtype):
    #         match subtype:
    #             case SubT.Start:
    #                 if prim_type == PrimT.Line: return prim[6:8]
    #                 if prim_type == PrimT.Arc: return prim[15:17]
    #             case SubT.Center:
    #                 if prim_type == PrimT.Line: return (prim[6:8] + prim[8:10]) / 2
    #                 if prim_type == PrimT.Arc:
    #                     dir = (prim[15:17] + prim[17:19]) / 2
    #                     r = prim[13:15]
    #                     dir = dir / torch.linalg.vector_norm(dir)
    #                     return prim[13:15]
    #             case SubT.End:
    #                 if prim_type == PrimT.Line: return prim[8:10]
    #                 if prim_type == PrimT.Arc: return prim[17:19]

    #     a_type, b_type = torch.argmax(prim_a[2:7]), torch.argmax(prim_b[2:7])
        
    #     if b_type < a_type: 
    #         prim_a, prim_b = prim_b, prim_a
    #         a_type, b_type = b_type, a_type

    #     match torch.argmax(constraint[8:17]):
    #         case ConstrT.Coincident:
    #             if [a_type, b_type] == [PrimT.Point, PrimT.Point]:
    #                 p1, p2 = prim_a[18,19] - prim_b[18,19]
    #                 return torch.linalg.vector_norm(p1 - p2)
                
    #             # elif [a_type, b_type] == [self.PrimT.Line, self.PrimT.Point]:
    #             #     m = torch.tensor([[0.0, -1.0], [1.0, 0.0]], device = prim_a.device).to(torch.double)
    #             #     b, a = prim_a[8,9], prim_a[6,7]
    #             #     dir = b - a
    #             #     p = prim_b[18,19]
    #             #     return torch.abs(p.dot(dir @ m) + b.dot(a @ m)) / torch.linalg.vector_norm(dir)
                
    #             # elif [a_type, b_type] == [self.PrimT.Arc, self.PrimT.Point]:
    #             #     c = prim_a[13, 14]
    #             #     r = prim_a[15].abs()
    #             #     p = prim_b[18,19]
    #             #     return torch.abs(torch.linalg.vector_norm(c - p) - r)
                
    #             # elif [a_type, b_type] == [self.PrimT.Circle, self.PrimT.Point]:
    #             #     c = prim_a[10, 11]
    #             #     r = prim_a[12]
    #             #     p = prim_b[18,19]
    #             #     return torch.abs(torch.linalg.vector_norm(c - p) - r)
                
    #             # elif [a_type, b_type] == [self.PrimT.Circle, self.PrimT.Circle]:
    #             #     c1, c2 = prim_a[10, 11], prim_b[10,11]
    #             #     r1, r2 = prim_a[12], prim_b[12]
    #             #     return torch.abs(torch.linalg.vector_norm(c1 - c2) - (r1+r2))
                
    #             # elif [a_type, b_type] == [self.PrimT.Circle, self.PrimT.Arc]:
    #             #     c1, c2 = prim_a[10, 11], prim_b[10,11]
    #             #     r1, r2 = prim_a[12], prim_b[12]
    #             #     return torch.abs(torch.linalg.vector_norm(c1 - c2) - (r1+r2))
                    
    #         # case self.ConstrT.Horizontal:
    #         #     # Horizontal
                    
    #         # case self.ConstrT.Vertical:
    #         #     # Vertical
                    
    #         # case self.ConstrT.Parallel:
    #         #     # Parallel
                    
    #         # case self.ConstrT.Perpendicular:
    #         #     # Perpendicular
                    
    #         # case self.ConstrT.Tangent:
    #         #     # Tangent
                    
    #         # case self.ConstrT.Midpoint:
    #         #     # Midpoint
                    
    #         # case self.ConstrT.Equal:
    #         #     # Equal
                    
    #         case _:
    #             # None
    #             return 0

    def diffusion_loss(self, pred_nodes : Tensor, true_nodes : Tensor, pred_edges : Tensor, true_edges : Tensor, params_mask : Tensor, loss_dict : dict, scales : Tensor | int = 1) -> Tensor:
        '''Edge Loss'''
        suba_labels = true_edges[...,0:4]    # batch_size x num_nodes (class index for each node)
        suba_logits = pred_edges[...,0:4] # .log()#.permute(0,2,1).contiguous() # batch_size x num_primitive_types x num_nodes
        suba_cross = ((-suba_logits.log() * suba_labels)).mean().clamp(0, 100)

        subb_labels = true_edges[...,4:8]    # batch_size x num_nodes (class index for each node)
        subb_logits = pred_edges[...,4:8] # .log()#.permute(0,2,1).contiguous() # batch_size x num_primitive_types x num_nodes
        subb_cross = ((-subb_logits.log() * subb_labels)).mean().clamp(0, 100)

        constraint_labels = true_edges[...,8:]    # batch_size x num_nodes (class index for each node)
        constraint_logits = pred_edges[...,8:] # .log()#.permute(0,2,1).contiguous() # batch_size x num_primitive_types x num_nodes
        constraint_cross = ((-constraint_logits.log() * constraint_labels)).mean().clamp(0, 100)

        edge_loss = suba_cross + subb_cross + constraint_cross
        
        '''Node Loss'''
        primitive_type_labels = true_nodes[:,:,2:7]    # batch_size x num_nodes (class index for each node)
        primitive_type_logits = pred_nodes[:,:,2:7] # .log()#.permute(0,2,1).contiguous() # batch_size x num_primitive_types x num_nodes
        node_cross = ((-primitive_type_logits.log() * primitive_type_labels)).mean().clamp(0, 100) # (-primitive_type_labels * primitive_type_logits).mean()
        
        construct_type_labels = true_nodes[:,:,0:2]    # batch_size x num_nodes (class index for each node)
        construct_type_logits = pred_nodes[:,:,0:2]# .log()#.permute(0,2,1).contiguous() # batch_size x num_primitive_types x num_nodes
        bce = ((-construct_type_logits.log() * construct_type_labels)).mean().clamp(0, 100) # (-construct_type_labels * construct_type_logits).mean()
        
        pred_params = pred_nodes[:,:,7:]
        target_params = true_nodes[:,:,7:]
        mse = ((pred_params - target_params) ** 2 * params_mask).sum() / params_mask.sum()

        node_loss = bce + node_cross + mse

        '''Constraint Enforcement Loss'''
        # batch_size, num_nodes, _ = true_nodes.shape
        # for b in range(batch_size):
        #     for i in range(num_nodes):
        #         for j in range(num_nodes):
        #             self.constraint_loss(pred_nodes[b, i, :], pred_nodes[b, j, :], true_edges[b, i, j, :])

        loss = 0.1 * edge_loss + node_loss

        loss_dict["node loss"] = bce.item() + node_cross.item() + mse.item()
        loss_dict["node construct"] = bce.item()
        loss_dict["node type"] = node_cross.item()
        loss_dict["node param"] = mse.item()
        loss_dict["edge loss"] = suba_cross.item() + subb_cross.item() + constraint_cross.item()
        loss_dict["edge suba"] = suba_cross.item()
        loss_dict["edge subb"] = subb_cross.item()
        loss_dict["edge type"] = constraint_cross.item()

        return loss

    def plot_loss(self, loss_dict : dict):
        self.writer.add_scalar("Training/Node_Loss",      loss_dict["node loss"],      self.global_step)
        self.writer.add_scalar("Training/Node_Construct", loss_dict["node construct"], self.global_step)
        self.writer.add_scalar("Training/Node_Type",      loss_dict["node type"],      self.global_step)
        self.writer.add_scalar("Training/Node_Param",     loss_dict["node param"],     self.global_step)
        self.writer.add_scalar("Training/Edge_Loss",      loss_dict["edge loss"],     self.global_step)
        self.writer.add_scalar("Training/Edge_SubB",      loss_dict["edge suba"],     self.global_step)
        self.writer.add_scalar("Training/Edge_SubA",      loss_dict["edge subb"],     self.global_step)
        self.writer.add_scalar("Training/Edge_Type",      loss_dict["edge type"],     self.global_step)

    @torch.no_grad()
    def validate(self):
        # self.validate_sampler.set_epoch(self.curr_epoch)
        pbar = tqdm(self.validate_loader) if self.gpu_id == 0 else self.validate_loader
        total_loss = 0
        for nodes, edges, params_mask in pbar:
            batch_size = nodes.size(0)
            nodes = nodes.to(self.gpu_id)
            edges = edges.to(self.gpu_id)
            params_mask = params_mask.to(self.gpu_id)

            t = torch.randint(low = 1, high = self.model.module.max_timestep, size = (batch_size,)).to(self.gpu_id)
            noised_nodes, noised_edges = self.model.module.noise_scheduler(nodes, edges, t)

            pred_nodes, pred_edges = self.model(noised_nodes, noised_edges, t)

            loss_dict = {} # dictionary to record loss values 
            loss = self.diffusion_loss(pred_nodes, nodes, pred_edges, edges, params_mask, loss_dict)

            total_loss += loss_dict["node loss"] + loss_dict["edge loss"]
            # assert loss.isfinite().all(), "Loss is non finite value"

            if self.gpu_id == 0: pbar.set_description(f"Validating Epoch {self.curr_epoch}  ")
        
        avg_loss = total_loss / len(pbar)
        if self.gpu_id == 0:
            self.save_checkpoint()
            print("---Saved Model Checkpoint---")
        
        
        if self.gpu_id == 0: self.writer.add_scalar("Validation/Loss", avg_loss, self.curr_epoch)

        if self.gpu_id == 0:
            fig, axes = plt.subplots(nrows = 2, ncols = 10, figsize=(40, 8))
            seed_node, seed_edge = self.model.module.noise_scheduler.sample_latent(1)
            sampled_nodes, sampled_edges = self.model.module.denoise(seed_node, seed_edge, axes)
            self.writer.add_figure("Validation/Visualization", fig, self.curr_epoch)
            plt.close(fig)

    def train(self):
        self.global_step = 0
        self.curr_epoch = 0

        while (self.curr_epoch < self.num_epochs):
            self.model.train()
            self.train_epoch()
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














# # %%
# import os
# from tqdm import tqdm
# from matplotlib import pyplot as plt
# from torch.utils.tensorboard.writer import SummaryWriter
# import torch
# from gumbel_diffusion_model import GD3PM, diffusion_loss
# from dataset1 import SketchDataset
# from torch.utils.data import DataLoader, Subset
# os.chdir('SketchGraphs/')
# import sketchgraphs.data as datalib
# os.chdir('../')

# # %%
# import torch.multiprocessing as mp
# from torch.utils.data.distributed import DistributedSampler
# from torch.distributed.optim import ZeroRedundancyOptimizer
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group

# # %%
# class MultiGPUTrainer:
#     def __init__(
#             self,
#             model: GD3PM,
#             train_set: Subset,
#             validate_set: Subset,
#             learning_rate: float,
#             gpu_id: int,
#             num_epochs: int,
#             experiment_string: str,
#             batch_size: int
#             ):
#         model.device = gpu_id
#         self.model = DDP(model.to(gpu_id), device_ids=[gpu_id])

#         # checkpoint_path = f"model_checkpoint_gumbeldiff_ddp_Adam_24crossattnblocks16heads256class256param256edge256time_betadist1,2_cont_nll_loss_mse_loss.pth"
#         # if os.path.exists(checkpoint_path):
#         #     map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_id}
#         #     self.model.load_state_dict(torch.load(checkpoint_path, map_location = map_location), strict = False)

#         self.train_sampler = DistributedSampler(train_set, drop_last = True)
#         self.train_loader = DataLoader(dataset = train_set, 
#                                        batch_size = batch_size, 
#                                        shuffle = False, 
#                                        pin_memory = True, 
#                                        sampler = self.train_sampler
#                                       )
#         self.validate_sampler = DistributedSampler(validate_set)
#         self.validate_loader = DataLoader(dataset = validate_set, 
#                                           batch_size = batch_size, 
#                                           shuffle = False, 
#                                           pin_memory = True, 
#                                           sampler = self.validate_sampler
#                                          )
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
#         self.timestep_distribution = torch.distributions.beta.Beta(torch.tensor([1.0]), torch.tensor([2.0]))
#         # self.optimizer = ZeroRedundancyOptimizer(self.model.parameters(),
#         #                                          optimizer_class = torch.optim.Adam,
#         #                                          lr = learning_rate
#         #                                         )
#         # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0 = 4, T_mult = 2, eta_min = 1e-5)
        
#         self.gpu_id = gpu_id
#         self.experiment_string = experiment_string
#         self.writer = SummaryWriter(f'runs3/{self.experiment_string}')
#         self.num_epochs = num_epochs

#         self.global_step = 0
#         self.curr_epoch = 0
#         self.min_validation_loss = float('inf')

#         self.record_freq = len(self.train_loader) // 50
#         if self.record_freq == 0: self.record_freq = 1
        
#         self.batch_size = batch_size

    
#     def train_batch(self, nodes : torch.Tensor, edges : torch.Tensor, params_mask : torch.Tensor) -> float:
#         self.optimizer.zero_grad()

#         nodes = nodes.to(self.gpu_id)
#         edges = edges.to(self.gpu_id)
#         params_mask = params_mask.to(self.gpu_id)

#         batch_size = nodes.size(0)
#         t = torch.randint(low = 1, high = self.model.module.max_timestep - 10, size = (batch_size,), device = self.gpu_id)
#         t = (self.timestep_distribution.sample((batch_size,)) * (self.model.module.max_timestep)).squeeze().int().to(self.gpu_id)
#         noised_nodes, noised_edges, true_node_noise, true_edge_noise = self.model.module.noise_scheduler(nodes, edges, t)
#         pred_node_noise, pred_edge_noise = self.model(noised_nodes, noised_edges, t)

#         loss_dict = {} # dictionary to record loss values
#         none_node_mask = (nodes[...,5] > 0).unsqueeze(-1) # true primitive is classified as none type
#         none_edge_mask = (edges[...,16] > 0).unsqueeze(-1) # true constraint is classified as none type
#         loss = diffusion_loss(pred_node_noise, pred_edge_noise, true_node_noise, true_edge_noise, params_mask, none_node_mask, none_edge_mask, 
#                               nodes, noised_nodes, self.model.module.noise_scheduler, t, loss_dict)

#         # assert loss.isfinite().all(), "Loss is non finite value"

#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), 1.0)
#         self.optimizer.step()

#         return loss_dict
    
#     def train_epoch(self):
#         self.train_sampler.set_epoch(self.curr_epoch)
#         iters = len(self.train_loader)
#         pbar = tqdm(self.train_loader) if self.gpu_id == 0 else self.train_loader
#         for idx, targets in enumerate(pbar):
#             nodes, edges, params_mask = targets
            
#             iter_loss_dict = self.train_batch(nodes, edges, params_mask)

#             self.global_step += 1
#             # self.scheduler.step(self.curr_epoch + idx / iters)
#             # if self.gpu_id == 0: self.writer.add_scalar("Learning Rate", self.scheduler.get_last_lr()[0], self.global_step)

#             if (self.global_step % self.record_freq == 0):
#                 if self.gpu_id == 0: self.plot_loss(iter_loss_dict) # self.writer.add_scalar("Training Loss", iter_loss, self.global_step)

#             iter_loss = iter_loss_dict["total loss"]
#             if self.gpu_id == 0: pbar.set_description(f"Training Epoch {self.curr_epoch} Iter Loss: {iter_loss}")
    
#     def plot_loss(self, loss_dict):
#         self.writer.add_scalar("Training/Total_Loss", loss_dict["total loss"],            self.global_step)
#         self.writer.add_scalar("Training/Node_Loss",  loss_dict["node loss"],             self.global_step)
#         self.writer.add_scalar("Training/Node_Construct",   loss_dict["node isconstruct loss"], self.global_step)
#         self.writer.add_scalar("Training/Node_Type", loss_dict["node type loss"],        self.global_step)
#         self.writer.add_scalar("Training/Node_Param",   loss_dict["node parameter loss"],   self.global_step)
#         self.writer.add_scalar("Training/Edge_Loss",  loss_dict["edge loss"],             self.global_step)
#         self.writer.add_scalar("Training/Edge_sub_a", loss_dict["edge sub_a loss"],       self.global_step)
#         self.writer.add_scalar("Training/Edge_sub_b", loss_dict["edge sub_b loss"],       self.global_step)
#         self.writer.add_scalar("Training/Edge_Type", loss_dict["edge type loss"],        self.global_step)
#         # self.writer.add_scalar("Training/KLD", loss_dict["kld"], self.global_step)
#         #self.writer.add_scalar("Training/Posterior_Collapse_Regularization", loss_dict["postcollapse_reg"], self.global_step)

#     @torch.no_grad()
#     def validate(self):
#         # self.validate_sampler.set_epoch(self.curr_epoch)
#         pbar = tqdm(self.validate_loader) if self.gpu_id == 0 else self.validate_loader
#         total_loss = 0
#         for nodes, edges, params_mask in pbar:
#             nodes = nodes.to(self.gpu_id)
#             edges = edges.to(self.gpu_id)
#             params_mask = params_mask.to(self.gpu_id)

#             batch_size = nodes.size(0)
#             t = torch.randint(low = 1, high = self.model.module.max_timestep - 10, size = (batch_size,), device = self.gpu_id)
#             noised_nodes, noised_edges, true_node_noise, true_edge_noise = self.model.module.noise_scheduler(nodes, edges, t)
#             pred_node_noise, pred_edge_noise = self.model(noised_nodes, noised_edges, t)
            
#             loss_dict = {}
#             none_node_mask = (nodes[...,5] > 0).unsqueeze(-1) # true primitive is classified as none type
#             none_edge_mask = (edges[...,16] > 0).unsqueeze(-1) # true constraint is classified as none type
#             loss = diffusion_loss(pred_node_noise, pred_edge_noise, true_node_noise, true_edge_noise, params_mask, none_node_mask, none_edge_mask,
#                                   nodes, noised_nodes, self.model.module.noise_scheduler, t, loss_dict)

#             total_loss += loss

#             assert loss.isfinite().all(), "Loss is non finite value"

#             if self.gpu_id == 0: pbar.set_description(f"Validating Epoch {self.curr_epoch}  ")
        
#         avg_loss = total_loss / len(pbar)
#         if avg_loss < self.min_validation_loss:
#             self.min_validation_loss = avg_loss
#             if self.gpu_id == 0: 
#                 self.writer.add_scalar("Validation/Loss", avg_loss, self.curr_epoch)
#                 self.save_checkpoint()
#                 print("---Saved Model Checkpoint---")
        
#         if self.gpu_id == 0:
#             # Render the actual CAD Sketch
#             noised_nodes, noised_edges = self.model.module.noise(nodes[0:4], edges[0:4])
#             denoised_nodes, denoised_edges = self.model.module.denoise(noised_nodes, noised_edges)
#             fig, axes = plt.subplots(nrows = 4, ncols = 2, figsize=(8, 16))
#             fig.suptitle(f"Target (left) vs Preds (right) for epoch {self.curr_epoch}")
#             for i in range(4):
#                 target_sketch = SketchDataset.preds_to_sketch(nodes[i].cpu(), edges[i].cpu())
#                 pred_sketch = SketchDataset.preds_to_sketch(denoised_nodes[i].cpu(), denoised_edges[i].cpu())
                
#                 datalib.render_sketch(target_sketch, axes[i, 0])
#                 # SketchDataset.superimpose_constraints(target_sketch, axes[i, 0])
#                 datalib.render_sketch(pred_sketch, axes[i, 1])
#                 # SketchDataset.superimpose_constraints(pred_sketch, axes[i, 1])
            
#             self.writer.add_figure(f"Validation/Epoch_Result_Visualization", fig, self.curr_epoch)
#             plt.close()

#             # # Render the graph visualization of the Sketch
#             # fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(16, 8))
#             # fig.suptitle(f"Target (left) vs Preds (right) for epoch {self.curr_epoch}")
#             # self.visualize_graph(nodes[0].cpu(), edges[0].cpu(), "target_graph")
#             # self.visualize_graph(pred_nodes[0].cpu(), pred_edges[0].cpu(), "pred_graph")
#             # axes[0].imshow(plt.imread("target_graph.png"))
#             # axes[1].imshow(plt.imread("pred_graph.png"))
#             # self.writer.add_figure(f"Validation/Epoch_Graph_Visualization", fig, self.curr_epoch)
#             # plt.close()

    
#     def visualize_graph(self, nodes, edges, filename):
#         sketch = SketchDataset.preds_to_sketch(nodes, edges)
#         seq = datalib.sketch_to_sequence(sketch)
#         graph = datalib.pgvgraph_from_sequence(seq)
#         datalib.render_graph(graph, f"{filename}.png")

#     def train(self):
#         self.global_step = 0
#         self.curr_epoch = 0

#         while (self.curr_epoch < self.num_epochs):
#             self.model.train()
#             self.train_epoch()
#             if self.curr_epoch % 5 == 0:
#                 self.model.eval()
#                 self.validate()
#             self.curr_epoch += 1
#             # print("Learning Rate: ", self.scheduler.get_last_lr())
    
#     def save_checkpoint(self):
#         checkpoint = self.model.state_dict()
#         torch.save(checkpoint, "model_checkpoint_"+self.experiment_string+".pth")

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
#                            learning_rate: float,
#                            batch_size: int,
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
#         learning_rate = learning_rate,
#         gpu_id = rank,
#         num_epochs = num_epochs,
#         experiment_string = experiment_string,
#         batch_size = batch_size
#     )

#     trainer.train()
    
#     destroy_process_group()


