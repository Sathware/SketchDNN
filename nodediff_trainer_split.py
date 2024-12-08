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
            world_size: int
            ):
        model.device = gpu_id
        self.world_size = world_size
        self.model = DDP(model.to(gpu_id), device_ids=[gpu_id], find_unused_parameters=True) #, device_ids=[gpu_id], find_unused_parameters=True
        # decay = 0.9999
        # self.ema_model = torch.optim.swa_utils.AveragedModel(self.model.module, device = gpu_id, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay))
        self.batch_size = 512
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
        sched = lambda epoch : 1 # if epoch >= 1 else 1e-2
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda = sched)

        if os.path.exists(f"checkpoint_nodediff_ddp_adam_5*24layers_1536dim_split.pth"):
            map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_id}
            checkpoint = torch.load(f"checkpoint_nodediff_ddp_adam_5*24layers_1536dim_split.pth", map_location = map_location)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        # self.optimizer = ZeroRedundancyOptimizer(self.model.parameters(),
        #                                          optimizer_class = torch.optim.Adam,
        #                                          lr = learning_rate
        #                                         )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0 = 4, T_mult = 2, eta_min = 1e-5)
        
        self.gpu_id = gpu_id
        self.experiment_string = experiment_string
        self.writer = SummaryWriter(f'runs7/{self.experiment_string}')
        self.num_epochs = num_epochs

        self.global_step = 0
        self.curr_epoch = 0
        self.min_validation_loss = float('inf')

        self.record_freq = len(self.train_loader) // 5
        if self.record_freq == 0: self.record_freq = 1

        # self.timestep_distribution = torch.distributions.beta.Beta(torch.tensor([1.6]), torch.tensor([2.0]))
        self.timestep_distribution = (1 - torch.linspace(-1, 1, 999) ** 4).clamp(2/5)

        self.curr_training_model_idx = 0
        self.freeze_non_training_models(self.curr_training_model_idx)

    def freeze_non_training_models(self, model_to_train_idx : int):
        # destroy_process_group()
        # self.model = self.model.module # Destroy DDP Instance

        # Send all parameters to CPU
        # self.model.to('cpu')
        # Freeze all models
        for model in self.model.module.architectures:
            # model.device = 'cpu'
            for param in model.parameters():
                param.requires_grad = False

        # Unfreeze current model
        for param in self.model.module.architectures[model_to_train_idx].parameters():
            param.requires_grad = True
        # Send current model to GPU
        # self.model.architectures[model_to_train_idx].device = self.gpu_id
        # self.model.architectures[model_to_train_idx].to(self.gpu_id)
        # init_process_group(backend="nccl", rank=self.gpu_id, world_size=self.world_size)
        # self.model = DDP(self.model) # Reconstruct DDP Instance
    
    def render_nodes(self, nodes, ax = None):
        SketchDataset.render_graph(nodes[...,1:].cpu(), torch.zeros(size = (24, 24, 17)).cpu(), ax)
    
    def train_batch(self, nodes : Tensor, params_mask : Tensor) -> float:
        nodes = nodes.to(self.gpu_id)
        params_mask = params_mask.to(self.gpu_id)

        batch_size = nodes.size(0)
        low_step, high_step = self.model.module.partitions[self.curr_training_model_idx]
        t = torch.randint(low = low_step, high = high_step, size = (batch_size,)).to(self.gpu_id)
        # t = torch.randint(low = 1, high = self.model.module.max_timestep, size = (batch_size,)).to(self.gpu_id)
        # t = torch.multinomial(self.timestep_distribution, batch_size, replacement = True).to(self.gpu_id) + 1
        noised_nodes, _ = self.model.module.noise_scheduler(nodes, t)

        pred_nodes = self.model(noised_nodes, t)

        loss_dict = {} # dictionary to record loss values
        # scales = torch.where(t <= (self.model.module.max_timestep / 5), 4, 1)
        # scales = scales.unsqueeze(1).unsqueeze(1)
        loss = self.diffusion_loss(pred_nodes, nodes, params_mask, loss_dict)

        if self.global_step % 250 == 249 and self.gpu_id == 0:
            fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(12, 4))
            fig.suptitle(f"Noised - Pred - True at time {t[0].item()}")
            self.render_nodes(noised_nodes[0].squeeze(0), axes[0])
            self.render_nodes(pred_nodes[0].squeeze(0), axes[1])
            self.render_nodes(nodes[0].squeeze(0), axes[2])
            self.writer.add_figure("Training/Visual", fig, self.global_step)
            plt.close()

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        # self.ema_model.update_parameters(self.model.module)

        del loss
        del pred_nodes

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
        
        node_cross = ((-primitive_type_logits.log() * primitive_type_labels)).mean().clamp(0, 100) # (-primitive_type_labels * primitive_type_logits).mean()
        
        construct_type_labels = true_nodes[:,:,0:2]    # batch_size x num_nodes (class index for each node)
        construct_type_logits = pred_nodes[:,:,0:2]# .log()#.permute(0,2,1).contiguous() # batch_size x num_primitive_types x num_nodes
        
        bce = ((-construct_type_logits.log() * construct_type_labels)).mean().clamp(0, 100) # (-construct_type_labels * construct_type_logits).mean()
        
        pred_params = pred_nodes[:,:,7:]
        target_params = true_nodes[:,:,7:]
        mse = ((pred_params - target_params) ** 2 * params_mask).sum() / params_mask.sum()

        node_loss = bce + node_cross + 12 * mse

        loss_dict["node loss"] = bce.item() + node_cross.item() + mse.item()
        loss_dict["node construct"] = bce.item()
        loss_dict["node type"] = node_cross.item()
        loss_dict["node param"] = mse.item()

        return node_loss

    def plot_loss(self, loss_dict : dict):
        self.writer.add_scalar(f"Training/Node_Loss{self.curr_training_model_idx}",      loss_dict["node loss"],      self.global_step)
        self.writer.add_scalar(f"Training/Node_Construct{self.curr_training_model_idx}", loss_dict["node construct"], self.global_step)
        self.writer.add_scalar(f"Training/Node_Type{self.curr_training_model_idx}",      loss_dict["node type"],      self.global_step)
        self.writer.add_scalar(f"Training/Node_Param{self.curr_training_model_idx}",     loss_dict["node param"],     self.global_step)

    @torch.no_grad()
    def validate(self):
        # self.validate_sampler.set_epoch(self.curr_epoch)
        pbar = tqdm(self.validate_loader) if self.gpu_id == 0 else self.validate_loader
        total_loss = 0
        for nodes, params_mask in pbar:
            nodes = nodes.to(self.gpu_id)
            params_mask = params_mask.to(self.gpu_id)

            batch_size = nodes.size(0)
            t = torch.randint(low = 1, high = self.model.module.max_timestep, size = (batch_size,)).to(self.gpu_id)
            noised_nodes, _ = self.model.module.noise_scheduler(nodes, t)

            pred_nodes = self.model(noised_nodes, t)

            loss_dict = {} # dictionary to record loss values 
            loss = self.diffusion_loss(pred_nodes, nodes, params_mask, loss_dict)

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
        self.global_step = 71162
        self.curr_epoch = 160

        while (self.curr_epoch < self.num_epochs):
            # self.model.train()
            self.train_epoch()
            if self.curr_epoch % 5 == 0:
                self.model.eval()
                self.model.to(self.gpu_id)
                self.validate()
                self.model.train()
                self.freeze_non_training_models(self.curr_training_model_idx)
            
            self.curr_training_model_idx = (self.curr_training_model_idx + 1) % len(self.model.module.architectures)
            self.freeze_non_training_models(self.curr_training_model_idx)
            # self.model.eval()
            # self.validate()
            self.curr_epoch += 1
            self.scheduler.step()
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
        world_size = world_size
    )

    trainer.train()
    
    destroy_process_group()


