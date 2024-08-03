# %%
import os
from typing import Sequence
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
# from torch.profiler import profile, record_function, ProfilerActivity
import torch
from denoise_block import GD3PM
from loss import diffusion_loss
from dataset1 import SketchDataset
from torch.utils.data import DataLoader, Subset, random_split, TensorDataset
os.chdir('SketchGraphs/')
import sketchgraphs.data as datalib
os.chdir('../')

# %%
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, d_model, device):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model, device = device)
        self.fc2 = nn.Linear(d_model, d_model, device = device)
        self.norm1 = nn.LayerNorm(d_model, device = device)
        self.norm2 = nn.LayerNorm(d_model, device = device)
        
    def forward(self, x):
        identity = x
        out = F.relu(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        return out + identity

class Sampler(nn.Module):
    def __init__(self, device):
        super(Sampler, self).__init__()
        d = 1024
        n_residual_layers=32
        self.max_timestep = 1000
        self.device = device
        
        self.fc_in = nn.Linear(d * 2, d, device = device)
        self.residual_layers = nn.ModuleList([ResidualBlock(d, device) for _ in range(n_residual_layers)])
        self.fc_out = nn.Linear(d, d, device = device)
        self.norm = nn.LayerNorm(d, device = device)
    
    def forward(self, x, t):
        # Concatenate time step with input
        t_embedding = self.time_embedding(t, x.size(-1))
        x = torch.cat([x, t_embedding], dim = -1)
        
        # Initial linear transformation
        x = F.relu(self.fc_in(x))
        
        # Pass through residual and attention layers
        for res_block in self.residual_layers:
            x = res_block(x)
        
        # Output linear transformation
        x = self.fc_out(self.norm(x))
        return x
    
    def time_embedding(self, t, dim):
        # Create a time embedding
        half_dim = dim // 2
        emb = torch.log(torch.tensor([10000]).to(self.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32).to(self.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# %%
class MultiGPUTrainer:
    def __init__(
            self,
            model: Sampler,
            train_set: TensorDataset,
            learning_rate: float,
            gpu_id: int,
            num_epochs: int,
            experiment_string: str,
            batch_size: int
            ):
        model.device = gpu_id
        self.model = DDP(model.to(gpu_id), device_ids=[gpu_id])
        # if os.path.exists(f"model_checkpoint_sampler_ddp_Adam_depth_32.pth"):
        #     map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_id}
        #     self.model.load_state_dict(torch.load(f"model_checkpoint_sampler_ddp_Adam_depth_32.pth", map_location = map_location))

        self.train_sampler = DistributedSampler(train_set, drop_last = True)
        self.train_loader = DataLoader(dataset = train_set, 
                                       batch_size = batch_size, 
                                       shuffle = False, 
                                       pin_memory = True, 
                                       sampler = self.train_sampler
                                      )
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        # self.optimizer = ZeroRedundancyOptimizer(self.model.parameters(),
        #                                          optimizer_class = torch.optim.Adam,
        #                                          lr = learning_rate
        #                                         )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0 = 4, T_mult = 2, eta_min = 1e-5)
        
        self.gpu_id = gpu_id
        self.experiment_string = experiment_string
        self.writer = SummaryWriter(f'runs3/{self.experiment_string}')
        self.num_epochs = num_epochs

        self.global_step = 0
        self.curr_epoch = 0
        self.min_validation_loss = float('inf')

        self.record_freq = len(self.train_loader) // 5
        if self.record_freq == 0: self.record_freq = 1
        
        self.batch_size = batch_size

        self.T = model.max_timestep
        self.a_bar = torch.cos(0.5 * torch.pi * (torch.arange(0.0, 1.0, 1/(self.T + 1)) + .008) / 1.008) ** 2
        self.a_bar = self.a_bar / self.a_bar[0]
        self.a_bar.to(gpu_id)

        self.a = self.a_bar[1:] / self.a_bar[:-1]
        self.a = torch.cat([self.a, torch.tensor([0.0])])

        self.sqrt_a = self.a.sqrt()
        self.sqrt_a_bar = self.a_bar.sqrt().to(gpu_id)
        self.sqrt_b_bar = (1 - self.a_bar).sqrt().to(gpu_id)

    
    def train_batch(self, mean : torch.Tensor, sdev : torch.Tensor) -> float:
        self.optimizer.zero_grad()

        mean = mean.to(self.gpu_id)
        sdev = sdev.to(self.gpu_id)

        latent = mean + sdev * torch.randn_like(mean)

        t = torch.randint(low = 1, high = self.T, size = (latent.size(0),)).to(self.gpu_id)
        latent_noise = torch.randn_like(latent).to(self.gpu_id)

        noised_latent = self.sqrt_a_bar[t,None] * latent + self.sqrt_b_bar[t,None] * latent_noise

        denoised_latent = self.model(noised_latent, t)

        loss = torch.nn.functional.mse_loss(denoised_latent, latent)
        if self.gpu_id == 0: 
            self.writer.add_scalar("MSE Loss", loss.item(), self.global_step)

        loss.backward()
        self.optimizer.step()

    def train_epoch(self):
        self.train_sampler.set_epoch(self.curr_epoch)
        iters = len(self.train_loader)
        pbar = tqdm(self.train_loader) if self.gpu_id == 0 else self.train_loader
        for idx, targets in enumerate(pbar):
            mean, sdev = targets
            
            self.train_batch(mean, sdev)

            self.global_step += 1

    def train(self):
        self.global_step = 0
        self.curr_epoch = 0

        while (self.curr_epoch < self.num_epochs):
            self.model.train()
            self.train_epoch()
            self.curr_epoch += 1
            if self.curr_epoch % 5 == 4 and self.gpu_id == 0:
                self.save_checkpoint()
            if self.gpu_id == 0:
                print("Epoch ", self.curr_epoch)
            # print("Learning Rate: ", self.scheduler.get_last_lr())
    
    def save_checkpoint(self):
        checkpoint = self.model.state_dict()
        torch.save(checkpoint, "model_checkpoint_"+self.experiment_string+".pth")

    @staticmethod
    def ddp_setup(rank, world_size):
        '''
        Args:
            rank: Unique identifier of each process
            world_size: The number of gpus (1 process per gpu)
        '''
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "44444"
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

# %%
def train_on_multiple_gpus(rank: int, 
                           world_size: int,
                           train_set: TensorDataset,
                           learning_rate: float,
                           batch_size: int,
                           num_epochs: int, 
                           experiment_string: str
                          ):
    MultiGPUTrainer.ddp_setup(rank, world_size)

    model = Sampler(rank)
    # if os.path.exists(f"best_model_checkpoint.pth"):
    #     model.load_state_dict(torch.load(f"best_model_checkpoint.pth"))

    trainer = MultiGPUTrainer(
        model = model,
        train_set = train_set,
        learning_rate = learning_rate,
        gpu_id = rank,
        num_epochs = num_epochs,
        experiment_string = experiment_string,
        batch_size = batch_size
    )

    trainer.train()
    
    destroy_process_group()


