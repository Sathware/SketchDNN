import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
import torch
from torch import Tensor
from diffusion_model import GD3PM
import config
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

torch.backends.cudnn.conv.fp32_precision = 'tf32'

class MultiGPUTrainer:
    def __init__(
            self,
            model: GD3PM,
            train_set: Subset,
            validate_set: Subset,
            learning_rate: float,
            batch_size: int,
            num_epochs: int,
            experiment_string: str,
            gpu_id: int
            ):
        self.gpu_id = gpu_id
        model.device = gpu_id
        self.model = DDP(model.to(gpu_id), device_ids=[gpu_id])
        self.batch_size = batch_size

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
        
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        
        self.experiment_string = experiment_string
        self.save_location = f'{config.TENSORBOARD_RUNS_PATH}/{self.experiment_string}'
        self.writer = SummaryWriter(self.save_location + "/logs")
        os.makedirs(name = self.save_location + f"/checkpoints", exist_ok = True) # directory for saving checkpoints

        self.num_epochs = num_epochs
        self.global_step = 0
        self.curr_epoch = 0
        self.min_validation_loss = float('inf')
    
    def train_batch(self, nodes : Tensor, params_mask : Tensor) -> float:
        self.optimizer.zero_grad()

        nodes = nodes.to(self.gpu_id)
        params_mask = params_mask.to(self.gpu_id)

        batch_size = nodes.size(0)
        t = torch.randint(low = 1, high = self.model.module.max_timestep, size = (batch_size,), device = self.gpu_id)
        noised_nodes = self.model.module.noise_scheduler(nodes, t)

        pred_nodes = self.model(noised_nodes, t)

        scales = torch.where(t <= 150, 16, 1)
        scales = scales.unsqueeze(1).unsqueeze(1)
        loss, loss_dict = self.diffusion_loss(pred_nodes, nodes, params_mask, scales)
        assert loss.isfinite().all(), "NaN was generated!"

        loss.backward()
        self.optimizer.step()

        return loss_dict
    
    def train_epoch(self):
        self.train_sampler.set_epoch(self.curr_epoch)
        pbar = tqdm(self.train_loader) if self.gpu_id == 0 else self.train_loader
        for idx, targets in enumerate(pbar):
            nodes, params_mask = targets
            
            iter_loss_dict = self.train_batch(nodes, params_mask)

            self.global_step += 1
            
            if self.gpu_id == 0: self.plot_loss(iter_loss_dict)

            iter_loss = iter_loss_dict["node loss"]
            if self.gpu_id == 0: pbar.set_description(f"Training Epoch {self.curr_epoch} Iter Loss: {iter_loss}")
    
    def diffusion_loss(self, pred_nodes : Tensor, true_nodes : Tensor, params_mask : Tensor, scales : Tensor | int = 1) -> Tensor:
        '''Primitive Loss'''
        primitive_type_label = true_nodes[:,:,2:7] # batch_size x num_nodes (class index for each node)
        primitive_type_preds = pred_nodes[:,:,2:7] # batch_size x num_primitive_types x num_nodes
        
        node_cross = ((-primitive_type_preds.log() * primitive_type_label)).mean() # (-primitive_type_labels * primitive_type_logits).mean()
        
        construct_type_label = true_nodes[:,:,0:2] # batch_size x num_nodes (class index for each node)
        construct_type_preds = pred_nodes[:,:,0:2] # batch_size x num_primitive_types x num_nodes
        
        bce = ((-construct_type_preds.log() * construct_type_label)).mean() # (-construct_type_labels * construct_type_logits).mean()
        
        primitive_params_true = true_nodes[:,:,7:]
        primitive_params_pred = pred_nodes[:,:,7:]
        mse = (scales * params_mask * (primitive_params_pred - primitive_params_true) ** 2).sum() / params_mask.sum()

        node_loss = bce + node_cross + mse

        loss_dict = {}
        loss_dict["node loss"] = bce.item() + node_cross.item() + mse.item()
        loss_dict["node construct"] = bce.item()
        loss_dict["node type"] = node_cross.item()
        loss_dict["node param"] = mse.item()

        return node_loss, loss_dict

    def plot_loss(self, loss_dict : dict):
        self.writer.add_scalar("Training/Node_Loss",      loss_dict["node loss"],      self.global_step)
        self.writer.add_scalar("Training/Node_Construct", loss_dict["node construct"], self.global_step)
        self.writer.add_scalar("Training/Node_Type",      loss_dict["node type"],      self.global_step)
        self.writer.add_scalar("Training/Node_Param",     loss_dict["node param"],     self.global_step)

    @torch.no_grad()
    def validate(self):
        self.validate_sampler.set_epoch(self.curr_epoch)
        pbar = tqdm(self.validate_loader) if self.gpu_id == 0 else self.validate_loader
        total_loss = 0
        for nodes, params_mask in pbar:
            batch_size = nodes.size(0)
            nodes = nodes.to(self.gpu_id)
            params_mask = params_mask.to(self.gpu_id)

            t = torch.randint(low = 1, high = self.model.module.max_timestep, size = (batch_size,)).to(self.gpu_id)
            noised_nodes = self.model.module.noise_scheduler(nodes, t)

            pred_nodes = self.model(noised_nodes, t)

            a_bar_t = self.model.module.noise_scheduler.a_bar[t]
            scales = torch.clamp(a_bar_t / (1 - a_bar_t), max = 16).unsqueeze(1).unsqueeze(1)
            loss, loss_dict = self.diffusion_loss(pred_nodes, nodes, params_mask, scales)

            total_loss += loss

            if self.gpu_id == 0: pbar.set_description(f"Validating Epoch {self.curr_epoch}  ")
        
        avg_loss = total_loss / len(pbar)
        if self.gpu_id == 0:
            self.save_checkpoint()
            print("---Saved Model Checkpoint---")
        
        if self.gpu_id == 0: self.writer.add_scalar("Validation/Loss", avg_loss, self.curr_epoch)

        if self.gpu_id == 0:
            fig, axes = plt.subplots(nrows = 2, ncols = config.NUM_TRAJECTORY_VISUALIZATION_IMAGES, figsize=(40, 8))
            sample = self.model.module.sample(1, axes)
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

            self.curr_epoch += 1
    
    def save_checkpoint(self):
        checkpoint = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict()
        epoch = self.curr_epoch
        torch.save(
            {
                "model": checkpoint,
                "optimizer": optimizer_state,
                "epoch": epoch
            }, self.save_location + f"/checkpoints/epoch_{epoch}.pth")

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

def train_on_multiple_gpus(rank: int, 
                           world_size: int,
                           train_set: Subset,
                           validate_set: Subset,
                           learning_rate: float,
                           batch_size: int,
                           num_epochs: int, 
                           experiment_string: str
                          ):
    MultiGPUTrainer.ddp_setup(rank, world_size)
    model = GD3PM(rank)

    trainer = MultiGPUTrainer(
        model = model,
        train_set = train_set,
        validate_set = validate_set,
        learning_rate = learning_rate,
        batch_size = batch_size,
        num_epochs = num_epochs,
        experiment_string = experiment_string,
        gpu_id = rank
    )

    trainer.train()
    
    destroy_process_group()


