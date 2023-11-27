# %%
import os
from typing import Sequence
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
# from torch.profiler import profile, record_function, ProfilerActivity
import torch
from model import GVAE
from loss import reconstruction_loss, kl_loss
from dataset import SketchDataset
from torch.utils.data import DataLoader, Subset, random_split
os.chdir('SketchGraphs/')
import sketchgraphs.data as datalib
os.chdir('../')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# %%
class MultiGPUTrainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_set: Subset,
            validate_set: Subset,
            learning_rate: float,
            gpu_id: int,
            num_epochs: int,
            experiment_string: str,
            batch_size: int
            ):
        model.device = gpu_id
        self.model = DDP(model.to(gpu_id), device_ids=[gpu_id])
        self.train_sampler = DistributedSampler(train_set)
        self.train_loader = DataLoader(dataset = train_set, 
                                       batch_size = batch_size, 
                                       shuffle = False, 
                                       pin_memory = True, 
                                       sampler = self.train_sampler
                                      )
        self.validate_sampler = DistributedSampler(validate_set)
        self.validate_loader = DataLoader(dataset = validate_set, 
                                          batch_size = batch_size, 
                                          shuffle = False, 
                                          pin_memory = True, 
                                          sampler = self.validate_sampler
                                         )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.9)
        
        self.gpu_id = gpu_id
        self.writer = SummaryWriter(f'runs/{experiment_string}')
        self.num_epochs = num_epochs

        self.global_step = 0
        self.curr_epoch = 0
        self.min_validation_loss = float('inf')

    
    def train_batch(self, nodes : torch.Tensor, edges : torch.Tensor) -> float:
        self.optimizer.zero_grad()

        nodes = nodes.to(self.gpu_id)
        edges = edges.to(self.gpu_id)

        pred_nodes, pred_edges, means, logvars = self.model(nodes, edges)

        # assert pred_nodes.isfinite().all(), "Model output for nodes has non finite values"
        # assert pred_edges.isfinite().all(), "Model output for edges has non finite values"
        # assert means.isfinite().all(),      "Model output for means has non finite values"
        # assert logvars.isfinite().all(),    "Model output for logvars has non finite values"

        loss = reconstruction_loss(pred_nodes, pred_edges, nodes, edges)
        loss += kl_loss(means, logvars)

        assert loss.isfinite().all(), "Loss is non finite value"

        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def train_epoch(self):
        self.train_sampler.set_epoch(self.curr_epoch)
        pbar = tqdm(self.train_loader)
        for nodes, edges in pbar:
            iter_loss = self.train_batch(nodes, edges)

            self.global_step += 1

            if (self.global_step % 100 == 99):
                if self.gpu_id == 0: self.writer.add_scalar("Training Loss", iter_loss, self.global_step)
            
            pbar.set_description(f"Training Epoch {self.curr_epoch} Iter Loss: {iter_loss}  ")
    
    @torch.no_grad()
    def validate(self):
        self.validate_sampler.set_epoch(self.curr_epoch)
        pbar = tqdm(self.validate_loader)
        total_loss = 0
        for nodes, edges in pbar:
            nodes = nodes.to(self.gpu_id)
            edges = edges.to(self.gpu_id)

            pred_nodes, pred_edges, means, logvars = self.model(nodes, edges)

            loss = reconstruction_loss(pred_nodes, pred_edges, nodes, edges)
            loss += kl_loss(means, logvars)

            total_loss += loss

            assert loss.isfinite().all(), "Loss is non finite value"

            pbar.set_description(f"Validating Epoch {self.curr_epoch}  ")
        
        avg_loss = total_loss / len(pbar)
        if avg_loss < self.min_validation_loss:
            self.min_validation_loss = avg_loss
            if self.gpu_id == 0: 
                self.writer.add_scalar("Validation Loss", avg_loss, self.curr_epoch)
                self.save_checkpoint()
        
        if self.gpu_id == 0:
            fig, axes = plt.subplots(nrows = 4, ncols = 2, figsize=(8, 16))
            fig.suptitle(f"Target (left) vs Preds (right) for epoch {self.curr_epoch}")
            for i in range(4):
                target_sketch = SketchDataset.preds_to_sketch(nodes[i].cpu(), edges[i].cpu())
                pred_sketch = SketchDataset.preds_to_sketch(pred_nodes[i].cpu(), pred_edges[i].cpu())
                
                datalib.render_sketch(target_sketch, axes[i, 0])
                datalib.render_sketch(pred_sketch, axes[i, 1])
            
            self.writer.add_figure(f"Epoch result visualization", fig, self.curr_epoch)
            plt.close()
                
    def train(self):
        self.global_step = 0
        self.curr_epoch = 0

        while (self.curr_epoch < self.num_epochs):
            self.model.train()
            self.train_epoch()
            self.model.eval()
            self.validate()
            self.curr_epoch += 1
            self.scheduler.step()
            # print("Learning Rate: ", self.scheduler.get_last_lr())
    
    def save_checkpoint(self):
        checkpoint = self.model.module.state_dict()
        torch.save(checkpoint, "best_model_checkpoint.pth")

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
                           train_set: Subset,
                           validate_set: Subset,
                           learning_rate: float,
                           batch_size: int,
                           num_epochs: int, 
                           experiment_string: str
                          ):
    MultiGPUTrainer.ddp_setup(rank, world_size)

    #dataset = SketchDataset(root = "data/")
    #train_set = Subset(dataset = dataset, indices = train_indices)
    #validate_set = Subset(dataset = dataset, indices = validate_indices)

    model = GVAE(rank)
    # if os.path.exists(f"best_model_checkpoint.pth"):
        # model.load_state_dict(torch.load(f"best_model_checkpoint.pth"))

    trainer = MultiGPUTrainer(
        model = model,
        train_set = train_set,
        validate_set = validate_set,
        learning_rate = learning_rate,
        gpu_id = rank,
        num_epochs = num_epochs,
        experiment_string = experiment_string,
        batch_size = batch_size
    )

    trainer.train()
    
    destroy_process_group()


