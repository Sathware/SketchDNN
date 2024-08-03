# %%
import os
from typing import Sequence
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
# from torch.profiler import profile, record_function, ProfilerActivity
import torch
from model2 import GVAE
from temp_model import Sharpener
from loss import reconstruction_loss, kl_loss
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

# %%
class MultiGPUTrainer:
    def __init__(
            self,
            model: Sharpener,
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
        self.vae = GVAE(gpu_id)
        # Load the original saved file with DataParallel
        state_dict = torch.load('checkpoints/model_checkpoint_gvae_ddp_Adam_mse-25_kld-.001_16layers16heads256hiddenencoder_16layers16heads256hiddendecoder_embedim1024_tempnodedim128_relu_after_node_layernorm.pth')
        # Create a new OrderedDict without the 'module.' prefix
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # Remove 'module.'
            new_state_dict[name] = v
        # Load the parameters into your model and Freeze
        self.vae.load_state_dict(new_state_dict)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # if os.path.exists(f"best_model_checkpoint.pth"):
        #     map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_id}
        #     self.model.load_state_dict(torch.load(f"best_model_checkpoint_custom.pth", map_location = map_location))

        self.train_sampler = DistributedSampler(train_set, drop_last = True)
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

    
    def train_batch(self, nodes : torch.Tensor, edges : torch.Tensor, params_mask : torch.Tensor) -> float:
        self.optimizer.zero_grad()

        nodes = nodes.to(self.gpu_id)
        edges = edges.to(self.gpu_id)
        params_mask = params_mask.to(self.gpu_id)

        noised_nodes, noised_edges, means, logvars = self.vae(nodes, edges)
        pred_nodes, pred_edges = self.model(noised_nodes, noised_edges)

        loss_dict = {} # dictionary to record loss values 
        loss = reconstruction_loss(pred_nodes, pred_edges, nodes, edges, params_mask, loss_dict) 

        if self.gpu_id == 0 and self.global_step % 400 == 399:
            # Render the actual CAD Sketch
            fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(12, 4))
            fig.suptitle(f"Target (left) vs Preds (center) vs Noised (right) for epoch {self.curr_epoch}")
            SketchDataset.render_graph(nodes[0].cpu(), edges[0].cpu(), axes[0])
            SketchDataset.render_graph(pred_nodes[0].cpu(), pred_edges[0].cpu(), axes[1])
            SketchDataset.render_graph(noised_nodes[0].cpu(), noised_edges[0].cpu(), axes[2])
                
            # SketchDataset.superimpose_constraints(target_sketch, axes[i, 0])
            # SketchDataset.superimpose_constraints(pred_sketch, axes[i, 1])
            
            self.writer.add_figure(f"Training/Visualization", fig, self.curr_epoch)
            plt.close()

        loss.backward()
        self.optimizer.step()

        loss_dict["total loss"] = loss.item()
        return loss_dict
    
    def train_epoch(self):
        self.train_sampler.set_epoch(self.curr_epoch)
        iters = len(self.train_loader)
        pbar = tqdm(self.train_loader) if self.gpu_id == 0 else self.train_loader
        for idx, targets in enumerate(pbar):
            nodes, edges, params_mask = targets
            
            iter_loss_dict = self.train_batch(nodes, edges, params_mask)

            self.global_step += 1
            # self.scheduler.step(self.curr_epoch + idx / iters)
            # if self.gpu_id == 0: self.writer.add_scalar("Learning Rate", self.scheduler.get_last_lr()[0], self.global_step)

            if (self.global_step % self.record_freq == 0):
                if self.gpu_id == 0: self.plot_loss(iter_loss_dict) # self.writer.add_scalar("Training Loss", iter_loss, self.global_step)

            iter_loss = iter_loss_dict["total loss"]
            if self.gpu_id == 0: pbar.set_description(f"Training Epoch {self.curr_epoch} Iter Loss: {iter_loss}")
    
    def plot_loss(self, loss_dict):
        self.writer.add_scalar("Training/Total_Loss", loss_dict["total loss"], self.global_step)
        self.writer.add_scalar("Training/Node_Loss", loss_dict["node loss"], self.global_step)
        self.writer.add_scalar("Training/Node_BCE", loss_dict["node bce"], self.global_step)
        self.writer.add_scalar("Training/Node_Cross", loss_dict["node cross"], self.global_step)
        self.writer.add_scalar("Training/Node_MSE", loss_dict["node mse"], self.global_step)
        self.writer.add_scalar("Training/Edge_Loss", loss_dict["edge loss"], self.global_step)
        self.writer.add_scalar("Training/Edge_sub_a", loss_dict["edge sub_a cross"], self.global_step)
        self.writer.add_scalar("Training/Edge_sub_b", loss_dict["edge sub_b cross"], self.global_step)
        self.writer.add_scalar("Training/Edge_Cross", loss_dict["edge cross"], self.global_step)

    @torch.no_grad()
    def validate(self):
        # self.validate_sampler.set_epoch(self.curr_epoch)
        pbar = tqdm(self.validate_loader) if self.gpu_id == 0 else self.validate_loader
        total_loss = 0
        for nodes, edges, params_mask in pbar:
            nodes = nodes.to(self.gpu_id)
            edges = edges.to(self.gpu_id)
            params_mask = params_mask.to(self.gpu_id)

            noised_nodes, noised_edges, means, logvars = self.vae(nodes, edges)
            pred_nodes, pred_edges = self.model(noised_nodes, noised_edges)

            loss_dict = {} # dictionary to record loss values 
            loss = reconstruction_loss(pred_nodes, pred_edges, nodes, edges, params_mask, loss_dict) 

            total_loss += loss


            if self.gpu_id == 0: pbar.set_description(f"Validating Epoch {self.curr_epoch}  ")
        
        avg_loss = total_loss / len(pbar)

        if self.gpu_id == 0: 
            self.writer.add_scalar("Validation/Loss", avg_loss, self.curr_epoch)
            self.save_checkpoint()
            print("---Saved Model Checkpoint---")
        
        if self.gpu_id == 0:
            # Render the actual CAD Sketch
            fig, axes = plt.subplots(nrows = 4, ncols = 3, figsize=(12, 16))
            fig.suptitle(f"Target (left) vs Preds (right) for epoch {self.curr_epoch}")
            for i in range(4):
                SketchDataset.render_graph(nodes[i].cpu(), edges[i].cpu(), axes[i,0])
                SketchDataset.render_graph(pred_nodes[i].cpu(), pred_edges[i].cpu(), axes[i,1])
                SketchDataset.render_graph(noised_nodes[i].cpu(), noised_edges[i].cpu(), axes[i,2])
            
            self.writer.add_figure(f"Validation/Epoch_Result_Visualization", fig, self.curr_epoch)
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

    model = Sharpener(rank)
    # if os.path.exists(f"best_model_checkpoint.pth"):
    #     model.load_state_dict(torch.load(f"best_model_checkpoint.pth"))

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


