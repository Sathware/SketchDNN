# %%
import os
from typing import Sequence
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
# from torch.profiler import profile, record_function, ProfilerActivity
import torch
from diffusion_model2 import DiffusionModel
from loss import diffusion_loss
from dataset1 import SketchDataset
from config import log_clip, node_mse_weight, node_cross_weight, node_bce_weight, edge_suba_weight, edge_subb_weight, edge_constraint_weight
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
            model: DiffusionModel,
            train_set: Subset,
            validate_set: Subset,
            learning_rate: float,
            gpu_id: int,
            num_epochs: int,
            experiment_string: str,
            batch_size: int
            ):
        model.device = gpu_id
        self.model = torch.compile(DDP(model.to(gpu_id), device_ids=[gpu_id]))
        # if os.path.exists(f"best_model_checkpoint.pth"):
        #     map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_id}
        #     self.model.load_state_dict(torch.load(f"best_model_checkpoint_custom.pth", map_location = map_location))

        self.train_sampler = DistributedSampler(train_set, drop_last = True)
        self.train_loader = DataLoader(dataset = train_set, 
                                       batch_size = batch_size, 
                                       shuffle = False, 
                                       pin_memory = True, 
                                       sampler = self.train_sampler,
                                       drop_last = True
                                      )
        self.validate_sampler = DistributedSampler(validate_set)
        self.validate_loader = DataLoader(dataset = validate_set, 
                                          batch_size = batch_size, 
                                          shuffle = False, 
                                          pin_memory = True, 
                                          sampler = self.validate_sampler,
                                          drop_last = True
                                         )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        # self.optimizer = ZeroRedundancyOptimizer(self.model.parameters(),
        #                                          optimizer_class = torch.optim.Adam,
        #                                          lr = learning_rate
        #                                         )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0 = 4, T_mult = 2, eta_min = 1e-5)
        
        self.gpu_id = gpu_id
        self.experiment_string = experiment_string
        self.writer = SummaryWriter(f'runs5/{self.experiment_string}')
        self.num_epochs = num_epochs

        self.global_step = 0
        self.curr_epoch = 0
        self.min_validation_loss = float('inf')

        self.record_freq = len(self.train_loader) // 20
        if self.record_freq == 0: self.record_freq = 1
        
        self.batch_size = batch_size
    
    def train_batch(self, nodes : torch.Tensor, edges : torch.Tensor, params_mask : torch.Tensor) -> float:
        self.optimizer.zero_grad()

        temp_nodes = nodes.to(self.gpu_id)
        temp_edges = edges.to(self.gpu_id)
        params_mask = params_mask.to(self.gpu_id)

        batch_size = nodes.size(0)
        t = torch.randint(low = 1, high = self.model.module.max_timestep, size = (batch_size,), device = self.gpu_id)

        node_noise = torch.randn_like(nodes).to(self.gpu_id)
        edge_noise = torch.randn_like(edges).to(self.gpu_id)

        temp_nodes[...,:7] = temp_nodes[...,:7].log().clip(min = log_clip)
        temp_edges = temp_edges.log().clip(min = log_clip)

        temp_nodes[...,:7] = self.model.module.sqrt_sched.sqrt_a_bar[t,None,None] * temp_nodes[...,:7] + self.model.module.sqrt_sched.sqrt_b_bar[t,None,None] * node_noise[...,:7]
        temp_nodes[...,7:] = self.model.module.cos_sched.sqrt_a_bar[t,None,None] * temp_nodes[...,7:] + self.model.module.cos_sched.sqrt_b_bar[t,None,None] * node_noise[...,7:]
        temp_edges = self.model.module.sqrt_sched.sqrt_a_bar[t,None,None,None] * temp_edges + self.model.module.sqrt_sched.sqrt_b_bar[t,None,None,None] * edge_noise

        temp_nodes, temp_edges = self.model.module.normalize_probs(temp_nodes, temp_edges)

        pred_nodes, pred_edges = self.model(temp_nodes, temp_edges, t)

        loss_dict = {} # dictionary to record loss values 

        loss = self.loss_fn(nodes.to(self.gpu_id), edges.to(self.gpu_id), pred_nodes, pred_edges, params_mask, loss_dict)

        if self.global_step % 200 == 199 and self.gpu_id == 0:
            fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(12, 4))
            fig.suptitle(f"True - Pred - Noised at time {t[0].item()}")
            SketchDataset.render_graph(nodes[0][...,1:].cpu(), edges[0].cpu(), axes[0])
            SketchDataset.render_graph(pred_nodes[0][...,1:].cpu(), pred_edges[0].cpu(), axes[1])
            SketchDataset.render_graph(temp_nodes[0][...,1:].cpu(), temp_edges[0].cpu(), axes[2])
            self.writer.add_figure("Training/Visual", fig, self.global_step)
            plt.close()

        loss.backward()
        self.optimizer.step()

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
            
            if self.gpu_id == 0: self.plot_loss(iter_loss_dict) # self.writer.add_scalar("Training Loss", iter_loss, self.global_step)

            iter_loss = iter_loss_dict["total loss"]
            if self.gpu_id == 0: pbar.set_description(f"Training Epoch {self.curr_epoch} Iter Loss: {iter_loss}")
    
    def loss_fn(self, nodes, edges, pred_nodes, pred_edges, params_mask, loss_dict):
        node_param = ((nodes[...,7:] - pred_nodes[...,7:]) ** 2 * params_mask).sum() / params_mask.sum()
        node_type = (-nodes[...,2:7] * (pred_nodes[...,2:7] - torch.logsumexp(input = pred_nodes[...,2:7], dim = -1, keepdim = True))).mean()
        node_bool = (-nodes[...,0:2] * (pred_nodes[...,0:2] - torch.logsumexp(input = pred_nodes[...,0:2], dim = -1, keepdim = True))).mean()
        edge_suba = (-edges[...,0:4] * (pred_edges[...,0:4] - torch.logsumexp(input = pred_edges[...,0:4], dim = -1, keepdim = True))).mean()
        edge_subb = (-edges[...,4:8] * (pred_edges[...,4:8] - torch.logsumexp(input = pred_edges[...,4:8], dim = -1, keepdim = True))).mean()
        edge_type = (-edges[...,8: ] * (pred_edges[...,8: ] - torch.logsumexp(input = pred_edges[...,8: ], dim = -1, keepdim = True))).mean()

        loss = node_mse_weight * node_param + node_cross_weight * node_type + node_bce_weight * node_bool + edge_suba_weight * edge_suba + edge_subb_weight * edge_subb + edge_constraint_weight * edge_type

        loss_dict["node bool"] = node_bool.item()
        loss_dict["node type"] = node_type.item()
        loss_dict["node param"] = node_param.item()

        loss_dict["edge suba"] = edge_suba.item()
        loss_dict["edge subb"] = edge_subb.item()
        loss_dict["edge type"] = edge_type.item()

        loss_dict["node loss"] = node_bool.item() + node_type.item() + node_param.item()
        loss_dict["edge loss"] = edge_suba.item() + edge_subb.item() + edge_type.item()
        loss_dict["total loss"] = loss_dict["node loss"] + loss_dict["edge loss"]

        return loss

    def plot_loss(self, loss_dict):
        self.writer.add_scalar("Training/Total_Loss", loss_dict["total loss"], self.global_step)
        self.writer.add_scalar("Training/Node_Loss", loss_dict["node loss"], self.global_step)
        self.writer.add_scalar("Training/Node_Bool", loss_dict["node bool"], self.global_step)
        self.writer.add_scalar("Training/Node_Type", loss_dict["node type"], self.global_step)
        self.writer.add_scalar("Training/Node_Param", loss_dict["node param"], self.global_step)
        self.writer.add_scalar("Training/Edge_Loss", loss_dict["edge loss"], self.global_step)
        self.writer.add_scalar("Training/Edge_sub_a", loss_dict["edge suba"], self.global_step)
        self.writer.add_scalar("Training/Edge_sub_b", loss_dict["edge subb"], self.global_step)
        self.writer.add_scalar("Training/Edge_Type", loss_dict["edge type"], self.global_step)

    @torch.no_grad()
    def validate(self):
        # self.validate_sampler.set_epoch(self.curr_epoch)
        pbar = tqdm(self.validate_loader) if self.gpu_id == 0 else self.validate_loader
        total_loss = 0
        validate_step = 0
        for nodes, edges, params_mask in pbar:
            temp_nodes = nodes.to(self.gpu_id)
            temp_edges = edges.to(self.gpu_id)
            params_mask = params_mask.to(self.gpu_id)

            batch_size = nodes.size(0)
            t = torch.randint(low = 1, high = self.model.module.max_timestep // 4, size = (batch_size,), device = self.gpu_id)

            node_noise = torch.randn_like(nodes).to(self.gpu_id)
            edge_noise = torch.randn_like(edges).to(self.gpu_id)

            temp_nodes[...,:7] = temp_nodes[...,:7].log().clip(min = log_clip)
            temp_edges = temp_edges.log().clip(min = log_clip)

            temp_nodes[...,:7] = self.model.module.sqrt_sched.sqrt_a_bar[t,None,None] * temp_nodes[...,:7] + self.model.module.sqrt_sched.sqrt_b_bar[t,None,None] * node_noise[...,:7]
            temp_nodes[...,7:] = self.model.module.cos_sched.sqrt_a_bar[t,None,None] * temp_nodes[...,7:] + self.model.module.cos_sched.sqrt_b_bar[t,None,None] * node_noise[...,7:]
            temp_edges = self.model.module.sqrt_sched.sqrt_a_bar[t,None,None,None] * temp_edges + self.model.module.sqrt_sched.sqrt_b_bar[t,None,None,None] * edge_noise

            temp_nodes, temp_edges = self.model.module.normalize_probs(temp_nodes, temp_edges)

            pred_nodes, pred_edges = self.model(temp_nodes, temp_edges, t)

            loss_dict = {} # dictionary to record loss values 

            loss = self.loss_fn(nodes.to(self.gpu_id), edges.to(self.gpu_id), pred_nodes, pred_edges, params_mask, loss_dict)

            if validate_step == 0 and self.gpu_id == 0:
                fig, axes = plt.subplots(nrows = 1, ncols = 10, figsize=(40, 4))
                fig.suptitle(f"Sample for {self.curr_epoch}")
                self.model.module.sample(1, axes)

                self.writer.add_figure("Validation/Visual", fig, self.curr_epoch)
                plt.close()

            total_loss += loss

            assert loss.isfinite().all(), "Loss is non finite value"

            if self.gpu_id == 0: pbar.set_description(f"Validating Epoch {self.curr_epoch}  ")

            validate_step = validate_step + 1
        
        avg_loss = total_loss / len(pbar)
        if self.gpu_id == 0:
                self.writer.add_scalar("Validation/Loss", avg_loss, self.curr_epoch)
                self.save_checkpoint()
                print("---Saved Model Checkpoint---")

    
    def visualize_graph(self, nodes, edges, filename):
        sketch = SketchDataset.preds_to_sketch(nodes, edges)
        seq = datalib.sketch_to_sequence(sketch)
        graph = datalib.pgvgraph_from_sequence(seq)
        datalib.render_graph(graph, f"{filename}.png")

    def train(self):
        self.global_step = 0
        self.curr_epoch = 0

        while (self.curr_epoch < self.num_epochs):
            self.model.train()
            self.train_epoch()
            if self.curr_epoch % 5 == 4:
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

    model = DiffusionModel(rank)
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


