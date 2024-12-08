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
from diff_model3 import GD3PM
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
            ):
        model.device = gpu_id
        self.model = DDP(model.to(gpu_id), device_ids=[gpu_id])
        # if os.path.exists(f"best_model_checkpoint.pth"):
        #     map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_id}
        #     self.model.load_state_dict(torch.load(f"best_model_checkpoint_custom.pth", map_location = map_location))
        self.batch_size = 130
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

        self.record_freq = len(self.train_loader) // 5
        if self.record_freq == 0: self.record_freq = 1

        # self.timestep_distribution = torch.distributions.beta.Beta(torch.tensor([1.6]), torch.tensor([2.0]))
    
    def train_batch(self, nodes : torch.Tensor, edges : torch.Tensor, params_mask : torch.Tensor) -> float:
        self.optimizer.zero_grad()

        nodes = nodes.to(self.gpu_id)
        edges = edges.to(self.gpu_id)
        params_mask = params_mask.to(self.gpu_id)

        batch_size = nodes.size(0)
        # t = (self.timestep_distribution.sample([batch_size,]).squeeze(-1) * self.model.module.max_timestep).int().to(self.gpu_id)
        t = torch.rand(size = (batch_size,), device = self.gpu_id)
        t = (torch.where(t < .75, t * .5/.75, (t - .75) * .5/.25 + 0.5) * self.model.module.max_timestep).int()
        noised_nodes, noised_edges, true_noise = self.model.module.noise_scheduler(nodes, edges, t)

        pred_nodes, pred_edges = self.model(noised_nodes, noised_edges, t)

        # assert pred_nodes.isfinite().all(), "Model output for nodes has non finite values"
        # assert pred_edges.isfinite().all(), "Model output for edges has non finite values"
        # assert means.isfinite().all(),      "Model output for means has non finite values"
        # assert logvars.isfinite().all(),    "Model output for logvars has non finite values"

        loss_dict = {} # dictionary to record loss values 
        loss = self.diffusion_loss(pred_nodes, pred_edges, nodes, edges, params_mask, loss_dict)

        if self.global_step % 1000 == 999 and self.gpu_id == 0:
            fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(12, 4))
            fig.suptitle(f"Noised - Pred - True at time {t[0].item()}")
            SketchDataset.render_graph(nodes[0].cpu().squeeze(0), edges[0].cpu().squeeze(0), axes[2])
            SketchDataset.render_graph(pred_nodes[0].cpu().squeeze(0), pred_edges[0].cpu().squeeze(0), axes[1])
            SketchDataset.render_graph(noised_nodes[0].cpu().squeeze(0), noised_edges[0].cpu().squeeze(0), axes[0])
            self.writer.add_figure("Training/Visual", fig, self.global_step)
            plt.close()

        loss.backward()
        self.optimizer.step()
        # if self.global_step % self.num_grad_accum_steps == self.num_grad_accum_steps - 1: 
        #     self.optimizer.step()
        #     self.optimizer.zero_grad()

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
    
    def diffusion_loss(self, pred_nodes : Tensor, pred_edges : Tensor, true_nodes : Tensor, true_edges : Tensor, params_mask : Tensor, loss_dict : dict) -> Tensor:
        '''Edge Loss'''
        # Only apply subnode loss to constraints that are not none -------
        subnode_a_labels = torch.argmax(true_edges[:,:,:,0:4], dim = 3)
        subnode_a_logits = pred_edges[:,:,:,0:4]#.permute(0, 3, 1, 2).contiguous()
        sub_a_cross_entropy = F.cross_entropy(
            input = subnode_a_logits.reshape(-1, 4), 
            target = subnode_a_labels.flatten(), 
            reduction = 'mean')

        subnode_b_labels = torch.argmax(true_edges[:,:,:,4:8], dim = 3)
        subnode_b_logits = pred_edges[:,:,:,4:8]#.permute(0, 3, 1, 2).contiguous()
        sub_b_cross_entropy = F.cross_entropy(
            input = subnode_b_logits.reshape(-1, 4), 
            target = subnode_b_labels.flatten(), 
            reduction = 'mean')
        
        constraint_type_labels = torch.argmax(true_edges[:,:,:,8:], dim = 3)
        constraint_type_logits = pred_edges[:,:,:,8:]#.permute(0, 3, 1, 2).contiguous()
        # There are far more none constraint types, so weigh them less
        constraint_cross_entropy = F.cross_entropy(
            input = constraint_type_logits.reshape(-1, 9), 
            target = constraint_type_labels.flatten(),
            # weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05]).to(pred_edges.device),
            reduction = 'mean')

        edge_loss = sub_a_cross_entropy + sub_b_cross_entropy + constraint_cross_entropy

        '''Node Loss'''
        # weight = torch.tensor([1.0, 2.0, 2.0, 1.0, 0.1]).to(pred_nodes.device)  # Weight circles, arcs, and points higher since they are much rarer than line and none types
        primitive_type_labels = torch.argmax(true_nodes[:,:,1:6], dim = 2)    # batch_size x num_nodes (class index for each node)
        primitive_type_logits = pred_nodes[:,:,1:6]#.permute(0,2,1).contiguous() # batch_size x num_primitive_types x num_nodes
        
        node_cross = F.cross_entropy(
            input = primitive_type_logits.reshape(-1, 5), 
            target = primitive_type_labels.flatten(),
            # weight = weight, 
            reduction = 'mean')
        
        pred_isconstruct = pred_nodes[:,:,0]
        target_isconstruct = true_nodes[:,:,0]
        bce = F.binary_cross_entropy_with_logits(
            input = pred_isconstruct, 
            target = target_isconstruct,
            # pos_weight = weight,
            reduction = 'mean')

        # pred_noise = pred_nodes[:,:,6:]
        # mse = ((pred_noise - true_noise) ** 2 * params_mask).sum() / params_mask.sum()
        pred_params = pred_nodes[:,:,6:]
        target_params = true_nodes[:,:,6:]
        mae = (torch.abs(pred_params - target_params) * params_mask).sum() / params_mask.sum()
        # mae = (torch.abs(pred_noise - true_noise) * params_mask).sum() / params_mask.sum() * 25

        node_loss = bce + node_cross + 4 * mae

        total_loss = node_loss + 0.1 * edge_loss

        loss_dict["edge loss"] = sub_a_cross_entropy.item() + sub_b_cross_entropy.item() + constraint_cross_entropy.item()
        loss_dict["edge sub_a cross"] = sub_a_cross_entropy.item()
        loss_dict["edge sub_b cross"] = sub_b_cross_entropy.item()
        loss_dict["edge cross"] = constraint_cross_entropy.item()
        loss_dict["node loss"] = bce.item() + node_cross.item() + mae.item()
        loss_dict["node bce"] = bce.item()
        loss_dict["node cross"] = node_cross.item()
        loss_dict["node mae"] = mae.item()
        loss_dict["total loss"] = bce.item() + node_cross.item() + mae.item() + sub_a_cross_entropy.item() + sub_b_cross_entropy.item() + constraint_cross_entropy.item()

        return total_loss

    def plot_loss(self, loss_dict : dict):
        self.writer.add_scalar("Training/Total_Loss", loss_dict["total loss"],       self.global_step)
        self.writer.add_scalar("Training/Node_Loss",  loss_dict["node loss"],        self.global_step)
        self.writer.add_scalar("Training/Node_BCE",   loss_dict["node bce"],         self.global_step)
        self.writer.add_scalar("Training/Node_Cross", loss_dict["node cross"],       self.global_step)
        self.writer.add_scalar("Training/Node_MAE",   loss_dict["node mae"],         self.global_step)
        self.writer.add_scalar("Training/Edge_Loss",  loss_dict["edge loss"],        self.global_step)
        self.writer.add_scalar("Training/Edge_sub_a", loss_dict["edge sub_a cross"], self.global_step)
        self.writer.add_scalar("Training/Edge_sub_b", loss_dict["edge sub_b cross"], self.global_step)
        self.writer.add_scalar("Training/Edge_Cross", loss_dict["edge cross"],       self.global_step)

    @torch.no_grad()
    def validate(self):
        # self.validate_sampler.set_epoch(self.curr_epoch)
        pbar = tqdm(self.validate_loader) if self.gpu_id == 0 else self.validate_loader
        total_loss = 0
        for nodes, edges, params_mask in pbar:
            nodes = nodes.to(self.gpu_id)
            edges = edges.to(self.gpu_id)
            params_mask = params_mask.to(self.gpu_id)

            batch_size = nodes.size(0)
            t = torch.randint(low = 1, high = self.model.module.max_timestep // 4, size = (batch_size,), device = self.gpu_id)
            noised_nodes, noised_edges, true_noise = self.model.module.noise_scheduler(nodes, edges, t)

            pred_nodes, pred_edges = self.model(noised_nodes, noised_edges, t)

            loss_dict = {} # dictionary to record loss values 
            loss = self.diffusion_loss(pred_nodes, pred_edges, nodes, edges, params_mask, loss_dict)

            total_loss += loss
            # assert loss.isfinite().all(), "Loss is non finite value"

            if self.gpu_id == 0: pbar.set_description(f"Validating Epoch {self.curr_epoch}  ")
        
        avg_loss = total_loss / len(pbar)
        # if avg_loss < self.min_validation_loss:
        #     self.min_validation_loss = avg_loss
        #     if self.gpu_id == 0:
        #         self.save_checkpoint()
        #         print("---Saved Model Checkpoint---")
        if self.gpu_id == 0:
            if avg_loss < self.min_validation_loss:
                self.min_validation_loss = avg_loss
                self.save_checkpoint()
                print("---Saved Model Checkpoint---")
        
        
        if self.gpu_id == 0: self.writer.add_scalar("Validation/Loss", avg_loss, self.curr_epoch)

        if self.gpu_id == 0:
            fig, axes = plt.subplots(nrows = 4, ncols = 1, figsize=(4, 16))
            fig.suptitle(f"Samples for epoch {self.curr_epoch}")
            sampled_nodes, sampled_edges = self.model.module.sample(4)
            for i in range(4):
                SketchDataset.render_graph(sampled_nodes[i].cpu().squeeze(0), sampled_edges[i].cpu().squeeze(0), axes[i])

            self.writer.add_figure("Validation/Visual", fig, self.curr_epoch)
            plt.close()

    
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


