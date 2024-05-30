# %%
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
import torch
from gumbel_diffusion_model import GD3PM, diffusion_loss
from dataset1 import SketchDataset
from torch.utils.data import DataLoader, Subset
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
            model: GD3PM,
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

        # checkpoint_path = f"model_checkpoint_gumbeldiff_ddp_Adam_24crossattnblocks16heads256class256param256edge256time_betadist1,2_cont_nll_loss_mse_loss.pth"
        # if os.path.exists(checkpoint_path):
        #     map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_id}
        #     self.model.load_state_dict(torch.load(checkpoint_path, map_location = map_location), strict = False)

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
        self.timestep_distribution = torch.distributions.beta.Beta(torch.tensor([1.0]), torch.tensor([2.0]))
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

        self.record_freq = len(self.train_loader) // 50
        if self.record_freq == 0: self.record_freq = 1
        
        self.batch_size = batch_size

    
    def train_batch(self, nodes : torch.Tensor, edges : torch.Tensor, params_mask : torch.Tensor) -> float:
        self.optimizer.zero_grad()

        nodes = nodes.to(self.gpu_id)
        edges = edges.to(self.gpu_id)
        params_mask = params_mask.to(self.gpu_id)

        batch_size = nodes.size(0)
        t = torch.randint(low = 1, high = self.model.module.max_timestep - 10, size = (batch_size,), device = self.gpu_id)
        t = (self.timestep_distribution.sample((batch_size,)) * (self.model.module.max_timestep)).squeeze().int().to(self.gpu_id)
        noised_nodes, noised_edges, true_node_noise, true_edge_noise = self.model.module.noise_scheduler(nodes, edges, t)
        pred_node_noise, pred_edge_noise = self.model(noised_nodes, noised_edges, t)

        loss_dict = {} # dictionary to record loss values
        none_node_mask = (nodes[...,5] > 0).unsqueeze(-1) # true primitive is classified as none type
        none_edge_mask = (edges[...,16] > 0).unsqueeze(-1) # true constraint is classified as none type
        loss = diffusion_loss(pred_node_noise, pred_edge_noise, true_node_noise, true_edge_noise, params_mask, none_node_mask, none_edge_mask, 
                              nodes, noised_nodes, self.model.module.noise_scheduler, t, loss_dict)

        # assert loss.isfinite().all(), "Loss is non finite value"

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), 1.0)
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

            if (self.global_step % self.record_freq == 0):
                if self.gpu_id == 0: self.plot_loss(iter_loss_dict) # self.writer.add_scalar("Training Loss", iter_loss, self.global_step)

            iter_loss = iter_loss_dict["total loss"]
            if self.gpu_id == 0: pbar.set_description(f"Training Epoch {self.curr_epoch} Iter Loss: {iter_loss}")
    
    def plot_loss(self, loss_dict):
        self.writer.add_scalar("Training/Total_Loss", loss_dict["total loss"],            self.global_step)
        self.writer.add_scalar("Training/Node_Loss",  loss_dict["node loss"],             self.global_step)
        self.writer.add_scalar("Training/Node_Construct",   loss_dict["node isconstruct loss"], self.global_step)
        self.writer.add_scalar("Training/Node_Type", loss_dict["node type loss"],        self.global_step)
        self.writer.add_scalar("Training/Node_Param",   loss_dict["node parameter loss"],   self.global_step)
        self.writer.add_scalar("Training/Edge_Loss",  loss_dict["edge loss"],             self.global_step)
        self.writer.add_scalar("Training/Edge_sub_a", loss_dict["edge sub_a loss"],       self.global_step)
        self.writer.add_scalar("Training/Edge_sub_b", loss_dict["edge sub_b loss"],       self.global_step)
        self.writer.add_scalar("Training/Edge_Type", loss_dict["edge type loss"],        self.global_step)
        # self.writer.add_scalar("Training/KLD", loss_dict["kld"], self.global_step)
        #self.writer.add_scalar("Training/Posterior_Collapse_Regularization", loss_dict["postcollapse_reg"], self.global_step)

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
            t = torch.randint(low = 1, high = self.model.module.max_timestep - 10, size = (batch_size,), device = self.gpu_id)
            noised_nodes, noised_edges, true_node_noise, true_edge_noise = self.model.module.noise_scheduler(nodes, edges, t)
            pred_node_noise, pred_edge_noise = self.model(noised_nodes, noised_edges, t)
            
            loss_dict = {}
            none_node_mask = (nodes[...,5] > 0).unsqueeze(-1) # true primitive is classified as none type
            none_edge_mask = (edges[...,16] > 0).unsqueeze(-1) # true constraint is classified as none type
            loss = diffusion_loss(pred_node_noise, pred_edge_noise, true_node_noise, true_edge_noise, params_mask, none_node_mask, none_edge_mask,
                                  nodes, noised_nodes, self.model.module.noise_scheduler, t, loss_dict)

            total_loss += loss

            assert loss.isfinite().all(), "Loss is non finite value"

            if self.gpu_id == 0: pbar.set_description(f"Validating Epoch {self.curr_epoch}  ")
        
        avg_loss = total_loss / len(pbar)
        if avg_loss < self.min_validation_loss:
            self.min_validation_loss = avg_loss
            if self.gpu_id == 0: 
                self.writer.add_scalar("Validation/Loss", avg_loss, self.curr_epoch)
                self.save_checkpoint()
                print("---Saved Model Checkpoint---")
        
        if self.gpu_id == 0:
            # Render the actual CAD Sketch
            noised_nodes, noised_edges = self.model.module.noise(nodes[0:4], edges[0:4])
            denoised_nodes, denoised_edges = self.model.module.denoise(noised_nodes, noised_edges)
            fig, axes = plt.subplots(nrows = 4, ncols = 2, figsize=(8, 16))
            fig.suptitle(f"Target (left) vs Preds (right) for epoch {self.curr_epoch}")
            for i in range(4):
                target_sketch = SketchDataset.preds_to_sketch(nodes[i].cpu(), edges[i].cpu())
                pred_sketch = SketchDataset.preds_to_sketch(denoised_nodes[i].cpu(), denoised_edges[i].cpu())
                
                datalib.render_sketch(target_sketch, axes[i, 0])
                # SketchDataset.superimpose_constraints(target_sketch, axes[i, 0])
                datalib.render_sketch(pred_sketch, axes[i, 1])
                # SketchDataset.superimpose_constraints(pred_sketch, axes[i, 1])
            
            self.writer.add_figure(f"Validation/Epoch_Result_Visualization", fig, self.curr_epoch)
            plt.close()

            # # Render the graph visualization of the Sketch
            # fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(16, 8))
            # fig.suptitle(f"Target (left) vs Preds (right) for epoch {self.curr_epoch}")
            # self.visualize_graph(nodes[0].cpu(), edges[0].cpu(), "target_graph")
            # self.visualize_graph(pred_nodes[0].cpu(), pred_edges[0].cpu(), "pred_graph")
            # axes[0].imshow(plt.imread("target_graph.png"))
            # axes[1].imshow(plt.imread("pred_graph.png"))
            # self.writer.add_figure(f"Validation/Epoch_Graph_Visualization", fig, self.curr_epoch)
            # plt.close()

    
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
            if self.curr_epoch % 5 == 0:
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

    model = GD3PM(rank)
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


