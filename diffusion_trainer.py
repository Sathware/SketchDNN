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

    
    def train_batch(self, nodes : torch.Tensor, edges : torch.Tensor, params_mask : torch.Tensor) -> float:
        self.optimizer.zero_grad()

        nodes = nodes.to(self.gpu_id)
        edges = edges.to(self.gpu_id)
        params_mask = params_mask.to(self.gpu_id)

        batch_size = nodes.size(0)
        t = torch.randint(low = 1, high = self.T, size = (batch_size,), device = self.gpu_id)
        # noisy_nodes, noisy_edges, true_noise = self.model.module.noise_scheduler(nodes, edges, timestep)

        node_noise = torch.randn_like(nodes).to(self.gpu_id)
        edge_noise = torch.randn_like(edges).to(self.gpu_id)

        noised_nodes = self.sqrt_a_bar[t,None,None] * nodes + self.sqrt_b_bar[t,None,None] * node_noise
        noised_edges = self.sqrt_a_bar[t,None,None,None] * edges + self.sqrt_b_bar[t,None,None,None] * edge_noise

        pred_node_noise, pred_edge_noise = self.model(noised_nodes, noised_edges, t)

        # assert pred_nodes.isfinite().all(), "Model output for nodes has non finite values"
        # assert pred_edges.isfinite().all(), "Model output for edges has non finite values"
        # assert means.isfinite().all(),      "Model output for means has non finite values"
        # assert logvars.isfinite().all(),    "Model output for logvars has non finite values"

        loss_dict = {} # dictionary to record loss values 
        # loss = diffusion_loss(pred_nodes, pred_edges, nodes, edges, true_noise, params_mask, loss_dict)

        # assert loss.isfinite().all(), "Loss is non finite value"
        # loss_scales = (2 * self.sqrt_a_bar[t,None,None]).clamp(1)
        node_type_loss = ((node_noise[...,:6] - pred_node_noise[...,:6]) ** 2).mean()
        node_param_loss = ( (node_noise[...,6:] - pred_node_noise[...,6:])  ** 2 * params_mask).mean()
        edge_loss = ((edge_noise - pred_edge_noise) ** 2).mean()
        loss = node_type_loss + 16 * node_param_loss + 0.1 * edge_loss

        loss_dict["node type"] = node_type_loss.item()
        loss_dict["node param"] = node_param_loss.item()
        loss_dict["node loss"] = node_type_loss.item() + node_param_loss.item()
        loss_dict["edge loss"] = edge_loss.item()
        loss_dict["total loss"] = node_type_loss.item() + node_param_loss.item() + edge_loss.item()

        # self.writer.add_scalar("Training/Node_Type_Loss", node_type_loss.item(), self.global_step)
        # self.writer.add_scalar("Training/Edge_Loss", edge_loss.item(), self.global_step)
        if self.global_step % 50 == 49 and self.gpu_id == 0:
            fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(12, 4))
            fig.suptitle(f"True - Pred - Noised at time {t[0].item()}")
            SketchDataset.render_graph(nodes[0].cpu().squeeze(0), edges[0].cpu().squeeze(0), axes[0])
            # SketchDataset.render_graph(denoised_nodes[0].cpu().squeeze(0), denoised_edges[0].cpu().squeeze(0), axes[1])
            SketchDataset.render_graph(( (noised_nodes[0] - self.sqrt_b_bar[t[0]] * pred_node_noise[0]) / self.sqrt_a_bar[t[0]]).cpu().squeeze(0), edges[0].cpu().squeeze(0), axes[1])
            SketchDataset.render_graph(noised_nodes[0].cpu().squeeze(0), noised_edges[0].cpu().squeeze(0), axes[2])
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
    
    def loss_fn(self, nodes, edges, denoised_nodes, denoised_edges, params_mask, loss_scales, loss_dict):
        # loss_scales = (2 * self.sqrt_a_bar[t,None,None]).clamp(1)
        node_type_loss = ((nodes[...,:6] - denoised_nodes[...,:6]) ** 2).mean()
        node_param_loss = (( loss_scales * (nodes[...,6:] - denoised_nodes[...,6:]) ) ** 2 * params_mask).mean()
        edge_loss = ((edges - denoised_edges) ** 2).mean()
        loss = node_type_loss + node_param_loss + 0.1 * edge_loss

        loss_dict["node type"] = node_type_loss.item()
        loss_dict["node param"] = node_param_loss.item()
        loss_dict["node loss"] = node_type_loss.item() + node_param_loss.item()
        loss_dict["edge loss"] = edge_loss.item()
        loss_dict["total loss"] = node_type_loss.item() + node_param_loss.item() + edge_loss.item()

        return loss, loss_dict

    def plot_loss(self, loss_dict):
        self.writer.add_scalar("Training/Total_Loss", loss_dict["total loss"], self.global_step)
        self.writer.add_scalar("Training/Node_Loss", loss_dict["node loss"], self.global_step)
        # self.writer.add_scalar("Training/Node_Bool", loss_dict["node bool"], self.global_step)
        self.writer.add_scalar("Training/Node_Type", loss_dict["node type"], self.global_step)
        self.writer.add_scalar("Training/Node_Param", loss_dict["node param"], self.global_step)
        self.writer.add_scalar("Training/Edge_Loss", loss_dict["edge loss"], self.global_step)
        # self.writer.add_scalar("Training/Edge_sub_a", loss_dict["edge sub_a"], self.global_step)
        # self.writer.add_scalar("Training/Edge_sub_b", loss_dict["edge sub_b"], self.global_step)
        # self.writer.add_scalar("Training/Edge_Type", loss_dict["edge type"], self.global_step)
        # self.writer.add_scalar("Training/KLD", loss_dict["kld"], self.global_step)
        #self.writer.add_scalar("Training/Posterior_Collapse_Regularization", loss_dict["postcollapse_reg"], self.global_step)

    @torch.no_grad()
    def validate(self):
        # self.validate_sampler.set_epoch(self.curr_epoch)
        pbar = tqdm(self.validate_loader) if self.gpu_id == 0 else self.validate_loader
        total_loss = 0
        validate_step = 0
        for nodes, edges, params_mask in pbar:
            nodes = nodes.to(self.gpu_id)
            edges = edges.to(self.gpu_id)
            params_mask = params_mask.to(self.gpu_id)

            # batch_size = nodes.size(0)
            # timestep = torch.randint(low = 1, high = self.model.module.max_steps, size = (batch_size,), device = self.gpu_id)
            # noisy_nodes, noisy_edges, true_noise = self.model.module.noise_scheduler(nodes, edges, timestep)
            # pred_nodes, pred_edges = self.model(noisy_nodes, noisy_edges, timestep)
            
            # loss = diffusion_loss(pred_nodes, pred_edges, nodes, edges, true_noise, params_mask)

            # total_loss += loss
            batch_size = nodes.size(0)
            t = torch.randint(low = 1, high = self.T, size = (batch_size,), device = self.gpu_id)
            # noisy_nodes, noisy_edges, true_noise = self.model.module.noise_scheduler(nodes, edges, timestep)

            node_noise = torch.randn_like(nodes).to(self.gpu_id)
            edge_noise = torch.randn_like(edges).to(self.gpu_id)

            noised_nodes = self.sqrt_a_bar[t,None,None] * nodes + self.sqrt_b_bar[t,None,None] * node_noise
            noised_edges = self.sqrt_a_bar[t,None,None,None] * edges + self.sqrt_b_bar[t,None,None,None] * edge_noise

            pred_node_noise, pred_edge_noise = self.model(noised_nodes, noised_edges, t)

            loss_dict = {} # dictionary to record loss values 

            # loss_scales = (2 * self.sqrt_a_bar[t,None,None]).clamp(1)
            node_type_loss = ((node_noise[...,:6] - pred_node_noise[...,:6]) ** 2).mean()
            node_param_loss = ((node_noise[...,6:] - pred_node_noise[...,6:] ) ** 2 * params_mask).mean()
            edge_loss = ((edge_noise - pred_edge_noise) ** 2).mean()
            loss = node_type_loss + node_param_loss + 0.1 * edge_loss

            loss_dict["node type"] = node_type_loss.item()
            loss_dict["node param"] = node_param_loss.item()
            loss_dict["node loss"] = node_type_loss.item() + node_param_loss.item()
            loss_dict["edge loss"] = edge_loss.item()
            loss_dict["total loss"] = node_type_loss.item() + node_param_loss.item() + edge_loss.item()

            if validate_step == 0 and self.gpu_id == 0:
                fig, axes = plt.subplots(nrows = 4, ncols = 3, figsize=(12, 16))
                fig.suptitle(f"True - Pred - Noised for epoch {self.curr_epoch}")
                for i in range(4):
                    SketchDataset.render_graph(nodes[i].cpu().squeeze(0), edges[i].cpu().squeeze(0), axes[i,0])
                    # SketchDataset.render_graph(denoised_nodes[i].cpu().squeeze(0), denoised_edges[i].cpu().squeeze(0), axes[i,1])
                    SketchDataset.render_graph(( (noised_nodes[i] - self.sqrt_b_bar[t[i]] * pred_node_noise[i]) / self.sqrt_a_bar[t[i]]).cpu().squeeze(0), edges[i].cpu().squeeze(0), axes[i,1])
                    SketchDataset.render_graph(noised_nodes[i].cpu().squeeze(0), noised_edges[i].cpu().squeeze(0), axes[i,2])
                    axes[i,2].set_title(f'timestep {t[i].item()}')

                self.writer.add_figure("Validation/Visual", fig, self.curr_epoch)
                plt.close()

            total_loss += loss

            assert loss.isfinite().all(), "Loss is non finite value"

            if self.gpu_id == 0: pbar.set_description(f"Validating Epoch {self.curr_epoch}  ")

            validate_step = validate_step + 1
        
        avg_loss = total_loss / len(pbar)
        # if avg_loss < self.min_validation_loss:
        #     self.min_validation_loss = avg_loss
        #     if self.gpu_id == 0:
        #         self.save_checkpoint()
        #         print("---Saved Model Checkpoint---")
        if self.gpu_id == 0:
                self.save_checkpoint()
                print("---Saved Model Checkpoint---")
        
        
        if self.gpu_id == 0: self.writer.add_scalar("Validation/Loss", avg_loss, self.curr_epoch)

            # # Render the graph visualization of the Sketch
            # fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(16, 8))
            # fig.suptitle(f"Target (left) vs Preds (right) for epoch {self.curr_epoch}")
            # self.visualize_graph(nodes[0].cpu(), edges[0].cpu(), "target_graph")
            # self.visualize_graph(pred_nodes[0].cpu(), pred_edges[0].cpu(), "pred_graph")
            # axes[0].imshow(plt.imread("target_graph.png"))
            # axes[1].imshow(plt.imread("pred_graph.png"))
            # self.writer.add_figure(f"Validation/Epoch_Graph_Visualization", fig, self.curr_epoch)
            # plt.close()

        # if self.gpu_id == 0:
        #     # Render the actual CAD Sketch
        #     noised_nodes, noised_edges = self.model.module.noise(nodes[0:4], edges[0:4])
        #     denoised_nodes, denoised_edges = self.model.module.denoise(noised_nodes, noised_edges)
        #     fig, axes = plt.subplots(nrows = 4, ncols = 2, figsize=(8, 16))
        #     fig.suptitle(f"Target (left) vs Preds (right) for epoch {self.curr_epoch}")
        #     for i in range(4):
        #         target_sketch = SketchDataset.preds_to_sketch(nodes[i].cpu(), edges[i].cpu())
        #         pred_sketch = SketchDataset.preds_to_sketch(denoised_nodes[i].cpu(), denoised_edges[i].cpu())
                
        #         datalib.render_sketch(target_sketch, axes[i, 0])
        #         # SketchDataset.superimpose_constraints(target_sketch, axes[i, 0])
        #         datalib.render_sketch(pred_sketch, axes[i, 1])
        #         # SketchDataset.superimpose_constraints(pred_sketch, axes[i, 1])
            
        #     self.writer.add_figure(f"Validation/Epoch_Result_Visualization", fig, self.curr_epoch)
        #     plt.close()

    
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


