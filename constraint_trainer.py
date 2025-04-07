# %%
import os
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
# from torch.profiler import profile, record_function, ProfilerActivity
import torch
from constraint_model import ConstraintModel
from torch.utils.data import DataLoader, Subset, random_split
from loss import edge_loss

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
            model: ConstraintModel,
            train_set: Subset,
            validate_set: Subset,
            gpu_id: int,
            num_epochs: int,
            experiment_string: str,
            ):
        model.device = gpu_id
        self.model = DDP(model.to(gpu_id), device_ids=[gpu_id])
        # if os.path.exists(f"best_model_checkpoint.pth"):
        #     map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_id}
        #     self.model.load_state_dict(torch.load(f"best_model_checkpoint_custom.pth", map_location = map_location))
        self.learning_rate = 1e-4
        self.batch_size = 512
        self.train_sampler = DistributedSampler(train_set, drop_last = True)
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
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
    
    def train_batch(self, nodes : torch.Tensor, edges : torch.Tensor) -> float:
        self.optimizer.zero_grad()

        nodes = nodes.to(self.gpu_id)
        edges = edges.to(self.gpu_id)

        t = torch.ones(nodes.size(0)).int().to(self.gpu_id) * 10
        nodes = self.model.module.noise_scheduler(nodes, t)
        pred_edges = self.model(nodes)

        # assert pred_nodes.isfinite().all(), "Model output for nodes has non finite values"
        # assert pred_edges.isfinite().all(), "Model output for edges has non finite values"
        # assert means.isfinite().all(),      "Model output for means has non finite values"
        # assert logvars.isfinite().all(),    "Model output for logvars has non finite values"

        loss_dict = {} # dictionary to record loss values 
        loss = edge_loss(pred_edges, edges, loss_dict)

        assert loss.isfinite().all(), "Loss is non finite value"

        loss.backward()
        self.optimizer.step()

        loss_dict["total loss"] = loss.item()
        return loss_dict
    
    def train_epoch(self):
        self.train_sampler.set_epoch(self.curr_epoch)
        iters = len(self.train_loader)
        pbar = tqdm(self.train_loader) if self.gpu_id == 0 else self.train_loader
        for idx, targets in enumerate(pbar):
            nodes, edges = targets
            
            iter_loss_dict = self.train_batch(nodes, edges)

            self.global_step += 1
            # self.scheduler.step(self.curr_epoch + idx / iters)
            # if self.gpu_id == 0: self.writer.add_scalar("Learning Rate", self.scheduler.get_last_lr()[0], self.global_step)

            if (self.global_step % self.record_freq == 0):
                if self.gpu_id == 0: self.plot_loss(iter_loss_dict) # self.writer.add_scalar("Training Loss", iter_loss, self.global_step)

            iter_loss = iter_loss_dict["total loss"]
            if self.gpu_id == 0: pbar.set_description(f"Training Epoch {self.curr_epoch} Iter Loss: {iter_loss}")
    
    def plot_loss(self, loss_dict):
        self.writer.add_scalar("Training/Edge_Loss", loss_dict["edge loss"], self.global_step)
        self.writer.add_scalar("Training/Edge_sub_a", loss_dict["edge sub_a cross"], self.global_step)
        self.writer.add_scalar("Training/Edge_sub_b", loss_dict["edge sub_b cross"], self.global_step)
        self.writer.add_scalar("Training/Edge_Cross", loss_dict["edge cross"], self.global_step)

    @torch.no_grad()
    def validate(self):
        # self.validate_sampler.set_epoch(self.curr_epoch)
        pbar = tqdm(self.validate_loader) if self.gpu_id == 0 else self.validate_loader
        total_loss = 0
        for nodes, edges in pbar:
            nodes = nodes.to(self.gpu_id)
            edges = edges.to(self.gpu_id)

            pred_edges = self.model(nodes)

            loss = edge_loss(pred_edges, edges)

            total_loss += loss

            assert loss.isfinite().all(), "Loss is non finite value"

            if self.gpu_id == 0: pbar.set_description(f"Validating Epoch {self.curr_epoch}  ")
        
        avg_loss = total_loss / len(pbar)
        if self.gpu_id == 0: self.writer.add_scalar("Validation/Loss", avg_loss, self.curr_epoch)

        if avg_loss < self.min_validation_loss:
            self.min_validation_loss = avg_loss 
            if self.gpu_id == 0:
                self.save_checkpoint()
                print("---Saved Model Checkpoint---")

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
                           num_epochs: int, 
                           experiment_string: str
                          ):
    MultiGPUTrainer.ddp_setup(rank, world_size)

    #dataset = SketchDataset(root = "data/")
    #train_set = Subset(dataset = dataset, indices = train_indices)
    #validate_set = Subset(dataset = dataset, indices = validate_indices)

    model = ConstraintModel(rank)
    # if os.path.exists(f"best_model_checkpoint.pth"):
    #     model.load_state_dict(torch.load(f"best_model_checkpoint.pth"))

    trainer = MultiGPUTrainer(
        model = model,
        train_set = train_set,
        validate_set = validate_set,
        gpu_id = rank,
        num_epochs = num_epochs,
        experiment_string = experiment_string
    )

    trainer.train()
    
    destroy_process_group()


