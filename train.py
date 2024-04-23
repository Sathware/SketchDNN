import torch
import torch.multiprocessing as mp
from torch.utils.data import random_split, TensorDataset
from dataset1 import SketchDataset
from diffusion_trainer import train_on_multiple_gpus

def main():
    # mp.freeze_support()
    print("---Loading Dataset into Memory---")
    dataset = SketchDataset(root="data/")
    dataset = TensorDataset(dataset.nodes[0:16], dataset.edges[0:16], dataset.node_params_mask[0:16])
    # train_set, validate_set, test_set = random_split(dataset = dataset, lengths = [0.05, 0.05, 0.9], generator = torch.Generator().manual_seed(4))
    print("---Finished Loading Dataset into Memory---")


    batch_size = 16
    learning_rate = 1e-5
    num_epochs = 1000
    experiment_string = "gd3pm_ddp_Adam_16layers16heads256hidden_test_overfit"

    world_size = 1 # int(torch.cuda.device_count())
    mp.spawn(train_on_multiple_gpus,
        args=(
            world_size, 
            dataset, # train_set
            dataset, # validate_set
            learning_rate,
            batch_size, 
            num_epochs, 
            experiment_string
            ), 
        nprocs=world_size)

if __name__ == "__main__":
    main()
    