import torch
import torch.multiprocessing as mp
from torch.utils.data import random_split
import config
from dataset import SketchDataset
from diffusion_trainer import train_on_multiple_gpus

def main():
    print("---Loading Dataset---")
    dataset = SketchDataset("data")
    train_set, validate_set, test_set = random_split(dataset = dataset, lengths = [0.9, 0.05, 0.05], generator = torch.Generator().manual_seed(config.DATASET_SPLIT_SEED))
    print("---Finished Loading Dataset---")
    
    experiment_string = "train_try_1"

    world_size = int(torch.cuda.device_count())
    mp.spawn(train_on_multiple_gpus,
        args=(
            world_size, 
            train_set,
            validate_set,
            config.LEARNING_RATE,
            config.BATCH_SIZE, 
            config.NUM_EPOCHS, 
            experiment_string
            ), 
        nprocs=world_size)

if __name__ == "__main__":
    main()
    