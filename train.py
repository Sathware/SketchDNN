import torch
import torch.multiprocessing as mp
from torch.utils.data import random_split, TensorDataset
from dataset1 import SketchDataset
from diffusion_trainer2 import train_on_multiple_gpus
from torch.utils.tensorboard.writer import SummaryWriter
# from sharpener_trainer import train_on_multiple_gpus

def main():
    # mp.freeze_support()
    print("---Loading Dataset into Memory---")
    dataset = SketchDataset(root="data/")
    # # dataset.nodes[...,:6] = 2 * dataset.nodes[...,:6] - 1
    # # dataset.edges = 2 * dataset.edges - 1
    dataset.nodes = torch.cat([1 - dataset.nodes[...,[0]], dataset.nodes], dim = -1)
    dataset = TensorDataset(dataset.nodes, dataset.edges, dataset.node_params_mask)
    train_set, validate_set, _ = random_split(dataset = dataset, lengths = [0.05, 0.05, 0.9], generator = torch.Generator().manual_seed(4))
    # embedding_dataset = torch.load("embedding_dataset.pth")
    print("---Finished Loading Dataset into Memory---")

    batch_size = 56
    learning_rate = 1e-4
    num_epochs = 10000
    # experiment_string = "sharpener_ddp_Adam_16tflayers"
    experiment_string = "softgaussdiff_ddp_adam_64tflayers_256hiddendim"

    # writer = SummaryWriter(f'runs5/{experiment_string}')
    # nodes, edges, _ = dataset[0]
    # nodes = nodes[...,1:]
    # print(nodes.shape, " ", edges.shape)
    # fig = SketchDataset.render_graph(nodes, edges)
    # writer.add_figure("Training/Visual", fig, 0)

    world_size = int(torch.cuda.device_count())
    # mp.spawn(train_on_multiple_gpus,
    #     args=(
    #         world_size, 
    #         embedding_dataset,
    #         learning_rate,
    #         batch_size, 
    #         num_epochs, 
    #         experiment_string
    #         ), 
    #     nprocs=world_size)
    mp.spawn(train_on_multiple_gpus,
        args=(
            world_size, 
            train_set, # train_set
            validate_set, # validate_set
            learning_rate,
            batch_size, 
            num_epochs, 
            experiment_string
            ), 
        nprocs=world_size)

if __name__ == "__main__":
    main()
    