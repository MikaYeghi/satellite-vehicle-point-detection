import os
import yaml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from data_generation.generator import Generator
from data_generation.parser import DataGenerationArgParser

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def main(rank, world_size, cfg):
    # Setup multi-GPU
    setup(rank, world_size)

    # Construct the device variable
    if cfg['DEVICE'] == 'cuda':
        device = "cuda:" + str(rank)
    elif cfg['DEVICE'] == 'cpu':
        device = "cpu:" + str(rank)
        print("WARNING: using CPU as the device for a multi-GPU scenario.")
    else:
        raise NotImplementedError
    
    # Initialize the generator object
    generator = Generator(cfg, rank, world_size, device=device)

    # Generate the data
    generator.generate()

if __name__ == '__main__':
    # Parse the arguments to extract the mandatory arguments
    argparser = DataGenerationArgParser()
    args = argparser.parse_args_as_dict()

    # Extract configs
    with open(args['config_file']) as f:
        cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
    
    world_size = cfg['NUM_GPUS']
    
    mp.spawn(
        main,
        args=(world_size, cfg),
        nprocs=world_size
    )
    
    cleanup()