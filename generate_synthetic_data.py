import yaml
import torch

from data_generation.generator import Generator
from data_generation.parser import DataGenerationArgParser

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Parse the arguments to extract the mandatory arguments
argparser = DataGenerationArgParser()
args = argparser.parse_args_as_dict()

# Extract configs
with open(args['config_file']) as f:
    cfg = yaml.load(f, Loader=yaml.UnsafeLoader)

# Initialize the generator object
generator = Generator(cfg, device=device)

# Generate the data
generator.generate()