import argparse

class DataGenerationArgParser:
    def __init__(self):
        self.parser = self.initialize_argparser()
    
    def initialize_argparser(self):
        # Create the parser
        parser  = argparse.ArgumentParser(description="Data generation argument parser")
        
        # Add arguments
        parser.add_argument('--config-file', type=str, help="path to the config file")
        
        return parser
    
    def parse_args(self):
        return self.parser.parse_args()
    
    def parse_args_as_dict(self):
        return vars(self.parser.parse_args())