import torch
from model import *
from genotypes import *
from ptflops import get_model_complexity_info

def print_complexity(network):
    macs, params = get_model_complexity_info(network, (3, 32, 32), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

if __name__ == "__main__":
    network = NetworkCIFAR(
        C=36, 
        num_classes=10, 
        layers=20, 
        auxiliary=False, 
        genotype=DARTS_newconv_epoch50)

    network.drop_path_prob = 0.2        # Placeholder - value is only for functionality and should not change complexity at all
    
    print_complexity(network)

