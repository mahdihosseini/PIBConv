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
    network = NetworkADP(
        C=36, 
        num_classes=10, 
        layers=15, 
        auxiliary=True, 
        genotype=NEWCONV_design_cin4_cifar10_DARTSsettings)

    network.drop_path_prob = 0.2        # Hardcoded - value is only for functionality and should not change complexity at all
    
    print_complexity(network)

