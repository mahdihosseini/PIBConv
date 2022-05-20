from venv import create
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
import math
import re
import numpy as np


N_CELLS = 28
N_EDGES = 14



def create_cell_name_dict(path):
    fname = os.path.join(path, 'weights_stat__.xlsx')
    df = pd.read_excel(fname, sheet_name=0)
    
    cell_name = df.columns.str.extract(r'(.*)\_epoch0',expand=False).dropna()
    
    return cell_name
    


def read_weights(path):
    fname = os.path.join(path, 'weights_stat__.xlsx')
    df = pd.read_excel(fname, sheet_name=0)
    
    ncols = df.shape[1] # exclude the first column
    #3D list: alphas[cell_id][edge_id]
    alphas = []
    
    
    # cells trained in each epoch
    for cell in range(N_CELLS):
        # epoch
        cell_weight = []
        for col in range(cell+1,ncols,N_CELLS):
            edges_weight = []
            # edge weight for one cell
            for row in range(N_EDGES):
                edges_weight.append(df.iloc[row, col])
            cell_weight.append(edges_weight)
        alphas.append(cell_weight)
    
    return alphas


def plot_weights(path, cell_edges_weight, cell_id, cell_name):
    epochs = len(cell_edges_weight)
    edges = len(cell_edges_weight[0])
    plt.figure() 
    for edge in range(edges):
        plt.plot(cell_edges_weight[:, edge], label=f'edge{edge}')
    
    plt.legend(loc="upper left")
    fname = os.path.join(path, f'cell{cell_id}.png')
    curr_node = cell_name[int(cell_id)]
    plt.title(f'Alpha for Node {curr_node}')
    plt.ylabel('Architecture Weight')
    plt.xlabel('Epoch')
    plt.savefig(f'{fname}')
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argparser')
    parser.add_argument('-path', action='store', help='e.g. ../save_data#')
    args = parser.parse_args()
    
    folder_name = args.path
    alphas = read_weights(folder_name)
    cell_name = np.array(create_cell_name_dict(folder_name))
    print(cell_name[2])
    print(len(alphas)) # num of cells per epoch
    print(len(alphas[0])) # num of epochs
    print(len(alphas[0][0])) # num of edges per cell
    
    alphas = np.array(alphas)
    
    num_cells = len(alphas)
    num_epochs = len(alphas[0])
    num_edges = len(alphas[0][0])
    for ncell in range(num_cells):
        plot_weights(folder_name, alphas[ncell][:][:], ncell, cell_name)
    
    
    
    
    
    


