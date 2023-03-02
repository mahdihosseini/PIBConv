from venv import create
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
import math
import re
import numpy as np
import glob


N_CELLS = 16
N_EDGES = 14



def create_cell_name_dict(path):
    gpaths = glob.glob(os.path.join(path, 'weights_stat*.xlsx'))
    print(gpaths)
    fname = gpaths[0]
    print(fname)
    print(type(fname))
    df = pd.read_excel(fname, sheet_name=0)
    
    cell_name = df.columns.str.extract(r'(.*)\_epoch0',expand=False).dropna()
    
    return cell_name
    


def read_weights(path, n_cells):
    gpaths = glob.glob(os.path.join(path, 'weights_stat*.xlsx'))
    fname = gpaths[0]
    print(fname)
    print(type(fname))
    df = pd.read_excel(fname, sheet_name=0)
    
    ncols = df.shape[1] # exclude the first column
    #3D list: alphas[cell_id][edge_id]
    alphas = []
    
    
    # cells trained in each epoch
    for cell in range(n_cells):
        # epoch
        cell_weight = []
        for col in range(cell+1,ncols,n_cells):
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
    cell_edges_weight = np.asarray(cell_edges_weight)
    for edge in range(edges):
        if edge == 0:
            print("=======")
            print(edge)
            print(type(cell_edges_weight))
            print(np.asarray(cell_edges_weight).shape)
        plt.plot(cell_edges_weight[:, edge], label=f'edge{edge}')
    
    plt.legend(loc="upper left")
    fname = os.path.join(path, f'cell{cell_id}.png')
    print(cell_id)
    curr_node = cell_name[int(cell_id)]
    plt.title(f'Alpha for Node {curr_node}')
    plt.ylabel('Architecture Weight')
    plt.xlabel('Epoch')
    plt.savefig(f'{fname}')
    plt.clf()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argparser')
    parser.add_argument('-path', action='store', help='e.g. ../save_data#')
    args = parser.parse_args()
    
    folder_name = args.path
    cell_name = np.array(create_cell_name_dict(folder_name))
    alphas = read_weights(folder_name, n_cells=len(cell_name))
    
    
    # print(cell_name[2])
    # print(len(alphas)) # num of cells per epoch
    # print(len(alphas[0])) # num of epochs
    # print(len(alphas[0][0])) # num of edges per cell
    
    alphas = np.array(alphas, dtype=object)
    
    num_cells = len(alphas)
    num_epochs = len(alphas[0])
    num_edges = len(alphas[0][0])
    for ncell in range(num_cells):
        plot_weights(folder_name, alphas[ncell][:][:], ncell, cell_name)