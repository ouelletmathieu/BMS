import pandas as pd
import matplotlib.pyplot as plt
from boolean.plot_util import get_average_x_y, colorVec, markerVec, get_quartile_x_y, get_density_x_y

"""This file contain two scripts :

      1. plot_mating_comparison: plot the comparison between the mating strategies in the supplementary material. 
      
"""

def plot_mating_comparison():

    node = 7

    path_random = "./data_example/mating_strategy/random_mating_nodes_"+ str(node) +".txt"
    path_mix = "./data_example/mating_strategy/swap_mating_nodes_"+ str(node) +".txt"
    path_cycle = './data_example/mating_strategy/cycle_mating_nodes_'+ str(node) +'.txt'

    df_random = pd.read_csv(path_random)
    df_mix = pd.read_csv(path_mix)
    df_cycle = pd.read_csv(path_cycle)

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    fig.subplots_adjust(wspace = 0.3 )

    ax = axs

    X,Y, STD = get_average_x_y(df_random, 'cycle1', 'cycle3', min_count = 5)
    X_mix,Y_mix, STD = get_average_x_y(df_mix, 'cycle1', 'cycle3', min_count = 5)
    X_cycle,Y_cycle, STD = get_average_x_y(df_cycle, 'cycle1', 'cycle3', min_count = 5)

    imsh = ax.plot(X,Y, marker='o',color=colorVec[2] , label='Random', ls='None')
    imsh = ax.plot(X_mix,Y_mix, marker='o',color=colorVec[1] , label='Mixed', ls='None')
    imsh = ax.plot(X_cycle,Y_cycle, marker='o',color=colorVec[0] , label='Cycle', ls='None')

    ax.set_xlim([0,11])

    ax.set_title(str('mating stategies'))
    ax.set( xlabel='parent\'s cycle', ylabel='child\'s cycle')
    ax.legend(loc=0)

    plt.savefig("./fig/mating_strategy.pdf", transparent=True)
        


if __name__ == "__main__":
    plot_mating_comparison()


