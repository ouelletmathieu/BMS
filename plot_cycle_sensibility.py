import pandas as pd
import matplotlib.pyplot as plt
from boolean.plot_util import get_average_x_y, colorVec, markerVec, get_quartile_x_y, get_density_x_y
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

"""This file contain two scripts :

      1. plot_cycle_sensibility: plot the average cycle lost shown in Figure 1. Need to run sampling_cycle_sensibility before for 5,6,7,8 nodes. 
      
"""

def plot_cycle_sensibility():


    path = './data_example/cycle_sensibility/edge_removed_nodes_'
    subpath = '.txt'
    nodes_nb = [5,6,7,8]


    dscale = 4/72
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(19, 6))

    for n, node in enumerate(nodes_nb) :
        
        #move the plot a little bit so it is easier to read
        move_by = dscale*(n - int(len(nodes_nb)/2))
        
        path_file = path+str(node)+subpath
        df = pd.read_csv(path_file)
        
        ax = axs[0]
        X,Y,STD = get_average_x_y(df, 'max_cycle', 'max_delta',  min_count = 1 )
        trans1 = ax.transData + ScaledTranslation(move_by, 0, fig.dpi_scale_trans)
        imsh = ax.errorbar(X,Y, yerr=STD, label=str(node)+' nodes (max.)', ls='None', transform=trans1, marker = markerVec[n], color = colorVec[n])
        ax.set_title(str('maximal sensibility'))
        ax.set( xlabel='max. cycle lenght', ylabel=' cycle lenght variation')
        ax.legend()
        ax.set_xlim([7, 17])
        
        ax = axs[1]
        trans1 = ax.transData + ScaledTranslation(move_by, 0, fig.dpi_scale_trans)
        X,Y, STD = get_average_x_y(df, 'max_cycle', 'avg_delta',  min_count = 1)
        imsh = ax.errorbar(X, Y, yerr=STD, label=str(node)+' nodes (avg.)', ls='None', transform=trans1 ,marker = markerVec[n], color = colorVec[n])
        ax.set_title(str('average sensibility'))
        ax.set( xlabel='max. cycle lenght', ylabel=' cycle lenght variation')
        ax.legend()
        ax.set_xlim([7, 17])
        
        ax = axs[2]
        trans1 = ax.transData + ScaledTranslation(move_by, 0, fig.dpi_scale_trans)
        X_pc, Y_pc, STD_pc = list(), list(), list()
        for i in range(len(X)):
            X_pc.append(X[i])
            Y_pc.append( Y[i]/(X[i]-1) )
            STD_pc.append( STD[i]/(X[i]-1) )
        imsh = ax.errorbar(X_pc, Y_pc, yerr=STD_pc, label=str(node)+' nodes (percent)', ls='None', transform=trans1,marker = markerVec[n], color = colorVec[n])
        ax.set_title(str('average sensibility'))
        ax.set( xlabel='max. cycle lenght', ylabel=' cycle lenght variation')
        ax.legend()
        ax.set_xlim([7, 17])
        
        plt.savefig("./fig/edge_removed.pdf", transparent=True)

if __name__ == "__main__":
    plot_cycle_sensibility()
