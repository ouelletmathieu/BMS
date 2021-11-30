import pandas as pd
import matplotlib.pyplot as plt
from boolean.plot_util import get_average_x_y, colorVec, markerVec, get_quartile_x_y


def main():

    path_file = "./data_example/1d_properties_nodes_7.txt"

    df = pd.read_csv(path_file)

    nrows, ncols = 2, 2

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*6, nrows*6))
    fig.subplots_adjust(wspace = 0.3 )

    #plot 1
    ax = axs[0][0]
    X,Y, STD = get_average_x_y(df, 'density', 'max_cycle', SEM=True, min_count = 3)
    imsh = ax.errorbar(X,Y, yerr=STD, marker=markerVec[0] ,color=colorVec[0] , label='Average', ls='None')
    ax.set_title(str('max. cycle (SEM)'))
    ax.set( xlabel='density', ylabel='max. cycle (average)')
    ax.legend(loc=0)
    X1,Y1 = get_quartile_x_y(df, 'density', 'max_cycle', quartile=99.95)
    ax2=ax.twinx()
    imsh2 = ax2.plot(X1,Y1, marker=markerVec[1],color=colorVec[1], ls='None', label = 'Maximum')
    ax2.set( xlabel='density', ylabel='max. cycle  (maximum)')
    ax2.legend(loc=1)
    
    #plot 2
    ax = axs[0][1]
    X,Y, STD = get_average_x_y(df, 'totalInhibition', 'max_cycle', SEM=True, min_count = 3)
    X1,Y1, STD1 = get_average_x_y(df, 'totalExcitation', 'max_cycle', SEM=True, min_count = 3)
    imsh = ax.errorbar(X,Y, yerr=STD,marker=markerVec[0],color=colorVec[0], ls='None', label='Inhibition')
    imsh2 = ax.errorbar(X1,Y1,  yerr=STD1,marker=markerVec[1],color=colorVec[1], ls='None', label='Excitation')
    ax.set_title(str('max. cycle by edge types (SEM)'))
    ax.set( xlabel='edge counts', ylabel='max. cycle (average)')
    ax.legend()

    #plot 3
    ax = axs[1][0]
    X2,Y2, STD2 = get_average_x_y(df, 'totalAutoInhibition', 'max_cycle', SEM=True, min_count = 3)
    X3,Y3, STD3 = get_average_x_y(df, 'totalAutoExcitation', 'max_cycle', SEM=True, min_count = 3)
    imsh3 = ax.errorbar(X2,Y2,yerr=STD2,marker=markerVec[0],color=colorVec[0], ls='None' , label='Auto Inhibition')
    imsh4 = ax.errorbar(X3,Y3, yerr=STD3,marker=markerVec[1],color=colorVec[1], ls='None', label='Auto Excitation')
    ax.set_title(str('max. cycle by self loop types (SEM)'))
    ax.set( xlabel='self loop counts', ylabel='max. cycle (average)')
    ax.legend()

    #plot 3
    ax = axs[1][1]
    X2,Y2, STD2 = get_average_x_y(df, 'mxc_inhib', 'max_cycle', SEM=True, min_count = 3)
    X3,Y3, STD3 = get_average_x_y(df, 'mxc_exct', 'max_cycle', SEM=True, min_count = 3)
    imsh3 = ax.errorbar(X2,Y2,yerr=STD2, marker=markerVec[0],color=colorVec[0], ls='None', label='Avg. Inhibition Cycle')
    imsh4 = ax.errorbar(X3,Y3,  yerr=STD3, marker=markerVec[1],color=colorVec[1], ls='None', label='Avg. Excitation Cycle')
    ax.set_title(str('max. cycle by loop lenght (SEM)'))
    ax.set( xlabel='avg. loop lenght', ylabel='max. cycle (average)')
    ax.legend()

    plt.savefig("./fig/1d_properties_random_graph.pdf", transparent=True)


if __name__ == "__main__":
    main()