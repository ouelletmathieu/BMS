import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os
import random
import ast
from boolean.boolean_util import listToArray, generateRandomValidNetwork
from boolean.plot_util import get_average_x_y, markerVec, colorVec
from boolean.genetic_util import get_pareto

n_random = 20*200
nodeNb = 7
nb_pareto_toKeep = 2
min_density = 0.2
 
dt_string = "02_11_08_08"
main_path = "./data_example/pareto_density_cycle/"+dt_string+"/"

path_wrong_motif = "./data_example/symmetry/symmetry_wrong_motif_network_nodes_"+str(nodeNb)+".txt"

fig_path = "./fig/balanced.pdf"


def main():
    df_wrong_motif = pd.read_csv(path_wrong_motif)

    
    mat_list_wrong_motif = []

    for mat_list in df_wrong_motif['mat'].to_list():
        mat_list = mat_list.replace(';',",")
        vec = ast.literal_eval(mat_list)
        mat_list_wrong_motif.append(listToArray(np.array(vec), 7))


    density_wrong_motif, balanced_wrong_motif, sem_wrong_motif = get_density_ratio_balanced_SEM(mat_list_wrong_motif)
    mat_list_random = [ generateRandomValidNetwork(7, 0.2+0.8*random.random()) for x in range(n_random) ] 
    density_random, balanced_random, sem_random = get_density_ratio_balanced_SEM(mat_list_random)



    pop_pareto = get_mat_for_balanced_ratio_pareto()
    density_pareto, balanced_pareto, sem_pareto = get_density_ratio_balanced_SEM(pop_pareto)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(2*4, 1*6))


    ax = axs[0]
    n=0
    imsh = ax.errorbar(density_pareto, balanced_pareto, yerr=sem_pareto, marker = markerVec[n], color = colorVec[n] , label='pareto', ls='None')
    n=1
    imsh = ax.errorbar(density_random, balanced_random, yerr=sem_random, marker = markerVec[n], color = colorVec[n] , label='random', ls='None')
    n=2
    imsh = ax.errorbar(density_wrong_motif, balanced_wrong_motif, yerr=sem_wrong_motif, marker = markerVec[n], color = colorVec[n] , label='wrong', ls='None')
    ax.legend()
    ax.set_xlim([0.15, 0.95])


    ax = axs[1]
    n=0
    density_pareto, distance_pareto, sem_distance_pareto = get_density_distance_balanced_SEM(pop_pareto)
    imsh = ax.errorbar(density_pareto, distance_pareto, yerr=sem_distance_pareto, marker = markerVec[n], color = colorVec[n] , label='pareto', ls='None')
    n=1
    density_random, distance_random, sem_distance_random = get_density_distance_balanced_SEM(mat_list_random)
    imsh = ax.errorbar(density_random, distance_random, yerr=sem_distance_random, marker = markerVec[n], color = colorVec[n] , label='random', ls='None')
    n=2
    density_wrong, distance_wrong, sem_distance_wrong = get_density_distance_balanced_SEM(mat_list_wrong_motif)
    imsh = ax.errorbar(density_wrong, distance_wrong, yerr=sem_distance_wrong, marker = markerVec[n], color = colorVec[n] , label='wrong', ls='None')
    ax.legend()
    ax.set_xlim([0.15, 0.95])


    plt.savefig(fig_path, transparent=True)

def get_mat_for_balanced_ratio_pareto():

    all_mat = []

    weights = [-1,1]
    
    list_run = glob.glob(main_path+"run*")
    
    list_file_foreach_run = list()
    list_index_foreach_run = list()
    list_pareto_pop = list()
    
    x_cycle = list()
    y_symm = list()
    
    ##############################################
    #   Getting the pareto front for each run    #
    ##############################################  
    
    for run in list_run:
        
        path_run, file_name_run = os.path.split(run)
        
        
        index_run = int(file_name_run.split('n')[1])
        
        list_pop = glob.glob(run+"/pop/*")
        last_pop_file = 0
        last_pop_index = 0
        
        for pop in list_pop:
            path, file_name = os.path.split(pop)
            index_pop = file_name.split('_')[0]
           
            if int(index_pop) > last_pop_index:
                last_pop_index = int(index_pop)
                last_pop_file = pop
 
        list_index_foreach_run.append(index_run)
        list_file_foreach_run.append(last_pop_file)
    
    
    n_run = len(list_run)
        
    for i in range(len(list_file_foreach_run)):
        
        df = pd.read_csv(list_file_foreach_run[i])
        
        mat_list = df["mat"]
        max_cycle_list = df["max_cycle"]
        density_list = df["density"]      
        
        pop = list()
        
        for i, mat_txt in enumerate(mat_list):
            density = density_list[i]
            cycle = max_cycle_list[i]
            vec = ast.literal_eval(mat_txt)
            pop.append([density, round(cycle), vec])
        
        pareto_pop = get_pareto(pop, weights, front_toKeep = nb_pareto_toKeep, min_density = min_density)
        list_pareto_pop.append(pareto_pop)
        
        for elem_of_front in pareto_pop:
            all_mat.append(listToArray(elem_of_front[2], nodeNb))

    return all_mat
            
def get_density_ratio_balanced_SEM(mat_list):
    
    
    density_list, balanced  = [], []
    
    for mat in mat_list:

        nodeNb = mat.shape[0]
        density = np.count_nonzero(mat)/(nodeNb**2)
        sum_input = np.sum(mat,0)
        ratio_balanced = np.count_nonzero(sum_input == 1)/nodeNb

        #rounding density to plot and average ot the nearest 0.05, example 0.36 -> 0.35 and 0.377 -> 0.40
        density = round(density * 10) / 10

        density_list.append(density)
        balanced.append(ratio_balanced)
    
    d = {'density': density_list, 'ratio': balanced}
    df = pd.DataFrame(data=d)

    X,Y, STD = get_average_x_y(df, 'density', 'ratio', SEM=True, min_count = 3)
    
    return X,Y, STD

def get_density_distance_balanced_SEM(mat_list):
    
    
    density_list, distance_balanced  = [], []
    
    for mat in mat_list:
    

        nodeNb = mat.shape[0]


        density = np.count_nonzero(mat)/(nodeNb**2)
        sum_input = np.sum(mat,0)
        
        tot = 0
        for i in sum_input:
            tot+= (i-1)**2
            
            
        distance= tot/nodeNb

        #rounding density to plot and average ot the nearest 0.05, example 0.36 -> 0.35 and 0.377 -> 0.40
        density = round(density * 10) / 10


        density_list.append(density)
        distance_balanced.append(distance)
    

    d = {'density': density_list, 'distance': distance_balanced}
    df = pd.DataFrame(data=d)

    X,Y, STD = get_average_x_y(df, 'density', 'distance', SEM=True, min_count = 3)
    
    return X,Y, STD


if __name__ == "__main__":
    main()
