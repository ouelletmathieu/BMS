import pandas as pd
import matplotlib.pyplot as plt
import random
from deap.benchmarks.tools import igd
import os
import glob
from matplotlib import gridspec
from boolean.genetic_util import get_pareto_from_pop_file
from boolean.plot_util import colorBlue_3pc, colorOrange, get_all

def script_plot_all_run_pareto():

    
    weights = [-1,1]
    
    dt_string = "02_11_08_08"
    main_path = "./data_example/pareto_density_cycle/"+dt_string+"/"
    path_random = "./data_example/1d_properties/1d_properties_nodes_7.txt"
    path_figure = "./fig/pareto_front.pdf"

    ###############################
    #            PARETO           #
    ###############################
    
    
    df_sample = pd.read_csv(path_random)
        
    list_run = glob.glob(main_path+"run*")
    
    list_pareto_foreach_run = list()
    list_index_foreach_run = list()
    list_file_foreach_run = list()
    
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
                
        pareto = get_pareto_from_pop_file(last_pop_file, weights)
        list_pareto_foreach_run.append(pareto)
        list_index_foreach_run.append(index_run)
        list_file_foreach_run.append(last_pop_file)
        
    fig = plt.figure(figsize=(6, 6)) 
    gs = gridspec.GridSpec(1, 1, width_ratios=[1]) 

    ax = plt.subplot(gs[0])
    ax.set_title(str("pareto front"))
    ax.set( xlabel='density', ylabel='max. cycle')
    
    x_sampled, y_sampled = get_all(df_sample, 'density', 'max_cycle')
    for i in range(len(x_sampled)):
        x_sampled[i] =  x_sampled[i] + 0.04*(random.random()-0.5)
        y_sampled[i] =  y_sampled[i] + 0.3*(random.random()-0.5)
    
        if x_sampled[i]  < 0 :
            x_sampled[i] = 0
        if x_sampled[i]  > 1 :
            x_sampled[i] = 1    
            
    x_radius_keep, y_radius_keep = 0.005, 0.01
    x_keep,y_keep  = [], []

    for i in range(len(x_sampled)):
        
        x_live,y_live = x_sampled[i], y_sampled[i]
        keep = True
        
        for j in range(len(x_keep)):
            if abs(x_live-x_keep[j])<x_radius_keep and abs(y_live-y_keep[j])<y_radius_keep:
                keep = False
                break           
        if keep:
            x_keep.append(x_live)
            y_keep.append(y_live)
    
    imsh =  ax.plot(x_keep, y_keep, 'o',color= colorBlue_3pc, ls='None') #, alpha=0.005)

    for i in range(len(list_pareto_foreach_run)):
        X = list()
        Y = list()
        
        for vec in list_pareto_foreach_run[i]:
            Y.append(vec[0])
            X.append(vec[1])
            
        X_sorted = [x for x , y in sorted(zip(X,Y), key=lambda pair: pair[0])]
        Y_sorted = [y for x , y in sorted(zip(X,Y), key=lambda pair: pair[0])]

        color_perturbed = list(colorOrange)
        color_perturbed[0] = color_perturbed[0] + 0.3*(random.random()-0.5)
        color_perturbed[1] = color_perturbed[1] + 0.3*(random.random()-0.5)
        color_perturbed[2] = color_perturbed[2] + 0.3*(random.random()-0.5)

        if color_perturbed[0] > 1.0 :
            color_perturbed[0] =1.0
        if color_perturbed[0] < 0 :
            color_perturbed[0] =0
        
        if color_perturbed[1] > 1.0 :
            color_perturbed[1] =1.0
        if color_perturbed[1] < 0 :
            color_perturbed[1] =0
        
        if color_perturbed[2] > 1.0 :
            color_perturbed[2] =1.0   
        if color_perturbed[2] < 0 :
            color_perturbed[2] =0 
             
        imsh = ax.plot(X_sorted,Y_sorted, marker='o',  color=  tuple(color_perturbed))

    plt.savefig(path_figure, transparent=True)
    
script_plot_all_run_pareto()