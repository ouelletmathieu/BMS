from boolean.plot_util import markerVec, colorVec
import matplotlib.pyplot as plt
from boolean.symm_util import  get_symmetry_number
import numpy as np
import pandas as pd
import glob
import os

nodeNb = 7
dt_string = "02_11_08_08"
main_path = "./data_example/pareto_density_cycle/"+dt_string+"/"

path_7node_wrong_motif = "./data_example/symmetry/symmetry_wrong_motif_network_nodes_7.txt"
path_7node_random = "./data_example/symmetry/symmetry_random_network_nodes_7.txt"

fig_path = "./fig/partial_symmetry.pdf"

def getAverage_sym_nb(path, density_min):
    
    width = 0.1
    bin_density = np.arange(0,1,width)
    mid_point = bin_density+ 0.5*width
    average_sym_ratio, average_sym_ratio_cycle, average_sym_ratio_transcient,count = [0]*len(bin_density), [0]*len(bin_density), [0]*len(bin_density), [0]*len(bin_density)
    df_wrong_motif = pd.read_csv(path)   

    list_wrong_motif_density = df_wrong_motif["density"].tolist()
    list_wrong_motif_symmetry_ratio,list_wrong_motif_symmetry_ratio_transcient = df_wrong_motif["nb_sym"].tolist(),df_wrong_motif["nb_sym_transcient"].tolist()
    list_wrong_motif_symmetry_ratio_cycle = df_wrong_motif["nb_sym_cycle"].tolist()

    if 'density_from_motif' in df_wrong_motif.columns:
        list_wrong_motif_density_from_motif = df_wrong_motif["density_from_motif"].tolist()
    else:
        list_wrong_motif_density_from_motif = [0]*len(list_wrong_motif_density)


    list_wrong_motif_density_high_motif,list_wrong_motif_density_from_motif_high_motif,list_wrong_motif_symmetry_ratio_high_motif = [],[],[]
    list_wrong_motif_symmetry_ratio_transcient_high_motif,list_wrong_motif_symmetry_ratio_cycle_high_motif = [],[]
    
    average_sym_ratio_dic = { i:[] for i in range(len(bin_density))}
    average_sym_ratio_transcient_dic = { i:[] for i in range(len(bin_density))}
    average_sym_ratio_cycle_dic = { i:[] for i in range(len(bin_density))}
            
            
    #check minimal density for motif
    for i in range(len(list_wrong_motif_density)):
        if list_wrong_motif_density_from_motif[i]/list_wrong_motif_density[i] >= density_min:
            list_wrong_motif_density_high_motif.append(list_wrong_motif_density[i])
            list_wrong_motif_density_from_motif_high_motif.append(list_wrong_motif_density_from_motif[i])
            list_wrong_motif_symmetry_ratio_high_motif.append(list_wrong_motif_symmetry_ratio[i])
            list_wrong_motif_symmetry_ratio_transcient_high_motif.append(list_wrong_motif_symmetry_ratio_transcient[i])
            list_wrong_motif_symmetry_ratio_cycle_high_motif.append(list_wrong_motif_symmetry_ratio_cycle[i])
            
    
    for i in range(len(list_wrong_motif_symmetry_ratio_high_motif)):
        if list_wrong_motif_density_high_motif[i]!= 1.0:
            index_density = int(list_wrong_motif_density_high_motif[i]/width)
            average_sym_ratio[index_density]+=list_wrong_motif_symmetry_ratio_high_motif[i]
            average_sym_ratio_transcient[index_density] +=list_wrong_motif_symmetry_ratio_transcient_high_motif[i]
            average_sym_ratio_cycle[index_density]+= list_wrong_motif_symmetry_ratio_cycle_high_motif[i]
            average_sym_ratio_dic[index_density].append(list_wrong_motif_symmetry_ratio_high_motif[i])
            average_sym_ratio_transcient_dic[index_density].append(list_wrong_motif_symmetry_ratio_transcient_high_motif[i])
            average_sym_ratio_cycle_dic[index_density].append(list_wrong_motif_symmetry_ratio_cycle_high_motif[i])
            count[index_density]+=1
    
    for i in range(len(count)):
        if count[i]>0:
            average_sym_ratio[i]= average_sym_ratio[i]/count[i]
            average_sym_ratio_cycle[i]= average_sym_ratio_cycle[i]/count[i]
            average_sym_ratio_transcient[i]= average_sym_ratio_transcient[i]/count[i]
    
    return mid_point, average_sym_ratio, average_sym_ratio_transcient, average_sym_ratio_cycle, [average_sym_ratio_dic, average_sym_ratio_transcient_dic, average_sym_ratio_cycle_dic]

def getAverage_sym_ratio(path, density_min):
    
    width = 0.1
    bin_density = np.arange(0,1,width)
    mid_point = bin_density+ 0.5*width
    average_sym_ratio = [0]*len(bin_density)
    average_sym_ratio_cycle = [0]*len(bin_density)   
    average_sym_ratio_transcient = [0]*len(bin_density)   
    count = [0]*len(bin_density)
    
    df_wrong_motif = pd.read_csv(path)   

    list_wrong_motif_symmetry_ratio, list_wrong_motif_symmetry_ratio_transcient = df_wrong_motif["ratio_sym"].tolist(), df_wrong_motif["ratio_sym_transcient"].tolist()
    list_wrong_motif_symmetry_ratio_cycle = df_wrong_motif["ratio_sym_cycle"].tolist()
    list_wrong_motif_density  = df_wrong_motif["density"].tolist()
    

    if 'density_from_motif' in df_wrong_motif.columns:
        list_wrong_motif_density_from_motif = df_wrong_motif["density_from_motif"].tolist()
    else:
        list_wrong_motif_density_from_motif = [0]*len(list_wrong_motif_density)

    list_wrong_motif_density_high_motif, list_wrong_motif_density_from_motif_high_motif, list_wrong_motif_symmetry_ratio_high_motif = [], [], []
    list_wrong_motif_symmetry_ratio_transcient_high_motif, list_wrong_motif_symmetry_ratio_cycle_high_motif = [], []
    
    average_sym_ratio_dic, average_sym_ratio_transcient_dic = { i:[] for i in range(len(bin_density))}, { i:[] for i in range(len(bin_density))}
    average_sym_ratio_cycle_dic = { i:[] for i in range(len(bin_density))}        
    
    #check minimal density for motif
    for i in range(len(list_wrong_motif_density)):

        
        if list_wrong_motif_density_from_motif[i]/list_wrong_motif_density[i] >= density_min:
            list_wrong_motif_density_high_motif.append(list_wrong_motif_density[i])
            list_wrong_motif_density_from_motif_high_motif.append(list_wrong_motif_density_from_motif[i])
            list_wrong_motif_symmetry_ratio_high_motif.append(list_wrong_motif_symmetry_ratio[i])
            list_wrong_motif_symmetry_ratio_transcient_high_motif.append(list_wrong_motif_symmetry_ratio_transcient[i])
            list_wrong_motif_symmetry_ratio_cycle_high_motif.append(list_wrong_motif_symmetry_ratio_cycle[i])
            
    for i in range(len(list_wrong_motif_symmetry_ratio_high_motif)):
        if list_wrong_motif_density_high_motif[i]!= 1.0:
            index_density = int(list_wrong_motif_density_high_motif[i]/width)
            average_sym_ratio[index_density]+=list_wrong_motif_symmetry_ratio_high_motif[i]
            average_sym_ratio_transcient[index_density] +=list_wrong_motif_symmetry_ratio_transcient_high_motif[i]
            average_sym_ratio_cycle[index_density]+= list_wrong_motif_symmetry_ratio_cycle_high_motif[i]
            count[index_density]+=1
            average_sym_ratio_dic[index_density].append(list_wrong_motif_symmetry_ratio_high_motif[i])
            average_sym_ratio_transcient_dic[index_density].append(list_wrong_motif_symmetry_ratio_transcient_high_motif[i])
            average_sym_ratio_cycle_dic[index_density].append(list_wrong_motif_symmetry_ratio_cycle_high_motif[i])
            
    for i in range(len(count)):
        if count[i]>0:
            average_sym_ratio[i]= average_sym_ratio[i]/count[i]
            average_sym_ratio_cycle[i]= average_sym_ratio_cycle[i]/count[i]
            average_sym_ratio_transcient[i]= average_sym_ratio_transcient[i]/count[i]
    
    return mid_point, average_sym_ratio, average_sym_ratio_transcient, average_sym_ratio_cycle, [average_sym_ratio_dic, average_sym_ratio_transcient_dic, average_sym_ratio_cycle_dic]
    
def script_get_pareto_symmetries(nodeNb, main_path):
 
    width = 0.1
    bin_density = np.arange(0,1,width)
    mid_point = bin_density+ 0.5*width
    
    average_sym_ratio, average_sym_ratio_cycle, average_sym_ratio_transcient, count = [0]*len(bin_density), [0]*len(bin_density), [0]*len(bin_density), [0]*len(bin_density)

    average_sym_ratio_dic, average_sym_ratio_cycle_dic = {i:[] for i in range(len(bin_density))}, {i:[] for i in range(len(bin_density))}
    average_sym_ratio_transcient_dic = {i:[] for i in range(len(bin_density))}
    average_sym_nb, average_sym_nb_cycle, average_sym_nb_transcient = [0]*len(bin_density), [0]*len(bin_density), [0]*len(bin_density) 
    
    average_sym_ratio_dic_nb, average_sym_ratio_cycle_dic_nb = {i:[] for i in range(len(bin_density))}, {i:[] for i in range(len(bin_density))}
    average_sym_ratio_transcient_dic_nb  = {i:[] for i in range(len(bin_density))}
    
  
    list_file_foreach_run = get_all_run_file(main_path) 
        
    for file in list_file_foreach_run:
        
        df_pop = pd.read_csv(file, encoding = 'utf-8')
        list_mat = df_pop["mat"].tolist()
        
        for mat_string in list_mat:
        
            mat =  np.reshape(np.matrix(mat_string),(nodeNb,nodeNb))

            density = np.count_nonzero(mat)/(nodeNb**2)
            nb_sym, nb_sym_transcient, nb_sym_cycle, ratio_sym, ratio_sym_transcient, ratio_sym_cycle, _ =  get_symmetry_number(mat)

            index_density = int(density/width)
            
            if density != 1.0:
                average_sym_ratio[index_density] += ratio_sym
                average_sym_ratio_cycle[index_density] += ratio_sym_cycle
                average_sym_ratio_transcient[index_density] += ratio_sym_transcient
                
                average_sym_ratio_dic[index_density].append(ratio_sym) 
                average_sym_ratio_cycle_dic[index_density].append(ratio_sym_cycle)
                average_sym_ratio_transcient_dic[index_density].append(ratio_sym_transcient)
                                                 
                count[index_density] +=1
                average_sym_nb[index_density]+= nb_sym
                average_sym_nb_cycle[index_density] += nb_sym_cycle
                average_sym_nb_transcient[index_density]+= nb_sym_transcient
                                                 
                average_sym_ratio_dic_nb[index_density].append(nb_sym)
                average_sym_ratio_cycle_dic_nb[index_density].append(nb_sym_cycle)
                average_sym_ratio_transcient_dic_nb[index_density].append(nb_sym_transcient)
  
    dic_avg = [average_sym_ratio_dic, average_sym_ratio_cycle_dic, average_sym_ratio_transcient_dic]
    dic_nb  = [average_sym_ratio_dic_nb, average_sym_ratio_cycle_dic_nb, average_sym_ratio_transcient_dic_nb]
                                                                                             
    for i in range(len(count)):
        if count[i]>0:
            average_sym_ratio[i] = average_sym_ratio[i]/count[i]
            average_sym_ratio_cycle[i] = average_sym_ratio_cycle[i]/count[i]
            average_sym_ratio_transcient[i] = average_sym_ratio_transcient[i]/count[i]
            average_sym_nb[i] = average_sym_nb[i]/count[i]
            average_sym_nb_cycle[i] = average_sym_nb_cycle[i]/count[i]
            average_sym_nb_transcient[i] = average_sym_nb_transcient[i]/count[i]
            
    return [mid_point, average_sym_ratio, average_sym_ratio_transcient, average_sym_ratio_cycle, dic_avg], [mid_point,  average_sym_nb, average_sym_nb_transcient, average_sym_nb_cycle, dic_nb]
    
def get_all_run_file(main_path) : 

    list_run = glob.glob(main_path+"run*")
    
    list_file_foreach_run = list()
    
    for run in list_run:
        
        list_pop = glob.glob(run+"/pop/*")
        last_pop_file = 0
        last_pop_index = 0

        for pop in list_pop:
            _, file_name = os.path.split(pop)
            index_pop = file_name.split('_')[0]
           
            if int(index_pop) > last_pop_index:
                last_pop_index = int(index_pop)
                last_pop_file = pop
            
        list_file_foreach_run.append(last_pop_file)
        
    
    return list_file_foreach_run

def plot_density_sym_ratio():
    


    array_ratio, array_nb = script_get_pareto_symmetries(nodeNb,main_path)
    mid_point, average_sym_ratio, _ , _, _ = getAverage_sym_ratio(path_7node_wrong_motif,0.7)
    mid_point_rand, average_sym_ratio_rand, _, _, _ = getAverage_sym_ratio(path_7node_random, -1)
    _, average_sym_ratio_pareto, _, _, _ = array_ratio
    mid_point, average_sym_nb, _ , _, _ = getAverage_sym_nb(path_7node_wrong_motif,0.7)
    mid_point_rand, average_sym_nb_rand, _, _, _ = getAverage_sym_nb(path_7node_random,-1)
    _, average_sym_nb_pareto, _, _, _ = array_nb

    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

    n=0
    axs[0].plot(mid_point_rand[1:],average_sym_ratio_pareto[1:],  marker = markerVec[n], label='pareto', color = colorVec[n], linestyle="None")
    axs[1].plot(mid_point[1:],average_sym_nb_pareto[1:],  marker = markerVec[n], label='pareto', color = colorVec[n], linestyle="None")

    n=1
    axs[0].plot(mid_point_rand[1:],average_sym_ratio_rand[1:],  marker = markerVec[n], label='random', color = colorVec[n], linestyle="None")
    axs[1].plot(mid_point[1:],average_sym_nb_rand[1:],  marker = markerVec[n], label='random', color = colorVec[n], linestyle="None")

    n=2
    axs[0].plot(mid_point[1:],average_sym_ratio[1:],  marker = markerVec[n], label='motif', color = colorVec[n], linestyle="None")
    axs[1].plot(mid_point[1:],average_sym_nb[1:],  marker = markerVec[n], label='motif', color = colorVec[n], linestyle="None")

    axs[1].legend()

    axs[0].set_xlabel("network density")
    axs[1].set_xlabel("network density")
    axs[0].set_ylabel("partial symmetries")
    axs[1].set_ylabel("complete symmetries ") 
    axs[0].set_xlim([0,1])
    axs[1].set_xlim([0,1])
    
    plt.savefig(fig_path, transparent=True)

plot_density_sym_ratio() 