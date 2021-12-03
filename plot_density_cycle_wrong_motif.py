import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest
from boolean.plot_util import markerVec, colorVec, get_density_x_y_with_count, two_proprotions_confint


xbound = 8
nodeNb = 7

path_file = "./data_example/1d_properties/1d_properties_nodes_7.txt"
path_wrong_motif = "./data_example/wrong_motif_network/wrong_motif_nodes_7.txt"
path_random_motif = "./data_example/wrong_motif_network/random_motif_nodes_7.txt"
path_fig = "./fig/density_cycle_wrong_motif.pdf"

#load data
df = pd.read_csv(path_file)
df_wrong_motif = pd.read_csv(path_wrong_motif)
df_random_motif = pd.read_csv(path_random_motif)

#random networks density
X,Y, count, total = get_density_x_y_with_count(df, 'max_cycle', check_cond = False)
X_capped, Y_capped = [], [] 
for i in range(len(X)):
    if X[i]< xbound:
        X_capped.append(X[i])
        Y_capped.append(Y[i])
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
axs[0].set_yscale('log')
imsh = axs[0].plot(X_capped,Y_capped, marker = markerVec[0],   color = colorVec[0], label=str(nodeNb)+' nodes', linestyle="None")
axs[0].set_title(str('max. cycle density'))
axs[0].set( xlabel='max. cycle lenght', ylabel='density (log)')
    
#wrong motifs networks density
X_wrong,Y_wrong, count_wrong, total_wrong = get_density_x_y_with_count(df_wrong_motif, 'max_cycle',  cond_col = 'density_from_motif', min_col = 0.75, toPrint = False)
X_wrong_capped,Y_wrong_capped = [],[]
for i in range(len(X_wrong)):
    if X_wrong[i]< xbound:
        X_wrong_capped.append(X_wrong[i])
        Y_wrong_capped.append(Y_wrong[i])
X_wrong, Y_wrong = X_wrong_capped, Y_wrong_capped
imsh = axs[0].plot(X_wrong_capped,Y_wrong_capped, marker = markerVec[1], color = colorVec[1], label='motifs (wrong motif)', linestyle="None")

#random motifs networks density
X_random_motif,Y_random_motif, count_random_motif, total_random_motif= get_density_x_y_with_count(df_random_motif, 'max_cycle',  cond_col = 'density_from_motif', min_col = 0.75)
X_random_motif_capped,Y_random_motif_capped = [], [] 
for i in range(len(X_random_motif)):
    if X_random_motif[i]< xbound:
        X_random_motif_capped.append(X_random_motif[i])
        Y_random_motif_capped.append(Y_random_motif[i])
X_random_motif, Y_random_motif = X_random_motif_capped, Y_random_motif_capped
imsh = axs[0].plot(X_random_motif_capped,Y_random_motif_capped, marker = markerVec[2], color = colorVec[2], label='motifs (random selection)', linestyle="None")
axs[0].legend()
axs[0].set_xlim([0,xbound])

list_x_ratio, list_y_ratio = list(), list()
pval_list_wrong, pval_list_random_motif = [0]*len(range(1,xbound)), [0]*len(range(1,xbound))
list_wrong_difference, list_wrong_confidence_lower, list_wrong_confidence_upper  = [0]*len(range(1,xbound)), [0]*len(range(1,xbound)), [0]*len(range(1,xbound))
list_random_motif_difference, list_random_motif_confidence_lower, list_random_motif_confidence_upper  = [0]*len(range(1,xbound)), [0]*len(range(1,xbound)), [0]*len(range(1,xbound))

n_done = 0
min_pval = 1*10**(-6)

for i in range(1,xbound):
    
    count_wrong_to_test = np.array([count[i], count_wrong[i]])
    nobs_wrong_to_test = np.array([total, total_wrong])
    stat, pval = proportions_ztest(count_wrong_to_test, nobs_wrong_to_test)
    if pval < min_pval :
        pval = min_pval
    pval_list_wrong[n_done] = pval
    diff, conf_interval = two_proprotions_confint(count[i], total, count_wrong[i], total_wrong, significance = 0.05)
    prop_count= count[i]/total
    
    print("random ratio", count[i], total,count[i]/total ,  sep= " , ")
    print("motif ratio", count_wrong[i], total_wrong,count_wrong[i]/total_wrong ,  sep= " , ")      
    print("stat: ", stat, "pval: ", pval, sep=" , ")
    print("difference: ", diff, "confidence: ", conf_interval, sep=" , ")
    
    list_wrong_difference[n_done] = diff/prop_count
    list_wrong_confidence_lower[n_done] = diff/prop_count - conf_interval[0]/prop_count
    list_wrong_confidence_upper[n_done] = conf_interval[1]/prop_count  - diff/prop_count
    print(diff/prop_count,conf_interval[0]/prop_count,conf_interval[1]/prop_count, sep = " , " )   
    
    count_random_to_test = np.array([count[i], count_random_motif[i]])
    nobs_random_to_test = np.array([total, total_random_motif])
    stat, pval = proportions_ztest(count_random_to_test, nobs_random_to_test)
    pval_list_random_motif[n_done] = pval
    if pval < min_pval :
        pval = min_pval
    diff, conf_interval = two_proprotions_confint(count[i], total, count_random_motif[i], total_random_motif, significance = 0.05)
    
    print("random ratio", count[i], total,count[i]/total ,  sep= " , ")
    print("motif ratio", count_random_motif[i], total_random_motif,count_random_motif[i]/total_random_motif ,  sep= " , ")      
    print("stat: ", stat, "pval: ", pval, sep=" , ")
    print("difference: ", diff, "confidence: ", conf_interval, sep=" , ")
    
    list_random_motif_difference[n_done] = diff/prop_count
    list_random_motif_confidence_lower[n_done] = diff/prop_count - conf_interval[0]/prop_count
    list_random_motif_confidence_upper[n_done] = conf_interval[1]/prop_count - diff/prop_count
    print(diff/prop_count,conf_interval[0]/prop_count,conf_interval[1]/prop_count, sep = " , " )
    
    print("\n\n\n")
    
    n_done+=1
    
    
imsh = axs[1].plot(X_random_motif_capped,pval_list_wrong, marker = markerVec[1],   color = colorVec[1], label='motifs (wrong motif)', linestyle="None")
imsh = axs[1].plot(X_random_motif_capped,pval_list_random_motif, marker = markerVec[2],   color = colorVec[2], label='motifs (random motif)', linestyle="None")
axs[1].set_yscale('log')
axs[1].set_ylim([1*10**(-6), 0.05])
axs[1].legend()

imsh = axs[2].errorbar(X_random_motif_capped, list_wrong_difference, yerr=[list_wrong_confidence_lower, list_wrong_confidence_upper], marker = markerVec[1], color = colorVec[1] ,label='motifs (wrong motif)' ,capthick=2, linestyle="None")
imsh = axs[2].errorbar(X_random_motif_capped, list_random_motif_difference, yerr=[list_random_motif_confidence_lower, list_random_motif_confidence_upper], marker = markerVec[2], color = colorVec[2] ,label='motifs (random motif)' ,capthick=2, linestyle="None")
axs[2].legend()


plt.savefig(path_fig, transparent=True)

