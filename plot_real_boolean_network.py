import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from boolean.plot_util import colorVec, markerVec

def main_compare_cat():
    
    path_random = "./data_example/real_boolean/random_network.txt"
    path_real = "./data_example/real_boolean/real_network_w_cat.csv"
    fig_path = "./fig/real_boolean.pdf"

    df_random = pd.read_csv(path_random, encoding = 'utf-8')
    df_real = pd.read_csv(path_real, encoding = 'utf-8')
    df_random['type'] = 'random'
    df_real = pd.concat([ df_random, df_real])
    ax = sns.violinplot(x="type", y="ratio_sym", data=df_real, color=colorVec[0], inner="stick")      
    ax.plot([-0.5,7.5], [0.5,0.5], color = colorVec[2])
    ax.plot([-0.5,7.5], [0,0], color="white")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    plt.savefig(fig_path, transparent=True)
    plt.clf()

    list_ratio = df_real["ratio_sym"].tolist()
    list_type = df_real["type"].tolist()
    dic_type = {}
    
    for i in range(len(list_ratio)):   
        if list_type[i] in dic_type:
            dic_type[list_type[i]].append(list_ratio[i])
        else:
            dic_type[list_type[i]] = [list_ratio[i]]
    
    list_type = ['cancer', 'development', 'cell_cycle', 'cell_fate', 'signaling', 'differentiation', 'T_cell_activation']
    print("\n\n\n\n\n")
    print("###################################")
    print("#            P-Value              #")
    print("###################################")
    print("name, stat, pval, mean")
    for t in list_type:        
        stat, pval = stats.ttest_ind( dic_type[t] , dic_type['random'], equal_var=False)
        print(t, stat, pval, np.mean(np.array(dic_type[t])))

def main_all_same(): 
    
    path_random = "/Users/mathieuouellet/Dropbox/Boolean_network/DATA/real_network/random_with_same_node_and_indegree.csv"
    path_real = "/Users/mathieuouellet/Dropbox/Boolean_network/DATA/real_network/real_network_sym.csv"
    fig_path = "./fig/real_boolean_all_cat.pdf"

    df_real = pd.read_csv(path_real, encoding = 'utf-8')
    df_random = pd.read_csv(path_random, encoding = 'utf-8')
    df_real['all'], df_real['type']  = '1', 'biological_network'
    df_random['all'], df_random['type'] ='1', 'random'
    df= pd.concat([ df_random, df_real])
    #df["all"] = ""
    #df = pd.concat([df_real.assign(orig='biological network'), df_random.assign(orig='random network')], axis=0)

    ax = sns.violinplot(x="all", y="ratio_sym", hue="type", data=df, split =True, inner="quartile", palette = colorVec)    
    #ax.plot([-0.3,0.4], [0.5,0.5], color="black")
    #ax.plot([-0.2,0.2], [0,0], color="white")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    
    plt.savefig(fig_path, transparent=True)
    plt.clf()
    stat, pval = stats.ttest_ind( df_real["ratio_sym"].tolist() , df_random["ratio_sym"].tolist(), equal_var=False)
    print("\n\n\n\n\n")
    print("###################################")
    print("#      P-Value  (all same)        #")
    print("###################################")
    print("stat, pval")
    print( stat, pval)
        
   
if __name__ == "__main__":
    main_compare_cat()
    main_all_same()