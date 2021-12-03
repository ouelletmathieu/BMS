import seaborn as sns
"""
def test():
    
    path_random = "/Users/mathieuouellet/Dropbox/Boolean_network/DATA/real_network/random_with_same_node_and_indegree.csv"
    path_real = "/Users/mathieuouellet/Dropbox/Boolean_network/DATA/real_network/real_network_sym_all_descr.csv"
    
    
    
    df_random = pd.read_csv(path_random, encoding = 'utf-8')

    
    
    
    df_real = pd.read_csv(path_real, encoding = 'utf-8')
    df_real = df_real[df_real.type != "senescence_life"]
    
    df_real = df_real[df_real.type != "senescence_life"]
    
    df_real = pd.concat([ df_random, df_real], axis=0)

    
    
    ax = sns.violinplot(x="type", y="ratio_sym", data=df_real, color=colorVec[0], inner="stick")
        
    ax.plot([-0.5,7.5], [0.5,0.5], color = colorVec[2])
    ax.plot([-0.5,7.5], [0,0], color="white")
    
    
    #list_mat = df_pop["mat"].tolist()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    
    plt.savefig("/Users/mathieuouellet/Dropbox/Boolean_network/FIGURES/OUTPUT/real_network_symmetries.pdf", transparent=True)
    
    list_ratio = df_real["ratio_sym"].tolist()
    list_type = df_real["type"].tolist()
    
    dic_type = {}
    
    for i in range(len(list_ratio)):
        
        if list_type[i] in dic_type:
            dic_type[list_type[i]].append(list_ratio[i])
        else:
            dic_type[list_type[i]] = [list_ratio[i]]
    
    list_type = ['cancer', 'development', 'cell_cycle', 'cell_fate', 'signaling', 'differentiation', 'T_cell_activation']
    
    list_pvalue = [-1]*len(list_type)
    
    
    for k, t in enumerate(list_type):        
        stat, pval = stats.ttest_ind( dic_type[t] , dic_type['random'], equal_var=False)
        print(t, stat, pval, np.mean(np.array(dic_type[t])))
        
        

test()
"""