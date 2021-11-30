import random
import networkx as nx
import os
from boolean.boolean_util import generateRandomValidNetwork, getGraphState, getMaxLength, getStat, constructGraph

def main():
    """     
    Create a txt file at location 'output_file' sampling a 'node' node networks containing a sample of 'nb_sample' element containing:
    [0]:  sample_nb
    [1]:  density of network
    [2]:  maximum cycle length
    [3]:  average cycle length over all cycles
    [4]:  number of attractor
    [5]:  number of fixed point
    [6]:  mxc_inhib
    [7]:  avgc_inhib
    [8]:  mxc_exct
    [9]:  avgc_exct
    [10]: mxc_all
    [11]: avgc_all
    [12]: totalInhibition,
    [13]: totalExcitation,
    [14]: totalAutoInhibition,
    [15]: totalAutoExcitation] 

            
    """

    nb_sample = 5000
    node = 8
    output_file = "./data_example/1d_properties_nodes_"+str(node)+".txt"


    file = open(output_file, 'a') 
    header = "sample_nb,density,max_cycle,avgcycle,nb_attractor,nb_fixed_point,mxc_inhib,avgc_inhib,mxc_exct,avgc_exct,mxc_all,avgc_all,totalInhibition,totalExcitation,totalAutoInhibition,totalAutoExcitation,\n"
    if os.stat(output_file).st_size < 10:
        file.writelines(header)

    
    
    for sample_nb in range(nb_sample):
        
        r = random.random()
        
        if r<0.2:
            r=0.2
            
        
        mat = generateRandomValidNetwork(node, r)
        
        graph = getGraphState(constructGraph(mat))
        cycle_list = list(nx.simple_cycles(graph[0]))
        max_cycle, avgcycle = getMaxLength(cycle_list)
        nb_attractor = len(cycle_list)
        nb_fixed_point = 0
        
        for cycle in cycle_list:
            if len(cycle)==1:
                nb_fixed_point+=1
                
        
        
        G_inhibition = nx.DiGraph()
        G_excitation = nx.DiGraph()
        G_all = nx.DiGraph()

        for i in range(node):
            G_excitation.add_node(i)
            G_all.add_node(i)
            G_inhibition.add_node(i)
        
        density = 0
        
        for i in range(node):
            for j in range(node):
                if mat[i,j]!=0:
                    density+=1
                    
        density = density/(node**2) 
        
        for i in range(node):
            for j in range(node):
                if mat[i,j]==1:
                    G_excitation.add_edge(i, j)
                    G_all.add_edge(i, j)
                elif mat[i,j]==-1:
                    G_inhibition.add_edge(i, j)
                    G_all.add_edge(i, j)
                    
        

        lstCycl_inhib = list(nx.simple_cycles(G_inhibition))
        mxc_inhib, avgc_inhib = getMaxLength(lstCycl_inhib)  

        lstCycl_exct = list(nx.simple_cycles(G_excitation))
        mxc_exct, avgc_exct = getMaxLength(lstCycl_exct) 

        lstCycl_all = list(nx.simple_cycles(G_all))
        mxc_all, avgc_all = getMaxLength(lstCycl_all) 

        
        matstat = getStat(mat)
        totalInhibition = matstat[0,0]
        totalExcitation = matstat[0,1]
        totalAutoInhibition = matstat[0,2]
        totalAutoExcitation = matstat[0,3]


        
        
        list_out = [sample_nb,'%.3f'%(density),max_cycle,'%.3f'%(avgcycle),nb_attractor,nb_fixed_point,
            mxc_inhib,'%.3f'%(avgc_inhib),mxc_exct,'%.3f'%(avgc_exct),mxc_all,'%.3f'%(avgc_all),
            totalInhibition,totalExcitation,totalAutoInhibition,totalAutoExcitation] 
        
        str_output = ""
        for elem in list_out:
            str_output += str(elem) + ","
        str_output+= "\n"

        file.writelines(str_output)

    file.close() 


if __name__ == "__main__":
    main()