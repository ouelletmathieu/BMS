import random
import networkx as nx
import os
from boolean.boolean_util import generateRandomValidNetwork, getGraphState, getMaxLength, constructGraph



def main():
    """     
    Create a txt file at location 'output_file' sampling a 'node' node networks containing a sample of 'nb_sample' element containing:
    [0]:  sample_nb
    [1]:  node (number of node)
    [2]:  maximum cycle length
    [3]:  maximum cycle length (it is there two times (it is a bug))
    [4]:  average cycle size 
    [5]:  max_delta (maximum length of cycle lost if one edge is changed)
    [6]:  avg_delta (average length of cycle lost is one edge is changed)
            
    """
    nb_sample = 50000
    node = 6
    #remove network with negligeable cycle since they are not of interest 
    min_cycle_to_consider = 7
    output_file = "./data_example/cycle_sensibility/edge_removed_nodes_"+str(node)+".txt"


    file = open(output_file, 'a') 
    header = "nb_sample,node,max_cycle,avgcycle,max_delta,avg_delta\n"
    if os.stat(output_file).st_size < 10:
        file.writelines(header)


    for nb_sample in range(nb_sample):

        r = random.random()

        if r<0.2:
            r=0.2

        mat = generateRandomValidNetwork(node, r)


        graph = getGraphState(constructGraph(mat))
        cycle_list = list(nx.simple_cycles(graph[0]))
        max_cycle, avgcycle = getMaxLength(cycle_list)


        if max_cycle>min_cycle_to_consider:

            avg_delta = 0
            max_delta = 0
            count_delta = 0

            for i in range(node):
                for j in range(node):

                    if(mat[i,j]!=0):

                        newmat = mat.copy()
                        newmat[i,j]=-1*newmat[i,j]
                        newgraph = getGraphState(constructGraph(newmat))
                        newcycle_list = list(nx.simple_cycles(newgraph[0]))
                        newmax_cycle, newavgcycle = getMaxLength(newcycle_list)


                        delta=max_cycle-newmax_cycle
                        avg_delta+=delta
                        count_delta+=1

                        if delta > max_delta:
                            max_delta = delta


                        newmat = mat.copy()
                        newmat[i,j]=0
                        newgraph = getGraphState(constructGraph(newmat))
                        newcycle_list = list(nx.simple_cycles(newgraph[0]))
                        newmax_cycle, newavgcycle = getMaxLength(newcycle_list)


                        delta=max_cycle-newmax_cycle
                        avg_delta+=delta
                        count_delta+=1

                        if delta > max_delta:
                            max_delta = delta

            avg_delta = avg_delta/count_delta

            list_out = [nb_sample,node,max_cycle,max_cycle,avgcycle,max_delta,avg_delta] 
                    
            str_output = ""
            for elem in list_out:
                str_output += str(elem) + ","
            str_output+= "\n"

            file.writelines(str_output)



if __name__ == "__main__":
    main()
