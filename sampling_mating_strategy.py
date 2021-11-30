import random
import networkx as nx
import os
from boolean.genetic_util import mate_cycle, mateMat
from boolean.boolean_util import generateRandomValidNetwork, getGraphState, getMaxLength, constructGraph, getDensity



def main():
    """     
    Create a txt file at location 'output_file' sampling a 'node' node networks containing a sample of 'nb_sample' element containing:
    [0]:  max cycle length of parent 1
    [1]:  max cycle length of parent 2
    [2]:  max cycle length of off spring
            
    """

    nb_sample = 3000
    node = 7

    output_file = "./data_example/mating_strategy/"
    type_mating = ["random_mating_nodes_", "cycle_mating_nodes_", "swap_mating_nodes_"]
    end_path = str(node)+".txt"



    #sample random mating 
    def func_random(mat1,mat2):
        r1 = getDensity(mat1)
        r2 = getDensity(mat2)
        node_nb = mat1.shape[0]
        return generateRandomValidNetwork(node_nb, (r1+r2)/2)

    path = output_file + type_mating[0] + end_path
    file = open(path, 'a') 
    header = "cycle1,cycle2,cycle3,\n"
    if os.stat(path).st_size < 10:
        file.writelines(header)

    for i in range(nb_sample):
        list_out = sampling_strat(func_random, node)         
        str_output = ""
        for elem in list_out:
            str_output += str(elem) + ","
        str_output+= "\n"
        file.writelines(str_output)

    file.close() 

    #sample cycle mating
    path = output_file + type_mating[1] + end_path
    file = open(path, 'a') 
    header = "cycle1,cycle2,cycle3,\n"
    if os.stat(path).st_size < 10:
        file.writelines(header)

    for i in range(nb_sample):
        list_out = sampling_strat(mate_cycle, node)         
        str_output = ""
        for elem in list_out:
            str_output += str(elem) + ","
        str_output+= "\n"
        file.writelines(str_output)
    
    file.close() 

    #sample swap mating
    path = output_file + type_mating[2] + end_path
    file = open(path, 'a') 
    header = "cycle1,cycle2,cycle3,\n"
    if os.stat(path).st_size < 10:
        file.writelines(header)

    for i in range(nb_sample):
        list_out = sampling_strat(mateMat, node)         
        str_output = ""
        for elem in list_out:
            str_output += str(elem) + ","
        str_output+= "\n"
        file.writelines(str_output)
    
    file.close() 

    """
    for i in range(50000):
    test_new_mating(mat1, mat2)
    #mat3 = generateRandomValidNetwork(7, (r1+r2)/2)
    #mat3, mat4 = mateMat(mat1,mat2)
    """


def sampling_strat(func_, node ):
    """Use the mating strategy func_ to mate two random matrix of random density 

    Returns:
        [int,int,int]: cycle of the two initial matrix and the cycle length of the mated one
    """

    r1 = random.random()
    if r1<0.2:
        r1=0.2
    mat1 = generateRandomValidNetwork(node, r1)
    
    r2 = random.random()
    if r2<0.2:
        r2=0.2
    mat2 = generateRandomValidNetwork(node, r2)

    graph1 = getGraphState(constructGraph(mat1))
    cycle1, avgcycle1 = getMaxLength(list(nx.simple_cycles(graph1[0])))

    graph2 = getGraphState(constructGraph(mat2))
    cycle2, avgcycle2 = getMaxLength(list(nx.simple_cycles(graph2[0])))

    mat3 = func_(mat1, mat2)

    if type(mat3) == tuple:
        mat3=mat3[0]

    graph3 = getGraphState(constructGraph(mat3))
    cycle3, avgcycle3 = getMaxLength(list(nx.simple_cycles(graph3[0])))

    return cycle1,cycle2,cycle3



if __name__ == "__main__":
    main()

