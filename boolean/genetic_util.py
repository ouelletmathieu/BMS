import random
import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from boolean.boolean_util import isValidRule, listToArray, getMaxCycleLength, getDensity, arrayToList, distanceBetweenList, generateRandomValidNetwork, constructGraph, get_1d_property
from boolean.motif_util import get_z_val_2_3
from boolean.adv_descriptor_util import get_tau_descriptor
import glob
import os

def mateMat(mat1,mat2):
    """Create two offsprings that are the complete mix of the two matrices mat1 and mat2. 
    Each element as 50% chance to go to either offsping. Both element is used. 
    If not good it return the two input matrix

    Returns:
        [mat1, mat2]:  the two offsprings, if one is not valid the two input matrix are output
    """
    tryNb = 0
    good = False
    n = mat1.shape[0]
    offspring1 = 0
    offspring2 = 0
    
    while not good and tryNb<1000:
        offspring1 = np.zeros((n,n))
        offspring2 = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                
                if(random.random()<0.5):
                    offspring1[i][j] = mat1[i][j]
                    offspring2[i][j] = mat2[i][j]
                else:
                    offspring1[i][j] = mat2[i][j]
                    offspring2[i][j] = mat1[i][j]
        
        if isValidRule(offspring1) and isValidRule(offspring2):
            good=True
        else:
            tryNb+=1
                
    if good:
        return offspring1, offspring2
    else:
        print("did not mate")
        return mat1, mat2

    
    
def mutateMat(p, mat1):
    """mutate a matrix mat switching the elements with a probability p. The matrix is destroyed

    Args:
        p (double [0,1]):  probability to switch each element 

    Returns:
        [matrix] : return the same matrix but modified
    """
    n = mat1.shape[0]
    newMat = 0
    tryNb = 0
    good = False
    
    while not good and tryNb<1000:
        newMat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if random.random()<p:
                    r = random.random()
                    if r >= 0.666:
                        newMat[i][j]=1
                    if r >= 0.333 and r < 0.666: 
                        newMat[i][j]=0
                    if r < 0.333:
                        newMat[i][j]=-1
                else:
                    newMat[i][j]=mat1[i][j]
    
        if isValidRule(newMat):
            good=True
        else:
            tryNb+=1
    
    if isValidRule(newMat):
        return newMat
    else:
        return mat1    
    
def selTournamentMathieu(individuals, k, tournsize, maxTry, toolbox):
    """set up a tournament for the 
    TODO finish to write this doc 
    Args:
        individuals ([type]): 
        k ([type]): 
        tournsize ([type]): 
        maxTry ([type]): 
        toolbox ([type]): 

    Returns:
        [type]: [description]
    """
    selected = []
    tryNb = 0
    
    while len(selected)<k and tryNb<maxTry:
        tryNb+=1
        testGroup = random.choices(individuals, k=tournsize)
        
        best = testGroup[0]
        bestfit = testGroup[0].fitness.values[0]
        
        for ind in testGroup:
            if ind.fitness.values[0] >= bestfit:
                bestfit = ind.fitness.values[0]
                best = ind
        
        same = False
        
        for ind in selected:
            if distanceBetweenList(best, ind)==0:
                same=True
                break
        
        if not same:
            selected.append(best)
            
    print("selected in ", tryNb)
    
    if len(selected)<k:
        print("not found enough we have a selected pop =, ", str(len(selected)) )
        print("number of  try =, ", str(tryNb) )    
        
        popNew = toolbox.population(k-selected)

        invalid_ind = [ind for ind in popNew if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        for ind in popNew: 
            selected.append(ind)
    
    return selected

def evalFitness(individual):
    """Evaluate the fitness for an individual. The fitness is 2 dimensional with 
    [0]: Max cycke length 
    [1]: Density
    Some randomness is added to the value so they cannot ever overlap
    """
    anti_overlap1 = (random.random()-0.5)*0.2
    anti_overlap2 = (random.random()-0.5)*0.01
    n = int(abs(math.sqrt(len(individual))))
    mat = listToArray(individual,n)
    return getMaxCycleLength(mat)+anti_overlap1, getDensity(mat)+anti_overlap2

def evalFitness1D(individual):
    """Evaluate the fitness for an individual. The fitness is 2 dimensional with 
    [0]: Max cycke length 
    """
    n = int(abs(math.sqrt(len(individual))))
    mat = listToArray(individual,n)

    return getMaxCycleLength(mat),

def mateList(ind1, ind2):
    """Mate the two individual 1 and 2 and output the offspring as individual 1 and 2 

    Returns:
        [ind1, ind2]: The two individual ind1 and ind2 but transformed 
    """
    n = int(abs(math.sqrt(len(ind1))))
    mat1 = listToArray(ind1,n)
    mat2 = listToArray(ind2,n)
    res = mateMat(mat1,mat2)
    res1 = arrayToList(res[0])
    res2 = arrayToList(res[1])
    
    for i in range(n):
        ind1[i] = res1[i]
        ind2[i] = res2[i]
        
    return ind1, ind2

def mutateList(individual, prob=1):
    """Mutate the individual with probaibility prob (as a parameter for the mutation)
    Returns:
        [ind1,]: The  individual ind1 but transformed 
    """
    n = int(abs(math.sqrt(len(individual))))
    mat1 = listToArray(individual,n)
    res = mutateMat(prob, mat1)
    res = arrayToList(res)
    
    for i in range(n):
        individual[i] = res[i]
    return individual,

def mating_1_cycle(mat1, mat2):
    """Internal method that mate two matrix mat1 and mat2 keeping the physical cycle 
    (see the paper)

    TODO add explanation in the function
    Returns:
        [mat]: new matrix
    """
    mat = [mat1,mat2]
    node = len(mat1)
    mat3 = np.zeros((node,node))
    
    nbElem1 = 0 
    nbElem2 = 0
    
    for i in range(node):
        for j in range(node):
            if mat1[i][j]!=0:
                nbElem1+=1
            if mat2[i][j]!=0:
                nbElem1+=1            
    
    G1 = constructGraph(mat1)
    G2 = constructGraph(mat2)


    cycle_list_1 = list(nx.simple_cycles(G1))
    cycle_list_2 = list(nx.simple_cycles(G2))                
    total_list = list()
    
    for cyc in cycle_list_1:
        total_list.append((0,cyc))
    for cyc in cycle_list_2:
        total_list.append((1,cyc))
        
    random.shuffle(total_list)

    targetElem = int((nbElem1+nbElem2)/2)
    liveElem = 0
    
    while liveElem<targetElem:
        
        if len(total_list)==0:
            break
        
        indexCycle, newCycle = total_list.pop()
        
        index = [] 
        typeEdge = []

        for i in range(len(newCycle)-1):
            index.append([newCycle[i],newCycle[i+1]])
            typeEdge.append(mat[indexCycle][newCycle[i],newCycle[i+1]])
        
        index.append([newCycle[-1],newCycle[0]])
        typeEdge.append(mat[indexCycle][newCycle[-1],newCycle[0]])
        
        doAble = True
        
        for n, ij in enumerate(index):
            
            if mat3[ij[0],ij[1]]!=typeEdge[n] and mat3[ij[0],ij[1]]!=0:
                doAble = False
                break
        
        if doAble :
            for n, ij in enumerate(index):
                if mat3[ij[0],ij[1]]==0:
                    liveElem+=1
                mat3[ij[0],ij[1]] = typeEdge[n]
                
    mat_selector = [0,1]
    
    if liveElem>=targetElem:
        return mat3
    else:
        
        list_index = []
        
        for i in range(node):
            for j in range(node):
                
                if mat3[i,j]==0 and (mat1[i,j]!=0 or mat2[i,j]!=0):
                    list_index.append((i,j))
        
        random.shuffle(list_index)
        random.shuffle(mat_selector)
        
        while targetElem>liveElem:
            ij = list_index.pop()
            
            if  mat[mat_selector[0]][ij[0],ij[1]]!=0 :
                mat3[ij[0],ij[1]]==mat[mat_selector[0]][ij[0],ij[1]]
            else:
                mat3[ij[0],ij[1]]==mat[mat_selector[1]][ij[0],ij[1]]

            liveElem+=1
                
        return mat3       
        
def mate_cycle(mat1, mat2):
    """Use the method mating_1_cycle many time to find an adequate mating. 

    Returns:
        [ind1, ind2]: The two individual ind1 and ind2 but transformed 
    """
    valid1 = False
    valid2 = False
    try1 = 0
    try2 = 0
    newmat1 = mat1
    newmat2 = mat2
    
    while not valid1 and try1<10:
        
        newmat1 = mating_1_cycle(mat1, mat2)
        
        if isValidRule(newmat1):
            valid1 = True
        try1+=1
    
    while not valid2 and try2<10:
        
        newmat2 = mating_1_cycle(mat1, mat2)
        
        if isValidRule(newmat2):
            valid2 = True
        try2+=1
    
    toReturn = [mat1, mat2]
    
    if valid1:
        toReturn[0]=newmat1
    else :
        r2 = (random.random()*0.8)+0.2
        toReturn[0] = generateRandomValidNetwork(len(mat1), r2)
    if valid2:
        toReturn[1]=newmat2
    else:
        r2 = (random.random()*0.8)+0.2
        toReturn[1] = generateRandomValidNetwork(len(mat1), r2)
    return toReturn[0], toReturn[1]
       
def mateList2(ind1, ind2):
    """Mate the two individual 1 and 2 and output the offspring as individual 1 and 2 using the 
    two type of mating 
    1. Cycle mating (mate_cycle) 
    2. Shuffling element (mateList)

    Returns:
        [ind1, ind2]: The two individual ind1 and ind2 but transformed 
    """   
    n = int(abs(math.sqrt(len(ind1))))
    mat1 = listToArray(ind1,n)
    mat2 = listToArray(ind2,n)
    res = 0
    if random.random()<0.5:
        res = mateMat(mat1, mat2)
    else:
        res = mate_cycle(mat1, mat2)
        
    res1 = arrayToList(res[0])
    res2 = arrayToList(res[1])
    
    for i in range(n):
        ind1[i] = res1[i]
        ind2[i] = res2[i]
        
    return ind1, ind2    

def randomPoint():
    """random -1,0,1 with equal property for the simulation (initial individuals)

    Returns:
        either -1,0,1
    """
    r = random.random()
    
    if r<0.333:
        return -1
    elif r<0.66666:
        return 0
    else:
        return 1

def evaluateMotif(gen, pop, list2motif, motif2Info, list3motif, motif3Info, nwayMotif, rootNode, list2mat, list3mat):
    """return list of 2-motif and 3-motifs z-values. See returns for the exact form of the output. 

    Args:
        gen (int): generation number
        pop ([vec,...]): population of networks
        list2motif ([networkX , ...]): list of graph (networkX) for each 2-motif
        motif2Info : Info for each 2-motifs (see method get2motif_graph())
        list3motif ([networkX , ...]): list of graph (networkX) for each 2-motif
        fmotif3Info :Info for each 3-motifs (see method get3motif_graph())
        nwayMotif ([int,...]): number of way to generate the same motif
        rootNode ([type]): Root of the decision tree for fast motif finding (see DecisionTree)
        list2mat ([np.array,...,]): list of 2-motif matrix 
        list3mat ([np.array,...,]): list of 3-motif matrix

    Returns:
        [[gen]+mean2vec , [gen]+std2vec],  [[gen]+mean3vec , [gen]+std3vec]
    """
    
    listzvalue = list()
    
    for index in range(len(pop)):
        zval = get_z_val_2_3(pop[index],list2motif,motif2Info,list3motif,motif3Info, nwayMotif, rootNode, list2mat, list3mat)
        listzvalue.append(zval)
        
    listOf2motif, listOf3motif = [], []
    nb = len(listzvalue)
    
    for i in range(len(listzvalue)):
        listOf2motif.append(listzvalue[i][0])
        listOf3motif.append(listzvalue[i][1])

    mean2vec, std2vec, mean3vec, std3vec  = [0]*len(listOf2motif[0]), [0]*len(listOf2motif[0]), [0]*len(listOf3motif[0]), [0]*len(listOf3motif[0])
    maxdyn = 0

    for n in range(nb):    
        if listOf2motif[n][-1] > maxdyn:
            maxdyn=listOf2motif[n][-1]; 
            
        for i in range(len(mean2vec)):
            mean2vec[i] += listOf2motif[n][i]/nb
        for i in range(len(mean3vec)):
            mean3vec[i] += listOf3motif[n][i]/nb

    for n in range(nb):
        for i in range(len(mean2vec)):
            std2vec[i] += ((listOf2motif[n][i] - mean2vec[i])**2)/nb
        for i in range(len(mean3vec)):
            std3vec[i] += ((listOf3motif[n][i] - mean3vec[i])**2)/nb

    for i in range(len(mean2vec)):
            std2vec[i] = math.pow(std2vec[i], 0.5)
    for i in range(len(mean2vec)):
            std3vec[i] = math.pow(std3vec[i], 0.5)
    
    
    return [[gen]+mean2vec , [gen]+std2vec],  [[gen]+mean3vec , [gen]+std3vec]

def get_tau_avg_std(pop):
    """formatting of tau descriptor properties for the optimization population 

    Args:
        vec : network in vector form (ie. list of element in the matrix in order starting with top left)

    Returns:
        average of : [node_pos_dumb,node_neg_dumb,node_all_dumb,node_pos_good, node_neg_good,node_all_good]
        std of : [node_pos_dumb,node_neg_dumb,node_all_dumb,node_pos_good, node_neg_good,node_all_good]
    """

    avg,std = [0]*6,[0]*6 
    list_tau_vec = list()
    
    for vec in pop:
        tauvec = get_tau_descriptor(vec)
        list_tau_vec.append(tauvec)
        
        for i, value in enumerate(tauvec):
            avg[i]+=value/len(pop)
    
    for tauvec in list_tau_vec:
        for i, value in enumerate(tauvec):
            std[i]+= ((value-avg[i])**2)/len(pop)
        
    for i in range(6):
        std[i] = math.sqrt(std[i])
        
    return avg, std  

def get_1d_property_avg_std(pop):
    """formatting of 1 dimensional properties for use in the optimization process (with a population)

    Args:
        pop : population from the optimization 

    Returns:
        average of : [ mxc_inhib, avgc_inhib, mxc_exct, avgc_exct, mxc_all, avgc_all, totalInhibition, totalExcitation, totalAutoInhibition, totalAutoExcitation]
        standard deviation of : [ mxc_inhib, avgc_inhib, mxc_exct, avgc_exct, mxc_all, avgc_all, totalInhibition, totalExcitation, totalAutoInhibition, totalAutoExcitation]
    """
    avg = [0]*10
    std = [0]*10
    list_1d_vec = list()
    
    for vec in pop:
        vec_1d = get_1d_property(vec)
        list_1d_vec.append(vec_1d)
        for i, value in enumerate(vec_1d):
            avg[i]+=value/len(pop)
    
    for vec_1d in list_1d_vec:
        for i, value in enumerate(vec_1d):
            std[i]+= ((value-avg[i])**2)/len(pop)
        
    for i in range(6):
        std[i] = math.sqrt(std[i])
        
    return avg, std

def get_pareto(list_fitness_associated, weights, front_toKeep = 1, min_density=0):
    """Create the pareto front for a 2d problem doesn't work in higher dimension. weights can be adjusted to allow minimization or maximization. 

    Args:
        list_fitness_associated ([[double, double, double], [d,d,d], ... ]): each element is [density, fitness1, fitness2] 
        weights ([double,double]): Indicate if the variable need to be minimized or maximised 
        front_toKeep (int, optional): number of layer of pareto front to keep. 1 keep all non dominated solution. Defaults to 1.
        min_density (int, optional): minimal density to consider. Defaults to 0.

    Returns:
        [[double, double, double], [d,d,d], ... ]: A subset of list_fitness_associated that are in the first front_toKeep pareto front
    """
    pareto = list()
    
    for i, ind1 in enumerate(list_fitness_associated):
        
        dominated_nb = 0
        if ind1[0]>min_density:
            for ind2 in list_fitness_associated:

                if ind1 != ind2:
                    if weights[0]<0:
                        if ind2[0] <= ind1[0]: 
                            if weights[1] < 0:
                                if ind2[1] <= ind1[1]:
                                    dominated_nb+=1

                            if weights[1] > 0:
                                if ind2[1] >= ind1[1]:
                                    dominated_nb+=1

                    if weights[0]>0:
                        if ind2[0] >= ind1[0]: 
                            if weights[1]<0:
                                if ind2[1] <= ind1[1]:
                                    dominated_nb+=1

                            if weights[1] > 0:
                                if ind2[1] >= ind1[1]:
                                    dominated_nb+=1

            if dominated_nb<front_toKeep:        
                pareto.append(list(ind1))

    return pareto


def get_pareto_from_pop_file(path, weights):
    """get  the pareto front for a 2d problem doesn't work in higher dimension. weights can be adjusted to allow minimization or maximization. 

    Args:
        path (file): path of the file
        weights ([double, double]): Indicate if the variable need to be minimized or maximised 

    Returns:
        [[double, double, double], [d,d,d], ... ]: A subset of list_fitness_associated that are in the pareto front. Each element is [density, fitness1, fitness2] 
    """    
    df = pd.read_csv(path)
    list_fitness = list()
    
    max_cycle = df["max_cycle"]
    density = df["density"]
    
    for i in range(len(max_cycle)):
        mx_cyc = round(max_cycle[i])
        list_fitness.append([max_cycle[i],density[i]])
   
    return get_pareto(list_fitness, weights)
    