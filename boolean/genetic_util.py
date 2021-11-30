import random
import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
from boolean_util import isValidRule, listToArray, getMaxCycleLength, getDensity, arrayToList, distanceBetweenList, generateRandomValidNetwork, constructGraph



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


def printPareto(pop):
    """Fast plot of the population pop
    """
    #TODO only work in notebook
    p = np.array([ind.fitness.values for ind in pop])
    plt.scatter(p[:, 1], p[:, 0], marker="o", s=24, label="Final Population")
    plt.show()



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
    