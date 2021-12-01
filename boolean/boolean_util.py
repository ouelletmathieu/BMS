import networkx as nx
import numpy as np
import sys
import random
import networkx as nx
import numpy as np
import sys
import random
import math

"""
list of matrix for isomorphism check, only done for 3 and 4 nodes.
"""
global T2_1, T2_2, T2, T2m
T2_1 = np.matrix([[1,0], [0,1]])
T2_2 = np.matrix([[0,1], [1,0]])

T2 = [T2_1,T2_2]
T2m = [T2_1,T2_2]

global T3_1, T3_2, T3_3, T3_4, T3_5, T3_6, T3, T3m 
T3_1 = np.matrix([[1,0,0], [0,1,0],[0,0,1]])
T3_2 = np.matrix([[1,0,0], [0,0,1],[0,1,0]])
T3_3 = np.matrix([[0,1,0], [1,0,0],[0,0,1]])
T3_4 = np.matrix([[0,1,0], [0,0,1],[1,0,0]])
T3_5 = np.matrix([[0,0,1], [1,0,0],[0,1,0]])
T3_6 = np.matrix([[0,0,1], [0,1,0],[1,0,0]])

T3 = [T3_1,T3_2,T3_3,T3_4,T3_5,T3_6]
T3m = [T3_1,T3_2,T3_3,T3_5,T3_4,T3_6 ]


global T4_1, T4_2, T4_3, T4_4, T4_5, T4_6 
global T4_7, T4_8, T4_9, T4_10, T4_11, T4_12
global T4_13, T4_14, T4_15, T4_16, T4_17, T4_18
global T4_19, T4_20, T4_21, T4_22, T4_23, T4_24
global T4, T4m 

T4_1 = np.matrix([[1,0,0,0], [0,1,0,0],[0,0,1,0],[0,0,0,1]])
T4_2 = np.matrix([[1,0,0,0], [0,1,0,0],[0,0,0,1],[0,0,1,0]])
T4_3 = np.matrix([[1,0,0,0], [0,0,1,0],[0,1,0,0],[0,0,0,1]])
T4_4 = np.matrix([[1,0,0,0], [0,0,1,0],[0,0,0,1],[0,1,0,0]])
T4_5 = np.matrix([[1,0,0,0], [0,0,0,1],[0,1,0,0],[0,0,1,0]])
T4_6 = np.matrix([[1,0,0,0], [0,0,0,1],[0,0,1,0],[0,1,0,0]])
T4_7 = np.matrix([[0,1,0,0], [1,0,0,0],[0,0,1,0],[0,0,0,1]])
T4_8 = np.matrix([[0,1,0,0], [1,0,0,0],[0,0,0,1],[0,0,1,0]])
T4_9 = np.matrix([[0,1,0,0], [0,0,1,0],[1,0,0,0],[0,0,0,1]])
T4_10 = np.matrix([[0,1,0,0], [0,0,1,0],[0,0,0,1],[1,0,0,0]])
T4_11 = np.matrix([[0,1,0,0], [0,0,0,1],[1,0,0,0],[0,0,1,0]])
T4_12 = np.matrix([[0,1,0,0], [0,0,0,1],[0,0,1,0],[1,0,0,0]])
T4_13 = np.matrix([[0,0,1,0], [1,0,0,0],[0,1,0,0],[0,0,0,1]])
T4_14 = np.matrix([[0,0,1,0], [1,0,0,0],[0,0,0,1],[0,1,0,0]])
T4_15 = np.matrix([[0,0,1,0], [0,1,0,0],[1,0,0,0],[0,0,0,1]])
T4_16 = np.matrix([[0,0,1,0], [0,1,0,0],[0,0,0,1],[1,0,0,0]])
T4_17 = np.matrix([[0,0,1,0], [0,0,0,1],[1,0,0,0],[0,1,0,0]])
T4_18 = np.matrix([[0,0,1,0], [0,0,0,1],[0,1,0,0],[1,0,0,0]])
T4_19 = np.matrix([[0,0,0,1], [1,0,0,0],[0,1,0,0],[0,0,1,0]])
T4_20 = np.matrix([[0,0,0,1], [1,0,0,0],[0,0,1,0],[0,1,0,0]])
T4_21 = np.matrix([[0,0,0,1], [0,1,0,0],[1,0,0,0],[0,0,1,0]])
T4_22 = np.matrix([[0,0,0,1], [0,1,0,0],[0,0,1,0],[1,0,0,0]])
T4_23 = np.matrix([[0,0,0,1], [0,0,1,0],[1,0,0,0],[0,1,0,0]])
T4_24 = np.matrix([[0,0,0,1], [0,0,1,0],[0,1,0,0],[1,0,0,0]])


T4 = [T4_1,T4_2,T4_3,T4_4,T4_5,T4_6,T4_7,T4_8,T4_9,T4_10,T4_11,T4_12,T4_13,
      T4_14,T4_15,T4_16,T4_17,T4_18,T4_19,T4_20,T4_21,T4_22,T4_23,T4_24]

T4m = [ np.transpose(mat) for mat in T4]



    
def constructGraph(adjacencyMatrix):
    """construct a networkx digraph given an adjacencyMatrix

    Args:
        adjacencyMatrix (np.array): nxn matrix with +-1 or 0 in each element, column = to, row = from  i.e. position [3,4] is from node 3 to node 4

    Returns:
        [networkx graph]: digraph where each node are connected when they are linked
    """

    G = nx.DiGraph()
    adj = adjacencyMatrix
    n=adj.shape[0]
    m=adj.shape[1]
    
    if m!=n :
        print("m is not equal to n")
        sys.exit(-1)
    
    for i in range(n):
        G.add_node(i, state=-1)
            
    for i in range(n):
        for j in range(m):
            if adj[i,j]==-1:
                G.add_edge(i, j, weight=-1)
                
            if adj[i,j]==1:
                G.add_edge(i, j, weight=1)   
    return G


def setRandomInitialState(graph):
    """
    create a random initial state for the graph graph (0 or 1 on each node)
    """
    for n in graph:
        graph.nodes[n]["state"]= random.randrange(2)

    

def setInitialState(graph, vector):
    """set the initial state of the graph. Vector is indexed by the index of the node.
    """
    for n in graph:
        graph.nodes[n]["state"]= vector[n]
        




def getNextState(graph):
    """This method generate the next state for a given graph for an initial state. 
    The method return a list of the state of each node (either 0 or 1) in the order the node are in the matrix.
    The graph is an object of nx.digraph which can contain a state and a weight for each node. 

    Args:
        graph (diGraph): networkx dirgraph with weight and state set

    Returns:
        [list]: return the next state 
    """    

    total = np.zeros( (graph.order()) )

    #go through all edges
    for u, v, weight in graph.edges(data='weight'):
        if weight==-1:  
            #inhibition
            if graph.nodes[u]["state"]==1:
                total[v]=total[v]-1
        if weight==1:  
            #print("excitation")
            if graph.nodes[u]["state"]==1:
                 total[v]=total[v]+1
       
    
    for n in range(graph.order()):
        #print(total[n])
        if total[n]<=0:
            graph.nodes[n]["state"]=0
        else:
            graph.nodes[n]["state"]=1
            
    return [graph.nodes[n]["state"] for n in range(graph.order())]



def getGraphState(graph):
    """Create a graph that represent the state space of the network
    also return a list of the possible initial state and the next state for each one

    TODO this method is slow the way it is designed

    Args:
        graph (diGraph): networkx dirgraph with weight 

    Returns:
        [diGraph, [initial state list], [final state list]]: the graph goes from state to state
    """    
    
    nbnode=graph.order()
    min = 0 
    max = 2**nbnode
    intialValue = []
    finalValue = []

    #we do that many times maybe put in calling method
    for n in range(max):
        newInit = [int(x) for x in reversed(bin(n)[2:])]
        newInit =  newInit + [0] * (nbnode - len(newInit)) 
        intialValue.append(newInit)
    
    
    for init in intialValue:
        setInitialState(graph, init )
        finalValue.append(getNextState(graph))

    StateG = nx.DiGraph()
    for i in range(n):
        StateG.add_node(i)
        
    for n in range(max) :
        initialRes = int("".join(str(x) for x in intialValue[n]), 2) 
        finalRes = int("".join(str(x) for x in finalValue[n]), 2) 
        StateG.add_edge(initialRes,finalRes)
   
    return StateG, intialValue, finalValue



def getAllRule(mat, nb):
    """This method generate all the possible rule for a given connection matrices. 
        Each connection can be either an inhibition or an excitation edge.  

    Args:
        mat ([type]): matrix of connection, if equal to 1 then a connection is considered
        nb ([type]):  number of rule to be generated. If -1 all rule are generated else it is a sampling

    Returns:
        [[np.array, ...]]: list of matrix (rules) 
    """

    listPos = []
    n = mat.shape[0]
    
    for i in range(n):
        for j in range(n):
            if mat[i,j]==1:
                listPos.append([i,j])

    nLink = len(listPos)
    max = 2**nLink
    possConn = [] 
    
    
    if nb==-1:
        for k in range(max):
            newInit = [int(x) for x in bin(k)[2:]]
            newInit = [0] * (nLink - len(newInit)) + newInit
            possConn.append(newInit)
    else:
        listNb = [random.randint(0,max) for i in range(nb)]
        for k in listNb:
            newInit = [int(x) for x in bin(k)[2:]]
            newInit = [0] * (nLink - len(newInit)) + newInit
            possConn.append(newInit)

    matList = []

    for k in range(len(possConn)):
        excitationList = possConn[k]
        livemat = np.zeros((n, n))

        for pos in range(len(listPos)):
            connection = listPos[pos]
            i = connection[0]
            j = connection[1]
            ex = excitationList[pos]
            if ex == 0:
                livemat[i,j]=-1
            else:
                livemat[i,j]=1
        matList.append(livemat)

    return matList


def getMaxLength(listOfList):
    """Return the maximum and the average of the lenth of the list inside a list 
    i.e. [ [1,1,1,1], [1,1], [1]] return max = 4 and mean = (4+2+1)/3

    Args:
        listOfList ([ list, ...]): A list that contain list of objects 

    Returns:
        [int, double]: maximum length and average length of list in the list 
    """    
    maximum =0 
    mean =0 
    if len(listOfList)!=0:
        for l in listOfList:
            mean+=len(l)
            if maximum<= len(l):
                maximum=len(l)

        return maximum, mean/len(listOfList)
    
    return 0,0



def getLegalMatrixList(size,nbKeep):
    """This method generate all the legal matrix representing connection for a number of node. 
    The condition to consider a matrix legal is that every node need to receive an input 
    but not every node need to have an output. The matrice is not symmetric since we keep 
    track of the directionality of the interaction. 

    Args:
        size (int): number of nodes
        nbKeep (int): number of solution to keep (if nbKeep=-1 they are all generated)

    Returns:
        [[np.array, ...]]: list of matrix (rules) 
    """    
    numberElement = size**2
    maxNb = 2**numberElement
    listMatrix =[]
    
    if nbKeep ==-1:
        listNb= range(0,maxNb)
    else:
        listNb = [random.randint(0,maxNb) for i in range(nbKeep)]
    
    for m in listNb:
        binList = [int(d) for d in str(bin(m))[2:]]
        binList = [0] * (numberElement - len(binList)) + binList
        mat = [[binList[y+x*size] for y in range(size)] for x in range(size)]

        good = True
        for row in mat:
            if np.sum(row)<1:
                good=False
        for column in np.matrix(mat).T:
            if np.sum(column)<1:
                good=False

        if good:
            listMatrix.append(np.matrix(mat))
            
    return listMatrix




def getInfoRule(mat):
    """return the number of  (non diag. exct.,  non diag. inhib.,  diag. exct.,  diag. inhib.)  connections 

    Args:
        mat ([type]): matrix rule

    Returns:
        [int, int, int, int]:  (non diag. exct.,  non diag. inhib.,  diag. exct.,  diag. inhib.)  connections 
    """    
    total=0
    nondiagPlus = 0
    nondiagMinus= 0
    diagPlus = 0
    diagMinus= 0
    
    n = mat.shape[0]
    
    for i in range(n):
        for j in range(n):
            
            if i == j:
                if mat[i,j]==1:
                    diagPlus+=1
                elif mat[i,j]==-1:
                    diagMinus+=1
            else:
                if mat[i,j]==1:
                    nondiagPlus+=1
                elif mat[i,j]==-1:
                    nondiagMinus+=1
            

    return np.array([nondiagPlus, nondiagMinus, diagPlus, diagMinus])



def getStat(rule):
    """return the number of  (inhibition,  excitation,  diag. -inhibition,  diag. -excitation)  connections 

    Args:
        mat ([type]): matrix rule

    Returns:
        [int, int, int, int]:  (inhibition,  excitation,  diag. -inhibition,  diag. -excitation)  connections 
    """    
    totalInhibition=0
    totalExcitation=0
    totalAutoInhibition=0
    totalAutoExcitation=0
    n = rule.shape[0]
    
    for i in range(n):
        for j in range(n):
            if rule[i,j]==-1:
                totalInhibition = totalInhibition+1
                if i==j:
                    totalAutoInhibition = totalAutoInhibition+1
            if rule[i,j]==1:
                totalExcitation = totalExcitation+1
                if i==j:
                    totalAutoExcitation = totalAutoExcitation+1

    return np.matrix([totalInhibition, totalExcitation, totalAutoInhibition, totalAutoExcitation])
        
    
"""

"""
def MatrixDistance(mat1, mat2):
    """compute the matrix distance given by sum of +1 if mat1[i][j] != mat2[i][j]

    Returns:
        [int]: numbers of terms that do not agree between the two matrices
    """
    dist=0
    
    for i in range(mat1.shape[0]):
        for j in range(mat1.shape[1]):
            if(mat1[i,j]!=mat2[i,j]):
                dist = dist+1 
    return dist




def AddRuleSpace(ruleGraph, mat, value, value2, value3, value4, dist=1):
    """This method is used to populate a graph that represent the relation in the space of rules
    giving the method a graph as rulespace a rule can be added. The Value 1,2,3 are just values 
    that can be stored to display in Gephi for example of for further analysis

    The method check for isomorphism 

    Args:
        RuleSpace ([type]): networkx graph 
        mat ([np.array]): matrix of the rule to add
        value<,2,3,4> (double or string): data to store in the graph for display
        dist (int, optional): [description]. Check how different two rules can (number of different element). 

    Raises:
        TypeError: Graph can only have 3 or 4 nodes rn for speed 
    """
    size = len(mat)
    index=0
    if ruleGraph.order()!= 0:
        index = ruleGraph.order()
        
    matrixString = ""
    a = np.squeeze(np.asarray(mat))
    for i in range(0,a.shape[0]):
        matrixString+="[ "+', '.join(['%d' %x for x in a[i,:]])+'],'
    
    ruleGraph.add_node(index, val1=value,rule=mat, ruleString = matrixString, val2 = value2, val3 = value3, val4 = value4)
    
    for n in range(ruleGraph.order()):
        if size == 4:
            if checkIfConjugate4(ruleGraph.nodes[n]["rule"], ruleGraph.nodes[index]["rule"], dist):
                if n!=index:
                    ruleGraph.add_edge(index,n)
                    
        if size == 3:
            if checkIfConjugate3(ruleGraph.nodes[n]["rule"], ruleGraph.nodes[index]["rule"], dist):
                if n!=index:
                    ruleGraph.add_edge(index,n)
        
        else :
            raise TypeError("can only be of size 3 or 4")





def checkIfConjugate2(mat1, mat2, distance):
    """Check if two matrix of dimension 2 only difer from n=distance terms considering isomorphism. 

    Args:
        distance ([int]): If set to 0 the two matrix need to be the same under isomorphism. Else we accept some difference

    Returns:
        [Boolean]: True if they are exacly distance from each other. Else False. 
    """
    for n in range(2):
        
        if MatrixDistance(T2m[n]*mat1*T2[n],mat2) == distance:
            return True
    
    return False


def checkIfConjugate3(mat1, mat2, distance):  
    """Check if two matrix of dimension 3 only difer from n=distance terms considering isomorphism. 

    Args:
        distance ([int]): If set to 0 the two matrix need to be the same under isomorphism. Else we accept some difference

    Returns:
        [Boolean]: True if they are exacly distance from each other. Else False. 
    """      
    for n in range(6):
        
        if MatrixDistance(T3m[n]*mat1*T3[n],mat2) == distance:
            return True
    
    return False


def checkIfConjugate4(mat1, mat2, distance):
    """Check if two matrix of dimension 4 only difer from n=distance terms considering isomorphism. 

    Args:
        distance ([int]): If set to 0 the two matrix need to be the same under isomorphism. Else we accept some difference

    Returns:
        [Boolean]: True if they are exacly distance from each other. Else False. 
    """      
    if distance == 0:
        if np.sum(mat1) != np.sum(mat2) :
            return False;
        if np.trace(mat1) != np.trace(mat2) :
            return False;
    
    for n in range(24):
        if MatrixDistance(T4m[n]*mat1*T4[n],mat2) == distance:
            return True
    
    return False


def countNonConjugate2(mat1):
    """Count the number of matrix that are different to this one when conjugating it for dimension 2

    Args:
        mat1 ([2x2 matrix]): rule

    Returns:
        [int]: number of matrix that are different to this one when conjugating it for dimension 2
    """    
    listmat = [] 
    for n in range(2):

        transformed = T2m[n]*mat1*T2[n]
        found = False
        for mat in listmat:
            if MatrixDistance(mat,transformed) == 0: 
                found = True
                break

        if not found:
            listmat.append(transformed)

    return listmat

def countNonConjugate3(mat1):
    """Count the number of matrix that are different to this one when conjugating it for dimension 3

    Args:
        mat1 ([3x3 matrix]): rule

    Returns:
        [int]: number of matrix that are different to this one when conjugating it for dimension 3
    """        
    listmat = [] 
    for n in range(6):
        
        transformed = T3m[n]*mat1*T3[n]
        found = False
        for mat in listmat:
            if MatrixDistance(mat,transformed) == 0: 
                found = True
                break

        if not found:
            listmat.append(transformed)

    return listmat
        


def contractGraph(G, distance, func):
    """Contract two nodes in graph G if the function between the two nodes are true 
    The graph is destroyed (modified)

    Args:
        G ([networkX]):  Graph to contract 
        distance ([Not used]): Not used 
        func ([func(node,node)]): [function that take two nodes as parameter and throw a boolean]

    Returns:
        The same graph but modified
    """    
    n=0
    m=0
    Glive = G
    nbNode = G.order()
    while n < nbNode:
        m=0
        while m < nbNode:
            if n!=m:
                if (Glive.has_node(n) and Glive.has_node(m)):
                    if func(Glive.nodes[n]["rule"], Glive.nodes[m]["rule"]):
                        Glive = nx.contracted_nodes(Glive, n, m)
            m += 1
        n += 1
        
    return Glive

def isValidRule(mat):
    """Check if a matrix (rule) is valid. A rule is valid if for each node there is at least one input. There is no condition on output.

    Returns:
        [boolean]: True if valid, false if not
    """    
    n = mat.shape[0]

    for row in mat:
        goodRow = False
        for conn in row :
            if conn != 0:
                goodRow = True
                break
                
        if not goodRow:
            return False    
            
    for column in np.matrix(mat).T:
        goodColumn = False
        for i in range(n) :
            if column[0,i] != 0:
                goodColumn = True
                break
                
        if not goodColumn:
            return False

    return True



def getMaxCycleLength(mat):
    """Return the maximum cycle length for the rule mat

    Args:
        mat ([np.array]): rule

    Returns:
        [int]: maximum cycle length 
    """
    
    graph = getGraphState(constructGraph(mat))
    undirected = graph[0].to_undirected()
    maxcycle, avgcycle = getMaxLength(list(nx.simple_cycles(graph[0])))
    
    return maxcycle


def getDensity(mat):
    """Return the density 
    """    
    n = mat.shape[0]
    connNb = 0
    
    for i in range(n):
        for j in range(n):
            if mat[i,j]!=0:
                connNb+=1
    
    return connNb/(n**2)

#TODO DUPLICATE 
def getDensity(mat):
    n = mat.shape[0]
    tot = 0
    for i in range(n):
        for j in range(n):
            if mat[i,j]!=0:
                tot+=1
    return tot/(n**2)

    
def generateRandomValidNetwork(size):
    n = size
    good = False
    newMat = 0
    
    while not good:
        newMat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                r = random.random()
                if r >= 0.666:
                    newMat[i][j]=1
                if r >= 0.333 and r < 0.666: 
                    newMat[i][j]=0
                if r < 0.333:
                    newMat[i][j]=-1
                    
        if isValidRule(newMat):
            good=True
        
    return newMat;
    
def generateRandomValidNetwork(size, density):
    n = size
    good = False
    newMat = 0
    p = density/2
    while not good:
        newMat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                r = random.random()
                if r >= density:
                    newMat[i][j]=0
                else:
                    if random.random()>0.5:
                        newMat[i][j]=1
                    else:
                        newMat[i][j]=-1
                    
        if isValidRule(newMat):
            good=True
        
    return newMat;


def listToArray(listMat, n):
    a_1d_array = np.array(listMat)
    reshaped_to_2d = np.reshape(a_1d_array, (n, n))
    return reshaped_to_2d


def arrayToList(mat):
    flatten_array = mat.flatten()
    listMat = flatten_array.tolist()
    return listMat  

def distanceBetweenList(list1, list2):
    n = len(list1)
    dist = 0
    for i in range(n):
        if list1[i]!=list2[i]:
            dist+=1
    
    return dist

def getNbSame(population):
    tot = 0
    n = len(population)
    print(n)
    for i in range(n):
        for j in range(i+1,n):
            if distanceBetweenList(population[i],population[j])==0:
                tot+=1
    return tot

def getAverage1To1Distance(population):
    total = 0
    for i in range(len(population)-1):
        total+=distanceBetweenList(population[i],population[i+1])
    
    return total/(len(population)-1)




def is_same_dynamic(list_dyn_couple, init_vec, final_vec):
    for couple in list_dyn_couple:
        val1 = couple[0]
        val2 = couple[1]
        index_val1 =init_vec.index(val1)
        if final_vec[index_val1]!=val2:
            return False
    
    return True


def get_1d_property(vec):
    """formatting of 1 dimensional properties for a network (vec)

    Args:
        vec : network in vector form (ie. list of element in the matrix in order starting with top left)

    Returns:
        [ mxc_inhib, avgc_inhib, mxc_exct, avgc_exct, mxc_all, avgc_all, totalInhibition, totalExcitation, totalAutoInhibition, totalAutoExcitation]
    """
        
    n = int(abs(math.sqrt(len(vec))))
    mat = listToArray(vec, n)
    G_inhibition, G_excitation, G_all = nx.DiGraph(), nx.DiGraph(), nx.DiGraph()

    for i in range(n):
        G_excitation.add_node(i)
        G_all.add_node(i)
        G_inhibition.add_node(i)
    
    density = 0
    
    for i in range(n):
        for j in range(n):
            if mat[i,j]!=0:
                density+=1
                
    density = density/(n**2) 
    
    for i in range(n):
        for j in range(n):
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
    totalInhibition, totalExcitation, totalAutoInhibition, totalAutoExcitation = matstat[0,0], matstat[0,1], matstat[0,2], matstat[0,3]
    
    return [ mxc_inhib, avgc_inhib, mxc_exct, avgc_exct, mxc_all, avgc_all, totalInhibition, totalExcitation, totalAutoInhibition, totalAutoExcitation]
         