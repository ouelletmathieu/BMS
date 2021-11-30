import networkx as nx
import numpy as np
import math
import networkx as nx
import numpy as np
import math
import itertools
import xlwt
import networkx.algorithms.isomorphism as iso
import xlsxwriter
from boolean.boolean_util import getDensity, getMaxCycleLength
from boolean.boolean_util import getInfoRule, checkIfConjugate3, checkIfConjugate2, constructGraph, listToArray, countNonConjugate2



def checkConnected2node(mat):
    """Test if 2 node are connected together
    """
    if mat[0,1]!=0 or mat[1,0]!=0:
        return True

    return False

def getAllMotif_2nodes_adjacency():
    """return all the potential rule matrix for 2 motifs.

    Returns:
        listmat, listInfo
    """
    nodes = 2
    lenght = nodes**2
    listmat = []
    listInfo = [] 
    
    for k in range(1,3**lenght):


        base3 =  [int(x) for x in ternaryDigit(k)]
        base3 = [0] * (lenght - len(base3)) + base3

    
        mat = np.zeros((nodes,nodes))
        
        for i in range(nodes):
            for j in range(nodes):
                if base3[i*nodes+j]==0:
                    mat[i,j]=0
                elif base3[i*nodes+j]==1:
                    mat[i,j]=1
                elif base3[i*nodes+j]==2:
                    mat[i,j]=-1
        if checkConnected2node(mat) :
            symm = False
            matInfo = getInfoRule(mat)
            
            for i in range(len(listmat)):
                if (matInfo==listInfo[i]).all():
                    if(checkIfConjugate2(listmat[i], mat, 0)):
                        symm = True
                        break

            if not symm:
                listmat.append(mat)
                listInfo.append(matInfo)
    
    return listmat, listInfo

def checkConnected3node(mat):
    """Test if 3 node are connected together
    """
    diag01 = False
    diag12 = False
    diag02 = False
    
    if mat[0,1]!=0 or mat[1,0]!=0:
        diag01=True
    if mat[1,2]!=0 or mat[2,1]!=0:
        diag12 = True
    if mat[0,2]!=0 or mat[2,0]!=0:
        diag02 = True
    
    if diag01 and (diag12 or diag02) :
        return True
    if diag12 and diag02:
        return True
    
    return False


def getAllMotif_3nodes_adjacency():
    """return all the potential rule matrix for 3 motifs. The process is slow.
    TODO make it faster 

    Returns:
        listmat, listInfo
    """
    nodes = 3
    lenght = nodes**2
    listmat = []
    listInfo = [] 
    print("total matrix = ", 3**lenght)
    notConn = 0
    
    for k in range(1,(3**lenght)):


        base3 =  [int(x) for x in ternaryDigit(k)]
        base3 = [0] * (lenght - len(base3)) + base3

    
        mat = np.zeros((nodes,nodes))
        
        for i in range(nodes):
            for j in range(nodes):
                if base3[i*nodes+j]==0:
                    mat[i,j]=0
                elif base3[i*nodes+j]==1:
                    mat[i,j]=1
                elif base3[i*nodes+j]==2:
                    mat[i,j]=-1
                
        if checkConnected3node(mat) :
            symm = False
            matInfo = getInfoRule(mat)
            
            for i in range(len(listmat)):
                if (matInfo==listInfo[i]).all():
                    if(checkIfConjugate3(listmat[i], mat, 0)):
                        symm = True
                        break

            if not symm:
                listmat.append(mat)
                listInfo.append(matInfo)
                if len(listmat)%250==0:
                    print(len(listmat))
        else:
            notConn += 1 
    print("number of notConn = ",notConn)
    print("number of accepted motif = ", len(listmat))
    return listmat, listInfo



def get2motif_graph():
    """Create networkX graph of all 2 node motifs
    Returns:
        listgraph, listInfo, listmat, listInfo
    """   
    listmat, listInfo = getAllMotif_2nodes_adjacency()
    listgraph = []
    
    for mat in listmat:
        listgraph.append(constructGraph(mat))
        
    return listgraph, listInfo, listmat, listInfo
        

def get3motif_graph():
    """Create networkX graph of all 3 node motifs
    Returns:
        listgraph, listInfo, listmat, listInfo
    """
    listmat, listInfo = getAllMotif_3nodes_adjacency()
    listgraph = []

    for mat in listmat:
        listgraph.append(constructGraph(mat))
        
    return listgraph, listInfo, listmat, listInfo
        
    
def get2motif(mat, list2mat, motifInfo):
    """get the count of 2 motif in the network mat. Slow way that do not use a decision tree

    Args:
        mat ([np.array]): [description]
        list3mat ([np.array, np.array, ... ]): list of all possible 3 motifs. Can be generated using get2motif_graph()
        motifInfo ([see get2motif_graph]): list of data describing the 3 motifs. Can be generated using get2motif_graph()

    Returns:
        [int, int, ...]: count of motifs in the graph for all the motif in list3mat
    """    
    graphRule = constructGraph(mat)
    em = iso.numerical_edge_match('weight', 1)
    count = [0]*len(list2mat)

    for sub_nodes in itertools.combinations(graphRule.nodes(),2):
        subg = graphRule.subgraph(sub_nodes)
        subgInfo = getInfoRule(  nx.adjacency_matrix(subg) )
        
        for m in range(len(list2mat)):
            if (subgInfo == motifInfo[m]).all() :
                if  nx.is_isomorphic(subg, list2mat[m],edge_match=em):
                    count[m]+=1
                    break
                
    return count;


def get3motif(mat, list3mat, motifInfo):
    """get the count of 3 motif in the network mat. Slow way that do not use a decision tree

    Args:
        mat ([np.array]): [description]
        list3mat ([np.array, np.array, ... ]): list of all possible 3 motifs. Can be generated using get3motif_graph()
        motifInfo ([see get3motif_graph]): list of data describing the 3 motifs. Can be generated using get3motif_graph()

    Returns:
        [int, int, ...]: count of motifs in the graph for all the motif in list3mat
    """
    graphRule = constructGraph(mat)
    
    em = iso.numerical_edge_match('weight', 1)
    count = [0]*len(list3mat)
 
    for sub_nodes in itertools.combinations(graphRule.nodes(),3):

        subg = graphRule.subgraph(sub_nodes)
        
        if nx.is_weakly_connected(subg):
            
            subgInfo = getInfoRule(  nx.adjacency_matrix(subg) )
            found = False



            for m in range(len(list3mat)):
                if (subgInfo == motifInfo[m]).all() :
                    if  nx.is_isomorphic(subg, list3mat[m],edge_match=em):
                        count[m]+=1
                        found = True
                        break

            if not found:
                print( nx.adjacency_matrix(subg))
                raise ValueError( str(nx.adjacency_matrix(subg)))
                
    return count



def ternaryDigit (n):
    if n == 0:
        return '0'
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))
    return ''.join(reversed(nums))

def nCr(n,r):
    """compute the number of combination (selecting r in n)
    """
    f = math.factorial
    return f(n) / f(r) / f(n-r)


def getProbabilty3motifs(mat, density, nodeNb, nwayMotif):
    """This function DO NOT gives the probability but gives the expected count of a 3 nodes motif for a random networks 
    TODO change the name
    Args:
        mat ([np.array nxn]): matrix of the motif to get the count
        density ([double]): density of the graph that would contain the motif
        nodeNb ([int]): number of node in the graph
        nwayMotif ([int]): number of way the motif can be created. Use the function countNonConjugate

    Returns:
        totalProb, standardDeviation: total count of expected motif and the standard deviation expected for the motif
    """
    
    nedge = 1 - density
    edge = density   
    nchoose3 = nCr(nodeNb, 3)
    pconn = 1 - ((nedge)**4)*(3*(edge)**2 + 6*nedge*edge + nedge**2)
    nbEdge = 0
    
    for i in range(3):
        for j in range(3):
            if mat[i,j]!=0:
                nbEdge+=1
    noedge = 9-nbEdge
    pmotif3 = ((nedge)**noedge)*((edge*0.5)**nbEdge)
    
    totalProb = nchoose3*nwayMotif*pmotif3
    
    standardDeviation = math.sqrt(totalProb*(1-nwayMotif*pmotif3))
    
    return totalProb, standardDeviation


def getProbabilty2motifs(mat, density, nodeNb):
    """This function DO NOT gives the probability but gives the expected count of a 2 nodes motif for a random networks 
    TODO change the name
    Args:
        mat ([np.array nxn]): matrix of the motif to get the count
        density ([double]): density of the graph that would contain the motif
        nodeNb ([int]): number of node in the graph

    Returns:
        totalProb, standardDeviation: total count of expected motif and the standard deviation expected for the motif
    """   
    nedge = 1 - density
    edge = density   
    nchoose2 = nCr(nodeNb, 2)
    pconn = 1 - (nedge)**2
    nwayMotif =  len(countNonConjugate2(mat))
    nbEdge = 0
    
    for i in range(2):
        for j in range(2):
            if mat[i,j]!=0:
                nbEdge+=1
                
    noedge = 4-nbEdge
    pmotif2 = ((nedge)**noedge)*((edge*0.5)**nbEdge)
    
    totalProb = nchoose2*nwayMotif*pmotif2

    standardDeviation = math.sqrt(totalProb*(1-nwayMotif*pmotif2))
    
    return totalProb, standardDeviation



def getZvalue3motifs(listMatGraph, MotifCount, density, nodeNb, nonConjugate_count):
    
    Zvalue = [0]*len(listMatGraph)

    for i in range(len(listMatGraph)):
        p,var = getProbabilty3motifs(listMatGraph[i], density, nodeNb,nonConjugate_count[i])
        
        if p==0 and MotifCount[i]==0:
            Zvalue[i]=0
        else:
            Zvalue[i]=(MotifCount[i]-p)/var
    
    return Zvalue


def getZvalue2motifs(listMatGraph, MotifCount, density, nodeNb):
    
    Zvalue = [0]*len(listMatGraph)
    
    for i in range(len(listMatGraph)):
        p,var = getProbabilty2motifs(listMatGraph[i], density, nodeNb)
        if p==0 and MotifCount[i]==0:
            Zvalue[i]=0
        else:
            Zvalue[i]=(MotifCount[i]-p)/var
            
    return Zvalue



#we added list2mat, list3mat to the method (need to keep the change)
def get_z_val_2_3(vec,list2motif,motif2Info,list3motif,fmotif3Info, nwayMotif, rootNode, list2mat, list3mat):
    
    n = int(abs(math.sqrt(len(vec))))
    mat = listToArray(vec, n)
    
    comp = getDensity(mat)
    dyn = getMaxCycleLength(mat)
    
    motif2Vec = get2motif(mat, list2motif, motif2Info)
    
    motif3Vec = get3motif_tree(mat, rootNode, list3motif)
    
    r = getDensity(mat)
    
    z2value = getZvalue2motifs(list2mat, motif2Vec, r, n)
    z3value = getZvalue3motifs(list3mat, motif3Vec, r, n,nwayMotif)
    
    return [z2value + [comp, dyn], z3value + [comp, dyn]]



def just_3_motif(vec, list2motif, motif2Info, list3motif, fmotif3Info, nwayMotif, rootNode, list2mat, list3mat):

    zval = get_z_val_2_3(vec,list2motif,motif2Info,list3motif,fmotif3Info, nwayMotif, rootNode, list2mat, list3mat)
    return zval[1][:-2]



def get3motif_tree(rule, rootNode, count_len):
    
    graphRule = constructGraph(rule)
    #em = iso.numerical_edge_match('weight', 1)
    count = [0]*len(count_len)
    
    for sub_nodes in itertools.combinations(graphRule.nodes(),3):

        mat = rule[np.ix_(sub_nodes,sub_nodes)]
        vec = [x for x in mat.flat]
        id_motif = rootNode.goToLeaf(0, vec)

        if id_motif != -2:
            count[id_motif[0]]+=1

    return count;    

def get_tree_input_3node():

    nodes = 3
    lenght = nodes**2
    list_1_reprentative = []
    all_matrix = []
    all_matrix_repr_nb = [] 
    listInfo = []
    
    for k in range(1,3**lenght):

        base3 =  [int(x) for x in ternaryDigit(k)]
        base3 = [0] * (lenght - len(base3)) + base3

    
        mat = np.zeros((nodes,nodes))
        
        for i in range(nodes):
            for j in range(nodes):
                if base3[i*nodes+j]==0:
                    mat[i,j]=0
                elif base3[i*nodes+j]==1:
                    mat[i,j]=1
                elif base3[i*nodes+j]==2:
                    mat[i,j]=-1
                    
        if checkConnected3node(mat) :
            
            symm = False
            matInfo = getInfoRule(mat)
            all_matrix.append(mat.flatten())
            
            for i in range(len(list_1_reprentative)):
                if (matInfo==listInfo[i]).all():
                    if(checkIfConjugate3(list_1_reprentative[i], mat, 0)):
                        symm = True
                        all_matrix_repr_nb.append(i)
                        break

            if not symm:
                list_1_reprentative.append(mat)
                listInfo.append(matInfo)
                if len(list_1_reprentative)%250==0:
                    print(len(list_1_reprentative))
                all_matrix_repr_nb.append(len(list_1_reprentative)-1)
    print("finish")
    
    return list_1_reprentative, all_matrix, all_matrix_repr_nb


class DecisionTree:
    
    def __init__(self, vec):
        self.parsed_vec = vec
        self.children_dict = dict()
        self.id = -1
        self.data = -1
        self.leaf = False
    
    @staticmethod
    def constructTree(list_vec, list_id, list_data):
        
        if len(list_vec)!=len(list_id) or len(list_id)!=len(list_data):
            print("not the same size")
            raise ValueError("should be the same size")
        
        rootNode =  DecisionTree([])
        
        for i in range(len(list_vec)):
            rootNode.addToDecisionTree(0, list_vec[i], list_id[i], list_data[i])

        return rootNode
    
                
    def addChildren(self,next_parsed, newNode ):
        self.children_dic[next_parsed, newNode]
    
    
    def __str__(self):
        return ""+str(self.parsed_vec)+" , "+str(self.id)+ " , " + str(self.data)
    
    
    def addToDecisionTree(self, active_id, vec, leaf_id, data):
        nextVal = int(vec[active_id])
        
        if not active_id==len(vec)-1:
            
            if nextVal in self.children_dict:
                self.children_dict[nextVal].addToDecisionTree(active_id+1,vec,leaf_id, data);
            else:
                
                newNode = DecisionTree(vec[0:active_id+1])
                self.children_dict[nextVal] = newNode
                self.children_dict[nextVal].addToDecisionTree(active_id+1,vec,leaf_id, data)
        
        else:
            
            newNode = DecisionTree(vec[0:active_id+1])
            newNode.id=leaf_id
            newNode.leaf=True
            newNode.data=data
            self.children_dict[nextVal] = newNode
            
            
    #return id, data if node is a leaf, return -2 if not found or not a leaf
    def goToLeaf(self,active_id, vec):
        
        
        nextVal = int(vec[active_id])
        
        #check if leaf
        if not active_id==len(vec)-1:
            #if next vector exist in dict
            if nextVal in self.children_dict:
                return self.children_dict[nextVal].goToLeaf(active_id+1, vec);
            else:
                return -2;
        
        else:
            return self.children_dict[nextVal].id, self.children_dict[nextVal].data
        
