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

from boolean_util import getInfoRule, checkIfConjugate3, checkIfConjugate2, constructGraph, listToArray, countNonConjugate2

def ternaryDigit (n):
    if n == 0:
        return '0'
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))
    return ''.join(reversed(nums))

def checkConnected2node(mat):
    
    if mat[0,1]!=0 or mat[1,0]!=0:
        return True

    return False

def getAllMotif_2nodes_adjacency():
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
    
    listmat, listInfo = getAllMotif_2nodes_adjacency()
    listgraph = []
    
    for mat in listmat:
        listgraph.append(constructGraph(mat))
        
    return listgraph, listInfo, listmat, listInfo
        

def get3motif_graph():
    
    listmat, listInfo = getAllMotif_3nodes_adjacency()
    listgraph = []
    
    
    for mat in listmat:
        listgraph.append(constructGraph(mat))
        
    return listgraph, listInfo, listmat, listInfo
        
    
def get2motif(rule, listgraph, motifInfo):
    
    graphRule = constructGraph(rule)
    em = iso.numerical_edge_match('weight', 1)
    count = [0]*len(listgraph)

    for sub_nodes in itertools.combinations(graphRule.nodes(),2):
        subg = graphRule.subgraph(sub_nodes)
        subgInfo = getInfoRule(  nx.adjacency_matrix(subg) )
        
        for m in range(len(listgraph)):
            if (subgInfo == motifInfo[m]).all() :
                if  nx.is_isomorphic(subg, listgraph[m],edge_match=em):
                    count[m]+=1
                    break
                
    return count;


def get3motif(rule, listgraph, motifInfo):
    
    graphRule = constructGraph(rule)
    
    em = iso.numerical_edge_match('weight', 1)
    count = [0]*len(listgraph)
 
    for sub_nodes in itertools.combinations(graphRule.nodes(),3):

        subg = graphRule.subgraph(sub_nodes)
        
        if nx.is_weakly_connected(subg):
            
            subgInfo = getInfoRule(  nx.adjacency_matrix(subg) )
            found = False



            for m in range(len(listgraph)):
                if (subgInfo == motifInfo[m]).all() :
                    if  nx.is_isomorphic(subg, listgraph[m],edge_match=em):
                        count[m]+=1
                        found = True
                        break

            if not found:
                print( nx.adjacency_matrix(subg))
                raise ValueError( str(nx.adjacency_matrix(subg)))
                
    return count


"""
TODO: Not used anymore probably safe to delete
"""
def get2MotifRatio(motifGraph, motifInfo, pop):
            
        count_ratio = [0]*len(motifGraph)
        
        #compile motif
        for ind in pop:
            
            n = int(abs(math.sqrt(len(ind))))
            mat1 = listToArray(ind,n)
            count = get2motif(mat1, motifGraph, motifInfo)
            tot = 0
            for c in count:
                tot+=c
            for i in range(len(count)):
                count[i] = count[i]/tot
                count_ratio[i] += count[i]
                
        for i in range(len(count)):
                count_ratio[i] = count_ratio[i]/len(pop)

        
        return count_ratio

    
def output(filename, sheet, bigList):
    
    
    book = xlwt.Workbook()
    sh = book.add_sheet(sheet)

    for i, lineOfData in enumerate(bigList, 1):
        for j, data in enumerate(lineOfData, 0):
            sh.write(i, j, data)
    
    book.save(filename) 
    book.close()
    
    
def outputXLSX(filename, sheet, bigList):
    
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()

    for i, lineOfData in enumerate(bigList, 1):
        for j, data in enumerate(lineOfData, 0):
            worksheet.write(i, j, data)
    
    workbook.close()
        

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

#doesnt give the probability but give the average count 
#for a random network
def getProbabilty3motifs(mat, density, nodeNb, nwayMotif):
    
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

#doesnt give the probability but give the average count 
#for a random network
def getProbabilty2motifs(mat, density, nodeNb):
    
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
