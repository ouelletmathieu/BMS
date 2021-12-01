import networkx as nx
import math
from itertools import combinations, chain
from boolean.boolean_util import constructGraph, listToArray



def get_minimum_node_set(cycle_list, node_list):
    
    s = node_list
    min_sub = node_list

    for sub in chain.from_iterable(combinations(s, r) for r in range(len(s)+1)):
        
        sub = list(sub)
        
        if len(sub)>0:
            one_per_cycle = True
        
            for cycle in cycle_list:
                this_cycle_ok = False
                for node in sub:
                    if node in cycle:
                        this_cycle_ok=True
                        break
                if not this_cycle_ok:
                    one_per_cycle = False
                    break
               
            if one_per_cycle:
                if len(sub) < len(min_sub):
                    min_sub = sub
    
    return min_sub


def get_minimum_node_set_all(cycle_list, node_list):
    
    s = node_list
    min_sub = [node_list]
    
    for sub in chain.from_iterable(combinations(s, r) for r in range(len(s)+1)):
        
        sub = list(sub)
        
        if len(sub)>0:
            
            one_per_cycle = True
        
            for cycle in cycle_list:
                this_cycle_ok = False
                for node in sub:
                    if node in cycle:
                        this_cycle_ok=True
                        break
                if not this_cycle_ok:
                    one_per_cycle = False
                    break
               
            if one_per_cycle:
                if len(sub) < len(min_sub[0]):
                    min_sub = [list(sub)]
                elif len(sub) == len(min_sub[0]):
                    min_sub.append(list(sub))
                
    return min_sub


def get_positive_cycle(cycle_list, G):
    
    positive_cycle_list = list()
    negative_cycle_list = list()
    
    for cycle in cycle_list:
        cyc = cycle.copy()
        cyc.append(cyc[0])
        
        neg_edg = 0
        
        for i in range(0,len(cyc)-1):
            edg = G.edges[cyc[i], cyc[i+1]]['weight']
            if edg ==-1:
                neg_edg+=1
        
        if neg_edg%2==0:
            positive_cycle_list.append(cyc)
        else:
            negative_cycle_list.append(cyc)     

    return positive_cycle_list, negative_cycle_list 


def countShared(list1, list2):
    count = 0
    
    for l1 in list1:
        for l2 in list2:
            if l1==l2:
                count+=1           
    return count
            
          
#not the good definition of tau
def get_tau_dumb(mat):

    G = constructGraph(mat)
    G_pos,G_neg = nx.DiGraph(), nx.DiGraph()
    
    for u, v, weight in G.edges(data='weight'):
        if weight is not None:
            if weight == -1:
                G_neg.add_edge(u,v)
            elif weight == 1:
                G_pos.add_edge(u,v)

    cycle_list_neg = list(nx.simple_cycles(G_neg))
    cycle_list_pos = list(nx.simple_cycles(G_pos))
    cycle_list_all = list(nx.simple_cycles(G))
    
    node_set_pos = get_minimum_node_set(cycle_list_pos, list(G_pos))
    node_set_neg = get_minimum_node_set(cycle_list_neg, list(G_neg))
    node_set_all = get_minimum_node_set(cycle_list_all, list(G))


    return node_set_pos, node_set_neg, node_set_all


def get_tau_good(mat):

    G = constructGraph(mat)

    cycle_list_all = list(nx.simple_cycles(G))
    positive_cycle_list, negative_cycle_list  = get_positive_cycle(cycle_list_all, G)
    node_set_all = get_minimum_node_set(cycle_list_all, list(G))
    node_set_pos = get_minimum_node_set(positive_cycle_list, list(G))
    node_set_neg = get_minimum_node_set(negative_cycle_list, list(G))
    
    return node_set_pos, node_set_neg, node_set_all 


def get_tau_ratio(mat):
    G = constructGraph(mat)

    cycle_list_all = list(nx.simple_cycles(G))
    positive_cycle_list, negative_cycle_list  = get_positive_cycle(cycle_list_all, G)
    list_node_pos = get_minimum_node_set_all(positive_cycle_list, list(G))
    list_node_neg = get_minimum_node_set_all(negative_cycle_list, list(G))
   
    min_shared = len(list(G))
    max_shared = 0
    
    for pos_set in list_node_pos:
        for neg_set in list_node_neg:
            count = countShared(pos_set, neg_set)
            
            if count > max_shared:
                max_shared = count
            if count < min_shared:
                min_shared = count
            
    return max_shared, min_shared
    

def get_tau_descriptor(vec):
    """formatting of tau descriptor properties for a network (vec)

    Args:
        vec : network in vector form (ie. list of element in the matrix in order starting with top left)

    Returns:
        [node_pos_dumb,node_neg_dumb,node_all_dumb,node_pos_good, node_neg_good,node_all_good]
    """

    n = int(abs(math.sqrt(len(vec))))
    mat = listToArray(vec, n)
        
    node_set_pos, node_set_neg, node_set_all = get_tau_dumb(mat)
    node_pos_good, node_neg_good, node_all_good = get_tau_good(mat)
    node_pos_dumb, node_neg_dumb, node_all_dumb  = len(node_set_pos), len(node_set_neg), len(node_set_all)    
    node_pos_good, node_neg_good,node_all_good = len(node_pos_good), len(node_neg_good), len(node_all_good)
    
    return [node_pos_dumb,node_neg_dumb,node_all_dumb,node_pos_good, node_neg_good,node_all_good]


