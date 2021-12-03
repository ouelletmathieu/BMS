from boolean.boolean_util import constructGraph, getGraphState, getMaxLength
import networkx as nx



def get_symmetry_number(mat):
    graph_mat = constructGraph(mat)
    graph = getGraphState(graph_mat)
    cycle_list = list(nx.simple_cycles(graph[0]))
    max_cycle, _ = getMaxLength(cycle_list)
    

    init_list = graph[1]
    final_list= graph[2]

    nb_sym = 0
    nb_sym_transcient = 0
    nb_sym_cycle = 0
    
    ratio_sym = 0
    ratio_sym_transcient = 0
    ratio_sym_cycle = 0
    n_transcient = 0
    n_cycle = 0
    
    list_in_cycle = []
    
    for cyc in cycle_list:
        for node in cyc:
            list_in_cycle.append(init_list[node])

    
    for i in range(len(init_list)):

        init_vec_not = [1-x for x in init_list[i]]
        final_vec_not = [1-x for x in final_list[i]]

        for j in range(len(init_list)):
            
            #find the good transition
            if init_list[j]==init_vec_not:
                
                #if perfect count it as a symmetric transition 
                if final_list[j]==final_vec_not:
                    nb_sym+=1
                    if init_list[j] in list_in_cycle:
                        nb_sym_cycle+=1
                    else:
                        nb_sym_transcient+=1  
                
                count_bit_sym =0
                for k in range(len(final_list[j])):
                    if final_list[j][k]==final_vec_not[k]:
                        count_bit_sym+=1
                
                ratio_bit_sym = count_bit_sym/len(final_list[j])
                ratio_sym += ratio_bit_sym
                
                if init_list[j] in list_in_cycle:
                        ratio_sym_cycle += ratio_bit_sym
                        n_cycle += 1
                else:
                        ratio_sym_transcient += ratio_bit_sym
                        n_transcient += 1
                
                
    ratio_sym = ratio_sym/len(init_list)
    if n_cycle > 0 :
        ratio_sym_cycle = ratio_sym_cycle/n_cycle
    if n_transcient > 0 :
        ratio_sym_transcient = ratio_sym_transcient/n_transcient
                    

    return nb_sym, nb_sym_transcient, nb_sym_cycle, ratio_sym, ratio_sym_transcient, ratio_sym_cycle, max_cycle

