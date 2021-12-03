import pandas as pd
import os
import random
from boolean.motif_util import get2motif_graph, get_tree_input_3node, DecisionTree, get3motif_graph, generate_wrong_motif_mat 
from boolean.boolean_util import countNonConjugate3
from sampling_1d_properties import get_header, print_1d_property_file
from boolean.print_util import print_list_file

nb_sample = 5000
nodeNb = 7
max_try = 10000

dt_string = "02_11_08_08"
main_path = "./data_example/wrong_motif_network/wrong_motif_nodes_"+str(nodeNb)+".txt"
path_cat = "./data_example/pareto_density_cycle/"+dt_string+"/motif_3_cat.csv"


file = open(main_path, 'a') 
header = get_header()
header = header.replace("\n", "")  + "density_from_motif,\n"
if os.stat(main_path).st_size < 10:
    file.writelines(header)


def random_sampling_wrong_motif():
    
    list3motif, motif3Info, list3mat, list33info =  get3motif_graph()
    
    #get motif
    df_3 = pd.read_csv(path_cat)   
    list_3 = df_3["3motif"].tolist()
    list_wrong_motif = list()
    for i in range(len(list_3)):
        if list_3[i]==-1 :
            list_wrong_motif.append(list3mat[i])
            
    #main loop for sample 
    for  k in range(nb_sample):
           mat, density_from_motif =  generate_wrong_motif_mat(list_wrong_motif, nodeNb, max_try)
           list_output = print_1d_property_file(k, mat, nodeNb, file)
           list_output.append(density_from_motif)
           print_list_file(list_output, file)

    file.close()


def random_sampling_random_motif():
    
    list3motif, motif3Info, list3mat, list33info =  get3motif_graph()
    
    #get motif
    df_3 = pd.read_csv(path_cat)   
    list_3_real = df_3["3motif"].tolist()
    list_wrong_motif = list()
    count_wrong_motif = 0
    for i in range(len(list_3_real)):
        if list_3_real[i]==-1 :
            list_wrong_motif.append(list3mat[i])
            count_wrong_motif+=1
            

    #main loop for sample 
    for  k in range(nb_sample):

        list_all_motif_id = list(range(len(list_3_real)))
        list_motif_to_use_id = random.sample(list_all_motif_id, count_wrong_motif)
        list_motif_to_use = [list3mat[id_mot] for  id_mot in list_motif_to_use_id]

        mat, density_from_motif =  generate_wrong_motif_mat(list_motif_to_use, nodeNb, max_try)
        list_output = print_1d_property_file(k, mat, nodeNb, file)
        list_output.append(density_from_motif)
        print_list_file(list_output, file)

    file.close()



random_sampling_wrong_motif()

