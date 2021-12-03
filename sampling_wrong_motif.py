import pandas as pd
import os
from boolean.motif_util import get3motif_graph, generate_wrong_motif_mat 
from sampling_1d_properties import get_header, print_1d_property_file, get_header_symmetry
from boolean.print_util import print_list_file, create_text_file
from sampling_1d_properties import print_symmetry_file

nb_sample = 5000
nodeNb = 7
max_try = 10000

dt_string = "02_11_08_08"
main_path = "./data_example/wrong_motif_network/wrong_motif_nodes_"+str(nodeNb)+".txt"
path_cat = "./data_example/pareto_density_cycle/"+dt_string+"/motif_3_cat.csv"
output_file_symmetry = "./data_example/symmetry/symmetry_wrong_motif_network_nodes_"+str(nodeNb)+".txt"


header = get_header().replace("\n", "")  + "density_from_motif,\n"
header_symmetry = get_header_symmetry().replace("\n", "") + "density_from_motif,\n"
file = create_text_file(main_path, header)
file_symmetry = create_text_file(output_file_symmetry, header_symmetry)


def random_sampling_wrong_motif():
    
    _, _, list3mat, _ =  get3motif_graph()
    
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
        list_output = print_1d_property_file(k, mat, nodeNb)
        list_output.append(density_from_motif)
        print_list_file(list_output, file)

        list_out_symmetry = print_symmetry_file(k, mat, nodeNb)
        list_out_symmetry.append(density_from_motif)
        print_list_file(list_out_symmetry,file_symmetry)


    file_symmetry.close() 
    file.close()




random_sampling_wrong_motif()

