import os
from boolean.motif_util import get2motif_graph, get3motif_graph, just_3_motif, DecisionTree, get_tree_input_3node
from boolean.boolean_util import arrayToList, countNonConjugate3, generateRandomValidNetwork



def script_sample_motif_z_value():
    """
    This script output the z_value of the motif found in a random graph. 
    Those value should be close to zero in magnitude when the sample is large enough. 
    The output is a file with 3286 rows for the 3286 motifs. 

    """

    nb_sample = 5000
    node = 7
    density = 0.7
    output_file = "./data_example/test/z_value_3_motifs_nodes_"+str(node)+".txt"

    #set up the files
    file = open(output_file, 'a') 
    header = "z_value,\n"
    if os.stat(output_file).st_size < 10:
        file.writelines(header)

    #list all possible motifs and build the decision tree
    list2motif, motif2Info, list2mat, list22info =  get2motif_graph()
    motif_3_repr, all_3_motif_mat, all_3_motif_repr_nb = get_tree_input_3node()
    mainTree = DecisionTree.constructTree(all_3_motif_mat, all_3_motif_repr_nb, all_3_motif_repr_nb)
    nwayMotif = []
    list3motif, motif3Info, list3mat, list33info =  get3motif_graph()
    for mat in motif_3_repr:
        nwayMotif.append(len(countNonConjugate3(mat)))

    #sample motifs
    z_list = [0]*len(list3motif)
    for i in range(nb_sample):
        mat = generateRandomValidNetwork(node, density)
        motif_3 = just_3_motif(arrayToList(mat), list2motif, motif2Info, list3motif, motif3Info, nwayMotif, mainTree, list2mat, list3mat)
        z_list = [x[0]+x[1] for x in zip(z_list,motif_3)]
    
    #print the output
    for i in range(len(motif_3)):
        list_out = [z_list[i]/nb_sample]
        str_output = ""
        for elem in list_out:
            str_output += str(elem) + ","
        str_output+= "\n"
        file.writelines(str_output)

    file.close() 


if __name__ == "__main__":
    script_sample_motif_z_value()