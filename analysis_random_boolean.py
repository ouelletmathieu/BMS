import PyBoolNet
from PyBoolNet import StateTransitionGraphs as STGs
from PyBoolNet import FileExchange
import itertools
from PyBoolNet import FileExchange
import os
from analysis_real_boolean import BooleanIter
from boolean.print_util import print_list_file, create_text_file


mainpath = "./GINSIM/kauffman_random/"
output = "./data_example/real_boolean/random_network.txt"
header = "file_name,n,k,topology,linkage,id,ratio_sym,\n"

def load_model_analysis():
    
    file = create_text_file(output, header, delete=True)
    
    model_list_fromdirectory = os.listdir(mainpath)
    model_list = []
    
    for model in model_list_fromdirectory:
        if model[0]=="r":
            model_list.append(["file",model])
    
    for source, model in model_list:

        bin_set = {0,1}

        if source == "repo":
            primes = PyBoolNet.Repository.get_primes(model)
        else:
            primes = FileExchange.bnet2primes(mainpath+model)
        
        n =0
        ls_elem =  []

        for node in primes:
            ls_elem.append(node)
            n+=1
        
        #if model is small enough we can test all transition 
        if n<17:

            iterator = itertools.product(bin_set,repeat = n)
            

        else:
            iterator = BooleanIter(n, 2**16)
     

        ntot = 0
        invert_tot = 0
        
        for vec_lst in iterator:

            vector_init = ""

            for v in vec_lst:
                vector_init+=str(v)

            vector_next = []

            neg_vector_inti = ""
            neg_vector_next = []


            for i in range(len(vector_init)):
                if vector_init[i]=="1":
                    neg_vector_inti += "0"
                else:
                    neg_vector_inti += "1"

            newVal =  STGs.successor_synchronous(primes, vector_init)

            for lm in ls_elem:
                vector_next.append(newVal[lm])

            neg_newVal = STGs.successor_synchronous(primes, neg_vector_inti)
            for lm in ls_elem:
                neg_vector_next.append(neg_newVal[lm])

            invert = 0
            for i in range(len(vector_next)):
                if vector_next[i]!=neg_vector_next[i]:
                    invert+=1
            ntot += n
            invert_tot +=  invert 

    
        list_param = model.split("_")
        
        output_list = [model,list_param[1],list_param[2],list_param[3],list_param[4],list_param[5],invert_tot/ntot]
        print_list_file(output_list, file)
        file.flush()

    file.close()
        
if __name__ == "__main__":
    load_model_analysis()
        
