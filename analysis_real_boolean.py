import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys
import random
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys
import random
import math
from deap.benchmarks.tools import igd
import itertools
import PyBoolNet
from PyBoolNet import StateTransitionGraphs as STGs
from PyBoolNet import FileExchange
import itertools
from PyBoolNet import FileExchange
from boolean.print_util import print_list_file, create_text_file


path_out = "./data_example/real_boolean/real_network_no_cat.txt"
header = "file_name,n_node,ratio_sym,\n"

def threshold( vec ):
    for i in range(len(vec)):
        if vec[i]>=1:
            vec[i] = 1
        else:
            vec[i] = 0
    return vec

def opposite(vec1,vec2):
    
    for i in range(len(vec1)):
        if vec1[i]==vec2[i]:
            return False

    return True

def getVar(mat,n):
    
    row_name = range(n)
    ls_row_compare = row_name = range(n)
    to_compare = itertools.product(ls_row_compare,repeat=2)
    
    max_val = 0
    
    for index in  to_compare:
        temp_sum = 0
        for i in range(n):
            temp_sum += 0.5*abs(mat[index[0],i]- mat[index[1],i])
        if temp_sum>max_val:
            max_val=temp_sum
    
    return max_val

class BooleanIter:
    
    def __init__(self, n, max_sample):
        self.n = n
        self.max_sample = max_sample
        self.sample = 0
  
    def __iter__(self):
        return self

    def __next__(self):
        rate = random.random()
        
        if self.sample != self.max_sample:
            ls = []
            for i in range(self.n):
                if random.random()< rate:
                    ls.append(0)
                else:
                    ls.append(1)
            self.sample+=1
            return ls
        else:
            raise StopIteration
      
def load_repository():

    file = create_text_file(path_out, header, delete=True)

    model_list_GinSim = []

    mainpath = "./GINSIM/cell_cycle/"
    model_list_GinSim_cell_cycle = ["Asymmetric_Cell_Division_in_Caulobacter_Crescentus_a.bnet",
                        "Asymmetric_Cell_Division_in_Caulobacter_Crescentus_b.bnet",
                         "boolean_cell_cycle.bnet",
                         "buddingYeastIrons2009_multi.bnet",
                         "buddingYeastOrlando2008_multi.bnet",
                         "drosophilaCellCycleVariants.bnet",
                         "ErbB2_model.bnet",
                         "fissionYeastDavidich2008.bnet",
                         "fissionYeastDavidich2008Modified_multi.bnet",
                         "MCP_budding_yeast_CC_multi.bnet",
                         "Traynard_Boolean_MamCC_Apr2016.bnet"
                        ]
    model_list_GinSim.extend([mainpath + s for s in model_list_GinSim_cell_cycle])
    #excluded too long : "core_engine_budding_yeast_CC_multi.bnet", "coupled_budding_yeast_CC_multi.bnet",
    #excluded duplicate :"Budding_yeast_exit_module.bnet",

    mainpath = "./GINSIM/cell_fate/"
    model_list_GinSim_cell_fate = ["Calzone__Cell_Fate.bnet", "CellFate_multiscale.bnet", "GINsim_HSPC_MSC_0.bnet", "phageLambda4_multi.bnet"]
    model_list_GinSim.extend([mainpath + s for s in model_list_GinSim_cell_fate])

    mainpath = "./GINSIM/development/"
    model_list_GinSim_development = ["ap_boundary.bnet",
                         "ap-1_else-0_wt_multi.bnet",
                         "Cacace_Tdev_2nov2019_multi.bnet",
                         "DrosoMesoLogModel_multi.bnet",
                         "full_network_multi.bnet",
                         "mechanistic_cellular_multi.bnet",
                         "p53Mdm2_tutorial_5march2018_multi.bnet",
                         "pairRule_multi.bnet",
                         "phenomenological_cellular_multi.bnet",
                         "primary_sex_determination_1_multi.bnet",
                         "primary_sex_determination_1_multi.bnet",
                         "reduced_network_0_multi.bnet",
                         "SeaUrchin_model_ginsim_revised_0_multi.bnet",
                         "SP_1cell_multi.bnet",
                         "SP_6cells.bnet",
                         "zebra_miR9_22jul2011.bnet"
                         ]
    model_list_GinSim.extend([mainpath + s for s in model_list_GinSim_development])
    #excluded  because only 4 multi node: "gapA_multi.bnet","gapB_multi.bnet","gapC_multi.bnet","gapD_multi.bnet",

    mainpath = "./GINSIM/differentiation/"
    model_list_GinSim_differentiation = ["Collombet_model_Bcell_Macrophages_PNAS_170215_multi.bnet",
                         "Frontiers-Th-Full-model-annotated_multi.bnet",
                         "TCRsig40.bnet",
                         "Th_17_multi.bnet",
                         "Th_differentiation_reduced_model_multi.bnet",
                         "ThIL17diff_29nov2020_multi.bnet"
                         ]
    model_list_GinSim.extend([mainpath + s for s in model_list_GinSim_differentiation])

    mainpath = "./GINSIM/signaling/"
    model_list_GinSim_signaling = ["Dpp__Pathway_11Jun2013_multi.bnet",
                         "EGF__Pathway_12Jun2013_0_multi.bnet",
                         "FGF_Pathway_12Jun2013.bnet",
                         "Hh__Pathway_11Jun2013_0.bnet",
                         "JakStat__Pathway_12Jun2013.bnet",
                         "MAPK_large_19june2013_multi.bnet",
                         "MAPK_red1_19062013.bnet",
                         "Mast_cell_activation_Annotated_19oct2014_multi.bnet",
                         "Notch__Pathway_12Jun2013_multi.bnet",
                         "Spz__Processing_12Jun2013_multi.bnet",
                         "Toll_Pathway_12Jun2013.bnet",
                         "Trp_reg_multi.bnet",
                         "VEGF_Pathway_12Jun2013_0.bnet",
                         "Wg_Pathway_11Jun2013.bnet"
                         ]    
    model_list_GinSim.extend([mainpath + s for s in model_list_GinSim_signaling])

    mainpath = "./GINSIM/T_cell_activation/"
    model_list_GinSim_T_cell_activation = ["Hernandez_TcellCheckPoints_13april2020_multi.bnet",
                         "ImmuneCheckpointInhibitors.bnet",
                         "RodriguezJorge_Merged_TCR_TLR5_Signalling_BooleanModel_15Jul2018.bnet",
                         "RodriguezJorge_TCR_Signalling_BooleanModel_17Jul2018.bnet",
                         "RodriguezJorge_TLR5_Signalling_BooleanModel_17Jul2018.bnet",
                         "TCR-REDOX-METABOLISM_2019-07-26_multi.bnet",
                         ]    
    model_list_GinSim.extend([mainpath + s for s in model_list_GinSim_T_cell_activation])
    
    model_list_repo = ["irons_yeast", "dinwoodie_life", "saadatpour_guardcell","tournier_apoptosis", "arellano_rootstem","faure_cellcycle","davidich_yeast","randomnet_n7k3","xiao_wnt5a","raf"]
    model_list_repo.extend(["dahlhaus_neuroplastoma","remy_tumorigenesis","klamt_tcr","grieco_mapk" ,"jaoude_thdiff"])
    
    model_list = []
    
    for model in model_list_repo:
        model_list.append(["repo",model])
    for model in model_list_GinSim:
        model_list.append(["file",model])
    
    for source, model in model_list:

        bin_set = {0,1}

        if source == "repo":
            primes = PyBoolNet.Repository.get_primes(model)
        else:
            primes = FileExchange.bnet2primes(model)
            
        n =0
        ls_elem =  []

        for node in primes:
            ls_elem.append(node)
            n+=1
        
        #if model is small enough we can test all transition 
        if n<18:
            iterator = itertools.product(bin_set,repeat = n)
        else:
            iterator = BooleanIter(n, 2**18)
        ntot = 0
        invert_tot = 0
        
        for vec_lst in iterator:
            vector_init, neg_vector_inti = "", ""
            for v in vec_lst:
                vector_init+=str(v)
            vector_next,neg_vector_next = [], []

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

        print_list_file([model,n,invert_tot/ntot], file)
        file.flush()

    file.close()
           
if __name__ == "__main__":
    load_repository()
        