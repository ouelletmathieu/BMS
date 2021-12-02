import random
import networkx as nx
import os
from boolean.boolean_util import generateRandomValidNetwork, arrayToList, countNonConjugate3, getNbSame, get_1d_property
from boolean.motif_util import DecisionTree, get_tree_input_3node, get3motif_graph, get2motif_graph
from boolean.genetic_util import randomPoint, evalFitness, mateList2, mutateList, distanceBetweenList, evaluateMotif, get_tau_avg_std, get_1d_property_avg_std
from boolean.adv_descriptor_util import get_tau_descriptor
import networkx as nx
import numpy as np
import random
from deap import base
from deap.benchmarks.tools import igd
from deap import creator
from deap import tools
import os
from multiprocessing import Pool
from datetime import datetime
import csv
import matplotlib.pyplot as plt

#############################################
#                Parameters                 #
#############################################

nNode = 7           #number of nodes 
npop = 600          #population for each generation
offspingSize = 400  #number of offspring generated at each generation 
MUTENTRY = 0.02     #probabiltiy of mutation per entry in the matrix 
maxGen = 300        #max generation for each optimization process 
delta_gen = 50      #number of generation between the data output
maxOptim = 500      #number of run of the full optim process to do
PROCESS = 10        #number of core used for the optimization 
adding_random = 600 #random element added each generation inside the pool (help to keep more diversity) 
file_path = "./data_example/pareto_density_cycle/"      #file path to store the result

#register the functions to use for the optimization process
toolbox = base.Toolbox() #toolbox for optimization from deap 
creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness)


toolbox.register("random", randomPoint)
toolbox.register("attr_float", toolbox.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, nNode**2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalFitness)
toolbox.register("mate", mateList2) 
toolbox.register("mutate", mutateList, prob=MUTENTRY)
toolbox.register("select", tools.selNSGA2)#,ref_points=ref_points)



def main():

    poolGen = Pool(PROCESS)
    toolbox.register("map", poolGen.map)

    #set up the machinery needed for motif analysis 
    list2motif, motif2Info, list2mat, list22info =  get2motif_graph()
    motif_3_repr, all_3_motif_mat, all_3_motif_repr_nb = get_tree_input_3node()
    list3mat = motif_3_repr
    mainTree = DecisionTree.constructTree(all_3_motif_mat, all_3_motif_repr_nb, all_3_motif_repr_nb)
    nwayMotif = []
    list3motif, motif3Info, list3mat, list33info =  get3motif_graph()
    for mat in motif_3_repr:
        nwayMotif.append( len(countNonConjugate3(mat)))

    #create file 
    now = datetime.now()
    dt_string = now.strftime("%d_%H_%M_%S")
    PathExcel = file_path+dt_string+"/"
    os.makedirs(PathExcel, exist_ok=True)




    #loop that run for each job of the otpimization process
    for indexOptim in range(maxOptim):
        #create files to store the run
        optimPath =PathExcel + "run" + str(indexOptim)+"/"
        os.makedirs(optimPath, exist_ok=True)
        optimPathPop = optimPath+"pop/"
        os.makedirs(optimPathPop, exist_ok=True)

        #create files to store the new data
        motif_2node_zvector,motif_3node_zvector  = list(),list()
        tau_vector, tau_vector_std = list(), list()
        vec_1d_list, vec_1d_std_list = list(), list()
        tracking_pop = list()
        pop = toolbox.population(npop)
        
        #create initial population 
        for ind in pop:
            density = (random.random()*0.8)+0.2
            mat = generateRandomValidNetwork(nNode, density)
            arr = arrayToList(mat)
            for i in range(len(arr)):
                ind[i]=arr[i]
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = toolbox.select(pop, len(pop))

        g = 0 #generation counter
        print(indexOptim, " initial pop: " ,  len(pop),"  ,  ", getNbSame(pop))
        
        #main loop for dealing with the generation 
        while g < maxGen:
            g = g + 1

            offspring = tools.selTournamentDCD(pop, offspingSize)
            offspring = [toolbox.clone(ind) for ind in offspring]
            #mating process 
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                toolbox.mate(child1, child2)
                
                toolbox.mutate(child1)
                toolbox.mutate(child2)
                
                del child1.fitness.values
                del child2.fitness.values

            #remove duplicate in offspring
            newList = list()   
            for i in range(len(offspring)):
                same = False
                for j in range(i+1, len(offspring)):
                    if distanceBetweenList(offspring[i], offspring[j])==0:
                        same=True;
                        break
                if not same:
                    newList.append(offspring[i])
            newList2 =  list()
            for i in range(len(newList)):
                same = False
                for ind in pop:
                    if distanceBetweenList(newList[i], ind)==0:
                        same=True;
                        break
                if not same:
                    newList2.append(newList[i])
            offspring = newList2
            #add randomness 
            new_pop = toolbox.population(adding_random)
            for ind in new_pop:
                density = (random.random()*0.8)+0.2
                mat = generateRandomValidNetwork(nNode, density)
                lst = arrayToList(mat)
                for i in range(len(lst)):
                    ind[i]=lst[i]
            #evaluate fitness
            invalid_ind = [ind for ind in new_pop if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            print("after first selection and remove similar again: ",str(g)," , " , len(offspring)," , ", getNbSame(offspring))  

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            print("number of invalid offspring: ", len(invalid_ind))  
            
            # Select the next generation population
            #pop = toolbox.select(pop + offspring + new_pop, npop)
            pop = toolbox.select(pop + offspring + new_pop, npop)

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            # Initialize statistics object
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean, axis=0)
            stats.register("std", np.std, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)
            logbook = tools.Logbook()
            logbook.header = "gen", "evals", "std", "min", "avg", "max"

            #if population statistic need to be stored
            if g==1 or g%delta_gen==0 or g==maxGen:
                printPareto(pop, optimPath+"pop_"+ str(g) +".pdf")

                #keep information about the population 
                motif_list_2_node, motif_list_3_node = evaluateMotif(g, pop, list2motif, motif2Info, list3motif, motif3Info, nwayMotif, mainTree, list2mat, list3mat)
                motif_2node_zvector.append(motif_list_2_node)
                motif_3node_zvector.append(motif_list_3_node)
                vectau, vectaustd = get_tau_avg_std(pop)
                tau_vector.append(vectau)
                tau_vector_std.append(vectaustd)
                vec_1d, vec_1d_std = get_1d_property_avg_std(pop)
                vec_1d_list.append(vec_1d)
                vec_1d_std_list.append(vec_1d_std)

                temp_list_pop = list()
                row_name = []
                row_name.append("g")
                row_name.append("max_cycle")
                row_name.append("density")            
                row_name.extend(["mxc_inhib", "avgc_inhib", "mxc_exct", "avgc_exct", "mxc_all", "avgc_all", "totalInhibition", "totalExcitation", "totalAutoInhibition", "totalAutoExcitation"])
                row_name.extend(["node_pos_dumb","node_neg_dumb","node_all_dumb","node_pos_good", "node_neg_good","node_all_good"])                
                row_name.append("mat")
                
                for ind in pop:
                    temp_list_ind = list()
                    temp_list_ind.append(g)
                    temp_list_ind.extend(ind.fitness.values)
                    vec_1d = get_1d_property(ind)
                    temp_list_ind.extend(vec_1d) 
                    vec_tau = get_tau_descriptor(ind)
                    temp_list_ind.extend(vec_tau)
                    temp_list_ind.append(ind)  
                    temp_list_pop.append(temp_list_ind)

                tracking_pop.append(temp_list_pop)
                pop_path = optimPathPop+str(g)+'_pop.csv'

                with open(pop_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row_name)
                    for pop_vec_list in temp_list_pop:
                        writer.writerow(pop_vec_list)

            print(indexOptim, " after first selection pop: ",str(g)," , " , len(pop)," , ", getNbSame(pop))
            # Compile statistics about the population
            record = stats.compile(pop)
            logbook.record(gen=g, evals=len(invalid_ind), **record)
            print(logbook.stream)

        #############################################
        #                 Printing                  #
        #############################################

        #print motif
        motif_3_path = optimPath+'3motifs.csv'
        motif_3_path_std = optimPath+'3motifs_std.csv'
        motif_2_path = optimPath+'2motifs.csv'
        motif_2_path_std = optimPath+'2motifs_std.csv'
        motif_tau = optimPath+'tau.csv'
        motif_tau_std = optimPath+'tau_std.csv'
        graph_1d =  optimPath+'1d.csv'
        graph_1d_std = optimPath+'1d_std.csv'


        with open(motif_3_path, 'w', newline='') as file:
            writer = csv.writer(file)
            list_motif_name = list()
            list_motif_name.append("gen")
            for i in range(len(list3mat)):
                list_motif_name.append("motif_"+str(i))
            writer.writerow(list_motif_name)
            for motif_list in motif_3node_zvector:
                writer.writerow(motif_list[0])

        with open(motif_3_path_std, 'w', newline='') as file:
            writer = csv.writer(file)
            list_motif_name = list()
            list_motif_name.append("gen")
            for i in range(len(list3mat)):
                list_motif_name.append("motif_"+str(i)+"_std")
            writer.writerow(list_motif_name)
            for motif_list in motif_3node_zvector:
                writer.writerow(motif_list[1])

        with open(motif_2_path, 'w', newline='') as file:
            writer = csv.writer(file)
            list_motif_name = list()
            list_motif_name.append("gen")
            for i in range(len(list2motif)):
                list_motif_name.append("motif_"+str(i))
            writer.writerow(list_motif_name)
            for motif_list in motif_2node_zvector:
                writer.writerow(motif_list[0])

        with open(motif_2_path_std, 'w', newline='') as file:
            writer = csv.writer(file)
            list_motif_name = list()
            list_motif_name.append("gen")
            for i in range(len(list2motif)):
                list_motif_name.append("motif_"+str(i)+"_std")
            writer.writerow(list_motif_name)
            for motif_list in motif_2node_zvector:
                writer.writerow(motif_list[1])

        with open(motif_tau, 'w', newline='') as file:
            writer = csv.writer(file)        
            writer.writerow(["node_pos_dumb","node_neg_dumb","node_all_dumb",
                            "node_pos_good","node_neg_good","node_all_good"])
            for vec_list in tau_vector:
                writer.writerow(vec_list)

        with open(motif_tau_std, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["node_pos_dumb_std","node_neg_dumb_std","node_all_dumb_std",
                            "node_pos_good_std","node_neg_good_std","node_all_good_std"])
            for vec_list in tau_vector_std:
                writer.writerow(vec_list)

        with open(graph_1d, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["mxc_inhib","avgc_inhib","mxc_exct",
                            "avgc_exct","mxc_all","avgc_all",
                            "totalInhibition","totalExcitation","totalAutoInhibition","totalAutoExcitation"])
            for vec_list in vec_1d_list:
                writer.writerow(vec_list)

        with open(graph_1d_std, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["mxc_inhib_std","avgc_inhib_std","mxc_exct_std",
                            "avgc_exct_std","mxc_all_std","avgc_all_std",
                            "totalInhibition_std","totalExcitation_std","totalAutoInhibition_std","totalAutoExcitation_std"])
            for vec_list in vec_1d_std_list:
                writer.writerow(vec_list)

def printPareto(pop, path):
    """Fast plot of the population pop
    """
    p = np.array([ind.fitness.values for ind in pop])
    plt.scatter(p[:, 1], p[:, 0], marker="o", s=24, label="Final Population")
    plt.savefig(path, transparent=True)


if __name__ == "__main__":
    main()


