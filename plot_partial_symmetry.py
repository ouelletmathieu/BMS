from boolean.plot_util import markerVec, colorVec
import matplotlib.pyplot as plt



def plot_density_sym_ratio():
    
    
    
    
    path_7node_wrong_motif = "/Users/mathieuouellet/Dropbox/Boolean_network/DATA/symmetries/7nodes_wrong_motif.txt"
    path_7node_randome = "/Users/mathieuouellet/Dropbox/Boolean_network/DATA/symmetries/7nodes_random.txt"

    
    array_ratio, array_nb = script_get_pareto_symmetries()
    
    mid_point, average_sym_ratio, average_sym_ratio_transcient , average_sym_ratio_cycle, dic = getAverage_sym_ratio(path_7node_wrong_motif,0.7)
    mid_point_rand, average_sym_ratio_rand, average_sym_ratio_transcient_rand, average_sym_ratio_cycle_rand, dic = getAverage_sym_ratio(path_7node_randome, -1)
    mid_point_pareto, average_sym_ratio_pareto, average_sym_ratio_transcient_pareto, average_sym_ratio_cycle_pareto, average_dic_list = array_ratio

    
    mid_point, average_sym_nb, average_sym_nb_transcient , average_sym_nb_cycle, dic = getAverage_sym_nb(path_7node_wrong_motif,0.7)
    mid_point_rand, average_sym_nb_rand, average_sym_nb_transcient_rand, average_sym_nb_cycle_rand, dic = getAverage_sym_nb(path_7node_randome,-1)
    mid_point_pareto, average_sym_nb_pareto, average_sym_nb_transcient_pareto, average_sym_nb_cycle_pareto, average_nb_dic_list = array_nb



    n=2
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))
    
    imsh00_0 = axs[0,0].plot(mid_point[1:],average_sym_ratio[1:],  marker = markerVec[n], label='motif', color = colorVec[n], linestyle="None")
    imsh = axs[0,1].plot(mid_point[1:],average_sym_ratio_cycle[1:],  marker = markerVec[n], label='motif', color = colorVec[n], linestyle="None")
    imsh = axs[0,2].plot(mid_point[1:],average_sym_ratio_transcient[1:],  marker = markerVec[n], label='motif', color = colorVec[n], linestyle="None")

    imsh10_0 = axs[1,0].plot(mid_point[1:],average_sym_nb[1:],  marker = markerVec[n], label='motif', color = colorVec[n], linestyle="None")
    imsh = axs[1,1].plot(mid_point[1:],average_sym_nb_cycle[1:],  marker = markerVec[n], label='motif', color = colorVec[n], linestyle="None")
    imsh = axs[1,2].plot(mid_point[1:],average_sym_nb_transcient[1:],  marker = markerVec[n], label='motif', color = colorVec[n], linestyle="None")



    n=1
    imsh00_1 = axs[0,0].plot(mid_point_rand[1:],average_sym_ratio_rand[1:],  marker = markerVec[n], label='random', color = colorVec[n], linestyle="None")
    imsh = axs[0,1].plot(mid_point_rand[1:],average_sym_ratio_cycle_rand[1:],  marker = markerVec[n], label='random', color = colorVec[n], linestyle="None")
    imsh = axs[0,2].plot(mid_point_rand[1:],average_sym_ratio_transcient_rand[1:],  marker = markerVec[n], label='random', color = colorVec[n], linestyle="None")
    
    imsh10_1 = axs[1,0].plot(mid_point[1:],average_sym_nb_rand[1:],  marker = markerVec[n], label='random', color = colorVec[n], linestyle="None")
    imsh = axs[1,1].plot(mid_point[1:],average_sym_nb_cycle_rand[1:],  marker = markerVec[n], label='random', color = colorVec[n], linestyle="None")
    imsh = axs[1,2].plot(mid_point[1:],average_sym_nb_transcient_rand[1:],  marker = markerVec[n], label='random', color = colorVec[n], linestyle="None")


    
    n=0
    imsh00_2 = axs[0,0].plot(mid_point_rand[1:],average_sym_ratio_pareto[1:],  marker = markerVec[n], label='pareto', color = colorVec[n], linestyle="None")
    imsh = axs[0,1].plot(mid_point_rand[1:],average_sym_ratio_cycle_pareto[1:],  marker = markerVec[n], label='pareto', color = colorVec[n], linestyle="None")
    imsh = axs[0,2].plot(mid_point_rand[1:],average_sym_ratio_transcient_pareto[1:],  marker = markerVec[n], label='pareto', color = colorVec[n], linestyle="None")
    
      
    imsh = axs[1,0].plot(mid_point[1:],average_sym_nb_pareto[1:],  marker = markerVec[n], label='pareto', color = colorVec[n], linestyle="None")
    imsh = axs[1,1].plot(mid_point[1:],average_sym_nb_cycle_pareto[1:],  marker = markerVec[n], label='pareto', color = colorVec[n], linestyle="None")
    imsh = axs[1,2].plot(mid_point[1:],average_sym_nb_transcient_pareto[1:],  marker = markerVec[n], label='pareto', color = colorVec[n], linestyle="None")

    axs[0,0].legend()
    axs[0,1].legend()
    axs[0,2].legend()
    axs[1,0].legend()
    axs[1,1].legend()
    axs[1,2].legend()
    
    plt.show()
    
    n=2
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    

    imsh00_0 = axs[0].plot(mid_point[1:],average_sym_ratio[1:],  marker = markerVec[n], label='motif', color = colorVec[n], linestyle="None")
    #imsh = axs[0,1].plot(mid_point[1:],average_sym_ratio_cycle[1:],  marker = markerVec[n], label='motif', color = colorVec[n], linestyle="None")
    #imsh = axs[0,2].plot(mid_point[1:],average_sym_ratio_transcient[1:],  marker = markerVec[n], label='motif', color = colorVec[n], linestyle="None")

    imsh10_0 = axs[1].plot(mid_point[1:],average_sym_nb[1:],  marker = markerVec[n], label='motif', color = colorVec[n], linestyle="None")
    #imsh = axs[1,1].plot(mid_point[1:],average_sym_nb_cycle[1:],  marker = markerVec[n], label='motif', color = colorVec[n], linestyle="None")
    #imsh = axs[1,2].plot(mid_point[1:],average_sym_nb_transcient[1:],  marker = markerVec[n], label='motif', color = colorVec[n], linestyle="None")



    n=1
    imsh00_1 = axs[0].plot(mid_point_rand[1:],average_sym_ratio_rand[1:],  marker = markerVec[n], label='random', color = colorVec[n], linestyle="None")
    #imsh = axs[0,1].plot(mid_point_rand[1:],average_sym_ratio_cycle_rand[1:],  marker = markerVec[n], label='random', color = colorVec[n], linestyle="None")
    #imsh = axs[0,2].plot(mid_point_rand[1:],average_sym_ratio_transcient_rand[1:],  marker = markerVec[n], label='random', color = colorVec[n], linestyle="None")
    
    imsh10_1 = axs[1].plot(mid_point[1:],average_sym_nb_rand[1:],  marker = markerVec[n], label='random', color = colorVec[n], linestyle="None")
    #imsh = axs[1,1].plot(mid_point[1:],average_sym_nb_cycle_rand[1:],  marker = markerVec[n], label='random', color = colorVec[n], linestyle="None")
    #imsh = axs[1,2].plot(mid_point[1:],average_sym_nb_transcient_rand[1:],  marker = markerVec[n], label='random', color = colorVec[n], linestyle="None")


    
    n=0
    imsh00_2 = axs[0].plot(mid_point_rand[1:],average_sym_ratio_pareto[1:],  marker = markerVec[n], label='pareto', color = colorVec[n], linestyle="None")
    #imsh = axs[0,1].plot(mid_point_rand[1:],average_sym_ratio_cycle_pareto[1:],  marker = markerVec[n], label='pareto', color = colorVec[n], linestyle="None")
    #imsh = axs[0,2].plot(mid_point_rand[1:],average_sym_ratio_transcient_pareto[1:],  marker = markerVec[n], label='pareto', color = colorVec[n], linestyle="None")
    
      
    imsh = axs[1].plot(mid_point[1:],average_sym_nb_pareto[1:],  marker = markerVec[n], label='pareto', color = colorVec[n], linestyle="None")
    #imsh = axs[1,1].plot(mid_point[1:],average_sym_nb_cycle_pareto[1:],  marker = markerVec[n], label='pareto', color = colorVec[n], linestyle="None")
    #imsh = axs[1,2].plot(mid_point[1:],average_sym_nb_transcient_pareto[1:],  marker = markerVec[n], label='pareto', color = colorVec[n], linestyle="None")

    #axs[0].legend()
    axs[1].legend()
    
    
    axs[0].set_xlabel("network density")
    axs[1].set_xlabel("network density")
    
    axs[0].set_ylabel("partial symmetries")
    axs[1].set_ylabel("complete symmetries ") 

    axs[0].set_xlim([0,1])
    axs[1].set_xlim([0,1])
    
    
    plt.savefig("/Users/mathieuouellet/Dropbox/Boolean_network/FIGURES/OUTPUT/partial_symmetries_ratio.pdf", transparent=True)


    
    
plot_density_sym_ratio() 