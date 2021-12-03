library(BoolNet)
library(rapportools)

id_list <- c(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19)
n_list <- c(7,9,11,13,15)
k_list <- c(2,3)
topology_list <- c("fixed", "homogeneous")
linkage_list <- c("uniform")
main_path = "./GINSIM/kauffman_random/"


for (n in n_list) {
  for (k in k_list) {
    for (topology in topology_list) {
      for (linkage  in linkage_list) {
        for (id in id_list){
          network <- FALSE
          try(network <- generateRandomNKNetwork(n, k, topology, linkage), silent = TRUE)
          if (is.boolean(network)==FALSE){
            file_path = paste(main_path,"randomnet","_",n,"_",k,"_",topology,"_",linkage,"_",id,"_.bnet",sep = "")
            saveNetwork(network, file_path)
          }
        }
      }    
    }    
  }
}
print(count)