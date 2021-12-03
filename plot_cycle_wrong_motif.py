import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from boolean.plot_util import get_average_x_y, colorVec, markerVec, get_binned_dict

path_file = "./data_example/1d_properties/1d_properties_nodes_7.txt"
path_wrong_motif = "./data_example/wrong_motif_network/wrong_motif_nodes_7.txt"
path_fig = "./fig/cycle_wrong_motif.pdf"

df = pd.read_csv(path_file)
df_wrong_motif = pd.read_csv(path_wrong_motif)

X_wrong,Y_wrong, STD_wrong = get_average_x_y(df_wrong_motif, 'density', 'max_cycle', SEM=True, min_count = 3,  cond_col = 'density_from_motif', min_col = 0.75)
dict_count_wrong, dict_y_wrong = get_binned_dict(df_wrong_motif, 'density', 'max_cycle' , cond_col = 'density_from_motif', min_col = 0.75)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
fig.subplots_adjust(wspace = 0.3 )

ax = axs[0]
X,Y, STD = get_average_x_y(df, 'density', 'max_cycle', SEM=True, min_count = 3)
dict_count, dict_y = get_binned_dict(df, 'density', 'max_cycle' )
imsh = ax.errorbar(X,Y, yerr=STD, marker=markerVec[0], color=colorVec[0] , label='random network', ls='None')
ax.errorbar(X_wrong,Y_wrong, yerr=STD_wrong, marker=markerVec[1], color=colorVec[1], label='wrong motifs', ls='None')
ax.set_title(str('max. cycle (SEM)'))
ax.set( xlabel='density', ylabel='max. cycle (average)')
ax.legend(loc=0)


ax = axs[1]
pvalue = [0]*len(X_wrong)
#compute p_value
for k in range(len(pvalue)):
    stat,pval = stats.ttest_ind( dict_y[X_wrong[k]] , dict_y_wrong[X_wrong[k]], equal_var=False)
    pvalue[k] = pval
imsh3 = ax.plot(X_wrong, pvalue, marker='o', color=colorVec[0], ls='None', label = 'random network')
ax.legend()
ax.set_title(str('max. cycle (Statistic) '))
ax.set( xlabel='density', ylabel='p-value (random vs. wrong motif)')


plt.savefig(path_fig, transparent=True)