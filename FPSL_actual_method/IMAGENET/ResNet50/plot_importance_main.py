import os
import matplotlib.pyplot as plt
import numpy as np
from variable_list import V

def display_imp(r,s,p, path_epochwise, epoch, store_path, curr_reduction):
    norm_mpn_list = []
    ### ResNet50 maximum prunable layers, net.max_layers() =32
    for l in range(32 - V.ig_l):
        print("layer number: ",l)
        normal_mpn_n = np.loadtxt(path_epochwise + '/imp_val/layer_' + str(l) + '.csv')
        print("multiplication of present & next layer normalized filter importance shape: ", normal_mpn_n.shape)
        norm_mpn_list.append(normal_mpn_n)

    # plt.clf()
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # fig.tight_layout()
    # r = np.int64(epoch/6)
    # s = np.int64((epoch - r*6)/3)
    for i in range(len(norm_mpn_list)):
        axs[r,s].scatter([i] * len(norm_mpn_list[i]), norm_mpn_list[i])


    sorted_norm_mpn = np.sort(np.hstack(norm_mpn_list))

    th = sorted_norm_mpn[np.int64(p*len(sorted_norm_mpn))]

    # axs[r,s].set(xlabel='Layer index', ylabel='Importance')
    axs[r,s].axhline(y=th, color='g', linestyle='-')
    axs[r,s].set_title('Epoch'+str(epoch)+'_'+str(round(curr_reduction,2))+'% FLOPs(↓)',
                       fontdict={'fontsize': 7, 'fontweight': 'medium'})
    axs[r,s].set_xticks(np.arange(1, 33, 4))
    axs[r,s].tick_params(axis="x", labelsize=6)
    # axs[r,s].xticks(np.arange(0, max(norm_mpn_list) + 1, 1))
    # axs[r,s].yticks(np.arange(0, 33, 1))

    # fig.suptitle('Epoch_'+str(epoch)+'_current reduction in FLOPs_'+str(round(curr_reduction,2))+'%')
    # # fig.savefig(store_path+'/'+ V.dataset_string+'_'+V.model_str+str(V.n_l)+'_epoch_'+str(epoch))
    #
    # plt.close(fig)
    #
    # plt.close()


### user defined flops reduction target(%) => retained FLOPs = (100- flops reduction)
reduction_in_flops = [62] #[63.5]
### pruning percentage per epoch (ppe) (%)
ppe = 2.5 #2


prune_results = V.base_path_results + '/reduction_in_flops_' + str(
    reduction_in_flops[0]) + '_percent_and_pruning_percentage_per_epoch_' + str(ppe)

os.makedirs(prune_results + '/final_imp_scores_plot', exist_ok=True)
per_red_FLOPs = np.loadtxt(prune_results + '/Training_Results_original_defined.csv', delimiter=',', usecols=4,
                           skiprows=1)
num_row = 3
num_column = 3

fig, axs = plt.subplots(nrows=num_row, ncols=num_column, dpi=400)
for ax in axs.flat:
    ax.set(xlabel='Layer index', ylabel='Importance')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
r = 0
s = 0
entry_no = 0
for e in range(1,26,3):
    path_epochwise = prune_results + '/training_epochs/epoch_' + str(e)
    r = np.int64(entry_no/num_column)
    s = entry_no - r*num_column
    entry_no = entry_no + 1
    display_imp(r,s, ppe/100, path_epochwise, e, prune_results + '/final_imp_scores_plot', round(np.float64(per_red_FLOPs[e]), 4))

fig.savefig(prune_results + '/final_imp_scores_plot'+'/'+ V.dataset_string+'_'+V.model_str+str(V.n_l))