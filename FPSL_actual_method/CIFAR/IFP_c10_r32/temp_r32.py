import numpy as np
import torch
import API_multi_6622 as api
from variable_list import V
import matplotlib.pyplot as plt

# net = api.Models(model=V.model_str, num_layers=V.n_l, num_transition_shape=1 * 1, num_linear_units=512, num_class=V.n_c).net()
net = api.Models(model=V.model_str, num_layers=V.n_l, num_class=V.n_c).net()
net.restore_checkpoint(V.restore_checkpoint_path)
print(net)
print(net.evaluate(V.dataset))
# features = net.get_features(dataset, 10, 0, return_type='mean', verbose=True)
# print(len(features))

p_list,n_list,mpn_list,norm_p_list,norm_n_list,norm_mpn_list = ([] for i in range(6))
for l in range(net.max_layers()-V.ig_l):
    print("layer number: ",l)
    weight_l, bias_l = net.get_weights(l)
    weight_l_next, bias_l_next = net.get_weights(l, next_conv=True)
    print("present_layer weight shape:", weight_l.shape)
    print("next_layer weight shape:",weight_l_next.shape)
    # present layer filter norm criteria (p_n)
    # if l == (net.max_layers()-(V.ig_l+1)):
    #     p_n = np.sum(np.abs(weight_l), axis=(1, 2, 3))
    #     # next layer filter norm criteria (n_n)
    #     n_n = np.sum(np.abs(weight_l_next), axis=0)
    #     normal_p_n = np.mean(np.abs(weight_l), axis=(1, 2, 3))
    #     normal_n_n = np.mean(np.abs(weight_l_next), axis=0)
    # else:
    #     p_n = np.sum(np.abs(weight_l), axis=(1, 2, 3))
    #     # next layer filter norm criteria (n_n)
    #     n_n = np.sum(np.abs(weight_l_next), axis=(0, 2, 3))
    #     normal_p_n = np.mean(np.abs(weight_l), axis=(1, 2, 3))
    #     normal_n_n = np.mean(np.abs(weight_l_next), axis=(0, 2, 3))
    #
    # #multiplication of present and next layer norm
    # mpn_n = np.multiply(p_n,n_n)
    # normal_mpn_n = np.multiply(normal_p_n, normal_n_n)
    # # print("present layer normalized filter importance shape: ", normal_p_n)
    # # print("next layer normalized filter importance shape: ", normal_n_n)
    # # print("multiplication of present & next layer normalized filter importance shape: ", normal_mpn_n)
    #
    # p_list.append(p_n)
    # n_list.append(n_n)
    # mpn_list.append(mpn_n)
    # norm_p_list.append(normal_p_n)
    # norm_n_list.append(normal_n_n)
    # norm_mpn_list.append(normal_mpn_n)

exit()
total_layers = net.max_layers()-1
plt.clf()


# fig, axs = plt.subplots(nrows=2, ncols=3)
fig, ax = plt.subplots(nrows=1, ncols=1)

for i in range(len(p_list)):
    # axs[0,0].scatter([i]*len(p_list[i]), p_list[i])
    # axs[0,1].scatter([i] * len(p_list[i]), n_list[i])
    # axs[0,2].scatter([i] * len(p_list[i]), mpn_list[i])
    # axs[1, 0].scatter([i] * len(p_list[i]), norm_p_list[i])
    # axs[1, 1].scatter([i] * len(p_list[i]), norm_n_list[i])
    # axs[1, 2].scatter([i] * len(p_list[i]), norm_mpn_list[i])
    ax.scatter([i] * len(p_list[i]), norm_mpn_list[i])

# plt.scatter(range(len(p_list[0])), p_list[0])
# axs[0,0].scatter(total_layers,p_list, color='red')
sorted_norm_mpn = np.sort(np.hstack(norm_mpn_list))
print("length of sorted array: ",len(sorted_norm_mpn))
th1_p = sorted_norm_mpn[np.int(0.6*len(sorted_norm_mpn))]
th2_p = sorted_norm_mpn[np.int(0.8*len(sorted_norm_mpn))]
# axs[0,0].set(xlabel='layer index', ylabel='p_n')
# # axs[0,1].scatter(total_layers,n_list, color='red')
# axs[0,1].set(xlabel='layer index', ylabel='n_n')
# # axs[0,2].scatter(total_layers,mpn_list, color='red')
# axs[0,2].set(xlabel='layer index', ylabel='mpn_n')
# # axs[1,0].scatter(total_layers,norm_p_list, color='blue')
# axs[1,0].set(xlabel='layer index', ylabel='normalized_p_n')
# # axs[1,1].scatter(total_layers,norm_n_list, color='blue')
# axs[1,1].set(xlabel='layer index', ylabel='normalized_n_n')
# # axs[1,2].scatter(total_layers,norm_mpn_list, color='blue')
# axs[1,2].axhline(y=th1_p, color='g', linestyle='-')
# axs[1,2].axhline(y=th2_p, color='r', linestyle='-')
# axs[1,2].set(xlabel='layer index', ylabel='normalized_mpn_n')
ax.axhline(y=th1_p, color='g', linewidth= 0.5, linestyle='-')
ax.axhline(y=th2_p, color='r', linewidth= 0.5, linestyle='-')
ax.set(xlabel='layer index', ylabel='normalized_mpn_n')
fig.suptitle(V.dataset_string+'_'+V.model_str+str(V.n_l)+'_filter_importance_all_details')
# fig.savefig(V.dataset_string+'_'+V.model_str+str(V.n_l)+'_filter_importance_metrics.pdf')
plt.close(fig)

plt.close()