from variable_list import V
import API_multi as api
import pruning_API_DFFP as prune_api
import numpy as np
import time
import os

p = 0.56
net = api.Models(model=V.model_str, num_layers=V.n_l, num_class= V.n_c).net()
init_filter_count = 0
retained_filter_count = 0
for j in range(0, net.max_layers()):
    init_filter_count += net.max_filters(layer=j)

net.restore_pruned_state(V.base_path_results + '/Pruning_Desired ' + str(p * 100) + '%' + '/retained_model')

for j in range(0, net.max_layers()):
    retained_filter_count += net.max_filters(layer=j)

f_retained_percentage  = (retained_filter_count*100)/init_filter_count
print("pruning fraction: ", p)
print("Total filters initial: ",init_filter_count)
print("Total filters final: ",retained_filter_count)
print("filters retained percentage: ",f_retained_percentage)