import csv
import API_multi as api
from flops_counter import count_flops
from variable_list import V

e = 100
reduction_in_flops = [64] #[64,69]
### pruning percentage per epoch (ppe) (%)
ppe = 2.5 #2
store_path = V.base_path_results + '/reduction_in_flops_' + str(
    reduction_in_flops[0]) + '_percent_and_pruning_percentage_per_epoch_' + str(ppe)

path = store_path+ '/training_epochs/epoch_'+str(e)

net = api.Models(model=V.model_str, num_layers=V.n_l, num_class=V.n_c, pretrained=False, version='B').net()
net.save_pruned_state(store_path+ '/state0')
macs, params = count_flops(store_path+'/state0', 'flops_counter')
macs = float(macs)
params = float(params)
print(macs, params)
net.restore_pruned_state(path + '/model after pruning/epoch_'
                             + str(e))
net.save_pruned_state(store_path+'/state1')
macs_pruned, params_pruned = count_flops(store_path+'/state1', 'flops_counter')
macs_pruned = float(macs_pruned)
params_pruned = float(params_pruned)
print(macs_pruned, params_pruned)
per_red_param = ((params - params_pruned) * 100 / params)
per_red_FLOPs = ((macs - macs_pruned) * 100 / macs)
print("FLOPs reduction percentage: ", per_red_FLOPs)
print("params reduction percentage: ", per_red_param)

row1 = ['base_FLOPs(M)', 'curr_FLOPs(M)', 'rdcn_FLOPs(%)', 'base_params(M)', 'curr_params(M)', 'rdcn_params(%)']
row2 = [round((macs / 1e6), 4), round((macs_pruned / 1e6), 4), round(per_red_FLOPs, 4),
        round((params / 1e6), 4), round((params_pruned / 1e6), 4), round(per_red_param, 4)]

with open(store_path + '/flops_and_params_of_pruned_model.csv', 'wt') as results_file:
    csv_writer = csv.writer(results_file)
    csv_writer.writerow(row1)
    csv_writer.writerow(row2)


# def main():
#     net = api.Models('ResNet', 50, version='B').net()
#
#     net.save_pruned_state('state0')
#     macs, params = count_flops('state0', 'flops_counter')
#     print(macs, params)
#
#     # Prune some filters
#     for i in range(net.max_layers()):
#         for j in range(20):
#             net.prune(i, j, False)
#
#     net.save_pruned_state('state1')
#     macs, params = count_flops('state1', 'flops_counter')
#     print(macs, params)
#
#
# def flops_parameters_count(net,path,epoch,image_dim):
#     macs, params = get_model_complexity_info(net, image_dim, as_strings=False, print_per_layer_stat=False,
#                                              verbose=False)
#
#     net.restore_pruned_state(path + '/model after pruning/epoch_'
#                              + str(epoch), arch_only=True)
#
#     macs_pruned, params_pruned = get_model_complexity_info(net, image_dim, as_strings=False,
#                                                            print_per_layer_stat=False,
#                                                            verbose=False)
#
#     return macs/1e6, macs_pruned/1e6, params/1e6, params_pruned/1e6
#
# if __name__ == '__main__':
#     main()
