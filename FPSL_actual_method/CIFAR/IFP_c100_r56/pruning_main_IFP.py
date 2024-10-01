from variable_list import V
import API_multi as api
import pruning_API_IFP as prune_api
import numpy as np
import time
import os
import csv

### starting epoch for prune and finetune(resume capability)
# Starting Epoch in fine tune (s_epoch= 0 by default/ last_checkpoint saved)
s_epoch = 0
### user defined flops reduction target(%) => retained FLOPs = (100- flops reduction)
reduction_in_flops = [52.6]
### pruning percentage per epoch (ppe) (%)
ppe = 1
### target FLOPs flag (1 means only finetune no prune as target flops already reached)
tflops_flag = 0
## Total finetune epochs & initial learning rate
f_epochs = 300
lr_init = 0.1


# n_batch = np.int64(np.ceil(V.dataset.num_train_images/V.b_size))

# ## Only network is assigned in main function, rest all are assigned in variable list
# net = api.Models(model=V.model_str, num_layers=V.n_l, num_class= V.n_c).net()
# print(net)
# net.restore_checkpoint(V.restore_checkpoint_path)
# print(net.evaluate(V.dataset))
### retrain & finetune time for each pruning fraction
def init_array(dim):
    arr = np.zeros(np.int64(dim))
    return arr

def to_array(val):
    return np.float64([val])

for r in range(0,len(reduction_in_flops)):
    prune_results = V.base_path_results + '/reduction_in_flops_'+ str(reduction_in_flops[r])+'_percent_and_pruning_percentage_per_epoch_'+str(ppe)

    if s_epoch == 0:
        os.makedirs(V.base_path_results, exist_ok=True)
        os.makedirs(prune_results, exist_ok=True)
        os.makedirs(prune_results + '/imp_score_visualization', exist_ok=True)
        os.makedirs(prune_results + '/final_result', exist_ok=True)
        os.makedirs(prune_results + '/training_epochs', exist_ok=True)
        os.makedirs(prune_results + '/final_pruned_model', exist_ok=True)
        os.makedirs(prune_results + '/finetuned_model', exist_ok=True)
        # os.makedirs(prune_results + '/best_epoch_within_target', exist_ok=True)
        os.makedirs(prune_results + '/training_last_epoch', exist_ok=True)
        ##compute importance for filters

    # prune_api.filter_norm_present_next_layer(prune_results+ '/training_epochs/epoch_'+str(s_epoch+1),net)
    # prune_api.find_global_threshold(prune_results+ '/training_epochs/epoch_'+str(s_epoch+1),ppe)
    prune_api.iterative_prune_and_finetune(ppe/100, reduction_in_flops[r], s_epoch, f_epochs,lr_init, tflops_flag, prune_results)

#         count = (p_end - p_init)/p_gap
#         ###retrain and finetune time initialization
#         curr_p, r_time_p, t_time_p, th_p, \
#         acc_bft_p_top_one, acc_bft_p_top_five, \
#         acc_aft_p_top_one, acc_aft_p_top_five,\
#         acc_o_top_one, acc_o_top_five= [init_array(count) for i in range(10)]
#         np.savetxt(V.base_path_results + '/final_result/retrain_and_finetune_time_details.txt',np.c_[curr_p, r_time_p, t_time_p],delimiter=",", header='pf, retrain_time, retrain_and_finetune_time',fmt='%f')
#         np.savetxt(V.base_path_results + '/final_result/accuracy_details.txt',np.c_[curr_p, acc_o_top_one,acc_o_top_five, acc_bft_p_top_one,acc_bft_p_top_five, acc_aft_p_top_one, acc_aft_p_top_five],
#                    delimiter=",", header='pf, base_t1, base_t5, pr_retrain_t1, pr_retrain_t5, finetune_t1, finetune_t5',fmt='%f')
#         np.savetxt(V.base_path_results + '/final_result/threshold_details.txt', np.c_[curr_p, th_p], delimiter=",", header='pf, threshold', fmt='%f')
#     else:
#         pf_count = np.int64((p_per - p_init) / p_gap)
#         curr_p, r_time_p, t_time_p = np.loadtxt(V.base_path_results + '/final_result/retrain_and_finetune_time_details.txt',skiprows=1, delimiter=",", unpack= True)
#         _, th_p = np.loadtxt(V.base_path_results + '/final_result/threshold_details.txt',skiprows=1, delimiter=",", unpack= True)
#         _, acc_o_top_one,acc_o_top_five, acc_bft_p_top_one,acc_bft_p_top_five, \
#         acc_aft_p_top_one, acc_aft_p_top_five= np.loadtxt(V.base_path_results + '/final_result/accuracy_details.txt',skiprows=1, delimiter=",", unpack= True)
#         ##desired pruning fraction (of filters)
#         p = p_per/100
#         th = prune_api.find_global_threshold(p)
#         start_time = time.time()
#         ## Prune_Retrain & Finetune creteas net inside function in API
#         layer_lr = layer_lr_init * (pf_count + 1)
#         te_acc_org, te_acc_prune_retrain = prune_api.prune_retrain_block(p,sl,r_flag,r_count,layer_lr,V.image_dim,V.upl)
#         r_end_time = time.time()
#         # Starting Epoch (s = 0 by default, otherwise multiple of lr_change_freq)
#         if r_flag == 1:
#             f_lr[0] = f_lr_init[0] * (pf_count + 1)
#             lr_divide_factor = lr_divide_factor_init * (1 + ((2 * (pf_count + 1)) / 100))
#             te_acc_pr_fine = prune_api.finetune_retained_model(p,s_e_f,f_epochs,f_lr,lr_change_freq,lr_divide_factor)
#         else:
#             if fine_method == 1:
#                 te_acc_pr_fine = prune_api.only_finetune_no_retrain(p,s_e_f,f_epochs,f_lr_init[0],lr_c_f,lr_c_s,lr_d_f,s_cp)
#             else:
#                 te_acc_pr_fine = prune_api.only_finetune_no_retrain_method_two(p, s_e_f, f_epochs, f_lr_init[0], s_cp)
#
#         end_time = time.time()
#         print("current p",curr_p)
#         # exit()
#         retrain_time = r_end_time-start_time
#         total_time = end_time-start_time
#         if np.isscalar(curr_p) == True:
#             curr_p = round(p, 4)
#             r_time_p = round(retrain_time, 4)
#             t_time_p = round(total_time, 4)
#             acc_o_top_one = round(te_acc_org[0], 4)
#             acc_o_top_five = round(te_acc_org[1], 4)
#             acc_bft_p_top_one = round(te_acc_prune_retrain[0], 4)
#             acc_bft_p_top_five = round(te_acc_prune_retrain[1], 4)
#             acc_aft_p_top_one = round(te_acc_pr_fine[0], 4)
#             acc_aft_p_top_five = round(te_acc_pr_fine[1], 4)
#             th_p = th
#         else:
#             curr_p[pf_count] = round(p,4)
#             r_time_p[pf_count] = round(retrain_time,4)
#             t_time_p[pf_count] = round(total_time,4)
#             acc_o_top_one[pf_count] = round(te_acc_org[0], 4)
#             acc_o_top_five[pf_count] = round(te_acc_org[1], 4)
#             acc_bft_p_top_one[pf_count] = round(te_acc_prune_retrain[0], 4)
#             acc_bft_p_top_five[pf_count] = round(te_acc_prune_retrain[1], 4)
#             acc_aft_p_top_one[pf_count] = round(te_acc_pr_fine[0], 4)
#             acc_aft_p_top_five[pf_count] = round(te_acc_pr_fine[1], 4)
#             th_p[pf_count] = th
#         # print("current p",curr_p)
#         # print("retrain time p", r_time_p)
#         np.savetxt(V.base_path_results + '/final_result/retrain_and_finetune_time_details.txt', np.c_[curr_p, r_time_p, t_time_p], delimiter=",",
#                    header='pf, retrain_time, retrain_and_finetune_time', fmt='%f')
#         np.savetxt(V.base_path_results + '/final_result/threshold_details.txt', np.c_[curr_p, th_p], delimiter=",",
#                    header='pf, threshold', fmt='%f')
#         np.savetxt(V.base_path_results + '/final_result/accuracy_details.txt', np.c_[curr_p, acc_o_top_one,acc_o_top_five, acc_bft_p_top_one,acc_bft_p_top_five, acc_aft_p_top_one, acc_aft_p_top_five], delimiter=",",
#                    header='pf, base_t1, base_t5, pr_retrain_t1, pr_retrain_t5, finetune_t1, finetune_t5', fmt='%f')
#
#
#
# print("current p",curr_p)
# print("finetune timep:",r_time_p)
