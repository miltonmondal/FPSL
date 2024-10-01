import torch
import os
import glob
import pandas
import csv
import numpy as np
import sys
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import API_multi as api
import API as api_for_flops  ###This single-gpu API is used to count flops
from variable_list import V
from flops_API import get_model_complexity_info

### contains seven functions (first two- only one time call)
##1.save_filter_importance_by_present_and_next_layer_filter_norm
##2. find_global_threshold


def filter_norm_present_next_layer(net, path_epochwise):
    #Create and save importance directly using filter norm using present and next layer
    os.makedirs(path_epochwise + "/imp_val", exist_ok=True)
    os.makedirs(path_epochwise + "/present_layer_filter_norm", exist_ok=True)
    os.makedirs(path_epochwise + "/next_layer_corr_channel_norm", exist_ok=True)
    os.makedirs(path_epochwise + "/mul_present_next_norm", exist_ok=True)
    os.makedirs(path_epochwise + "/present_layer_filter_mean", exist_ok=True)
    os.makedirs(path_epochwise + "/next_layer_corr_channel_mean", exist_ok=True)
    os.makedirs(path_epochwise + "/mul_present_next_mean", exist_ok=True)


    for l in range(net.max_layers() - V.ig_l):
        print("layer number: ", l)
        weight_l, bias_l = net.get_weights(l)
        weight_l_next, bias_l_next = net.get_weights(l, next_conv=True)
        # print("weight_l shape: ", weight_l.shape)
        # print("weight_l_next shape: ", weight_l_next.shape)

        # present layer filter norm criteria (p_n)
        p_n = torch.sum(torch.abs(weight_l), dim= (1,2,3)).data.cpu().numpy()
        #next layer filter norm criteria (n_n)
        n_n = torch.sum(torch.abs(weight_l_next), dim= (0,2,3)).data.cpu().numpy()
        #multiplication of present and next layer norm
        mpn_n = np.multiply(p_n,n_n)
        normal_p_n = torch.mean(torch.abs(weight_l), dim=(1, 2, 3)).data.cpu().numpy()
        normal_n_n = torch.mean(torch.abs(weight_l_next), dim=(0,2,3)).data.cpu().numpy()
        # importance = np.multiply(normal_p_n, normal_n_n)
        importance = mpn_n/ weight_l.shape[0]

        # print("present layer normalized filter importance shape: ", normal_p_n)
        # print("next layer normalized filter importance shape: ", normal_n_n)
        print("multiplication of present & next layer normalized filter importance shape: ", importance)
        # print("present_layer_shape: ", weight_l.shape)
        print("importance shape:", importance.shape)
        p_layer = []
        n_layer = []
        mpn_layer = []
        normal_p_layer = []
        normal_n_layer = []
        importance_layer = []


        for j in range(net.max_filters(layer=l)):
            p_layer.append(p_n[j])
            n_layer.append(n_n[j])
            mpn_layer.append(mpn_n[j])
            normal_p_layer.append(normal_p_n[j])
            normal_n_layer.append(normal_n_n[j])
            importance_layer.append(importance[j])




        print("importance of imp of layer " + str(l) + "filter", importance_layer)
        np.savetxt(path_epochwise + '/imp_val/layer_' + str(l) + '.csv', importance_layer)
        np.savetxt(path_epochwise + '/present_layer_filter_norm/layer_' + str(l) + '.csv', p_layer)
        np.savetxt(path_epochwise + '/next_layer_corr_channel_norm/layer_' + str(l) + '.csv', n_layer)
        np.savetxt(path_epochwise + '/mul_present_next_norm/layer_' + str(l) + '.csv', mpn_layer)
        np.savetxt(path_epochwise + '/present_layer_filter_mean/layer_' + str(l) + '.csv', normal_p_layer)
        np.savetxt(path_epochwise + '/next_layer_corr_channel_mean/layer_' + str(l) + '.csv', normal_n_layer)

####### find_global_threshold is not used rather threshold_for_prunable_layers is used ####

def find_global_threshold(p, path_epochwise):
    # Maximum allowable pruning per layer (limits sometime)

    def concatinate(indir,p):
        fileList = glob.glob(indir + "/*.csv")
        dfList = []
        for filename in fileList:
            print(filename)
            df = pandas.read_csv(filename, header=None)
            dfList.append(df)
        concatDF = pandas.concat(dfList, axis=0).to_numpy()

        concatDF1 = np.sort(concatDF, axis=0)

        for k in range(len(concatDF1) - 1):
            if concatDF1[k] <= concatDF1[k + 1]:
                print("true")
            else:
                break
        # T_V = 0.7 means 70% pruning we want, 0.7>0.6 = 70% pruning is greater than 60%
        # Exception case for a layer: (majority(90%)/all) smaller than thresold -> layer wise thrsold is T_V [[[[GOLDEN RULE]]]]
        # T_V=concatDF1[int(0.55*len(concatDF1))]

        ##### Percentange of Pruning Required #####
        T_V = concatDF1[int(p * len(concatDF1))]
        # os.makedirs(path_epochwise + '/Pruning_Desired ' + str(p * 100) + '%', exist_ok=True)
        os.makedirs(path_epochwise + '/Threshold', exist_ok=True)
        np.savetxt(path_epochwise + '/Threshold' + '/p_f' + str(p) + '.csv', T_V)
        return T_V
    threshold = concatinate(path_epochwise + "/imp_val",p)

    return threshold

def threshold_for_prunable_layers(net, p, epoch, path_epochwise, store_path, target_flag):
    if target_flag == 0:
        imp_list = []
        if epoch > 1:
            chk = np.loadtxt(store_path+ '/training_epochs/epoch_'+str(epoch-1) + '/model after pruning/prune_logic_epoch_' + str(epoch-1) + '.csv', dtype=str)
            for l in range(net.max_layers() - V.ig_l):
                imp_score = np.loadtxt(path_epochwise + '/imp_val/layer_' + str(l) + '.csv')
                if chk[l] == 'N':
                    imp_list.append(imp_score)
        else:
            for l in range(net.max_layers() - V.ig_l):
                imp_score = np.loadtxt(path_epochwise + '/imp_val/layer_' + str(l) + '.csv')
                imp_list.append(imp_score)

        print("imp list", imp_list)
        imp_list = list(np.concatenate(imp_list).flat)
        imp_array = np.asarray(imp_list)
        sorted_array = np.sort(imp_array)
        th_index = np.int64(p*len(sorted_array))-1
        th_value = sorted_array[th_index]
    else:
        th_value = np.loadtxt(store_path + '/final_pruned_model/Threshold' + '/p_f' + str(p) + '.csv', dtype='float32')

    # print("threshold: ", th_value)
    os.makedirs(path_epochwise + '/Threshold', exist_ok=True)
    np.savetxt(path_epochwise + '/Threshold' + '/p_f' + str(p) + '.csv', [th_value])

def display_importance(net, p, path_epochwise, epoch, store_path, curr_reduction):
    p_list,n_list,mpn_list,norm_p_list,norm_n_list,norm_mpn_list = ([] for i in range(6))
    for l in range(net.max_layers() - V.ig_l):
        print("layer number: ",l)
        p_n = np.loadtxt(path_epochwise + '/present_layer_filter_norm/layer_' + str(l) + '.csv')
        #next layer filter norm criteria (n_n)
        n_n = np.loadtxt(path_epochwise + '/next_layer_corr_channel_norm/layer_' + str(l) + '.csv')
        #multiplication of present and next layer norm
        mpn_n = np.loadtxt(path_epochwise + '/mul_present_next_norm/layer_' + str(l) + '.csv')
        normal_p_n = np.loadtxt(path_epochwise + '/present_layer_filter_mean/layer_' + str(l) + '.csv')
        normal_n_n = np.loadtxt(path_epochwise + '/next_layer_corr_channel_mean/layer_' + str(l) + '.csv')
        normal_mpn_n = np.loadtxt(path_epochwise + '/imp_val/layer_' + str(l) + '.csv')

        print("present layer normalized filter importance shape: ", normal_p_n)
        print("next layer normalized filter importance shape: ", normal_n_n)
        print("multiplication of present & next layer normalized filter importance shape: ", normal_mpn_n)

        p_list.append(p_n)
        n_list.append(n_n)
        mpn_list.append(mpn_n)
        norm_p_list.append(normal_p_n)
        norm_n_list.append(normal_n_n)
        norm_mpn_list.append(normal_mpn_n)

    plt.clf()
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    fig, axs = plt.subplots(nrows=2, ncols=3)
    fig.tight_layout()

    for i in range(len(p_list)):
        # ax.scatter([i] * len(p_list[i]), norm_mpn_list[i])
        axs[0,0].scatter([i]*len(p_list[i]), p_list[i])
        axs[0,1].scatter([i] * len(p_list[i]), n_list[i])
        axs[0,2].scatter([i] * len(p_list[i]), mpn_list[i])
        axs[1, 0].scatter([i] * len(p_list[i]), norm_p_list[i])
        axs[1, 1].scatter([i] * len(p_list[i]), norm_n_list[i])
        axs[1, 2].scatter([i] * len(p_list[i]), norm_mpn_list[i])


    ## plt.scatter(range(len(p_list[0])), p_list[0])
    ## axs[0,0].scatter(total_layers,p_list, color='red')
    # sorted_norm_mpn = np.sort(np.hstack(norm_mpn_list))
    # print("length of sorted array: ",len(sorted_norm_mpn))
    # th1_p = sorted_norm_mpn[np.int64(0.6*len(sorted_norm_mpn))]
    # th2_p = sorted_norm_mpn[np.int64(0.8*len(sorted_norm_mpn))]
    # ax.axhline(y=th1_p, color='g', linewidth= 0.5, linestyle='-')
    # ax.axhline(y=th2_p, color='r', linewidth= 0.5, linestyle='-')
    # ax.set(xlabel='layer index', ylabel='normalized_mpn_n')


    sorted_p = np.sort(np.hstack(p_list))
    sorted_n = np.sort(np.hstack(n_list))
    sorted_mpn = np.sort(np.hstack(mpn_list))
    sorted_norm_p = np.sort(np.hstack(norm_p_list))
    sorted_norm_n = np.sort(np.hstack(norm_n_list))
    sorted_norm_mpn = np.sort(np.hstack(norm_mpn_list))


    th_00 = sorted_p[np.int64(p*len(sorted_p) )]
    th_01 = sorted_n[np.int64(p*len(sorted_n))]
    th_02 = sorted_mpn[np.int64(p*len(sorted_mpn))]
    th_10 = sorted_norm_p[np.int64(p*len(sorted_norm_p))]
    th_11 = sorted_norm_n[np.int64(p*len(sorted_norm_mpn))]
    th_12 = sorted_norm_mpn[np.int64(p*len(sorted_norm_mpn))]

    axs[0,0].set(xlabel='layer index', ylabel='p_n')
    axs[0,0].axhline(y=th_00, color='g', linestyle='-')

    axs[0,1].set(xlabel='layer index', ylabel='n_n')
    axs[0,1].axhline(y=th_01, color='g', linestyle='-')

    axs[0,2].set(xlabel='layer index', ylabel='mpn_n')
    axs[0,2].axhline(y=th_02, color='g', linestyle='-')

    axs[1,0].set(xlabel='layer index', ylabel='normalized_p_n')
    axs[1,0].axhline(y=th_10, color='g', linestyle='-')

    axs[1,1].set(xlabel='layer index', ylabel='normalized_n_n')
    axs[1,1].axhline(y=th_11, color='g', linestyle='-')

    axs[1,2].set(xlabel='layer index', ylabel='normalized_mpn_n')
    axs[1,2].axhline(y=th_12, color='g', linestyle='-')



    fig.suptitle('epoch_'+str(epoch)+'_current reduction in FLOPs_'+str(curr_reduction)+'%')
    fig.savefig(store_path+'/'+ V.dataset_string+'_'+V.model_str+str(V.n_l)+'_epoch_'+str(epoch))
    plt.close(fig)

    plt.close()

def flops_parameters_count(path,epoch,image_dim):
    net = api_for_flops.Models(model=V.model_str, num_layers=V.n_l, num_class=V.n_c).net()
    macs, params = get_model_complexity_info(net, image_dim, as_strings=False, print_per_layer_stat=False,
                                             verbose=False)

    # net = api_for_flops.Models(model=V.model_str, num_layers=V.n_l).net()

    net.restore_pruned_state(path + '/model after pruning/epoch_'
                             + str(epoch), arch_only=True)

    macs_pruned, params_pruned = get_model_complexity_info(net, image_dim, as_strings=False,
                                                           print_per_layer_stat=False,
                                                           verbose=False)

    return macs, params, macs_pruned, params_pruned


def iterative_prune_and_finetune(p, df, c_epoch, t_epoch, lr, target_flag ,store_path):

    def my_loss(output, data, labels):
        return torch.nn.CrossEntropyLoss()(output, labels)


    def learningrateprovider(epoch, first_step=150, second_step=225,
                             lr=0.1, mul_fac=0.1):
        if epoch > second_step:
            lear = lr * np.power(mul_fac, 2)
        elif epoch > first_step:
            lear = lr * np.power(mul_fac, 1)
        else:
            lear = lr
        return lear

    net = api.Models(model=V.model_str, num_layers=V.n_l, num_class=V.n_c).net()
    net.restore_checkpoint(V.restore_checkpoint_path)
    te_acc_org = net.evaluate(V.dataset, train_images=False)


    target_flops_flag = target_flag
    for e in range(c_epoch+1, t_epoch + 1):

        path_epochwise = store_path+ '/training_epochs/epoch_'+str(e)

        os.makedirs(path_epochwise + '/model after pruning', exist_ok=True)
        os.makedirs(path_epochwise + '/model after finetuning', exist_ok=True)

        if e > 1:
            net = api.Models(model=V.model_str, num_layers=V.n_l, num_class=V.n_c).net()
            net.restore_pruned_state(store_path+ '/training_epochs/epoch_'+str(e-1) + '/model after pruning/epoch_'
                                  + str(e-1))
            net.restore_checkpoint(store_path+ '/training_epochs/epoch_'+str(e-1) + '/model after finetuning/epoch_'
                                   + str(e-1) + '.ckpt')


        filter_norm_present_next_layer(net, path_epochwise)
        # find_global_threshold(p, path_epochwise)
        threshold_for_prunable_layers(net, p, e, path_epochwise, store_path, target_flops_flag)

        threshold_value = np.loadtxt(path_epochwise + '/Threshold' + '/p_f' + str(p) + '.csv', dtype='float32')
        i_f_l, r_f_l, p_r_l, c_l = ([] for i in range(4))
        print("baseline_test_accuary: ", te_acc_org)
        init_num_filters = 0
        final_num_filters = 0

        if target_flops_flag == 0:
            for j in range(0, net.max_layers()-V.ig_l):
                desired_layer = j
                a = net.max_filters(desired_layer)
                init_num_filters += a
                i_f_l.append(a)
                # print(net)
                for i in range(net.max_filters(layer=desired_layer)):
                    temp_importance = np.loadtxt(path_epochwise + '/imp_val/layer_' + str(desired_layer) + '.csv',
                                                 dtype='float32')
                    # print("HIIIIIIIIIIII j is", j)
                    # print("temp importance:", temp_importance)
                    # print(temp_importance.size)
                    # print(type(temp_importance))


                    ordered_importance = np.sort(temp_importance)
                    # print("ordered_importance",ordered_importance)
                    ordered_indices = np.argsort(temp_importance)

                    ## two filters are retained (instead of whole layer pruning)
                    if i>= net.max_filters(layer=j)-2:
                        print("all small then threshold....................................................................")
                        ## retaining 1 filter at least in a layer to get rid of layer collapse
                        p_indices = ordered_indices[0:net.max_filters(layer=j)-2]
                        indicator = 'A'
                        break


                    if ordered_importance[i] > threshold_value:
                        print("j is", j)
                        print("normal case!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        p_indices = ordered_indices[0:i]
                        indicator = 'N'
                        break





                sorted_arr = np.sort(p_indices)
                p_indices = sorted_arr[::-1]
                print("p_indices", p_indices)



                for i in range(len(p_indices)):
                    net.prune(layer=desired_layer, filter=p_indices[i], verbose=False)

                b = net.max_filters(desired_layer)
                final_num_filters += b
                retain_percent = (b * 100) / a
                print("retain_percentage_wrt_previous" + str(retain_percent) + "layer" + str(desired_layer))
                r_f_l.append(b)
                p_r_l.append(retain_percent)
                c_l.append(indicator)

        te_acc_pruned = net.evaluate(V.dataset, train_images=False)
        net.save_pruned_state(path_epochwise + '/model after pruning/epoch_'
                                       + str(e))

        # optim = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        optim = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        net.attach_optimizer(optim)
        net.attach_loss_fn(my_loss)
        lear = learningrateprovider(e)
        print("learning rate:", lear)
        net.change_optimizer_learning_rate(lear)

        net.start_training(V.dataset, eval_freq=10000, epoch=1, loss_in_db=True)
        net.save_checkpoint(path_epochwise + "/model after finetuning/epoch_" + str(e) + '.ckpt')

        acc_test, test_top5 = net.evaluate(V.dataset)

        macs, params, macs_pruned, params_pruned = flops_parameters_count(path_epochwise,e,V.image_dim)
        per_red_param = ((params - params_pruned) * 100 / params)
        per_red_FLOPs = ((macs - macs_pruned) * 100 / macs)

        display_importance(net, p, path_epochwise,  e, store_path+ '/imp_score_visualization', round(per_red_FLOPs, 4))
        if target_flops_flag == 0:
            #### A file should contain the information that how did we get the pruned model
            #### & currently what is the pruned model details (model_pruning_details)
            row1a = ['model', 'dataset', 'n_c', 'b_size', 'target_FLOPs_rdcn(%)', 'pruning(%)_epochwise', 'filters_rdcn(%)_curr_epoch']
            row1b = [V.model_str + str(V.n_l), V.dataset_string, V.n_c, V.b_size, df, p*100, round((1 - (final_num_filters/init_num_filters))*100,4)]
            row2a = ['init_params(million)', 'final_params', 'params_pruned', 'params_pruned(%)']
            row2b = [round((params / 1e6), 4), round((params_pruned / 1e6), 4), round(((params - params_pruned) / 1e6), 4),
                     round(per_red_param, 4)]
            row3a = ['init_FLOPs(million)', 'final_FLOPs', 'FLOPs_reduced', 'reduction_in_FLOPs(%)']
            row3b = [round((macs / 1e6), 4), round((macs_pruned / 1e6), 4), round(((macs - macs_pruned) / 1e6), 4),
                     round(per_red_FLOPs, 4)]
            row4a = ['Number of Filters in each layer for Initial Model:']
            row4b = i_f_l
            row5a = ['Number of Filters in each layer for Pruned Model:']
            row5b = r_f_l
            row6a = ['Percentage(%) of Filters in each layer after pruning:']
            row6b = list(np.around(np.array(p_r_l), 3))
            row7a = ['Final Pruning logic applied layerwise :']
            row7b = c_l

            print("cl is: ", c_l)
            np.savetxt(path_epochwise + '/model after pruning/prune_logic_epoch_' + str(e) + '.csv', c_l,  fmt="%s")

            with open(path_epochwise + '/model after pruning/prune_summary_epoch_' + str(e) + '.csv',
                      'wt') as results_file:
                csv_writer = csv.writer(results_file)
                csv_writer.writerow(row1a)
                csv_writer.writerow(row1b)
                csv_writer.writerow(row2a)
                csv_writer.writerow(row2b)
                csv_writer.writerow(row3a)
                csv_writer.writerow(row3b)
                csv_writer.writerow(row4a)
                csv_writer.writerow(row4b)
                csv_writer.writerow(row5a)
                csv_writer.writerow(row5b)
                csv_writer.writerow(row6a)
                csv_writer.writerow(row6b)
                csv_writer.writerow(row7a)
                csv_writer.writerow(row7b)
                csv_writer.writerow(['Testing Accuarcy for pruned model: ' + str(te_acc_pruned)])
##### NEW OBSERVATION for iterative pruning
            with open(store_path + '/Iterative_pruning_modes.csv', 'a') as ipm_file:
                csv_writer = csv.writer(ipm_file)
                csv_writer.writerow(row7b)

            with open(store_path + '/Iterative_pruning_details.csv', 'a') as ipd_file:
                csv_writer = csv.writer(ipd_file)
                csv_writer.writerow(row5b)


        # if per_red_FLOPs >= df:
        #     if target_flops_flag == 0:
        #         prev_best = [0,0]
        #     else:
        #         prev_best = np.loadtxt(store_path + "/best_epoch_within_target/best_epoch.txt")
        #
        #     target_flops_flag = 1
        #     os.makedirs(store_path + '/final_pruned_model/Threshold', exist_ok=True)
        #     os.makedirs(store_path + '/final_pruned_model/model_details', exist_ok=True)
        #     np.savetxt(store_path + '/final_pruned_model/Threshold' + '/p_f' + str(p) + '.csv', [threshold_value])
        #     net.save_pruned_state(store_path + '/final_pruned_model/model_details')
        #     prev_best_epoch = prev_best[0]
        #     prev_best_acc_test = prev_best[1]
        #     if acc_test >= prev_best_acc_test:
        #         net.save_checkpoint(store_path + "/best_epoch_within_target/best_epoch.ckpt")
        #         np.savetxt(store_path + "/best_epoch_within_target/best_epoch.txt", [e, acc_test])

        if per_red_FLOPs >= df:
            target_flops_flag = 1
            os.makedirs(store_path + '/final_pruned_model/Threshold', exist_ok=True)
            os.makedirs(store_path + '/final_pruned_model/model_details', exist_ok=True)
            np.savetxt(store_path + '/final_pruned_model/Threshold' + '/p_f' + str(p) + '.csv', [threshold_value])

        last_epoch = e
        net.save_checkpoint(store_path + "/training_last_epoch/last_epoch.ckpt")
        np.savetxt(store_path + "/training_last_epoch/last_epoch.txt",
                   [last_epoch, acc_test])

        # print("Training Accuracy after " + str(e) + " epochs: " + str(acc_train))
        print("Testing Accuracy after " + str(e) + " epochs: " + str(acc_test))
        if e == 1:
            head_row = ['epoch', 'lr', 'acc_test', 'acc_test_top5', 'rdcn_FLOPs(%)', 'rdcn_params(%)' ]
            with open(store_path + '/Training_Results_original_defined.csv', 'wt') as results_file:
                csv_writer = csv.writer(results_file)
                csv_writer.writerow(head_row)
        row = [e, lear, round(acc_test, 4), round(test_top5, 4), round(per_red_FLOPs, 4), round(per_red_param, 4) ]
        with open(store_path + '/Training_Results_original_defined.csv', 'a') as results_file:
            csv_writer = csv.writer(results_file)
            csv_writer.writerow(row)

    if last_epoch == t_epoch:
        net.restore_checkpoint(store_path + "/training_last_epoch/last_epoch.ckpt")
        acc_train, train_top5 = net.evaluate(V.dataset, train_images=True)
        acc_test, test_top5 = net.evaluate(V.dataset)
        row1 = ['curr_FLOPs', 'rdcn_FLOPs(%)', 'curr_params', 'rdcn_params(%)',
                'tr_acc', 't5_tr_acc', 'b_te', 'b_t5_te', 'te_acc', 't5_te_acc']
        row2 = [round((macs_pruned/ 1e6), 4), round(per_red_FLOPs, 4),round((params_pruned / 1e6), 4), round(per_red_param, 4),
                round(acc_train, 4), round(train_top5, 4), round(te_acc_org[0], 4), round(te_acc_org[1], 4),  round(acc_test, 4), round(test_top5, 4)]
        with open(store_path + '/Training_Results_last_epoch.csv', 'wt') as results_file:
            csv_writer = csv.writer(results_file)
            csv_writer.writerow(row1)
            csv_writer.writerow(row2)


