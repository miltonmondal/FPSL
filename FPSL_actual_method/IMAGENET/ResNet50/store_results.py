import torch
import math
from variable_list import V
import API_multi as api
import pruning_API_IFP as prune_api
from flops_API import get_model_complexity_info
import numpy as np
import time
import os
import csv


def flops_parameters_count(path,epoch,image_dim):
    net = api.Models(model=V.model_str, num_layers=V.n_l, num_class=V.n_c, pretrained=False, version='B').net()
    macs, params = get_model_complexity_info(net, image_dim, as_strings=False, print_per_layer_stat=False,
                                             verbose=False)

    # net = api_for_flops.Models(model=V.model_str, num_layers=V.n_l).net()

    net.restore_pruned_state(path + '/model after pruning/epoch_'
                             + str(epoch), arch_only=True)

    macs_pruned, params_pruned = get_model_complexity_info(net, image_dim, as_strings=False,
                                                           print_per_layer_stat=False,
                                                           verbose=False)

    return macs, params, macs_pruned, params_pruned


def results_print(p, df, c_epoch, t_epoch, lr,store_path):

    def my_loss(output, data, labels):
        return torch.nn.CrossEntropyLoss()(output, labels)

    ###### learning rate provider is different for ImageNet ResNet combination #####
    def learningrateprovider(epoch, warmup_epochs=5, max_epoch_for_cosine=90, base_lr=lr, final_lr=1e-4,
                             warmup_begin_lr=1e-2, lr_factor=0.256):
        max_steps = max_epoch_for_cosine - warmup_epochs
        if epoch < warmup_epochs:
            increase = (base_lr - warmup_begin_lr) / float(warmup_epochs)
            lear = warmup_begin_lr + (increase * float(epoch))
            # cool down condition
        elif epoch > max_epoch_for_cosine:
            lear = final_lr
        else:
            lear_cos = final_lr + (base_lr - final_lr) * (
                        1 + math.cos(math.pi * (epoch - warmup_epochs) / max_steps)) / 2
            # lr_factor is for lr reduced in each step [epoch 30, 60]
            lear = lear_cos*(lr_factor ** np.int(epoch/30))
            # lear = lear_cos

        return lear

    net = api.Models(model=V.model_str, num_layers=V.n_l, num_class=V.n_c, pretrained=True, version='B').net()
    te_acc_org = net.evaluate(V.dataset, train_images=False)

    for e in range(c_epoch+1, t_epoch + 1):

        path_epochwise = store_path+ '/training_epochs/epoch_'+str(e)

        if e > 1:
            net = api.Models(model=V.model_str, num_layers=V.n_l, num_class=V.n_c, pretrained=False, version='B').net()

            net.restore_pruned_state(store_path+ '/training_epochs/epoch_'+str(e-1) + '/model after pruning/epoch_'
                                  + str(e-1))
            net.restore_checkpoint(store_path+ '/training_epochs/epoch_'+str(e-1) + '/model after finetuning/epoch_'
                                   + str(e-1) + '.ckpt')


        te_acc_pruned = net.evaluate(V.dataset, train_images=False)
        lear = learningrateprovider(e)
        print("learning rate:", lear)

        acc_test, test_top5 = net.evaluate(V.dataset)

        macs, params, macs_pruned, params_pruned = flops_parameters_count(path_epochwise,e,V.image_dim)
        per_red_param = ((params - params_pruned) * 100 / params)
        per_red_FLOPs = ((macs - macs_pruned) * 100 / macs)

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
        net.restore_checkpoint(store_path + "/best_epoch_within_target/best_epoch.ckpt")
        acc_train, train_top5 = net.evaluate(V.dataset, train_images=True)
        acc_test, test_top5 = net.evaluate(V.dataset)
        best = [acc_test, test_top5]
        row1 = ['curr_FLOPs', 'rdcn_FLOPs(%)', 'curr_params', 'rdcn_params(%)',
                'tr_acc', 't5_tr_acc', 'b_te', 'b_t5_te', 'te_acc', 't5_te_acc']
        row2 = [round((macs_pruned/ 1e6), 4), round(per_red_FLOPs, 4),round((params_pruned / 1e6), 4), round(per_red_param, 4),
                round(acc_train, 4), round(train_top5, 4), round(te_acc_org[0], 4), round(te_acc_org[1], 4),  round(acc_test, 4), round(test_top5, 4)]
        row3 = ['base_FLOPs', 'base_params']
        row4 = [round((macs / 1e6), 4), round((params / 1e6), 4)]
        with open(store_path + '/Training_Results_best_epoch.csv', 'wt') as results_file:
            csv_writer = csv.writer(results_file)
            csv_writer.writerow(row1)
            csv_writer.writerow(row2)
            csv_writer.writerow(row3)
            csv_writer.writerow(row4)

### starting epoch for prune and finetune(resume capability)
# Starting Epoch in fine tune (s_epoch= 0 by default/ last_checkpoint saved)
s_epoch = 0

### user defined flops reduction target(%) => retained FLOPs = (100- flops reduction)
reduction_in_flops = [64] #[64,69]
### pruning percentage per epoch (ppe) (%)
ppe = 2.5 #2
## Total finetune epochs & initial learning rate
f_epochs = 100
lr_init = 0.256

prune_results = V.base_path_results + '/reduction_in_flops_'+ str(reduction_in_flops[0])+'_percent_and_pruning_percentage_per_epoch_'+str(ppe)

results_print(ppe/100, reduction_in_flops[0], s_epoch, f_epochs, prune_results)