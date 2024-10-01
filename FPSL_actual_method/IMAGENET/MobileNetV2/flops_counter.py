import API_multi as api
from flops_counter_api import get_model_complexity_info
from variable_list import V
import sys
import os


def count_flops(checkpoint, flop_counter_pyfile):
    command = 'CUDA_VISIBLE_DEVICES=0 python ' + flop_counter_pyfile + '.py ' + checkpoint
    os.system(command)

    with open('.flops_temp', 'r') as temp:
        mac, params = temp.read().strip().split(',')

    return mac, params


def save_flops(net):
    macs, params = get_model_complexity_info(net, V.image_dim, as_strings=False, print_per_layer_stat=False,
                                             verbose=False)
    with open('.flops_temp', 'w+') as temp:
        temp.writelines(str(macs) + ',' + str(params))



def model_reinit(checkpoint):

    ##############################
    # Re-initialize the model here
    # Add restore_checkpoint
    ##############################

    net = api.Models(model=V.model_str, num_class=V.n_c).net()
    net.restore_pruned_state(checkpoint)

    ##############################

    save_flops(net)  # Don't modify this line


if __name__ == '__main__':
    model_reinit(sys.argv[1])
