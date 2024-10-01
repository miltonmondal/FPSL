import torch
import API_multi as api
import numpy as np
from variable_list import V

def main():

    net = api.Models(model=V.model_str, num_class=V.n_c).net()

    dataset = api.Datasets('CIFAR10', 128)
    print("total layers: ",net.max_layers())
    print(net)

    for l in range(net.max_layers()):
        print("layer number: ", l)
        weight_l1, _ = net.get_weights(l, next_conv=0)
        weight_l2, _ = net.get_weights(l, next_conv=1)
        weight_l3, _  = net.get_weights(l, next_conv=2)
        # print(weight_l_1)
        # print(weight_l_2)
        print("first weight shape:", weight_l1.shape)
        print("second weight shape:", weight_l2.shape)
        print("third weight shape:", weight_l3.shape)
        n1 = torch.sum(torch.abs(weight_l1), dim= (1,2,3)).data.cpu().numpy()
        n2 = torch.sum(torch.abs(weight_l2), dim= (1,2,3)).data.cpu().numpy()
        n3 = torch.sum(torch.abs(weight_l3), dim= (0,2,3)).data.cpu().numpy()
        n12 = np.multiply(n1,n2)
        mpn_n = np.multiply(n12, n3)
        importance = mpn_n / weight_l2.shape[0]
        print("n1 shape:", n1.shape)
        print("n2 shape:", n2.shape)
        print("n3 shape:", n3.shape)
        print("mpn shape:", mpn_n.shape)
        # exit()

        # # present layer filter norm criteria (p_n)
        # p_n = torch.sum(torch.abs(weight_l), dim= (1,2,3)).data.cpu().numpy()
        # #next layer filter norm criteria (n_n)
        # n_n = torch.sum(torch.abs(weight_l_next), dim= (0,2,3)).data.cpu().numpy()
        # #multiplication of present and next layer norm
        # mpn_n = np.multiply(p_n,n_n)
        # normal_p_n = torch.mean(torch.abs(weight_l), dim=(1, 2, 3)).data.cpu().numpy()
        # normal_n_n = torch.mean(torch.abs(weight_l_next), dim=(0,2,3)).data.cpu().numpy()
        # # importance = np.multiply(normal_p_n, normal_n_n)
        # importance = mpn_n / weight_l.shape[0]
        # # print("present layer normalized filter importance shape: ", normal_p_n)
        # # print("next layer normalized filter importance shape: ", normal_n_n)
        # print("multiplication of present & next layer normalized filter importance shape: ", importance)
        # # print("present_layer_shape: ", weight_l.shape)
        # print("importance shape:", importance.shape)


if __name__ == '__main__':
    main()
