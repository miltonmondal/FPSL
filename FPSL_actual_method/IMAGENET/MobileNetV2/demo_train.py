import torch
import API_multi as api


def main():

    net = api.Models('ResNet', 50).net()

    dataset = api.Datasets('ImageNet', 256)
    optim = torch.optim.SGD(net.parameters(), lr=1e-3)

    def loss(out, data, labels):
        return torch.nn.CrossEntropyLoss()(out, labels)

    net.attach_optimizer(optim)
    net.attach_loss_fn(loss)
    net.start_training(dataset, 1000, 1)


if __name__ == '__main__':
    main()
