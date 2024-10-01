import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import torchvision, sys, math, os, time, tqdm, sklearn.metrics
import time

import copy, glob
from torchvision.io import read_image, ImageReadMode


class Models:
    def __init__(
        self,
        model,
        num_layers=None,
        version=None,
        num_transition_shape=None,
        num_linear_units=None,
        num_class=None,
        pretrained=False,
        with_conv_multipliers=False,
        phase_shift_mode=None,
        phase_shift_fn=None,
    ):
        self._model_list = ['VGG', 'ResNet', 'MobileNetV2']
        assert model in self._model_list, 'Model must be either ' + ' or '.join(self._model_list)
        self._model = model
        self._num_layers = num_layers
        self._version = version
        self._num_transition_shape = num_transition_shape
        self._num_linear_units = num_linear_units
        self._num_class = num_class
        self._pretrained = pretrained
        self._with_conv_multipliers = with_conv_multipliers
        self._phase_shift_mode = phase_shift_mode
        self._phase_shift_fn = phase_shift_fn
        if model == 'ResNet':
            if self._version is None:
                print('No resnet version specified. Using version A (He et. al.)')
                self._version = 'A'
            if self._version == 'B':
                print('ResNet version B (Improved version)')
                assert num_layers in [50, 101, 152], 'ResNet version "B" is only applicable for layers=[50, 101, 152]'
            else:
                print('ResNet version A (He et. al.)')
        if self._pretrained:
            print('Loading pretrained model.')

    def net(self):
        if self._model == 'VGG':
            net = self._VGG_(
                self._num_layers, self._num_transition_shape, self._num_linear_units, self._num_class, self._pretrained
            )
        if self._model == 'ResNet':
            net = self._ResNet_(
                self._num_layers,
                self._num_class,
                self._version,
                self._pretrained,
                with_conv_multipliers=self._with_conv_multipliers,
                phase_shift_mode=self._phase_shift_mode,
                phase_shift_fn=self._phase_shift_fn,
            )
        if self._model == 'MobileNetV2':
            net = self._MobileNet_(
                self._num_class,
                self._pretrained,
            )

        return net

    class _Agnostic_:
        def start_training(
            self,
            dataset,
            eval_freq,
            epoch,
            preprocessFn=None,
            loss_in_db=False,
            final_eval=True,
        ):
            collate_loss, correct_predictions, total_processed = (
                torch.zeros(1).cuda(),
                torch.zeros(1).cuda(),
                torch.zeros(1).cuda(),
            )
            tic = time.time()
            # return
            for e in range(1, epoch + 1):
                epoch_tic = time.time()
                for data, labels in dataset.train_images:
                    if preprocessFn is not None:
                        data = torch.from_numpy(preprocessFn(data.numpy())).cuda()
                    labels = labels.cuda(None, non_blocking=True)
                    self.train()
                    output = self.forward(data)
                    loss = self._loss_fn(output, data, labels)
                    self._optim.zero_grad()
                    loss.backward()
                    self._optim.step()
                    self._global_step += 1
                    collate_loss += loss
                    predictions = output.argmax(dim=1, keepdim=True)
                    correct_predictions += predictions.eq(labels.view_as(predictions)).sum()
                    total_processed += labels.size()[0]
                    sys.stdout.write('Iter: %i\r' % self._global_step)
                    sys.stdout.flush()
                    if self._global_step % eval_freq == 0:
                        toc = time.time()
                        loss_value = collate_loss.item() / total_processed
                        if loss_in_db:
                            loss_value = -10 * np.log10(loss_value.cpu())
                        training_accuracy = 100 * (correct_predictions / total_processed).item()
                        top1, top5 = self.evaluate(dataset)
                        speed = total_processed / (toc - tic)
                        print(
                            'Iter: %d, Loss %.4f Acc(Train Eval Top5) %.2f %.2f %.2f @%dim/s'
                            % (self._global_step, loss_value, training_accuracy, top1, top5, speed)
                        )
                        collate_loss, correct_predictions, total_processed = (
                            torch.zeros(1).cuda(),
                            torch.zeros(1).cuda(),
                            torch.zeros(1).cuda(),
                        )
                        tic = time.time()
                print('\rEpoch', e, 'completed in', time.time() - epoch_tic, 'seconds')
            if final_eval:
                final_acc = self.evaluate(dataset)
                print('Final eval accuracy (Top1, Top5):', final_acc)

        def evaluate(
            self,
            dataset,
            verbose=False,
            confusion_matrix=False,
            train_images=False,
            mode='eval',
        ):
            if mode == 'eval':
                self.eval()
            elif mode == 'train':
                self.train()
            if train_images == False:
                dataset_images = dataset.eval_images
                dataset_num_images = dataset.num_eval_images
            else:
                dataset_images = dataset.train_images
                dataset_num_images = dataset.num_train_images
            correct_predictions = torch.zeros(1).cuda()
            correct_topk_predictions = torch.zeros(1).cuda()
            collected_labels, collected_predictions = [], []
            with torch.no_grad():
                if verbose:
                    data_generator = tqdm.tqdm(dataset_images)
                else:
                    data_generator = dataset_images
                for data, labels in data_generator:
                    data = data.cuda()
                    labels = labels.cuda()
                    output = self.forward(data)
                    predictions = output.argmax(dim=1, keepdim=True)
                    correct_predictions += predictions.eq(labels.view_as(predictions)).sum().item()
                    _, predictions_topk = output.topk(5, 1, True, True)
                    predictions_topk = predictions_topk.t()
                    correct_topk_predictions += predictions_topk.eq(
                        labels.view(1, -1).expand_as(predictions_topk)
                    ).sum()

                    if confusion_matrix:
                        collected_labels.extend(list(labels.cpu().numpy()))
                        collected_predictions.extend(list(predictions.view_as(labels).cpu().numpy()))

            if confusion_matrix:
                return (
                    100 * (correct_predictions / dataset_num_images).item(),
                    100 * (correct_topk_predictions / dataset_num_images).item(),
                    sklearn.metrics.confusion_matrix(collected_labels, collected_predictions),
                )
            else:
                return (
                    100 * (correct_predictions / dataset_num_images).item(),
                    100 * (correct_topk_predictions / dataset_num_images).item(),
                )

        def init_conv_layers(self):
            print('Initializing conv layers')
            for m in self.modules_copy:
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                    try:
                        m.bias.data.zero_()
                    except:
                        pass

        def attach_loss_fn(self, loss_fn):
            self._loss_fn = loss_fn

        def attach_optimizer(self, optim):
            self._optim = optim

        def change_optimizer_learning_rate(self, lr):
            self._optim.param_groups[0]['lr'] = lr

        def max_layers(self):
            return len(self._idx)

        def max_filters(self, layer):
            idx, iden = self._idx[layer]
            return self.modules[idx].out_channels

        def compute_gradients(self, dataset, num_batches=None, verbose=False):
            if verbose:
                data_generator = tqdm.tqdm(dataset.train_images)
            else:
                data_generator = dataset.train_images

            loss_metric = nn.CrossEntropyLoss()
            self.eval()
            self.zero_grad()

            i = 0
            for data, labels in data_generator:
                data, labels = data.cuda(), labels.cuda()
                output = self.forward(data)
                loss = loss_metric(output, labels)
                loss.backward()
                if i == num_batches:
                    break
                i += 1

        def get_gradients(self, layer, filter=None):
            return self.get_weights(layer, filter, grad=True)

        def get_features(
            self,
            dataset,
            num_batches,
            after_layer,
            layer_type='conv',
            return_type='tensor',
            verbose=False,
        ):

            assert layer_type in ['conv', 'bn', 'relu'], 'layer_type must be [\'conv\', \'bn\', \'relu\']'
            assert return_type in ['tensor', 'mean'], (
                'return_type must be either [' + ' '.join(['tensor', 'mean']) + ']'
            )

            class SaveFeatures:
                def __init__(self, layer):
                    self.handle = layer.register_forward_hook(self.hook_fn)

                def hook_fn(self, layer, input, output):
                    self.features = output

                def remove(self):
                    self.handle.remove()

            if layer_type == 'conv':
                required_layer = self.get_layer_from_tree(self._idx[after_layer][0])
            elif layer_type == 'bn' or layer_type == 'relu':
                required_layer = self.get_layer_from_tree(self._idx[after_layer][0] + 1)
            save_features = SaveFeatures(required_layer)

            if return_type == 'tensor':
                features_set, labels_set = [], []
            elif return_type == 'mean':
                features_sum, labels_set = [], []
                [features_sum.append(torch.zeros(1)) for _ in range(self._num_class)]
                [labels_set.append(0) for _ in range(self._num_class)]

            for i, (data, labels) in enumerate(dataset.train_images):

                with torch.no_grad():
                    self.forward(data.cuda())
                features = save_features.features
                if layer_type == 'relu':
                    features = F.relu(features.clone(), inplace=True)

                if return_type == 'tensor':
                    features_set.append(features.cpu().numpy())
                    labels_set.append(labels.cpu().numpy())
                elif return_type == 'mean':
                    for k, label in enumerate(labels):
                        features_sum[label] = features_sum[label] + features[k : k + 1, ...].cpu()
                        labels_set[label] += 1

                if verbose:
                    sys.stdout.write('\rFetching features: %i/%i' % (i + 1, num_batches))
                    sys.stdout.flush()

                if i == num_batches - 1:
                    break

            save_features.remove()
            print('\n')

            if return_type == 'tensor':
                return np.concatenate(features_set), np.concatenate(labels_set)
            elif return_type == 'mean':
                features_sum = [
                    features_sum[i].__mul__(torch.Tensor([1 / labels_set[i]])).cpu().numpy()
                    if labels_set[i] != 0
                    else np.float32([0])
                    for i in range(self._num_class)
                ]
                return features_sum, labels_set

        def get_weights(self, layer, filter=None, grad=False, next_conv=0):
            conv_idx = self._idx[layer][0]
            required_layer = self.get_layer_from_tree(conv_idx)

            flag_found = False
            while next_conv > 0:
                for i in range(conv_idx + 1, conv_idx + 10):
                    required_layer = self.get_layer_from_tree(i)
                    if isinstance(required_layer, nn.Conv2d):
                        next_conv -= 1
                        if next_conv == 0:
                            flag_found = True
                        else:
                            conv_idx = i
                        break

            if next_conv > 0 and not flag_found:
                raise NotImplementedError('proximity search for conv layer failed')

            conv_weight = required_layer.weight.data.clone()
            if grad:
                conv_weight = required_layer.weight.grad.data.clone()
            try:
                conv_bias = required_layer.bias.data.clone()
                if grad:
                    conv_bias = required_layer.bias.grad.data.clone()
            except:
                conv_bias = torch.zeros(0)
            if filter is not None:
                conv_weight = conv_weight[filter : filter + 1]
                conv_weight = conv_weight.cpu().numpy()
                try:
                    conv_bias = conv_bias[filter : filter + 1]
                    conv_bias = conv_bias.cpu().numpy()
                except:
                    pass
            return conv_weight, conv_bias

        def save_checkpoint(self, location=None):
            if location is None:
                location = self._checkpoint
            torch.save(self.state_dict(), location)
            print('Checkpoint saved at', location)

        def restore_checkpoint(self, location=None):
            if location is None:
                location = self._checkpoint
            try:
                self.load_state_dict(torch.load(location))
            except:
                self.resnet.load_state_dict(torch.load(location))
            print('Restoring checkpoint from', location)

        def save_pruned_state(self, name):
            try:
                os.makedirs(name)
            except FileExistsError:
                print('Warning: A pruned_state with name=' + name + ' already exists. Overwriting...')

            file = open(name + '/pruned_state.txt', 'w+')
            for state in self._pruned_state:
                layer, filter = state
                file.write(str(layer) + ',' + str(filter) + '\n')
            file.close()
            torch.save(self.state_dict(), name + '/pruned_weights.ckpt')

        def restore_pruned_state(self, name, initial_state=False):
            file = open(name + '/pruned_state.txt', 'r').read().strip().split('\n')
            self._pruned_state = []
            for data in file:
                if data != '':
                    layer, filter = data.strip().split(',')
                    layer, filter = int(layer), int(filter)
                    self.prune(layer, filter, verbose=False)
            if not initial_state:
                self.load_state_dict(torch.load(name + '/pruned_weights.ckpt'))

        def prune_conv_layer(self, conv0, filter, in_out):
            conv0_in_channels = conv0.in_channels
            conv0_out_channels = conv0.out_channels
            conv0_kernel_size = conv0.kernel_size[0]
            conv0_stride = conv0.stride[0]
            conv0_padding = conv0.padding[0]
            conv0_groups = conv0.groups
            conv0_weight = conv0.weight.data.clone()
            try:
                conv0_bias = conv0.bias.data.clone()
            except:
                pass

            if in_out == 'out':
                conv0_target_weight = self.delete_index(conv0_weight, at_index=filter)
                try:
                    conv0_target_bias = self.delete_index(conv0_bias, at_index=filter)
                    conv0.__init__(
                        conv0_in_channels,
                        conv0_out_channels - 1,
                        conv0_kernel_size,
                        conv0_stride,
                        conv0_padding,
                        1,
                        conv0_groups,
                    )
                except:
                    conv0.__init__(
                        conv0_in_channels,
                        conv0_out_channels - 1,
                        conv0_kernel_size,
                        conv0_stride,
                        conv0_padding,
                        1,
                        conv0_groups,
                        bias=False,
                    )
            elif in_out == 'in':
                conv0_target_weight = self.delete_index(conv0_weight, at_index=filter, dim=1)
                try:
                    conv0_target_bias = conv0_bias
                    conv0.__init__(
                        conv0_in_channels - 1,
                        conv0_out_channels,
                        conv0_kernel_size,
                        conv0_stride,
                        conv0_padding,
                        1,
                        conv0_groups,
                    )
                except:
                    conv0.__init__(
                        conv0_in_channels - 1,
                        conv0_out_channels,
                        conv0_kernel_size,
                        conv0_stride,
                        conv0_padding,
                        1,
                        conv0_groups,
                        bias=False,
                    )
            elif in_out == 'inout':
                conv0_target_weight = self.delete_index(conv0_weight, at_index=filter, dim=0)
                try:
                    conv0_target_bias = conv0_bias
                    conv0.__init__(
                        conv0_in_channels - 1,
                        conv0_out_channels - 1,
                        conv0_kernel_size,
                        conv0_stride,
                        conv0_padding,
                        1,
                        conv0_groups - 1,
                    )
                except:
                    conv0.__init__(
                        conv0_in_channels - 1,
                        conv0_out_channels - 1,
                        conv0_kernel_size,
                        conv0_stride,
                        conv0_padding,
                        1,
                        conv0_groups - 1,
                        bias=False,
                    )

            conv0.weight.data = conv0_target_weight
            try:
                conv0.bias.data = conv0_target_bias
            except:
                pass

        def prune_batchnorm_layer(self, bn, filter):
            bn_num_features = bn.num_features
            bn_weight = bn.weight.data.clone()
            bn_bias = bn.bias.data.clone()
            bn_running_mean = bn.running_mean.data.clone()
            bn_running_var = bn.running_var.data.clone()

            bn_target_num_features = bn_num_features - 1
            bn_target_weight = self.delete_index(bn_weight, at_index=filter)
            bn_target_bias = self.delete_index(bn_bias, at_index=filter)
            bn_target_running_mean = self.delete_index(bn_running_mean, at_index=filter)
            bn_target_running_var = self.delete_index(bn_running_var, at_index=filter)

            bn.__init__(bn_target_num_features)
            bn.weight.data = bn_target_weight
            bn.bias.data = bn_target_bias
            bn.running_mean.data = bn_target_running_mean
            bn.running_var.data = bn_target_running_var

        def prune_linear_layer(self, ln, filter, rc):
            ln_in_features = ln.in_features
            ln_out_features = ln.out_features
            ln_weight = ln.weight.data.clone()
            ln_bias = ln.bias.data.clone()

            if rc == 'row':
                ln_target_weight = self.delete_index(ln_weight, filter)
                ln_target_bias = self.delete_index(ln_bias, filter)
                ln.__init__(ln_in_features, ln_out_features - 1)
            elif rc == 'col':
                ln_target_weight = self.delete_index(ln_weight, filter, dim=1)
                ln_target_bias = ln_bias
                ln.__init__(ln_in_features - 1, ln_out_features)
            ln.weight.data = ln_target_weight
            ln.bias.data = ln_target_bias

        def delete_index(self, tensor, at_index, dim=0):
            if dim == 0:
                return torch.cat((tensor[:at_index, ...], tensor[at_index + 1 :, ...]))
            elif dim == 1:
                return torch.cat((tensor[:, :at_index, ...], tensor[:, at_index + 1 :, ...]), dim=dim)

    class _MobileNet_(nn.Module, _Agnostic_):
        def __init__(self, num_class=None, pretrained=False):
            super().__init__()
            self._global_step = 0
            self._checkpoint = './weights/mobilenetv2.ckpt'
            self._pruned_state = []
            self._idx = []
            self._tree = [[], []]
            self._optim = None
            self._loss_fn = None
            self._batch_multiplier = None
            self._num_class = num_class or 1000

            self.net = torchvision.models.mobilenet_v2(pretrained=pretrained)
            self.net = nn.DataParallel(self.net)
            self.modules = list(self.modules())
            module_type = [type(m).__name__ for m in self.modules]

            i, skip_first = 0, True
            while i < len(self.modules):
                m_type = module_type[i]
                if m_type == 'InvertedResidual':
                    if skip_first:
                        skip_first = False
                    else:
                        self._idx.append((i + 3, 'C'))
                        i += 11
                i += 1

            # For CIFAR
            if self._num_class == 10 or self._num_class == 100:
                self.modules[5].stride = (1, 1)
                self.modules[23].stride = (1, 1)
                self.modules[47].stride = (1, 1)

            self._super_string = super.__str__(self)
            super_string_split = self._super_string.split('\n')
            for string in super_string_split[1:-2]:
                if string[-1] == '(':
                    self._tree.append(self._tree[-1].copy())
                    self._tree[-1].append(0)
                elif string[-2:] == ' )':
                    self._tree[-1].pop(-1)
                    self._tree[-1][-1] += 1
                else:
                    self._tree.append(self._tree[-1].copy())
                    self._tree[-1][-1] += 1

            if self._num_class == 10 or self._num_class == 100:
                lin_idx = 214
                lin0 = self.get_layer_from_tree(lin_idx)
                lin0_in_features = lin0.in_features
                lin0.__init__(lin0_in_features, num_class)
                nn.init.normal_(lin0.weight, 0, 0.01)
                nn.init.zeros_(lin0.bias)

            self.modules_copy = self.modules
            self.cuda()

        def __str__(self):
            p_str = super().__str__()
            p_str += '\n\n\n----------------------\nIndex wise Module List\n----------------------\n\n'
            p_str += '\n'.join([str(i) + ' ' + type(m).__name__ for i, m in enumerate(self.modules)])
            p_str += '\n\n\n----------------\nPrunable Indices\n----------------\n\n'
            p_str += 'Identifier:ExternalIndex:InternalIndex LayerInfo\n\n'
            for i, (idx, iden) in enumerate(self._idx):
                conv_layer = self.modules[idx]
                p_str += (
                    iden
                    + ':'
                    + str(i)
                    + ':'
                    + str(idx)
                    + ' Conv_'
                    + str(conv_layer.kernel_size[0])
                    + 'x'
                    + str(conv_layer.kernel_size[1])
                    + '_s'
                    + str(conv_layer.stride[0])
                    + '_('
                    + str(conv_layer.in_channels)
                    + ', '
                    + str(conv_layer.out_channels)
                    + ')\n'
                )
            return p_str

        def get_layer_from_tree(self, internal_idx, net=None):
            def u(module, k):
                if len(k) == 1:
                    return list(module.named_children())[k[0]][1]
                else:
                    kk = [k[0]]
                    k = k[1:]
                    return u(list(module.named_children())[kk[0]][1], k)

            if net is None:
                net = self.net
            return u(net, self._tree[internal_idx])

        def forward(self, x):
            return self.net(x)

        def prune(self, layer, filter, verbose=True):
            if verbose:
                print(
                    'Pruning config:' + 'CBCBC',
                    '- conv:out,bn:elem,conv:inout,bn:elem,conv:in at layer=' + str(layer) + ' location=' + str(filter),
                )
            idx, iden = self._idx[layer]
            self.prune_conv_layer(self.get_layer_from_tree(idx), filter, 'out')
            self.prune_batchnorm_layer(self.get_layer_from_tree(idx + 1), filter)
            self.prune_conv_layer(self.get_layer_from_tree(idx + 4), filter, 'inout')
            self.prune_batchnorm_layer(self.get_layer_from_tree(idx + 5), filter)
            self.prune_conv_layer(self.get_layer_from_tree(idx + 7), filter, 'in')

            self.net.cuda()
            self._pruned_state.append((layer, filter))

    class _ResNet_(nn.Module, _Agnostic_):
        class B1(nn.Module):
            def __init__(self, f, dummy_var=False, with_conv_multipliers=False):
                super().__init__()
                if with_conv_multipliers:
                    Conv2d = Conv2dWithMultiplier
                else:
                    Conv2d = nn.Conv2d
                self.b1 = nn.Sequential(
                    Conv2d(f, f, 3, 1, 1), nn.BatchNorm2d(f), nn.ReLU(), Conv2d(f, f, 3, 1, 1), nn.BatchNorm2d(f)
                )

            def forward(self, x):
                return F.relu(self.b1(x) + x)

        class B3(nn.Module):
            def __init__(self, f, first_block=False, with_conv_multipliers=False):
                super().__init__()
                if with_conv_multipliers:
                    Conv2d = Conv2dWithMultiplier
                else:
                    Conv2d = nn.Conv2d
                if first_block:
                    fin, fout, s = f // 2, f, 2
                else:
                    fin, fout, s = f, f, 1
                self.b3 = nn.Sequential(
                    Conv2d(fin, fout, 3, s, 1),
                    nn.BatchNorm2d(fout),
                    nn.ReLU(),
                    Conv2d(fout, fout, 3, 1, 1),
                    nn.BatchNorm2d(fout),
                )
                self.b3_skip = nn.Sequential(Conv2d(fin, fout, 3, s, 1), nn.BatchNorm2d(fout))

            def forward(self, x):
                return F.relu(self.b3(x) + self.b3_skip(x))

        class B4(nn.Module):
            class PaddedAdd(nn.Module):
                def __init__(self, first_block=False):
                    super().__init__()
                    self.first_block = first_block

                def forward(self, x, y):
                    if self.first_block:
                        x = F.max_pool2d(x, 2)

                    if x.shape[1] < y.shape[1]:
                        x = torch.cat(
                            (x, torch.zeros(x.shape[0], y.shape[1] - x.shape[1], x.shape[2], x.shape[3]).cuda()), dim=1
                        )
                    elif x.shape[1] > y.shape[1]:
                        y = torch.cat(
                            (y, torch.zeros(y.shape[0], x.shape[1] - y.shape[1], y.shape[2], y.shape[3]).cuda()), dim=1
                        )
                    else:
                        pass
                    return x + y

            class Identity(nn.Module):
                def __init__(self, f, first_block=False):
                    super().__init__()
                    self.first_block = first_block
                    self.mat = torch.eye(n=f).cuda()

                def forward(self, x):
                    if self.first_block:
                        x = F.max_pool2d(x, 2)

            def __init__(self, f, first_block=False, with_conv_multipliers=False):
                super().__init__()
                if first_block:
                    fin, fout, s = f // 2, f, 2
                else:
                    fin, fout, s = f, f, 1
                if with_conv_multipliers:
                    Conv2d = Conv2dWithMultiplier
                else:
                    Conv2d = nn.Conv2d
                self.b4 = nn.Sequential(
                    Conv2d(fin, fout, 3, s, 1),
                    nn.BatchNorm2d(fout),
                    nn.ReLU(),
                    Conv2d(fout, fout, 3, 1, 1),
                    nn.BatchNorm2d(fout),
                )
                self.b4_add = self.PaddedAdd(first_block)

            def forward(self, x):
                return F.relu(self.b4_add(x, self.b4(x)))

        class Classifier(nn.Module):
            def __init__(self, num_class):
                super().__init__()
                if num_class is None:
                    num_class = 10
                self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
                self.classifier = nn.Linear(8 * 8, num_class)

            def forward(self, x):
                x = self.gap(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x

        def B_x(self, x, n, f, with_conv_multipliers=False):
            Bx = eval('self.B' + str(x))
            super_block = nn.Sequential(Bx(f, True, with_conv_multipliers=with_conv_multipliers))
            for i in range(n - 1):
                super_block.add_module(str(i + 1), Bx(f, with_conv_multipliers=with_conv_multipliers))
            return super_block

        def __init__(
            self,
            num_layers,
            num_class,
            version=None,
            pretrained=False,
            first_block=False,
            with_conv_multipliers=False,
            phase_shift_mode=None,
            phase_shift_fn=None,
        ):
            super().__init__()
            assert num_layers in [
                18,
                34,
                50,
                101,
                152,
                20,
                32,
                44,
                56,
                110,
            ], 'num_layers must be [18, 34, 50, 101, 152, 20, 32, 44, 56, 110]'

            self._global_step = 0
            self._checkpoint = './weights/resnet' + str(num_layers) + '.ckpt'
            self._pruned_state = []
            self._idx, self._lin_idx = [], []
            self._tree = [[], []]
            self._optim = None
            self._loss_fn = None
            self._phase_shift_fn = phase_shift_fn
            self._batch_multiplier = None
            self._num_class = num_class or 1000

            # For ImageNet
            if num_layers in [18, 34, 50, 101, 152]:
                self.resnet = eval(
                    'torchvision.models.resnet' + str(num_layers) + '(pretrained=' + str(pretrained) + ')'
                )
                if not pretrained:
                    lin = (self.resnet.fc.in_features, self._num_class)
                    self.resnet.fc = torch.nn.Linear(lin[0], lin[1])
                if pretrained and self._num_class != 1000:
                    raise KeyError('pretrained=True only available when num_class=1000')
                self.resnet = nn.DataParallel(self.resnet)
                self._num_class, self._num_linear_units = 1000, 512
                self.modules = list(self.modules())
                module_type = [type(m).__name__ for m in self.modules]
                i = 0
                while i < len(self.modules):
                    m_type = module_type[i]
                    if m_type == 'Sequential' and module_type[i + 1] == 'Bottleneck':
                        self._idx.append((i + 2, 'C'))
                        self._idx.append((i + 4, 'C'))
                        if version == 'A' and num_layers in [50, 101, 152]:
                            if self.modules[i + 4].stride == (2, 2):
                                self.modules[i + 2].stride = (2, 2)
                                self.modules[i + 4].stride = (1, 1)
                        i += 12
                    elif m_type == 'Bottleneck':
                        self._idx.append((i + 1, 'C'))
                        self._idx.append((i + 3, 'C'))
                        i += 8
                    elif m_type == 'BasicBlock':
                        self._idx.append((i + 1, 'C'))
                        i += 6
                    else:
                        i += 1

            # For CIFAR10
            if with_conv_multipliers:
                Conv2d = Conv2dWithMultiplier
            else:
                Conv2d = nn.Conv2d
            resnet_n = (num_layers - 2) // 6
            if num_layers in [20, 32, 44, 56, 110]:
                if phase_shift_mode == 1 or phase_shift_mode == 3:
                    first_conv = Conv2dWithPhaseShift(
                        in_channels=3,
                        out_channels=4,
                        kernel_size=3,
                        phase_shift_fn=self._phase_shift_fn,
                        stride=1,
                        padding=1,
                    )
                else:
                    first_conv = Conv2d(3, 16, 3, 1, 1)
                self.resnet = nn.Sequential(
                    first_conv,
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    self.B_x(x=1, n=resnet_n, f=16, with_conv_multipliers=with_conv_multipliers),
                    self.B_x(x=4, n=resnet_n, f=32, with_conv_multipliers=with_conv_multipliers),
                    self.B_x(x=4, n=resnet_n, f=64, with_conv_multipliers=with_conv_multipliers),
                    self.Classifier(self._num_class),
                )
                if phase_shift_mode == 2 or phase_shift_mode == 3:
                    last_conv_parent, last_conv_token = get_all_parent_layers(self.resnet, nn.Conv2d)[-1]
                    last_conv = last_conv_parent[int(last_conv_token)]
                    setattr(
                        last_conv_parent,
                        last_conv_token,
                        Conv2dWithPhaseShift(
                            in_channels=last_conv.in_channels,
                            out_channels=last_conv.out_channels // 4,
                            kernel_size=last_conv.kernel_size,
                            phase_shift_fn=self._phase_shift_fn,
                            stride=last_conv.stride,
                            padding=last_conv.padding,
                        ),
                    )
                self.resnet = nn.DataParallel(self.resnet)
                self._num_class, self._num_linear_units = 10, 64
                self.modules = list(self.modules())
                module_type = [type(m).__name__ for m in self.modules]
                i = 0
                while i < len(self.modules):
                    m_type = module_type[i]
                    if m_type == 'B1':
                        self._idx.append((i + 2, 'C'))
                        i += 7
                    elif m_type == 'B3':
                        self._idx.append((i + 2, 'C'))
                        self._idx.append((i + 5, 'D'))
                        self._idx.append((i + 8, 'S'))
                        i += 10
                    elif m_type == 'B4':
                        self._idx.append((i + 2, 'C'))
                        i += 7
                    else:
                        i += 1

            self.init_state = [copy.deepcopy(self.resnet)]

            self._super_string = super.__str__(self)
            super_string_split = self._super_string.split('\n')
            for string in super_string_split[1:-2]:
                if string[-1] == '(':
                    self._tree.append(self._tree[-1].copy())
                    self._tree[-1].append(0)
                elif string[-2:] == ' )':
                    self._tree[-1].pop(-1)
                    self._tree[-1][-1] += 1
                else:
                    self._tree.append(self._tree[-1].copy())
                    self._tree[-1][-1] += 1

            for i, m in enumerate(self.modules):
                if isinstance(m, nn.Linear):
                    self._lin_idx.append((i, 'L'))

            self.modules_copy = self.modules
            self.cuda()

        def __str__(self):
            p_str = super().__str__()
            p_str += '\n\n\n----------------------\nIndex wise Module List\n----------------------\n\n'
            p_str += '\n'.join([str(i) + ' ' + type(m).__name__ for i, m in enumerate(self.modules)])
            p_str += '\n\n\n----------------\nPrunable Indices\n----------------\n\n'
            p_str += 'Identifier:ExternalIndex:InternalIndex LayerInfo\n\n'
            for i, (idx, iden) in enumerate(self._idx):
                conv_layer = self.modules[idx]
                p_str += (
                    iden
                    + ':'
                    + str(i)
                    + ':'
                    + str(idx)
                    + ' Conv_'
                    + str(conv_layer.kernel_size[0])
                    + 'x'
                    + str(conv_layer.kernel_size[1])
                    + '_s'
                    + str(conv_layer.stride[0])
                    + '_('
                    + str(conv_layer.in_channels)
                    + ', '
                    + str(conv_layer.out_channels)
                    + ')\n'
                )
            return p_str

        def get_layer_from_tree(self, internal_idx, net=None):
            def u(module, k):
                if len(k) == 1:
                    return list(module.named_children())[k[0]][1]
                else:
                    kk = [k[0]]
                    k = k[1:]
                    return u(list(module.named_children())[kk[0]][1], k)

            if net is None:
                net = self.resnet
            return u(net, self._tree[internal_idx])

        def forward(self, x):
            return self.resnet(x)

        def prune(self, layer, filter, verbose=True):
            first_idx, first_iden = self._idx[layer]
            second_idx, second_iden = first_idx + 1, 'B'
            if first_iden == 'C':
                third_idx, third_iden = first_idx + 2, 'C'
                if not isinstance(self.get_layer_from_tree(third_idx), nn.Conv2d):
                    third_idx += 1
                pruning_config = first_iden + second_iden + third_iden
            elif first_iden == 'D':
                third_idx, third_iden = self._idx[layer + 1]
                fourth_idx, fourth_iden = third_idx + 1, 'B'
                try:
                    fifth_idx, fifth_iden = self._idx[layer + 2]
                    sixth_idx, sixth_iden = self._idx[layer + 4]
                    pruning_config = first_iden + second_iden + third_iden + fourth_iden + fifth_iden + sixth_iden
                except:
                    fifth_idx, fifth_iden = self._lin_idx[0]
                    pruning_config = first_iden + second_iden + third_iden + fourth_iden + fifth_iden
            elif first_iden == 'S':
                third_idx, third_iden = self._idx[layer - 1]
                fourth_idx, fourth_iden = third_idx + 1, 'B'
                try:
                    fifth_idx, fifth_iden = self._idx[layer + 1]
                    sixth_idx, sixth_iden = self._idx[layer + 3]
                    pruning_config = first_iden + second_iden + third_iden + fourth_iden + fifth_iden + sixth_iden
                except:
                    fifth_idx, fifth_iden = self._lin_idx[0]
                    pruning_config = first_iden + second_iden + third_iden + fourth_iden + fifth_iden

            assert pruning_config in [
                'CBC',
                'DBSBCS',
                'DBSBL',
                'SBDBCS',
                'SBDBL',
            ], 'Error: No suitable pruning config found.'

            if pruning_config == 'CBC':
                if verbose:
                    print(
                        'Pruning config:' + pruning_config,
                        '- conv:out,bn:elem,conv:in at layer=' + str(layer) + ' location=' + str(filter),
                    )
                self.prune_conv_layer(self.get_layer_from_tree(first_idx), filter, 'out')
                self.prune_batchnorm_layer(self.get_layer_from_tree(second_idx), filter)
                self.prune_conv_layer(self.get_layer_from_tree(third_idx), filter, 'in')
            elif pruning_config == 'DBSBCS' or pruning_config == 'SBDBCS':
                if verbose:
                    print(
                        'Pruning config:' + pruning_config,
                        '- conv:out,bn:elem,conv:out,bn:elem,conv:in,conv:in at location=' + str(filter),
                    )
                self.prune_conv_layer(self.get_layer_from_tree(first_idx), filter, 'out')
                self.prune_batchnorm_layer(self.get_layer_from_tree(second_idx), filter)
                self.prune_conv_layer(self.get_layer_from_tree(third_idx), filter, 'out')
                self.prune_batchnorm_layer(self.get_layer_from_tree(fourth_idx), filter)
                self.prune_conv_layer(self.get_layer_from_tree(fifth_idx), filter, 'in')
                self.prune_conv_layer(self.get_layer_from_tree(sixth_idx), filter, 'in')
            elif pruning_config == 'DBSBL' or pruning_config == 'SBDBL':
                if verbose:
                    print(
                        'Pruning config:' + pruning_config,
                        '- conv:out,bn:elem,conv:out,bn:elem,lin:col at location=' + str(filter),
                    )
                self.prune_conv_layer(self.get_layer_from_tree(first_idx), filter, 'out')
                self.prune_batchnorm_layer(self.get_layer_from_tree(second_idx), filter)
                self.prune_conv_layer(self.get_layer_from_tree(third_idx), filter, 'out')
                self.prune_batchnorm_layer(self.get_layer_from_tree(fourth_idx), filter)
                self.prune_linear_layer(self.get_layer_from_tree(fifth_idx), filter, 'col')

            self.resnet.cuda()
            self._pruned_state.append((layer, filter))

        def set_layer_multiplier(self, layer, multiplier_array):
            idx, iden = self._idx[layer]
            multiplier_array = torch.from_numpy(multiplier_array)
            with torch.no_grad():
                self.modules[idx].multiplier[0, :, 0, 0] = multiplier_array

        def get_layer_multiplier(self, layer):
            idx, iden = self._idx[layer]
            with torch.no_grad():
                return self.modules[idx].multiplier[0, :, 0, 0].cpu().numpy()

        def zero_out_filter(self, layer, filter):
            required_layer = self.get_layer_from_tree(self._idx[layer][0])
            with torch.no_grad():
                required_layer.weight[filter] = 0
                try:
                    required_layer.bias[filter] = 0
                except:
                    pass

        def init_filter(self, layer, filter):
            required_layer = self.get_layer_from_tree(self._idx[layer][0])
            required_layer_init = self.get_layer_from_tree(self._idx[layer][0], net=self.init_state[0])
            with torch.no_grad():
                required_layer.weight[filter] = required_layer_init.weight[filter]
                required_layer.bias[filter] = required_layer_init.bias[filter]

        def freeze_filter(self, layer, filter):
            required_layer = self.get_layer_from_tree(self._idx[layer][0])
            with torch.no_grad():
                required_layer.weight[filter].requires_grad = False

        def watch_filter(self, layer, filter=None, net=None):
            required_layer = self.get_layer_from_tree(self._idx[layer][0], net)
            with torch.no_grad():
                if filter is not None:
                    print(required_layer.weight[filter].abs().sum())
                    try:
                        print(required_layer.bias[filter])
                    except:
                        pass
                else:
                    print(required_layer.weight.abs().sum(dim=(1, 2, 3)))
                    try:
                        print(required_layer.bias)
                    except:
                        pass

    class _VGG_(nn.Module, _Agnostic_):
        def __init__(self, num_layers, num_transition_shape, num_linear_units, num_class, pretrained=False):
            super().__init__()

            assert num_layers in [11, 13, 16, 19], 'VGG num_layers != [11, 13, 16, 19]'

            self._global_step = 0
            self._checkpoint = './weights/vgg' + str(num_layers) + '-' + str(num_class) + '.ckpt'
            self._pruned_state = []
            self._optim = None
            self._idx = []
            self._prune_linear_units = num_transition_shape
            self._num_class = num_class
            self._collected_loss, self._collected_metrics = [], [[0, 0, 0]]
            self._loss_fn = None
            self._batch_multiplier = None

            self.modules_copy = list(self.modules())
            self._fe = eval(
                'torchvision.models.vgg' + str(num_layers) + '_bn(' + 'pretrained=' + str(pretrained) + ').features'
            )

            self._c = nn.Sequential(
                nn.Linear(num_transition_shape * 512, num_linear_units),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(num_linear_units, num_linear_units),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(num_linear_units, num_class),
            )

            for i, m in enumerate(self._fe.modules()):
                if isinstance(m, nn.Conv2d):
                    self._idx.append((i - 1, 'C'))
            for i, m in enumerate(self._c.children()):
                if isinstance(m, nn.Linear):
                    self._idx.append((i + len(self._fe), 'L'))

        def __str__(self):
            conv_modules = self.__get_modules__(children='conv')
            linear_modules = self.__get_modules__(children='linear')
            p_str = super.__str__(self) + '\n\nPrunable Indices\n\n'
            for i, (idx, iden) in enumerate(self._idx):
                if iden == 'C':
                    layer = conv_modules[str(idx)]
                    p_str += str(i) + ': Convolution(' + str(layer.in_channels) + ', ' + str(layer.out_channels) + ')\n'
                if iden == 'L':
                    layer = linear_modules[str(idx - len(self._fe))]
                    p_str += str(i) + ': Linear(' + str(layer.in_features) + ', ' + str(layer.out_features) + ')\n'
            return p_str

        def __get_modules__(self, children='conv'):
            if children == 'conv':
                return list(self.children())[0]._modules
            elif children == 'linear':
                return list(self.children())[1]._modules
            else:
                return None

        def forward(self, x, layer=None, layer_type='conv'):
            if layer is not None:
                idx, iden = self._idx[layer]
                if iden == 'C':
                    if layer_type == 'conv':
                        with torch.no_grad():
                            return self._fe[: idx + 1](x)
                    elif layer_type == 'bn':
                        with torch.no_grad():
                            return self._fe[: idx + 2](x)
                    elif layer_type == 'relu':
                        with torch.no_grad():
                            return self._fe[: idx + 3](x)
                elif iden == 'L':
                    with torch.no_grad():
                        x = self._fe(x)
                        x = x.view(x.size(0), -1)
                        return self._c[: (idx - len(self._fe) + 1)](x)
            else:
                x = self._fe(x)
                x = x.view(x.size(0), -1)
                return self._c(x)

        def max_filters(self, layer):
            idx, iden = self._idx[layer]
            if iden == 'C':
                conv_modules = self.__get_modules__(children='conv')
                return conv_modules[str(idx)].out_channels
            elif iden == 'L':
                linear_modules = self.__get_modules__(children='linear')
                return linear_modules[str(idx - len(self._fe))].out_features

        def get_weights(self, layer, filter=None, grad=False):
            idx, iden = self._idx[layer]

            if iden == 'C':
                conv_modules = self.__get_modules__(children='conv')
                conv_weight = conv_modules[str(idx)].weight.data.clone()
                conv_bias = conv_modules[str(idx)].bias.data.clone()

                if grad:
                    conv_weight = conv_modules[str(idx)].weight.grad.data.clone()
                    conv_bias = conv_modules[str(idx)].bias.grad.data.clone()

                if filter is not None:
                    conv_weight = conv_weight[filter : filter + 1]
                    conv_bias = conv_bias[filter : filter + 1]

                return conv_weight.cpu().numpy(), conv_bias.cpu().numpy()

            elif iden == 'L':
                linear_modules = self.__get_modules__(children='linear')
                linear_weight = linear_modules[str(idx - len(self._fe))].weight.data.clone()
                linear_bias = linear_modules[str(idx - len(self._fe))].bias.data.clone()

                if grad:
                    linear_weight = linear_modules[str(idx - len(self._fe))].weight.grad.data.clone()
                    linear_bias = linear_modules[str(idx - len(self._fe))].bias.grad.data.clone()

                if filter is not None:
                    linear_weight = linear_weight[filter : filter + 1, ...]
                    linear_bias = linear_bias[filter : filter + 1, ...]

                return linear_weight.cpu().numpy(), linear_bias.cpu().numpy()

        def prune(self, layer, filter, verbose=True):
            conv_modules = self.__get_modules__(children='conv')
            linear_modules = self.__get_modules__(children='linear')

            first_idx, first_iden = self._idx[layer]
            if first_iden == 'C':
                second_idx, second_iden = self._idx[layer][0] + 1, 'B'
                third_idx, third_iden = self._idx[layer + 1]
                pruning_config = first_iden + second_iden + third_iden
            elif first_iden == 'L':
                second_idx, second_iden = self._idx[layer + 1]
                pruning_config = first_iden + second_iden

            if pruning_config == 'CBC':
                self.prune_conv_layer(conv_modules[str(first_idx)], filter, 'out')
                self.prune_batchnorm_layer(conv_modules[str(second_idx)], filter)
                self.prune_conv_layer(conv_modules[str(third_idx)], filter, 'in')
                if verbose:
                    print(
                        'Pruning conv(out):fil='
                        + str(filter)
                        + ', bn:elem='
                        + str(filter)
                        + ', conv(in):idx='
                        + str(filter)
                    )
            elif pruning_config == 'CBL':
                self.prune_conv_layer(conv_modules[str(first_idx)], filter, 'out')
                self.prune_batchnorm_layer(conv_modules[str(second_idx)], filter)
                self.prune_linear_layer(
                    linear_modules[str(third_idx - len(self._fe))], filter, 'col', self._prune_linear_units
                )
                if verbose:
                    print(
                        'Pruning conv(out):fil='
                        + str(filter)
                        + ', bn:elem='
                        + str(filter)
                        + ', linear(col):idx='
                        + str(filter)
                    )
            elif pruning_config == 'LL':
                self.prune_linear_layer(linear_modules[str(first_idx - len(self._fe))], filter, 'row')
                self.prune_linear_layer(linear_modules[str(second_idx - len(self._fe))], filter, 'col')
                if verbose:
                    print(
                        'Pruning linear(row):idx='
                        + str(filter)
                        + ', bn:elem='
                        + str(filter)
                        + ', linear(col):idx='
                        + str(filter)
                    )

            self._pruned_state.append((layer, filter))


class Conv2dWithMultiplier(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self.register_buffer('multiplier', torch.ones(1, self.out_channels, 1, 1))

    def forward(self, x):
        x = super().forward(x)
        x = x.mul(self.multiplier)
        return x

    def __str__(self):
        ss = super().__str__()
        return ss[:6] + 'WithMultiplier' + ss[6:]


class Conv2dWithPhaseShift(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        phase_shift_fn,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

        assert phase_shift_fn is not None, 'You forgot to attach phase shift fn'
        self._phase_shift_fn = phase_shift_fn

    def forward(self, x):
        return self._phase_shift_fn(x, self.weight, self.bias, self.stride, self.padding)

    def __str__(self):
        ss = super().__str__()
        return ss[:6] + 'WithPhaseShift' + ss[6:]


def get_all_parent_layers(net, type):
    layers = []

    for name, l in net.named_modules():
        if isinstance(l, type):
            tokens = name.strip().split('.')

            layer = net
            for t in tokens[:-1]:
                if not t.isnumeric():
                    layer = getattr(layer, t)
                else:
                    layer = layer[int(t)]

            layers.append([layer, tokens[-1]])

    return layers


class Datasets:
    def __init__(self, dataset, train_batch_size, input_resize=None, eval_batch_size=512):
        self._dataset_list = ['MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet', 'TinyImageNet', 'FakeData', 'FakeDataHighRes']
        assert dataset in self._dataset_list, 'Dataset must be in ' + ' or '.join(self._dataset_list)

        if input_resize:
            print('Input resize command received:', input_resize)
            print('Input resize is only implemented for CIFAR datasets')

        self._train_batch_size = train_batch_size
        self._root = '/home/milton/Datasets'

        if dataset == 'MNIST':
            self._train_dataset = torchvision.datasets.MNIST(
                self._root,
                train=True,
                download=False,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.1306604762738429,), (0.30810780717887876,)),
                    ]
                ),
            )
            self._eval_dataset = torchvision.datasets.MNIST(
                self._root,
                train=False,
                download=False,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.1306604762738429,), (0.30810780717887876,)),
                    ]
                ),
            )
        elif dataset == 'CIFAR10':
            if not input_resize:
                input_resize = 32
            self._train_dataset = torchvision.datasets.CIFAR10(
                self._root,
                train=True,
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.Resize(input_resize),
                        torchvision.transforms.RandomCrop(input_resize, padding=4),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ]
                ),
            )
            self._eval_dataset = torchvision.datasets.CIFAR10(
                self._root,
                train=False,
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize(input_resize),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ]
                ),
            )
        elif dataset == 'CIFAR100':
            if not input_resize:
                input_resize = 32
            self._train_dataset = torchvision.datasets.CIFAR100(
                self._root,
                train=True,
                download=False,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.Resize(input_resize),
                        torchvision.transforms.RandomCrop(input_resize, padding=4),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            (0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)
                        ),
                    ]
                ),
            )
            self._eval_dataset = torchvision.datasets.CIFAR100(
                self._root,
                train=False,
                download=False,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize(input_resize),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            (0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)
                        ),
                    ]
                ),
            )
        elif dataset == 'ImageNet':
            self._train_dataset = torchvision.datasets.ImageNet(
                self._root,
                split='train',
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.RandomResizedCrop(224),
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ]
                ),
            )
            self._eval_dataset = torchvision.datasets.ImageNet(
                self._root,
                split='val',
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize(256),
                        torchvision.transforms.CenterCrop(224),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ]
                ),
            )
        elif dataset == 'TinyImageNet':
            id_dict = {}
            for i, line in enumerate(open(self._root + '/tiny-imagenet-200/wnids.txt', 'r')):
                id_dict[line.replace('\n', '')] = i

            class TrainTinyImageNetDataset(Dataset):
                def __init__(self, id_dict, root, transform=None):
                    self.filenames = glob.glob(root + "/tiny-imagenet-200/train/*/*/*.JPEG")
                    self.transform = transform
                    self.id_dict = id_dict

                def __len__(self):
                    return len(self.filenames)

                def __getitem__(self, idx):
                    img_path = self.filenames[idx]
                    image = read_image(img_path)
                    if image.shape[0] == 1:
                        image = read_image(img_path, ImageReadMode.RGB)
                    label = self.id_dict[img_path.split('/')[-3]]
                    if self.transform:
                        image = self.transform(image.type(torch.FloatTensor))
                    return image, label

            class TestTinyImageNetDataset(Dataset):
                def __init__(self, id_dict, root, transform=None):
                    self.filenames = glob.glob(root + "/tiny-imagenet-200/val/images/*.JPEG")
                    self.transform = transform
                    self.id_dict = id_dict
                    self.cls_dic = {}
                    for i, line in enumerate(open(root + '/tiny-imagenet-200/val/val_annotations.txt', 'r')):
                        a = line.split('\t')
                        img, cls_id = a[0], a[1]
                        self.cls_dic[img] = self.id_dict[cls_id]

                def __len__(self):
                    return len(self.filenames)

                def __getitem__(self, idx):
                    img_path = self.filenames[idx]
                    image = read_image(img_path)
                    if image.shape[0] == 1:
                        image = read_image(img_path, ImageReadMode.RGB)
                    label = self.cls_dic[img_path.split('/')[-1]]
                    if self.transform:
                        image = self.transform(image.type(torch.FloatTensor))
                    return image, label

            transform = torchvision.transforms.Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))
            self._train_dataset = TrainTinyImageNetDataset(id_dict=id_dict, root=self._root, transform=transform)
            self._eval_dataset = TestTinyImageNetDataset(id_dict=id_dict, root=self._root, transform=transform)
        elif dataset == 'FakeData':
            self._train_dataset = torchvision.datasets.FakeData(
                image_size=(3, 32, 32), transform=torchvision.transforms.ToTensor()
            )
            self._eval_dataset = torchvision.datasets.FakeData(
                image_size=(3, 32, 32), transform=torchvision.transforms.ToTensor()
            )
        elif dataset == 'FakeDataHighRes':
            self._train_dataset = torchvision.datasets.FakeData(
                image_size=(3, 224, 224), transform=torchvision.transforms.ToTensor()
            )
            self._eval_dataset = torchvision.datasets.FakeData(
                image_size=(3, 224, 224), transform=torchvision.transforms.ToTensor()
            )

        self.train_images = torch.utils.data.DataLoader(
            self._train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True
        )
        self.eval_images = torch.utils.data.DataLoader(
            self._eval_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True
        )

        self.num_train_images = self._train_dataset.__len__()
        self.num_eval_images = self._eval_dataset.__len__()


def one_hot(labels, num_class):
    return torch.cuda.FloatTensor(labels.size(0), num_class).zero_().scatter_(1, labels.unsqueeze(1), 1)
