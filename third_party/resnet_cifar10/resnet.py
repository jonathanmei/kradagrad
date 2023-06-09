# --- Note from Kradagrad authors ---
# Besides this note, this file a lightly modified version of ResNet for CIFAR-10/100 to allow softplus.
#     URL: https://github.com/akamaster/pytorch_resnet_cifar10
#     commit: d5489e8
#     accessed: 2022_11_23
# ----------- End of note -----------
'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m, batch_norm: bool=True):
    classname = m.__class__.__name__
    if batch_norm:
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight)
    else:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0.0)
            nn.init.xavier_uniform_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', activation='relu', batch_norm=True):
        super(BasicBlock, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) if batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if batch_norm else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes) if batch_norm else nn.Identity()
                )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out) if self.activation == 'relu' else F.softplus(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out) if self.activation == 'relu' else F.softplus(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, activation='relu', batch_norm=True):
        super(ResNet, self).__init__()
        self.activation = activation
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16) if batch_norm else nn.Identity()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, activation=activation, batch_norm=batch_norm)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, activation=activation, batch_norm=batch_norm)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, activation=activation, batch_norm=batch_norm)
        self.linear = nn.Linear(64, num_classes)

        _weights_init_fun = lambda x: _weights_init(x, batch_norm=batch_norm)
        self.apply(_weights_init_fun)

    def _make_layer(self, block, planes, num_blocks, stride, activation='relu', batch_norm=True):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, activation=activation, batch_norm=batch_norm))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out) if self.activation == 'relu' else F.softplus(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet_factory(size):
    def resnet(num_classes=10, activation='relu', batch_norm=True):
        return ResNet(BasicBlock, [size] * 3, num_classes=num_classes, activation=activation, batch_norm=batch_norm)
    return resnet

resnet20 = resnet_factory(3)
resnet32 = resnet_factory(5)
resnet44 = resnet_factory(7)
resnet56 = resnet_factory(9)
resnet110 = resnet_factory(18)
resnet1202 = resnet_factory(200)

def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
