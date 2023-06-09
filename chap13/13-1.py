# 数据增广
import torch

from torchvision import transforms
from matplotlib import pyplot as plt
from torch import nn
from d2l import load_cifar10, try_all_gpus, train_ch13, resnet18

batch_size, devices, net = 256, try_all_gpus(), resnet18(10, 3)


def init_weights(m):
  if type(m) in [nn.Linear, nn.Conv2d]:
    nn.init.xavier_uniform_(m.weight)


net.apply(init_weights)


def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
  train_iter = load_cifar10(True, train_augs, batch_size)
  test_iter = load_cifar10(False, test_augs, batch_size)
  loss = nn.CrossEntropyLoss(reduction="none")
  trainer = torch.optim.Adam(net.parameters(), lr=lr)
  train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)


train_augs = transforms.Compose([
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor()])

test_augs = transforms.Compose([
  transforms.ToTensor()])

train_with_data_aug(train_augs, test_augs, net)

plt.pause(0)
