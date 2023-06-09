import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import load_data_fashion_mnist, train_ch6


net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.ReLU(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.ReLU(),
    nn.Linear(84, 10))


lr, num_epochs, batch_size = 0.2, 10, 256
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
train_ch6(net, train_iter, test_iter, num_epochs, lr)


plt.pause(0)


