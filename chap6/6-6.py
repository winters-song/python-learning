import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import load_data_fashion_mnist, train_ch6, try_gpu

# LeNet（LeNet-5）
# net = nn.Sequential(
#     nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#     nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#     nn.Flatten(),
#     nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
#     nn.Linear(120, 84), nn.Sigmoid(),
#     nn.Linear(84, 10))


net = nn.Sequential(
  nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
  nn.MaxPool2d(kernel_size=2, stride=2),
  nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
  nn.MaxPool2d(kernel_size=2, stride=2),
  nn.Dropout(0.25),
  nn.Flatten(),
  nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
  nn.Dropout(0.5),
  nn.Linear(120, 84), nn.ReLU(),
  nn.Dropout(0.5),
  nn.Linear(84, 10))


batch_size, lr, num_epochs = 256, 0.005, 20
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())
plt.pause(0)


