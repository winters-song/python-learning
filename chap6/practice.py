import os

import matplotlib.pyplot as plt
import torch
from model import MLP
from d2l import load_data_fashion_mnist, train_ch6, try_gpu

net = MLP()

filename = 'mlp.params'
need_init = True

if os.path.exists( './' + filename):
  net.load_state_dict(torch.load(filename))
  net.eval()
  need_init = False
  print("data loaded")

batch_size, lr, num_epochs = 256, 0.005, 10
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu(), need_init)

torch.save(net.state_dict(), filename)
plt.pause(0)

