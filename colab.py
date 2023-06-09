import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline

import time
import hashlib
import os
import tarfile
import zipfile
import requests


def try_gpu(i=0):
  """Return gpu(i) if exists, otherwise return cpu().

  Defined in :numref:`sec_use_gpu`"""
  if torch.cuda.device_count() >= i + 1:
    return torch.device(f'cuda:{i}')
  return torch.device('cpu')


class Timer:
  """Record multiple running times."""

  def __init__(self):
    """Defined in :numref:`subsec_linear_model`"""
    self.times = []
    self.start()

  def start(self):
    """Start the timer."""
    self.tik = time.time()

  def stop(self):
    """Stop the timer and record the time in a list."""
    self.times.append(time.time() - self.tik)
    return self.times[-1]

  def avg(self):
    """Return the average time."""
    return sum(self.times) / len(self.times)

  def sum(self):
    """Return the sum of time."""
    return sum(self.times)

  def cumsum(self):
    """Return the accumulated time."""
    return np.array(self.times).cumsum().tolist()


# 加载Fashion_MNIST数据集
def load_data_fashion_mnist(batch_size, resize=None):
  """下载Fashion-MNIST数据集，然后将其加载到内存中"""
  trans = [transforms.ToTensor()]
  if resize:
    trans.insert(0, transforms.Resize(resize))
  trans = transforms.Compose(trans)
  mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
  mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
  return (data.DataLoader(mnist_train, batch_size, shuffle=True),
          data.DataLoader(mnist_test, batch_size, shuffle=False))


# 创建迭代器
def load_array(data_arrays, batch_size, is_train=True):
  dataset = data.TensorDataset(*data_arrays)
  return data.DataLoader(dataset, batch_size, shuffle=is_train)


# 造数据
def synthetic_data(w, b, num_examples):
  """生成y=Xw+b+噪声  : 期望，标准差，size(样本量，权重数)"""
  X = torch.normal(0, 1, (num_examples, len(w)))
  y = torch.matmul(X, w) + b
  y += torch.normal(0, 0.01, y.shape)
  return X, y.reshape((-1, 1))


# 展示图片列表
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
  """绘制图像列表"""
  figsize = (num_cols * scale, num_rows * scale)
  plt.figure(figsize=figsize)
  _, axes = plt.subplots(num_rows, num_cols)
  axes = axes.flatten()
  for i, (ax, img) in enumerate(zip(axes, imgs)):
    if torch.is_tensor(img):
      # 图片张量
      ax.imshow(img.numpy())
    else:
      # PIL图片
      ax.imshow(img)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    if titles:
      ax.set_title(titles[i])
  return axes


def use_svg_display():
  """Use the svg format to display a plot in Jupyter.

  Defined in :numref:`sec_calculus`"""
  backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
  """Set the figure size for matplotlib.

  Defined in :numref:`sec_calculus`"""
  use_svg_display()
  plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
  """Set the axes for matplotlib.

  Defined in :numref:`sec_calculus`"""
  axes.set_xlabel(xlabel)
  axes.set_ylabel(ylabel)
  axes.set_xscale(xscale)
  axes.set_yscale(yscale)
  axes.set_xlim(xlim)
  axes.set_ylim(ylim)
  if legend:
    axes.legend(legend)
  axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
  """Plot data points.

  Defined in :numref:`sec_calculus`"""
  if legend is None:
    legend = []

  set_figsize(figsize)
  axes = axes if axes else plt.gca()

  # Return True if `X` (tensor or list) has 1 axis
  def has_one_axis(X):
    return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
            and not hasattr(X[0], "__len__"))

  if has_one_axis(X):
    X = [X]
  if Y is None:
    X, Y = [[]] * len(X), X
  elif has_one_axis(Y):
    Y = [Y]
  if len(X) != len(Y):
    X = X * len(Y)
  axes.cla()
  for x, y, fmt in zip(X, Y, fmts):
    if len(x):
      axes.plot(x, y, fmt)
    else:
      axes.plot(y, fmt)
  set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
  plt.show(block=False)


# 累加器（计算总损失）
class Accumulator:
  """在n个变量上累加"""

  def __init__(self, n):
    self.data = [0.0] * n

  def add(self, *args):
    self.data = [a + float(b) for a, b in zip(self.data, args)]

  def reset(self):
    self.data = [0.0] * len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]


# softmax
def accuracy(y_hat, y):
  """计算预测正确的数量"""
  if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
    y_hat = y_hat.argmax(axis=1)
  cmp = y_hat.type(y.dtype) == y
  return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
  """计算在指定数据集上模型的精度"""
  if isinstance(net, torch.nn.Module):
    net.eval()  # 将模型设置为评估模式
  metric = Accumulator(2)  # 正确预测数、预测总数
  with torch.no_grad():
    for X, y in data_iter:
      metric.add(accuracy(net(X), y), y.numel())
  return metric[0] / metric[1]


class Animator:
  """在动画中绘制数据"""

  def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
               ylim=None, xscale='linear', yscale='linear',
               fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
               figsize=(3.5, 2.5)):
    # 增量地绘制多条线
    if legend is None:
      legend = []

    plt.figure(figsize=figsize)
    self.fig, self.axes = plt.subplots(nrows, ncols)
    if nrows * ncols == 1:
      self.axes = [self.axes, ]

    def set_axes():
      # 使用lambda函数捕获参数
      plt.xlabel(xlabel)
      plt.ylabel(ylabel)
      plt.xlim(xlim)
      plt.ylim(ylim)
      plt.xscale(xscale)
      plt.yscale(yscale)
      plt.legend(legend)

    self.set_axes = set_axes
    self.X, self.Y, self.fmts = None, None, fmts

  def add(self, x, y):
    # 向图表中添加多个数据点
    if not hasattr(y, "__len__"):
      y = [y]
    n = len(y)
    if not hasattr(x, "__len__"):
      x = [x] * n
    if not self.X:
      self.X = [[] for _ in range(n)]
    if not self.Y:
      self.Y = [[] for _ in range(n)]
    for i, (a, b) in enumerate(zip(x, y)):
      if a is not None and b is not None:
        self.X[i].append(a)
        self.Y[i].append(b)
    self.axes[0].cla()
    for x, y, fmt in zip(self.X, self.Y, self.fmts):
      self.axes[0].plot(x, y, fmt)
    self.set_axes()


def get_fashion_mnist_labels(labels):
  """返回Fashion-MNIST数据集的文本标签"""
  text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
  return [text_labels[int(i)] for i in labels]


def linreg(X, w, b):
  """线性回归模型"""
  return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
  """均方损失"""
  return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
  """Minibatch stochastic gradient descent.

  Defined in :numref:`sec_linear_scratch`"""
  with torch.no_grad():
    for param in params:
      param -= lr * param.grad / batch_size
      param.grad.zero_()


def evaluate_loss(net, data_iter, loss):
  """评估给定数据集上模型的损失"""
  metric = Accumulator(2)  # 损失的总和,样本数量
  for X, y in data_iter:
    out = net(X)
    y = y.reshape(out.shape)
    l = loss(out, y)
    metric.add(l.sum(), l.numel())
  return metric[0] / metric[1]


DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'


def download(name, cache_dir=os.path.join('..', 'data')):
  """下载一个DATA_HUB中的文件，返回本地文件名"""
  assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
  url, sha1_hash = DATA_HUB[name]
  os.makedirs(cache_dir, exist_ok=True)
  fname = os.path.join(cache_dir, url.split('/')[-1])
  if os.path.exists(fname):
    sha1 = hashlib.sha1()
    with open(fname, 'rb') as f:
      while True:
        data = f.read(1048576)
        if not data:
          break
        sha1.update(data)
    if sha1.hexdigest() == sha1_hash:
      return fname  # 命中缓存
  print(f'正在从{url}下载{fname}...')
  r = requests.get(url, stream=True, verify=True)
  with open(fname, 'wb') as f:
    f.write(r.content)
  return fname


def download_extract(name, folder=None):
  """下载并解压zip/tar文件"""
  fname = download(name)
  base_dir = os.path.dirname(fname)
  data_dir, ext = os.path.splitext(fname)
  if ext == '.zip':
    fp = zipfile.ZipFile(fname, 'r')
  elif ext in ('.tar', '.gz'):
    fp = tarfile.open(fname, 'r')
  else:
    assert False, '只有zip/tar文件可以被解压缩'
  fp.extractall(base_dir)
  return os.path.join(base_dir, folder) if folder else data_dir


def download_all():
  """下载DATA_HUB中的所有文件"""
  for name in DATA_HUB:
    download(name)


DATA_HUB['kaggle_house_train'] = (
  DATA_URL + 'kaggle_house_pred_train.csv',
  '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
  DATA_URL + 'kaggle_house_pred_test.csv',
  'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')


def evaluate_accuracy_gpu(net, data_iter, device=None):
  """Compute the accuracy for a model on a dataset using a GPU.

  Defined in :numref:`sec_lenet`"""
  if isinstance(net, nn.Module):
    net.eval()  # Set the model to evaluation mode
    if not device:
      device = next(iter(net.parameters())).device
  # No. of correct predictions, no. of predictions
  metric = Accumulator(2)

  with torch.no_grad():
    for X, y in data_iter:
      if isinstance(X, list):
        # Required for BERT Fine-tuning (to be covered later)
        X = [x.to(device) for x in X]
      else:
        X = X.to(device)
      y = y.to(device)
      metric.add(accuracy(net(X), y), y.numel())
  return metric[0] / metric[1]


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device=None, need_init=True):
  def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
      nn.init.xavier_uniform_(m.weight)

  if need_init:
    net.apply(init_weights)

  if device:
    net.to(device)

  optimizer = torch.optim.SGD(net.parameters(), lr=lr)
  loss = nn.CrossEntropyLoss()
  animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                      legend=['train loss', 'train acc', 'test acc'])
  timer, num_batches = Timer(), len(train_iter)
  for epoch in range(num_epochs):
    # 训练损失之和，训练准确率之和，样本数
    metric = Accumulator(3)
    net.train()
    for i, (X, y) in enumerate(train_iter):
      timer.start()
      optimizer.zero_grad()
      X, y = X.to(device), y.to(device)
      y_hat = net(X)
      l = loss(y_hat, y)
      l.backward()
      optimizer.step()
      with torch.no_grad():
        metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
      timer.stop()
      train_l = metric[0] / metric[2]
      train_acc = metric[1] / metric[2]
      if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
        animator.add(epoch + (i + 1) / num_batches,
                     (train_l, train_acc, None))
    test_acc = evaluate_accuracy_gpu(net, test_iter)
    animator.add(epoch + 1, (None, None, test_acc))
    print(f'epoch {epoch + 1},'
          f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
  print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
        f'test acc {test_acc:.3f}')
  print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec ')
