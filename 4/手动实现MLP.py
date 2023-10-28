import torch
from torch import nn
from d2l import torch as d2l

import util

# 获取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)         # 小数据量

# print(len(train_iter), len(test_iter))
# 235 40


# 初始化模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

w1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))

w2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [w1, b1, w2, b2]
# print(params)


def relu(X):
    """
    用于创建一个与输入张量 X 具有相同形状的张量，但所有元素的值都初始化为零。
    具体来说，这个函数会生成一个全零张量，其形状与输入张量 X 相同。
    :param X:
    :return:
    """
    a = torch.zeros_like(X)
    return torch.max(X, a)


# 定义模型
def net(X):
    X = X.reshape(-1, num_inputs)
    H = relu(X@w1 + b1)
    return (H@w2 + b2)


# 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')


# 训练
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
util.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2l.plt.show()

# 预测
util.predict_ch3(net, test_iter, 6)




