import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

# 多项式最大阶数
max_degree = 20
# 训练数据和测试数据的大小
n_train, n_test = 100, 100
true_w = np.zeros(max_degree)
# print(true_w)
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])
# 设置随机特征
features = np.random.normal(size=(n_train + n_test, 1))
# print(features)
# 打乱
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))

for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i+1)

# labels的维度
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]]


# 对模型进行训练和测试
def evaluate_loss(net, data_iter, loss):
    """
    评估给定数据集上模型的损失
    :param net:
    :param data_iter:
    :param loss:
    :return:
    """
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

