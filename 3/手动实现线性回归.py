import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
feature, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_dataset(data_arry, batch_size, is_train=True):
    """
    构造数据迭代器
    :param data_arry: 数据
    :param batch_size: 批量大小
    :param is_train: 是否训练
    :return: Data.DataLoader  数据加载器
    """
    dataSet = data.TensorDataset(*data_arry)
    return data.DataLoader(dataSet, batch_size, shuffle=is_train)


data_iter = load_dataset((feature, labels), 10)
"""
上述已经生成相关数据，可以打印出来看看
"""


# print(next(iter(data_iter)))


def linreg(X, w, b):
    """
    定义回归模型
    :param X:
    :param w:
    :param b:
    :return:
    """
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """
    均方损失
    :param y_hat:
    :param y:
    :return:
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def SGD(params, lr, batch_size):
    """
    小批量随机梯度下降
    :param params: 模型
    :param lr: 学习率
    :param batch_size: 批量大小
    :return:
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()  # 每次清零


w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

for epoch in range(10):
    for X, y in data_iter:
        l = squared_loss(linreg(X, w, b), y)
        l.sum().backward()
        SGD([w, b], 0.03, 10)
    with torch.no_grad():
        trian_l = squared_loss(linreg(feature, w, b), labels)
        print(f'epoch{epoch+1} loss:{float(trian_l.mean()):f}')


print(f'w的估计误差：{true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差：{true_b - b}')
