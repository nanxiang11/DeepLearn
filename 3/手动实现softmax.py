import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from torch import nn

from util import Accumulator, accuracy, Animator, evaluate_accuracy


def get_dataloader_workers():
    """
    使用4个进程来读取数据
    :return:
    """
    return 4


# 拉取数据
def load_data_fashion_mnist(batch_size, resize=None):
    """
    下载数据集，将其加载在内存中
    :param batch_size:
    :param resize:
    :return:
    """
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))


"""
mnist_test: 这是要加载的数据集，即 Fashion-MNIST 的测试集。
batch_size: 每个批次的样本数量。
shuffle: 表示是否在每个 epoch（遍历数据集的一次完整迭代）之前打乱数据集的顺序。在测试集上通常不需要打乱顺序，因此设置为 False。
num_workers: 表示用于数据加载的子进程数量。
"""

# 划分迭代32次
train_iter, test_iter = load_data_fashion_mnist(256)

# 尝试打印一下
# print(next(iter(train_iter)))


# 初始化参数
w = torch.normal(0, 0.01, size=(784, 10), requires_grad=True)
b = torch.zeros(10, requires_grad=True)

# 回忆加法操作作
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


# print(X.sum(0, keepdim=True))
# print(X.sum(1, keepdim=True))


# 定义softmax函数
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


# 验证函数
# X = torch.normal(0, 1, (2, 5))
# X_prob = softmax(X)
# print(X)
# print(X_prob)


# 定义模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1, w.shape[0])), w) + b)


# 定义损失函数
def cross_entopy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


# 优化算法
trainer = d2l.sgd([w, b], 0.1, 256)


# 训练
def train_epoch_ch3(net, train_iter, loss, updater):
    """
    训练模型一轮
    :param net:是要训练的神经网络模型
    :param train_iter:是训练数据的数据迭代器，用于遍历训练数据集
    :param loss:是用于计算损失的损失函数
    :param updater:是用于更新模型参数的优化器
    :return:
    """
    if isinstance(net, torch.nn.Module):  # 用于检查一个对象是否属于指定的类（或类的子类）或数据类型。
        net.train()

    # 训练损失总和， 训练准确总和， 样本数
    metric = Accumulator(3)

    for X, y in train_iter:  # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):  # 用于检查一个对象是否属于指定的类（或类的子类）或数据类型。
            # 使用pytorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()  # 方法用于计算损失的平均值
            updater.step()
        else:
            # 使用定制（自定义）的优化器和损失函数
            l.sum().backward()
            updater(X.shape())
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """
    训练模型（）
    :param net:
    :param train_iter:
    :param test_iter:
    :param loss:
    :param num_epochs:
    :param updater:
    :return:
    """
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        trans_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, trans_metrics + (test_acc,))
        train_loss, train_acc = trans_metrics
        print(trans_metrics)
        # assert train_loss < 0.5, train_loss
        # assert train_acc <= 1 and train_acc > 0.7, train_acc
        # assert test_acc <= 1 and test_acc > 0.7, test_acc


num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entopy, num_epochs, trainer)
d2l.plt.show()


def predict_ch3(net, test_iter, n=6):
    """
    进行预测
    :param net:
    :param test_iter:
    :param n:
    :return:
    """
    global X, y
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + "\n" + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n]
    )
    d2l.plt.show()


predict_ch3(net, test_iter)
