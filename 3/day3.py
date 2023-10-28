import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from torch import nn

from IPython import display  # 模块来显示各种媒体类型

from util import Accumulator, accuracy, Animator, evaluate_accuracy

# 使用 svg 格式在 Jupyter 中显示绘图
d2l.use_svg_display()

# 读取数据集
# 通过ToTensor实例将图像数据从PIL类型变换为32位浮点类型
# 并除以255使得所有像素的数值均为0-1
"""
FashionMNIST数据集由10个类别的图形组成，每个类别由训练数据集中的6000张图像和测试数据集中1000张图像组成，
因此，训练集和测试集分别包含60000和10000张图像注意测试集是拿来评估模型的
Fashion-MNIST 中包含的 10 个类别分别为 
t-shirt(T 恤)、trouser(裤子)、pullover(套衫)、dress (连衣裙)、coat (外套)、
sandal(凉鞋)、shirt(衬衫)、sneaker (运动鞋)、bag(包)和 ankle boot (短靴)。
"""
trans = transforms.ToTensor()  # 生成实例
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)


# 通道数为1 高度和宽度为28 torch.Size([1, 28, 28])

# 标签转文字
def get_fashion_mnist_labels(labels):
    """
    返回一个Fashion-MNIST数据集的文本标签
    labels： 这个其实是下标
    :return:
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
                   'sneaker', 'bag', 'ankle boot']

    return [text_labels[int(i)] for i in labels]


# 可视化样本
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """
    它使用Matplotlib库来创建一个图像网格，将图像列表中的图像按行和列排列，并可选择性地为每个图像添加标题。
    :param imgs: 包含要绘制的图像的列表
    :param num_rows:指定要绘制的图像行数
    :param num_cols:指定要绘制的图像列数
    :param titles:可选参数，包含与每个图像对应的标题的列表
    :param scale:缩放因子，用于控制绘图的尺寸
    :return:
    """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()  # 将这个多维数组展平为一个一维数组，以便您可以更轻松地遍历和操作每个子图，而不必关心它们的原始排列。
    for i, (ax, img) in enumerate(zip(axes, imgs)):  # ax 是一个子图对象，img 是要在子图上显示的图像。
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图像
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)  # 设置子图图像x, y的坐标轴不可见
        ax.axes.get_yaxis().set_visible(False)
        if titles:  # 如果有标签就加上标签
            ax.set_title(titles[i])
    return axes


# 抽取前面几个样本
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
d2l.plt.show()

# 读取小批量数据
batch_size = 256


def get_dataloader_workers():
    """
    使用4个进程来读取数据
    :return:
    """
    return 4


# 该函数就是为了整合上述所有代码
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

# 查看
# for X, y in train_iter:
#     print(X.shape, X.dtype, y.shape, y.dtype)
#     break


# torch不会隐式地调整输入的形状因此，我们在线性层前定义了展平层flatten来调整网络的输入形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_weights(m):
    """
    初始化权重
    :param m:
    :return:
    """
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


# 将自定义的初始化函数应用到模型的所有权重上
net.apply(init_weights)

# 定义损失函数
loss = nn.CrossEntropyLoss(reduction="none")


# 优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)


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
        animator.add(epoch + 1, trans_metrics + (test_acc, ))
        train_loss, train_acc = trans_metrics
        print(trans_metrics)
        # assert train_loss < 0.5, train_loss
        # assert train_acc <= 1 and train_acc > 0.7, train_acc
        # assert test_acc <= 1 and test_acc > 0.7, test_acc



num_epochs = 10
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
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

