import torch
from torch import nn


# 为了方便起见，我们定义一个计算卷积的函数
# 此函数初始化卷积层权重，并对输入和输出扩大或缩减相应的纬度
def comp_conv2d(conv2d, X):
    # 这里的（1， 1）这里表示批量大小和通道大小
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个纬度（批量大小和通道）
    return Y.reshape(Y.shape[2:])


conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
print(comp_conv2d(conv2d, X).shape)
# torch.Size([8, 8])


# 步幅
# 下面我们将高度和宽度步幅设置为2，从而将输入的高度和宽度减半
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)
# torch.Size([4, 4])
