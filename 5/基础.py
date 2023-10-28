import torch
from torch import nn
from torch.nn import functional as F

X = torch.rand(2, 20)


# 自定义块
class MLP(nn.Module):
    # 用模型参数声明层这里，我们声明两个全连接层
    def __init__(self):
        # 调用MLP的父类Module的构造函数类执行必要的初始化
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    # 定义模型的向前传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模型块中定义
        return self.out(F.relu(self.hidden(X)))


# net = MLP()
# print(net(X))


# 在向前传播中定义一些自己定义的
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数，因此在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


# net = FixedHiddenMLP()
# print(net(X))

class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


# chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
# print(chimera(X))


# 参数管理
net = nn.Sequential(nn.Linear(4, 8),
                    nn.ReLU(), nn.Linear(8, 1))

X = torch.rand(size=(2, 4))


# print(net(X))
# print(net[2].state_dict())
# print(type(net[2].bias))
# print(net[2].bias)
# print(net[2].bias.data)
#
# # 判断梯度
# print(net[2].weight.grad == None)

# 一次访问所有参数
# print(*[(name, param.shape) for name, param in net[0].named_parameters()])
# print(*[(name, param.shape) for name, param in net.named_parameters()])
# print(net.state_dict()['2.bias'].data)


# 从嵌套块收集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block{i}', block1())
    return net


# rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
# print(rgnet(X))
# print(rgnet)


# 参数初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

# net.apply(init_normal)
# print(net[0].weight.data[0], net[0].bias.data[0])


def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
# net.apply(init_constant)
# print(net[0].weight.data[0], net[0].bias.data[0])

# 对不同层不同初始化
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

# net[0].apply(init_xavier)
# net[2].apply(init_42)
# print(net[0].weight.data[0])
# print(net[2].weight.data)


# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


# net.apply(my_init)
# net[0].weight[:2]


# 参数绑定
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))

net(X)
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])

