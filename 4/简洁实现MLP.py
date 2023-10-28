import torch
from torch import nn
from d2l import torch as d2l

import util


# 获取数据
train_iter, test_iter = d2l.load_data_fashion_mnist(256)

# 定义模型
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)


# 初始化
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)


batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

util.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()


util.predict_ch3(net, test_iter, 6)




