import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

from torch import nn   # 神经网络


true_w = torch.tensor([2, -3.4])
true_b = 4.2
feature, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 绘图
f = feature[:, 1].numpy()
l = labels.numpy()
d2l.set_figsize()
d2l.plt.scatter(f, l, 1)
d2l.plt.show()




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
# temp = next(iter(data_iter))  # 迭代数据

# Sequential将多个层串联在一起。当给定输入数据时，该实例将数据传入第一层，然后将数据第一层输出作为第二层的输入
# Linear全连接层
net = nn.Sequential(nn.Linear(2, 1))

# net[0]为第一层
# weight.data访问权重  bias.data访问偏执
# w d normal_重写参数   b d fill_ 重写
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)


# 定义损失函数
loss = nn.MSELoss()     # MSELoss 也被称为平方L2范数 返回所有样本损失的平均值

# 随机梯度
# optim模块中实现了该算法的许多变体
trainer = torch.optim.SGD(net.parameters(), lr=0.03)


num_epochs = 10

for epoch in range(num_epochs):             # 不断的优化参数
    for X, y in data_iter:                  # 每次获取小批量数据
        l = loss(net(X), y)                 # 通过调用net(X)生成预测并计算损失l(向前传播）
        trainer.zero_grad()
        l.backward()                        # 通过反向传播来计算梯度
        trainer.step()                      # 通过优化器来更新模型参数
    l = loss(net(feature), labels)          # 计算每轮后的损失
    print(f'epoch{epoch+1}, loss{l:f}')     # 打印


# 评估模型

w = net[0].weight.data
b = net[0].bias.data
print("w估计误差-->", true_w - w.reshape(true_w.shape))
print("b估计误差-->", true_b - b)




