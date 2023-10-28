import torch
from torch import nn

# print(torch.device('cpu'))
# print(torch.device('cuda'))
# print(torch.device('cuda:1'))
#
# print(torch.cuda.device_count())

# GPU编号从0开始
def try_gpu(i=0):
    """
    如果存在，则返回gpu(i), 负责返回cpu
    :param i:
    :return:
    """
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """
    返回所有可用的GPU，如果没有GPU则返回[cpu(),]
    :return:
    """
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


# print(try_gpu())
# print(try_gpu(10))
# print(try_all_gpus())

# 默认情况下张量是存储在cpu是创建的
x = torch.tensor([1, 2, 3])
print(x.device)
# cpu

# 将数据存储在GPU上
x = torch.ones(2, 3, device=try_gpu())
'''
tensor([[1., 1., 1.],
        [1., 1., 1.]], device='cuda:0')
'''
print(x)

# 将张量创建在第二个GPU
# Y = torch.rand(2, 3, device=try_gpu(1))

# 张量复制
# Z = x.cuda(1)
# print(x)
# print(Z)
# print(Y + Z)
#
# print(Z.cuda(1) is Z)

# 神经网络与GPU
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())

print(net(x))
'''
tensor([[0.3482],
        [0.3482]], device='cuda:0', grad_fn=<AddmmBackward0>)
'''
print(net[0].weight.data.device)
# cuda:0

