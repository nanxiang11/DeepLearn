import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4).reshape(2, 2)
# print(x)
# torch.save(x, 'x-file')

x2 = torch.load('x-file')
# print(x2)
y = torch.zeros(4)

# 列表存储
torch.save([x, y], 'x-file')
x3, y3 = torch.load('x-file')
# print(x3, y3)


# 字典存储
mydict = {'x': x3, 'y': y3}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
# print(mydict2)


# 加载和保存模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


net = MLP()
x = torch.randn(size=(2, 20))
y = net(x)

print(y)


torch.save(net.state_dict(), "mlp.params")

clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())

y_clone = clone(x)
print(y == y)