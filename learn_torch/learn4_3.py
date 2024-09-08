#nn.module类
import torch
import torch.nn as nn
import torch.nn.functional as F


#module嵌套module
class Model_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        #二维卷积也是module

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

#apply（）
@torch.no_grad()   #修饰器，表示在下面函数中不需要梯度运算
def init_weights(m):  #参数初始化为1
    print(m, type(m))
    if type(m) == nn.Linear:
        m.weight.fill_(1.0)
        print(m.weight)

net = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 2))
net.apply(init_weights)  #调用apply

#load_state_dict 导入参数和buffer值

#require_grad_  设置是否梯度更新













