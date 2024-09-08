###learn1 : tensor操作


import torch
import numpy as np

#创建tensor
a = torch.tensor(2)
print(type(a))
print(a)

#从列表创建
data = [ [[1,2,3],[2,3,4.]], [[12,22,32],[2,22,4.]]]
b = torch.tensor(data)
print(b,b.dtype)

#从numpy数组创建
data2 = np.random.normal((2,3))
c = torch.tensor(data2)
print(c, c.dtype)

#从另一个张量创建
d = torch.ones_like(b) #与b形状、dtype相同的全1向量
print(d)
d2 = torch.zeros_like(b)
d3 = torch.rand_like(b)
print(d3)

#随机生成
e = torch.rand((2,2,2,2))
print(e)
print(e.dtype)
e1 = torch.ones((2,2))
e2 = torch.zeros((1))


#属性
print(e.shape)
print(e.device) #默认情况下在cpu上创建

#在有gpu时移动到gpu运算
if torch.cuda.is_available():
    e = e.to('cuda')

#操作(100+种)api
#1 判断
print(torch.is_tensor(a))
print(torch.is_nonzero(e2))
print(torch.numel(e))  #元素数量

#2 创建

#遍历 arange
f1 = torch.arange(5, 14, 3)
print(f1)

for i in torch.arange(10):
    print("epoch:", i)

#torch.eye

#torch.full
f2 = torch.full((2,2),5)
print(f2)

#3
#cat 连接两个张量，要求被连接之外的维度形状相同
f3 = torch.cat([torch.rand([2,2,4]), torch.rand([2,3,4])], dim = 1)
#注意维度从零开始计算
print(f3)

#4 运算 按元素计算


x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x**y, torch.exp(x))

#如果形状不同但是维度相同，会通过广播机制变换形状
xa = torch.arange(3).reshape((3, 1))
yb = torch.arange(2).reshape((1, 2))
print(xa+yb)

#逻辑判断
print(x == y)

print(x.sum())

#5 区域 赋值、取值  同numpy数组

#关于内存,如何进行原地操作以减少内存开销
before = id(y)
y = y + x
print(id(y) == before)  #此时内存地址变化


before = id(y)
y += x
print(id(y) == before)  #这样内存不变


Z = torch.zeros_like(y)
print('id(Z):', id(Z))
Z[:] = x + y
print('id(Z):', id(Z))  #这样内存不变
