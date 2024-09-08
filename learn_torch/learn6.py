###自动求导

import torch

#假设我们想对函数𝑦=2𝐱⊤𝐱关于列向量𝐱求导
x = torch.arange(4.0)
x.requires_grad_(True)  #指把梯度存储下来，通过 x.gard 访问

#以上等价于
x = torch.arange(4.0, requires_grad= True)

#y
y = 2 * torch.dot(x, x) #内积

#调用反向传播函数计算y关于x每个分量的梯度
y.backward()

print('1', x.grad == 4 * x)


###下面计算x的另一个新函数
x.grad.zero_()  #此时清除梯度(置零)，否则会默认累加

y = x.sum()
y.backward()
print('2', x.grad)

#另一个新函数
x.grad.zero_()
y = x * x  #此时y为向量

##注意在深度学习中，目的不是计算微分矩阵之和，而是批量中每个样本单独计算的偏导数之和
y.sum().backward()  #此时求和后是标量
print('3', x.grad)



#另一个新函数
x.grad.zero_()
y = x * x
u = y.detach()  #此时把y当做一个常数而不是关于x的导数赋给u
z = u * x  #u是常数

z.sum().backward()
print('4', x.grad, x.grad == u)

#此时y对x求导,y还是x的函数
x.grad.zero_()
y.sum().backward()

print('5', x.grad, x.grad == u, x.grad == 2 * x)

###对控制流也可以求导
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)  #size为空，一个标量

print(a)
d = f(a)
d.backward()

print('6', a.grad == d/a)