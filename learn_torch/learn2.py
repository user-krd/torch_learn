import torch
import numpy as np

#chunk 分割张量
a = torch.rand([3,2,])

print(a)
b1, b2 = torch.chunk(a, chunks=2, dim=0)

# print(b1, b2)

#gather 按索引取值，详见文档

#reshape 改变形状，元素顺序不变
a2 = torch.reshape(a, [2,3,1])
print(a2)
a3 = torch.reshape(a,[-1])
print(a3)

#scatter 类似gather
#scatter_add_

#split 划分,相比与chunk可以传入列表指定具体的划分

#squeeze 压缩，移除所有为1的维度

a22 = torch.squeeze(a2)
print(a22)

#stack 堆叠张量（会提高维度）

#take 按索引将张量看作一维顺序取值

#tile 复制拷贝

#transpose 转置
x = torch.rand([2,3,4,5])

x2 = x.transpose(2,1)

print(x2.shape)

#unbind 在给定的维度拆分张量
x3 = torch.unbind(torch.rand([4,3]), dim=1)

print(x3)

#unsqueeze 新增一个维度，元素数目不变

#where 判断并返回

#4
#随机种子
#manual_seed(seed) 设置种子

#伯努利采样
#bernoulli


#高斯分布
#normal
#randn


#均匀分布 rand、randint……

#随机组合 randperm









