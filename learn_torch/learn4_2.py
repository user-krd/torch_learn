import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#nn 模型层分析

input_image = torch.rand(3,28,28) #假设是3张28*28图片
print("初始", input_image.size())

#nn.Flatten nn.扁平化  在部分连续维度上进行铺平
flatten = nn.Flatten() #实例化flatten，此处未传入参数：从第1维至-1维铺平
flat_image = flatten(input_image)
print("FLATTEN铺平后", flat_image.size())

#nn.Linear nn.线性层
layer1 = nn.Linear(in_features=28*28, out_features=20) #实例化一个线性层
hidden1 = layer1(flat_image)
#print(layer1.weight, layer1.bias)
print("线性层输出", hidden1.size())

#nn.ReLU 非线性激活
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)  #实例化该类，作用后所有负数变为0
print(f"After ReLU: {hidden1}")

#nn.Sequential 一种有序的容器，数据将会有序的经过其中的模块
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
) #实例化
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)  #传入数据
print("经过网络后", logits.size())

#nn.Softmax
softmax = nn.Softmax(dim=1) #softmax层
pred_probab = softmax(logits)
print("最终概率", pred_probab.size(), pred_probab)



