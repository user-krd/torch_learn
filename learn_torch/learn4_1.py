####learn4 : torch.nn 神经网络模型 neural networks

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#选择设备
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


#继承自nn的module

class NeuralNetwork(nn.Module):
    #定义所有模块
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),   #线性层mlp（前馈网络）
            nn.ReLU(),   #非线性激活函数
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    #前项运算
    def forward(self, x):
        x = self.flatten(x)   #维度展开铺平
        logits = self.linear_relu_stack(x)
        return logits

#调用在对应设备上
model = NeuralNetwork().to(device)
print(model)

#输入
X = torch.rand(1, 28, 28, device=device)  #随机张量
logits = model(X) #传入x，得到最终输出
pred_probab = nn.Softmax(dim=1)(logits)  #实例化softmax层，dim=1指第1维度上进行归一化
y_pred = pred_probab.argmax(1)  #概率的最大值为分类值
print(f"Predicted class: {y_pred}")



#模型参数parameter
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():  #调用的是nn.model的一个方法，layer也可以调用
    print(f"parameter: {name} | Size: {param.size()} | Values : {param[:2]} \n")
#注意： linear层有parameter，relu没有
