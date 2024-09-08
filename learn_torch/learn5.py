######learn5：数据集和预处理



import os
#创建数据集
os.makedirs(os.path.join('..', 'learn_data'), exist_ok=True) #创建文件夹
data_file = os.path.join('..', 'learn_data', 'house_tiny.csv')


with open(data_file, 'w') as f:        #打开文件（w权限）
    f.write('NumRooms,Alley,Price\n')  #列名
    f.write('NA,Pave,127500\n')        #每行表示一个样本数据
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')


#读取csv文件  pandas
import pandas as pd
data = pd.read_csv(data_file)
print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]  #将标签分离


#处理缺失的数据   fillna函数
inputs1 = inputs.fillna("----") #用指定值
print(inputs1)

inputs2 = inputs.ffill()  #用前一个替代
print(inputs2)

inputs3 = inputs.bfill()  #用后一个替代
print(inputs3)

inputs1 = inputs.fillna(3333, limit=2) #每列只填充2个
print(inputs1)

inputs = inputs.fillna(inputs.mean(numeric_only = True))#用平均值填充，关键字指只有数字计算均值
print(inputs)

###对于str的值或离散的值  可以单独给NaN分类
inputs = pd.get_dummies(inputs, dummy_na= True, dtype= int) #dummy_na 指给nan单独建立类
                                                #dtype=int 用0,1而不是true，false
print(inputs)

#转张量
import torch
x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x, y)


# import torch
# import numpy as np
# import os
# import pandas as pd
# from torchvision.io import read_image
# import torch
# from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
#
# x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
# y = torch.relu(x)
# plt.plot(x.detach(), y.detach(), color= 'r',)
# plt.ylabel('relu(x)')
# plt.show()
