import torch
import numpy as np

###dataset && dataloader
import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


###datasets
#自定义的dataset需要继承torch的Dataset类
#必须实现三个函数

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
                         #文件路径

        self.img_labels = pd.read_csv(annotations_file)  #读取文件
        self.img_dir = img_dir  #照片目录
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)  #返回样本大小

    def __getitem__(self, idx):  #通过idx返回一个样本
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

        image = read_image(img_path)  #读取图像

        label = self.img_labels.iloc[idx, 1] #读取标签

        if self.transform:
            image = self.transform(image) #对数据进行进一步预处理

        if self.target_transform:
            label = self.target_transform(label) #对label进行处理
        return image, label



###dataloader
#利用dataset构建batch，每个训练周期后对样本进行打乱（shuffle）

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))

