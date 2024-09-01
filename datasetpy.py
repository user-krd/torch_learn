from torch.utils.data import Dataset
import os
from PIL import Image

#读取单个图片
imagepath = "./dataset/train/ants/0013035.jpg"
img001 = Image.open(imagepath)
#print(img001.size)

#获取文件夹下所有文件的名称
datasetpath = "./dataset/train/ants"
imagelist = os.listdir(datasetpath)
#print(imagelist)




class mydata(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir#主文件夹位置
        self.label_dir = label_dir#标签文件夹名称
        self.path = os.path.join(self.root_dir,self.label_dir)#合体
        self.img_path = os.listdir(self.path)#获取所有文件名

    def __getitem__(self, idx):#获取每一个
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

#实例化
root_dir = "./dataset/train"
ants_label_dir = "ants"
ants_dataset = mydata(root_dir, ants_label_dir)

bees_label_dir = "bees"
bees_dataset = mydata(root_dir, bees_label_dir)

#整合为train
train_data = ants_dataset + bees_dataset
img,label= train_data[123]
img.show()
