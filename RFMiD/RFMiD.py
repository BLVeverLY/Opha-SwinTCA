#encoding=gbk
import torch.nn as nn
import pandas as pd
import torch
#print(torch.__version__)
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import pandas as pd
import cv2
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils import data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score,classification_report
import numpy as np
from torchvision import transforms as tfs
import torchvision

from PIL import Image
import random

# img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C

class SingleLabel(data.Dataset):
    def __init__(self,data_folder, label_path, transform = None):
        super().__init__()
        self.img_path = data_folder
        self.lp = pd.read_csv(label_path).values
        self.transform = transform

    def __len__(self):
        return len(self.lp)

    def __getitem__(self, idex):
        # label, img_name, = self.df[idex]
        for i in range(1,len(self.lp[idex])):
            if self.lp[idex][i] != 0:
                label = np.array(i-1)
                break  
        # label = int(self.lp[idex][1]) # "Disease_Risk"
        img_name = self.img_path + '/' + str(self.lp[idex][0]) + '.png'
        img = Image.open(img_name)
        img=transforms.Resize((224,224))(img)
        img=transforms.ToTensor()(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, img, label

class pix2pix(data.Dataset):
    def __init__(self,data_folder, label_path, transform = None):
        super().__init__()
        self.img_path = data_folder
        self.lp = pd.read_csv(label_path).values
        self.transform = transform

    def __len__(self):
        return len(self.lp)

    def __getitem__(self, idex):
        for i in range(1,len(self.lp[idex])):
            if self.lp[idex][i] != 0:
                label = np.array(i-1)
                break
        # label = int(self.lp[idex][1]) # "Disease_Risk"
        img_name1 = self.img_path + '/' + str(self.lp[idex][0]) + '_fake.png'
        img1 = Image.open(img_name1)
        img1=transforms.Resize((224,224))(img1)
        # img1 = cv2.imread(img_name1)[:, :, ::-1]
        # img1 = cv2.resize(img1, (224, 224))
        # img1 = img1.transpose(2, 0, 1)
        img1=transforms.ToTensor()(img1)
        img_name2 = self.img_path + '/' + str(self.lp[idex][0]) + '_real.png'
        img2 = Image.open(img_name2)
        img2=transforms.Resize((224,224))(img2)
        # img2 = cv2.imread(img_name2)[:, :, ::-1]
        # img2 = cv2.resize(img2, (224, 224))
        # img2 = img2.transpose(2, 0, 1)
        img2=transforms.ToTensor()(img2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label # OCT, fundus

class cyclegan(data.Dataset):
    def __init__(self,data_folder, label_path, transform = None):
        super().__init__()
        self.img_path = data_folder
        self.lp = pd.read_csv(label_path).values
        self.transform = transform

    def __len__(self):
        return len(self.lp)

    def __getitem__(self, idex):
        for i in range(1,len(self.lp[idex])):
            if self.lp[idex][i] != 0:
                label = np.array(i-1) 
                break  
        # label = int(self.lp[idex][1]) # "Disease_Risk"
        img_name1 = self.img_path + '/' + str(self.lp[idex][0]) + '_fake_B.png'
        img1 = Image.open(img_name1)
        img1=transforms.Resize((224,224))(img1)
        img1=transforms.ToTensor()(img1)
        img_name2 = self.img_path + '/' + str(self.lp[idex][0]) + '_real_A.png'
        img2 = Image.open(img_name2)
        img2=transforms.Resize((224,224))(img2)
        img2=transforms.ToTensor()(img2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label # OCT, fundus

class diffusion(data.Dataset):
    def __init__(self,data_folder, diffusion_csv, label_path, transform = None):
        super().__init__()
        self.img_path = data_folder
        self.lp = pd.read_csv(label_path).values
        self.transform = transform
        self.dc = pd.read_csv(diffusion_csv).values
        self.dict_of_list = {0:[],1:[],2:[],3:[],4:[],5:[]}
        for i in range(len(self.dc)):
            self.dict_of_list[self.dc[i][1]].append(self.dc[i][0])
        random.seed(49)

    def __len__(self):
        return len(self.lp)

    def __getitem__(self, idex):
        for i in range(1,len(self.lp[idex])):
            if self.lp[idex][i] != 0:
                label = i-1
                break
        # label = int(self.lp[idex][1]) # "Disease_Risk"
        img_name1 = self.img_path + '/' + str(self.lp[idex][0]) + '.png'
        img1 = Image.open(img_name1)
        img1=transforms.Resize((224,224))(img1)
        img1=transforms.ToTensor()(img1)
        
        img_name2 = random.choice(self.dict_of_list[label])
        img2 = Image.open(img_name2)
        img2=transforms.Resize((224,224))(img2)
        img2=transforms.ToTensor()(img2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, np.array(label) # OCT, fundus

    
class MultiLabel(data.Dataset):
    def __init__(self,data_folder, label_path, transform = None):
        super().__init__()
        self.img_path = data_folder
        self.lp = pd.read_csv(label_path).values
        self.transform = transform

    def __len__(self):
        return len(self.lp)

    def __getitem__(self, idex):
        # label, img_name, = self.df[idex]
        # print(self.lp[idex][2:])
        # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0] ## 45
        label = self.lp[idex][2:].astype(np.int) # label = int(self.lp[idex][2:])
        img_name = self.img_path + '/' + str(self.lp[idex][0]) + '.png'
        img = Image.open(img_name)
        img=transforms.Resize((224,224))(img)
        img=transforms.ToTensor()(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


if __name__ == '__main__':
    batch_size = 16
    train_data = SingleLabel(data_folder="/home/.../RFMiD_All_Classes_Dataset/1. Original Images/a. Training Set",
                        label_path="/home/.../RFMiD_All_Classes_Dataset/2. Groundtruths/a. RFMiD_Training_Labels.csv")