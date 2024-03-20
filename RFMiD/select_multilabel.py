import torch
import pandas as pd
#print(torch.__version__)
import torchvision.transforms as transforms
import pandas as pd
from torch.utils import data
import numpy as np

from PIL import Image
import csv

# train_dataset = torchvision.datasets.ImageFolder('/home/ssd1/ljy/test/'
#                             ,transform=transforms.Compose([
#                                                             transforms.Resize((224,224)),
#                                                             transforms.ToTensor()
#                                                         ]))
# img = Image.open('/home/ssd1/ljy/test/test1/万丽OD_000.jpg')
# # img.convert("RGB")
# img=transforms.Resize((224,224))(img)
# # img=img.resize((224,224))
# img=transforms.ToTensor()(img)

# img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C

class SingleLabel(data.Dataset):
    def __init__(self,data_folder, label_path, csv_name, transform = None):
        super().__init__()
        self.img_path = data_folder
        self.lp = pd.read_csv(label_path).values
        self.transform = transform
        n = 0; rows = []
        for row in range(len(self.lp)):
            for i in range(2,len(self.lp[1])):
                if self.lp[row][i] != 0:
                    for j in range(2,len(self.lp[1])):
                        if self.lp[row][j] != 0 and j != i:
                            n = n + 1
                            rows.append(row)
                            # print(row,i,j)
                            break
                    break
                            # print(row)
        print(csv_name)
        print(n)
        # print(rows)
        df = pd.read_csv(label_path)
        df = df.drop(rows,axis=0)
        cols = [x for i, x in enumerate(df.columns[1:]) if df[x].sum() == 0]
        print(cols)
        df2 = df.drop(cols, axis=1)  # 利用drop方法将含有特定数值的列删除
        df2.to_csv(csv_name)
        print(len(df2.values))
        
                

    def __len__(self):
        return len(self.lp)

    def __getitem__(self, idex):
        # label, img_name, = self.df[idex]
        label = int(self.lp[idex][1]) # "Disease_Risk"
        img_name = self.img_path + '/' + str(self.lp[idex][0]) + '.png'
        img = Image.open(img_name)
        img=transforms.Resize((224,224))(img)
        img=transforms.ToTensor()(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


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
    train_data = SingleLabel(data_folder="/home/....RFMiD_All_Classes_Dataset/1. Original Images/a. Training Set",
                        label_path="/home/.../RFMiD_All_Classes_Dataset/2. Groundtruths/a. RFMiD_Training_Labels.csv", 
                        csv_name = "train2.csv")

    test_data = SingleLabel(data_folder="/home/.../RFMiD_All_Classes_Dataset/1. Original Images/c. Testing Set",
                            label_path="/home/.../RFMiD_All_Classes_Dataset/2. Groundtruths/c. RFMiD_Testing_Labels.csv", 
                            csv_name = "test2.csv")

    val_data = SingleLabel(data_folder="/home/.../RFMiD_All_Classes_Dataset/1. Original Images/b. Validation Set",
                           label_path="/home/.../RFMiD_All_Classes_Dataset/2. Groundtruths/b. RFMiD_Validation_Labels.csv",
                           csv_name = "val2.csv")




