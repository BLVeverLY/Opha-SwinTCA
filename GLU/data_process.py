import torchvision
from torch.utils.data import Dataset
import numpy as np
import re
import random
from torchvision import transforms
from itertools import product

def PairedImages(disease, path, k, method):
    ImageLabeldataset = torchvision.datasets.ImageFolder(path
                            ,transform=transforms.Compose([
                                                            transforms.Resize((224,224)),
                                                            transforms.ToTensor()
                                                        ]))
    # person_id_all_list 所有人的ID或name
    person_id_all_list = []
    for i in range(len(ImageLabeldataset.imgs)): # 0,1,...,len-1
        dir = ImageLabeldataset.imgs[i][0].split('/')
        if dir[k-1] == disease+'OCT' or dir[k-1] == disease+'retina':
                dir2 = dir[k]
        person_id_all_list.append(dir2)
    person_id_all_set = set(person_id_all_list)
    # person_id_set 这些人中有符合要求的配对图片的人的ID或name
    person_id_list = []
    person_1_model_dict = dict.fromkeys(person_id_all_set, 0)
    person_2_model_dict = dict.fromkeys(person_id_all_set, 0)
    for i in range(len(ImageLabeldataset.imgs)): # 0,1,...,len-1
        dir = ImageLabeldataset.imgs[i][0].split('/')
        dir2 = ""
        dir2 = dir[k]
        if dir[k-1] == disease+'OCT' or dir[k-1] == disease+'retina':
            if dir[k-1] == disease+'OCT':
                person_1_model_dict[dir2] = 1
            if dir[k-1] == disease+'retina':
                person_2_model_dict[dir2] = 1
            if person_1_model_dict[dir2] and person_2_model_dict[dir2]:
                person_id_list.append(dir2)
    person_id_set = set(person_id_list)
    print(path+": ", ImageLabeldataset.class_to_idx)
    print(disease, len(person_id_set))
    return ImageLabeldataset, person_id_set


# def merge_ImageFolder(dataset, sub_dataset):
#     dataset.classes.extend(sub_dataset.classes)
#     dataset.classes = sorted(list(set(dataset.classes)))
#     dataset.class_to_idx.update(sub_dataset.class_to_idx)
#     dataset.samples.extend(sub_dataset.samples)
#     dataset.targets.extend(sub_dataset.targets)
#     return


def ConstructDict(dict_of_list, disease, ImageLabeldataset, k, person_id_set, method): # , method):
    n = m = 0
    for i in range(len(ImageLabeldataset.imgs)): # 0,1,...,len-1
        dir = ImageLabeldataset.imgs[i][0].split('/')
        dir2 = dir[k]
        # print(dir2)
        if dir2 in person_id_set:
                if dir2 not in dict_of_list:
                    dict_of_list[dir2]={'IMAGE1': [], 'IMAGE2': []}
                if dir[k-1] == disease + 'OCT':
                    dict_of_list[dir2]['IMAGE1'].append(i); n=n+1
                    # self.dict_of_list[dir2]['IMAGE1'].append(ImageLabeldataset1[i][0])
                if dir[k-1] == disease + 'retina照相':
                    dict_of_list[dir2]['IMAGE2'].append(i); m=m+1
    return n, m


class TwoImageData(Dataset): #继承Dataset
    def __init__(self, isTrain = False, gpu = 5):
        super(TwoImageData, self).__init__()
        random.seed(49)
        self.gpu = gpu

        # /home/ssd1/ljy/fundus_OCT/data/RVO_test/RVOOCT/70046鲍道南-04/ k=8
        # /home/newljy/1_3AMD/train/Neovascular AMD/Neovascular AMDOCT/0096032L-86F/
        # /home/newljy/2_2GLU/train/G/GOCT/P 1/
        if isTrain:
            path1 = '/home/.../GLU/train/G/'; k1 = 7
            path2 = '/home/.../GLU/train/N/'; k2 = 7
        else:
            path1 = '/home/.../GLU/val/G/'; k1 = 7
            path2 = '/home/.../GLU/val/N/'; k2 = 7
        
        self.ImageLabeldataset1, self.person_id_set1 = PairedImages("G", path1, k1, "ID")
        self.ImageLabeldataset2, self.person_id_set2 = PairedImages("N", path2, k2, "ID")

        self.all_person_id_list = list(self.person_id_set1) + list(self.person_id_set2)
        # self.dict_of_list[id] = {'IMAGE1': [], 'IMAGE2': []} # OCT, retina
        # d={'a':{'aa':[1,2,3],'bb':[5]},'b':{'aa':[4,7]}}
        self.dict_of_list = {}
        n1, m1 = ConstructDict(self.dict_of_list, "G", self.ImageLabeldataset1, k1, self.person_id_set1, "ID")
        n2, m2 = ConstructDict(self.dict_of_list, "N", self.ImageLabeldataset2, k2, self.person_id_set2, "ID")

        print("number of people:")
        print(len(self.person_id_set1),len(self.person_id_set2))
        print("number of OCT images:")
        print(n1,n2)
        print("number of fundus images:")
        print(m1,m2)

        # max1 = max2 = 0; min1 = min2 = 9999
        # for key,value in self.dict_of_list.items():
        #     print(len(self.dict_of_list[key]['IMAGE1']), len(self.dict_of_list[key]['IMAGE2']))
        #     if len(self.dict_of_list[key]['IMAGE1'])>max1:
        #         max1=len(self.dict_of_list[key]['IMAGE1'])
        #     if len(self.dict_of_list[key]['IMAGE1'])<min1:
        #         min1=len(self.dict_of_list[key]['IMAGE1'])
        #     if len(self.dict_of_list[key]['IMAGE2'])>max2:
        #         max2=len(self.dict_of_list[key]['IMAGE2'])
        #     if len(self.dict_of_list[key]['IMAGE2'])<min2:
        #         min2=len(self.dict_of_list[key]['IMAGE2'])
        # print(max1, min1, max2, min2)
        self.all_person_id_list.sort()

    def __len__(self):#返回整个数据集的大小
        # return len(self.id_list)
        return len(self.all_person_id_list)
    
    def __getitem__(self, index):#根据索引index返回dataset[index]
        person_id = self.all_person_id_list[index]
        if person_id in self.person_id_set1:
            label = 0 # glaucoma
            image1 = self.ImageLabeldataset1[random.choice(self.dict_of_list[person_id]['IMAGE1'])][0]
            image2 = self.ImageLabeldataset1[random.choice(self.dict_of_list[person_id]['IMAGE2'])][0]
        elif person_id in self.person_id_set2:
            label = 1 # normal
            image1 = self.ImageLabeldataset2[random.choice(self.dict_of_list[person_id]['IMAGE1'])][0]
            image2 = self.ImageLabeldataset2[random.choice(self.dict_of_list[person_id]['IMAGE2'])][0]
        
        label = np.array(label)
        return image1, image2, label, person_id         