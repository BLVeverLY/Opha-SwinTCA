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
    # person_id_all_list # every patient's ID or name
    person_id_all_list = []
    for i in range(len(ImageLabeldataset.imgs)): # 0,1,...,len-1
        dir = ImageLabeldataset.imgs[i][0].split('/')
        if dir[k-1] == disease+'OCT' or dir[k-1] == disease+'retina':
            if method=="ID":
                # dir2 = dir[k].split('-')
                # person_id_all_list.append(dir2[0])
                dir2 = dir[k][0:5] # ID refers to the first 6 digits
            elif method=="name":
                dir2 = "".join(re.findall(r'[\u4e00-\u9fa5]', dir[k])) 
            person_id_all_list.append(dir2)
    person_id_all_set = set(person_id_all_list)
    # person_id_set # people have the ID or name of the person who matches the required matching picture
    person_id_list = []
    person_1_model_dict = dict.fromkeys(person_id_all_set, 0)
    person_2_model_dict = dict.fromkeys(person_id_all_set, 0)
    for i in range(len(ImageLabeldataset.imgs)): # 0,1,...,len-1
        dir = ImageLabeldataset.imgs[i][0].split('/')
        dir2 = ""
        if method=="ID":
            dir2 = dir[k][0:5] 
        elif method=="name":
            dir2 = "".join(re.findall(r'[\u4e00-\u9fa5]', dir[k])) 
        if dir[k-1] == disease+'OCT' or dir[k-1] == disease+'retina':
            if dir[k-1] == disease+'OCT':
                person_1_model_dict[dir2] = 1
            if dir[k-1] == disease+'retina照相':
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
        if method=="ID":
            # dir2 = dir[6].split('-')
            dir2 = dir[k][0:5] # ID
            if dir2 in person_id_set:
                if dir2 not in dict_of_list:
                    dict_of_list[dir2]={'IMAGE1': [], 'IMAGE2': []}
                if dir[k-1] == disease + 'OCT':
                    dict_of_list[dir2]['IMAGE1'].append(i); n=n+1
                    # self.dict_of_list[dir2]['IMAGE1'].append(ImageLabeldataset1[i][0])
                if dir[k-1] == disease + 'retina照相':
                    dict_of_list[dir2]['IMAGE2'].append(i); m=m+1
                    # self.dict_of_list[dir2]['IMAGE2'].append(ImageLabeldataset1[i][0])
        elif method=="name":
            dir3 = "".join(re.findall(r'[\u4e00-\u9fa5]', dir[k]))  # name
            if dir3!="" and dir3 in person_id_set:
                if dir3 not in dict_of_list:
                    dict_of_list[dir3]={'IMAGE1': [], 'IMAGE2': []}
                if dir[k-1] == disease + 'OCT':
                    dict_of_list[dir3]['IMAGE1'].append(i); n=n+1
                if dir[k-1] == disease + 'retina照相':
                    dict_of_list[dir3]['IMAGE2'].append(i); m=m+1
    return n, m


class TwoImageData(Dataset): #继承Dataset
    def __init__(self, isTrain = False, gpu = 5):
        super(TwoImageData, self).__init__()
        random.seed(49)
        self.gpu = gpu

        if isTrain:
            path1 = '/home/.../DR_train'; k1=8 # k: position of person ID or name
            path2 = '/home/.../AMD_train'; k2=8
            path3 = '/home/.../HW_train'; k3=8
            path4 = '/home/.../RVO_train'; k4=8
            path5 = '/home/.../glaucoma_train'; k5=8
            path6 = '/home/.../normal_train'; k6=8
        else:
            path1 = '/home/.../DR_test'; k1=8
            path2 = '/home/.../AMD_test'; k2=8
            path3 = '/home/.../HW_test'; k3=8
            path4 = '/home/.../RVO_test'; k4=8
            path5 = '/home/.../glaucoma_test'; k5=8
            path6 = '/home/.../normal_test'; k6=8

        self.ImageLabeldataset1, self.person_id_set1 = PairedImages("DR", path1, k1, "ID")
        self.ImageLabeldataset2, self.person_id_set2 = PairedImages("AMD", path2, k2, "ID")
        self.ImageLabeldataset3, self.person_id_set3 = PairedImages("HW", path3, k3, "ID")
        self.ImageLabeldataset4, self.person_id_set4 = PairedImages("RVO", path4, k4, "ID")
        self.ImageLabeldataset5, self.person_id_set5 = PairedImages("glaucoma", path5, k5, "ID")
        self.ImageLabeldataset6, self.person_id_set6 = PairedImages("normal", path6, k6, "ID")

        self.all_person_id_list = list(self.person_id_set1) + \
                                  list(self.person_id_set2) + \
                                  list(self.person_id_set3) + \
                                  list(self.person_id_set4) + \
                                  list(self.person_id_set5) + \
                                  list(self.person_id_set6)
        # self.dict_of_list[id] = {'IMAGE1': [], 'IMAGE2': []} # OCT, retina
        # d={'a':{'aa':[1,2,3],'bb':[5]},'b':{'aa':[4,7]}}
        self.dict_of_list = {}
        n1, m1 = ConstructDict(self.dict_of_list, "DR", self.ImageLabeldataset1, k1, self.person_id_set1, "ID")
        n2, m2 = ConstructDict(self.dict_of_list, "AMD", self.ImageLabeldataset2, k2, self.person_id_set2, "ID")
        n3, m3 = ConstructDict(self.dict_of_list, "HW", self.ImageLabeldataset3, k3, self.person_id_set3, "ID")
        n4, m4 = ConstructDict(self.dict_of_list, "RVO", self.ImageLabeldataset4, k4, self.person_id_set4, "ID")
        n5, m5 = ConstructDict(self.dict_of_list, "glaucoma", self.ImageLabeldataset5, k5, self.person_id_set5, "ID")
        n6, m6 = ConstructDict(self.dict_of_list, "normal", self.ImageLabeldataset6, k6, self.person_id_set6, "ID")

        print("number of people:")
        print(len(self.person_id_set1)+len(self.person_id_set7),len(self.person_id_set2)+len(self.person_id_set8),
              len(self.person_id_set3)+len(self.person_id_set9),len(self.person_id_set4)+len(self.person_id_set10),
              len(self.person_id_set5)+len(self.person_id_set11),len(self.person_id_set12)+len(self.person_id_set13))
        print("number of OCT images:")
        print(n1,n2,n3,n4,n5,n6)
        print("number of fundus images:")
        print(m1,m2,m3,m4,m5,m6)

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

    def __len__(self):# returns the size of the entire dataset
        # return len(self.id_list)
        return len(self.all_person_id_list)
    
    def __getitem__(self, index):# return dataset based on index dataset[index]
        person_id = self.all_person_id_list[index]
        if person_id in self.person_id_set1:
            label = 0 # DR
            image1 = self.ImageLabeldataset1[random.choice(self.dict_of_list[person_id]['IMAGE1'])][0]
            image2 = self.ImageLabeldataset1[random.choice(self.dict_of_list[person_id]['IMAGE2'])][0]
        elif person_id in self.person_id_set2:
            label = 1 # AMD
            image1 = self.ImageLabeldataset2[random.choice(self.dict_of_list[person_id]['IMAGE1'])][0]
            image2 = self.ImageLabeldataset2[random.choice(self.dict_of_list[person_id]['IMAGE2'])][0]
        elif person_id in self.person_id_set3:
            label = 2 # HW
            image1 = self.ImageLabeldataset3[random.choice(self.dict_of_list[person_id]['IMAGE1'])][0]
            image2 = self.ImageLabeldataset3[random.choice(self.dict_of_list[person_id]['IMAGE2'])][0]
        elif person_id in self.person_id_set4:
            label = 3 # RVO
            image1 = self.ImageLabeldataset4[random.choice(self.dict_of_list[person_id]['IMAGE1'])][0]
            image2 = self.ImageLabeldataset4[random.choice(self.dict_of_list[person_id]['IMAGE2'])][0]
        elif person_id in self.person_id_set5:
            label = 4 # glaucoma
            image1 = self.ImageLabeldataset5[random.choice(self.dict_of_list[person_id]['IMAGE1'])][0]
            image2 = self.ImageLabeldataset5[random.choice(self.dict_of_list[person_id]['IMAGE2'])][0]
        elif person_id in self.person_id_set6:
            label = 5 # normal
            print(self.dict_of_list[person_id]['IMAGE1'])
            image1 = self.ImageLabeldataset6[random.choice(self.dict_of_list[person_id]['IMAGE1'])][0]
            image2 = self.ImageLabeldataset6[random.choice(self.dict_of_list[person_id]['IMAGE2'])][0]
        
        label = np.array(label)
        return image1, image2, label, person_id         