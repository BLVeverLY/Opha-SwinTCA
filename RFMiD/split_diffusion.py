import torch
import pandas as pd
#print(torch.__version__)
import torchvision
import pandas as pd
from torch.utils import data
import numpy as np

from PIL import Image
import csv


    
dataset = torchvision.datasets.ImageFolder('/home/newljy/SD_gen_data/')
print(dataset.class_to_idx)
# {'OCT_age-related_macular_degeneration': 0, 'OCT_diabetics_retinopathy': 1, 'OCT_glaucoma': 2, 'OCT_high_myopia': 3, 'OCT_normal': 4, 'OCT_retinal_vein_occlusion': 5, 'fundus_age-related_macular_degeneration': 6, 'fundus_diabetics_retinopathy': 7, 'fundus_glaucoma': 8, 'fundus_high_myopia': 9, 'fundus_normal': 10, 'fundus_retinal_vein_occlusion': 11}

# with open('OCT_SD.csv',"w") as csvfile: 
#     writer = csv.writer(csvfile)
#     # writer.writerow(["ID", "DR", "AMD", "HM", "RVO", "Glaucoma", "Normal"])
#     writer.writerow(["path","type"])
#     for i in range(len(dataset.imgs)):
#         if dataset.imgs[i][1] == 0:
#             writer.writerow([dataset.imgs[i][0],1])
#         elif dataset.imgs[i][1] == 1:
#             writer.writerow([dataset.imgs[i][0],0])
#         elif dataset.imgs[i][1] == 2:
#             writer.writerow([dataset.imgs[i][0],4])
#         elif dataset.imgs[i][1] == 3:
#             writer.writerow([dataset.imgs[i][0],2])
#         elif dataset.imgs[i][1] == 4:
#             writer.writerow([dataset.imgs[i][0],5])
#         elif dataset.imgs[i][1] == 5:
#             writer.writerow([dataset.imgs[i][0],3])

# with open('fundus_SD.csv',"w") as csvfile: 
#     writer = csv.writer(csvfile)
#     # writer.writerow(["ID", "DR", "AMD", "HM", "RVO", "Glaucoma", "Normal"])
#     writer.writerow(["path","type"])
#     for i in range(len(dataset.imgs)):
#         if dataset.imgs[i][1] == 6:
#             writer.writerow([dataset.imgs[i][0],1])
#         elif dataset.imgs[i][1] == 7:
#             writer.writerow([dataset.imgs[i][0],0])
#         elif dataset.imgs[i][1] == 8:
#             writer.writerow([dataset.imgs[i][0],4])
#         elif dataset.imgs[i][1] == 9:
#             writer.writerow([dataset.imgs[i][0],2])
#         elif dataset.imgs[i][1] == 10:
#             writer.writerow([dataset.imgs[i][0],5])
#         elif dataset.imgs[i][1] == 11:
#             writer.writerow([dataset.imgs[i][0],3])

f1 = open('OCT_train.csv', 'w')
f2 = open('OCT_test.csv', 'w')
writer1 = csv.writer(f1)
writer2 = csv.writer(f2)
writer1.writerow(["path","type"])
writer2.writerow(["path","type"])
df = pd.read_csv("OCT_SD.csv").values
n0=n1=n2=n3=n4=n5=0
for i in range(len(df)):
    if df[i][1]==0:
        n0=n0+1
        if n0<80:
            writer1.writerow(df[i])
        else:
            writer2.writerow(df[i])
    elif df[i][1]==1:
        n1=n1+1
        if n1<80:
            writer1.writerow(df[i])
        else:
            writer2.writerow(df[i])
    elif df[i][1]==2:
        n2=n2+1
        if n2<80:
            writer1.writerow(df[i])
        else:
            writer2.writerow(df[i])
    elif df[i][1]==3:
        n3=n3+1
        if n3<80:
            writer1.writerow(df[i])
        else:
            writer2.writerow(df[i])
    elif df[i][1]==4:
        n4=n4+1
        if n4<80:
            writer1.writerow(df[i])
        else:
            writer2.writerow(df[i])
    elif df[i][1]==5:
        n5=n5+1
        if n5<80:
            writer1.writerow(df[i])
        else:
            writer2.writerow(df[i])

f1 = open('fundus_train.csv', 'w')
f2 = open('fundus_test.csv', 'w')
writer1 = csv.writer(f1)
writer2 = csv.writer(f2)
writer1.writerow(["path","type"])
writer2.writerow(["path","type"])
df = pd.read_csv("fundus_SD.csv").values
n0=n1=n2=n3=n4=n5=0
for i in range(len(df)):
    if df[i][1]==0:
        n0=n0+1
        if n0<80:
            writer1.writerow(df[i])
        else:
            writer2.writerow(df[i])
    elif df[i][1]==1:
        n1=n1+1
        if n1<80:
            writer1.writerow(df[i])
        else:
            writer2.writerow(df[i])
    elif df[i][1]==2:
        n2=n2+1
        if n2<80:
            writer1.writerow(df[i])
        else:
            writer2.writerow(df[i])
    elif df[i][1]==3:
        n3=n3+1
        if n3<80:
            writer1.writerow(df[i])
        else:
            writer2.writerow(df[i])
    elif df[i][1]==4:
        n4=n4+1
        if n4<80:
            writer1.writerow(df[i])
        else:
            writer2.writerow(df[i])
    elif df[i][1]==5:
        n5=n5+1
        if n5<80:
            writer1.writerow(df[i])
        else:
            writer2.writerow(df[i])