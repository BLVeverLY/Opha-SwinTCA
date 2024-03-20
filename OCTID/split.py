import torch
import pandas as pd
#print(torch.__version__)
import torchvision
import pandas as pd
from torch.utils import data
import numpy as np

from PIL import Image
import csv



dataset = torchvision.datasets.ImageFolder('/home/newljy/OCTID/')
print(dataset.class_to_idx)
# {'OCT_AMD': 0, 'OCT_DR': 1, 'OCT_normal': 2}

# DR0, AMD1, normal2
f1 = open('train.csv', 'w')
f2 = open('test.csv', 'w')
writer1 = csv.writer(f1)
writer2 = csv.writer(f2)
writer1.writerow(["path","type"])
writer2.writerow(["path","type"])
n1=n2=n3=0
for i in range(len(dataset.imgs)):
    if dataset.imgs[i][1] == 0:
        if n1<22:
            writer2.writerow([dataset.imgs[i][0],1])
        else:
            writer1.writerow([dataset.imgs[i][0],1])
        n1=n1+1
    elif dataset.imgs[i][1] == 1:
        if n2<12:
            writer2.writerow([dataset.imgs[i][0],0])
        else:
            writer1.writerow([dataset.imgs[i][0],0])
        n2=n2+1
    elif dataset.imgs[i][1] == 2:
        if n3<42:
            writer2.writerow([dataset.imgs[i][0],2])
        else:
            writer1.writerow([dataset.imgs[i][0],2])
        n3=n3+1
