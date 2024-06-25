
import os
import torch
import nibabel as nib
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import SimpleITK as sitk
from scipy.ndimage import rotate
import nibabel as nib
import numpy as np
import os


class Feature_Folder(torch.utils.data.Dataset):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.t1_dataset=[]
        self.t2_dataset=[]
        self.t1c_dataset=[]
        self.count=0
        self.y=[]
        self.sid_dict =[]

    def __len__(self):
        return len(self.t1_dataset)


    def __add__(self,sid,t1_feature,t2_feature,t1c_feature,target):
        self.t1_dataset.append(t1_feature)
        self.t2_dataset.append(t2_feature)
        self.t1c_dataset.append(t1c_feature)
        self.y.append(target)
        self.sid_dict.append(sid)

    def __getitem__(self,index):
        return (self.t1_dataset[index],self.t2_dataset[index],self.t1c_dataset[index],self.sid_dict[index],self.y[index])
    
    def gety(self):
        return self.y


