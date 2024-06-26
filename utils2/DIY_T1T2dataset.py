import imp
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



class DIY_T1T2set(torch.utils.data.Dataset):
    def __init__(self, T1_dataset=[],T2_dataset=[]):

        
        self.T1_dataset=T1_dataset
        self.T2_dataset=T2_dataset

        self.type=['T1','T2']


    def __len__(self):
        return len(self.T1_dataset)


    def __getitem__(self,index):
        data1=self.T1_dataset[index]
        data2=self.T2_dataset[index]
        return (data1, data2)




