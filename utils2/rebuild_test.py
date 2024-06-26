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

from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
import torchio as tio

import shutil


def nii_loader(path):
    img = nib.load(str(path))
    return img

def read_table(path):
    return(pd.read_excel(path).values) # default to first sheet

def get_ND_bounding_box(label, margin):
    """
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = label.shape
    if(type(margin) is int ):
        margin = [margin]*len(input_shape)
    assert(len(input_shape) == len(margin))
    indxes = np.nonzero(label)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    return idx_min, idx_max


def preprocess_sigle_img(image, threshold=0):
    "standardize voxels with value > threshold"
    image = image.astype(np.float32)
    print("a tra: ",image[:,:,50])
    idx_min,idx_max=get_ND_bounding_box(image)
    mean = np.mean(image)
    std = np.sqrt(image)

    image-=mean
    #image/=std
    return image

def gety(data_path):
    y=[]
    count_0=0
    count_1=0
    x=[]
    for i, f in enumerate(os.listdir(data_path)):
            fn = f.split("_")
            print("f: ",f)
            x.append(f)
            sid=0
            try:
                    sid=int(fn[0])
            except:
                print("it's the_select_file")
            
            if sid<61: 
                count_0+=1
                y.append(0)
            else: 
                count_1+=1
                y.append(1) 
    print("count_0: ",count_0," ;count_1: ",count_1)
    print("y: ",y)
    return x,y




if __name__=="__main__":
        
    data_path="/opt/chenxingru/Pineal_region/5_28_all_regis_AXandTra_best_file/"

    train_save_path="/opt/chenxingru/Pineal_region/9_24_regis_AXandTra_best_f_NoTest/"
    test_save_path="/opt/chenxingru/Pineal_region/9_24_regis_AXandTra_best_f_Test/"
    if not os.path.exists(train_save_path):
        os.makedirs(train_save_path)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    split=StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)
    train=[]
    test=[]
    x,y=np.array(gety(data_path))

    for train_index, test_index in split.split(x,y):
        print("train_index: ",train_index," ;test_index: ",test_index)
        x_train,x_test = x[train_index], x[test_index]
        y_train,y_test = y[train_index], y[test_index]
        print("train: ",train," ; test: ",test)
    print(x[train_index], x[test_index])
    print(y_train,y_test)


    for i, f in enumerate(x_train):
        path_save = train_save_path+f # chenxingru/.../001_liu/7397192/3_2_T1.nii.gz
        path_get= data_path+f
        print("\n[select_SAG]  tmp_path: "+path_get+" \n save_file:  "+path_save)
        shutil.copyfile(path_get,path_save)

    for i, f in enumerate(x_test):
        path_save = test_save_path+f # chenxingru/.../001_liu/7397192/3_2_T1.nii.gz
        path_get= data_path+f
        print("\n[select_SAG]  tmp_path: "+path_get+" \n save_file:  "+path_save)
        shutil.copyfile(path_get,path_save)


   

