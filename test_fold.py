from audioop import cross
import enum
from genericpath import exists
import os
import torch
import json
from re import T  
from pickletools import optimize
import datetime, warnings
import tensorboardX
import numpy as np
import torch.nn as nn
from utils2.config import opt
#from model import ScaleDense
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings("ignore")
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torch.optim as optim

#from new_load_data import DIY_Folder

from load_data_2 import DIY_Folder
from utils2.earlystopping import EarlyStopping
from utils2.avgmeter import AverageMeter
from utils2.metrics import Metrics
from sklearn.model_selection import train_test_split
from model.resnet_3d import ResNet10,ResNet18,ResNet34

from model.diy_resnet_3d import DIY_ResNet10,DIY_ResNet18
import matplotlib.pyplot as plt
from utils2.weighted_CE import Weighted_CE
from utils2.self_KL import SelfKL
import time

import torchio as tio
from sklearn.model_selection import cross_validate
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from statistics import mean

if __name__=="__main__": 
    begin_time = time.time()

    data_path=opt.data_path
    test_data_path = opt.testdata_path

    print("=========== start train the brain age estimation model =========== \n")
    print(" ==========> Using {} processes for data loader.".format(opt.num_workers))


    #load the training_data  and test_data (training_data will be splited later for cross_validation)
    total_file = DIY_Folder(data_path=data_path,train_form = opt.train_form,root_mask_radiomics = opt.root_bbx_path,use_radiomics=opt.use_radiomics,istest=False)
    print("len(total_file): ",len(total_file))
    print("total_file.gety(): ",total_file.gety())
    #root_radiomics="",root_mask_radiomics="",use_radiomics=True, norm_radiomics=True
    test_file = DIY_Folder(data_path=test_data_path,train_form = opt.train_form,root_mask_radiomics = opt.test_root_bbx_path,use_radiomics=opt.use_radiomics,istest=True)

    test_data = test_file.select_dataset(data_idx=[i for i in range(len(test_file))], aug=False,use_secondimg=True,noSameRM=opt.noSameRM,usethird=opt.usethird)
    
    for i in range(3):
        print("i=================: ",i)
        k=5
        splits=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)

        for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(total_file)),total_file.gety())):
            print("fold: ",fold)

            train_sids = total_file.getsid(train_idx)
            valid_sids = total_file.getsid(val_idx)
            print("train_sids: ",train_sids)
            print("val_sids: ",valid_sids)
        