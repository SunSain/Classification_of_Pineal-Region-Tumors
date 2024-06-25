import os
envpath = '/opt/anaconda39/lib/python3.9/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
from audioop import cross
import enum
from genericpath import exists

import torch
import json
from re import T  
from pickletools import optimize
import datetime, warnings
import tensorboardX
import numpy as np
import torch.nn as nn
from utils2.config_3class_noclinical import opt
#from model import ScaleDense
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings("ignore")
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils2.weighting.min_norm_solvers import MinNormSolver, gradient_normalizers
import torch.optim as optim

#from new_load_data import DIY_Folder

from load_data_23 import DIY_Folder
from utils2.earlystopping import EarlyStopping
from utils2.avgmeter import AverageMeter
from utils2.metrics import Metrics
from sklearn.model_selection import train_test_split
from model.resnet_3d_noclinical import ResNet10,ResNet18,ResNet34,ResNet24,ResNet30

from model.diy_resnet_3d import DIY_ResNet10,DIY_ResNet18


from utils2.norm_radiomics_dataset import  Unite_norm_radiomics
from torch.utils.data import WeightedRandomSampler
from utils2.FLoss import Focal_Loss
import seaborn as sns
from utils2.batchaverage import BatchCriterion
from load_3modality_only import DIY_Folder_3M

import subprocess
from utils2.single_batchavg import SingleBatchCriterion
free_gpu_id = opt.free_gpu_id
print("free_gpu_id: ",free_gpu_id)


if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_device(0)
    DEVICE=torch.device('cuda')
else:
    DEVICE=torch.device('cpu')
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
print('DEVICE: ',DEVICE)

best_pretrained_model_path="/home/chenxr/Pineal_region/after_12_08/Results/New_attention_2/Two/three_modality_best_attention/model_result/ResNet18_BatchAvg_0401_T1__k-fold-sub-fold-1__best_model.pth.tar"
best_pretrained_model_path="/home/chenxr/Pineal_region/after_12_08/Results/New_attention/Two/three_modality_best_attention/model_result/ResNet18_BatchAvg_0401_T1__k-fold-sub-fold-2__best_model.pth.tar"
best_pretrained_model_path="/home/chenxr/Pineal_region/after_12_08/Results/New_attention_2/Two/three_modality_best_attention/model_result/ResNet18_BatchAvg_0401_T1__k-fold-sub-fold-1__best_model.pth.tar"
best_pretrained_model_path="/home/chenxr/Pineal_region/after_12_08/Results/attention/Two/three_modality_best_attention/model_result/ResNet18_BatchAvg_0401_T1__k-fold-sub-fold-2__best_model.pth.tar"
best_pretrained_model_path="/home/chenxr/Pineal_region/after_12_08/Results/tmp_test/Two/T1C_SingleBatchAvg_selfKL_Composed_RM_composed_ResNet18/model_result/ResNet18_SingleBatchAvg_selfKL_0401_T1__k-fold-sub-fold-2__best_model.pth.tar"
model = ResNet18(num_classes=2,input_channel=3,use_radiomics=False,feature_align=True,use_clinical=False)
            

checkpoint = torch.load(best_pretrained_model_path,map_location = DEVICE)
if checkpoint['state_dict'].get( "fc_two.weight",None) !=None:
        checkpoint['state_dict']['fc.weight']=checkpoint['state_dict']['fc_two.weight']
        checkpoint['state_dict']['fc.bias']=checkpoint['state_dict']['fc_two.bias']
        del checkpoint['state_dict']['fc_two.weight']
        del checkpoint['state_dict']['fc_two.bias']
model.load_state_dict(checkpoint['state_dict'])




