
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
from utils2.config_attention import opt
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

from load_data_23 import DIY_Folder
from utils2.earlystopping import EarlyStopping
from utils2.avgmeter import AverageMeter
from utils2.metrics import Metrics
from sklearn.model_selection import train_test_split
from model.resnet_3d import ResNet10,ResNet18,ResNet34,ResNet24,ResNet30

from model.diy_resnet_3d import DIY_ResNet10,DIY_ResNet18
import matplotlib.pyplot as plt
from utils2.weighted_CE import Weighted_CE

from copy import deepcopy
from utils2.self_KL import SelfKL
"""
from model.vgg11 import VGG11_bn
from model.vgg13 import VGG13_bn
from model.Inception2 import Inception2
from model.vgg16 import VGG16_bn
from model.SEResnet import seresnet18
"""
from model import tencent_resnet


import torchio as tio
from sklearn.model_selection import cross_validate
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_curve, auc,roc_auc_score,confusion_matrix ,precision_score,f1_score,recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from statistics import mean
import time
import math

from utils2.norm_radiomics_dataset import  Unite_norm_radiomics
from torch.utils.data import WeightedRandomSampler
from utils2.FLoss import Focal_Loss
import seaborn as sns
from utils2.batchaverage import BatchCriterion
from load_3modality_only import DIY_Folder_3M

import subprocess
from load_feature_dataset import Feature_Folder
from new_load_feature_dataset import New_Feature_Folder
from model.wrong_multi_attention import Self_Attention
from model.concatenation_mlp import Concateation_MLP
from model.transformer import Transformer
free_gpu_id = opt.free_gpu_id
import pandas as pd
print("free_gpu_id: ",free_gpu_id)


if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_device(1)
    DEVICE=torch.device('cuda')
    print("DEVICE: ",torch.cuda.current_device())
else:
    DEVICE=torch.device('cpu')
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
print('DEVICE: ',DEVICE)




#/home/chenxr/Pineal_region/after_12_08/Results/old_T1C/Two/pretrained_constrain_composed_batchavg_ResNet18/model_result/ResNet18_BatchAvg_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar
#/home/chenxr/Pineal_region/after_12_08/Results/old_T2/Two/new_pretrained_batchavg_constrain_composed_ResNet18/model_result/
#/home/chenxr/Pineal_region/after_12_08/Results/old_T2/Two/new_pretrained_selfKL_composed_ResNet18/ResNet18_SelfKL_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar
#/home/chenxr/Pineal_region/after_12_08/Results/old_T1/Two/new_real_batchavg_constrain_composed_ResNet18/ResNet18_SelfKL_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar

def save_feature(t1_train_box,t1_valid_box,t1_test_box,fold,ps):
    feature_csv_output_dir = "/opt/chenxingru/Pineal_region/after_12_08/after_12_08/saved_features"
    if not os.path.exists(feature_csv_output_dir):
        os.mkdir(feature_csv_output_dir)
        
    train_features=pd.DataFrame(data=t1_train_box)
    train_features.to_csv(os.path.join(feature_csv_output_dir,ps+"_"+str(fold)+"_train.csv"),mode='a+',index=None,header=None)

    valid_features=pd.DataFrame(data=t1_valid_box)
    valid_features.to_csv(os.path.join(feature_csv_output_dir,ps+"_"+str(fold)+"_valid.csv"),mode='a+',index=None,header=None)
    
    test_features=pd.DataFrame(data=t1_test_box)
    test_features.to_csv(os.path.join(feature_csv_output_dir,ps+"_"+str(fold)+"_test.csv"),mode='a+',index=None,header=None)


def add_feature(sid,feat,target):
        id_feat=torch.unsqueeze(sid, dim=1)
        id_feat=id_feat.numpy()
        print("id_feat: ",id_feat)
        print("target: ",target)
        id_feat=np.concatenate((id_feat, feat.cpu().detach().numpy()), axis=1)
        id_feat=np.concatenate((id_feat, torch.unsqueeze(target,dim=1).cpu().detach().numpy()), axis=1)

        print("id_feat: ",id_feat)
        print("id_feat.shape: ",id_feat.shape)
        print("target: ", torch.unsqueeze(target,dim=1))
        return id_feat
        
        
"""def build_feature_csv(t1_model,t2_model,t1c_model,train_dataloaer, valid_dataloader,test_dataloader):
    t1_model.eval()
    t2_model.eval()
    t1c_model.eval()
    train_dataset = Feature_Folder(num_classes=2)
    valid_dataset = Feature_Folder(num_classes=2)
    test_dataset = Feature_Folder(num_classes=2)
    t1_train_box=[]
    t2_train_box=[]
    t1c_train_box=[]
    t1_test_box=[]
    t2_test_box=[]
    t1c_test_box=[]
    t1_valid_box=[]
    t2_valid_box=[]
    t1c_valid_box=[]
    for i, stuff in enumerate(train_dataloaer):
        (T1cimg,T1img,T2img,sid,target, saffix,sradiomics) = stuff
        target = torch.from_numpy(np.expand_dims(target,axis=1))
        target = convert(target)
        
        t1_img = T1img.to(DEVICE3)
        t2_img = T2img.to(DEVICE1)
        t1c_img = T1cimg.to(DEVICE2)
        input_affix = saffix.to(DEVICE3)
        #input_img = torch.reshape(input_img, [input_img.shape[0],1,input_img.shape[1],input_img.shape[2],input_img.shape[3]])
        t1_img_out,t1_feat = t1_model(t1_img,input_affix,[])
        id_feat = add_feature(sid,t1_feat,target)
        t1_train_box.extend(id_feat.tolist())
        
        input_affix = saffix.to(DEVICE1)
        t2_img_out,t2_feat = t2_model(t2_img,input_affix,[])
        id_feat = add_feature(sid,t2_feat,target)
        t2_train_box.extend(id_feat.tolist())
        
        input_affix = saffix.to(DEVICE2)
        t1c_img_out,t1c_feat = t1c_model(t1c_img,input_affix,[])
        id_feat = add_feature(sid,t1c_feat,target)
        t1c_train_box.extend(id_feat.tolist())
        
        train_dataset.__add__(sid,t1_feat,t2_feat,t1c_feat,target)

    for i, stuff in enumerate(valid_dataloader):
        (T1cimg,T1img,T2img,sid,target, saffix,sradiomics) = stuff
        target = torch.from_numpy(np.expand_dims(target,axis=1))
        target = convert(target)
        
        t1_img = T1img.to(DEVICE3)
        t2_img = T2img.to(DEVICE1)
        t1c_img = T1cimg.to(DEVICE2)
        input_affix = saffix.to(DEVICE3)
        #input_img = torch.reshape(input_img, [input_img.shape[0],1,input_img.shape[1],input_img.shape[2],input_img.shape[3]])
        t1_img_out,t1_feat = t1_model(t1_img,input_affix,[])
        id_feat = add_feature(sid,t1_feat,target)
        t1_valid_box.extend(id_feat.tolist())
        
        input_affix = saffix.to(DEVICE1)
        t2_img_out,t2_feat = t2_model(t2_img,input_affix,[])
        id_feat = add_feature(sid,t2_feat,target)
        t2_valid_box.extend(id_feat.tolist())
        
        input_affix = saffix.to(DEVICE2)
        t1c_img_out,t1c_feat = t1c_model(t1c_img,input_affix,[])
        id_feat = add_feature(sid,t1c_feat,target)
        t1c_valid_box.extend(id_feat.tolist())
        
        valid_dataset.__add__(sid,t1_feat,t2_feat,t1c_feat,target)
        
    for i, stuff in enumerate(test_dataloader):
        (T1cimg,T1img,T2img,sid,target, saffix,sradiomics) = stuff
        target = torch.from_numpy(np.expand_dims(target,axis=1))
        target = convert(target)
        t1_img = T1img.to(DEVICE3)
        t2_img = T2img.to(DEVICE1)
        t1c_img = T1cimg.to(DEVICE2)
        input_affix = saffix.to(DEVICE3)
        #input_img = torch.reshape(input_img, [input_img.shape[0],1,input_img.shape[1],input_img.shape[2],input_img.shape[3]])
        t1_img_out,t1_feat = t1_model(t1_img,input_affix,[])
        id_feat = add_feature(sid,t1_feat,target)
        t1_test_box.extend(id_feat.tolist())
        
        input_affix = saffix.to(DEVICE1)
        t2_img_out,t2_feat = t2_model(t2_img,input_affix,[])
        id_feat = add_feature(sid,t2_feat,target)
        t2_test_box.extend(id_feat.tolist())
        
        input_affix = saffix.to(DEVICE2)
        t1c_img_out,t1c_feat = t1c_model(t1c_img,input_affix,[])
        id_feat = add_feature(sid,t1c_feat,target)
        t1c_test_box.extend(id_feat.tolist())
        
        test_dataset.__add__(sid,t1_feat,t2_feat,t1c_feat,target)
        
    print("t1_train_box features.shape: ",np.array(t1_train_box).shape)
    print("t2_train_box features.shape: ",np.array(t2_train_box).shape)
    print("t1c_train_box features.shape: ",np.array(t1c_train_box).shape)
    print("t1_valid_box features.shape: ",np.array(t1_valid_box).shape)
    print("t2_valid_box features.shape: ",np.array(t2_valid_box).shape)
    print("t1c_valid_box features.shape: ",np.array(t1c_valid_box).shape)
    print("t1_test_box features.shape: ",np.array(t1_test_box).shape)
    print("t2_test_box features.shape: ",np.array(t2_test_box).shape)
    print("t1c_test_box features.shape: ",np.array(t1c_test_box).shape)
       
    return train_dataset, valid_dataset, test_dataset,t1_train_box,t2_train_box,t1c_train_box,t1_valid_box,t2_valid_box,t1c_valid_box,t1_test_box,t2_test_box,t1c_test_box
"""

def build_feature_csv(model,train_dataloaer, valid_dataloader,test_dataloader,ps):
    model.eval()
    train_dataset = Feature_Folder(num_classes=2)
    valid_dataset = Feature_Folder(num_classes=2)
    test_dataset = Feature_Folder(num_classes=2)
    t1_train_box=[]
    t1_test_box=[]
    
    t1_valid_box=[]
    print("ps: ",ps)
    for i, stuff in enumerate(train_dataloaer):
        (T1cimg,T1img,T2img,sid,target, saffix,sradiomics) = stuff
        target = torch.from_numpy(np.expand_dims(target,axis=1))
        target = convert(target)
        print("traing sid: ",sid)
        if sid ==49:
            continue
        torch.cuda.empty_cache()
        t1_model = model.to(DEVICE)
        if ps=="T1":
            img =  T1img.to(DEVICE)
        elif ps =="T2":
            img = T2img.to(DEVICE)
        else:
            img = T1cimg.to(DEVICE)
        input_affix = saffix.to(DEVICE)
        print("CURRENT DEVICE: ",torch.cuda.current_device())
        #input_img = torch.reshape(input_img, [input_img.shape[0],1,input_img.shape[1],input_img.shape[2],input_img.shape[3]])
        t1_img_out,t1_feat = t1_model(img,input_affix,[])
        id_feat = add_feature(sid,t1_feat,target)
        t1_train_box.extend(id_feat.tolist())
        train_dataset.__add__(sid,t1_feat,[],[],target)
        del img
        del input_affix
        del t1_model
        torch.cuda.empty_cache()

    for i, stuff in enumerate(valid_dataloader):
        (T1cimg,T1img,T2img,sid,target, saffix,sradiomics) = stuff
        target = torch.from_numpy(np.expand_dims(target,axis=1))
        target = convert(target)
        print("validate sid: ",sid)
        torch.cuda.empty_cache()
        t1_model = model.to(DEVICE)
        if ps=="T1":
            img =  T1img.to(DEVICE)
        elif ps =="T2":
            img = T2img.to(DEVICE)
        else:
            img = T1cimg.to(DEVICE)
        input_affix = saffix.to(DEVICE)
        #input_img = torch.reshape(input_img, [input_img.shape[0],1,input_img.shape[1],input_img.shape[2],input_img.shape[3]])
        t1_img_out,t1_feat = t1_model(img,input_affix,[])
        id_feat = add_feature(sid,t1_feat,target)
        t1_valid_box.extend(id_feat.tolist())
        
        valid_dataset.__add__(sid,t1_feat,[],[],target)
        del img
        del input_affix
        del t1_model
        torch.cuda.empty_cache()
        
    for i, stuff in enumerate(test_dataloader):
        (T1cimg,T1img,T2img,sid,target, saffix,sradiomics) = stuff
        target = torch.from_numpy(np.expand_dims(target,axis=1))
        target = convert(target)
        print("testing sid: ",sid)
        torch.cuda.empty_cache()
        t1_model = model.to(DEVICE)
        if ps=="T1":
            img =  T1img.to(DEVICE)
        elif ps =="T2":
            img = T2img.to(DEVICE)
        else:
            img = T1cimg.to(DEVICE)
        input_affix = saffix.to(DEVICE)
        #input_img = torch.reshape(input_img, [input_img.shape[0],1,input_img.shape[1],input_img.shape[2],input_img.shape[3]])
        t1_img_out,t1_feat = t1_model(img,input_affix,[])
        id_feat = add_feature(sid,t1_feat,target)
        t1_test_box.extend(id_feat.tolist())
        
        test_dataset.__add__(sid,t1_feat,[],[],target)
        del img
        del input_affix
        del t1_model
        torch.cuda.empty_cache()
        
    print("t1_train_box features.shape: ",np.array(t1_train_box).shape)
    print("t1_valid_box features.shape: ",np.array(t1_valid_box).shape)
    print("t1_test_box features.shape: ",np.array(t1_test_box).shape)
       
    return train_dataset, valid_dataset, test_dataset,t1_train_box,t1_valid_box,t1_test_box


def load_feature_model(best_model_path,input_channel,feature_align):
    model = ResNet18(num_classes=2,input_channel=input_channel,use_radiomics=False,feature_align=feature_align)
    
    try:
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['state_dict'])
    except:
        print("no pretrained path")

    return model

def get_pred(output): #[[0.1,0.9], [0.8,0.2]]→ [1,0]
    pred = output.cpu()  
    pred=pred.max(1,keepdim=True)[1]
    pred=convert(pred).cpu()  
    return pred 


def get_corrects(output, target):
    target = target.data.cpu()
    pred = get_pred(output)
    correct=pred.eq(target).sum().item()
    return pred,correct



def save_checkpoint(state, is_best, out_dir, model_name,pps,epoch,fold):
    #checkpoint_path = out_dir+model_name+'_'+opt.lossfunc+opt.ps+pps+"_fold_epoch_"+str(fold)+"_"+str(epoch)+'_checkpoint.pth.tar'
    best_model_path = out_dir+model_name+'_'+opt.lossfunc+opt.ps+pps+'_best_model.pth.tar'

    #print("checkpoint_path: ",checkpoint_path)
    #torch.save(state, checkpoint_path)
    if is_best:    
        print("save_model_path: ",best_model_path)
        torch.save(state, best_model_path)
        print("=======>   This is the best model !!! It has been saved!!!!!!\n\n")

def load_curr_best_checkpoint(model,out_dir,model_name,pps):
    best_model_path = out_dir+model_name+'_'+opt.lossfunc+opt.ps+pps+'_best_model.pth.tar'
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def convert(original): #[[0],[1],[0],[0]]→ [0,1,0,0]
    target=torch.Tensor(len(original))
    for i in range(len(original)):
        target[i]=original[i][0]
    target=target.type(torch.LongTensor).to(DEVICE)
    return target

def cal_metrics(CM):
    tn=CM[0][0]
    tp=CM[1][1]
    fp=CM[0][1]
    fn=CM[1][0] 
    num = CM.shape[0]
    num=2
    tps = [0 for i in range(num)]
    fps = [0 for i in range(num)]
    fns = [0 for i in range(num)]
    tns = [0 for i in range(num)]
    acc=np.sum(np.diag(CM)/np.sum(CM))
    mets = [] 
    for i in range(num):
        tp_0 = CM[i][i]
        fp_0 = sum(CM[j][i] for j in range(num) if  j != i)
        fn_0 = sum(CM[i][j] for j in range(num) if j!=i)
        tn_0 = sum(sum(CM[j][k] for j in range(num) if j!=i)for k in range(num) if k!=i)
        
        sen_0=tp_0/(tp_0+fn_0)   
        pre_0=tp_0/(tp_0+fp_0)       
        F1_0= (2*sen_0*pre_0)/(sen_0+pre_0)      
        spe_0 = tn_0/(tn_0+fp_0)
       
        met_0= Metrics()
        met_0.update(a=0,b=0,c=0,acc=acc,sen=sen_0,pre=pre_0,F1=F1_0,spe=spe_0,auc=0.0,CM=CM)
        mets.append(met_0)
    pre = sum(mets[i].pre for i in range(num))/num
    sen = sum(mets[i].sen for i in range(num))/num
    spe = sum(mets[i].spe for i in range(num))/num
    F1 = sum(mets[i].F1 for i in range(num))/num
    if num==2:
        met_0 = Metrics()
        met_0.update(a=0,b=0,c=0,acc=0,sen=0,pre=0,F1=0,spe=0,auc=0.0,CM=None)
        mets.append(met_0)
    return acc,pre,sen,spe,F1,mets
    


def validate(valid_loader, model, criterion,labels,multi_M):
    
    losses = AverageMeter()
    Loss_0 = AverageMeter()
    Loss_1 = AverageMeter()
    Loss_2 = AverageMeter()
    Loss_3 = AverageMeter()

    maes = AverageMeter()
    CM=0
    total_target=[]
    total_out=[]
    total_pred=[]
    total_sid = []
    model.eval() #because if allow model.eval, the bn wouldn't work, and the train_set'result would be different(model.train & model.eval)

    with torch.no_grad():
        for i, stuff  in enumerate(valid_loader):
        
            (t1_feature,t2_feature,t1c_feature ,sid,target) = stuff
            target = torch.from_numpy(np.expand_dims(target,axis=1))
            input_t1_feature = t1_feature.to(DEVICE)
            input_t2_feature = t2_feature.to(DEVICE)
            input_t1c_feature = t1c_feature.to(DEVICE)

            input_target = target.squeeze(1).to(DEVICE)

            out = model(input_t1_feature,input_t2_feature,input_t1c_feature)
            print("validate input_target: ",input_target)
            print("out: ",out)
            spare_criterion = nn.CrossEntropyLoss().to(DEVICE)
            loss = spare_criterion(out,input_target)

            losses.update(loss*input_t1_feature.size(0),input_t1_feature.size(0))
            Loss_0.update(0 ,input_t1_feature.size(0))
            Loss_1.update(0 ,input_t1_feature.size(0))
            Loss_2.update(0 ,input_t1_feature.size(0))
            Loss_3.update(0,input_t1_feature.size(0))

            pred, mae = get_corrects(output=out, target=target)
            print("valid_pred:",pred)
            print("out: ",out)
            print("target: ",target)
            maes.update(mae, input_t1_feature.size(0))

            # collect every output/pred/target, combine them together for total metrics'calculation
            total_target.extend(target)
            total_out.extend(torch.softmax(out,dim=1).cpu().numpy())
            total_pred.extend(pred.cpu().numpy())
 
            CM+=confusion_matrix(target.cpu(), pred.cpu(),labels= labels)
        
        print("total_sid: ",total_sid)
        a=torch.tensor(total_target)
        b=torch.tensor(total_out)
        c=torch.tensor(total_pred)    
        # total metrics'calcultion
        print("a",a)
        print("b",b)
        print("c",c)

        auc = roc_auc_score(a.cpu(),b.cpu()[:,1])  
        acc,pre,sen,spe,F1,mets = cal_metrics(CM)
        F1 = f1_score(a.cpu(),c.cpu(),average="weighted")
        pre = precision_score(a.cpu(),c.cpu(),average="weighted")
        sen = recall_score(a.cpu(),c.cpu(),average="weighted")
        
        auc = roc_auc_score(a.cpu(),b.cpu()[:,1])

        met= Metrics()

        met.update(a=a,b=b,c=c, acc=acc,sen=sen,pre=pre,F1=F1,spe=spe,auc=auc,CM=CM)

        #mets=[met_0,met_1,met_2]
        
        return losses.avg,losses, maes.avg, maes, met,[Loss_0.avg,Loss_1.avg,Loss_2.avg,Loss_3.avg],mets
    

def train(train_loader, model, criterion, optimizer, epoch,labels):
    
    losses = AverageMeter()
    maes = AverageMeter()
    Loss_0 = AverageMeter()
    Loss_1 = AverageMeter()
    Loss_2 = AverageMeter()
    Loss_3 = AverageMeter()


    for i, stuff  in enumerate(train_loader):
        
        (t1_feature,t2_feature,t1c_feature ,sid,target) = stuff
        target = torch.from_numpy(np.expand_dims(target,axis=1))


        input_t1_feature = t1_feature.to(DEVICE)
        input_t2_feature = t2_feature.to(DEVICE)
        input_t1c_feature = t1c_feature.to(DEVICE)

        input_target = target.squeeze(1).to(DEVICE)

        model.train()
        model.zero_grad()

        out = model(input_t1_feature,input_t2_feature,input_t1c_feature)
        
        spare_criterion = nn.CrossEntropyLoss()
        print("input_target: ",input_target)
        print("out: ",out)
        loss = spare_criterion(out,input_target)

        losses.update(loss*input_t1_feature.size(0),input_t1_feature.size(0))
        Loss_0.update(0 ,input_t1_feature.size(0))
        Loss_1.update(0 ,input_t1_feature.size(0))
        Loss_2.update(0 ,input_t1_feature.size(0))
        Loss_3.update(0,input_t1_feature.size(0))


        pred, mae = get_corrects(output=out, target=target)
        print("train_pred:",pred)
        print("out: ",out)
        print("target: ",target)
        maes.update(mae, input_t1_feature.size(0))
        
        CM=confusion_matrix(target.cpu(), pred.cpu(),labels= labels)
        
        acc,pre,sen,spe,F1,mets  = cal_metrics(CM)


        if i%opt.print_freq ==0:
            print(
                'Epoch: [{0} / {1}]   [step {2}/{3}]\t'
                  'Loss ({loss.avg:.4f})\t'
                  'Acc ({acc.avg:.4f})\t'
                  'Acc2 ({acc2:.4f})\t'
                  'sen ({sen:.4f})\t'
                  'pre ({pre:.4f})\t'
                  'F1 ({F1:.4f})\t'
                  'spe ({spe:.4f})\t'.format
                  ( epoch, opt.epochs, i, len(train_loader)
                  , loss=losses, acc=maes, acc2=acc, sen=sen, pre=pre, F1=F1, spe=spe )
                )
            print("CM: ",CM)
        loss.backward()
        print("loss:",loss," ; loss.grad: ",loss.grad,"; output is leaf? ",out.is_leaf," ; out.grad:",out.grad)

        optimizer.step()
    #total metrics'calcultion
    acc,pre,sen,spe,F1 ,mets = cal_metrics(CM)

    
    met= Metrics()
    met.update(a=0,b=0,c=0, auc=0, acc=acc,sen=sen,pre=pre,F1=F1,spe=spe,CM=CM)

    return losses.avg,losses, maes.avg, maes, met,[Loss_0.avg,Loss_1.avg,Loss_2.avg,Loss_3.avg]



def main(output_path):

    begin_time = time.time()
    json_path = os.path.join(opt.output_dir, 'hyperparameter.json')
    with open(json_path,'w') as jsf:
        jsf.write(json.dumps(vars(opt)
                                , indent=4
                                , separators=(',',':')))
    # record metrics into a txt
    record_file_path= opt.output_dir+opt.model+'_'+opt.lossfunc+opt.ps+'_4_record.txt'
    record_file = open(record_file_path, "a")
    
    

    print("=========== start train the brain age estimation model =========== \n")
    print(" ==========> Using {} processes for data loader.".format(opt.num_workers))

    #load the training_data  and test_data (training_data will be splited later for cross_validation)

    multi_M = True
   
    
    load_data_time = time.time()-begin_time
    print("[....Loading data OK....]: %dh, %dmin, %ds "%(int(load_data_time/3600),int(load_data_time/60),int(load_data_time%60)))
    print("[....Loading data OK....]: %dh, %dmin, %ds "%(int(load_data_time/3600),int(load_data_time/60),int(load_data_time%60)),file=record_file)
    
    print("[... loading basic settings ...]")
    if multi_M:
        print("<Begin to multi-modality train!>")
    else:
        print("<Begin to single-modality train!>")
        
    begin_time = time.time()

    epochs = opt.epochs
    usesecond = False
    num_classes = 2
    input_channel = 1
    
    target_names =  ['class 0', 'class 1','class 2'] if num_classes==3  else ['class 0', 'class 1']
    labels = [0,1,2] if num_classes==3 else [0,1]

    


    sum_writer = tensorboardX.SummaryWriter(opt.output_dir)
    print(" ==========> All settled. Training is getting started...")
    print(" ==========> Training takes {} epochs.".format(epochs))
    print(" ==========> output_dir: ",opt.output_dir)
    print(" ==========> task: ",num_classes)

    # split the training_data into K fold with StratifiedKFold(shuffle = True)
    k=3
    splits=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)

    # to record metrics: the best_acc of k folds, best_acc of each fold
    foldperf={}
    fold_best_auc,best_fold,best_epoch=-1,1,0
    fold_record_valid_metrics,fold_record_matched_test_metrics,fold_aucs,test_fold_aucs,fold_record_matched_train_metrics,train_fold_aucs=[],[],[],[],[],[]
    fold_best_statedict = None  

    chores_time = time.time()-begin_time
    print("[....Chores time....]: %dh, %dmin, %ds "%(int(chores_time/3600),int(chores_time/60),int(chores_time%60)))
    begin_time = time.time()

    #================= begin to train, choose 1 of k folds as validation =================================
    print("======================== start train ================================================ \n")
  
    total_comman_total_file=[1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 27, 28, 29, 30, 31, 34, 35, 36, 37, 39, 40, 42, 43, 44, 46, 48, 49, 51, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 95, 96, 97, 100, 102, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122]
    test_comman_total_file=[3, 4, 8, 16, 20, 22, 25, 26, 38, 45, 47, 56, 64, 65, 66, 75, 93, 94, 98, 99, 101, 111, 120]
    y_box = [0 if i<61 else 1 for i in total_comman_total_file] 
    """
    data_path=opt.data_path
    test_data_path = opt.testdata_path
    total_file = DIY_Folder_3M(num_classes = opt.num_classes,data_path=data_path,train_form = opt.train_form,root_mask_radiomics = opt.root_bbx_path,use_radiomics=opt.use_radiomics,istest=False,sec_pair_orig=opt.sec_pair_orig,multiclass = opt.multiclass,vice_path = ""
                                   ,multi_M = True,t1_path = opt.t1_path
                                   , t2_path=opt.t2_path, t1c_path = opt.t1c_path)

    test_file = DIY_Folder_3M(num_classes = opt.num_classes,data_path=test_data_path,train_form = opt.train_form,root_mask_radiomics = opt.test_root_bbx_path,use_radiomics=opt.use_radiomics,istest=True,sec_pair_orig= opt.sec_pair_orig,multiclass = opt.multiclass
                                  ,multi_M = True,t1_path = opt.t1_test_path
                                   , t2_path=opt.t2_test_path, t1c_path = opt.t1c_test_path)
    fake_test_data,test_sids = test_file.select_dataset(data_idx=[i for i in range(len(test_file))], aug=False,use_secondimg=opt.usesecond,noSameRM=opt.noSameRM, usethird = opt.usethird,comman_total_file=test_comman_total_file)
    print("fake_test_sids: ",test_sids)
    fake_test_loader= torch.utils.data.DataLoader(fake_test_data
                                                , batch_size = opt.batch_size
                                                , num_workers = opt.num_workers
                                                , pin_memory = True
                                                , drop_last = False
                                                )
    k=3
    splits=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)
    
    arries = ["T1","T2","T1C"]
    for total_count, ps in enumerate(arries):
        if total_count!=0:
            continue
        result_collection={}
        for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(total_comman_total_file)),y_box)):
            fold_result_collection={}
            fake_vali_data,vali_sids=total_file.select_dataset(data_idx=val_idx, aug=False,use_secondimg=False,noSameRM=opt.noSameRM, usethird=False,comman_total_file=total_comman_total_file)

            fake_train_data,train_sids=total_file.select_dataset(data_idx=train_idx, aug=False,aug_form="",use_secondimg=False,noSameRM=False, usethird=opt.usethird,comman_total_file=total_comman_total_file)


            fake_train_loader = DataLoader(fake_train_data, batch_size=opt.batch_size, num_workers = opt.num_workers
                                                    , pin_memory = True
                                                    , drop_last = False)

            fake_valid_loader = DataLoader(fake_vali_data, batch_size=opt.batch_size, num_workers = opt.num_workers
                                                    , shuffle = True
                                                    , pin_memory = True
                                                    , drop_last = False)

            best_model_t1c_fold_path = "/home/chenxr/Pineal_region/after_12_08/Results/old_T1C/Two/pretrained_constrain_composed_batchavg_ResNet18/model_result/ResNet18_BatchAvg_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar"%fold 
            best_model_t2_fold_path = "/home/chenxr/Pineal_region/after_12_08/Results/old_T2/Two/new_pretrained_selfKL_composed_ResNet18/ResNet18_SelfKL_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar"%fold
            best_model_t1_fold_path = "/home/chenxr/Pineal_region/after_12_08/Results/old_T1/Two/new_real_batchavg_constrain_composed_ResNet18/ResNet18_SelfKL_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar"%fold
            if ps == "T1":
                best_path = best_model_t1_fold_path
            elif ps =="T2":
                best_path = best_model_t2_fold_path
            else:
                best_path = best_model_t1c_fold_path
            t1model=load_feature_model(best_path,input_channel,feature_align=True)
            #t2model=load_feature_model(best_model_t2_fold_path,input_channel,feature_align=True)
            #t1cmodel=load_feature_model(best_model_t1c_fold_path,input_channel,feature_align=True)
            t1model=t1model.to(DEVICE)
            #t2model=t2model.to(DEVICE1)
            #t1cmodel=t1cmodel.to(DEVICE2)
            
            train_dataset, valid_dataset, test_dataset,t1_train_box,t1_valid_box,t1_test_box = build_feature_csv(t1model,fake_train_loader, fake_valid_loader,fake_test_loader,ps=ps)
            save_feature(t1_train_box,t1_valid_box,t1_test_box)
            result_collection[fold]=fold_result_collection
    """
    total_comman_total_file=[1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 27, 28, 29, 30, 31, 34, 35, 36, 37, 39, 40, 42, 43, 44, 46, 48, 49, 51, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 95, 96, 97, 100, 102, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122]
    test_comman_total_file=[3, 4, 8, 16, 20, 22, 25, 26, 38, 45, 47, 56, 64, 65, 66, 75, 93, 94, 98, 99, 101, 111, 120]
    y_box = [0 if i<61 else 1 for i in total_comman_total_file] 
    k=3
    splits=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)
    csv_root="/home/chenxr/Pineal_region/after_12_08/Results/saved_features/"
    
    
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(total_comman_total_file)),y_box)):
        print("train_idx: ",train_idx)
        train_sids = [total_comman_total_file[i] for i in train_idx]
        print("train_sids: ",train_sids)
        t1_train_path=os.path.join(csv_root,"T1_%d_train.csv"%fold)
        t2_train_path=os.path.join(csv_root,"T2_%d_train.csv"%fold)
        t1c_train_path=os.path.join(csv_root,"T1C_%d_train.csv"%fold)
        t1_valid_path=os.path.join(csv_root,"T1_%d_valid.csv"%fold)
        t2_valid_path=os.path.join(csv_root,"T2_%d_valid.csv"%fold)
        t1c_valid_path=os.path.join(csv_root,"T1C_%d_valid.csv"%fold)
        t1_test_path=os.path.join(csv_root,"T1_%d_test.csv"%fold)
        t2_test_path=os.path.join(csv_root,"T2_%d_test.csv"%fold)
        t1c_test_path=os.path.join(csv_root,"T1C_%d_test.csv"%fold)
        train_dataset = New_Feature_Folder(2,t1_train_path,t2_train_path,t1c_train_path)
        valid_dataset = New_Feature_Folder(2,t1_valid_path,t2_valid_path,t1c_valid_path)
        test_dataset = New_Feature_Folder(2,t1_test_path,t2_test_path,t1c_test_path)
        
        print("test_dataset.getsid   :",test_dataset.getsid())
        print("test_comman_total_file:",test_comman_total_file)
        print("train_dataset.getsid   :",train_dataset.getsid())
        print("train_sids             :",train_sids)
        if sorted(test_dataset.getsid())!=test_comman_total_file:
            assert 3==4
            
        if sorted(train_dataset.getsid())!=train_sids:
            assert 3==4
             
        torch.cuda.empty_cache()
        #model=Self_Attention()
        model = Transformer(514,256)
        model = model.to(DEVICE)
        criterion =  nn.CrossEntropyLoss().to(DEVICE)
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers = opt.num_workers
                                                , pin_memory = True
                                                , drop_last = False
                                                ,shuffle=True)

        valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, num_workers = opt.num_workers
                                                , shuffle = True
                                                , pin_memory = True
                                                , drop_last = False)
        test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers = opt.num_workers
                                                , shuffle = True
                                                , pin_memory = True
                                                , drop_last = False)
    
        optimizer = optim.Adam(model.parameters(), lr = opt.lr, weight_decay = opt.weight_decay)#decay=0

        # 设置warm up的轮次为20次
        warm_up_iter = opt.warm_up_iter
        lr_step = opt.lr_step
        T_max = opt.epochs	# 周期
        lr_max = opt.max_lr	# 最大值: 0.1
        lr_min = opt.min_lr	# 最小值: 1e-5

        # 为param_groups[0] (即model.layer2) 设置学习率调整规则 - Warm up + Cosine Anneal
        lambda0 = lambda cur_iter: (cur_iter % 10+1 )*10**(cur_iter // 5)*1e-4 if  cur_iter < warm_up_iter else \
                                   (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.1

        print("step lr :",(lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (20-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.1)
        # LambdaLR
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
        
        early_stopping = EarlyStopping(patience = opt.patience, verbose=True)
        history = {'train_loss': [], 'valid_loss': [],'test_loss':[],'train_acc':[],'valid_acc':[], 'test_acc':[],'train_auc':[],'test_auc':[],'valid_auc':[],'lr':[]}


        
        train_loss_box ,train_acc_box= [],[]
        valid_loss_box,valid_acc_box=[],[]
        t_acc_record,v_acc_record=[],[]
        test_acc_record,test_loss_box,test_acc_box=[],[],[]

        saved_metrics,saved_epoch=[],[]

        best_auc, sofar_valid_acc=-1,-1

        best_statedict = deepcopy(model.state_dict())
        best_epoch=0


        for epoch in range(opt.epochs):

            train_loss, train_losses, train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3]= train(train_loader = train_loader
                                                , model = model
                                                , criterion = criterion
                                                , optimizer = optimizer
                                                , epoch = epoch
                                                , labels = labels)

            train_loss_box.append(train_loss.detach().cpu())
            train_acc_box.append(train_acc)
            t_acc_record.append(train_accs.list)
            #================================== every epoch's metrics record =================================================

            train_loss, train_losses, validating_train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3],train_mets= validate(valid_loader = train_loader, model = model, criterion = criterion, labels = labels, multi_M= multi_M)
            train_a,train_b,train_c = train_met.a, train_met.b, train_met.c
            train_pre, train_rec, train_F1, train_spe = train_met.pre,train_met.sen,train_met.F1, train_met.spe
            print("[Each Epoch]: training_train_acc: ",train_acc, " ; validating_acc: ",validating_train_acc)
            
            valid_loss, valid_losses, valid_acc, valid_accs, valid_met,[valid_loss_0,valid_loss_1,valid_loss_2,valid_loss_3],valid_mets= validate(valid_loader = valid_loader, model = model, criterion = criterion, labels = labels, multi_M= multi_M)
            valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c
            valid_pre, valid_rec, valid_F1, valid_spe = valid_met.pre,valid_met.sen, valid_met.F1, valid_met.spe
            valid_met_0,valid_met_1,valid_met_2 = valid_mets[0],valid_mets[1],valid_mets[2]

            test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3],test_mets= validate(valid_loader = test_loader, model = model, criterion = criterion, labels = labels, multi_M= multi_M)
            test_a,test_b,test_c = test_met.a, test_met.b, test_met.c
            test_pre,test_rec, test_F1,test_spe = test_met.pre,test_met.sen, test_met.F1, test_met.spe
            test_met_0,test_met_1,test_met_2 = test_mets[0],test_mets[1],test_mets[2]
            

            valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu()[:,1])
            test_auc = roc_auc_score(test_a.cpu(),test_b.cpu()[:,1])
            train_auc = roc_auc_score(train_a.cpu(),train_b.cpu()[:,1])

            valid_loss_box.append(valid_loss.detach().cpu())
            valid_acc_box.append(valid_acc)
            test_loss_box.append(test_loss)       
            test_acc_box.append(test_acc)

            
            v_acc_record.append(valid_accs.list)       
            test_acc_record.append(test_accs.list)  

            scheduler.step()# step lr

            for param_group in optimizer.param_groups:
                print("\n*learning rate {:.2e}*\n" .format(param_group['lr']))
                history['lr'].append(param_group['lr'])
            #print("train/loss: ",train_loss)
            #print("valid/loss: ",valid_loss)
            #print("train/acc: ",train_acc)
            #print("valid/acc: ",valid_acc)

            sum_write_Mark=opt.sum_write_Mark

            # if two_class, valid_met_2 ==zero
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"loss", {'train':train_loss,'valid':valid_loss,'test':test_loss},epoch)
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"acc", {'train':train_acc,'valid':valid_acc,'test':test_acc},epoch)

            #sum_writer.add_scalar(opt.model+str(fold)+sum_write_Mark+"train/auc", train_auc,epoch)
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"auc", {'valid':valid_auc,'test':test_auc},epoch)    

            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"pre", {'train':train_pre,'valid':valid_pre,'test':test_pre},epoch)
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"F1", {'train':train_F1,'valid':valid_F1,'test':test_F1},epoch)
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"recall", {'train':train_rec,'valid':valid_rec,'test':test_rec},epoch)
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"spe", {'train':train_spe,'valid':valid_spe,'test':test_spe},epoch)

            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"train_subloss", {'loss_0':train_loss_0,'loss_1':train_loss_1,'loss_2':train_loss_2,'loss_3':train_loss_3},epoch)
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"loss_0", {'train':train_loss_0,'valid':valid_loss_0,'test':test_loss_0},epoch)
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"loss_1", {'train':train_loss_1,'valid':valid_loss_1,'test':test_loss_1},epoch)
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"loss_2", {'train':train_loss_2,'valid':valid_loss_2,'test':test_loss_2},epoch)
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"loss_3", {'train':train_loss_3,'valid':valid_loss_3,'test':test_loss_3},epoch)

            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"valid-pre", {'0':valid_met_0.pre,'1':valid_met_1.pre,'2':valid_met_2.pre},epoch)
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"valid-sen", {'0':valid_met_0.sen,'1':valid_met_1.sen,'2':valid_met_2.sen},epoch)
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"valid-F1", {'0':valid_met_0.F1,'1':valid_met_1.F1,'2':valid_met_2.F1},epoch)
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"valid-recall", {'0':valid_met_0.sen,'1':valid_met_1.sen,'2':valid_met_2.sen},epoch)

            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"test-F1", {'0':test_met_0.F1,'1':test_met_1.F1,'2':test_met_2.F1},epoch)
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"test-pre", {'0':test_met_0.pre,'1':test_met_1.pre,'2':test_met_2.pre},epoch)
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"test-sen", {'0':test_met_0.sen,'1':test_met_1.sen,'2':test_met_2.sen},epoch)
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"test-recall", {'0':test_met_0.sen,'1':test_met_1.sen,'2':test_met_2.sen},epoch)

            is_best=False

            
            if valid_auc > best_auc:
                best_auc = valid_auc
                sofar_valid_acc = valid_acc
                sofar_valid_auc = valid_auc
                sofar_valid_metrics = valid_met
                sofar_test_metrics = test_met
                sofar_test_auc = test_auc
                sofar_train_metrics=train_met
                sofar_train_auc=train_auc
                

                is_best=True
                saved_metrics.append(valid_acc)
                saved_epoch.append(epoch)
                best_epoch=epoch
                best_statedict=deepcopy(model.state_dict())

                pps="_k-fold-sub-fold-"+str(fold)+"_"
                print("【FOLD: %d】====> Best at epoch %d, valid auc: %f , valid acc: %f\n"%(fold,epoch, valid_auc, valid_acc))
                save_checkpoint({ 'epoch': epoch
                                , 'arch': opt.model
                                , 'state_dict': best_statedict
                                , 'fold':fold}
                                , is_best
                                , opt.output_dir
                                , model_name = opt.model
                                , pps=pps
                                , fold=fold
                                , epoch=epoch
                                )
            early_stopping(valid_auc)
            if early_stopping.early_stop:
                print("====== Early stopping ====")
                break


            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Valid Loss:{:.3f} AVG Training Acc {:.2f} % AVG Valid Acc {:.2f} %".format(epoch + 1,
                                                                                                                opt.epochs,
                                                                                                                train_loss,
                                                                                                                valid_loss,
                                                                                                                train_acc,
                                                                                                                valid_acc))
            if epoch==0:
                print("Begining Epoch:{}/{} AVG Training Loss:{:.3f} AVG Valid Loss:{:.3f} ||| AVG Training Acc {:.2f} AVG Valid Acc {:.2f} ||| AVG Valid Auc {:.2f}%".format(epoch + 1,opt.epochs,train_loss,valid_loss,train_acc,valid_acc, valid_auc),file=record_file)

            history['train_loss'].append(train_loss.cpu().detach().numpy())
            history['valid_loss'].append(valid_loss.cpu().detach().numpy())
            history['train_acc'].append(train_acc)
            history['valid_acc'].append(valid_acc)
            #history['train_auc'].append(train_auc)
            history['valid_auc'].append(valid_auc)        
        
        if fold_best_statedict==None:
            fold_best_statedict=best_statedict
        is_best=False
        if fold_best_auc < best_auc:
            print("fold: %d, change fold_best_auc from %f to %f "%(fold,fold_best_auc,best_auc))
            fold_best_auc = best_auc
            best_fold = fold
            fold_best_statedict=best_statedict

            is_best=True
            saved_metrics.append(fold_best_auc)
            saved_epoch.append(best_epoch)
            pps="_total_best_"
            print("【Total best fresh!】====> Best at fold %d, epoch %d, valid auc: %f  acc: %f\n"%(fold,epoch, best_auc, sofar_valid_acc))
            save_checkpoint({ 'epoch': best_epoch
                            , 'arch': opt.model
                            , 'state_dict': fold_best_statedict
                            , 'fold': best_fold}                
                            , is_best
                            , opt.output_dir
                            , model_name = opt.model
                            , pps=pps
                            , fold=best_fold
                            , epoch=best_epoch
                            )
        fold_aucs.append(sofar_valid_auc)
        fold_record_valid_metrics.append(sofar_valid_metrics)
        fold_record_matched_test_metrics.append(sofar_test_metrics)
        test_fold_aucs.append(sofar_test_auc)
        
        fold_record_matched_train_metrics.append(sofar_train_metrics)
        train_fold_aucs.append(sofar_train_auc)
        
        
         
    
        # to print & record the best result of this fold
        pps="_k-fold-sub-fold-"+str(fold)+"_"
        model = load_curr_best_checkpoint(model,out_dir = opt.output_dir,model_name = opt.model,pps = pps)
        #model.load_state_dict(best_statedict)        
        train_loss, train_losses, train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3],train_mets =validate(valid_loader = train_loader
                                    , model = model
                                    , criterion = criterion
                                    , labels = labels
                                    , multi_M= multi_M)
        train_a,train_b,train_c = train_met.a, train_met.b, train_met.c

        valid_loss, valid_losses, valid_acc, valid_accs, valid_met,[valid_loss_0,valid_loss_1,valid_loss_2,valid_loss_3],valid_mets= validate(valid_loader = valid_loader
                                    , model = model
                                    , criterion = criterion
                                    , labels = labels
                                    , multi_M= multi_M)
        valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c

        test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3] ,test_mets = validate(valid_loader = test_loader
                                    , model = model
                                    , criterion = criterion
                                    , labels = labels
                                    , multi_M= multi_M)
        test_a,test_b,test_c = test_met.a, test_met.b, test_met.c

        if opt.num_classes==3:
            valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu() ,multi_class = 'ovo',labels=[0,1,2])
            test_auc = roc_auc_score(test_a.cpu(),test_b.cpu() ,multi_class = 'ovo',labels=[0,1,2])
            train_auc = roc_auc_score(train_a.cpu(),train_b.cpu() ,multi_class = 'ovo',labels=[0,1,2])
        else:
            valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu()[:,1])
            test_auc = roc_auc_score(test_a.cpu(),test_b.cpu()[:,1])
            train_auc = roc_auc_score(train_a.cpu(),train_b.cpu()[:,1])
            
        recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_epoch",record_file = None,ps_thing=[best_fold,fold,fold_best_auc,best_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets, labels = labels)
        recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_epoch",record_file = record_file,ps_thing=[best_fold,fold,fold_best_auc,best_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets, labels = labels)


        plt.subplot(2,3,1)
        plt.plot(history['train_loss'],'r--',label='t_loss')
        plt.plot(history['valid_loss'],'b--',label='v_loss')
        plt.title("train_valid_loss")


        plt.subplot(2,3,3)
        plt.plot(history['train_acc'],'r--',label='t_acc')
        plt.plot(history['valid_acc'],'b--',label='v_acc')
        plt.title("train_valid_acc")


        plt.subplot(2,3,5)
        plt.plot(history['lr'],'b--',label='learning_rate')
        plt.title("learning_rate")

        plt.savefig(opt.pic_output_dir+opt.model+'_'+opt.lossfunc+opt.ps+'_k-fold-'+str(fold)+'_loss_acc.png')

        #plt.show()
        plt.close()
        

        foldperf['fold{}'.format(fold+1)] = history 


    os.system('echo " === TRAIN mae mtc:{:.5f}" >> {}'.format(train_loss, output_path))

    # to get averaged metrics of k folds
    record_avg(fold_record_valid_metrics,fold_record_matched_test_metrics,record_file,fold_aucs,test_fold_aucs,fold_record_matched_train_metrics,train_fold_aucs)

    # to print & record the best result of total
    model.load_state_dict(fold_best_statedict)        
    train_loss, train_losses, train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3],train_mets =validate(valid_loader = train_loader
                                , model = model
                                , criterion = criterion
                                , labels = labels
                                , multi_M= multi_M)
    train_a,train_b,train_c = train_met.a, train_met.b, train_met.c

    valid_loss, valid_losses, valid_acc, valid_accs, valid_met,[valid_loss_0,valid_loss_1,valid_loss_2,valid_loss_3],valid_mets= validate(valid_loader = valid_loader
                                , model = model
                                , criterion = criterion
                                , labels = labels
                                , multi_M= multi_M)
    valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c

    test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3] ,test_mets= validate(valid_loader = test_loader
                                , model = model
                                , criterion = criterion
                                , labels = labels
                                , multi_M= multi_M)
    test_a,test_b,test_c = test_met.a, test_met.b, test_met.c

    valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu()[:,1])
    test_auc = roc_auc_score(test_a.cpu(),test_b.cpu()[:,1])
    train_auc = roc_auc_score(train_a.cpu(),train_b.cpu()[:,1])

    recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_fold",record_file = None,ps_thing=[best_fold,best_fold,fold_best_auc,fold_best_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets, labels = labels)
    recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_fold",record_file = record_file,ps_thing=[best_fold,best_fold,fold_best_auc,fold_best_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets, labels = labels)


    torch.cuda.empty_cache()
    sum_writer.close()
    record_file.close()
    print("=======training end========")
    end_time = time.time()
    print("Total training time: ",end_time-begin_time)



def record_avg(fold_record_valid_metrics,fold_record_matched_test_metrics,record_file,fold_aucs,test_fold_aucs,fold_record_matched_train_metrics,train_fold_aucs):

    fold_accs,fold_f1s,fold_pres,fold_recalls,fold_spes=[],[],[],[],[]
    test_fold_accs,test_fold_f1s,test_fold_pres,test_fold_recalls,test_fold_spes=[],[],[],[],[]
    train_fold_accs,train_fold_f1s,train_fold_pres,train_fold_recalls,train_fold_spes=[],[],[],[],[]


    for i, metrics in enumerate(fold_record_valid_metrics):
        acc,F1, pre, rec,spe = metrics.acc,metrics.F1, metrics.pre, metrics.sen, metrics.spe
        fold_accs.append(acc)
        fold_f1s.append(F1)
        fold_pres.append(pre)
        fold_recalls.append(rec)
        fold_spes.append(spe)
    for i,metrics in enumerate(fold_record_matched_test_metrics):
        acc,F1, pre, rec,spe = metrics.acc,metrics.F1, metrics.pre, metrics.sen, metrics.spe
        test_fold_accs.append(acc)
        test_fold_f1s.append(F1)
        test_fold_pres.append(pre)
        test_fold_recalls.append(rec)
        test_fold_spes.append(spe)
        
    
    for i,metrics in enumerate(fold_record_matched_train_metrics):
        acc,F1, pre, rec,spe = metrics.acc,metrics.F1, metrics.pre, metrics.sen, metrics.spe
        train_fold_accs.append(acc)
        train_fold_f1s.append(F1)
        train_fold_pres.append(pre)
        train_fold_recalls.append(rec)
        train_fold_spes.append(spe)
        
    
    record_file.write("\nTrain<== acc records , avg: %f ==>"%(mean(train_fold_accs)))
    record_file.write(str(train_fold_accs))    
    record_file.write("<== auc records , avg: %f ==>"%(mean(train_fold_aucs)))
    record_file.write(str(train_fold_aucs))
    record_file.write("<== F1 records , avg: %f ==>"%(mean(train_fold_f1s)))
    record_file.write(str(train_fold_f1s))
    record_file.write("<== pre records , avg: %f ==>"%(mean(train_fold_pres)))
    record_file.write(str(train_fold_pres))
    record_file.write("<== recall records , avg: %f ==>"%(mean(train_fold_recalls)))
    record_file.write(str(train_fold_recalls))
    record_file.write("<== spe records , avg: %f ==>"%(mean(train_fold_spes)))
    record_file.write(str(train_fold_spes))


    
    record_file.write("\nvALID<== acc records , avg: %f ==>"%(mean(fold_accs)))
    record_file.write(str(fold_accs))    
    record_file.write("<== auc records , avg: %f ==>"%(mean(fold_aucs)))
    record_file.write(str(fold_aucs))
    record_file.write("<== F1 records , avg: %f ==>"%(mean(fold_f1s)))
    record_file.write(str(fold_f1s))
    record_file.write("<== pre records , avg: %f ==>"%(mean(fold_pres)))
    record_file.write(str(fold_pres))
    record_file.write("<== recall records , avg: %f ==>"%(mean(fold_recalls)))
    record_file.write(str(fold_recalls))
    record_file.write("<== spe records , avg: %f ==>"%(mean(fold_spes)))
    record_file.write(str(fold_spes))

    record_file.write("\nTEST<== acc records , avg: %f ==>"%(mean(test_fold_accs)))
    record_file.write(str(test_fold_accs))    
    record_file.write("<== auc records , avg: %f ==>"%(mean(test_fold_aucs)))
    record_file.write(str(test_fold_aucs))
    record_file.write("<== F1 records , avg: %f ==>"%(mean(test_fold_f1s)))
    record_file.write(str(test_fold_f1s))
    record_file.write("<== pre records , avg: %f ==>"%(mean(test_fold_pres)))
    record_file.write(str(test_fold_pres))
    record_file.write("<== recall records , avg: %f ==>"%(mean(test_fold_recalls)))
    record_file.write(str(test_fold_recalls))
    record_file.write("<== spe records , avg: %f ==>"%(mean(test_fold_spes)))
    record_file.write(str(test_fold_spes))

    print("[Avg]====acc===auc======F1====pre===recall===spe")
    print("[train]=%.3f-|-%.3f-|-%.3f-|-%.3f-|-%.3f-|-%.3f"%(mean(train_fold_accs),mean(train_fold_aucs),mean(train_fold_f1s),mean(train_fold_pres),mean(train_fold_recalls),mean(train_fold_spes)))
    print("[valid]=%.3f-|-%.3f-|-%.3f-|-%.3f-|-%.3f-|-%.3f"%(mean(fold_accs),mean(fold_aucs),mean(fold_f1s),mean(fold_pres),mean(fold_recalls),mean(fold_spes)))
    print("[test]==%.3f-|-%.3f-|-%.3f-|-%.3f-|-%.3f-|-%.3f"%(mean(test_fold_accs),mean(test_fold_aucs),mean(test_fold_f1s),mean(test_fold_pres),mean(test_fold_recalls),mean(test_fold_spes)))


def  recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="",record_file = None,ps_thing=[],target_names=[],train_mets=None,valid_mets=None,test_mets=None,labels=[0,1,2]):
    
    train_a, train_c,train_acc,train_pre, train_rec, train_F1, train_spe = train_met.a, train_met.c,train_met.acc,train_met.pre,train_met.sen,train_met.F1, train_met.spe
    valid_a,valid_c,valid_acc,valid_pre, valid_rec, valid_F1, valid_spe = valid_met.a,  valid_met.c, valid_met.acc,valid_met.pre,valid_met.sen, valid_met.F1, valid_met.spe
    test_a, test_c, test_acc, test_pre,  test_rec,   test_F1,test_spe   =  test_met.a,   test_met.c, test_met.acc,      test_met.pre,test_met.sen, test_met.F1, test_met.spe

    [best_fold,fold,fold_best_auc,best_auc,best_epoch] = ps_thing

    if record_form == "best_fold":
        print("\n============ Total End ============\n")
        if record_file!=None: print("\n============ Total End ============\n",file=record_file)

    head_str = "\n[Fold end]" if record_form=="best_epoch" else "\n[Total end]"
    if record_file == None:
        print(head_str+"best_fold %d & now_fold %d  |||| best_vali_auc %f  & now_vali_auc %f ==========================================\n"%(best_fold, fold, fold_best_auc,best_auc ))
        print("\n================================== [Fold: %d [Train] best at epoch %d ] ==========================================\n"%(fold, best_epoch))                                                                                                    
        print(classification_report(train_a.cpu(), train_c.cpu(), target_names=target_names, labels= labels))
        print("\n================================== [Fold: %d [Valid] best at epoch %d ] ==========================================\n"%(fold, best_epoch)) 
        print(classification_report(valid_a.cpu(), valid_c.cpu(), target_names=target_names, labels= labels))
        print("\n================================== [Fold: %d [Test] best at epoch %d ] ==========================================\n"%(fold, best_epoch)) 
        print(classification_report(test_a.cpu(), test_c.cpu(), target_names=target_names, labels= labels))

        print("\n================================== [Fold: %d AUC best at epoch %d ] ==========================================\n"%(fold,best_epoch))
        print("[train]: %f \t[valid]: %f \t[test]: %f \n"%(train_auc, valid_auc, test_auc))
        print("\n================================== [Fold: %d acc at epoch %d ] ==========================================\n"%(fold,best_epoch))
        print("[train]: %f \t[valid]: %f \t[test]: %f \n"%(train_acc, valid_acc, test_acc))
        print("\n================================== [Fold: %d F1 best at epoch %d ] ==========================================\n"%(fold,best_epoch))
        print("[train]: %f \t[valid]: %f \t[test]: %f \n"%(train_F1, valid_F1, test_F1))
        print("\n================================== [Fold: %d spe best at epoch %d ] ==========================================\n"%(fold,best_epoch))
        print("[train]: %f \t[valid]: %f \t[test]: %f \n"%(train_spe, valid_spe, test_spe))       
        print("\n================================== [Fold: %d sen best at epoch %d ] ==========================================\n"%(fold,best_epoch))
        print("[train]: %f \t[valid]: %f \t[test]: %f \n"%(train_rec, valid_rec, test_rec))
        
        print("[train]: <pre_0>: %f \t<pre_1>: %f \t<pre_2>: %f \n"%(train_mets[0].pre,train_mets[1].pre,train_mets[2].pre))
        print("[valid]: <pre_0>: %f \t<pre_1>: %f \t<pre_2>: %f \n"%(valid_mets[0].pre,valid_mets[1].pre,valid_mets[2].pre))
        print("[test] : <pre_0>: %f \t<pre_1>: %f \t<pre_2>: %f \n"%(test_mets[0].pre,test_mets[1].pre,test_mets[2].pre))
        print("[train_avg_pre]: %f \t[valid_avg_pre]: %f \t[test_avg_pre]: %f \t"%((train_mets[0].pre+train_mets[1].pre+train_mets[2].pre)/2,(valid_mets[0].pre+valid_mets[1].pre+valid_mets[2].pre)/2,(test_mets[0].pre+test_mets[1].pre+test_mets[2].pre)/2))
 
        print("[train]: <recall_0>: %f \t<recall_1>: %f \t<recall_2>: %f \n"%(train_mets[0].sen,train_mets[1].sen,train_mets[2].sen))
        print("[valid]: <recall_0>: %f \t<recall_1>: %f \t<recall_2>: %f \n"%(valid_mets[0].sen,valid_mets[1].sen,valid_mets[2].sen))
        print("[test] : <recall_0>: %f \t<recall_1>: %f \t<recall_2>: %f \n"%(test_mets[0].sen,test_mets[1].sen,test_mets[2].sen))
        print("[train_avg_recall]: %f \t[valid_avg_recall]: %f \t[test_avg_recall]: %f \t"%((train_mets[0].sen+train_mets[1].sen+train_mets[2].sen)/2,(valid_mets[0].sen+valid_mets[1].sen+valid_mets[2].sen)/2,(test_mets[0].sen+test_mets[1].sen+test_mets[2].sen)/2))
 
        print("[train]: <F1_0>: %f \t<F1_1>: %f \t<F1_2>: %f \n"%(train_mets[0].F1,train_mets[1].F1,train_mets[2].F1))
        print("[valid]: <F1_0>: %f \t<F1_1>: %f \t<F1_2>: %f \n"%(valid_mets[0].F1,valid_mets[1].F1,valid_mets[2].F1))
        print("[test] : <F1_0>: %f \t<F1_1>: %f \t<F1_2>: %f \n"%(test_mets[0].F1,test_mets[1].F1,test_mets[2].F1))


        print("[train_avg_F1]: %f \t[valid_avg_F1]: %f \t[test_avg_F1]: %f \t"%((train_mets[0].F1+train_mets[1].F1+train_mets[2].F1)/2,(valid_mets[0].F1+valid_mets[1].F1+valid_mets[2].F1)/2,(test_mets[0].F1+test_mets[1].F1+test_mets[2].F1)/2))

        print("[train]: <spe_0>: %f \t<spe_1>: %f \t<spe_2>: %f \n"%(train_mets[0].spe,train_mets[1].spe,train_mets[2].spe))     
        print("[valid]: <spe_0>: %f \t<spe_1>: %f \t<spe_2>: %f \n"%(valid_mets[0].spe,valid_mets[1].spe,valid_mets[2].spe))
        print("[test] : <spe_0>: %f \t<spe_1>: %f \t<spe_0>: %f \n"%(test_mets[0].spe,test_mets[1].spe,test_mets[2].spe))
        print("[train_avg_spe]: %f \t[valid_avg_spe]: %f \t[test_avg_spe]: %f \t"%((train_mets[0].spe+train_mets[1].spe+train_mets[2].spe)/2,(valid_mets[0].spe+valid_mets[1].spe+valid_mets[2].spe)/2,(test_mets[0].spe+test_mets[1].spe+test_mets[2].spe)/2))
   
    #===============================================================
    
    else:

        print(head_str+"best_fold %d & now_fold %d  |||| best_vali_auc %f  & now_vali_auc %f ==========================================\n"%(best_fold, fold, fold_best_auc,best_auc),file=record_file)
        print("\n================================== [Fold: %d [Train] best at epoch %d loss: %f ] ==========================================\n"%(fold, best_epoch,train_loss),file=record_file)                                                                                                    
        print(classification_report(train_a.cpu(), train_c.cpu(), target_names=target_names, labels= labels),file=record_file)
        print("\n================================== [Fold: %d [Valid] best at epoch %d loss: %f ] ==========================================\n"%(fold, best_epoch, valid_loss),file=record_file) 
        print(classification_report(valid_a.cpu(), valid_c.cpu(), target_names=target_names, labels= labels),file=record_file)
        print("\n================================== [Fold: %d [Test] best at epoch %d loss: %f ] ==========================================\n"%(fold, best_epoch, test_loss),file=record_file) 
        print(classification_report(test_a.cpu(), test_c.cpu(), target_names=target_names, labels= labels),file=record_file)

        print("\n================================== [Fold: %d AUC best at epoch %d ] ==========================================\n"%(fold,best_epoch),file=record_file)
        print("[train]: %f \t[valid]: %f \t[test]: %f \n"%(train_auc, valid_auc, test_auc),file=record_file)

        print("\n================================== [Fold: %d acc at epoch %d ] ==========================================\n"%(fold,best_epoch),file=record_file)
        print("[train]: %f \t[valid]: %f \t[test]: %f \n"%(train_acc, valid_acc, test_acc),file=record_file)
        print("\n================================== [Fold: %d F1 best at epoch %d ] ==========================================\n"%(fold,best_epoch),file=record_file)
        print("[train]: %f \t[valid]: %f \t[test]: %f \n"%(train_F1, valid_F1, test_F1),file=record_file)
        print("\n================================== [Fold: %d spe best at epoch %d ] ==========================================\n"%(fold,best_epoch),file=record_file)
        print("[train]: %f \t[valid]: %f \t[test]: %f \n"%(train_spe, valid_spe, test_spe),file=record_file)       
        print("\n================================== [Fold: %d sen best at epoch %d ] ==========================================\n"%(fold,best_epoch),file=record_file)
        print("[train]: %f \t[valid]: %f \t[test]: %f \n"%(train_rec, valid_rec, test_rec),file=record_file)

        print("[test] : <pre_0>: %f \t<pre_1>: %f \t<pre_2>: %f \n"%(test_mets[0].pre,test_mets[1].pre,test_mets[2].pre),file=record_file)
        print("[valid]: <pre_0>: %f \t<pre_1>: %f \t<pre_2>: %f \n"%(valid_mets[0].pre,valid_mets[1].pre,valid_mets[2].pre),file=record_file)
        print("[train]: <pre_0>: %f \t<pre_1>: %f \t<pre_2>: %f \n"%(train_mets[0].pre,train_mets[1].pre,train_mets[2].pre),file=record_file)
        print("[train_avg_pre]: %f \t[valid_avg_pre]: %f \t[test_avg_pre]: %f \n"%((train_mets[0].pre+train_mets[1].pre+train_mets[2].pre)/2,(valid_mets[0].pre+valid_mets[1].pre+valid_mets[2].pre)/2,(test_mets[0].pre+test_mets[1].pre+test_mets[2].pre)/2),file=record_file)
 
        print("[train]: <recall_0>: %f \t<recall_1>: %f \t<recall_2>: %f \n"%(train_mets[0].sen,train_mets[1].sen,train_mets[2].sen),file=record_file)
        print("[valid]: <recall_0>: %f \t<recall_1>: %f \t<recall_2>: %f \n"%(valid_mets[0].sen,valid_mets[1].sen,valid_mets[2].sen),file=record_file)
        print("[test] : <recall_0>: %f \t<recall_1>: %f \t<recall_2>: %f \n"%(test_mets[0].sen,test_mets[1].sen,test_mets[2].sen),file=record_file)
        print("[train_avg_recall]: %f \t[valid_avg_recall]: %f \t[test_avg_recall]: %f \n"%((train_mets[0].sen+train_mets[1].sen+train_mets[2].sen)/2,(valid_mets[0].sen+valid_mets[1].sen+valid_mets[2].sen)/2,(test_mets[0].sen+test_mets[1].sen+test_mets[2].sen)/2),file=record_file)
 
        print("[train]: <F1_0>: %f \t<F1_1>: %f \t<F1_2>: %f \n"%(train_mets[0].F1,train_mets[1].F1,train_mets[2].F1),file=record_file)
        print("[valid]: <F1_0>: %f \t<F1_1>: %f \t<F1_2>: %f \n"%(valid_mets[0].F1,valid_mets[1].F1,valid_mets[2].F1),file=record_file)
        print("[test] : <F1_0>: %f \t<F1_1>: %f \t<F1_2>: %f \n"%(test_mets[0].F1,test_mets[1].F1,test_mets[2].F1),file=record_file)


        print("[train_avg_F1]: %f \t[valid_avg_F1]: %f \t[test_avg_F1]: %f \n"%((train_mets[0].F1+train_mets[1].F1+train_mets[2].F1)/2,(valid_mets[0].F1+valid_mets[1].F1+valid_mets[2].F1)/2,(test_mets[0].F1+test_mets[1].F1+test_mets[2].F1)/2),file=record_file)

        print("[train]: <spe_0>: %f \t<spe_1>: %f \t<spe_2>: %f \n"%(train_mets[0].spe,train_mets[1].spe,train_mets[2].spe),file=record_file)     
        print("[valid]: <spe_0>: %f \t<spe_1>: %f \t<spe_2>: %f \n"%(valid_mets[0].spe,valid_mets[1].spe,valid_mets[2].spe),file=record_file)
        print("[test] : <spe_0>: %f \t<spe_1>: %f \t<spe_0>: %f \n"%(test_mets[0].spe,test_mets[1].spe,test_mets[2].spe),file=record_file)
        print("[train_avg_spe]: %f \t[valid_avg_spe]: %f \t[test_avg_spe]: %f \n"%((train_mets[0].spe+train_mets[1].spe+train_mets[2].spe)/2,(valid_mets[0].spe+valid_mets[1].spe+valid_mets[2].spe)/2,(test_mets[0].spe+test_mets[1].spe+test_mets[2].spe)/2),file=record_file)
 
if __name__ == "__main__":
    output_path = os.path.join(opt.output_dir, 'result')
    print("output_path: ",output_path)

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    if not os.path.exists(opt.pic_output_dir):
        os.makedirs(opt.pic_output_dir)      
    print("=> training beigin. \n")
    #os.system('echo "train {}"  >>  {}'.format(datetime.datetime.now(),output_path))
    main(output_path)
