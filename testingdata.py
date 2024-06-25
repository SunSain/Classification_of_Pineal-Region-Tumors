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
from model.resnet_3d_noclinical import ResNet10,ResNet18,ResNet34,ResNet24,ResNet30

from model.diy_resnet_3d import DIY_ResNet10,DIY_ResNet18
import matplotlib.pyplot as plt
from utils2.weighted_CE import Weighted_CE

from copy import deepcopy
from utils2.self_KL import SelfKL
from utils2.single_batchavg import SingleBatchCriterion
from model.mix_attention  import MixTransformer
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
from model.transformer import Transformer
from model.concatenation_mlp import Concateation_MLP
from model.self_attention_concatenation_mlp import SelfAttention_MLP_Transformer
from model.self_attention_crossattention_mlp import SelfAttention_CrossAttention_Transformer
from model.transformer_ver2_ResMlp import CrossAttention_ResTransformer
from medcam import medcam

import pandas as pd

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_device(2)
    DEVICE=torch.device('cuda')
    print("DEVICE: ",torch.cuda.current_device())
else:
    DEVICE=torch.device('cpu')
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
print('DEVICE: ',DEVICE)

import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAM, \
                            ScoreCAM, \
                            GradCAMPlusPlus, \
                            AblationCAM, \
                            XGradCAM, \
                            EigenCAM, \
                            EigenGradCAM, \
                            LayerCAM, \
                            FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

def reshape_transform(tensor, height=14, width=14):
    # 去掉cls token
    result = tensor[:, 1:, :].reshape(tensor.size(0),
    height, width, tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def gradcam(model,input_tensor,use_cuda,target_category):
    cam = GradCAM(model=model,
                target_layers=[model.blocks[-1].norm1],
                # 这里的target_layer要看模型情况,
                # 比如还有可能是：target_layers = [model.blocks[-1].ffn.norm]
                use_cuda=use_cuda,
                reshape_transform=reshape_transform)
    image_path = "xxx.jpg"
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))

    # 预处理图像
    input_tensor = preprocess_image(rgb_img,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

    # 看情况将图像转换为批量形式
    # input_tensor = input_tensor.unsqueeze(0)
    if use_cuda:
        input_tensor = input_tensor.cuda()
    grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)
    grayscale_cam = grayscale_cam[0, :]

    # 将 grad-cam 的输出叠加到原始图像上
    visualization = show_cam_on_image(rgb_img, grayscale_cam)

    # 保存可视化结果
    cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
    cv2.imwrite('cam.jpg', visualization)

#/home/chenxr/Pineal_region/after_12_08/Results/old_T1C/Two/pretrained_constrain_composed_batchavg_ResNet18/model_result/ResNet18_BatchAvg_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar
#/home/chenxr/Pineal_region/after_12_08/Results/old_T2/Two/new_pretrained_batchavg_constrain_composed_ResNet18/model_result/
#/home/chenxr/Pineal_region/after_12_08/Results/old_T2/Two/new_pretrained_selfKL_composed_ResNet18/ResNet18_SelfKL_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar
#/home/chenxr/Pineal_region/after_12_08/Results/old_T1/Two/new_real_batchavg_constrain_composed_ResNet18/ResNet18_SelfKL_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar

def save_feature(t1_train_box,t1_valid_box,t1_test_box,fold,ps):
    feature_csv_output_dir = "/home/chenxr/Pineal_region/after_12_08/Results/saved_features_version"
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
        #train_dataset.__add__(sid,t1_feat,[],[],target)
        del img
        del input_affix
        del t1_model
        del t1_feat
        del t1_img_out
        del target
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
        
        #valid_dataset.__add__(sid,t1_feat,[],[],target)
        del img
        del input_affix
        del t1_model
        del t1_feat
        del t1_img_out
        del target
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
        
        #test_dataset.__add__(sid,t1_feat,[],[],target)
        del img
        del input_affix
        del t1_model
        del t1_feat
        del t1_img_out
        del target
        torch.cuda.empty_cache()
        
    print("t1_train_box features.shape: ",np.array(t1_train_box).shape)
    print("t1_valid_box features.shape: ",np.array(t1_valid_box).shape)
    print("t1_test_box features.shape: ",np.array(t1_test_box).shape)
       
    return train_dataset, valid_dataset, test_dataset,t1_train_box,t1_valid_box,t1_test_box

def load_cnn_feature_model(best_model_path,input_channel,feature_align):
    model = ResNet18(num_classes=2,input_channel=input_channel,use_radiomics=False,feature_align=feature_align)
    #print("Loading model: ",best_model_path)
    try:
        checkpoint = torch.load(best_model_path,map_location = DEVICE)
        if checkpoint['state_dict'].get( "fc_two.weight",None) !=None:
                     checkpoint['state_dict']['fc.weight']=checkpoint['state_dict']['fc_two.weight']
                     checkpoint['state_dict']['fc.bias']=checkpoint['state_dict']['fc_two.bias']
                     del checkpoint['state_dict']['fc_two.weight']
                     del checkpoint['state_dict']['fc_two.bias']
        model.load_state_dict(checkpoint['state_dict'])
        for param in model.parameters():
            param.requires_grad = False
    except:
        print("no cnn pretrained path: ",best_model_path)
        assert 0==1
    model.eval()
    return model

def load_feature_model(best_model_path,input_channel,feature_align):
    model = ResNet18(num_classes=2,input_channel=input_channel,use_radiomics=False,feature_align=feature_align)
    
    try:
        checkpoint = torch.load(best_model_path,map_location = DEVICE)
        if checkpoint['state_dict'].get( "fc_two.weight",None) !=None:
                     checkpoint['state_dict']['fc.weight']=checkpoint['state_dict']['fc_two.weight']
                     checkpoint['state_dict']['fc.bias']=checkpoint['state_dict']['fc_two.bias']
                     del checkpoint['state_dict']['fc_two.weight']
                     del checkpoint['state_dict']['fc_two.bias']
        model.load_state_dict(checkpoint['state_dict'])
    except:
        print("no attention pretrained path")

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



def load_curr_best_checkpoint(model,out_dir,model_name,fold):

    best_model_path = out_dir+'/ResNet18_CE_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar'%fold
    print("Loading model: ",best_model_path)
    checkpoint = torch.load(best_model_path,map_location = DEVICE)
    if checkpoint['state_dict'].get( "fc_two.weight",None) !=None:
                     checkpoint['state_dict']['fc.weight']=checkpoint['state_dict']['fc_two.weight']
                     checkpoint['state_dict']['fc.bias']=checkpoint['state_dict']['fc_two.bias']
                     del checkpoint['state_dict']['fc_two.weight']
                     del checkpoint['state_dict']['fc_two.bias']
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
    print("\ncalc_metric: ",CM," \n mets: ",mets)
    pre = sum(mets[i].pre for i in range(num))/num
    sen = sum(mets[i].sen for i in range(num))/num
    spe = sum(mets[i].spe for i in range(num))/num
    F1 = sum(mets[i].F1 for i in range(num))/num
    if num==2:
        met_0 = Metrics()
        met_0.update(a=0,b=0,c=0,acc=0,sen=0,pre=0,F1=0,spe=0,auc=0.0,CM=None)
        mets.append(met_0)
    return acc,pre,sen,spe,F1,mets
    

def validate2(valid_loader, model, criterion,labels,multi_M,lossfunc):
    
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
            (T1cimg,T1C_auged_data,T1img,T1_auged_data,T2img,T2_auged_data,sid,target,saffix,sradiomics) = stuff
            target = torch.from_numpy(np.expand_dims(target,axis=1))
            target = convert(target)
            total_sid.append(sid)


            t1_img =  T1img.to(DEVICE)
            t2_img = T2img.to(DEVICE)
            t1c_img = T1cimg.to(DEVICE)
            input_affix = saffix.to(DEVICE)
            
            input_target = target.to(DEVICE)
            #print("input_affix: ",infput_affix)
            #input_img = torch.reshape(input_img, [input_img.shape[0],1,input_img.shape[1],input_img.shape[2],input_img.shape[3]])
            
            out,feat = model(input_img,input_affix)
            
            
            spare_criterion = nn.CrossEntropyLoss()     
            spare_selfKL_criterion = SelfKL(num_classes=opt.num_classes,lambda_0=opt.lambda_0,lambda_1=opt.lambda_1,lambda_2=opt.lambda_2,lambda_3=opt.lambda_2, CE_or_KL=opt.CE_or_KL) 
            if(lossfunc == 'SelfKL' or lossfunc == 'Weighted_SelfKL'):
                loss,loss_0 ,loss_1,loss_2,loss_3 = criterion(out, target)
            elif(lossfunc=="BatchAvg"):
                loss_0 = spare_criterion(out,target)
                loss_1 = criterion(feat, target)
                loss_KL,_,_,_,_ =spare_selfKL_criterion(out,target)
                loss = loss_0+loss_1+loss_KL
                loss_2,loss_3 =0.,0.
                loss=loss_0+loss_1
            elif opt.lossfunc=="SingleBatchAvg_selfKL":
                loss_0 = spare_criterion(out,target)
                if len(sid)<2:
                    loss=0.0
                else:
                    loss_1 = criterion(allfeats)
                loss_2,loss_3=0.,0.
                loss=loss_0+loss_1
            else:
                print("Not selfkl, criterion ")
                loss= criterion(out, target)
                loss_0 ,loss_1,loss_2,loss_3 =0.,0.,0.,0.

            

            #input_img.size(0) = batch_size 
            # the CE's ouput is averaged, so 
            losses.update(loss*input_img.size(0),input_img.size(0))
            Loss_0.update(loss_0*input_img.size(0),input_img.size(0))
            Loss_1.update(loss_1*input_img.size(0),input_img.size(0))
            Loss_2.update(loss_2*input_img.size(0),input_img.size(0))
            Loss_3.update(loss_3*input_img.size(0),input_img.size(0))

            pred, mae = get_corrects(output=out, target=target)
            print("valid_pred:",pred)
            print("out: ",out)
            print("target: ",target)
            maes.update(mae, input_img.size(0))

            # collect every output/pred/target, combine them together for total metrics'calculation
            total_target.extend(target)
            total_out.extend(torch.softmax(out,dim=1).cpu().numpy())
            total_pred.extend(pred.cpu().numpy())
 
            CM+=confusion_matrix(target.cpu(), pred.cpu(),labels= labels)
            del input_img
            del input_affix
        
        print("total_sid: ",total_sid)
        a=torch.tensor(total_target)
        b=torch.tensor(total_out)
        c=torch.tensor(total_pred)    
        # total metrics'calcultion
        print("a",a)
        print("b",b)
        print("c",c)

        auc = roc_auc_score(a.cpu(),b.cpu()[:,1])
        
        new_CM= confusion_matrix(a.cpu(),c.cpu())

        
        acc,pre,sen,spe,F1,mets = cal_metrics(CM)
        F1 = f1_score(a.cpu(),c.cpu(),average="weighted")
        pre = precision_score(a.cpu(),c.cpu(),average="weighted")
        sen = recall_score(a.cpu(),c.cpu(),average="weighted")
        
        
        print("new_F1: ",F1)
        print("pre:",pre)
        print("recall: ",sen)
        print("new_CM:", new_CM)
        print("CM: ",CM)  
        print('Confusion Matirx : ',CM,'[Metrics]-Accuracy(mean): ' , acc,'- Sensitivity : ',sen*100,'- Specificity : ',spe*100,'- Precision: ',pre*100,'- F1 : ',F1*100,'- auc : ',auc*100)

        # Metrics is a DIY_package to store all metrics(acc, auc, F1,pre, recall, spe,CM, outpur, pred,target)
        met= Metrics()

        met.update(a=a,b=b,c=c, acc=acc,sen=sen,pre=pre,F1=F1,spe=spe,auc=auc,CM=CM)

        #mets=[met_0,met_1,met_2]
        
        return losses.avg,losses, maes.avg, maes, met,[Loss_0.avg,Loss_1.avg,Loss_2.avg,Loss_3.avg],mets  



def validate3(valid_loader, model, criterion,labels,multi_M, t1_best_path, t2_best_path, t1c_best_path,lossfunc):
    
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

    torch.cuda.empty_cache()
    t1model=load_cnn_feature_model(t1_best_path,input_channel=1,feature_align=True)
    t1_model = t1model.to(DEVICE)
    t2model=load_cnn_feature_model(t2_best_path,input_channel=1,feature_align=True)
    t2_model = t2model.to(DEVICE)
    t1cmodel=load_cnn_feature_model(t1c_best_path,input_channel=1,feature_align=True)
    t1c_model = t1cmodel.to(DEVICE)

    with torch.no_grad():
        for i, stuff  in enumerate(valid_loader):
            
            (T1cimg,T1C_auged_data,T1img,T1_auged_data,T2img,T2_auged_data,sid,target,saffix,sradiomics) = stuff
            target = torch.from_numpy(np.expand_dims(target,axis=1))
            target = convert(target)
            torch.cuda.empty_cache()

            t1_img =  T1img.to(DEVICE)
            t2_img = T2img.to(DEVICE)
            t1c_img = T1cimg.to(DEVICE)
            input_affix = saffix.to(DEVICE)
            print("CURRENT DEVICE: ",torch.cuda.current_device())
            #input_img = torch.reshape(input_img, [input_img.shape[0],1,input_img.shape[1],input_img.shape[2],input_img.shape[3]])
            t1_img_out,t1_feature = t1_model(t1_img,input_affix )
            t2_img_out,t2_feature = t2_model(t2_img,input_affix )
            t1c_img_out,t1c_feature = t1c_model(t1c_img,input_affix )

            input_t1_feature = t1_feature.to(DEVICE)
            input_t2_feature = t2_feature.to(DEVICE)
            input_t1c_feature = t1c_feature.to(DEVICE)

            input_target = target.to(DEVICE)

            out,feat = model(input_t1_feature,input_t2_feature,input_t1c_feature)

            t1_auged_img =  T1_auged_data.to(DEVICE)
            t2_auged_img = T2_auged_data.to(DEVICE)
            t1c_auged_img = T1C_auged_data.to(DEVICE)
            print("CURRENT DEVICE: ",torch.cuda.current_device())
            #input_img = torch.reshape(input_img, [input_img.shape[0],1,input_img.shape[1],input_img.shape[2],input_img.shape[3]])
            t1_img_auged_out,t1_auged_feature = t1_model(t1_auged_img,input_affix )
            t2_img_auged_out,t2_auged_feature = t2_model(t2_auged_img,input_affix )
            t1c_img_auged_out,t1c_auged_feature = t1c_model(t1c_auged_img,input_affix )

            input_t1_auged_feature = t1_auged_feature.to(DEVICE)
            input_t2_auged_feature = t2_auged_feature.to(DEVICE)
            input_t1c_auged_feature = t1c_auged_feature.to(DEVICE)

            input_target = target.to(DEVICE)

            auged_out,auged_feat = model(input_t1_auged_feature,input_t2_auged_feature,input_t1c_auged_feature)
            
            if(lossfunc== 'SelfKL' or lossfunc == 'Weighted_SelfKL'):
                spare_selfKL_criterion = SelfKL(num_classes=2,lambda_0=1,lambda_1=0,lambda_2=9,lambda_3=0, CE_or_KL=True) 
                loss,loss_0 ,loss_1,loss_2,loss_3 = spare_selfKL_criterion(out, target)
            elif(lossfunc=="SingleBatchAvg"):
                criterion=SingleBatchCriterion()
                if len(sid)<2:
                    loss=0.0
                else:
                    allfeats = torch.cat((feat[0],auged_feat[0]), 0)
                    print("allfeats_0: ",allfeats)
                    allfeats = allfeats.squeeze()
                    print("allfeats_1: ",allfeats)
                    loss = criterion(allfeats)
            else:
                loss=0.0
            spare_criterion = nn.CrossEntropyLoss()
            loss+= spare_criterion(out,target)


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
            print("out2: ",out)
            total_pred.extend(pred.cpu().numpy())
 
            CM+=confusion_matrix(target.cpu(), pred.cpu(),labels= labels)
        

        a=torch.tensor(total_target)
        b=torch.tensor(total_out)
        c=torch.tensor(total_pred)   
        print("total_sid: ",total_sid)
        print("a",a)
        print("b",b)
        print("c",c) 
        # total metrics'calcultion
        #print("a",a)
        #print("b",b)
        #print("c",c)

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
    



def validate(valid_loader, model, criterion,labels,multi_M,lossfunc):
    
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
            if multi_M:
                (img,masked_img,_,_,_,_,sid,target, saffix,sradiomics) = stuff
            else:
                (img,masked_img,sid,target, saffix,sradiomics) = stuff
            target = torch.from_numpy(np.expand_dims(target,axis=1))
            target = convert(target)
            total_sid.append(sid)


            input_img = img.to(DEVICE)
            input_affix = saffix.to(DEVICE)
            input_radiomics = sradiomics.to(DEVICE)
            input_target = target.to(DEVICE)
            #print("input_affix: ",infput_affix)
            #input_img = torch.reshape(input_img, [input_img.shape[0],1,input_img.shape[1],input_img.shape[2],input_img.shape[3]])
            
            out,feat = model(input_img,input_affix)
            
            
            spare_criterion = nn.CrossEntropyLoss()     
            spare_selfKL_criterion = SelfKL(num_classes=opt.num_classes,lambda_0=opt.lambda_0,lambda_1=opt.lambda_1,lambda_2=opt.lambda_2,lambda_3=opt.lambda_2, CE_or_KL=opt.CE_or_KL) 
            if(lossfunc == 'SelfKL' or lossfunc == 'Weighted_SelfKL'):
                loss,loss_0 ,loss_1,loss_2,loss_3 = criterion(out, target)
            elif(lossfunc=="BatchAvg"):
                loss_0 = spare_criterion(out,target)
                loss_1 = criterion(feat, target)
                loss_KL,_,_,_,_ =spare_selfKL_criterion(out,target)
                loss = loss_0+loss_1+loss_KL
                loss_2,loss_3 =0.,0.
                loss=loss_0+loss_1
            elif opt.lossfunc=="SingleBatchAvg_selfKL":
                loss_0 = spare_criterion(out,target)
                if len(sid)<2:
                    loss=0.0
                else:
                    loss_1 = criterion(allfeats)
                loss_2,loss_3=0.,0.
                loss=loss_0+loss_1
            else:
                print("Not selfkl, criterion ")
                loss= criterion(out, target)
                loss_0 ,loss_1,loss_2,loss_3 =0.,0.,0.,0.

            

            #input_img.size(0) = batch_size 
            # the CE's ouput is averaged, so 
            losses.update(loss*input_img.size(0),input_img.size(0))
            Loss_0.update(loss_0*input_img.size(0),input_img.size(0))
            Loss_1.update(loss_1*input_img.size(0),input_img.size(0))
            Loss_2.update(loss_2*input_img.size(0),input_img.size(0))
            Loss_3.update(loss_3*input_img.size(0),input_img.size(0))

            pred, mae = get_corrects(output=out, target=target)
            print("valid_pred:",pred)
            print("out: ",out)
            print("target: ",target)
            maes.update(mae, input_img.size(0))

            # collect every output/pred/target, combine them together for total metrics'calculation
            total_target.extend(target)
            total_out.extend(torch.softmax(out,dim=1).cpu().numpy())
            total_pred.extend(pred.cpu().numpy())
 
            CM+=confusion_matrix(target.cpu(), pred.cpu(),labels= labels)
            del input_img
            del input_affix
        
        print("total_sid: ",total_sid)
        a=torch.tensor(total_target)
        b=torch.tensor(total_out)
        c=torch.tensor(total_pred)    
        # total metrics'calcultion
        print("a",a)
        print("b",b)
        print("c",c)

        auc = roc_auc_score(a.cpu(),b.cpu()[:,1])
        
        new_CM= confusion_matrix(a.cpu(),c.cpu())

        
        acc,pre,sen,spe,F1,mets = cal_metrics(CM)
        F1 = f1_score(a.cpu(),c.cpu(),average="weighted")
        pre = precision_score(a.cpu(),c.cpu(),average="weighted")
        sen = recall_score(a.cpu(),c.cpu(),average="weighted")
        
        
        print("new_F1: ",F1)
        print("pre:",pre)
        print("recall: ",sen)
        print("new_CM:", new_CM)
        print("CM: ",CM)  
        print('Confusion Matirx : ',CM,'[Metrics]-Accuracy(mean): ' , acc,'- Sensitivity : ',sen*100,'- Specificity : ',spe*100,'- Precision: ',pre*100,'- F1 : ',F1*100,'- auc : ',auc*100)

        # Metrics is a DIY_package to store all metrics(acc, auc, F1,pre, recall, spe,CM, outpur, pred,target)
        met= Metrics()

        met.update(a=a,b=b,c=c, acc=acc,sen=sen,pre=pre,F1=F1,spe=spe,auc=auc,CM=CM)

        #mets=[met_0,met_1,met_2]
        
        return losses.avg,losses, maes.avg, maes, met,[Loss_0.avg,Loss_1.avg,Loss_2.avg,Loss_3.avg],mets  

#def main(output_dir_box,color_box,data_path,test_data_path,test_roc_save_path,label_box,title):


def multi_resnet_main(t1_path,t2_path,t1c_path,model_name,savepath,model_path,lossfunc):
    begin_time = time.time()
    num_classes = 2
    target_names =  ['class 0', 'class 1','class 2'] if num_classes==3  else ['class 0', 'class 1']
    labels = [0,1,2] if num_classes==3 else [0,1]
    print("======================== start testing ================================================ \n")
    test_comman_total_file=[3, 4, 8, 16, 20, 22, 25, 26, 38, 45, 47, 56, 64, 65, 66, 75, 93, 94, 98, 99, 101, 111, 120]

    test_data_path = t1_path

    test_file = DIY_Folder_3M(num_classes = opt.num_classes,data_path=test_data_path,train_form = None,root_mask_radiomics = opt.test_root_bbx_path,use_radiomics=opt.use_radiomics,istest=True,sec_pair_orig= opt.sec_pair_orig,multiclass = opt.multiclass
                                  ,multi_M = True,t1_path = opt.t1_test_path
                                   , t2_path=opt.t2_test_path, t1c_path = opt.t1c_test_path
                                   , extra_test=False,
                                   valid_sid_box=test_comman_total_file)
    test_loader= torch.utils.data.DataLoader(fake_test_data
                                                , batch_size = opt.batch_size
                                                , num_workers = opt.num_workers
                                                , pin_memory = True
                                                , drop_last = False
                                                )
    k=3
    load_data_time = time.time()-begin_time
    all_fpr=[]
    all_tpr=[]
    all_testauc=[]
    total_comman_total_file=[1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 27, 28, 29, 30, 31, 34, 35, 36, 37, 39, 40, 42, 43, 44, 46, 48, 49, 51, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 95, 96, 97, 100, 102, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122]

    
    testauc_record_file_path = savepath+"/new_test_ROC_2.txt"
    testauc_record_file=open(testauc_record_file_path,"w")

    use_clinical = True if not model_name=="Baseline" and not model_name=="Baseline_withBestAugmentation" else False 
    k=3
    splits=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)
    arries = ["T1","T2","T1C"]
    foldperf={}
    fold_best_auc,best_fold,best_epoch=-1,1,0
    fold_record_valid_metrics,fold_record_matched_test_metrics,fold_aucs,test_fold_aucs,fold_record_matched_train_metrics,train_fold_aucs=[],[],[],[],[],[]
    fold_record_extra_test_metrics=[]
    fold_record_extra_test_aucs=[]
    fold_record_total_test_withextra_test_metrics=[]
    total_test_withextra_test_fold_aucs=[]
    
    fold_test_tpr=[]
    fold_test_fpr=[]
    fold_test_roc=[]
    
    fold_test_CM=[]
    
    fold_best_statedict = None  

    result_collection={}
    tprs=[]
    mean_fpr=np.linspace(0,1,100)
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(total_comman_total_file)),y_box)):
        torch.cuda.empty_cache()
        fold_result_collection={}

        #model = Transformer(514,256)
        model = ResNet18(num_classes=num_classes,input_channel=3,use_radiomics=False,feature_align=False,use_clinical=use_clinical)
        
        pps="_k-fold-sub-fold-"+str(fold)+"_"
        model = load_curr_best_checkpoint(model,out_dir = model_path,model_name = "ResNet18",fold=fold)
        model = model.to(DEVICE)
        
        criterion =  nn.CrossEntropyLoss().to(DEVICE)

        # 设置warm up的轮次为20次

        train_loss_box ,train_acc_box= [],[]
        valid_loss_box,valid_acc_box=[],[]
        t_acc_record,v_acc_record=[],[]
        test_acc_record,test_loss_box,test_acc_box=[],[],[]

        saved_metrics,saved_epoch=[],[]

        best_auc, sofar_valid_acc=-1,-1

        best_epoch=0

        test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3],test_mets= validate2(valid_loader = test_loader, model = model, criterion = criterion, labels = labels, multi_M= multi_M,lossfunc=lossfunc)
        test_a,test_b,test_c = test_met.a, test_met.b, test_met.c
        test_pre,test_rec, test_F1,test_spe = test_met.pre,test_met.sen, test_met.F1, test_met.spe
        test_met_0,test_met_1,test_met_2 = test_mets[0],test_mets[1],test_mets[2]

        test_auc = roc_auc_score(test_a.cpu(),test_b.cpu()[:,1])
            

        #extra_test_loss, extra_test_losses, extra_test_acc, extra_test_accs, extra_test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3] ,extra_test_mets = validate(valid_loader = extra_test_loader, model = model, criterion = criterion, labels = labels, multi_M= multi_M)
        #extra_test_a,extra_test_b,extra_test_c = extra_test_met.a, extra_test_met.b, extra_test_met.c 
        
    
        #total_test_withextra_test_loss, total_test_withextra_test_losses, total_test_withextra_test_acc, test_accs, total_test_withextra_test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3] ,total_test_withextra_test_mets = validate(valid_loader = total_test_withextra_loader, model = model, criterion = criterion, labels = labels, multi_M= multi_M)
        #total_test_withextra_test_a,total_test_withextra_test_b,total_test_withextra_test_c = total_test_withextra_test_met.a, total_test_withextra_test_met.b, total_test_withextra_test_met.c

        print("test_a: ",test_a)
        print("test_b: ",test_b)
        print("test_c: ",test_c)
        fpr1, tpr1, thersholds = roc_curve(test_a.cpu(), test_b.cpu()[:,1])#用1的预测
        
        roc_auc1 = auc(fpr1, tpr1)
        print("test_roc_auc: ",roc_auc1)
        
        fold_test_tpr.append(tpr1)
        fold_test_fpr.append(fpr1)


        draw_heatmap(test_met.CM,title="fold_"+str(fold)+"_",save_path=savepath+'/'+str(fold))
        fold_record_matched_test_metrics.append(test_met)
        test_fold_aucs.append(test_auc)
        fold_test_CM.append(test_met.CM)
        
    os.system('echo " === TRAIN mae mtc:{:.5f}"'.format(test_loss))
    print("fold_test_tpr: ",fold_test_tpr)
    print("fold_test_fpr: ",fold_test_fpr)
    
    testauc_record_file.write("\nfold_test_fpr:\n")
    testauc_record_file.write(str(fold_test_fpr))
    testauc_record_file.write("\nfold_test_tpr:\n")
    testauc_record_file.write(str(fold_test_tpr))
    testauc_record_file.write("\ntest_fold_aucs:\n")
    testauc_record_file.write(str(test_fold_aucs))
    avg_testCM_save_path=savepath+"/"+"_test_CM.png"
    plt.figure(figsize=(16,16))

    sns.heatmap(np.mean(fold_test_CM,0), annot=True, square=True,cmap='Blues')
    plt.ylim(0, 3)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    

    plt.savefig(avg_testCM_save_path)
    plt.close()
    
    torch.cuda.empty_cache()
    record_file.close()
    testauc_record_file.close()
    print("=======training end========")
    end_time = time.time()
    print("Total training time: ",end_time-begin_time)

def multi_main(t1_path,t2_path,t1c_path,model_name,savepath,model_path,lossfunc):
    begin_time = time.time()
    num_classes = 2
    target_names =  ['class 0', 'class 1','class 2'] if num_classes==3  else ['class 0', 'class 1']
    labels = [0,1,2] if num_classes==3 else [0,1]
    print("======================== start testing ================================================ \n")
    test_comman_total_file=[3, 4, 8, 16, 20, 22, 25, 26, 38, 45, 47, 56, 64, 65, 66, 75, 93, 94, 98, 99, 101, 111, 120]
    #test_comman_total_file=[3]
    test_data_path = t1_path

    test_file = DIY_Folder_3M(num_classes = opt.num_classes,data_path=test_data_path,train_form = None,root_mask_radiomics = opt.test_root_bbx_path,use_radiomics=opt.use_radiomics,istest=True,sec_pair_orig= opt.sec_pair_orig,multiclass = opt.multiclass
                                  ,multi_M = True,t1_path = opt.t1_test_path
                                   , t2_path=opt.t2_test_path, t1c_path = opt.t1c_test_path
                                   , extra_test=False,
                                   valid_sid_box=test_comman_total_file)
    fake_test_data,test_sids = test_file.select_dataset(data_idx=[i for i in range(len(test_comman_total_file))], aug=False,use_secondimg=opt.usesecond,noSameRM=opt.noSameRM, usethird = opt.usethird,comman_total_file=test_comman_total_file)

    test_loader= torch.utils.data.DataLoader(fake_test_data
                                                , batch_size = opt.batch_size
                                                , num_workers = opt.num_workers
                                                , pin_memory = True
                                                , drop_last = False
                                                )
    k=3
    load_data_time = time.time()-begin_time
    all_fpr=[]
    all_tpr=[]
    all_testauc=[]
    total_comman_total_file=[1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 27, 28, 29, 30, 31, 34, 35, 36, 37, 39, 40, 42, 43, 44, 46, 48, 49, 51, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 95, 96, 97, 100, 102, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122]

    
    testauc_record_file_path = savepath+"/new_test_ROC_2.txt"
    testauc_record_file=open(testauc_record_file_path,"w")

    use_clinical = True if not model_name=="Baseline" and not model_name=="Baseline_withBestAugmentation" else False 
    k=3
    splits=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)
    arries = ["T1","T2","T1C"]
    foldperf={}
    fold_best_auc,best_fold,best_epoch=-1,1,0
    fold_record_valid_metrics,fold_record_matched_test_metrics,fold_aucs,test_fold_aucs,fold_record_matched_train_metrics,train_fold_aucs=[],[],[],[],[],[]
    fold_record_extra_test_metrics=[]
    fold_record_extra_test_aucs=[]
    fold_record_total_test_withextra_test_metrics=[]
    total_test_withextra_test_fold_aucs=[]
    
    fold_test_tpr=[]
    fold_test_fpr=[]
    fold_test_roc=[]
    
    fold_test_CM=[]
    
    fold_best_statedict = None  

    result_collection={}
    tprs=[]
    mean_fpr=np.linspace(0,1,100)
    y_box = [0 if i<61 else 1 for i in total_comman_total_file] 
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(total_comman_total_file)),y_box)):
        
        best_model_t1c_fold_path="/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_13/model_result/ResNet18_CE_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar"%fold
        
        best_model_t1_fold_path="/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_mask1_composed_RM_Singlebatchavg_37/model_result/ResNet18_CE_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar"%fold
        best_model_t2_fold_path="/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T2_mask1_composed_RM_Singlebatchavg_37/model_result/ResNet18_CE_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar"%fold

        
        t1_best_path = best_model_t1_fold_path
        t2_best_path = best_model_t2_fold_path
        t1c_best_path = best_model_t1c_fold_path

        torch.cuda.empty_cache()
        dim=514
        model_dict={
            'Multi_CrossAttention':Transformer(514,256,new_arch=False),
            'Multi_Feature_Concatenation':Concateation_MLP(514,2),
            'Multi_SelfAttention': SelfAttention_MLP_Transformer(514,256),
            'Multi_MixAttention':MixTransformer(514,256,w1=0.5,w2=0.5,w3=0.5)
        }
        model = model_dict[model_name]
        pps="_k-fold-sub-fold-"+str(fold)+"_"
        model = load_curr_best_checkpoint(model,out_dir =model_path,model_name =model_name,fold=fold)

        model = model.to(DEVICE)
        
        criterion =  nn.CrossEntropyLoss().to(DEVICE)

        # 设置warm up的轮次为20次

        train_loss_box ,train_acc_box= [],[]
        valid_loss_box,valid_acc_box=[],[]
        t_acc_record,v_acc_record=[],[]
        test_acc_record,test_loss_box,test_acc_box=[],[],[]

        saved_metrics,saved_epoch=[],[]

        best_auc, sofar_valid_acc=-1,-1

        best_epoch=0
        test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3],test_mets= validate3(valid_loader = test_loader, model = model, criterion = criterion, labels = labels, multi_M= True
                                                                                                                                                            , t1_best_path = best_model_t1_fold_path
                                                , t2_best_path = best_model_t2_fold_path
                                                , t1c_best_path = best_model_t1c_fold_path,
                                                lossfunc=lossfunc)

        test_a,test_b,test_c = test_met.a, test_met.b, test_met.c
        test_pre,test_rec, test_F1,test_spe = test_met.pre,test_met.sen, test_met.F1, test_met.spe
        test_met_0,test_met_1,test_met_2 = test_mets[0],test_mets[1],test_mets[2]

        test_auc = roc_auc_score(test_a.cpu(),test_b.cpu()[:,1])
            

        #extra_test_loss, extra_test_losses, extra_test_acc, extra_test_accs, extra_test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3] ,extra_test_mets = validate(valid_loader = extra_test_loader, model = model, criterion = criterion, labels = labels, multi_M= multi_M)
        #extra_test_a,extra_test_b,extra_test_c = extra_test_met.a, extra_test_met.b, extra_test_met.c 
        
    
        #total_test_withextra_test_loss, total_test_withextra_test_losses, total_test_withextra_test_acc, test_accs, total_test_withextra_test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3] ,total_test_withextra_test_mets = validate(valid_loader = total_test_withextra_loader, model = model, criterion = criterion, labels = labels, multi_M= multi_M)
        #total_test_withextra_test_a,total_test_withextra_test_b,total_test_withextra_test_c = total_test_withextra_test_met.a, total_test_withextra_test_met.b, total_test_withextra_test_met.c

        print("test_a: ",test_a)
        print("test_b: ",test_b)
        print("test_c: ",test_c)
        fpr1, tpr1, thersholds = roc_curve(test_a.cpu(), test_b.cpu()[:,1])#用1的预测
        
        roc_auc1 = auc(fpr1, tpr1)
        print("test_roc_auc: ",roc_auc1)
        
        fold_test_tpr.append(tpr1)
        fold_test_fpr.append(fpr1)


        draw_heatmap(test_met.CM,title="fold_"+str(fold)+"_",save_path=savepath+'/'+str(fold))

        fold_record_matched_test_metrics.append(test_met)
        test_fold_aucs.append(test_auc)
        fold_test_CM.append(test_met.CM)
        
    os.system('echo " === TRAIN mae mtc:{:.5f}"'.format(test_loss))
    print("fold_test_tpr: ",fold_test_tpr)
    print("fold_test_fpr: ",fold_test_fpr)
    
    testauc_record_file.write("\nfold_test_fpr:\n")
    testauc_record_file.write(str(fold_test_fpr))
    testauc_record_file.write("\nfold_test_tpr:\n")
    testauc_record_file.write(str(fold_test_tpr))
    testauc_record_file.write("\ntest_fold_aucs:\n")
    testauc_record_file.write(str(test_fold_aucs))
    avg_testCM_save_path=savepath+"/"+"_test_CM.png"
    plt.figure(figsize=(16,16))

    sns.heatmap(np.mean(fold_test_CM,0), annot=True, square=True,cmap='Blues')
    plt.ylim(0, 3)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    

    plt.savefig(avg_testCM_save_path)
    plt.close()
    testauc_record_file.close()    
    record_avg(fold_record_matched_test_metrics,test_fold_aucs)
    draw_all_roc(savepath,label=model_name,auc=mean(test_fold_aucs),title="")
    torch.cuda.empty_cache()

    print("=======training end========")
    end_time = time.time()
    print("Total training time: ",end_time-begin_time)

def single_main(testdata_path,model_name,savepath,model_path,lossfunc,label):
    begin_time = time.time()
    num_classes = 2
    target_names =  ['class 0', 'class 1','class 2'] if num_classes==3  else ['class 0', 'class 1']
    labels = [0,1,2] if num_classes==3 else [0,1]
    print("======================== start testing ================================================ \n")
    test_comman_total_file=[3, 4, 8, 16, 20, 22, 25, 26, 38, 45, 47, 56, 64, 65, 66, 75, 93, 94, 98, 99, 101, 111, 120]
    #test_comman_total_file=[3]
    test_data_path = testdata_path

    test_file = DIY_Folder_3M(num_classes = opt.num_classes,data_path=test_data_path,train_form = None,root_mask_radiomics = opt.test_root_bbx_path,use_radiomics=opt.use_radiomics,istest=True,sec_pair_orig= opt.sec_pair_orig,multiclass = opt.multiclass
                                  ,multi_M = True,t1_path = opt.t1_test_path
                                   , t2_path=opt.t2_test_path, t1c_path = opt.t1c_test_path
                                   , extra_test=False,
                                   valid_sid_box=test_comman_total_file)
    fake_test_data,test_sids = test_file.select_dataset(data_idx=[i for i in range(len(test_comman_total_file))], aug=False,use_secondimg=opt.usesecond,noSameRM=opt.noSameRM, usethird = opt.usethird,comman_total_file=test_comman_total_file)
    
    test_loader= torch.utils.data.DataLoader(fake_test_data
                                                , batch_size = opt.batch_size
                                                , num_workers = opt.num_workers
                                                , pin_memory = True
                                                , drop_last = False
                                                )
    k=3
    load_data_time = time.time()-begin_time
    all_fpr=[]
    all_tpr=[]
    all_testauc=[]
    total_comman_total_file=[1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 27, 28, 29, 30, 31, 34, 35, 36, 37, 39, 40, 42, 43, 44, 46, 48, 49, 51, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 95, 96, 97, 100, 102, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122]
    y_box = [0 if i<61 else 1 for i in total_comman_total_file] 
    
    testauc_record_file_path = savepath+"/new_test_ROC_2.txt"
    testauc_record_file=open(testauc_record_file_path,"w")

    use_clinical = True if not model_name=="Baseline" and not model_name=="Baseline_withBestAugmentation" else False 
    k=3
    splits=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)
    arries = ["T1","T2","T1C"]
    foldperf={}
    fold_best_auc,best_fold,best_epoch=-1,1,0
    fold_record_valid_metrics,fold_record_matched_test_metrics,fold_aucs,test_fold_aucs,fold_record_matched_train_metrics,train_fold_aucs=[],[],[],[],[],[]
    fold_record_extra_test_metrics=[]
    fold_record_extra_test_aucs=[]
    fold_record_total_test_withextra_test_metrics=[]
    total_test_withextra_test_fold_aucs=[]
    
    fold_test_tpr=[]
    fold_test_fpr=[]
    fold_test_roc=[]
    
    fold_test_CM=[]
    
    fold_best_statedict = None  

    result_collection={}
    tprs=[]
    mean_fpr=np.linspace(0,1,100)
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(total_comman_total_file)),y_box)):
        torch.cuda.empty_cache()
        fold_result_collection={}

        #model = Transformer(514,256)
        print("use_clinical: ",use_clinical)
        model = ResNet18(num_classes=num_classes,input_channel=1,use_radiomics=False,feature_align=False,use_clinical=use_clinical)
        
        pps="_k-fold-sub-fold-"+str(fold)+"_"
        model = load_curr_best_checkpoint(model,out_dir = model_path,model_name = "ResNet18",fold=fold)
        model = model.to(DEVICE)
        
        criterion =  nn.CrossEntropyLoss().to(DEVICE)

        # 设置warm up的轮次为20次

        train_loss_box ,train_acc_box= [],[]
        valid_loss_box,valid_acc_box=[],[]
        t_acc_record,v_acc_record=[],[]
        test_acc_record,test_loss_box,test_acc_box=[],[],[]

        saved_metrics,saved_epoch=[],[]

        best_auc, sofar_valid_acc=-1,-1

        best_epoch=0

        test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3],test_mets= validate(valid_loader = test_loader, model = model, criterion = criterion, labels = labels, multi_M= True,lossfunc=lossfunc)
        test_a,test_b,test_c = test_met.a, test_met.b, test_met.c
        test_pre,test_rec, test_F1,test_spe = test_met.pre,test_met.sen, test_met.F1, test_met.spe
        test_met_0,test_met_1,test_met_2 = test_mets[0],test_mets[1],test_mets[2]

        test_auc = roc_auc_score(test_a.cpu(),test_b.cpu()[:,1])
            

        #extra_test_loss, extra_test_losses, extra_test_acc, extra_test_accs, extra_test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3] ,extra_test_mets = validate(valid_loader = extra_test_loader, model = model, criterion = criterion, labels = labels, multi_M= multi_M)
        #extra_test_a,extra_test_b,extra_test_c = extra_test_met.a, extra_test_met.b, extra_test_met.c 
        
    
        #total_test_withextra_test_loss, total_test_withextra_test_losses, total_test_withextra_test_acc, test_accs, total_test_withextra_test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3] ,total_test_withextra_test_mets = validate(valid_loader = total_test_withextra_loader, model = model, criterion = criterion, labels = labels, multi_M= multi_M)
        #total_test_withextra_test_a,total_test_withextra_test_b,total_test_withextra_test_c = total_test_withextra_test_met.a, total_test_withextra_test_met.b, total_test_withextra_test_met.c

        print("test_a: ",test_a)
        print("test_b: ",test_b)
        print("test_c: ",test_c)
        fpr1, tpr1, thersholds = roc_curve(test_a.cpu(), test_b.cpu()[:,1])#用1的预测
        
        roc_auc1 = auc(fpr1, tpr1)
        print("test_roc_auc: ",roc_auc1)
        
        fold_test_tpr.append(tpr1)
        fold_test_fpr.append(fpr1)


        draw_heatmap(test_met.CM,title="fold_"+str(fold)+"_",save_path=savepath+'/'+str(fold))

        fold_record_matched_test_metrics.append(test_met)
        test_fold_aucs.append(test_auc)
        fold_test_CM.append(test_met.CM)
        
    os.system('echo " === TRAIN mae mtc:{:.5f}"'.format(test_loss))
    print("fold_test_tpr: ",fold_test_tpr)
    print("fold_test_fpr: ",fold_test_fpr)
    
    testauc_record_file.write("\nfold_test_fpr:\n")
    testauc_record_file.write(str(fold_test_fpr))
    testauc_record_file.write("\nfold_test_tpr:\n")
    testauc_record_file.write(str(fold_test_tpr))
    testauc_record_file.write("\ntest_fold_aucs:\n")
    testauc_record_file.write(str(test_fold_aucs))
    avg_testCM_save_path=savepath+"/"+"_test_CM.png"
    plt.figure(figsize=(16,16))

    sns.heatmap(np.mean(fold_test_CM,0), annot=True, square=True,cmap='Blues')
    plt.ylim(0, 3)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    

    plt.savefig(avg_testCM_save_path)
    plt.close()
    testauc_record_file.close()    
    draw_all_roc(savepath,label=label,auc=mean(test_fold_aucs),title="")
    
    torch.cuda.empty_cache()

    print("=======training end========")
    end_time = time.time()
    
    record_avg(fold_record_matched_test_metrics,test_fold_aucs)
    
    print("Total training time: ",end_time-begin_time)

def record_avg(fold_record_matched_test_metrics,test_fold_aucs):

    fold_accs,fold_f1s,fold_pres,fold_recalls,fold_spes=[],[],[],[],[]
    test_fold_accs,test_fold_f1s,test_fold_pres,test_fold_recalls,test_fold_spes=[],[],[],[],[]
    train_fold_accs,train_fold_f1s,train_fold_pres,train_fold_recalls,train_fold_spes=[],[],[],[],[]

    for i,metrics in enumerate(fold_record_matched_test_metrics):
        acc,F1, pre, rec,spe = metrics.acc,metrics.F1, metrics.pre, metrics.sen, metrics.spe
        test_fold_accs.append(acc)
        test_fold_f1s.append(F1)
        test_fold_pres.append(pre)
        test_fold_recalls.append(rec)
        test_fold_spes.append(spe)
        
    
    
    print("[Avg]====acc===auc======F1====pre===recall===spe")
    print("[test]==%.3f-|-%.3f-|-%.3f-|-%.3f-|-%.3f-|-%.3f"%(mean(test_fold_accs),mean(test_fold_aucs),mean(test_fold_f1s),mean(test_fold_pres),mean(test_fold_recalls),mean(test_fold_spes)))



def draw_all_roc(savepath,label,auc,title):
    import matplotlib
    import matplotlib.pyplot as plt 
    from matplotlib import font_manager 
    
        
    plt.figure(figsize=(9,9))
    
    testauc_record_file_path =savepath+"/new_test_ROC_2.txt"
    try:
        fold_test_fpr,fold_test_tpr,test_fold_aucs=read_txt(testauc_record_file_path)
        print("\nfold_test_fpr,fold_test_tpr,test_fold_aucs: ",test_fold_aucs)
    except:
        print("No new_test_ROC_2")
    
    #画单条ROC曲线
    plt.plot(fold_test_fpr, fold_test_tpr, color='r',linestyle="-", label=label+' (auc = {0:.3f})'.format(auc),lw=2)
#画坐标轴
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限,以免和边缘重合,更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.yticks( size = 20)
    plt.xticks( size = 20)
    plt.xlabel('False Positive Rate',fontsize=20, )
    plt.ylabel('True Positive Rate',fontsize=20,)  # 可以使用中文,但需要导入一些库即字体
    plt.title(title,fontsize=20)
    plt.plot([0, 1], [0, 1],'grey',linestyle="--",alpha=0.3)
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(savepath+"/ROC.png")
    plt.close()

import numpy as np
import re
import ast
from collections import OrderedDict
def read_txt(path):
    print("read_txt(path): ",path)
    fold_test_tpr=""
    fold_test_fpr=""
    test_fold_aucs=""
    with open(path, 'r') as file:
        lines = file.read()
    strs=""
    mark=0
    print("\nRead finished")
    special_chars = r"\b(fold_test_tpr:|fold_test_fpr:|test_fold_aucs:)\b"

    lines=lines.replace("fold_test_tpr:"," \nfold_test_tpr: \n")
    lines=lines.replace("fold_test_fpr:"," \nfold_test_fpr: \n")
    lines=lines.replace("test_fold_aucs:"," \ntest_fold_aucs: \n")
    lines  = re.sub(special_chars, r'\1\n', lines)

    lines = lines.split('\n')

    for line in lines:
        line=line.strip()
        if line=="fold_test_fpr:":
            if mark==1:
                break
            mark=1
            strs=""
        elif line=="fold_test_tpr:":
            fold_test_fpr=strs
            strs=""
        elif line=="test_fold_aucs:":
            fold_test_tpr=strs
            strs=""     
        else:
            strs+=line.strip()  
    test_fold_aucs=strs
    print("test_fold_aucs:",test_fold_aucs)
    
    s=test_fold_aucs    
    s = s.replace('array(', '')
    s = s.replace(')', '')
    test_fold_aucs = eval(s)

    s=fold_test_fpr    
    s = s.replace('array(', '')
    s = s.replace(')', '')
    fold_test_fpr = eval(s)
    
    
    s=fold_test_tpr    
    s = s.replace('array(', '')
    s = s.replace(')', '')
    fold_test_tpr = eval(s)
    
    fprdict={}
    fprcountdict={}
    for i,fprbox in enumerate(fold_test_fpr):
        tprbox=fold_test_tpr[i]
        for j, num in enumerate(fprbox):
            fpr=fprbox[j]
            tpr=tprbox[j]
            ini_tpr=fprdict.get(fpr,0)
            ini_tpr+=tpr
            fprdict[fpr]=ini_tpr
            
            count=fprcountdict.get(fpr,0)
            count+=1
            fprcountdict[fpr]=count
    print("fprdict: ",fprdict)
    fprdict = OrderedDict(sorted(fprdict.items()))
    new_fpr_box=[]
    new_tpr_box=[]
    for key in fprdict:
        val = fprdict[key]
        val = val/fprcountdict[key]
        new_fpr_box.append(key)
        new_tpr_box.append(val)
    
    mean_fpr = np.linspace(0, 1, 100)
    tprs=[]
    for i ,lis in enumerate(fold_test_fpr):
        mean_fpr = np.linspace(0, 1, 100)
        interp_tpr = np.interp(mean_fpr, fold_test_fpr[i], fold_test_tpr[i])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    return mean_fpr,mean_tpr,mean_auc


def  draw_heatmap(test_CM,title,save_path):
        
        test_save_path = save_path+"_test.png"

        plt.figure(figsize=(16,16))
        


    
        
        plt.subplot(1,1,1)
        sns.heatmap(test_CM, annot=True,square=True, cmap='Blues')

        plt.ylim(0, 3)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(title+"test")
        
        plt.savefig(test_save_path)
        plt.close()
 
def avg_cm():
    
    all_cm=[]
    pred=[1,0,0,1]
    target=[1,0,0,0]
    labels=[0,1]
    CM=confusion_matrix(target, pred,labels= labels)
    print("CM: ",CM)
    all_cm.append(CM)


    pred=[0,0,0,1]
    target=[1,0,0,1]
    labels=[0,1]
    CM=confusion_matrix(target, pred,labels= labels)
    print("CM: ",CM)
    all_cm.append(CM)
    
    pred=[1,0,0,1]
    target=[1,1,1,0]
    labels=[0,1]
    CM=confusion_matrix(target, pred,labels= labels)
    print("CM: ",CM)
    all_cm.append(CM)
    print("all_cm: ",all_cm)
    print(np.mean(all_cm,0))


if __name__ == "__main__":
    
    T1_path=opt.t1_test_path
    T2_path=opt.t2_test_path
    T1C_path=opt.t1c_test_path
    model_name="Multi_MixAttention"
    savepath="./T1"
    if os.path.exists(savepath):
        import shutil
        folder_path = savepath
        try:
            shutil.rmtree(folder_path)
            print(f"文件夹 {folder_path} 及其所有内容已被删除。")
        except Exception as e:
            print(f"删除文件夹时出错: {e}")
            
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    t1_mode_path="/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_SelfKL_7/model_result/"
    t2_mode_path=""
    t1c_mode_path=""
    multi_model="/opt/chenxingru/Pineal_region/after_12_08/230321_data/03_21_seg_No_regis_T1+C_best_Notest"
    lossfunc="SingleBatchAvg"
    multi_task=0
    if multi_task==1:
        multi_main(T1_path,T2_path,T1C_path,model_name,savepath,multi_model,lossfunc)
    else:
            
        if not T1_path=="T1_Dataset":
            single_main(T1_path,model_name,savepath,model_path=t1_mode_path,lossfunc=lossfunc,label="T1")
        if not T2_path=="T2_Dataset":
            single_main(T1_path,model_name,savepath,t2_mode_path,lossfunc,label="T2")
        if not T1C_path=="T1C_Dataset":
            single_main(T1C_path,model_name,savepath,t1c_mode_path,lossfunc,label="T1C")
