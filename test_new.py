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



def load_curr_best_checkpoint(model,out_dir,model_name,pps,hyf):

    best_model_path = out_dir+model_name+'_'+hyf["lossfunc"]+hyf["ps"]+pps+'_best_model.pth.tar'
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
            if(opt.lossfunc == 'SelfKL' or opt.lossfunc == 'Weighted_SelfKL'):
                loss,loss_0 ,loss_1,loss_2,loss_3 = criterion(out, target)
            elif(opt.lossfunc=="BatchAvg"):
                loss_0 = spare_criterion(out,target)
                loss_1 = criterion(feat, target)
                loss_KL,_,_,_,_ =spare_selfKL_criterion(out,target)
                loss = loss_0+loss_1+loss_KL
                loss_2,loss_3 =0.,0.
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
        
        """
        new_F1 = f1_score(a.cpu(),c.cpu(),average="weighted")
        pre = precision_score(a.cpu(),c.cpu(),average="weighted")
        recall = recall_score(a.cpu(),c.cpu(),average="weighted")
        """
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
    
def main(filepath_boxex,colorboxex,data_path,testdata_path,test_roc_save_path_boxex,labelboxex,title_boxex):
 
    begin_time = time.time()

    multi_M = True

    begin_time = time.time()

    num_classes = 2

    
    target_names =  ['class 0', 'class 1','class 2'] if num_classes==3  else ['class 0', 'class 1']
    labels = [0,1,2] if num_classes==3 else [0,1]

    # to record metrics: the best_acc of k folds, best_acc of each fold

    #================= begin to train, choose 1 of k folds as validation =================================
    print("======================== start testing ================================================ \n")
  
  
    total_comman_total_file=[1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 27, 28, 29, 30, 31, 34, 35, 36, 37, 39, 40, 42, 43, 44, 46, 48, 49, 51, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 95, 96, 97, 100, 102, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122]
    test_comman_total_file=[3, 4, 8, 16, 20, 22, 25, 26, 38, 45, 47, 56, 64, 65, 66, 75, 93, 94, 98, 99, 101, 111, 120]
                        
    #total_comman_total_file=[1, 2, 59, 60, 61, 62, 63, 121, 122] #32,33,41,50,53,83,89,103
    #test_comman_total_file=[3, 4, 65, 66]
    
    y_box = [0 if i<61 else 1 for i in total_comman_total_file] 
    data_path=data_path
    test_data_path = testdata_path
    total_file = DIY_Folder_3M(num_classes = opt.num_classes,data_path=data_path,train_form = None,root_mask_radiomics = opt.root_bbx_path,use_radiomics=opt.use_radiomics,istest=False,sec_pair_orig=opt.sec_pair_orig,multiclass = opt.multiclass,vice_path = ""
                                   ,multi_M = True,t1_path = opt.t1_path
                                   , t2_path=opt.t2_path, t1c_path = opt.t1c_path,
                                   valid_sid_box=total_comman_total_file
                                   )

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
    
    extra_test_file= []#[123,124,125,126,127,130,131,132,134,138,139,141,143,144,145,146,148,149,150,152,153,156]
    extra_test_data,extra_sids=test_file.select_dataset(data_idx=[i for i in range(len(extra_test_file))], aug=False,use_secondimg=opt.usesecond,noSameRM=opt.noSameRM, usethird = opt.usethird,comman_total_file=extra_test_file)
    extra_test_loader= torch.utils.data.DataLoader(extra_test_data
                                                , batch_size = opt.batch_size
                                                , num_workers = opt.num_workers
                                                , pin_memory = True
                                                , drop_last = False
                                                )
    
    total_test_withextra_file = test_comman_total_file+extra_test_file
    print("total_test_withextra_file: ",total_test_withextra_file)
    total_test_withextra_data,extra_sids=test_file.select_dataset(data_idx=[i for i in range(len(total_test_withextra_file))], aug=False,use_secondimg=opt.usesecond,noSameRM=opt.noSameRM, usethird = opt.usethird,comman_total_file=total_test_withextra_file)
    total_test_withextra_loader= torch.utils.data.DataLoader(total_test_withextra_data
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
    
    for indexex, output_dir_box in enumerate(filepath_boxex):
        
        color_box=colorboxex[indexex]
        test_roc_save_path=test_roc_save_path_boxex[indexex]
        label_box=labelboxex[indexex]
        title=title_boxex[indexex]
        
        for output_path in output_dir_box:
            
            json_path = os.path.join(output_path, 'hyperparameter.json')
            
            jsf=open(json_path,'r')
            hyf=json.load(jsf,strict=False)

            print("hyf: ",hyf)
            output_dir = hyf["output_dir"]
            record_file_path= hyf["output_dir"]+hyf["model"]+'_'+hyf["lossfunc"]+hyf["ps"]+'_Testing_record.txt'
            record_file = open(record_file_path, "a")
            
            testauc_record_file_path = hyf["output_dir"]+"new_test_ROC_2.txt"
            testauc_record_file=open(testauc_record_file_path,"w")
            
            extra_record_file_path=hyf["output_dir"]+hyf["model"]+'_'+hyf["lossfunc"]+hyf["ps"]+'_Extra_Testing_record.txt'
            extra_record_file=open(extra_record_file_path, "a")
            use_clinical = hyf.get("use_clinical",True)
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
                fake_vali_data,vali_sids=total_file.select_dataset(data_idx=val_idx, aug=False,use_secondimg=False,noSameRM=hyf["noSameRM"], usethird=False,comman_total_file=total_comman_total_file)
                print("val_idx: ",val_idx)
                print("train_idx: ",train_idx)
                print("fake vali_sids: ",vali_sids)
                print("Got train set")
                fake_train_data,train_sids=total_file.select_dataset(data_idx=train_idx, aug=False,aug_form="",use_secondimg=False,noSameRM=False, usethird=hyf["usethird"],comman_total_file=total_comman_total_file)
                print("fake train_sids: ",train_sids)

                train_loader = DataLoader(fake_train_data, batch_size=hyf["batch_size"], num_workers = hyf["num_workers"]
                                                        , shuffle = True
                                                        , pin_memory = True
                                                        , drop_last = False)

                valid_loader = DataLoader(fake_vali_data, batch_size=hyf["batch_size"], num_workers = hyf["num_workers"]
                                                        , shuffle = True
                                                        , pin_memory = True
                                                        , drop_last = False)

    
                torch.cuda.empty_cache()
                #model=Self_Attention()

                #model = Transformer(514,256)
                model = ResNet18(num_classes=num_classes,input_channel=1,use_radiomics=False,feature_align=False,use_clinical=hyf.get('use_clinical',True))
                if hyf['model']=="ResNet34":
                    model = ResNet34(num_classes=num_classes,input_channel=1,use_radiomics=False,feature_align=False,use_clinical=False)
                elif hyf['model']=="ResNet10":
                    model = ResNet10(num_classes=num_classes,input_channel=1,use_radiomics=False,feature_align=False,use_clinical=False)
                
                pps="_k-fold-sub-fold-"+str(fold)+"_"
                model = load_curr_best_checkpoint(model,out_dir = hyf["output_dir"],model_name = hyf["model"],pps = pps,hyf=hyf)
                print("Loda model dir: ",str(output_path)+str(hyf["model"])+'_'+hyf["lossfunc"]+hyf["ps"]+pps+'_best_model.pth.tar')
                #model = medcam.inject(model, output_dir=hyf["output_dir"]+"/attention_maps", save_maps=True)
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

                
        
                
                train_loss, train_losses, validate_train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3],train_mets= validate(valid_loader = train_loader, model = model, criterion = criterion, labels = labels, multi_M= multi_M)
                train_a,train_b,train_c = train_met.a, train_met.b, train_met.c
                train_pre, train_rec, train_F1, train_spe = train_met.pre,train_met.sen,train_met.F1, train_met.spe
                torch.cuda.empty_cache()
                valid_loss, valid_losses, valid_acc, valid_accs, valid_met,[valid_loss_0,valid_loss_1,valid_loss_2,valid_loss_3],valid_mets= validate(valid_loader = valid_loader, model = model, criterion = criterion, labels = labels, multi_M= multi_M)
                valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c
                valid_pre, valid_rec, valid_F1, valid_spe = valid_met.pre,valid_met.sen, valid_met.F1, valid_met.spe
                valid_met_0,valid_met_1,valid_met_2 = valid_mets[0],valid_mets[1],valid_mets[2]
                torch.cuda.empty_cache()
                test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3],test_mets= validate(valid_loader = test_loader, model = model, criterion = criterion, labels = labels, multi_M= multi_M)
                test_a,test_b,test_c = test_met.a, test_met.b, test_met.c
                test_pre,test_rec, test_F1,test_spe = test_met.pre,test_met.sen, test_met.F1, test_met.spe
                test_met_0,test_met_1,test_met_2 = test_mets[0],test_mets[1],test_mets[2]
                torch.cuda.empty_cache()

                valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu()[:,1])
                test_auc = roc_auc_score(test_a.cpu(),test_b.cpu()[:,1])
                train_auc = roc_auc_score(train_a.cpu(),train_b.cpu()[:,1])
                    

                #extra_test_loss, extra_test_losses, extra_test_acc, extra_test_accs, extra_test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3] ,extra_test_mets = validate(valid_loader = extra_test_loader, model = model, criterion = criterion, labels = labels, multi_M= multi_M)
                #extra_test_a,extra_test_b,extra_test_c = extra_test_met.a, extra_test_met.b, extra_test_met.c 
                
            
                #total_test_withextra_test_loss, total_test_withextra_test_losses, total_test_withextra_test_acc, test_accs, total_test_withextra_test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3] ,total_test_withextra_test_mets = validate(valid_loader = total_test_withextra_loader, model = model, criterion = criterion, labels = labels, multi_M= multi_M)
                #total_test_withextra_test_a,total_test_withextra_test_b,total_test_withextra_test_c = total_test_withextra_test_met.a, total_test_withextra_test_met.b, total_test_withextra_test_met.c

                print("test_a: ",test_a)
                print("test_b: ",test_b)
                print("test_c: ",test_c)
                fpr1, tpr1, thersholds = roc_curve(test_a.cpu(), test_b.cpu()[:,1])#用1的预测
                fpr2, tpr2, thersholds = roc_curve(valid_a.cpu(), valid_b.cpu()[:,1])#用1的预测
                fpr3, tpr3, thersholds = roc_curve(train_a.cpu(), train_b.cpu()[:,1])#用1的预测
                
                roc_auc1 = auc(fpr1, tpr1)
                roc_auc2 = auc(fpr2, tpr2)
                roc_auc3 = auc(fpr3, tpr3)
                print("test_roc_auc: ",roc_auc1)
                
                fold_test_tpr.append(tpr1)
                fold_test_fpr.append(fpr1)


                
                plt.figure(figsize=(6,6))
                plt.plot(fpr1, tpr1, 'k--', label='Test ROC (area = {0:.4f})'.format(roc_auc1), lw=2)
                
                tprs.append(np.interp(mean_fpr,fpr1,tpr1))
                tprs[-1][0]=0.0
                
                plt.plot(fpr2, tpr2, 'b--', label='Valid ROC (area = {0:.4f})'.format(roc_auc2), lw=2)
                plt.plot(fpr3, tpr3, 'g--', label='Train ROC (area = {0:.4f})'.format(roc_auc3), lw=2)

                plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限,以免和边缘重合,更好的观察图像的整体
                plt.ylim([-0.05, 1.05])
                
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')  # 可以使用中文,但需要导入一些库即字体
                plt.title('Test ROC Curve')
                plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                plt.legend(loc="lower right")
                #plt.show()
                test_roc_save_path_single = hyf["pic_output_dir"]+"/"+str(fold)+"_test_roc.png"
                plt.savefig(test_roc_save_path_single)
                plt.close()
        
                
                #extra_test_auc = roc_auc_score(extra_test_a.cpu(),extra_test_b.cpu()[:,1])
                #total_test_withextra_auc=roc_auc_score(total_test_withextra_test_a.cpu(),total_test_withextra_test_b.cpu()[:,1])
                    
                recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_epoch",record_file = None,ps_thing=[best_fold,fold,fold_best_auc,best_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets, labels = labels)
                recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_epoch",record_file = record_file,ps_thing=[best_fold,fold,fold_best_auc,best_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets, labels = labels)

                #recording(test_met,extra_test_met,total_test_withextra_test_met,test_loss,extra_test_loss,total_test_withextra_test_loss,test_auc,extra_test_auc,total_test_withextra_auc,record_form="best_epoch",record_file = None,ps_thing=[best_fold,fold,fold_best_auc,best_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets, labels = labels)
                #recording(test_met,extra_test_met,total_test_withextra_test_met,test_loss,extra_test_loss,total_test_withextra_test_loss,test_auc,extra_test_auc,total_test_withextra_auc,record_form="best_epoch",record_file = extra_record_file,ps_thing=[best_fold,fold,fold_best_auc,best_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets, labels = labels)
                
                fig_dir = os.path.join(hyf["pic_output_dir"],"Testing")
                if not os.path.exists(fig_dir):
                    os.mkdir(fig_dir)
                draw_heatmap(train_met.CM,valid_met.CM, test_met.CM,title="fold_"+str(fold)+"_",save_path=fig_dir+'/'+str(fold))
                
                fold_aucs.append(valid_auc)
                fold_record_valid_metrics.append(valid_met)
                fold_record_matched_test_metrics.append(test_met)
                test_fold_aucs.append(test_auc)
                fold_record_matched_train_metrics.append(train_met)
                train_fold_aucs.append(train_auc)
                fold_test_CM.append(test_met.CM)
                


            os.system('echo " === TRAIN mae mtc:{:.5f}" >> {}'.format(train_loss, output_path))

            # to get averaged metrics of k folds
            record_avg(fold_record_valid_metrics,fold_record_matched_test_metrics,record_file,fold_aucs,test_fold_aucs,fold_record_matched_train_metrics,train_fold_aucs)
            #record_avg(fold_record_extra_test_metrics,fold_record_matched_test_metrics,record_file,fold_record_extra_test_aucs,test_fold_aucs,fold_record_total_test_withextra_test_metrics,total_test_withextra_test_fold_aucs)
            
            


            print("fold_test_tpr: ",fold_test_tpr)
            print("fold_test_fpr: ",fold_test_fpr)
            
            testauc_record_file.write("\nfold_test_fpr:\n")
            testauc_record_file.write(str(fold_test_fpr))
            testauc_record_file.write("\nfold_test_tpr:\n")
            testauc_record_file.write(str(fold_test_tpr))
            testauc_record_file.write("\ntest_fold_aucs:\n")
            testauc_record_file.write(str(test_fold_aucs))
            

            #plt.plot(fpr2, tpr2, 'b--', label='Valid ROC (area = {0:.4f})'.format(roc_auc2), lw=2)
            #plt.plot(fpr3, tpr3, 'g--', label='Train ROC (area = {0:.4f})'.format(roc_auc3), lw=2)


        
            avg_testCM_save_path=hyf["pic_output_dir"]+"/"+str(fold)+"_test_CM.png"
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

"""
        plt.figure(figsize=(6,6)) 
        print("all_fpr: ",all_fpr)
        print("all_tpr: ",all_tpr)
        print("all_testauc: ",all_testauc)
        
        for i, path in enumerate(output_dir_box):
            
            fold_test_fpr=all_fpr[i]
            fold_test_tpr=all_tpr[i]
            test_fold_aucs=all_testauc[i]
            plt.plot(fold_test_fpr, fold_test_tpr, color_box[i]+"-", label=label_box[i], lw=2)

        plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限,以免和边缘重合,更好的观察图像的整体
        plt.ylim([-0.05, 1.05])
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')  # 可以使用中文,但需要导入一些库即字体
        plt.title(title)
        plt.plot([0, 1], [0, 1],'r--')
        plt.legend(loc="lower right")
        #plt.show()
        plt.savefig(test_roc_save_path+".png")
        plt.close()
"""
import numpy as np
import re
import ast
from collections import OrderedDict
def read_txt(path):
    # 读取文本文件内容
    #path="/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_433_CE/model_result/test_ROC.txt"

    fold_test_tpr=""
    fold_test_fpr=""
    test_fold_aucs=""

    with open(path, 'r') as file:
        lines = file.read()
    strs=""

    mark=0


    # 使用正则表达式替换特殊字符后面的内容
    # 假设我们想在 !, #, $, %, &, * 后面添加换行符
    special_chars = r"\b(fold_test_tpr:|fold_test_fpr:|test_fold_aucs:)\b"

    #print("lines: ",lines)
    lines=lines.replace("fold_test_tpr:"," \nfold_test_tpr: \n")
    lines=lines.replace("fold_test_fpr:"," \nfold_test_fpr: \n")
    lines=lines.replace("test_fold_aucs:"," \ntest_fold_aucs: \n")
    lines  = re.sub(special_chars, r'\1\n', lines)

    
    #print("line2: ",lines)
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
    #print("fold_test_tpr:",fold_test_tpr)
    #print("fold_test_fpr:",fold_test_fpr)
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
    """
    print("new_tpr_box: ",new_tpr_box)
    print("mean_auc: ",mean_auc)
    new_auc=auc(new_fpr_box, new_tpr_box)
    
    print("new_auc: ",new_auc)
    print("new_tpr_box: ",new_tpr_box)
    #fold_test_fpr = np.mean(fold_test_fpr,0)
    #fold_test_tpr = np.mean(fold_test_tpr,0)
    #test_fold_aucs = np.mean(test_fold_aucs,0)
    print("\nfold_test_fpr,fold_test_tpr,test_fold_aucs: ",fold_test_fpr,fold_test_tpr,test_fold_aucs)
    #print("\n\n2fold_test_fpr,fold_test_tpr,test_fold_aucs: ",np.mean(fold_test_fpr,axis=1),np.mean(fold_test_tpr,axis=1),np.mean(test_fold_aucs))
    """
    return fold_test_fpr[2],fold_test_tpr[2],test_fold_aucs[2]


def draw_all_roc(output_dir_box,test_roc_save_path,color_box,label_box,title,auc_box,subtitle):
    import matplotlib
    import matplotlib.pyplot as plt 
    from matplotlib import font_manager 

    font_path = "/home/chenxr"
    font_files = font_manager.findSystemFonts(fontpaths=font_path)
    
    for file in font_files:
        font_manager.fontManager.addfont(file)

    plt.figure(figsize=(9,9))
    for i, output_path in enumerate(output_dir_box):
        print("i: ",i)
        print("outpath: ",output_path)
        json_path = os.path.join(output_path, 'hyperparameter.json')
        jsf=open(json_path,'r')
        hyf=json.load(jsf,strict=False)
        testauc_record_file_path = hyf["output_dir"]+"new_test_ROC_2.txt"
        print("path: ",output_path)
        try:
            fold_test_fpr,fold_test_tpr,test_fold_aucs=read_txt(testauc_record_file_path)
        except:
            try:
                testauc_record_file_path = hyf["output_dir"]+"new_test_ROC.txt"
                fold_test_fpr,fold_test_tpr,test_fold_aucs=read_txt(testauc_record_file_path)
            except:
                testauc_record_file_path = hyf["output_dir"]+"test_ROC.txt"
                fold_test_fpr,fold_test_tpr,test_fold_aucs=read_txt(testauc_record_file_path)
            
        print("\nfold_test_fpr,fold_test_tpr,test_fold_aucs: ",test_fold_aucs)
        #画单条ROC曲线
        plt.plot(fold_test_fpr, fold_test_tpr, color=color_box[i],linestyle="-", label=label_box[i]+' (auc = {0:.3f})'.format(auc_box[i]),lw=2)
    #画坐标轴
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限,以免和边缘重合,更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.text(-0.15,1.13, subtitle, fontsize=22,fontproperties = 'Times New Roman')
    plt.yticks(fontproperties = 'Times New Roman', size = 20)
    plt.xticks(fontproperties = 'Times New Roman', size = 20)
    from matplotlib import rcParams

    config = {
        "font.family": 'serif',
        "font.size": 12,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)
    
    plt.xlabel('假阳率',fontsize=25,fontproperties = 'SimSun')
    plt.ylabel('真阳率',fontsize=25,fontproperties = 'SimSun')  # 可以使用中文,但需要导入一些库即字体
    plt.plot([0, 1], [0, 1],'grey',linestyle="--",alpha=0.3)
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 18},loc="lower right")
    #plt.show()
    plt.savefig(test_roc_save_path+".png")
    plt.close()

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

def  draw_heatmap(train_CM,valid_CM, test_CM,title,save_path):
        
        train_save_path = save_path+"_train.png"
        valid_save_path = save_path+"_valid.png"
        test_save_path = save_path+"_test.png"

        plt.figure(figsize=(16,16))
        
        plt.subplot(1,3,1)
        sns.heatmap(train_CM, annot=True, square=True,cmap='Blues')

        plt.ylim(0, 3)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(title+"train")
        
   
        
        plt.subplot(1,3,2)
        sns.heatmap(valid_CM, annot=True,square=True, cmap='Blues')

        plt.ylim(0, 3)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(title+"valid")
    
        
        plt.subplot(1,3,3)
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

"""
all_fpr:  [array([0.        , 0.08333333, 0.16666667, 0.16666667, 0.25      , 0.25      , 0.33333333, 0.33333333, 0.5       , 0.5       ,0.75      , 0.75      , 1.        ]), 
           array([0.  , 0.  , 0.  , 0.25, 0.25, 0.75, 0.75, 1.  ]), 
           array([0.        , 0.08333333, 0.25      , 0.25      , 0.33333333,0.33333333, 0.66666667, 0.66666667, 1.        ]), 
           
           array([0.        , 0.08333333, 0.16666667, 0.16666667, 0.25      ,0.25      , 0.33333333, 0.33333333, 0.83333333, 0.83333333,1.        ]), 
           array([0.        , 0.08333333, 0.08333333, 0.16666667, 0.16666667,0.25      , 0.25      , 0.41666667, 0.41666667, 0.75      ,0.75      , 1.        ]), 
           array([0.        , 0.08333333, 0.08333333, 0.16666667, 0.16666667,0.33333333, 0.33333333, 0.58333333, 0.58333333, 0.66666667,0.66666667, 0.83333333, 0.83333333, 1.        ])]

all_tpr:  [array([0.        , 0.        , 0.        , 0.09090909, 0.09090909, 0.72727273, 0.72727273, 0.81818182, 0.81818182, 0.90909091,0.90909091, 1.        , 1.        ]), 
           array([0.        , 0.09090909, 0.27272727, 0.27272727, 0.81818182, 0.81818182, 1.        , 1.        ]), 
           array([0.        , 0.        , 0.        , 0.81818182, 0.81818182,0.90909091, 0.90909091, 1.        , 1.        ]), 
           
           array([0.        , 0.        , 0.        , 0.72727273, 0.72727273,0.81818182, 0.81818182, 0.90909091, 0.90909091, 1.        ,1.        ]), 
           array([0.        , 0.        , 0.45454545, 0.45454545, 0.63636364,0.63636364, 0.72727273, 0.72727273, 0.90909091, 0.90909091,1.        , 1.        ]), 
           array([0.        , 0.        , 0.45454545, 0.45454545, 0.54545455,0.54545455, 0.63636364, 0.63636364, 0.81818182, 0.81818182,0.90909091, 0.90909091, 1.        , 1.        ])]
all_testauc:  [0.6818181818181818, 0.7272727272727273, 0.7045454545454546, 0.75, 0.7651515151515151, 0.6742424242424242] 
"""
def test_mean():
    fold_test_fpr=[0.      ,   0.08333333 ,0.08333333 ,0.16666667 ,0.16666667 ,0.33333333,
    0.33333333 ,0.58333333 ,0.58333333, 0.66666667 ,0.66666667 ,0.83333333,
    0.83333333, 1.        ]
    fold_test_tpr=[0.   ,      0.      ,   0.45454545 ,0.45454545 ,0.54545455 ,0.54545455,
    0.63636364 ,0.63636364 ,0.81818182 ,0.81818182 ,0.90909091, 0.90909091,
    1.   ,      1.        ]
    test_fold_aucs=[0.6060606060606061, 0.7424242424242424, 0.6515151515151516]
    mean_fpr=np.mean(fold_test_fpr,0)
    mean_tpr=np.mean(fold_test_tpr,0)
    
    mean_auc=auc(fold_test_fpr,fold_test_tpr)#计算平均AUC值
    print("mean_auc: ",mean_auc)

def single():
    colorboxex=[]
    labelboxed=[]
    test_roc_save_path_boxex=[]
    filepath_boxex=[]
    title_boxex=[]
    """
    T1C_output_path = [       
        
         "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_Flip_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_Affine_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_composed_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_mask1_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_mask1_composed_RM_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Use_Clinical/Two/T1C_composed_RM/model_result/",
        ]
    data_path='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1C' 
    testdata_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1C"


    model_layers = [       
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRIz/Two/T1C_resnet34_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRIz/Two/T1C_resnet10_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_ce/model_result/",
        
        ]
    label_box=["resnet34","resnet10","resnet18"]
    data_path='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1C' 
    testdata_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1C"
    test_roc_save_path='/home/chenxr/model_layers_Ablation_2'
    color_box=["b","g","k"]


    

   
    t1_new_path=[
        "/home/chenxr/Pineal_region/after_12_08/Results/old_T1/Two/new_real_batchavg_constrain_composed_ResNet18/model_result/"
    ]
    label_box=["resnet34","resnet10","resnet18"]
    data_path='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1' 
    testdata_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1"
    test_roc_save_path='/home/chenxr/model_layers_Ablation_2'
    color_box=["b","g","k"]
    print("=> Drawing beigin. \n")
    
    filepath_boxex.append(t1_new_path)
    colorboxex.append(color_box)
    labelboxed.append(label_box)
    test_roc_save_path_boxex.append(test_roc_save_path)
    title_boxex.append("ResNet_Ablation")
    """
    T2_output_path=[
        "/home/chenxr/Pineal_region/after_12_08/Results/old_T2/Two/new_pretrained_selfKL_composed_ResNet18/model_result/",
    ]
    data_path='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T2' 
    testdata_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T2"
    
    test_roc_save_path='/home/chenxr/t1C_augmentation_Ablation_2'
    color_box=["k","b","g","m","y","r"]
    label_box=['None',"Flip","Affine","Composed","RM","Composed_RM"]
    print("=> Drawing beigin. \n")
    #main(T1C_output_path,color_box,data_path,testdata_path,test_roc_save_path,label_box,title="T1C_Augmentation_Ablation")
    filepath_boxex.append(T2_output_path)
    colorboxex.append(color_box)
    labelboxed.append(label_box)
    test_roc_save_path_boxex.append(test_roc_save_path)
    title_boxex.append("T1C_Augmentation_Ablation")
    

    main(filepath_boxex,colorboxex,data_path,testdata_path,test_roc_save_path_boxex,labelboxed,title_boxex)
def special_drawing():
    T2_AUG_output_path = [       
         "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_Flip_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_Affine_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_composed_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_mask1_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_mask1_composed_RM_CE/model_result/",
        ]
    auc_box=[0.605,0.719,0.716,0.698,0.613,0.723]
    subtitle="c"
    test_roc_save_path='/home/chenxr/t2_augmentation_Ablation_2'
    color_box=["grey","lightgreen","lightblue","#D6BAF2","green","r","r"]
    label_box=['None',"Flip","Affine","Composed","RM","Composed_RM"]
    draw_all_roc(T2_AUG_output_path,test_roc_save_path,color_box,label_box,title="",auc_box=auc_box,subtitle=subtitle)


    T2_output_path = [       
        
         "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_composed_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_mask1_composed_RM_CE/model_result/",
        
                "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T2_composed_ce/model_result/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/old_T2/Two/new_pretrained_selfKL_composed_ResNet18/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_composed_RM_Matrix_MGDA/model_result/",
        ]  
    auc_box=[0.732,0.742,0.780,0.735,0.770]
    subtitle="c"
    test_roc_save_path='/home/chenxr/t2_test_roc.png'
    color_box=["grey","lightgreen","lightblue","#D6BAF2","r"]
    label_box=["Baseline","Composed","Composed_RM","Composed_RM_Clinical","Composed_RM_Clinical_AllLoss"]
    #draw_all_roc(T2_output_path,test_roc_save_path,color_box,label_box,title="T2_Test_ROC")
    draw_all_roc(T2_output_path,test_roc_save_path,color_box,label_box,title="",auc_box=auc_box,subtitle=subtitle)

    T1_output_path=[
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_composed_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_mask1_composed_RM_CE/model_result/",
        
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_composed_RM_2/model_result/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/Single_batchavg/Two/T1_singlebatchavg_composed_ResNet18/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/SelfKL/Two/T1C_mask1_composed_RM_1090/model_result/"
    ]
    auc_box=[0.712,0.710,0.773,0.710,0.780]
    subtitle="b"
    test_roc_save_path='/home/chenxr/t1_test_roc.png'
    color_box=["grey","lightgreen","lightblue","#D6BAF2","r"]
    label_box=["Baseline","Composed","Composed_RM","Composed_RM_Clinical","Composed_RM_Clinical_AllLoss"]
    draw_all_roc(T1_output_path,test_roc_save_path,color_box,label_box,title="",auc_box=auc_box,subtitle=subtitle)
    
    T1_output_path = [       
        
         "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_Flip_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_Affine_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_composed_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_mask1_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_composed_RM_CE_3/model_result/",
        ]
    auc_box=[0.712,0.722,0.745,0.710,0.750,0.711]
    subtitle="b"
    test_roc_save_path='/home/chenxr/t1_aug_test_roc.png'
    
    color_box=["grey","lightgreen","lightblue","#D6BAF2","green","r"]
    label_box=['None',"Flip","Affine","Composed","RM","Composed_RM"]
    draw_all_roc(T1_output_path,test_roc_save_path,color_box,label_box,title="",auc_box=auc_box,subtitle=subtitle)
    



if __name__ == "__main__":
    
    special_drawing()
def common_drawing():
    #avg_cm()
    #test_mean()

    t1c_sing_batch=[
        "/home/chenxr/Pineal_region/after_12_08/Results/old_T1C/Two/none_composed_ResNet18/model_result/",
        
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_7/model_result/",
                "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_6/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_9/model_result/",

        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_8/model_result/",

        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_10/model_result/",

        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_11/model_result/",

        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_12/model_result/",

        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_13/model_result/",

        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_14/model_result/",

        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_15/model_result/",

        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_16/model_result/",
    
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_17/model_result/",
              "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_2/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_4/model_result/",

        
    ]
    auc_box=[0.803,0.788,0.833,0.826,0.783,0.826,0.828,0.836,0.848,0.841,0.821,0.828,0.818,0.823,0.755]

    subtitle=""
    test_roc_save_path='/home/chenxr/t1c_Sing_Batch_2'
    color_box=["black","grey","#FFB6C1","#D3D3D3","#90EE90","#D6BAF2",
               "#E0FFFF","#F08080","r","lightblue","#E6E6FA","#FAFAD2",
               "#FFFF99","#B0E0E6","#FFC0CB"]
    
    label_box=['CE','Weights_1',"Weights_2","Weights_3","Weights_4","Weights_5","Weights_6",
               'Weights_7',"Weights_8","Weights_9","Weights_10","Weights_11","Weights_12",
               "Weights_13","Weights_14",]
    draw_all_roc(t1c_sing_batch,test_roc_save_path,color_box,label_box,title="",auc_box=auc_box,subtitle=subtitle)
  


    t1_singlebatchavg_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_composed_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weightd_clinical/Two/T1_mask1_composed_RM_singlebatchavg_ce_19/model_result/",
         "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_mask1_composed_RM_Singlebatchavg_37/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Single_batchavg/Two/T1_singlebatchavg_composed_ResNet18/model_result/", 
       
        "/home/chenxr/Pineal_region/after_12_08/Results/weightd_clinical/Two/T1_mask1_composed_RM_singlebatchavg_ce_91/model_result/"
        
    ]
    auc_box=[]
    subtitle="b"
    
    test_roc_save_path='/home/chenxr/t1_sing_test_roc.png'
    color_box=["lightblue","k","r","lightgreen","blue","grey","g","r"]
    label_box=["Baseline","Sing:CE=1:9","Sing:CE=3:7","Sing:CE=1:1","Sing:CE=9:1","y"]


    T1C_output_path = [       
         "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_composed_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_mask1_composed_RM_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Use_Clinical/Two/T1C_composed_RM/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_13/model_result/",
        ]
    auc_box=[0.765,0.768,0.816,0.803,0.848]
    subtitle="a"
    test_roc_save_path='/home/chenxr/t1C_test_roc.png'
    color_box=["grey","lightgreen","lightblue","#D6BAF2","r"]
    label_box=["Baseline","Composed","Composed_RM","Composed_RM_Clinical","Composed_RM_Clinical_AllLoss"]
    draw_all_roc(T1C_output_path,test_roc_save_path,color_box,label_box,title="",auc_box=auc_box,subtitle=subtitle)
    

    T1C_output_path2 = [       
        "/home/chenxr/Pineal_region/after_12_08/Results/Use_Clinical/Two/T1C_composed_RM/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_Singlebatchavg_ce_new_37/model_result/" , 
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_mask1_composed_RM_Batchavg_ce_91/model_result/", 
        "/home/chenxr/Pineal_region/after_12_08/Results/SelfKL/Two/T1C_pretrained_mask1_composed_RM_1090/model_result/",
        
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_13/model_result/",
        ]
    auc_box=[0.803,0.833,0.816,0.801,0.848]
    subtitle="a"
    test_roc_save_path='/home/chenxr/t1c_loss_test_roc'
    color_box=["black","lightgreen","lightblue","#D6BAF2","r"]
    label_box=["Baseline","SampleAvgLoss","CategoryAvgLoss","MatrixLoss","AllLoss"]
    draw_all_roc(T1C_output_path2,test_roc_save_path,color_box,label_box,title="",auc_box=auc_box,subtitle=subtitle)
    
    #draw_all_roc(T1C_output_path2,test_roc_save_path,color_box,label_box,title="T1C_Loss_Test_ROC")

    T1_output_path2 = [       
                "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_composed_RM_2/model_result/",
         "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical_T1/Two/T1_composed_RM_SingleBatchavg_11/model_result",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_Singlebatchavg_ce_new_55/model_result",
        '/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_composed_RM_Matrix_MGDA/model_result/',
        "/home/chenxr/Pineal_region/after_12_08/Results/SelfKL/Two/T1C_mask1_composed_RM_1090/model_result/"
        ]
    auc_box=[0.710,0.745,0.770,0.770,0.780]
    subtitle="b"
    test_roc_save_path='/home/chenxr/t1_loss_test_roc'
    color_box=["black","lightgreen","lightblue","#D6BAF2","r"]
    label_box=["Baseline","SampleAvgLoss","CategoryAvgLoss","MatrixLoss","AllLoss"]
    draw_all_roc(T1_output_path2,test_roc_save_path,color_box,label_box,title="",auc_box=auc_box,subtitle=subtitle)
    
    #draw_all_roc(T1C_output_path2,test_roc_save_path,color_box,label_box,title="T1_Loss_Test_ROC")
    
    T2_output_path = [       
            "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T2_composed_ce/model_result/",
    #"/home/chenxr/Pineal_region/after_12_08/Results/old_T2/Two/new_pretrained_selfKL_composed_ResNet18/model_result/",
         "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_mask1_ce/model_result/",
    
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Matrix_MGDA_4/model_result",
        "/home/chenxr/Pineal_region/after_12_08/Results/SelfKL/Two/T2_mask1_composed_RM_1090/model_result/",
    
    
    "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_composed_RM_Matrix_MGDA/model_result/",
    ]  
    auc_box=[0.735,0.753,0.758,0.770,0.770]
    subtitle="c"
    test_roc_save_path='/home/chenxr/t2_loss_test_roc'
    color_box=["black","lightgreen","lightblue","#D6BAF2","r"]
    label_box=["Baseline","SampleAvgLoss","CategoryAvgLoss","MatrixLoss","AllLoss"]
    draw_all_roc(T2_output_path,test_roc_save_path,color_box,label_box,title="",auc_box=auc_box,subtitle=subtitle)
    
    


    resnetasyer=[
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRIz/Two/T1C_resnet10_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRIz/Two/T1C_resnet34_ce/model_result/", 
    ]


    T1C_AUG_output_path = [       
        
         "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_Flip_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_Affine_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_composed_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_mask1_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_mask1_composed_RM_CE/model_result/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/Use_Clinical/Two/T1C_composed_RM/model_result/",
        ]
    auc_box=[0.765,0.783,0.722,0.768,0.717,0.816]
    subtitle="a"
    test_roc_save_path='/home/chenxr/t1C_augmentation_Ablation_2'
    color_box=["grey","lightgreen","lightblue","#D6BAF2","green","r","r"]
    label_box=['None',"Flip","Affine","Composed","RM","Composed_RM"]
    draw_all_roc(T1C_AUG_output_path,test_roc_save_path,color_box,label_box,title="",auc_box=auc_box,subtitle=subtitle)
    
    print("=> Drawing beigin. \n")


    print("=> Drawing beigin. \n")


    T1_AUG_output_path = [       
        
         "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_Flip_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_Affine_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_composed_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_mask1_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_composed_RM_CE_3/model_result/",
        ]
    auc_box=[]
    subtitle="b"
    test_roc_save_path='/home/chenxr/t1_augmentation_Ablation_2'
    color_box=["grey","lightgreen","lightblue","#D6BAF2","green","r","r"]
    label_box=['None',"Flip","Affine","Composed","RM","Composed_RM"]
    print("=> Drawing beigin. \n")
    draw_all_roc(T1_AUG_output_path,test_roc_save_path,color_box,label_box,title="",auc_box=auc_box,subtitle=subtitle)
    


    T1C_singlebatchavg_pathlist = [       
        "/home/chenxr/Pineal_region/after_12_08/Results/old_T1C/Two/none_composed_ResNet18/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_Singlebatchavg_ce_new_91/model_result/",  
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_Singlebatchavg_ce_new_55/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_Singlebatchavg_ce_new_37/model_result/" , 
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_Singlebatchavg_ce_new_19/model_result/",                             
        ]
    auc_box=[0.803,0.813,0.778,0.833,0.795]
    subtitle="a"
    test_roc_save_path='/home/chenxr/t1c_singlebatchavg_Ablation_2'
    color_box=["grey","g","lightblue","r","#D6BAF2"]
    label_box=['CE',"CE:SampleAvg=9:1","CE:SampleAvg=1:1","CE:SampleAvg=3:7","CE:SampleAvg=1:9"]
    print("=> Drawing beigin. \n")
    draw_all_roc(T1C_singlebatchavg_pathlist,test_roc_save_path,color_box,label_box,title="",auc_box=auc_box,subtitle=subtitle)
    



    T2_SingleBatchavg_output_path = [       
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T2_composed_ce/model_result/",
         #"/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T2_mask1_composed_RM_Singlebatchavg_37/model_result/",
         "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_mask1_ce/model_result/",
         "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T2_singlebatchavg_ce/model_result/",
         "/home/chenxr/Pineal_region/after_12_08/Results/weightd_clinical/Two/T2_mask1_composed_RM_singlebatchavg_ce_64/model_result/",
        ]  
    auc_box=[0.735,0.753,0.689,0.619]
    subtitle="c" 
    test_roc_save_path='/home/chenxr/t2_singlebatchavg_Ablation_2'
    color_box=["grey","r","lightblue","lightgreen"]
    label_box=['CE',"CE:SampleAvg=3:7","CE:SampleAvg=1:1","CE:SampleAvg=6:4"]
    draw_all_roc(T2_SingleBatchavg_output_path,test_roc_save_path,color_box,label_box,title="",auc_box=auc_box,subtitle=subtitle)
    
    print("=> Drawing beigin. \n")

    t1_singlebatchavg_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_composed_ce/model_result/",
            "/home/chenxr/Pineal_region/after_12_08/Results/weightd_clinical/Two/T1_mask1_composed_RM_singlebatchavg_ce_91/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Single_batchavg/Two/T1_singlebatchavg_composed_ResNet18/model_result/", 
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_mask1_composed_RM_Singlebatchavg_37/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weightd_clinical/Two/T1_mask1_composed_RM_singlebatchavg_ce_19/model_result/",
        
    ]
    auc_box=[]
    subtitle="b"
    
    t1_singlebatchavg_new_box=[
         "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_composed_RM_2/model_result/",
         "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical_T1/Two/T1_composed_RM_SingleBatchavg_91/model_result",
         "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical_T1/Two/T1_composed_RM_SingleBatchavg_11/model_result",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical_T1/Two/T1_composed_RM_SingleBatchavg_37/model_result",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical_T1/Two/T1_composed_RM_SingleBatchavg_19/model_result",
        
       
    ]
    auc_box=[0.710,0.720,0.745,0.692,0.689]
    subtitle="b"
    test_roc_save_path='/home/chenxr/t1_singlebatchavg_Ablation_2'
    color_box=["grey","lightblue","r","g","#D6BAF2"]
    label_box=['CE',"CE:SampleAvg=9:1","CE:SampleAvg=1:1","CE:SampleAvg=3:7","CE:SampleAvg=1:9"]
    draw_all_roc(t1_singlebatchavg_new_box,test_roc_save_path,color_box,label_box,title="",auc_box=auc_box,subtitle=subtitle)
    
    #draw_all_roc(t1_singlebatchavg_new_box,test_roc_save_path,color_box,label_box,title="T1_SingBatchAvg_Test_ROC")
    

    t1c_batchavg_clinical_composedRM_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/Use_Clinical/Two/T1C_composed_RM/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_mask1_composed_RM_Batchavg_ce_91/model_result/", 
    ]
    auc_box=[0.803,0.816]
    subtitle="a"
    test_roc_save_path='/home/chenxr/t1c_lebatchavg_Ablation_2'
    color_box=["grey","r","g","lightblue","#D6BAF2"]
    label_box=['CE',"CE:CategoryAvg=9:1"]
    draw_all_roc(t1c_batchavg_clinical_composedRM_box,test_roc_save_path,color_box,label_box,title="",auc_box=auc_box,subtitle=subtitle)
    
    
    t1_batchavg_clinical_composedRM_box= [
                "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_composed_RM_2/model_result/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/old_T1/Two/new_real_batchavg_constrain_composed_ResNet18/model_result/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/weightd_clinical/Two/T1_mask1_composed_RM_batchavg_ce_91/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_Singlebatchavg_ce_new_55/model_result",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_SelfKL_7/model_result",
        
    ]
    auc_box=[0.710,0.770,0.747]
    subtitle="b"
    test_roc_save_path='/home/chenxr/t1_lebatchavg_Ablation_2'
    color_box=["grey","r","g","lightblue","#D6BAF2"]
    label_box=['CE',"CE:CategoryAvg=1:1","CE:CategoryAvg=9:1"]
    draw_all_roc(t1_batchavg_clinical_composedRM_box,test_roc_save_path,color_box,label_box,title="",auc_box=auc_box,subtitle=subtitle)
    
    #draw_all_roc(t1_batchavg_clinical_composedRM_box,test_roc_save_path,color_box,label_box,title="T1_BatchAvg_Test_ROC")
    
    
    T2_Batchavg_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T2_composed_ce/model_result/",
        
        #"/home/chenxr/Pineal_region/after_12_08/Results/old_T2/Two/new_pretrained_batchavg_constrain_composed_ResNet18/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Matrix_MGDA_4/model_result",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical_T2/Two/T2_composed_RM_batchavg_91/model_result/",
    ]
    auc_box=[0.735,0.758,0.712]
    subtitle="c"
    test_roc_save_path='/home/chenxr/t2_batchavg_Ablation_2'
    color_box=["grey","r","lightblue","lightblue","#D6BAF2"]
    label_box=['CE',"CE:CategoryAvg=1:1","CE:CategoryAvg=9:1"]
    draw_all_roc(T2_Batchavg_box,test_roc_save_path,color_box,label_box,title="",auc_box=auc_box,subtitle=subtitle)
    
    #draw_all_roc(T2_Batchavg_box,test_roc_save_path,color_box,label_box,title="T2_Matrix_Test_ROC")
    
    
    t1_matrix_box=[
                "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_composed_RM_2/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/SelfKL/Two/T1_mask1_composed_RM_1090/model_result/",  
        '/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_composed_RM_Matrix_MGDA/model_result/',
    ]


    auc_box=[0.710,0.746,0.770]
    subtitle="b"
    test_roc_save_path='/home/chenxr/t1_matrix_Ablation_2'
    color_box=["grey","g","r","lightblue","#D6BAF2"]
    label_box=['CE',"MatrixLoss","MatrixLoss2"]
    draw_all_roc(t1_matrix_box,test_roc_save_path,color_box,label_box,title="",auc_box=auc_box,subtitle=subtitle)
    
    #draw_all_roc(t1_matrix_box,test_roc_save_path,color_box,label_box,title="T1_Matrix_Test_ROC")
    
    t2_matrix_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T2_composed_ce/model_result/",
        
        "/home/chenxr/Pineal_region/after_12_08/Results/SelfKL/Two/T2_mask1_composed_RM_1090/model_result/",
    ]
    auc_box=[0.735,0.770,]
    subtitle="c"
    test_roc_save_path='/home/chenxr/t2_matrix_Ablation_2'
    color_box=["grey","r","g","lightblue","#D6BAF2"]
    label_box=['CE',"MatrixLoss"]
    draw_all_roc(t2_matrix_box,test_roc_save_path,color_box,label_box,title="",auc_box=auc_box,subtitle=subtitle)
    
    #draw_all_roc(t2_matrix_box,test_roc_save_path,color_box,label_box,title="T2_Matrix_Test_ROC")


    t1c_matrix_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/Use_Clinical/Two/T1C_composed_RM/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Matrix_MGDA/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Matrix_MGDA_2/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Matrix_MGDA_3/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Matrix_MGDA_4/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/SelfKL/Two/T1C_pretrained_mask1_composed_RM_1090/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_SelfKL_7/model_result/",
        
    ]
    auc_box=[0.803,0.795,0.795,0.778,0.763,0.801,0.75]
    subtitle="a"
    test_roc_save_path='/home/chenxr/t1c_matrix_Ablation_2'
    color_box=["black","#D6BAF2","g","grey","lightblue","r","lightgreen"]
    label_box=['CE',"MatrixLoss0","MatrixLoss2","MatrixLoss3","MatrixLoss4","MatrixLoss5","MatrixLoss6"]
    draw_all_roc(t1c_matrix_box,test_roc_save_path,color_box,label_box,title="",auc_box=auc_box,subtitle=subtitle)
    
    
    mix_attention_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_13/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_811_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_622_CE/model_result/",
        
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_555_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_433_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_333_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_122_CE/model_result/",
        

        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_111_CE/model_result/",
    ]

    auc_box=[0.848,0.813,0.896,0.866,0.889,0.806,0.889,0.919]
    subtitle=""
    test_roc_save_path='/home/chenxr/mixattention_Ablation_2'
    color_box=["black","grey","lightpink","#D6BAF2","lightblue","lightgreen","#FFA54F","r"]
    label_box=["T1C","w1:(w2+w3)=8:2","w1:(w2+w3)=6:4","w1:(w2+w3)=5:5","w1:(w2+w3)=4.5:5.5",'w1:(w2+w3)=4:6',"w1:(w2+w3)=3:7","w1:(w2+w3)=1:2"]
    draw_all_roc(mix_attention_box,test_roc_save_path,color_box,label_box,title="",auc_box=auc_box,subtitle=subtitle)
  
    attention_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_13/model_result/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/concatenation/Two/Three_modality/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical_T1/Two/T1_composed_RM_SingleBatchavg_19/model_result",
        "/home/chenxr/Pineal_region/after_12_08/Results/New_concatenation_mlp/Two/three_modality_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/SelfAttention_MLP_Transformer/Two/CE_new/model_result/",
        ""
        "/home/chenxr/Pineal_region/after_12_08/Results/New_attention_3_cnn+attention/Two/three_modality_best_attention/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_111_CE/model_result/",
    ]
    "/home/chenxr/Pineal_region/after_12_08/Results/SelfAttention_concatenation_mlp/Two/three_modality_SelfKL"
    auc_box=[0.848,0.687,0.843,0.816,0.846,0.919]
    subtitle=""
    test_roc_save_path='/home/chenxr/attention_Ablation_2'
    color_box=["black","#D6BAF2","lightblue","lightgreen","sandybrown","r"]
    label_box=['T1C',"Multi_Image_Conca","Feature_Conca","SelfAttention","CrossAttention","MixAttention",]
    draw_all_roc(attention_box,test_roc_save_path,color_box,label_box,title="",auc_box=auc_box,subtitle=subtitle)
  
    #draw_all_roc(t1c_matrix_box,test_roc_save_path,color_box,label_box,title="T1C_Matrix_Test_ROC")


    for i, output_path in enumerate(t1c_sing_batch):
        print("i: ",i)
        print("outpath: ",output_path)
        output_path=output_path+"/model_result"
        json_path = os.path.join(output_path, 'hyperparameter.json')
        jsf=open(json_path,'r')
        hyf=json.load(jsf,strict=False)
        print("CE:matrix:sing:batch",hyf["lambda_CE"],hyf["lambda_SelfKL"],hyf["lambda_Sing"],hyf["lambda_Batch"])

    
    
    
    """
    T1C_output_path = [       
        # "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_ce/model_result/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_composed_CE/model_result/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_mask1_composed_RM_CE/model_result/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/Use_Clinical/Two/T1C_composed_RM/model_result/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_Singlebatchavg_ce_new_37/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_13/model_result/",

        ]
    test_roc_save_path='/home/chenxr/t1c_test_roc.png'
    color_box=["lightblue","k","lightgreen","blue","r","grey","g"]
    label_box=["Baseline","Composed_CE","Composed_RM_CE","Composed_RM_Clinical_CE","Composed_RM_Clinical_AllLoss"]
    """
    #draw_all_roc(T1C_output_path,test_roc_save_path,color_box,label_box,title="T1C")
    
    """
    T1_output_path = [       
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_811_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_622_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_111_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_555_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_433_CE/model_result/",
        
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_333_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_122_CE/model_result/",

        ]
    data_path='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1' 
    testdata_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1"
    test_roc_save_path='/home/chenxr/test_roc.png'
    color_box=["lightblue","k","r","lightgreen","blue","grey","g"]
    label_box=["w1:(w2+w3)=8:2","w1:(w2+w3)=6:4","w1:(w2+w3)=5:5","w1:(w2+w3)=4.5:5.5",'w1:(w2+w3)=4:6',"w1:(w2+w3)=3:7","w1:(w2+w3)=2:8"]

    draw_all_roc(T1_output_path,test_roc_save_path,color_box,label_box,title="Mix_attention")
    """
    #path="/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_433_CE/model_result/test_ROC.txt"
    #read_txt(path)

   
    
    
    

 


def others():
###############################################    

    colorboxex=[]
    labelboxed=[]
    test_roc_save_path_boxex=[]
    filepath_boxex=[]
    title_boxex=[]
    T1_output_path = [       
        
         "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_Flip_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_Affine_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_composed_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_mask1_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_mask1_composed_RM_CE/model_result/",
        ]
    data_path='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1' 
    testdata_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1"
    test_roc_save_path='/home/chenxr/t1_augmentation_Ablation_2'
    color_box=["k","b","g","m","y","r"]
    label_box=['None',"Flip","Affine","Composed","RM","Composed_RM"]

    filepath_boxex.append(T1_output_path)
    colorboxex.append(color_box)
    labelboxed.append(label_box)
    test_roc_save_path_boxex.append(test_roc_save_path)
    title_boxex.append("T1_Augmentation_Ablation")
    print("=> Drawing beigin. \n")
    #main(T1_output_path,color_box,data_path,testdata_path,test_roc_save_path,label_box,title="T1_Augmentation_Ablation")
 
    t1_singlebatchavg_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_composed_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weightd_clinical/Two/T1_mask1_composed_RM_singlebatchavg_ce_19/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Single_batchavg/Two/T1_singlebatchavg_composed_ResNet18/model_result/", 
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_mask1_composed_RM_Singlebatchavg_37/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weightd_clinical/Two/T1_mask1_composed_RM_singlebatchavg_ce_91/model_result/"
        
    ]
    data_path='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1' 
    testdata_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1"
    test_roc_save_path='/home/chenxr/t1_singlebatchavg_Ablation_2'
    color_box=["k","r","b","g","m","y"]
    label_box=['CE',"CE:Sing=1:1","CE:Sing=3:7","CE:Sing=9:1"]
    
    filepath_boxex.append(t1_singlebatchavg_box)
    colorboxex.append(color_box)
    labelboxed.append(label_box)
    test_roc_save_path_boxex.append(test_roc_save_path)
    title_boxex.append("T1_SingleBatchAvg_Ablation")

    print("=> Drawing beigin. \n")
    
    
    t1_batchavg_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_composed_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Single_batchavg/Two/T1_pretrained_batchavg_constrain_composed_ResNet18/model_result/", 
        "/home/chenxr/Pineal_region/after_12_08/Results/weightd_clinical/Two/T1_mask1_composed_RM_batchavg_ce_91/model_result/",  
    ]
    data_path='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1' 
    testdata_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1"
    test_roc_save_path='/home/chenxr/t1_batchavg_Ablation_2'
    color_box=["k","r","g","b","y"]
    label_box=['CE',"CE:Batch=1:1","CE:Batch=9:1"]
    
    filepath_boxex.append(t1_batchavg_box)
    colorboxex.append(color_box)
    labelboxed.append(label_box)
    test_roc_save_path_boxex.append(test_roc_save_path)
    title_boxex.append("T1_BatchAvg_Ablation")

    print("=> Drawing beigin. \n")
    
    main(filepath_boxex,colorboxex,data_path,testdata_path,test_roc_save_path_boxex,labelboxed,title_boxex)

#########################
    colorboxex=[]
    labelboxed=[]
    test_roc_save_path_boxex=[]
    filepath_boxex=[]
    title_boxex=[]

    T2_output_path = [       
        
         "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_Flip_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_Affine_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_composed_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_mask1_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_mask1_composed_RM_CE/model_result/",
        ]   
    label_box=['None',"Flip","Affine","Composed","RM","Composed_RM"]
    data_path='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T2' 
    testdata_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T2"
    test_roc_save_path='/home/chenxr/t2_augmentation_Ablation_2'
    color_box=["k","b","g","m","y","r"]
   
    print("=> Drawing beigin. \n")
    #os.system('echo "train {}"  >>  {}'.format(datetime.datetime.now(),output_path))
    #main(T2_output_path,color_box,data_path,testdata_path,test_roc_save_path,label_box,title="T2_Augmentation_Ablation")
    
    
    filepath_boxex.append(T2_output_path)
    colorboxex.append(color_box)
    labelboxed.append(label_box)
    test_roc_save_path_boxex.append(test_roc_save_path)
    title_boxex.append("T2_Augmentation_Ablation")

    print("=> Drawing beigin. \n")

    T2_output_path = [       
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T2_composed_ce/model_result/",
        
         "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T2_mask1_composed_RM_Singlebatchavg_37/model_result/",
         "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T2_singlebatchavg_ce/model_result/",
         "/home/chenxr/Pineal_region/after_12_08/Results/weightd_clinical/Two/T2_mask1_composed_RM_singlebatchavg_ce_64/model_result/",
        ]   
    data_path='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T2' 
    testdata_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T2"
    test_roc_save_path='/home/chenxr/t2_SingleBatchAvg_Ablation_2'
    color_box=["k","r","g","m","y","b"]


    label_box=['CE',"CE:Sing=3:7","CE:Sing=1:1","CE:Sing=6:4"]
    
    filepath_boxex.append(T2_output_path)
    colorboxex.append(color_box)
    labelboxed.append(label_box)
    test_roc_save_path_boxex.append(test_roc_save_path)
    title_boxex.append("T2_SingleBatchavg_Ablation")
    filepath_boxex.append(T2_output_path)
    colorboxex.append(color_box)
    labelboxed.append(label_box)
    test_roc_save_path_boxex.append(test_roc_save_path)
    title_boxex.append("T2_SingleBatchavg_Ablation")

    print("=> Drawing beigin. \n")
    
    T2_Batchavg_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T2_composed_ce/model_result/",
        
        "/home/chenxr/Pineal_region/after_12_08/Results/old_T2/Two/new_pretrained_batchavg_constrain_composed_ResNet18/model_result/",
    ]

    data_path='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T2' 
    testdata_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T2"
    test_roc_save_path='/home/chenxr/t2_segmentation_Ablation_2'
    color_box=["k","r","g","m","y","b"]

    label_box=['CE',"CE:Sing=1:1"]
    filepath_boxex.append(T2_Batchavg_box)
    colorboxex.append(color_box)
    labelboxed.append(label_box)
    test_roc_save_path_boxex.append(test_roc_save_path)
    title_boxex.append("T2_Batchavg_Ablation")
    filepath_boxex.append(T2_output_path)
    colorboxex.append(color_box)
    labelboxed.append(label_box)
    test_roc_save_path_boxex.append(test_roc_save_path)
    title_boxex.append("T2_Batchavg_Ablation")

    print("=> Drawing beigin. \n")

    main(filepath_boxex,colorboxex,data_path,testdata_path,test_roc_save_path_boxex,labelboxed,title_boxex)
    
###############################################




    colorboxex=[]
    labelboxed=[]
    test_roc_save_path_boxex=[]
    filepath_boxex=[]
    title_boxex=[]
    
    T1C_output_path = [       
        
         "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_Flip_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_Affine_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_composed_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_mask1_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_mask1_composed_RM_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Use_Clinical/Two/T1C_composed_RM/model_result/",
        ]
    data_path='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1C' 
    testdata_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1C"
    test_roc_save_path='/home/chenxr/t1C_augmentation_Ablation_2'
    color_box=["k","b","g","m","y","r"]
    label_box=['None',"Flip","Affine","Composed","RM","Composed_RM"]
    print("=> Drawing beigin. \n")
    #main(T1C_output_path,color_box,data_path,testdata_path,test_roc_save_path,label_box,title="T1C_Augmentation_Ablation")
    
    filepath_boxex.append(T1C_output_path)
    colorboxex.append(color_box)
    labelboxed.append(label_box)
    test_roc_save_path_boxex.append(test_roc_save_path)
    title_boxex.append("T1C_Augmentation_Ablation")

    print("=> Drawing beigin. \n")
    
    #main(filepath_boxex,colorboxex,data_path,testdata_path,test_roc_save_path_boxex,labelboxed,title_boxex)
 ##   

    #main(model_layers,color_box,data_path,testdata_path,test_roc_save_path,label_box,title="ResNet_Ablation")    
##

    T1C_singlebatchavg_pathlist = [       
        "/home/chenxr/Pineal_region/after_12_08/Results/old_T1C/Two/none_composed_ResNet18/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_Singlebatchavg_ce_new_19/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_Singlebatchavg_ce_new_55/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_Singlebatchavg_ce_new_91/model_result/",  
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_Singlebatchavg_ce_new_37/model_result/" ,                              
        ]
    data_path='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1C' 
    testdata_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1C"
    test_roc_save_path='/home/chenxr/t1c_singlebatchavg_Ablation_2'
    color_box=["k","b","g","m","r"]
    label_box=['CE',"CE:Sing=1:9","CE:Sing=1:1","CE:Sing=9:1","CE:Sing=3:7"]
    print("=> Drawing beigin. \n")
    #main(T1C_singlebatchavg_pathlist,color_box,data_path,testdata_path,test_roc_save_path,label_box,title="T1C_Singlebatchavg_Ablation")  

    filepath_boxex.append(T1C_singlebatchavg_pathlist)
    colorboxex.append(color_box)
    labelboxed.append(label_box)
    test_roc_save_path_boxex.append(test_roc_save_path)
    title_boxex.append("T1C_Singlebatchavg_Ablation")
##
    T1C_batchavg_pathlist = [       
        "/home/chenxr/Pineal_region/after_12_08/Results/old_T1C/Two/none_composed_ResNet18/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_mask1_composed_RM_Batchavg_ce_91/model_result/",
        ]
    
    data_path='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1C' 
    testdata_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1C"
    test_roc_save_path='/home/chenxr/t1c_batchavg_Ablation_2'
    color_box=["k","b","g","m","y"]
    label_box=['CE',"CE:Batch=1:9"]
    print("=> Drawing beigin. \n")
    
    filepath_boxex.append(T1C_batchavg_pathlist)
    colorboxex.append(color_box)
    labelboxed.append(label_box)
    test_roc_save_path_boxex.append(test_roc_save_path)
    title_boxex.append("T1C_Batchavg_Ablation")
    
    #main(T1C_batchavg_pathlist,color_box,data_path,testdata_path,test_roc_save_path,label_box,title="T1C_Batchavg_Ablation")  
    main(filepath_boxex,colorboxex,data_path,testdata_path,test_roc_save_path_boxex,labelboxed,title_boxex)


    colorboxex=[]
    labelboxed=[]
    test_roc_save_path_boxex=[]
    filepath_boxex=[]
    title_boxex=[]

    
    #main(filepath_boxex,colorboxex,data_path,testdata_path,test_roc_save_path_boxex,labelboxed,title_boxex)
 ##   
    
    model_layers = [       
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRIz/Two/T1C_resnet34_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRIz/Two/T1C_resnet10_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_ce/model_result/",
        
        ]
    label_box=["resnet34","resnet10","resnet18"]
    data_path='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1C' 
    testdata_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1C"
    test_roc_save_path='/home/chenxr/model_layers_Ablation_2'
    color_box=["b","g","k"]
   
    print("=> Drawing beigin. \n")
    filepath_boxex.append(model_layers)
    colorboxex.append(color_box)
    labelboxed.append(label_box)
    test_roc_save_path_boxex.append(test_roc_save_path)
    title_boxex.append("ResNet_Ablation")

######


def draw_auc():
        
    T1_output_path = [       
        
         "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_Flip_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_Affine_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_composed_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_mask1_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_mask1_composed_RM_CE/model_result/",
        ]

    t1_singlebatchavg_box=[
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_composed_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weightd_clinical/Two/T1_mask1_composed_RM_singlebatchavg_ce_19/model_result/",
         "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_mask1_composed_RM_Singlebatchavg_37/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Single_batchavg/Two/T1_singlebatchavg_composed_ResNet18/model_result/", 
       
        "/home/chenxr/Pineal_region/after_12_08/Results/weightd_clinical/Two/T1_mask1_composed_RM_singlebatchavg_ce_91/model_result/"
        
    ]
    
    T2_output_path = [       
        
         "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_Flip_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_Affine_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_composed_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_mask1_ce/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_mask1_composed_RM_CE/model_result/",
        ]  
    test_roc_save_path='/home/chenxr/t2_sing_test_roc.png'
    color_box=["lightblue","k","r","lightgreen","blue","grey","g","r"]
    label_box=["Baseline","Sing:CE=1:9","Sing:CE=3:7","Sing:CE=1:1","Sing:CE=9:1","y"]

    draw_all_roc(T2_output_path,test_roc_save_path,color_box,label_box,title="T2_SingleBatchAvg")
    
    """
    T1C_output_path = [       
        # "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_ce/model_result/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_composed_CE/model_result/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_mask1_composed_RM_CE/model_result/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/Use_Clinical/Two/T1C_composed_RM/model_result/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_Singlebatchavg_ce_new_37/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_13/model_result/",

        ]
    test_roc_save_path='/home/chenxr/t1c_test_roc.png'
    color_box=["lightblue","k","lightgreen","blue","r","grey","g"]
    label_box=["Baseline","Composed_CE","Composed_RM_CE","Composed_RM_Clinical_CE","Composed_RM_Clinical_AllLoss"]
    """
    #draw_all_roc(T1C_output_path,test_roc_save_path,color_box,label_box,title="T1C")
    
    """
    T1_output_path = [       
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_811_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_622_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_111_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_555_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_433_CE/model_result/",
        
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_333_CE/model_result/",
        "/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_122_CE/model_result/",

        ]
    data_path='/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1' 
    testdata_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1"
    test_roc_save_path='/home/chenxr/test_roc.png'
    color_box=["lightblue","k","r","lightgreen","blue","grey","g"]
    label_box=["w1:(w2+w3)=8:2","w1:(w2+w3)=6:4","w1:(w2+w3)=5:5","w1:(w2+w3)=4.5:5.5",'w1:(w2+w3)=4:6',"w1:(w2+w3)=3:7","w1:(w2+w3)=2:8"]

    draw_all_roc(T1_output_path,test_roc_save_path,color_box,label_box,title="Mix_attention")
    """
    path="/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_add_add_433_CE/model_result/test_ROC.txt"
    read_txt(path)



