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
from model.mix_attention  import MixTransformer

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
                # 这里的target_layer要看模型情况，
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
        print("validating sid: ",sid)
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
    
    try:
        checkpoint = torch.load(best_model_path,map_location = DEVICE)
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



def load_curr_best_checkpoint(model,best_model_path):
    checkpoint = torch.load(best_model_path,map_location = DEVICE)
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
    


def validate(valid_loader, model, criterion,labels,multi_M, t1_best_path, t2_best_path, t1c_best_path,hyf):
    
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
            t1_img_out,t1_feature = t1_model(t1_img,input_affix,[])
            t2_img_out,t2_feature = t2_model(t2_img,input_affix,[])
            t1c_img_out,t1c_feature = t1c_model(t1c_img,input_affix,[])

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
            t1_img_auged_out,t1_auged_feature = t1_model(t1_auged_img,input_affix,[])
            t2_img_auged_out,t2_auged_feature = t2_model(t2_auged_img,input_affix,[])
            t1c_img_auged_out,t1c_auged_feature = t1c_model(t1c_auged_img,input_affix,[])

            input_t1_auged_feature = t1_auged_feature.to(DEVICE)
            input_t2_auged_feature = t2_auged_feature.to(DEVICE)
            input_t1c_auged_feature = t1c_auged_feature.to(DEVICE)

            input_target = target.to(DEVICE)

            auged_out,auged_feat = model(input_t1_auged_feature,input_t2_auged_feature,input_t1c_auged_feature)
            
            if(hyf["lossfunc"] == 'SelfKL' or hyf["lossfunc"] == 'Weighted_SelfKL'):
                spare_selfKL_criterion = SelfKL(num_classes=hyf["num_classes"],lambda_0=hyf["lambda_0"],lambda_1=hyf["lambda_1"],lambda_2=hyf["lambda_2"],lambda_3=hyf["lambda_2"], CE_or_KL=hyf["CE_or_KL"]) 
                loss,loss_0 ,loss_1,loss_2,loss_3 = spare_selfKL_criterion(out, target)
            elif(hyf["lossfunc"]=="SingleBatchAvg"):
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



def normalizeImg(self,img, brain_mask_img):
    img_data=sitk.GetArrayFromImage(img)
    mask_data = sitk.GetArrayFromImage(brain_mask_img)
    brain_mask = mask_data > 0
    brain_mean = img_data[brain_mask].mean()
    brain_std = img_data[brain_mask].std()
    img_data = (img_data - brain_mean) / brain_std

    img= array2image(img_data,img)

    croporpad = tio.CropOrPad(
    (400,400,23),
    mask_name='brain_mask',
    )
    img = croporpad(img)
    return img

def preprocess(datapath):
    preprocessor = Preprocessor(target_spacing=[0.45,0.45,6.5])
    img, brain_mask_img =preprocessor.run(img_path=sub_path)
    print(sid,"_Spacing2: ",img.GetSpacing())
    img=normalizeImg(img,brain_mask_img)
    img = sitk.GetArrayFromImage(img).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img, dtype= np.float32)
    return img

def main(T1path,T2path,T1Cpath,t1model,t2model,t1cmodel,multi_model):
    input_affix=[0,1]
    number=0
    out=[0,0]
    if T1path!="T1 Directory":
        t1image= preprocess(T1path)
        torch.cuda.empty_cache()
        t1model=load_cnn_feature_model(t1model,input_channel=1,feature_align=True)
        t1_model = t1model.to(DEVICE)
        t1_img=t1image.to(DEVICE)
        t1_img_out,t1_feature = t1_model(t1_img,input_affix,[])
        number+=1
        out+=t1_img_out
        
    if T2path!="T2 Directory":
        t2image= preprocess(T2path)
        torch.cuda.empty_cache()
        t2model=load_cnn_feature_model(t2model,input_channel=1,feature_align=True)
        t2_model = t2model.to(DEVICE)
        t2_img=t2image.to(DEVICE)
        t2_img_out,t2_feature = t2_model(t2_img,input_affix,[])
        number+=1
        out+=t2_img_out
        
    if T1Cpath!="T1C Directory":
        t1cimage= preprocess(T1Cpath)
        torch.cuda.empty_cache()
        t1cmodel=load_cnn_feature_model(t1cmodel,input_channel=1,feature_align=True)
        t1c_model = t1cmodel.to(DEVICE)
        t1c_img=t1cimage.to(DEVICE)
        t1c_img_out,t1c_feature = t1c_model(t1c_img,input_affix,[])
        number+=1
        out+=t3_img_out
    out=out/number
    
    if multi_model!="":
        model = MixTransformer(dim,256,w1=0.5,w2=0.5,w3=0.5)
        model = load_curr_best_checkpoint(model,out_dir =multi_model)
        input_t1_feature = t1_feature.to(DEVICE)
        input_t2_feature = t2_feature.to(DEVICE)
        input_t1c_feature = t1c_feature.to(DEVICE)
        out,feat = model(input_t1_feature,input_t2_feature,input_t1c_feature)
        
    return out
        


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
 
 

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-T1path', '--T1path', default=' ', help="The path of pretrained models")
    parser.add_argument('-T2path', '--T2path', default=' ',help="The name of selected model")
    parser.add_argument('-T1Cpath', '--T1Cpath', default=' ', help="The path of pretrained models")
    parser.add_argument('-t1_model', '--t1_model', default=' ',help="The name of selected model")
    parser.add_argument('-t2_model', '--t2_model', default=' ', help="The path of pretrained models")
    parser.add_argument('-t1c_model', '--t1c_model', default=' ',help="The name of selected model")
    parser.add_argument('-multi_model', '--multi_model', default=' ',help="The name of selected model")
    args = parser.parse_args()
    T1path = args.T1path
    T2path = args.T2path
    T1Cpath = args.T1Cpath
    t1_model=args.t1_model
    t2_model=args.t2_model
    t1c_model=args.t1c_model
    multi_model=args.multi_model
    
    print("=> training beigin. \n")
    #os.system('echo "train {}"  >>  {}'.format(datetime.datetime.now(),output_path))
    out=main(T1path,T2path,T1Cpath,t1_model,t2_model,t1c_model,multi_model)
    return out
