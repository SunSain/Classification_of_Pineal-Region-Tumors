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
from utils2.config_3class import opt
<<<<<<< HEAD
from load_data_3class import DIY_Folder
=======
from load_data_23 import DIY_Folder
>>>>>>> 3a4a4f2 (20240625-code)
import seaborn as sns
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


from utils2.earlystopping import EarlyStopping
from utils2.avgmeter import AverageMeter
from utils2.metrics import Metrics
from sklearn.model_selection import train_test_split
from model.resnet_3d import ResNet10,ResNet18,ResNet34,ResNet24,ResNet30

from model.diy_resnet_3d import DIY_ResNet10,DIY_ResNet18
import matplotlib.pyplot as plt
from utils2.weighted_CE import Weighted_CE
from utils2.self_KL import SelfKL
from utils2.FLoss import Focal_Loss
<<<<<<< HEAD
=======
import pandas as pd
>>>>>>> 3a4a4f2 (20240625-code)

"""
from model.vgg11 import VGG11_bn
from model.vgg13 import VGG13_bn
from model.Inception2 import Inception2
from model.vgg16 import VGG16_bn
from model.SEResnet import seresnet18
"""
#from model import tencent_resnet


import torchio as tio
from sklearn.model_selection import cross_validate
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from sklearn.model_selection import StratifiedKFold

<<<<<<< HEAD
from sklearn.metrics import roc_curve, auc,roc_auc_score,confusion_matrix ,precision_score,f1_score,recall_score
=======
from sklearn.metrics import roc_curve, auc,roc_auc_score,confusion_matrix ,precision_score,f1_score,recall_score,precision_recall_curve
>>>>>>> 3a4a4f2 (20240625-code)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from statistics import mean
import time
import math
from model import tencent_resnet
import ast
from utils2.batchaverage import BatchCriterion
<<<<<<< HEAD
=======
from load_3modality_only import DIY_Folder_3M

>>>>>>> 3a4a4f2 (20240625-code)

target_names = ['class 0', 'class 1','class 2']



if torch.cuda.is_available():
    torch.cuda.empty_cache()
<<<<<<< HEAD
    torch.cuda.set_device(0)
=======
    torch.cuda.set_device(3)
>>>>>>> 3a4a4f2 (20240625-code)
    DEVICE=torch.device('cuda')
else:
    DEVICE=torch.device('cpu')
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
print('DEVICE: ',DEVICE)

test_outdir_box=[]
test_model_box=[]
test_loss_box=[]
test_train_box = []
test_ps_box = []

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


def load_curr_best_checkpoint(model,out_dir,model_name,lossfunc,ps,pps):
    best_model_path = out_dir+model_name+'_'+lossfunc+ps+pps+'_best_model.pth.tar'
    print("best_model_path: ",best_model_path)
    
    
    try:
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['state_dict'])
    except:
        checkpoint = torch.load(best_model_path)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint['state_dict'])
    best_epoch = checkpoint['epoch']
    return model,best_epoch

def convert(original): #[[0],[1],[0],[0]]→ [0,1,0,0]
    target=torch.Tensor(len(original))
    for i in range(len(original)):
        target[i]=original[i][0]
    target=target.type(torch.LongTensor).to(DEVICE)
    return target



<<<<<<< HEAD
def normal_validate(valid_loader, model, criterion, lossfunc,affix_box):
=======
def normal_validate(valid_loader, model, criterion, lossfunc,affix_box,multi_M):
>>>>>>> 3a4a4f2 (20240625-code)
        
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
    lambda_0,lambda_1,lambda_2,lambda_3,CE_or_KL=affix_box[0],affix_box[1],affix_box[2],affix_box[3],affix_box[4]
    model.eval() #because if allow model.eval, the bn wouldn't work, and the train_set'result would be different(model.train & model.eval)
<<<<<<< HEAD

    with torch.no_grad():
        
        for i, (img,_,target, saffix,sradiomics) in enumerate(valid_loader):
            target = torch.from_numpy(np.expand_dims(target,axis=1))
            target = convert(target)


=======
    features=[]
    with torch.no_grad():
        
        for i, stuff  in enumerate(valid_loader):

            if multi_M:
                (img,t1_img,t2_img,sid,target, saffix, sradiomics) = stuff
            else:
                (img,sid,target, saffix, sradiomics) = stuff
            target = torch.from_numpy(np.expand_dims(target,axis=1))
            target = convert(target)

            id_feat=torch.unsqueeze(sid, dim=1)
            id_feat=id_feat.numpy()
            print("id_feat: ",id_feat)
            print("target: ",target)
            
>>>>>>> 3a4a4f2 (20240625-code)
            input_img = img.to(DEVICE)
            input_affix = saffix.to(DEVICE)
            input_radiomics = sradiomics.to(DEVICE)
            input_target = target.to(DEVICE)
            #print("input_affix: ",input_affix)
            #input_img = torch.reshape(input_img, [input_img.shape[0],1,input_img.shape[1],input_img.shape[2],input_img.shape[3]])
            
            if(lossfunc=="BatchAvg"):
                out,feat = model(input_img,input_affix,input_radiomics)
                print("out_ :",out)
                #print("feat_:" ,feat)
            else:
<<<<<<< HEAD
                out = model(input_img,input_affix,input_radiomics)
            
            spare_criterion = nn.CrossEntropyLoss()     
            spare_selfKL_criterion = SelfKL(num_classes=3,lambda_0=lambda_0,lambda_1=lambda_1,lambda_2=lambda_2,lambda_3=lambda_2, CE_or_KL= CE_or_KL) 
=======
                out,feat = model(input_img,input_affix,input_radiomics)
                
            id_feat=np.concatenate((id_feat, feat.cpu().numpy()), axis=1)
            id_feat=np.concatenate((id_feat, torch.unsqueeze(target,dim=1).cpu().numpy()), axis=1)

            print("id_feat: ",id_feat)
            print("id_feat.shape: ",id_feat.shape)
            print("target: ", torch.unsqueeze(target,dim=1))
            features.extend(id_feat.tolist())
            print("features.shape: ",np.array(features).shape)
            spare_criterion = nn.CrossEntropyLoss()     
            spare_selfKL_criterion = SelfKL(num_classes=2,lambda_0=lambda_0,lambda_1=lambda_1,lambda_2=lambda_2,lambda_3=lambda_2, CE_or_KL= CE_or_KL) 
>>>>>>> 3a4a4f2 (20240625-code)
            if(lossfunc == 'SelfKL' or lossfunc == 'Weighted_SelfKL'):
                loss,loss_0 ,loss_1,loss_2,loss_3 = criterion(out, target)
            elif(lossfunc=="BatchAvg"):
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
            print("pred:",pred)
            print("out: ",out)
            print("target: ",target)
            maes.update(mae, input_img.size(0))

            # collect every output/pred/target, combine them together for total metrics'calculation
            total_target.extend(target)
            total_out.extend(torch.softmax(out,dim=1).cpu().numpy())
            total_pred.extend(pred.cpu().numpy())
 
            CM+=confusion_matrix(target.cpu(), pred.cpu(),labels=[0,1,2])
        
        a=torch.tensor(total_target)
        b=torch.tensor(total_out)
        c=torch.tensor(total_pred)    
        # total metrics'calcultion
        print("a",a)
        print("b",b)
        print("c",c)
<<<<<<< HEAD
        auc = roc_auc_score(a.cpu(),b.cpu(),multi_class = 'ovo',labels=[0,1,2])
=======
        #auc = roc_auc_score(a.cpu(),b.cpu(),multi_class = 'ovo',labels=[0,1,2])
        auc = roc_auc_score(a.cpu(),b.cpu()[:,1])

>>>>>>> 3a4a4f2 (20240625-code)
        new_F1 = f1_score(a.cpu(),c.cpu(),average="weighted")
        pre = precision_score(a.cpu(),c.cpu(),average="weighted")
        recall = recall_score(a.cpu(),c.cpu(),average="weighted")
        new_CM= confusion_matrix(a.cpu(),c.cpu())
        print("new_F1: ",new_F1)
        print("pre:",pre)
        print("recall: ",recall)
        print("new_CM:", new_CM)
        print("CM: ",CM)
        
        acc,met_0,met_1,met_2 = cal_metrics(CM)
<<<<<<< HEAD
        auc = roc_auc_score(a.cpu(),b.cpu(),multi_class = 'ovo',labels=[0,1,2])
        
        avg_pre = (met_0.pre+met_1.pre+met_2.pre)/3
        avg_sen = (met_0.sen+met_1.sen+met_2.sen)/3
        avg_spe = (met_0.spe+met_1.spe+met_2.spe)/3
        avg_F1 = (met_0.F1+met_1.F1+met_2.F1)/3     
=======
        #auc = roc_auc_score(a.cpu(),b.cpu(),multi_class = 'ovo',labels=[0,1,2])
        auc = roc_auc_score(a.cpu(),b.cpu()[:,1])
        
        avg_pre = (met_0.pre+met_1.pre+met_2.pre)/2
        avg_sen = (met_0.sen+met_1.sen+met_2.sen)/2
        avg_spe = (met_0.spe+met_1.spe+met_2.spe)/2
        avg_F1 = (met_0.F1+met_1.F1+met_2.F1)/2     
>>>>>>> 3a4a4f2 (20240625-code)
        print('Confusion Matirx : ',CM,'[Metrics]-Accuracy(mean): ' , acc,'- Sensitivity : ',avg_sen*100,'- Specificity : ',avg_spe*100,'- Precision: ',avg_pre*100,'- F1 : ',avg_F1*100,'- auc : ',auc*100)

        # Metrics is a DIY_package to store all metrics(acc, auc, F1,pre, recall, spe,CM, outpur, pred,target)
        met= Metrics()

        met.update(a=a,b=b,c=c, acc=acc,sen=recall,pre=pre,F1=new_F1,spe=avg_spe,auc=auc,CM=CM)

        mets=[met_0,met_1,met_2]
        
        
        
<<<<<<< HEAD
        return losses.avg,losses, maes.avg, maes, met,[Loss_0.avg,Loss_1.avg,Loss_2.avg,Loss_3.avg],mets
    
def normal_masked_constrained_validate(valid_loader, model, criterion,lossfunc,affix_box):
=======
        return losses.avg,losses, maes.avg, maes, met,[Loss_0.avg,Loss_1.avg,Loss_2.avg,Loss_3.avg],mets,features
    
def normal_masked_constrained_validate(valid_loader, model, criterion,lossfunc,affix_box,multi_M):
>>>>>>> 3a4a4f2 (20240625-code)
    
    Losses = AverageMeter()
    Loss_0 = AverageMeter()

    Loss_1 = AverageMeter()
    
    Loss_2 = AverageMeter()
    Loss_3 = AverageMeter()
    
    maes = AverageMeter()
    CM=0
    total_target=[]
    total_out=[]
    total_pred=[]

    constrain_lambd=affix_box[0]
    model.eval() #because if allow model.eval, the bn wouldn't work, and the train_set'result would be different(model.train & model.eval)

<<<<<<< HEAD
    with torch.no_grad():
        for i, (img,masked_img,_,target, saffix,sradiomics) in enumerate(valid_loader):
            target = torch.from_numpy(np.expand_dims(target,axis=1))
            target = convert(target)

=======
    features=[]
    with torch.no_grad():
        for i, stuff in enumerate(valid_loader):
            if multi_M:
                (img,masked_img,_,_,_,_,sid,target, saffix,sradiomics) = stuff
            else:
                (img,masked_img,sid,target, saffix,sradiomics) = stuff
            target = torch.from_numpy(np.expand_dims(target,axis=1))
            target = convert(target)
            
            id_feat=torch.unsqueeze(sid, dim=1)
            id_feat=id_feat.numpy()
            print("id_feat: ",id_feat)
            print("target: ",target)
>>>>>>> 3a4a4f2 (20240625-code)

            img = img.to(DEVICE)
            input_affix = saffix.to(DEVICE)
            masked_img = masked_img.to(DEVICE)
            input_radiomics=sradiomics.to(DEVICE)
            #input_img = torch.reshape(input_img, [input_img.shape[0],1,input_img.shape[1],input_img.shape[2],input_img.shape[3]])
            if( lossfunc=="BatchAvg"):
                img_out,feat = model(img,input_affix,input_radiomics)
                masked_img_out,masked_feat = model(masked_img,input_affix,input_radiomics)

            else:
<<<<<<< HEAD
                img_out = model(img,input_affix,input_radiomics)
                masked_img_out = model(masked_img,input_affix,input_radiomics)
=======
                img_out,feat = model(img,input_affix,input_radiomics)
                masked_img_out,masked_feat = model(masked_img,input_affix,input_radiomics)
                
                
                
            id_feat=np.concatenate((id_feat, feat.cpu().numpy()), axis=1)
            id_feat=np.concatenate((id_feat, torch.unsqueeze(target,dim=1).cpu().numpy()), axis=1)

            print("id_feat: ",id_feat)
            print("id_feat.shape: ",id_feat.shape)
            print("target: ", torch.unsqueeze(target,dim=1))
            features.extend(id_feat.tolist())
            print("features.shape: ",np.array(features).shape)
>>>>>>> 3a4a4f2 (20240625-code)

            spare_criterion = nn.CrossEntropyLoss()            
            if( lossfunc == 'SelfKL' or  lossfunc == 'Weighted_SelfKL'):
                    loss_0,_ ,_,_,_ = criterion(img_out, target)
                    loss_1,_ ,_,_,_ = criterion(masked_img_out, target)
                    loss_2,_ ,_,_,_ = criterion(masked_img_out, F.softmax(img_out,dim=1),False)
            
            elif( lossfunc == 'Weighted_CE' or  lossfunc == 'FLoss' ):
                loss_0 = criterion(img_out, target) 
                loss_1 = criterion(masked_img_out, target)
                loss_2 = criterion(masked_img_out, F.softmax(img_out,dim=1),False) 
            elif( lossfunc=="BatchAvg"):
                loss_0 = criterion(feat, target) + spare_criterion(img_out, target)
                loss_1 = criterion(masked_feat, target) + spare_criterion(masked_img_out, target)
                loss_2 = spare_criterion(masked_img_out, F.softmax(img_out,dim=1))
            else:        
                loss_0 = criterion(img_out, target) 
                loss_1 = criterion(masked_img_out, target)
                loss_2 = criterion(masked_img_out, F.softmax(img_out,dim=1)) 

            
            loss =  constrain_lambd * (loss_0 + loss_1) + (1.0- constrain_lambd)*loss_2
                
            Losses.update(loss*img.size(0),img.size(0)) 
            
            Loss_0.update(loss_0*img.size(0),img.size(0))
            Loss_1.update(loss_1*img.size(0),img.size(0))
            Loss_2.update(loss_2*img.size(0),img.size(0))
            Loss_3.update(  constrain_lambd * (loss_0 + loss_1)*img.size(0),img.size(0))



            pred, mae = get_corrects(output=img_out, target=target)
            maes.update(mae, img.size(0))

            # collect every output/pred/target, combine them together for total metrics'calculation
            total_target.extend(target)
            print("vlidate: target: ",target)
            total_out.extend(torch.softmax(img_out,dim=1).cpu().numpy())
            total_pred.extend(pred.cpu().numpy())

            CM+=confusion_matrix(target.cpu(), pred.cpu(),labels=[0,1,2])
            
        a=torch.tensor(total_target)
        b=torch.tensor(total_out)
        c=torch.tensor(total_pred)    
        # total metrics'calcultion
<<<<<<< HEAD
        auc = roc_auc_score(a.cpu(),b.cpu(),multi_class = 'ovo',labels=[0,1,2])
=======
        #auc = roc_auc_score(a.cpu(),b.cpu(),multi_class = 'ovo',labels=[0,1,2])
        auc = roc_auc_score(a.cpu(),b.cpu()[:,1])
>>>>>>> 3a4a4f2 (20240625-code)
        new_F1 = f1_score(a.cpu(),c.cpu(),average="weighted")
        pre = precision_score(a.cpu(),c.cpu(),average="weighted")
        recall = recall_score(a.cpu(),c.cpu(),average="weighted")
        new_CM= confusion_matrix(a.cpu(),c.cpu())
        print("new_F1: ",new_F1)
        print("pre:",pre)
        print("recall: ",recall)
        print("new_CM:", new_CM)
        print("CM: ",CM)

        
        acc,met_0,met_1,met_2 = cal_metrics(CM)
<<<<<<< HEAD
        auc = roc_auc_score(a.cpu(),b.cpu(),multi_class = 'ovo',labels=[0,1,2])
        avg_pre = (met_0.pre+met_1.pre+met_2.pre)/3
        avg_sen = (met_0.sen+met_1.sen+met_2.sen )/3
        avg_spe = (met_0.spe+met_1.spe+met_2.spe)/3
        avg_F1 = (met_0.F1+met_1.F1+met_2.F1)/3         
        met= Metrics() 
        met.update(a=a,b=b,c=c, acc=acc,sen=recall,pre=pre,F1=avg_F1,spe=avg_spe,auc=auc,CM=CM)
       
        mets = [met_0,met_1,met_2]
        
        return Losses.avg,Losses, maes.avg, maes, met,[Loss_0.avg,Loss_1.avg,Loss_2.avg,Loss_3.avg],mets
=======
        #auc = roc_auc_score(a.cpu(),b.cpu(),multi_class = 'ovo',labels=[0,1,2])
        auc = roc_auc_score(a.cpu(),b.cpu()[:,1])
        avg_pre = (met_0.pre+met_1.pre+met_2.pre)/2
        avg_sen = (met_0.sen+met_1.sen+met_2.sen )/2
        avg_spe = (met_0.spe+met_1.spe+met_2.spe)/2
        avg_F1 = (met_0.F1+met_1.F1+met_2.F1)/2         
        met= Metrics() 
        met.update(a=a,b=b,c=c, acc=acc,sen=recall,pre=pre,F1=new_F1,spe=avg_spe,auc=auc,CM=CM)
       
        mets = [met_0,met_1,met_2]
        
        return Losses.avg,Losses, maes.avg, maes, met,[Loss_0.avg,Loss_1.avg,Loss_2.avg,Loss_3.avg],mets,features
>>>>>>> 3a4a4f2 (20240625-code)


def cal_metrics(CM):
    tn=CM[0][0]
    tp=CM[1][1]
    fp=CM[0][1]
    fn=CM[1][0]
    
    tp_0=CM[0][0]
    fp_0 = CM[1][0]+CM[2][0]
    fn_0= CM[0][1]+CM[0][2]
    tn_0= CM[1][1]+CM[1][2]+CM[2][1]+CM[2][2]
    
    tp_1=CM[1][1]
    fp_1= CM[0][1]+CM[2][1]
    fn_1= CM[1][0]+CM[1][2]
    tn_1= CM[0][0]+CM[0][2]+CM[2][0]+CM[2][2]
    
    tp_2=CM[2][2]
    fp_2= CM[0][2]+CM[1][2]
    fn_2= CM[2][0]+CM[2][1]
    tn_2 = CM[0][0]+CM[0][1]+CM[1][0]+CM[1][1]
    
     
    acc=np.sum(np.diag(CM)/np.sum(CM))
    sen_0=tp_0/(tp_0+fn_0)
    sen_1=tp_1/(tp_1+fn_1)
    sen_2=tp_2/(tp_2+fn_2)
    pre_0=tp_0/(tp_0+fp_0)
    pre_1=tp_1/(tp_1+fp_1)
    pre_2=tp_2/(tp_2+fp_2)
    
    F1_0= (2*sen_0*pre_0)/(sen_0+pre_0)
    F1_1= (2*sen_1*pre_1)/(sen_1+pre_1)
    F1_2= (2*sen_2*pre_2)/(sen_2+pre_2)
    spe_0 = tn_0/(tn_0+fp_0)
    spe_1 = tn_1/(tn_1+fp_1)
    spe_2 = tn_2/(tn_2+fp_2)
    met_0= Metrics()
    met_1= Metrics()
    met_2= Metrics()
    met_0.update(a=0,b=0,c=0,acc=acc,sen=sen_0,pre=pre_0,F1=F1_0,spe=spe_0,auc=0.0,CM=CM)
    met_1.update(a=0,b=0,c=0,acc=acc,sen=sen_1,pre=pre_1,F1=F1_1,spe=spe_1,auc=0.0,CM=CM)
    met_2.update(a=0,b=0,c=0,acc=acc,sen=sen_2,pre=pre_2,F1=F1_2,spe=spe_2,auc=0.0,CM=CM)
    
    return acc,met_0,met_1,met_2

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


def  recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="",record_file = None,ps_thing=[],target_names=[],train_mets=None,valid_mets=None,test_mets=None):
    
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
        print(classification_report(train_a.cpu(), train_c.cpu(), target_names=target_names, labels=[0,1,2]))
        print("\n================================== [Fold: %d [Valid] best at epoch %d ] ==========================================\n"%(fold, best_epoch)) 
        print(classification_report(valid_a.cpu(), valid_c.cpu(), target_names=target_names, labels=[0,1,2]))
        print("\n================================== [Fold: %d [Test] best at epoch %d ] ==========================================\n"%(fold, best_epoch)) 
        print(classification_report(test_a.cpu(), test_c.cpu(), target_names=target_names, labels=[0,1,2]))

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
<<<<<<< HEAD
        print("[train_avg_pre]: %f \t[valid_avg_pre]: %f \t[test_avg_pre]: %f \t"%((train_mets[0].pre+train_mets[1].pre+train_mets[2].pre)/3,(valid_mets[0].pre+valid_mets[1].pre+valid_mets[2].pre)/3,(test_mets[0].pre+test_mets[1].pre+test_mets[2].pre)/3))
=======
        print("[train_avg_pre]: %f \t[valid_avg_pre]: %f \t[test_avg_pre]: %f \t"%((train_mets[0].pre+train_mets[1].pre+train_mets[2].pre)/2,(valid_mets[0].pre+valid_mets[1].pre+valid_mets[2].pre)/2,(test_mets[0].pre+test_mets[1].pre+test_mets[2].pre)/2))
>>>>>>> 3a4a4f2 (20240625-code)
 
        print("[train]: <recall_0>: %f \t<recall_1>: %f \t<recall_2>: %f \n"%(train_mets[0].sen,train_mets[1].sen,train_mets[2].sen))
        print("[valid]: <recall_0>: %f \t<recall_1>: %f \t<recall_2>: %f \n"%(valid_mets[0].sen,valid_mets[1].sen,valid_mets[2].sen))
        print("[test] : <recall_0>: %f \t<recall_1>: %f \t<recall_2>: %f \n"%(test_mets[0].sen,test_mets[1].sen,test_mets[2].sen))
<<<<<<< HEAD
        print("[train_avg_recall]: %f \t[valid_avg_recall]: %f \t[test_avg_recall]: %f \t"%((train_mets[0].sen+train_mets[1].sen+train_mets[2].sen)/3,(valid_mets[0].sen+valid_mets[1].sen+valid_mets[2].sen)/3,(test_mets[0].sen+test_mets[1].sen+test_mets[2].sen)/3))
=======
        print("[train_avg_recall]: %f \t[valid_avg_recall]: %f \t[test_avg_recall]: %f \t"%((train_mets[0].sen+train_mets[1].sen+train_mets[2].sen)/2,(valid_mets[0].sen+valid_mets[1].sen+valid_mets[2].sen)/2,(test_mets[0].sen+test_mets[1].sen+test_mets[2].sen)/2))
>>>>>>> 3a4a4f2 (20240625-code)
 
        print("[train]: <F1_0>: %f \t<F1_1>: %f \t<F1_2>: %f \n"%(train_mets[0].F1,train_mets[1].F1,train_mets[2].F1))
        print("[valid]: <F1_0>: %f \t<F1_1>: %f \t<F1_2>: %f \n"%(valid_mets[0].F1,valid_mets[1].F1,valid_mets[2].F1))
        print("[test] : <F1_0>: %f \t<F1_1>: %f \t<F1_2>: %f \n"%(test_mets[0].F1,test_mets[1].F1,test_mets[2].F1))


<<<<<<< HEAD
        print("[train_avg_F1]: %f \t[valid_avg_F1]: %f \t[test_avg_F1]: %f \t"%((train_mets[0].F1+train_mets[1].F1+train_mets[2].F1)/3,(valid_mets[0].F1+valid_mets[1].F1+valid_mets[2].F1)/3,(test_mets[0].F1+test_mets[1].F1+test_mets[2].F1)/3))
=======
        print("[train_avg_F1]: %f \t[valid_avg_F1]: %f \t[test_avg_F1]: %f \t"%((train_mets[0].F1+train_mets[1].F1+train_mets[2].F1)/2,(valid_mets[0].F1+valid_mets[1].F1+valid_mets[2].F1)/2,(test_mets[0].F1+test_mets[1].F1+test_mets[2].F1)/2))
>>>>>>> 3a4a4f2 (20240625-code)

        print("[train]: <spe_0>: %f \t<spe_1>: %f \t<spe_2>: %f \n"%(train_mets[0].spe,train_mets[1].spe,train_mets[2].spe))     
        print("[valid]: <spe_0>: %f \t<spe_1>: %f \t<spe_2>: %f \n"%(valid_mets[0].spe,valid_mets[1].spe,valid_mets[2].spe))
        print("[test] : <spe_0>: %f \t<spe_1>: %f \t<spe_0>: %f \n"%(test_mets[0].spe,test_mets[1].spe,test_mets[2].spe))
<<<<<<< HEAD
        print("[train_avg_spe]: %f \t[valid_avg_spe]: %f \t[test_avg_spe]: %f \t"%((train_mets[0].spe+train_mets[1].spe+train_mets[2].spe)/3,(valid_mets[0].spe+valid_mets[1].spe+valid_mets[2].spe)/3,(test_mets[0].spe+test_mets[1].spe+test_mets[2].spe)/3))
=======
        print("[train_avg_spe]: %f \t[valid_avg_spe]: %f \t[test_avg_spe]: %f \t"%((train_mets[0].spe+train_mets[1].spe+train_mets[2].spe)/2,(valid_mets[0].spe+valid_mets[1].spe+valid_mets[2].spe)/2,(test_mets[0].spe+test_mets[1].spe+test_mets[2].spe)/2))
>>>>>>> 3a4a4f2 (20240625-code)
   
    #===============================================================
    
    else:

        print(head_str+"best_fold %d & now_fold %d  |||| best_vali_auc %f  & now_vali_auc %f ==========================================\n"%(best_fold, fold, fold_best_auc,best_auc),file=record_file)
        print("\n================================== [Fold: %d [Train] best at epoch %d loss: %f ] ==========================================\n"%(fold, best_epoch,train_loss),file=record_file)                                                                                                    
        print(classification_report(train_a.cpu(), train_c.cpu(), target_names=target_names, labels=[0,1,2]),file=record_file)
        print("\n================================== [Fold: %d [Valid] best at epoch %d loss: %f ] ==========================================\n"%(fold, best_epoch, valid_loss),file=record_file) 
        print(classification_report(valid_a.cpu(), valid_c.cpu(), target_names=target_names, labels=[0,1,2]),file=record_file)
        print("\n================================== [Fold: %d [Test] best at epoch %d loss: %f ] ==========================================\n"%(fold, best_epoch, test_loss),file=record_file) 
        print(classification_report(test_a.cpu(), test_c.cpu(), target_names=target_names, labels=[0,1,2]),file=record_file)

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
<<<<<<< HEAD
        print("[train_avg_pre]: %f \t[valid_avg_pre]: %f \t[test_avg_pre]: %f \n"%((train_mets[0].pre+train_mets[1].pre+train_mets[2].pre)/3,(valid_mets[0].pre+valid_mets[1].pre+valid_mets[2].pre)/3,(test_mets[0].pre+test_mets[1].pre+test_mets[2].pre)/3),file=record_file)
=======
        print("[train_avg_pre]: %f \t[valid_avg_pre]: %f \t[test_avg_pre]: %f \n"%((train_mets[0].pre+train_mets[1].pre+train_mets[2].pre)/2,(valid_mets[0].pre+valid_mets[1].pre+valid_mets[2].pre)/2,(test_mets[0].pre+test_mets[1].pre+test_mets[2].pre)/2),file=record_file)
>>>>>>> 3a4a4f2 (20240625-code)
 
        print("[train]: <recall_0>: %f \t<recall_1>: %f \t<recall_2>: %f \n"%(train_mets[0].sen,train_mets[1].sen,train_mets[2].sen),file=record_file)
        print("[valid]: <recall_0>: %f \t<recall_1>: %f \t<recall_2>: %f \n"%(valid_mets[0].sen,valid_mets[1].sen,valid_mets[2].sen),file=record_file)
        print("[test] : <recall_0>: %f \t<recall_1>: %f \t<recall_2>: %f \n"%(test_mets[0].sen,test_mets[1].sen,test_mets[2].sen),file=record_file)
<<<<<<< HEAD
        print("[train_avg_recall]: %f \t[valid_avg_recall]: %f \t[test_avg_recall]: %f \n"%((train_mets[0].sen+train_mets[1].sen+train_mets[2].sen)/3,(valid_mets[0].sen+valid_mets[1].sen+valid_mets[2].sen)/3,(test_mets[0].sen+test_mets[1].sen+test_mets[2].sen)/3),file=record_file)
=======
        print("[train_avg_recall]: %f \t[valid_avg_recall]: %f \t[test_avg_recall]: %f \n"%((train_mets[0].sen+train_mets[1].sen+train_mets[2].sen)/2,(valid_mets[0].sen+valid_mets[1].sen+valid_mets[2].sen)/2,(test_mets[0].sen+test_mets[1].sen+test_mets[2].sen)/2),file=record_file)
>>>>>>> 3a4a4f2 (20240625-code)
 
        print("[train]: <F1_0>: %f \t<F1_1>: %f \t<F1_2>: %f \n"%(train_mets[0].F1,train_mets[1].F1,train_mets[2].F1),file=record_file)
        print("[valid]: <F1_0>: %f \t<F1_1>: %f \t<F1_2>: %f \n"%(valid_mets[0].F1,valid_mets[1].F1,valid_mets[2].F1),file=record_file)
        print("[test] : <F1_0>: %f \t<F1_1>: %f \t<F1_2>: %f \n"%(test_mets[0].F1,test_mets[1].F1,test_mets[2].F1),file=record_file)


<<<<<<< HEAD
        print("[train_avg_F1]: %f \t[valid_avg_F1]: %f \t[test_avg_F1]: %f \n"%((train_mets[0].F1+train_mets[1].F1+train_mets[2].F1)/3,(valid_mets[0].F1+valid_mets[1].F1+valid_mets[2].F1)/3,(test_mets[0].F1+test_mets[1].F1+test_mets[2].F1)/3),file=record_file)

        print("[train]: <spe_0>: %f \t<spe_1>: %f \t<spe_2>: %f \n"%(train_mets[0].spe,train_mets[1].spe,train_mets[2].spe),file=record_file)     
        print("[valid]: <spe_0>: %f \t<spe_1>: %f \t<spe_2>: %f \n"%(valid_mets[0].spe,valid_mets[1].spe,valid_mets[2].spe),file=record_file)
        print("[test] : <spe_0>: %f \t<spe_1>: %f \t<spe_0>: %f \n"%(test_mets[0].spe,test_mets[1].spe,test_mets[2].spe),file=record_file)
        print("[train_avg_spe]: %f \t[valid_avg_spe]: %f \t[test_avg_spe]: %f \n"%((train_mets[0].spe+train_mets[1].spe+train_mets[2].spe)/3,(valid_mets[0].spe+valid_mets[1].spe+valid_mets[2].spe)/3,(test_mets[0].spe+test_mets[1].spe+test_mets[2].spe)/3),file=record_file)
=======
        print("[train_avg_F1]: %f \t[valid_avg_F1]: %f \t[test_avg_F1]: %f \n"%((train_mets[0].F1+train_mets[1].F1+train_mets[2].F1)/2,(valid_mets[0].F1+valid_mets[1].F1+valid_mets[2].F1)/2,(test_mets[0].F1+test_mets[1].F1+test_mets[2].F1)/2),file=record_file)

        print("[train]: <spe_0>: %f \t<spe_1>: %f \t<spe_2>: %f \n"%(train_mets[0].spe,train_mets[1].spe,train_mets[2].spe),file=record_file)     
        
        print("[valid]: <spe_0>: %f \t<spe_1>: %f \t<spe_2>: %f \n"%(valid_mets[0].spe,valid_mets[1].spe,valid_mets[2].spe),file=record_file)
        print("[test] : <spe_0>: %f \t<spe_1>: %f \t<spe_0>: %f \n"%(test_mets[0].spe,test_mets[1].spe,test_mets[2].spe),file=record_file)
        print("[train_avg_spe]: %f \t[valid_avg_spe]: %f \t[test_avg_spe]: %f \n"%((train_mets[0].spe+train_mets[1].spe+train_mets[2].spe)/2,(valid_mets[0].spe+valid_mets[1].spe+valid_mets[2].spe)/2,(test_mets[0].spe+test_mets[1].spe+test_mets[2].spe)/2),file=record_file)
>>>>>>> 3a4a4f2 (20240625-code)
   
def  draw_heatmap(train_CM,valid_CM, test_CM,title,save_path):
        
        train_save_path = save_path+"_train.png"
        valid_save_path = save_path+"_valid.png"
<<<<<<< HEAD
        test_save_path = save_path+"_test.png"
=======
        test_save_path = save_path+"_test0.png"
>>>>>>> 3a4a4f2 (20240625-code)

        plt.figure(figsize=(30,30))
        
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


<<<<<<< HEAD

def main():
    
    # bbx/bbx_pair_orig, constrain part
    data_path=opt.data_path
    test_data_path = opt.testdata_path   
    root_mask_radiomics =  opt.root_bbx_path
    test_root_mask_radiomics = opt.test_root_bbx_path
=======
def main(output_dir):
    
    """
    # bbx/bbx_pair_orig, constrain part
    data_path=opt.t2_path
    test_data_path = opt.t2_test_path   
    root_mask_radiomics =  ""
    test_root_mask_radiomics = ""
>>>>>>> 3a4a4f2 (20240625-code)
    train_form = "None"
    
    #train_form = "none"
    
<<<<<<< HEAD
    num_classes=3
    input_channel=1
    multiclass=True
    
    use_radiomics=True
    usethird= False
    sec_pair_orig=False
    CE_or_KL=True
   
    total_file = DIY_Folder(data_path=data_path,train_form = train_form,root_mask_radiomics = root_mask_radiomics,use_radiomics=use_radiomics,istest=False,sec_pair_orig=sec_pair_orig,multiclass = multiclass)
    test_file = DIY_Folder(data_path=test_data_path,train_form = train_form,root_mask_radiomics = test_root_mask_radiomics,use_radiomics=use_radiomics,istest=True,sec_pair_orig= sec_pair_orig,multiclass = multiclass)
=======
    num_classes=2
    input_channel=1
    multiclass=False
    
    use_radiomics=False
    usethird= False
    sec_pair_orig=False
    CE_or_KL=True
    
   
    total_file = DIY_Folder(data_path=data_path,train_form = train_form,root_mask_radiomics = root_mask_radiomics,use_radiomics=use_radiomics,istest=False,sec_pair_orig=sec_pair_orig,multiclass = multiclass,vice_path = "")
    test_file = DIY_Folder(data_path=test_data_path,train_form = train_form,root_mask_radiomics = test_root_mask_radiomics,use_radiomics=use_radiomics,istest=True,sec_pair_orig= sec_pair_orig,multiclass = multiclass,vice_path = "")
>>>>>>> 3a4a4f2 (20240625-code)

    print("len(total_file): ",len(total_file))
    print("total_file.gety(): ",total_file.gety())
    #root_radiomics="",root_mask_radiomics="",use_radiomics=True, norm_radiomics=True
<<<<<<< HEAD

=======
    """
>>>>>>> 3a4a4f2 (20240625-code)

    bbx=["/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/bbx_composed_sampler_masked_constrained_train_Weighted_CE_45_30_15_ResNet34/",
                 "/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/bbx_composed_sampler_ResNet34/"
                 ]
    output_dirs=[
<<<<<<< HEAD
        "/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/radiomics_composed_sampler_ResNet18/",
=======
        #"/home/chenxr/Pineal_region/after_12_08/Results/old_T1C/Two/pretraine_composed_SelfKL_ResNet18/",
        "/home/chenxr/Pineal_region/after_12_08/Results/old_T1C/Two/pretraine_composed_SelfKL_BatchAvg_ResNet18/",
        "/home/chenxr/Pineal_region/after_12_08/Results/old_T1C/Two/pretrained_constrain_composed_batchavg_ResNet18/",
        "/home/chenxr/Pineal_region/after_12_08/Results/old_T1/Two/none_sampler_composed_ResNet18/",
        "/home/chenxr/Pineal_region/after_12_08/Results/old_T1/Two/new_real_batchavg_constrain_composed_ResNet18/",
        "/home/chenxr/Pineal_region/after_12_08/Results/old_T1/Two/new_pretrained_selfKL_composed_ResNet18/",
        "/home/chenxr/Pineal_region/after_12_08/Results/old_T1/Two/new_real_batchavg_constrain_composed_ResNet18/",
        "/home/chenxr/Pineal_region/after_12_08/Results/old_T1/Two/none_batchavg_constrain_composed_ResNet18/",
        "/home/chenxr/Pineal_region/after_12_08/Results/old_T2/Two/none_composed_ResNet18/",
        "/home/chenxr/Pineal_region/after_12_08/Results/old_T2/Two/new_pretrained_selfKL_composed_ResNet18/",
        "/home/chenxr/Pineal_region/after_12_08/Results/old_T2/Two/new_pretrained_batchavg_constrain_composed_ResNet18/",
        
        
        #"/home/chenxr/Pineal_region/after_12_08/Results/old_T1C/Two/none_composed_ResNet18/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/0629_T2/Two/none_sampler_composed_ResNet18/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/0704_T1C/Two/new_data/pretrained_bbx_seg_sampler_constrain_batchavg_selfKL_composed_ResNet18/",
        "/home/chenxr/Pineal_region/after_12_08/Results/0704_T1/Two/new_data/none_sampler_composed_ResNet18/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/0704_T1C/Two/new_data/new_constrain_batchavg_selfkl_sampler_composed_ResNet18/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/0704_T1C/Two/new_data/none_sampler_Composed_ResNet18/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/0704_T1C/Two/new_data/pretrained_bbx_seg_sampler_constrain_batchavg_selfKL_composed_ResNet18/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/0704_T1/Two/new_data/none_sampler_composed_ResNet18/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/0630_all/Two/new_pretrained_sampler_constrain_batchavg_selfKL_composed_ResNet18/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/0630_all/Two/none_sampler_composed_ResNet18/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/0629_T2/Two/none_sampler_batchavg_constrain_composed_ResNet18/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/0627_T1/Two/new_data/none_sampler_constrained_batchavg_SelfKL_ResNet18/"
        #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/Two/new_data/none_sampler_constrained_batchavg_SelfKL_Composed_ResNet18/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/Two/new_data/none_sampler_constrained_batchavg_SelfKL_Composed_ResNet18/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/Two/none_sampler_constrained_batchavg_SelfKL_ResNet18/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/Two/multiclass/no_MGDA/none_sampler_constrain_BatchAvg_ResNet18/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/Two/none_sampler_constrained_SelfKL_ResNet18/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/Two/multiclass/no_MGDA/none_sampler_constrain_BatchAvg_ResNet18/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/radiomics_composed_sampler_ResNet18/",
>>>>>>> 3a4a4f2 (20240625-code)
        #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/none_composed_sampler_BatchAvg_ResNet18/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/none_sampler_SelfKL_1090_ResNet18/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/MGDA/none_composed_sampler_BatchAvg_selfKL_1090_hasPretrain_ResNet18/",
        
        "/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/MGDA/new_bbx_composed_constrain_sampler_selfKL_ResNet18/",
                 "/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/new_bbx_composed_sampler_ResNet18/",
                 "/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/new_bbx_composed_sampler_selfKL_1090_ResNet18/",
                 "/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/radiomics_composed_sampler_ResNet18/",
                 "/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/MGDA/radiomics_composed_constrain_sampler_ResNet18/",
                 "/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/MGDA/radiomics_composed_constrain_sampler_selfKL_ResNet18/"
        #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/MGDA/none_composed_sampler_Weighted_SelfKL_1090_221_ResNet24/",
                 #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/none_composed_sampler_constrain_Weighted_SelfKL_5444_321_ResNet24/",
        #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/MGDA/none_composed_sampler_Weighted_SelfKL_1090_221_ResNet24/",
                 #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/MGDA/none_composed_sampler_constrained_SelfKL_1090_ResNet24/",
                 #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/MGDA/none_composed_sampler_SelfKL_ResNet24/",
                 #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/none_composed_sampler_contrain_Weighted_SelfKL_1090_521_ResNet24/",
                 #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/none_composed_sampler_constrain_Weighted_SelfKL_5444_321_ResNet24/",
                 #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/none_composed_sampler_constrain_SelfKL_5444_ResNet24/",
                "/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/MGDA/new_bbx_composed_constrain_sampler_selfKL_ResNet18/",
                 "/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/new_bbx_composed_sampler_SelfKL_1090_ResNet24/",
                 "/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/radiomics_composed_sampler_selfKL_1090_ResNet24/",
                 "/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/radiomics_composed_sampler_ResNet24/",
                 "/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/new_bbx_composed_sampler_ResNet24/",
                
        #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/none_composed_sampler_masked_constrained_CE_ResNet34/",
                #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/none_composed_sampler_contrain_Weighted_SelfKL_5444_521_ResNet24/",
                #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/none_composed_sampler_SelfKL_10990_ResNet24/",
                #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/none_composed_sampler_Weighted_SelfKL_1090_421_ResNet24/",
                #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/none_composed_sampler_SelfKL_1090_ResNet24/",
                #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/MGDA/none_composed_sampler_constrain_ResNet24/",
                 #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/bbx_composed_sampler_Weighted_selfKL_45_30_15_5222_ResNet18",
                # "/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/bbx_composed_sampler_Weighted_selfKL_45_30_15_5444_ResNet18/",
                 #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/none_composed_sampler_masked_constrained_Weighted_SelfKL_321_5444_ResNet34/",
                 #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/none_composed_sampler_masked_constrained_Weighted_CE_321_ResNet34/",
                 "/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/none_composed_sampler_ResNet34/",
                 #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/none_composed_sampler_ResNet18/",
                 #"/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/none_composed_sampler_Weighted_CE_45_30_15_ResNet18",
                 "/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/none_composed_sampler_Weighted_SelfKL_45_30_15_5444_ResNet34/",
                 ]

    
<<<<<<< HEAD
    print("output_dirs: ",output_dirs)
    
    for i, output_dir in enumerate(output_dirs):
    
        hyf_path=os.path.join(output_dir,"model_result")
        json_path = os.path.join(hyf_path, 'hyperparameter.json')
        jsf=open(json_path,'r')
        hyf=json.load(jsf,strict=False)

        print("hyf: ",hyf)
        output_dir = hyf["output_dir"]
        pic_output_dir = hyf["pic_output_dir"]
        lossfunc= hyf["lossfunc"]
        ps = hyf["ps"]
        model_name = hyf["model"]
        print("model_name: ",model_name)
        usesecond=hyf["usesecond"]
        constrain_lambd=hyf["constrain_lambd"]

        #use_radiomics=hyf["use_radiomics"]
        #noSameRM= hyf["noSameRM"]
        noSameRM=True
        use_radiomics=True
        

        batch_size = 4
        lambda_0,lambda_1,lambda_2 ,lambda_3= hyf["lambda_0"],hyf["lambda_1"],hyf["lambda_2"],hyf["lambda_3"]
        aug_form = hyf["aug_form"]
        loss_weight=hyf["loss_weight"]
        
        num_workers=hyf["num_workers"]
        
        try:
            FL_gamma=hyf["FL_gamma"]
        except:
            FL_gamma=2
        
        
            
        # record metrics into a txt
        record_file_path= output_dir+model_name+'_'+lossfunc+ps+'_test_record.txt'
        record_file = open(record_file_path, "w")

        #load the training_data  and test_data (training_data will be splited later for cross_validation)
        test_data = test_file.select_dataset(data_idx=[i for i in range(len(test_file))], aug=False,use_secondimg=usesecond,noSameRM=noSameRM, usethird = usethird)
        test_loader= torch.utils.data.DataLoader(test_data
                                                    , batch_size = batch_size
                                                    , num_workers = num_workers
                                                    , pin_memory = True
                                                    , drop_last = False
                                                    )

        
        loss_func_dict = { 'CE' : nn.CrossEntropyLoss().to(DEVICE)
                        , 'Weighted_CE' : Weighted_CE(classes_weight= loss_weight,n_classes=num_classes)
                        , 'SelfKL' :  SelfKL(num_classes=num_classes,lambda_0= lambda_0,lambda_1= lambda_1,lambda_2= lambda_2,lambda_3= lambda_2, CE_or_KL= CE_or_KL)                    
                        , 'Weighted_SelfKL': SelfKL(num_classes=num_classes,lambda_0= lambda_0,lambda_1= lambda_1,lambda_2= lambda_2,lambda_3= lambda_2, CE_or_KL= CE_or_KL,classes_weight= loss_weight)
                        , 'FLoss':Focal_Loss(num_classes=num_classes,gamma= FL_gamma)
                        , 'BatchAvg': BatchCriterion(num_classes=num_classes,DEVICE=DEVICE)
                        }
        feature_align = True if  lossfunc=='BatchAvg'  else False
        
        criterion = loss_func_dict[lossfunc]

        
        noemal_valid_func_dict={"None": normal_validate
                                ,"none": normal_validate
                    ,"masked_constrained_train": normal_masked_constrained_validate
                    }
        validating = noemal_valid_func_dict[train_form]
        affix_box_list={'None':[lambda_0,lambda_1,lambda_2,lambda_3,CE_or_KL]
                        ,'none':[lambda_0,lambda_1,lambda_2,lambda_3,CE_or_KL]
                        ,"masked_constrained_train":[0.5]
                        }
        affix_box=affix_box_list[train_form]
        print(" ==========> All settled. testing is getting started...")


        # split the training_data into K fold with StratifiedKFold(shuffle = True)
        k=5
        splits=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)
        

        # to record metrics: the best_acc of k folds, best_acc of each fold
        best_fold=1
        best_epoch=0
        
        fold_record_valid_metrics,fold_record_matched_test_metrics,fold_aucs,test_fold_aucs,fold_record_matched_train_metrics,train_fold_aucs=[],[],[],[],[],[]
        #================= begin to train, choose 1 of k folds as validation =================================
        total_train_CM,total_valid_CM,total_test_CM=[],[],[]
        print("======================== start test ================================================ \n")

        for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(total_file)),total_file.gety())):


            print("\n============ FOLD %d ============\n"%(fold),file=record_file)
            print('Fold {}'.format(fold))
            


            #valid_data(no augmentation: aug=False) & train_data(need augmentation:aug = True)
            print("Get valid set")
            vali_data=total_file.select_dataset(data_idx=val_idx, aug=False,use_secondimg=usesecond,noSameRM=noSameRM, usethird=usethird)
            print("Got train set")
            train_data=total_file.select_dataset(data_idx=train_idx, aug=True,aug_form=aug_form,use_secondimg=usesecond,noSameRM=noSameRM, usethird=usethird)
            

            if use_radiomics:
                radio_mean, radio_std = train_data.calc_own_radiomics_mean()
                vali_data.inject_other_mean(radio_mean, radio_std)
                test_data.inject_other_mean(radio_mean, radio_std)
                
                print("train_data. radio_mean: ",train_data.radio_mean)
                train_data.input_unit_radiomics_mean(radio_mean, radio_std)
                vali_data.input_unit_radiomics_mean(radio_mean, radio_std)        
                test_data.input_unit_radiomics_mean(radio_mean, radio_std)
                print("train_data. radio_mean: ",train_data.radio_mean)


            train_loader = DataLoader(train_data, batch_size=batch_size, num_workers = num_workers
                                                    , shuffle = True
                                                    , pin_memory = True
                                                    , drop_last = False)

            valid_loader = DataLoader(vali_data, batch_size=batch_size, num_workers = num_workers
                                                    , shuffle = True
                                                    , pin_memory = True
                                                    , drop_last = False)
            
        
            # choose model
           
            if model_name == "tencent_resnet10":
                model = tencent_resnet.resnet10(sample_input_W=400,
                sample_input_H=400,
                sample_input_D=23,
                shortcut_type='B',
                no_cuda=False,
                num_seg_classes=num_classes,
                input_channel=input_channel,
                feature_align=feature_align
                )
            elif model_name == "tencent_resnet18":
                model = tencent_resnet.resnet18(sample_input_W=400,
                sample_input_H=400,
                sample_input_D=23,
                shortcut_type='B',
                no_cuda=False,
                num_seg_classes=num_classes,
                input_channel=input_channel,
                feature_align=feature_align
                )
            elif model_name == "tencent_resnet34":
                model = tencent_resnet.resnet34(sample_input_W=400,
                sample_input_H=400,
                sample_input_D=23,
                shortcut_type='B',
                no_cuda=False,
                num_seg_classes=num_classes,
                input_channel=input_channel,
                feature_align=feature_align
                )
            elif model_name == "ResNet18":
                model = ResNet18(num_classes=num_classes,input_channel=input_channel,use_radiomics=use_radiomics,feature_align=feature_align)
            elif model_name == "ResNet34":
                model = ResNet34(num_classes=num_classes,input_channel=input_channel,use_radiomics=use_radiomics,feature_align=feature_align)
            elif model_name == "ResNet10":
                model = ResNet10(num_classes=num_classes,input_channel=input_channel,use_radiomics=use_radiomics,feature_align=feature_align)
            elif model_name == "ResNet24":
                model = ResNet24(num_classes=num_classes,input_channel=input_channel,use_radiomics=use_radiomics,feature_align=feature_align)
            elif model_name == "ResNet30":
                model = ResNet30(num_classes=num_classes,input_channel=input_channel,use_radiomics=use_radiomics,feature_align=feature_align)
            elif model_name == "DIY_ResNet18":
                model = DIY_ResNet18(num_classes=num_classes,input_channel=input_channel)
            elif model_name == "DIY_ResNet10":
                model = DIY_ResNet10(num_classes=num_classes,input_channel=input_channel)
            else:
                print("[ERROR: ] Wrong model chosen\n")
            
            model = model.to(DEVICE)
            print("model: ",model)

            pps="_k-fold-sub-fold-"+str(fold)+"_"
            model, best_epoch = load_curr_best_checkpoint(model,out_dir = output_dir,model_name = model_name,pps = pps,lossfunc=lossfunc,ps = ps)
            #model.load_state_dict(best_statedict)        
            train_loss, train_losses, train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3],train_mets  =validating(valid_loader = train_loader
                                        , model = model
                                        , criterion = criterion
                                        ,  lossfunc=lossfunc
                                        ,affix_box=affix_box
                                        )
            train_a,train_b,train_c = train_met.a, train_met.b, train_met.c
            train_acc,train_pre, train_rec, train_F1, train_spe =train_met.acc, train_met.pre,train_met.sen,train_met.F1, train_met.spe

            valid_loss, valid_losses, valid_acc, valid_accs, valid_met, [valid_loss_0,valid_loss_1,valid_loss_2,valid_loss_3],valid_mets= validating(valid_loader = valid_loader
                                        , model = model
                                        , criterion = criterion
                                        ,  lossfunc=lossfunc
                                        ,affix_box=affix_box
                                        )
            valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c
            valid_acc,valid_pre, valid_rec, valid_F1, valid_spe = valid_met.acc,valid_met.pre,valid_met.sen, valid_met.F1, valid_met.spe

            test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3],test_mets = validating(valid_loader = test_loader
                                        , model = model
                                        , criterion = criterion
                                        ,  lossfunc=lossfunc
                                        ,affix_box=affix_box
                                        )
            test_a,test_b,test_c = test_met.a, test_met.b, test_met.c
            test_acc,test_pre,test_rec, test_F1,test_spe = test_met.acc,test_met.pre,test_met.sen, test_met.F1, test_met.spe

            train_auc = roc_auc_score(train_a.cpu(),train_b.cpu() ,multi_class = 'ovo',labels=[0,1,2])
            valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu(),multi_class = 'ovo',labels=[0,1,2])
            test_auc = roc_auc_score(test_a.cpu(),test_b.cpu(),multi_class = 'ovo',labels=[0,1,2])
            
            train_fold_aucs.append(train_auc)
            fold_aucs.append(valid_auc)
            test_fold_aucs.append(test_auc)
            sofar_valid_metrics = valid_met
            fold_record_valid_metrics.append(sofar_valid_metrics)
            fold_record_matched_test_metrics.append(test_met)
            
            fold_record_matched_train_metrics.append(train_met)

            recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_epoch",record_file = None,ps_thing=[best_fold,fold,valid_auc,valid_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets)
            recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_epoch",record_file = record_file,ps_thing=[best_fold,fold,valid_auc,valid_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets)
            
            draw_heatmap(train_met.CM,valid_met.CM, test_met.CM,title="fold_"+str(fold)+"_",save_path=pic_output_dir+'_k-fold_'+str(fold))
            if total_train_CM==[]:
                total_train_CM=train_met.CM
                total_valid_CM = valid_met.CM
                total_test_CM=test_met.CM
            else:
                total_train_CM+=train_met.CM
                total_valid_CM+=valid_met.CM
                total_test_CM+=test_met.CM


        os.system('echo " ================================= "')


        os.system('echo " === Test mae mtc:{:.5f}" >> {}'.format(train_loss, output_dir))

        # to get averaged metrics of k folds
        record_avg(fold_record_valid_metrics,fold_record_matched_test_metrics,record_file,fold_aucs,test_fold_aucs,fold_record_matched_train_metrics,train_fold_aucs)

        # to print & record the best result of total  
        pps="_total_best_"
        
        model, best_epoch = load_curr_best_checkpoint(model,out_dir = output_dir,model_name = model_name,pps = pps,lossfunc=lossfunc,ps=ps)  
        train_loss, train_losses, train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3],train_mets =validating(valid_loader = train_loader
=======
    #print("output_dirs: ",output_dirs)
    
    #for i, output_dir in enumerate(output_dirs):
    output_dir="/home/chenxr/Pineal_region/after_12_08/Results/united/old_T1/Two/pretrained_batchavg_constrain_composed_ResNet18/"
    hyf_path=os.path.join(output_dir,"model_result")
    json_path = os.path.join(hyf_path, 'hyperparameter.json')
    jsf=open(json_path,'r')
    hyf=json.load(jsf,strict=False)

    print("hyf: ",hyf)
    output_dir = hyf["output_dir"]
    pic_output_dir = hyf["pic_output_dir"]
    pic_output_dir=pic_output_dir+"test_data_3/"
    feature_csv_output_dir = pic_output_dir
    
    if not os.path.exists(pic_output_dir):
        os.mkdir(pic_output_dir)
    lossfunc= hyf["lossfunc"]
    ps = hyf["ps"]
    model_name = hyf["model"]
    print("model_name: ",model_name)
    usesecond=hyf["usesecond"]
    constrain_lambd=hyf["constrain_lambd"]

    #use_radiomics=hyf["use_radiomics"]
    #noSameRM= hyf["noSameRM"]
    noSameRM=True
    use_radiomics=False
    

    batch_size = 4
    lambda_0,lambda_1,lambda_2 ,lambda_3= hyf["lambda_0"],hyf["lambda_1"],hyf["lambda_2"],hyf["lambda_3"]
    aug_form = hyf["aug_form"]
    loss_weight=hyf["loss_weight"]
    
    num_workers=hyf["num_workers"]
    
    try:
        FL_gamma=hyf["FL_gamma"]
    except:
        FL_gamma=2
        
    total_train_a=[]
    total_train_b=[]
    total_valid_a=[]
    total_valid_b=[]
    total_test_a=[]
    total_test_b=[]
    best_auc = 0
    best_fold = 0
    
    try:
        multi_M = hyf["multi_M"]
    except:
        multi_M = False
    
    num_classes=hyf["num_classes"]
    data_path = hyf["data_path"]
    test_data_path=hyf["testdata_path"]
    train_form = hyf["train_form"]
    root_bbx_path = hyf["root_bbx_path"]
    test_root_bbx_path = hyf["test_root_bbx_path"]
    sec_pair_orig = hyf["sec_pair_orig"]
    multiclass = hyf["multiclass"]
    usethird = hyf["usethird"]
    CE_or_KL = hyf["CE_or_KL"]
    
    input_channel = 2 if usesecond else 1
    input_channel = 3 if multi_M else input_channel
    input_channel = 4 if multi_M and usesecond else input_channel
    
    print("multi_M: ",multi_M)
    
        
    # record metrics into a txt
    record_file_path= output_dir+model_name+'_'+lossfunc+ps+'_test_record_new5.txt'
    record_file = open(record_file_path, "w")

    #load the training_data  and test_data (training_data will be splited later for cross_validation)
    total_comman_total_file=[1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 27, 28, 29, 30, 31, 34, 35, 36, 37, 39, 40, 42, 43, 44, 46, 48, 49, 51, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 95, 96, 97, 100, 102, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122]
    test_comman_total_file=[3, 4, 8, 16, 20, 22, 25, 26, 38, 45, 47, 56, 64, 65, 66, 75, 93, 94, 98, 99, 101, 111, 120]
    #comman_total_file=[1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 27, 28, 29, 30, 31, 34, 35, 36, 37, 39, 40, 42, 43, 44, 46, 48, 49, 51, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 95, 96, 97, 100, 102, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122]
    y_box = [0 if i<61 else 1 for i in total_comman_total_file] 
    
    if multi_M:
        total_file = DIY_Folder_3M(num_classes = num_classes,data_path=data_path,train_form = train_form,root_mask_radiomics = root_bbx_path,use_radiomics= use_radiomics,istest=False,sec_pair_orig= sec_pair_orig,multiclass =  multiclass,vice_path = ""
                                ,multi_M = multi_M,t1_path =  opt.t1_path
                                , t2_path= opt.t2_path, t1c_path =  opt.t1c_path)

    else:
        total_file = DIY_Folder(num_classes =  num_classes,data_path=data_path,train_form =  train_form,root_mask_radiomics =  root_bbx_path,use_radiomics= use_radiomics,istest=False,sec_pair_orig= sec_pair_orig,multiclass =  multiclass,vice_path = "")

    
    if  multi_M:
        test_file = DIY_Folder_3M(num_classes =  num_classes,data_path=test_data_path,train_form =  train_form,root_mask_radiomics =  test_root_bbx_path,use_radiomics= use_radiomics,istest=True,sec_pair_orig=  sec_pair_orig,multiclass =  multiclass
                                ,multi_M = multi_M,t1_path = opt.t1_test_path
                                , t2_path=opt.t2_test_path, t1c_path = opt.t1c_test_path)
    else:
        test_file = DIY_Folder(num_classes = num_classes,data_path=test_data_path,train_form = train_form,root_mask_radiomics = test_root_bbx_path,use_radiomics= use_radiomics,istest=True,sec_pair_orig= sec_pair_orig,multiclass = multiclass)
    test_data,test_side = test_file.select_dataset(data_idx=[i for i in range(len(test_file))], aug=False,use_secondimg=usesecond,noSameRM=noSameRM, usethird = usethird,comman_total_file=test_comman_total_file)
    test_loader= torch.utils.data.DataLoader(test_data
                                            , batch_size = batch_size
                                            , num_workers = num_workers
                                            , pin_memory = True
                                            , drop_last = False
                                            )
    
    loss_func_dict = { 'CE' : nn.CrossEntropyLoss().to(DEVICE)
                    , 'Weighted_CE' : Weighted_CE(classes_weight= loss_weight,n_classes=num_classes)
                    , 'SelfKL' :  SelfKL(num_classes=num_classes,lambda_0= lambda_0,lambda_1= lambda_1,lambda_2= lambda_2,lambda_3= lambda_2, CE_or_KL= CE_or_KL)                    
                    , 'Weighted_SelfKL': SelfKL(num_classes=num_classes,lambda_0= lambda_0,lambda_1= lambda_1,lambda_2= lambda_2,lambda_3= lambda_2, CE_or_KL= CE_or_KL,classes_weight= loss_weight)
                    , 'FLoss':Focal_Loss(num_classes=num_classes,gamma= FL_gamma)
                    , 'BatchAvg': BatchCriterion(num_classes=num_classes,DEVICE=DEVICE)
                    }
    feature_align = True if  lossfunc=='BatchAvg'  else False
    
    criterion = loss_func_dict[lossfunc]

    
    noemal_valid_func_dict={"None": normal_validate
                            ,"none": normal_validate
                ,"masked_constrained_train": normal_masked_constrained_validate
                }
    validating = noemal_valid_func_dict[train_form]
    affix_box_list={'None':[lambda_0,lambda_1,lambda_2,lambda_3,CE_or_KL]
                    ,'none':[lambda_0,lambda_1,lambda_2,lambda_3,CE_or_KL]
                    ,"masked_constrained_train":[0.5]
                    }
    affix_box=affix_box_list[train_form]
    print(" ==========> All settled. testing is getting started...")


    # split the training_data into K fold with StratifiedKFold(shuffle = True)
    k=3
    splits=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)
    

    # to record metrics: the best_acc of k folds, best_acc of each fold

    best_epoch=0
    
    fold_record_valid_metrics,fold_record_matched_test_metrics,fold_aucs,test_fold_aucs,fold_record_matched_train_metrics,train_fold_aucs=[],[],[],[],[],[]
    #================= begin to train, choose 1 of k folds as validation =================================
    total_train_CM,total_valid_CM,total_test_CM=[],[],[]
    print("======================== start test ================================================ \n")

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(total_comman_total_file)),y_box)):


        print("\n============ FOLD %d ============\n"%(fold),file=record_file)
        print('Fold {}'.format(fold))
        


        #valid_data(no augmentation: aug=False) & train_data(need augmentation:aug = True)
        print("Get valid set")
        vali_data, vali_sids=total_file.select_dataset(data_idx=val_idx, aug=False,use_secondimg=usesecond,noSameRM=noSameRM, usethird=usethird,comman_total_file=total_comman_total_file)
        print("Got train set")
        train_data, train_sids=total_file.select_dataset(data_idx=train_idx, aug=False,aug_form=None,use_secondimg=usesecond,noSameRM=noSameRM, usethird=usethird,comman_total_file=total_comman_total_file)
        

        if use_radiomics:
            radio_mean, radio_std = train_data.calc_own_radiomics_mean()
            vali_data.inject_other_mean(radio_mean, radio_std)
            test_data.inject_other_mean(radio_mean, radio_std)
            
            print("train_data. radio_mean: ",train_data.radio_mean)
            train_data.input_unit_radiomics_mean(radio_mean, radio_std)
            vali_data.input_unit_radiomics_mean(radio_mean, radio_std)        
            test_data.input_unit_radiomics_mean(radio_mean, radio_std)
            print("train_data. radio_mean: ",train_data.radio_mean)


        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers = num_workers
                                                , shuffle = True
                                                , pin_memory = True
                                                , drop_last = False)

        valid_loader = DataLoader(vali_data, batch_size=batch_size, num_workers = num_workers
                                                , shuffle = True
                                                , pin_memory = True
                                                , drop_last = False)
        
    
        # choose model
        
        if model_name == "tencent_resnet10":
            model = tencent_resnet.resnet10(sample_input_W=400,
            sample_input_H=400,
            sample_input_D=23,
            shortcut_type='B',
            no_cuda=False,
            num_seg_classes=num_classes,
            input_channel=input_channel,
            feature_align=feature_align
            )
        elif model_name == "tencent_resnet18":
            model = tencent_resnet.resnet18(sample_input_W=400,
            sample_input_H=400,
            sample_input_D=23,
            shortcut_type='B',
            no_cuda=False,
            num_seg_classes=num_classes,
            input_channel=input_channel,
            feature_align=feature_align
            )
        elif model_name == "tencent_resnet34":
            model = tencent_resnet.resnet34(sample_input_W=400,
            sample_input_H=400,
            sample_input_D=23,
            shortcut_type='B',
            no_cuda=False,
            num_seg_classes=num_classes,
            input_channel=input_channel,
            feature_align=feature_align
            )
        elif model_name == "ResNet18":
            model = ResNet18(num_classes=num_classes,input_channel=input_channel,use_radiomics=use_radiomics,feature_align=feature_align)
        elif model_name == "ResNet34":
            model = ResNet34(num_classes=num_classes,input_channel=input_channel,use_radiomics=use_radiomics,feature_align=feature_align)
        elif model_name == "ResNet10":
            model = ResNet10(num_classes=num_classes,input_channel=input_channel,use_radiomics=use_radiomics,feature_align=feature_align)
        elif model_name == "ResNet24":
            model = ResNet24(num_classes=num_classes,input_channel=input_channel,use_radiomics=use_radiomics,feature_align=feature_align)
        elif model_name == "ResNet30":
            model = ResNet30(num_classes=num_classes,input_channel=input_channel,use_radiomics=use_radiomics,feature_align=feature_align)
        elif model_name == "DIY_ResNet18":
            model = DIY_ResNet18(num_classes=num_classes,input_channel=input_channel)
        elif model_name == "DIY_ResNet10":
            model = DIY_ResNet10(num_classes=num_classes,input_channel=input_channel)
        else:
            print("[ERROR: ] Wrong model chosen\n")
        
        model = model.to(DEVICE)
        print("model: ",model)

        pps="_k-fold-sub-fold-"+str(fold)+"_"
        try:
            model, best_epoch = load_curr_best_checkpoint(model,out_dir = output_dir,model_name = model_name,pps = pps,lossfunc=lossfunc,ps = ps)
        except:
            print("no fold: ",fold)
            continue
        #model.load_state_dict(best_statedict)        
        train_loss, train_losses, train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3],train_mets, train_features  =validating(valid_loader = train_loader
>>>>>>> 3a4a4f2 (20240625-code)
                                    , model = model
                                    , criterion = criterion
                                    ,  lossfunc=lossfunc
                                    ,affix_box=affix_box
<<<<<<< HEAD
                                    )
        train_acc,train_pre, train_rec, train_F1, train_spe =train_met.acc, train_met.pre,train_met.sen,train_met.F1, train_met.spe

        valid_loss, valid_losses, valid_acc, valid_accs, valid_met,[valid_loss_0,valid_loss_1,valid_loss_2,valid_loss_3],valid_mets= validating(valid_loader = valid_loader
=======
                                    ,multi_M = multi_M
                                    )
        train_a,train_b,train_c = train_met.a, train_met.b, train_met.c
        train_acc,train_pre, train_rec, train_F1, train_spe =train_met.acc, train_met.pre,train_met.sen,train_met.F1, train_met.spe

        valid_loss, valid_losses, valid_acc, valid_accs, valid_met, [valid_loss_0,valid_loss_1,valid_loss_2,valid_loss_3],valid_mets, valid_features= validating(valid_loader = valid_loader
>>>>>>> 3a4a4f2 (20240625-code)
                                    , model = model
                                    , criterion = criterion
                                    ,  lossfunc=lossfunc
                                    ,affix_box=affix_box
<<<<<<< HEAD
=======
                                    ,multi_M = multi_M
>>>>>>> 3a4a4f2 (20240625-code)
                                    )
        valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c
        valid_acc,valid_pre, valid_rec, valid_F1, valid_spe = valid_met.acc,valid_met.pre,valid_met.sen, valid_met.F1, valid_met.spe

<<<<<<< HEAD
        test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3],test_mets = validating(valid_loader = test_loader
=======
        test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3],test_mets,test_features = validating(valid_loader = test_loader
>>>>>>> 3a4a4f2 (20240625-code)
                                    , model = model
                                    , criterion = criterion
                                    ,  lossfunc=lossfunc
                                    ,affix_box=affix_box
<<<<<<< HEAD
                                   )
=======
                                    ,multi_M = multi_M
                                    )
>>>>>>> 3a4a4f2 (20240625-code)
        test_a,test_b,test_c = test_met.a, test_met.b, test_met.c
        test_acc,test_pre,test_rec, test_F1,test_spe = test_met.acc,test_met.pre,test_met.sen, test_met.F1, test_met.spe


<<<<<<< HEAD
        train_auc = roc_auc_score(train_a.cpu(),train_b.cpu() ,multi_class = 'ovo',labels=[0,1,2])
        valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu(),multi_class = 'ovo',labels=[0,1,2])
        test_auc = roc_auc_score(test_a.cpu(),test_b.cpu(),multi_class = 'ovo',labels=[0,1,2])


        recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_fold",record_file = None,ps_thing=[best_fold,best_fold,valid_auc,valid_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets)
        recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_fold",record_file = record_file,ps_thing=[best_fold,best_fold,valid_auc,valid_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets)

        draw_heatmap(train_met.CM,valid_met.CM, test_met.CM,title="fold_"+str(fold)+"_",save_path=pic_output_dir+'_total_'+str(fold))

        total_train_CM=total_train_CM/4.0
        total_test_CM=total_test_CM/5.0
        draw_heatmap(total_train_CM,total_valid_CM, total_test_CM,title="total_",save_path=pic_output_dir+'_truly')

        torch.cuda.empty_cache()
        record_file.close()
        jsf.close()


if __name__ == "__main__":

    main()
=======
        #train_auc = roc_auc_score(train_a.cpu(),train_b.cpu() ,multi_class = 'ovo',labels=[0,1,2])
        #valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu(),multi_class = 'ovo',labels=[0,1,2])
        #test_auc = roc_auc_score(test_a.cpu(),test_b.cpu(),multi_class = 'ovo',labels=[0,1,2])
        train_auc = roc_auc_score(train_a.cpu(),train_b.cpu()[:,1])
        valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu()[:,1])
        test_auc = roc_auc_score(test_a.cpu(),test_b.cpu()[:,1])
        if best_auc<valid_auc:
            best_auc = valid_auc
            best_fold = fold
        
        #train_features = train_features.numpy()
        train_features=pd.DataFrame(data=train_features)
        train_features.to_csv(feature_csv_output_dir+"_"+str(fold)+"_train.csv",mode='a+',index=None,header=None)
        print("\n[write into csv]\n")

        valid_features=pd.DataFrame(data=valid_features)
        valid_features.to_csv(feature_csv_output_dir+"_"+str(fold)+"_valid.csv",mode='a+',index=None,header=None)
        
#            test_features = test_features.data.numpy()
#            test_features = test_features.tolist()
        test_features=pd.DataFrame(data=test_features)
        test_features.to_csv(feature_csv_output_dir+"_"+str(fold)+"_test.csv",mode='a+',index=None,header=None)
        
        
        train_fold_aucs.append(train_auc)
        fold_aucs.append(valid_auc)
        test_fold_aucs.append(test_auc)
        sofar_valid_metrics = valid_met
        fold_record_valid_metrics.append(sofar_valid_metrics)
        fold_record_matched_test_metrics.append(test_met)
        
        fold_record_matched_train_metrics.append(train_met)

        recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_epoch",record_file = None,ps_thing=[best_fold,fold,valid_auc,valid_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets)
        recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_epoch",record_file = record_file,ps_thing=[best_fold,fold,valid_auc,valid_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets)
        
        
        draw_heatmap(train_met.CM,valid_met.CM, test_met.CM,title="fold_"+str(fold)+"_",save_path=pic_output_dir+'_k-fold_'+str(fold))
        
        print("train_a: ",train_a)
        print("train_b: ",train_b)
        print("valid_a: ",valid_a)
        print("valid_b: ",valid_b)
        plt.subplot(2,3,1)
        precision,recall,thresholds = precision_recall_curve(train_a.cpu(), train_b.cpu()[:,1])
        plt.plot(precision,recall)

        plt.subplot(2,3,2)
        precision,recall,thresholds = precision_recall_curve(valid_a.cpu(), valid_b.cpu()[:,1])
        plt.plot(precision,recall)

        plt.subplot(2,3,3)
        precision,recall,thresholds = precision_recall_curve(test_a.cpu(), test_b.cpu()[:,1])
        plt.plot(precision,recall)
        plt.savefig(pic_output_dir+'_k-fold-'+str(fold)+'_proc.png')

        #plt.show()
        plt.close()


        plt.subplot(2,3,1)
        fpr,tpr,thresholds = roc_curve(train_a.cpu(), train_b.cpu()[:,1])
        plt.plot(fpr,tpr)

        plt.subplot(2,3,2)
        fpr,tpr,thresholds = roc_curve(valid_a.cpu(), valid_b.cpu()[:,1])
        plt.plot(fpr,tpr)

        plt.subplot(2,3,3)
        fpr,tpr,thresholds = roc_curve(test_a.cpu(), test_b.cpu()[:,1])
        plt.plot(fpr,tpr)
        plt.savefig(pic_output_dir+'_k-fold-'+str(fold)+'_roc.png')

        #plt.show()
        plt.close()
        
        
        total_train_a.extend(train_a.cpu())
        total_train_b.extend(train_b.cpu())
        total_valid_a.extend(valid_a.cpu())
        total_valid_b.extend(valid_b.cpu())
        total_test_a.extend(test_a.cpu())
        total_test_b.extend(test_b.cpu())
        
        if total_train_CM==[]:
            total_train_CM=train_met.CM
            total_valid_CM = valid_met.CM
            total_test_CM=test_met.CM
        else:
            total_train_CM+=train_met.CM
            total_valid_CM+=valid_met.CM
            total_test_CM+=test_met.CM


    os.system('echo " ================================= "')


    os.system('echo " === Test mae mtc:{:.5f}" >> {}'.format(train_loss, output_dir))

    # to get averaged metrics of k folds
    record_avg(fold_record_valid_metrics,fold_record_matched_test_metrics,record_file,fold_aucs,test_fold_aucs,fold_record_matched_train_metrics,train_fold_aucs)

    # to print & record the best result of total  
    #pps="_total_best_"
    
    #model, best_epoch = load_curr_best_checkpoint(model,out_dir = output_dir,model_name = model_name,pps = pps,lossfunc=lossfunc,ps=ps)  
    pps="_k-fold-sub-fold-"+str(best_fold)+"_"
    model, best_epoch = load_curr_best_checkpoint(model,out_dir = output_dir,model_name = model_name,pps = pps,lossfunc=lossfunc,ps = ps)
        
    train_loss, train_losses, train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3],train_mets,features =validating(valid_loader = train_loader
                                , model = model
                                , criterion = criterion
                                ,  lossfunc=lossfunc
                                ,affix_box=affix_box
                                ,multi_M = multi_M
                                )
    train_acc,train_pre, train_rec, train_F1, train_spe =train_met.acc, train_met.pre,train_met.sen,train_met.F1, train_met.spe

    valid_loss, valid_losses, valid_acc, valid_accs, valid_met,[valid_loss_0,valid_loss_1,valid_loss_2,valid_loss_3],valid_mets, features= validating(valid_loader = valid_loader
                                , model = model
                                , criterion = criterion
                                ,  lossfunc=lossfunc
                                ,affix_box=affix_box
                                ,multi_M = multi_M
                                )
    valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c
    valid_acc,valid_pre, valid_rec, valid_F1, valid_spe = valid_met.acc,valid_met.pre,valid_met.sen, valid_met.F1, valid_met.spe

    test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3],test_mets, features = validating(valid_loader = test_loader
                                , model = model
                                , criterion = criterion
                                ,  lossfunc=lossfunc
                                ,affix_box=affix_box
                                ,multi_M = multi_M
                                )
    test_a,test_b,test_c = test_met.a, test_met.b, test_met.c
    test_acc,test_pre,test_rec, test_F1,test_spe = test_met.acc,test_met.pre,test_met.sen, test_met.F1, test_met.spe


    #train_auc = roc_auc_score(train_a.cpu(),train_b.cpu() ,multi_class = 'ovo',labels=[0,1,2])
    #valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu(),multi_class = 'ovo',labels=[0,1,2])
    #test_auc = roc_auc_score(test_a.cpu(),test_b.cpu(),multi_class = 'ovo',labels=[0,1,2])
    train_auc = roc_auc_score(train_a.cpu(),train_b.cpu()[:,1])
    valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu()[:,1])
    test_auc = roc_auc_score(test_a.cpu(),test_b.cpu()[:,1])


    recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_fold",record_file = None,ps_thing=[best_fold,best_fold,valid_auc,valid_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets)
    recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_fold",record_file = record_file,ps_thing=[best_fold,best_fold,valid_auc,valid_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets)

    draw_heatmap(train_met.CM,valid_met.CM, test_met.CM,title="fold_"+str(fold)+"_",save_path=pic_output_dir+'_total_'+str(fold))



    total_train_CM=total_train_CM/4.0
    total_test_CM=total_test_CM
    draw_heatmap(total_train_CM,total_valid_CM, total_test_CM,title="total_",save_path=pic_output_dir+'_truly')

    plt.subplot(2,3,1)
    precision,recall,thresholds = precision_recall_curve(total_train_a, total_train_b[:,1])
    plt.plot(precision,recall)

    plt.subplot(2,3,2)
    precision,recall,thresholds = precision_recall_curve(total_valid_a, total_valid_b[:,1])
    plt.plot(precision,recall)

    plt.subplot(2,3,3)
    precision,recall,thresholds = precision_recall_curve(total_test_a, total_test_b[:,1])
    plt.plot(precision,recall)
    plt.savefig(pic_output_dir+'_total_proc.png')

    #plt.show()
    plt.close()


    plt.subplot(2,3,1)
    fpr,tpr,thresholds = roc_curve(total_train_a, total_train_b[:,1])
    plt.plot(fpr,tpr)

    plt.subplot(2,3,2)
    fpr,tpr,thresholds = roc_curve(total_valid_a, total_valid_b[:,1])
    plt.plot(fpr,tpr)

    plt.subplot(2,3,3)
    fpr,tpr,thresholds = roc_curve(total_test_a, total_test_b[:,1])
    plt.plot(fpr,tpr)
    plt.savefig(pic_output_dir+'_total_roc.png')

    #plt.show()
    plt.close()
    





    torch.cuda.empty_cache()
    record_file.close()
    jsf.close()


if __name__ == "__main__":
    output_dir=opt.output_dir
    print("output_dir: ",output_dir)
    main(output_dir)
>>>>>>> 3a4a4f2 (20240625-code)

    