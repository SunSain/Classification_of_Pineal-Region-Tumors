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
import matplotlib.pyplot as plt
from utils2.weighted_CE import Weighted_CE

from copy import deepcopy
from utils2.self_KL import SelfKL,Self_L1,Self_L2,Self_L3
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
import timm.optim
import timm.scheduler
import math

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
    torch.cuda.set_device(3)
    DEVICE=torch.device('cuda')
else:
    DEVICE=torch.device('cpu')
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
print('DEVICE: ',DEVICE)

spare_criterion = nn.CrossEntropyLoss()
spare_selfKL_criterion = SelfKL(num_classes=opt.num_classes,lambda_0=opt.lambda_0,lambda_1=opt.lambda_1,lambda_2=opt.lambda_2,lambda_3=opt.lambda_2, CE_or_KL=opt.CE_or_KL) 
batchavg_class_criterion = BatchCriterion(num_classes=opt.num_classes,DEVICE=DEVICE)
singlebatchavg_class_criterion=SingleBatchCriterion()

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
    if not os.path.exists(best_model_path):
        return model
    checkpoint = torch.load(best_model_path,map_location = DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    return model
    

def convert(original): #[[0],[1],[0],[0]]→ [0,1,0,0]
    target=torch.Tensor(len(original))
    for i in range(len(original)):
        target[i]=original[i][0]
    target=target.type(torch.LongTensor).to(DEVICE)
    return target


def selfKL_weighted_validate(valid_loader, model, criterion,labels,multi_M):
    
    tasks=['L0','L1','L2','L3']
    loss_fn = {'L0':nn.CrossEntropyLoss(),'L1':Self_L1(),'L2':Self_L2(),'L3':Self_L3()}
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
        for i, (img,auged_img,sid,target, saffix, sradiomics)  in enumerate(valid_loader):

            target = torch.from_numpy(np.expand_dims(target,axis=1))
            input_img = img.to(DEVICE)
            input_affix = saffix.to(DEVICE)
            input_radiomics = sradiomics.to(DEVICE)
            input_auged_img = auged_img.to(DEVICE)
            target = convert(target)
            
            input_target = target.to(DEVICE)


            out,feat = model(input_img,input_affix)
            auged_out,auged_feat = model(input_auged_img,input_affix)
            
            new_target=(F.one_hot(target,num_classes=out.size(1))).float()  
            new_target=torch.clamp(new_target,min=0.0001,max=1.0)   
            new_out=F.softmax(out,dim=1)
            new_auged_out=F.softmax(auged_out,dim=1)

            losses_box=[]
            for i, t in enumerate(tasks):
                loss_0 = loss_fn[t](new_out, new_target)
                loss_1 = loss_fn[t](new_auged_out, new_target)
                loss=(loss_0+loss_1)/2
                losses_box.append(loss)

            #loss_1=0
            #loss_2=0
            Loss_0.update(losses_box[0]*input_img.size(0),input_img.size(0))
            Loss_1.update(losses_box[1]*input_img.size(0),input_img.size(0))
            Loss_2.update(losses_box[2]*input_img.size(0),input_img.size(0))
            Loss_3.update(losses_box[3]*input_img.size(0),input_img.size(0))

            loss =  opt.lambda_0*losses_box[0]+opt.lambda_1*losses_box[1]+opt.lambda_2*losses_box[2]+opt.lambda_3*losses_box[3]
            #0.01,0.9,0.18,0.01

            losses.update(loss*input_img.size(0),input_img.size(0))

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
        
        print("total_sid: ",total_sid)
        a=torch.tensor(total_target)
        b=torch.tensor(total_out)
        c=torch.tensor(total_pred)    
        # total metrics'calcultion
        print("a",a)
        print("b",b)
        print("c",c)

        if opt.num_classes==3:
            auc = roc_auc_score(a.cpu(),b.cpu(),multi_class = 'ovo',labels=[0,1,2])
        else:
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
        
        if opt.num_classes==3:
            auc = roc_auc_score(a.cpu(),b.cpu(),multi_class = 'ovo',labels=[0,1,2])
        else:
            auc = roc_auc_score(a.cpu(),b.cpu()[:,1])
        
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
 
def selfKL_weighted_train(train_loader, model, criterion, optimizer, epoch,labels,multi_M):
    tasks=['L0','L1','L2','L3']
    loss_fn = {'L0':nn.CrossEntropyLoss(),'L1':Self_L1(),'L2':Self_L2(),'L3':Self_L3()}
    losses = AverageMeter()
    maes = AverageMeter()
    Loss_0 = AverageMeter()
    Loss_1 = AverageMeter()
    Loss_2 = AverageMeter()
    Loss_3 = AverageMeter()

    
    for i, (img,auged_img,sid,target, saffix, sradiomics)  in enumerate(train_loader):

        target = torch.from_numpy(np.expand_dims(target,axis=1))
        input_img = img.to(DEVICE)
        input_affix = saffix.to(DEVICE)
        input_radiomics = sradiomics.to(DEVICE)
        input_auged_img = auged_img.to(DEVICE)
        target = convert(target)
        
        input_target = target.to(DEVICE)

        model.train()
        model.zero_grad()

        out,feat = model(input_img,input_affix)
        auged_out,auged_feat = model(input_auged_img,input_affix)

        new_target=(F.one_hot(target,num_classes=out.size(1))).float()  
        new_target=torch.clamp(new_target,min=0.0001,max=1.0)   
        new_out=F.softmax(out,dim=1)
        new_auged_out=F.softmax(auged_out,dim=1)

        losses_box=[]
        for i, t in enumerate(tasks):
            loss_0 = loss_fn[t](new_out, new_target)
            loss_1 = loss_fn[t](new_auged_out, new_target)
            loss=(loss_0+loss_1)/2
            losses_box.append(loss)

        #loss_1=0
        #loss_2=0
        Loss_0.update(losses_box[0]*input_img.size(0),input_img.size(0))
        Loss_1.update(losses_box[1]*input_img.size(0),input_img.size(0))
        Loss_2.update(losses_box[2]*input_img.size(0),input_img.size(0))
        Loss_3.update(losses_box[3]*input_img.size(0),input_img.size(0))

        loss =  opt.lambda_0*losses_box[0]+opt.lambda_1*losses_box[1]+opt.lambda_2*losses_box[2]+opt.lambda_3*losses_box[3]
        #0.01,0.9,0.18,0.01
        #0.9,0.2,2.1,0.9

        print("New total loss: ",loss)
        
        losses.update(loss*input_img.size(0),input_img.size(0))



        pred, mae = get_corrects(output=out, target=target)
        print("train_pred:",pred)
        print("out: ",out)
        print("target: ",target)
        maes.update(mae, input_img.size(0))
        
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



def All_weighted_train(train_loader, model, criterion, optimizer, epoch,labels,multi_M):
    tasks=['CE','SingleBatchAvg',"BatchAvg",'SelfKL']
    losses = AverageMeter()
    maes = AverageMeter()
    Loss_0 = AverageMeter()
    Loss_1 = AverageMeter()
    Loss_2 = AverageMeter()
    Loss_3 = AverageMeter()

    
    for i, (img,auged_img,sid,target, saffix, sradiomics)  in enumerate(train_loader):

        target = torch.from_numpy(np.expand_dims(target,axis=1))
        input_img = img.to(DEVICE)
        input_affix = saffix.to(DEVICE)
        input_radiomics = sradiomics.to(DEVICE)
        input_auged_img = auged_img.to(DEVICE)
        target = convert(target)
        
        input_target = target.to(DEVICE)

        model.train()
        model.zero_grad()

        out,feat = model(input_img,input_affix)
        auged_out,auged_feat = model(input_auged_img,input_affix)
        
        allfeats = torch.cat((feat,auged_feat), 0)
        
        spare_criterion = nn.CrossEntropyLoss()
        spare_selfKL_criterion = SelfKL(num_classes=opt.num_classes,lambda_0=opt.lambda_0,lambda_1=opt.lambda_1,lambda_2=opt.lambda_2,lambda_3=opt.lambda_2, CE_or_KL=opt.CE_or_KL) 
        
        for i, t in enumerate(tasks):
            if(t=="BatchAvg"):
                loss = batchavg_class_criterion(feat, target)
                bloss = batchavg_class_criterion(auged_feat, target)
                loss_2=loss+bloss
                loss_2=loss_2/2.0
            elif(t=="SingleBatchAvg"):
                allfeats = torch.cat((feat,auged_feat), 0)
                if len(sid)<2:
                    loss_3=0.0
                else:
                    loss_3 = singlebatchavg_class_criterion(allfeats)
                    
                loss_3=loss_3/2.0
            elif(t=="SelfKL"):
                loss,_,_,_,_= spare_selfKL_criterion(out,target)
                bloss,_ ,_,_,_ = spare_selfKL_criterion(auged_out,target)
                closs,_ ,_,_,_ = spare_selfKL_criterion(auged_out,F.softmax(out,dim=1),False)
                loss_1=loss+bloss+closs
                loss_1=loss_1/3.0
            else:
                loss = spare_criterion(out, target)
                bloss = spare_criterion(auged_out, target)
                loss_0=loss+bloss
                loss_0=loss_0/2.0

        #loss_1=0
        #loss_2=0
        Loss_0.update(loss_0*input_img.size(0),input_img.size(0))
        Loss_1.update(loss_1*input_img.size(0),input_img.size(0))
        Loss_2.update(loss_2*input_img.size(0),input_img.size(0))
        Loss_3.update(loss_3*input_img.size(0),input_img.size(0))

        #loss =  torch.add(0.3303436040878296*loss_0,0.6696563959121704*loss_3).requires_grad_(True)
        #loss =  0.225*loss_0+0.1*loss_2+0.5*loss_3+0.175*loss_1
        loss =  opt.lambda_CE*loss_0+opt.lambda_SelfKL*loss_1+opt.lambda_Batch*loss_2+opt.lambda_Sing*loss_3
        
         #0.9,0.2,2.1,0.9
         #0.15,0.1,0.1,0.65

            

        print("New total loss: ",loss)
        
        losses.update(loss*input_img.size(0),input_img.size(0))



        pred, mae = get_corrects(output=out, target=target)
        print("train_pred:",pred)
        print("out: ",out)
        print("target: ",target)
        maes.update(mae, input_img.size(0))
        
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


def MGDA_train(train_loader, model, criterion, optimizer, epoch,n_iter=0,writer = None,labels=[0,1],tasks=["CE","SingleBatchAvg"],loss_fn = {"CE":spare_criterion,"SelfKL":spare_selfKL_criterion,"SingleBatchAvg":singlebatchavg_class_criterion}):
    orig_tasks=tasks
    losses = AverageMeter()
    maes = AverageMeter()
    Loss_0 = AverageMeter()
    Loss_1 = AverageMeter()
    Loss_2 = AverageMeter()
    Loss_3 = AverageMeter()


    for i, (img,auged_img,sid,target, saffix, sradiomics) in enumerate(train_loader):
        n_iter+=1
        target = torch.from_numpy(np.expand_dims(target,axis=1))


        input_img = img.to(DEVICE)
        input_affix = saffix.to(DEVICE)
        input_radiomics = sradiomics.to(DEVICE)
        
        input_auged_img = auged_img.to(DEVICE)
        #print("input_affix: ",input_affix)
        

        # convert the input_img's shape: [8,91,109,91] → [8,1,91,109,91] ,to match the input_img of model
        #input_img = torch.reshape(input_img, [input_img.shape[0],1,input_img.shape[1],input_img.shape[2],input_img.shape[3]])
        #convert target's shape: [[1],[0]] → [1,0]
        target = convert(target)
        
        input_target = target.to(DEVICE)

        model.train()
        model.zero_grad()

        out,feat = model(input_img,input_affix)
        auged_out,auged_feat = model(input_auged_img,input_affix)
        
        allfeats = torch.cat((feat,auged_feat), 0)

        spare_criterion = nn.CrossEntropyLoss()
        spare_selfKL_criterion = SelfKL(num_classes=opt.num_classes,lambda_0=opt.lambda_0,lambda_1=opt.lambda_1,lambda_2=opt.lambda_2,lambda_3=opt.lambda_2, CE_or_KL=opt.CE_or_KL) 
        batchavg_class_criterion = BatchCriterion(num_classes=opt.num_classes,DEVICE=DEVICE)
        singlebatchavg_class_criterion=SingleBatchCriterion()
        loss_data = {}
        grads = {}
        scale = {}
        for t in tasks:
            # Comptue gradients of each loss function wrt parameters
            optimizer.zero_grad()
            out_t,feat = model(input_img,input_affix)
            auged_out,auged_feat = model(input_auged_img,input_affix)

            
            if(t=="BatchAvg"):
                loss = loss_fn[t](feat, target)
                bloss = loss_fn[t](auged_feat, target)
                loss=loss+bloss
                loss=loss/2.0
            elif(t=="SingleBatchAvg"):
                allfeats = torch.cat((feat,auged_feat), 0)
                if len(sid)<2:
                    loss=0.0
                    tasks=['CE']
                    continue
                else:
                    loss = loss_fn[t](allfeats)
                    loss=loss/2.0

            elif(t=="SelfKL"):
                loss,_,_,_,_= spare_selfKL_criterion(out_t,target)
                bloss,_ ,_,_,_ = spare_selfKL_criterion(auged_out, target)
                closs,_ ,_,_,_ = spare_selfKL_criterion(auged_out, F.softmax(out_t,dim=1),False)
                loss=loss+bloss+closs
                loss=loss/3.0
            else:
                loss = loss_fn[t](out_t, target)
                bloss = loss_fn[t](auged_out, target)
                loss=loss+bloss
                loss=loss/2.0
            print("task t: ",t)
            print("per loss: ",loss)
            print("loss.grad: ",loss.grad)
            
            loss.backward()
            loss_data[t] = loss
            
            grads[t] = []
            for param in model.parameters():
                if param.grad is not None:
                    #print("param.grad is not none")
                    grads[t].append(Variable(param.grad.data.clone(), requires_grad=False))
        rep = out_t
        
        #print("grads SingleBatchAvg: ",grads["SingleBatchAvg"])
        #print("grads BatchAvg: ",grads["BatchAvg"])
        print("out_t: ",out_t)
        
        # Normalize all gradients, this is optional and not included in the paper.
        if len(tasks)>1:
            gn = gradient_normalizers(grads, loss_data, 'loss+')
            print("gn: ",gn)
            for t in tasks:
                for gr_i in range(len(grads[t])):
                    #print("grads[t][gr_i]: ",grads[t][gr_i])
                    grads[t][gr_i] = grads[t][gr_i] / gn[t]

            # Frank-Wolfe iteration to compute scales.
            #print("grads: ",grads)
            sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
            for i, t in enumerate(tasks):
                scale[t] = float(sol[i])
        else:
            for i, t in enumerate(tasks):
                scale[t] = 1.0
        print("scale: ",scale)
        print("loss_data: ",loss_data)
        loss=0.

        allfeats = torch.cat((feat,auged_feat), 0)
        if len(sid)<2:
            loss_3=0.0
        else:
            loss_3 = singlebatchavg_class_criterion(allfeats)

        #loss = batchavg_class_criterion(feat, target)
        
        #bloss = batchavg_class_criterion(auged_feat, target)
        #loss_2=loss+bloss
        loss_2=0.0

        loss,_,_,_,_= spare_selfKL_criterion(out_t,target)
        
        bloss,_ ,_,_,_ = spare_selfKL_criterion(auged_out,target)
        
        closs,_ ,_,_,_ = spare_selfKL_criterion(auged_out,F.softmax(out_t,dim=1),False)
        loss_1=loss+bloss+closs

        loss = spare_criterion(out_t, target)
        bloss = spare_criterion(auged_out, target)
        loss_0=loss+bloss

        
        Loss_0.update(loss_0*input_img.size(0),input_img.size(0))
        Loss_1.update(loss_1*input_img.size(0),input_img.size(0))
        Loss_2.update(loss_2*input_img.size(0),input_img.size(0))
        Loss_3.update(loss_3*input_img.size(0),input_img.size(0))


        pred, mae = get_corrects(output=out, target=target)
        print("train_pred:",pred)
        print("out: ",out)
        print("target: ",target)
        maes.update(mae, input_img.size(0))
        
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
        optimizer.zero_grad()
        
        rep,feat = model(input_img,input_affix)
        auged_out,auged_feat = model(input_auged_img,input_affix)
        

        for i, t in enumerate(tasks):
            if(t=="BatchAvg"):
                loss = loss_fn[t](feat, target)
                bloss = loss_fn[t](auged_feat, target)
                loss_t=loss+bloss
                loss_t=loss_t/2.0
            elif(t=="SingleBatchAvg"):
                allfeats = torch.cat((feat,auged_feat), 0)
                if len(sid)<2:
                    loss=0.0
                else:
                    loss = loss_fn[t](allfeats)
                    
                loss_t=loss
                loss_t=loss_t/2.0
            elif(t=="SelfKL"):
                loss,_,_,_,_= spare_selfKL_criterion(out_t,target)
                bloss,_ ,_,_,_ = spare_selfKL_criterion(auged_out, target)
                closs,_ ,_,_,_ = spare_selfKL_criterion(auged_out, F.softmax(out_t,dim=1),False)
                loss_t=loss+bloss+closs
                loss_t=loss_t/3.0
            else:
                loss = loss_fn[t](out_t, target)
                bloss = loss_fn[t](auged_out, target)
                loss_t=loss+bloss
                loss_t=loss_t/2.0
            loss_data[t] = loss_t
            if i > 0:
                loss = loss + scale[t]*loss_t
                print("New scale and new loss val: ",scale[t], "; loss_"+t, " ;val: ",loss_t)
            else:
                loss = scale[t]*loss_t
                print("New scale and new loss val: ",scale[t], "; loss_"+t, " ;val: ",loss_t)
        print("New total loss: ",loss)
        
        losses.update(loss*input_img.size(0),input_img.size(0))
                
        writer.add_scalar('training_loss', loss, n_iter)
        for t in tasks:
            writer.add_scalar('training_loss_{}'.format(t), loss_data[t], n_iter)
            writer.add_scalar("scale_{}".format(t),scale[t],n_iter)

        print("loss:",loss," ; loss.grad: ",loss.grad,"; output is leaf? ",rep.is_leaf," ; out.grad:",rep.grad," ;scale: ",scale)
        loss.backward()
        optimizer.step()
        tasks=orig_tasks
    #total metrics'calcultion
    
    acc,pre,sen,spe,F1 ,mets = cal_metrics(CM)

    
    met= Metrics()
    met.update(a=0,b=0,c=0, auc=0, acc=acc,sen=sen,pre=pre,F1=F1,spe=spe,CM=CM)

    
    return losses.avg,losses, maes.avg, maes, met,[Loss_0.avg,Loss_1.avg,Loss_2.avg,Loss_3.avg],n_iter,scale
    


def MGDA_validate(valid_loader, model, criterion,n_iter=0,writer = None,labels =None, multi_M=True ,ps='valid',tasks=["CE","SingleBatchAvg"],loss_fn = {"CE":spare_criterion,"SelfKL":spare_selfKL_criterion,"BatchAvg":batchavg_class_criterion,"SingleBatchAvg":singlebatchavg_class_criterion}):
    
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
    tot_loss = {}
    tot_loss['all'] = 0.0
    for t in tasks:
            tot_loss[t] = 0.0
    num_val_batches=0
    num_val_items = 0
    model.eval() #because if allow model.eval, the bn wouldn't work, and the train_set'result would be different(model.train & model.eval)

    with torch.no_grad():
        
        for i, (img,auged_img,sid,target, saffix, sradiomics) in enumerate(valid_loader):


            input_img = img.to(DEVICE)
            input_affix = saffix.to(DEVICE)
            input_radiomics = sradiomics.to(DEVICE)
            input_auged_img = auged_img.to(DEVICE)
            
            target = torch.from_numpy(np.expand_dims(target,axis=1))
            target = convert(target)
            input_target = target.to(DEVICE)
            
            out,feat = model(input_img,input_affix)
            auged_out,auged_feat = model(input_auged_img,input_affix)
            
            allfeats = torch.cat((feat,auged_feat), 0)
            if len(sid)<2:
                loss_3=0.0
            else:
                loss_3 = singlebatchavg_class_criterion(allfeats)
                loss_3=loss_3/2.0

            #loss = batchavg_class_criterion(feat, target)
            
            #bloss = batchavg_class_criterion(auged_feat, target)
            #loss_2=loss+bloss
            #loss_2=loss_2/2.0
            loss_2=0.0

            loss,_,_,_,_= spare_selfKL_criterion(out,target)
            
            bloss,_ ,_,_,_ = spare_selfKL_criterion(auged_out,target)
            
            closs,_ ,_,_,_ = spare_selfKL_criterion(auged_out,F.softmax(out,dim=1),False)
            loss_1=loss+bloss+closs
            loss_1=loss_1/3.0

            loss = spare_criterion(out, target)
            bloss = spare_criterion(auged_out, target)
            loss_0=loss+bloss
            loss_0=loss_0/2.0
            
            loss=loss_0+loss_1+loss_2+loss_3
            
            loss_box={"CE":loss_0,"SelfKL":loss_1,"BatchAvg":loss_2,"SingleBatchAvg":loss_3}
            num_val_batches+=1
            num_val_items+=input_img.size(0)
            
            if not writer==None:
                for t in tasks:
                    tot_loss['all'] += loss_box[t]
                    tot_loss[t] += loss_box[t]
            #input.size(0) = batch_size 
            # the CE's ouput is averaged, so 
            losses.update(loss*input_img.size(0),input_img.size(0))
            Loss_0.update(loss_0*input_img.size(0),input_img.size(0))
            Loss_1.update(loss_1*input_img.size(0),input_img.size(0))
            Loss_2.update(loss_2*input_img.size(0),input_img.size(0))
            Loss_3.update(loss_3*input_img.size(0),input_img.size(0))

            pred, mae = get_corrects(output=out, target=target)
            maes.update(mae, input_img.size(0))

            # collect every output/pred/target, combine them together for total metrics'calculation
            total_target.extend(target)
            total_out.extend(torch.softmax(out,dim=1).cpu().numpy())
            total_pred.extend(pred.cpu().numpy())
 
            CM+=confusion_matrix(target.cpu(), pred.cpu(),labels= labels)
        
        a=torch.tensor(total_target)
        b=torch.tensor(total_out)
        c=torch.tensor(total_pred)    
        # total metrics'calcultion
        if opt.num_classes==3:
            auc = roc_auc_score(a.cpu(),b.cpu(),multi_class = 'ovo',labels=[0,1,2])
        else:
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
        
        
        if not writer==None:
            for t in tasks:
                writer.add_scalar(ps+'_loss_{}'.format(t), tot_loss[t]/num_val_batches, n_iter)
            writer.add_scalar(ps+'_loss', tot_loss['all']/num_val_items, n_iter)
              
        return losses.avg,losses, maes.avg, maes, met,[Loss_0.avg,Loss_1.avg,Loss_2.avg,Loss_3.avg],mets
    

def masked_constrained_train(train_loader, model, criterion, optimizer, epoch,labels,multi_M):
    
    Losses = AverageMeter()
    Loss_0 = AverageMeter()
    Loss_1 = AverageMeter()
    Loss_2 = AverageMeter()
    Loss_3 = AverageMeter()
    maes = AverageMeter()

    for i, stuff in enumerate(train_loader):
        if multi_M:
                (img,masked_img,_,_,_,_,sid,target, saffix,sradiomics) = stuff
        else:
                (img,masked_img,sid,target, saffix,sradiomics) = stuff
        target = torch.from_numpy(np.expand_dims(target,axis=1))
        img = img.to(DEVICE)
        masked_img = masked_img.to(DEVICE)
        input_affix = saffix.to(DEVICE)
        input_radiomics = sradiomics.to(DEVICE)
        #print("Intrainning, img.size: ",img.size)
        # convert the input_img's shape: [8,91,109,91] → [8,1,91,109,91] ,to match the input_img of model
        #input_img = torch.reshape(input_img, [input_img.shape[0],1,input_img.shape[1],input_img.shape[2],input_img.shape[3]])
        #convert target's shape: [[1],[0]] → [1,0]
        target = convert(target)

        model.train()
        model.zero_grad()

        img_out,feat = model(img,input_affix)
        masked_img_out,masked_feat = model(masked_img,input_affix)

            
        spare_criterion = nn.CrossEntropyLoss() 
        spare_selfKL_criterion = SelfKL(num_classes=opt.num_classes,lambda_0=opt.lambda_0,lambda_1=opt.lambda_1,lambda_2=opt.lambda_2,lambda_3=opt.lambda_2, CE_or_KL=opt.CE_or_KL) 

        if(opt.lossfunc == 'SelfKL' or opt.lossfunc == 'Weighted_SelfKL'):
                loss_0,_ ,_,_,_ = criterion(img_out, target)
                loss_1,_ ,_,_,_ = criterion(masked_img_out, target)
                loss_2,_ ,_,_,_ = criterion(masked_img_out, F.softmax(img_out,dim=1),False)
        elif(opt.lossfunc == 'Weighted_CE' or opt.lossfunc == 'FLoss' ):
            loss_0 = criterion(img_out, target) 
            loss_1 = criterion(masked_img_out, target)
            loss_2 = criterion(masked_img_out, F.softmax(img_out,dim=1),False) 
        elif(opt.lossfunc=="BatchAvg"):
            loss_0 = criterion(feat, target) + spare_criterion(img_out, target)
            loss_1 = criterion(masked_feat, target) + spare_criterion(masked_img_out, target)
            loss_2 = spare_criterion(masked_img_out, F.softmax(img_out,dim=1))
            loss_KL,_,_,_,_ =spare_selfKL_criterion(img_out,target)
            loss_0 +=loss_KL
            loss_KL_2,_,_,_,_ =spare_selfKL_criterion(masked_img_out,target)
            loss_1+=loss_KL_2
        elif(opt.lossfunc=="SingleBatchAvg"):
                allfeats = torch.cat((feat,masked_feat), 0)
                if len(sid)<2:
                    loss=0.0
                else:
                    loss = criterion(allfeats)
                loss_0 = loss+ spare_criterion(img_out, target)
                loss_1 = loss+ spare_criterion(masked_img_out, target)
                loss_2 = spare_criterion(masked_img_out, F.softmax(img_out,dim=1))

                loss_KL,_,_,_,_ =spare_selfKL_criterion(img_out,target)
                loss_0 +=loss_KL
                loss_KL_2,_,_,_,_ =spare_selfKL_criterion(masked_img_out,target)
                loss_1+=loss_KL_2
        elif(opt.lossfunc=="AllBatchAvg"):
                allfeats = torch.cat((feat,masked_feat), 0)
                if len(sid)<2:
                    loss=0.0
                else:
                    loss = criterion(allfeats)
                print("Validating criterion: ",criterion)
                batchavg_class_criterion = BatchCriterion(num_classes=opt.num_classes,DEVICE=DEVICE)
                loss_0 = loss+batchavg_class_criterion(feat, target) + spare_criterion(img_out, target)
                loss_1 = loss+batchavg_class_criterion(masked_feat, target) + spare_criterion(masked_img_out, target)
                loss_2 = spare_criterion(masked_img_out, F.softmax(img_out,dim=1))
                loss_KL,_,_,_,_ =spare_selfKL_criterion(img_out,target)
                loss_0 +=loss_KL
                loss_KL_2,_,_,_,_ =spare_selfKL_criterion(masked_img_out,target)
                loss_1+=loss_KL_2 
        else:        
            loss_0 = criterion(img_out, target) 
            loss_1 = criterion(masked_img_out, target)
            loss_2 = criterion(masked_img_out, F.softmax(img_out,dim=1)) 

        loss = opt.constrain_lambd * (loss_0 + loss_1) + loss_2
        

        #input_img.size(0) = batch_size 
        # the CE's ouput is averaged, so 
        Losses.update(loss*img.size(0),img.size(0)) 
        Loss_0.update(loss_0*img.size(0),img.size(0))
        Loss_1.update(loss_1*img.size(0),img.size(0))
        Loss_2.update(loss_2*img.size(0),img.size(0))
        Loss_3.update(opt.constrain_lambd * (loss_0 + loss_1)*img.size(0),img.size(0))

        pred, mae = get_corrects(output=img_out, target=target)

        maes.update(mae, img_out.size(0))
        CM=confusion_matrix(target.cpu(), pred.cpu(),labels= labels)
        
        acc,pre,sen,spe,F1,mets = cal_metrics(CM)

        if i%opt.print_freq ==0:
            print(
                'Epoch: [{0} / {1}]   [step {2}/{3}]\t'
                  'Loss ({loss.avg:.4f})\t'
                  'Loss_0 ({loss_0.avg:.4f})\t'
                  'Loss_1 ({loss_1.avg:.4f})\t'
                  'Loss_2 ({loss_2.avg:.4f})\t'
                  'Acc ({acc.avg:.4f})\t'
                  'Acc2 ({acc2:.4f})\t'
                  'sen ({sen:.4f})\t'
                  'pre ({pre:.4f})\t'
                  'F1 ({F1:.4f})\t'
                  'spe ({spe:.4f})\t'.format
                  ( epoch, opt.epochs, i, len(train_loader)
                  , loss=Losses,loss_0=Loss_0, loss_1=Loss_1,loss_2=Loss_2, acc=maes, acc2=acc, sen=sen, pre=pre, F1=F1, spe=spe )
                )
            print("CM: ",CM)
        loss.backward()
        optimizer.step()
    # total metrics'calcultion
    acc,pre,sen,spe,F1,mets  = cal_metrics(CM)


    met= Metrics()
    met.update(a=0,b=0,c=0, auc=0, acc=acc,sen=sen,pre=pre,F1=F1,spe=spe,CM=CM)

    return Losses.avg,Losses, maes.avg, maes, met,[Loss_0.avg,Loss_1.avg,Loss_2.avg,Loss_3.avg],None
    

def All_weighted_validate(valid_loader, model, criterion,labels,multi_M):
    
    tasks=['CE','SingleBatchAvg','BatchAvg','SelfKL']
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
        for i, (img,auged_img,sid,target, saffix, sradiomics)  in enumerate(valid_loader):

            target = torch.from_numpy(np.expand_dims(target,axis=1))
            input_img = img.to(DEVICE)
            input_affix = saffix.to(DEVICE)
            input_radiomics = sradiomics.to(DEVICE)
            input_auged_img = auged_img.to(DEVICE)
            target = convert(target)
            
            input_target = target.to(DEVICE)


            out,feat = model(input_img,input_affix)
            auged_out,auged_feat = model(input_auged_img,input_affix)
            
            allfeats = torch.cat((feat,auged_feat), 0)
            
            spare_criterion = nn.CrossEntropyLoss()
            spare_selfKL_criterion = SelfKL(num_classes=opt.num_classes,lambda_0=opt.lambda_0,lambda_1=opt.lambda_1,lambda_2=opt.lambda_2,lambda_3=opt.lambda_2, CE_or_KL=opt.CE_or_KL) 
            
            for i, t in enumerate(tasks):
                if(t=="BatchAvg"):
                    loss = batchavg_class_criterion(feat, target)
                    bloss = batchavg_class_criterion(auged_feat, target)
                    loss_2=loss+bloss
                    loss_2=loss_2/2.0
                elif(t=="SingleBatchAvg"):
                    allfeats = torch.cat((feat,auged_feat), 0)
                    if len(sid)<2:
                        loss_3=0.0
                    else:
                        loss_3 = singlebatchavg_class_criterion(allfeats)
                        
                    loss_3=loss_3/2.0
                elif(t=="SelfKL"):
                    loss,_,_,_,_= spare_selfKL_criterion(out,target)
                    bloss,_ ,_,_,_ = spare_selfKL_criterion(auged_out,target)
                    closs,_ ,_,_,_ = spare_selfKL_criterion(auged_out,F.softmax(out,dim=1),False)
                    loss_1=loss+bloss+closs
                    loss_1=loss_1/3.0
                else:
                    loss = spare_criterion(out, target)
                    bloss = spare_criterion(auged_out, target)
                    loss_0=loss+bloss
                    loss_0=loss_0/2.0

            #loss_1=0
            #loss_2=0
            Loss_0.update(loss_0*input_img.size(0),input_img.size(0))
            Loss_1.update(loss_1*input_img.size(0),input_img.size(0))
            Loss_2.update(loss_2*input_img.size(0),input_img.size(0))
            Loss_3.update(loss_3*input_img.size(0),input_img.size(0))

            #loss = torch.add(0.3303436040878296*loss_0,0.6696563959121704*loss_3).requires_grad_(True)
            #loss =  0.1*loss_0+0.9*loss_3
            #loss =  0.225*loss_0+0.1*loss_2+0.5*loss_3+0.175*loss_1
            loss =  opt.lambda_CE*loss_0+opt.lambda_SelfKL*loss_1+opt.lambda_Batch*loss_2+opt.lambda_Sing*loss_3
            

            losses.update(loss*input_img.size(0),input_img.size(0))

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
        
        print("total_sid: ",total_sid)
        a=torch.tensor(total_target)
        b=torch.tensor(total_out)
        c=torch.tensor(total_pred)    
        # total metrics'calcultion
        print("a",a)
        print("b",b)
        print("c",c)

        if opt.num_classes==3:
            auc = roc_auc_score(a.cpu(),b.cpu(),multi_class = 'ovo',labels=[0,1,2])
        else:
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
        
        if opt.num_classes==3:
            auc = roc_auc_score(a.cpu(),b.cpu(),multi_class = 'ovo',labels=[0,1,2])
        else:
            auc = roc_auc_score(a.cpu(),b.cpu()[:,1])
        
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
    
def masked_constrained_validate(valid_loader, model, criterion,labels,multi_M):
    
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

    model.eval() #because if allow model.eval, the bn wouldn't work, and the train_set'result would be different(model.train & model.eval)

    with torch.no_grad():
        for i, stuff in enumerate(valid_loader):
            if multi_M:
                (img,masked_img,_,_,_,_,sid,target, saffix,sradiomics) = stuff
            else:
                (img,masked_img,sid,target, saffix,sradiomics) = stuff
            target = torch.from_numpy(np.expand_dims(target,axis=1))
            target = convert(target)


            img = img.to(DEVICE)
            input_affix = saffix.to(DEVICE)
            masked_img = masked_img.to(DEVICE)
            input_radiomics=sradiomics.to(DEVICE)
            #input_img = torch.reshape(input_img, [input_img.shape[0],1,input_img.shape[1],input_img.shape[2],input_img.shape[3]])
            img_out,feat = model(img,input_affix)
            masked_img_out,masked_feat = model(masked_img,input_affix)

            spare_criterion = nn.CrossEntropyLoss()      
            spare_selfKL_criterion = SelfKL(num_classes=opt.num_classes,lambda_0=opt.lambda_0,lambda_1=opt.lambda_1,lambda_2=opt.lambda_2,lambda_3=opt.lambda_2, CE_or_KL=opt.CE_or_KL) 
      
            if(opt.lossfunc == 'SelfKL' or opt.lossfunc == 'Weighted_SelfKL'):
                 loss_0,_ ,_,_,_ = criterion(img_out, target)
                 loss_1,_ ,_,_,_ = criterion(masked_img_out, target)
                 loss_2,_ ,_,_,_ = criterion(masked_img_out, F.softmax(img_out,dim=1),False)
            
            elif(opt.lossfunc == 'Weighted_CE' or opt.lossfunc == 'FLoss' ):
                loss_0 = criterion(img_out, target) 
                loss_1 = criterion(masked_img_out, target)
                loss_2 = criterion(masked_img_out, F.softmax(img_out,dim=1),False) 
            elif(opt.lossfunc=="BatchAvg"):
                loss_0 = criterion(feat, target) + spare_criterion(img_out, target)
                loss_1 = criterion(masked_feat, target) + spare_criterion(masked_img_out, target)
                loss_2 = spare_criterion(masked_img_out, F.softmax(img_out,dim=1))

                loss_KL,_,_,_,_ =spare_selfKL_criterion(img_out,target)
                loss_0 +=loss_KL
                loss_KL_2,_,_,_,_ =spare_selfKL_criterion(masked_img_out,target)
                loss_1+=loss_KL_2
            elif(opt.lossfunc=="SingleBatchAvg"):
                allfeats = torch.cat((feat,masked_feat), 0)
                if len(sid)<2:
                    loss=0.0
                else:
                    loss = criterion(allfeats)
                loss_0 = loss+ spare_criterion(img_out, target)
                loss_1 = loss+ spare_criterion(masked_img_out, target)
                loss_2 = spare_criterion(masked_img_out, F.softmax(img_out,dim=1))

                loss_KL,_,_,_,_ =spare_selfKL_criterion(img_out,target)
                loss_0 +=loss_KL
                loss_KL_2,_,_,_,_ =spare_selfKL_criterion(masked_img_out,target)
                loss_1+=loss_KL_2
            elif(opt.lossfunc=="AllBatchAvg"):
                allfeats = torch.cat((feat,masked_feat), 0)
                if len(sid)<2:
                    loss=0.0
                else:
                    loss = criterion(allfeats)
                print("Validating criterion: ",criterion)
                batchavg_class_criterion = BatchCriterion(num_classes=opt.num_classes,DEVICE=DEVICE)
                loss_0 = loss+batchavg_class_criterion(feat, target) + spare_criterion(img_out, target)
                loss_1 = loss+batchavg_class_criterion(masked_feat, target) + spare_criterion(masked_img_out, target)
                loss_2 = spare_criterion(masked_img_out, F.softmax(img_out,dim=1))
                loss_KL,_,_,_,_ =spare_selfKL_criterion(img_out,target)
                loss_0 +=loss_KL
                loss_KL_2,_,_,_,_ =spare_selfKL_criterion(masked_img_out,target)
                loss_1+=loss_KL_2 
            else:        
                loss_0 = criterion(img_out, target) 
                loss_1 = criterion(masked_img_out, target)
                loss_2 = criterion(masked_img_out, F.softmax(img_out,dim=1)) 

            
            loss = opt.constrain_lambd * (loss_0 + loss_1) + (1.0-opt.constrain_lambd)*loss_2
                
            Losses.update(loss*img.size(0),img.size(0)) 
            
            Loss_0.update(loss_0*img.size(0),img.size(0))
            Loss_1.update(loss_1*img.size(0),img.size(0))
            Loss_2.update(loss_2*img.size(0),img.size(0))
            Loss_3.update( opt.constrain_lambd * (loss_0 + loss_1)*img.size(0),img.size(0))



            pred, mae = get_corrects(output=img_out, target=target)
            maes.update(mae, img.size(0))

            # collect every output/pred/target, combine them together for total metrics'calculation
            total_target.extend(target)
            print("vlidate: target: ",target)
            total_out.extend(torch.softmax(img_out,dim=1).cpu().numpy())
            total_pred.extend(pred.cpu().numpy())
 
            CM+=confusion_matrix(target.cpu(), pred.cpu(),labels= labels)
            
        a=torch.tensor(total_target)
        b=torch.tensor(total_out)
        c=torch.tensor(total_pred)    
        # total metrics'calcultion
        if opt.num_classes==3:
                auc = roc_auc_score(a.cpu(),b.cpu(),multi_class = 'ovo',labels=[0,1,2])
        else:
            auc = roc_auc_score(a.cpu(),b.cpu()[:,1])
        
        new_F1 = f1_score(a.cpu(),c.cpu(),average="weighted")
        pre = precision_score(a.cpu(),c.cpu(),average="weighted")
        recall = recall_score(a.cpu(),c.cpu(),average="weighted")
        
        new_CM= confusion_matrix(a.cpu(),c.cpu())
        print("new_F1: ",new_F1)
        print("pre:",pre)
        print("recall: ",recall)
        print("new_CM:", new_CM)
        print("CM: ",CM)
        
        
        acc,pre,sen,spe,F1,mets = cal_metrics(CM)
        F1 = f1_score(a.cpu(),c.cpu(),average="weighted")
        pre = precision_score(a.cpu(),c.cpu(),average="weighted")
        sen = recall_score(a.cpu(),c.cpu(),average="weighted")
        
        if opt.num_classes==3:
            auc = roc_auc_score(a.cpu(),b.cpu(),multi_class = 'ovo',labels=[0,1,2])
        else:
            auc = roc_auc_score(a.cpu(),b.cpu()[:,1])
   
        print('Confusion Matirx : ',CM,'[Metrics]-Accuracy(mean): ' , acc,'- Sensitivity : ', sen*100,'- Specificity : ',spe*100,'- Precision: ',pre*100,'- F1 : ', F1*100,'- auc : ',auc*100)

        # Metrics is a DIY_package to store all metrics(acc, auc, F1,pre, recall, spe,CM, outpur, pred,target)
        met= Metrics() 
        met.update(a=a,b=b,c=c, acc=acc,sen=recall,pre=pre,F1= F1,spe= spe,auc=auc,CM=CM)
       
        #mets = [met_0,met_1,met_2]

        
        return Losses.avg,Losses, maes.avg, maes, met,[Loss_0.avg,Loss_1.avg,Loss_2.avg,Loss_3.avg],mets



def single_batchav_train(train_loader, model, criterion, optimizer, epoch,labels,multi_M):
    
    losses = AverageMeter()
    maes = AverageMeter()
    Loss_0 = AverageMeter()
    Loss_1 = AverageMeter()
    Loss_2 = AverageMeter()
    Loss_3 = AverageMeter()


    for i, stuff  in enumerate(train_loader):
        (img,auged_img,sid,target, saffix, sradiomics) = stuff
        target = torch.from_numpy(np.expand_dims(target,axis=1))
        print("Training sid:",sid)

        input_img = img.to(DEVICE)
        input_affix = saffix.to(DEVICE)
        input_radiomics = sradiomics.to(DEVICE)
        
        input_auged_img = auged_img.to(DEVICE)
        #print("input_affix: ",input_affix)
        

        # convert the input_img's shape: [8,91,109,91] → [8,1,91,109,91] ,to match the input_img of model
        #input_img = torch.reshape(input_img, [input_img.shape[0],1,input_img.shape[1],input_img.shape[2],input_img.shape[3]])
        #convert target's shape: [[1],[0]] → [1,0]
        target = convert(target)
        
        input_target = target.to(DEVICE)

        model.train()
        model.zero_grad()

        out,feat = model(input_img,input_affix)
        auged_out,auged_feat = model(input_auged_img,input_affix)
        
        allfeats = torch.cat((feat,auged_feat), 0)

        spare_criterion = nn.CrossEntropyLoss()
        spare_selfKL_criterion = SelfKL(num_classes=opt.num_classes,lambda_0=opt.lambda_0,lambda_1=opt.lambda_1,lambda_2=opt.lambda_2,lambda_3=opt.lambda_2, CE_or_KL=opt.CE_or_KL) 
        batchavg_class_criterion = BatchCriterion(num_classes=opt.num_classes,DEVICE=DEVICE)
        
        if len(sid)<2:
                loss=0.0
        else:
                loss = criterion(allfeats)
        loss_0 = spare_criterion(out,target)
        loss_1 = spare_criterion(auged_out,target)
        
        if opt.lossfunc =="AllBatchavg":
            loss = opt.single_batchavg_lamda*loss
            loss_2 = opt.batchavg_lamda*batchavg_class_criterion(feat,input_target)
            loss_3 = opt.batchavg_lamda*batchavg_class_criterion(auged_feat,input_target)
        elif opt.lossfunc=="SingleBatchAvg_selfKL":
            loss_KL,_ ,_,_,_ =spare_selfKL_criterion(out,target)
            loss_2 = loss_KL*opt.selfKL_lamda/2
            loss_KL,_ ,_,_,_ =spare_selfKL_criterion(auged_out,target)
            loss_3 = loss_KL*opt.selfKL_lamda/2
        else:
            loss_2,loss_3 = 0.,0.
        #loss_KL,_ ,_,_,_ =spare_selfKL_criterion(out,target)
        loss+=loss_0+loss_1+loss_2+loss_3

        #input_img.size(0) = batch_size 
        # the CE's ouput is averaged, so 
        losses.update(loss*input_img.size(0),input_img.size(0))
        Loss_0.update(loss_0*input_img.size(0),input_img.size(0))
        Loss_1.update(loss_1*input_img.size(0),input_img.size(0))
        Loss_2.update(loss_2*input_img.size(0),input_img.size(0))
        Loss_3.update(loss_3*input_img.size(0),input_img.size(0))


        pred, mae = get_corrects(output=out, target=target)
        print("train_pred:",pred)
        print("out: ",out)
        print("target: ",target)
        maes.update(mae, input_img.size(0))
        
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

def single_batchavg_validate(valid_loader, model, criterion,labels,multi_M):
    
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
            (img,auged_img,sid,target, saffix, sradiomics) = stuff
            print("Validating sid:",sid)
            target = torch.from_numpy(np.expand_dims(target,axis=1))
            target = convert(target)
            total_sid.append(sid)


            input_img = img.to(DEVICE)
            input_affix = saffix.to(DEVICE)
            input_radiomics = sradiomics.to(DEVICE)
            input_target = target.to(DEVICE)
            
            input_auged_img = auged_img.to(DEVICE)
            #print("input_affix: ",infput_affix)
            #input_img = torch.reshape(input_img, [input_img.shape[0],1,input_img.shape[1],input_img.shape[2],input_img.shape[3]])
            
            out,feat = model(input_img,input_affix)
            auged_out,auged_feat = model(input_auged_img,input_affix)
            
            allfeats = torch.cat((feat,auged_feat), 0)

            spare_criterion = nn.CrossEntropyLoss()
            spare_selfKL_criterion = SelfKL(num_classes=opt.num_classes,lambda_0=opt.lambda_0,lambda_1=opt.lambda_1,lambda_2=opt.lambda_2,lambda_3=opt.lambda_2, CE_or_KL=opt.CE_or_KL) 
            batchavg_class_criterion = BatchCriterion(num_classes=opt.num_classes,DEVICE=DEVICE)
            if len(sid)<2:
                loss=0.0
            else:
                loss = criterion(allfeats)
        
            loss_0 = spare_criterion(out,target)
            loss_1 = spare_criterion(auged_out,target)
            if opt.lossfunc =="AllBatchavg":
                loss = opt.single_batchavg_lamda*loss
                loss_2 = opt.batchavg_lamda*batchavg_class_criterion(feat,input_target)
                loss_3 = opt.batchavg_lamda*batchavg_class_criterion(auged_feat,input_target)
            elif opt.lossfunc=="SingleBatchAvg_selfKL":
                loss_KL,_ ,_,_,_ =spare_selfKL_criterion(out,target)
                loss_2 = loss_KL*opt.selfKL_lamda/2
                loss_KL,_ ,_,_,_ =spare_selfKL_criterion(auged_out,target)
                loss_3 = loss_KL*opt.selfKL_lamda/2
            else:
                loss_2,loss_3 = 0.,0.
            #loss_KL,_ ,_,_,_ =spare_selfKL_criterion(out,target)
            loss+=loss_0+loss_1+loss_2+loss_3
            #loss = loss_0+loss_1+loss_KL
            
            

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
        
        print("total_sid: ",total_sid)
        a=torch.tensor(total_target)
        b=torch.tensor(total_out)
        c=torch.tensor(total_pred)    
        # total metrics'calcultion
        print("a",a)
        print("b",b)
        print("c",c)

        if opt.num_classes==3:
            auc = roc_auc_score(a.cpu(),b.cpu(),multi_class = 'ovo',labels=[0,1,2])
        else:
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
        
        if opt.num_classes==3:
            auc = roc_auc_score(a.cpu(),b.cpu(),multi_class = 'ovo',labels=[0,1,2])
        else:
            auc = roc_auc_score(a.cpu(),b.cpu()[:,1])
        
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
    
    pre = (met_0.pre+met_1.pre+met_2.pre)/2
    sen = (met_0.sen+met_1.sen+met_2.sen)/2
    spe = (met_0.spe+met_1.spe+met_2.spe)/3
    F1 = (met_0.F1+met_1.F1+met_2.F1)/3 
    
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
    
    pre = (met_0.pre+met_1.pre+met_2.pre)/3
    sen = (met_0.sen+met_1.sen+met_2.sen)/3
    spe = (met_0.spe+met_1.spe+met_2.spe)/3
    F1 = (met_0.F1+met_1.F1+met_2.F1)/3 
    return acc,met_0,met_1,met_2,pre,sen,spe,F1

def get_weights_for_balanced_classes(dataset,nclasses,multi_M):
    count=[0]*nclasses
    print("dataset: ",dataset)
    print("dataset[0]:",dataset[0])
    for i, item in enumerate(dataset):
        if opt.train_form =="masked_constrained_train":
            if multi_M:
                (img,_,t1_img,t1_img_aug,t2_img,t2_img_aug,_,target, saffix,sradiomics) = item
            else:
                (data,auged_data,sid,target,saffix,sradiomics) = item
                print("(data,auged_data,sid,target,saffix,sradiomics): ",data,auged_data,sid,target,saffix,sradiomics)
                print("target: ",target)
        else:
            if multi_M:
                (img,t1_img,t2_img ,_,target, saffix, sradiomics) = item
            else:
                (img,auged_img,_,target, saffix,sradiomics) = item
        count[target]+=1
        print("i: target: ",i,";",target)
    w_per_class = [0.]*nclasses
    print("count: ",count)
    N=float(sum(count))
    for i in range(nclasses):
        w_per_class[i] = N/float(count[i])
    weight = [0]*len(dataset)

    if opt.train_form =="masked_constrained_train":
        for idx, item  in enumerate(dataset):
            if multi_M:
                (img,_,t1_img,t1_img_aug,t2_img,t2_img_aug,_,target, saffix,sradiomics) = item
            else:
                (data,auged_data,sid,target,saffix,sradiomics) = item
            weight[idx]=w_per_class[target]
                
    else:
        for idx, item  in enumerate(dataset):
            if multi_M:
                (img,t1_img,t2_img,_,target, saffix, sradiomics) = item
            else:
                (img,auged_data,_,target, saffix,sradiomics) = item
            weight[idx]=w_per_class[target]
            
    print("count: ",count)
    print("w_per_class: ",w_per_class)
    print("weight: ",weight)
    return weight

class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())

def custom_lr_scheduler(epoch):
        # 学习率初始值
        
        warm_up_iter = opt.warm_up_iter
        lr_step = opt.lr_step
        T_max = opt.epochs	# 周期
        lr_max = opt.max_lr	# 最大值: 0.1
        lr_min = opt.min_lr	# 最小值: 1e-5
        lr = 1e-8
        
        
        lambda0 = lambda cur_iter: (cur_iter % 10+1 )*10**(cur_iter // 5)*1e-4 if  cur_iter < warm_up_iter else  \
                            (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.1
        # 学习率下降到一定值后上升再下降的逻辑
        if epoch < 20:
            lr = (epoch % 10+1 )*10**(epoch // 5)*1e-4
        elif epoch<0.5*opt.epochs:
            lr = (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (epoch-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.1
        else:
            lr = (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (epoch-0.5*opt.epochs)/(T_max-warm_up_iter)*math.pi)))/0.1
        
        return lr

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

    testauc_record_file_path = opt.output_dir+"test_ROC.txt"
    testauc_record_file=open(testauc_record_file_path,"a")
    
    data_path=opt.data_path
    test_data_path = opt.testdata_path

    print("=========== start train the brain age estimation model =========== \n")
    print(" ==========> Using {} processes for data loader.".format(opt.num_workers))

    #load the training_data  and test_data (training_data will be splited later for cross_validation)

    multi_M = opt.multi_m
   
    total_comman_total_file=[1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 27, 28, 29, 30, 31, 34, 35, 36, 37, 39, 40, 42, 43, 44, 46, 48, 49, 51, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 95, 96, 97, 100, 102, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122]
    test_comman_total_file=[3, 4, 8, 16, 20, 22, 25, 26, 38, 45, 47, 56, 64, 65, 66, 75, 93, 94, 98, 99, 101, 111, 120]
    #total_comman_total_file=[1, 2, 59, 60, 61, 62, 63, 121, 122] #32,33,41,50,53,83,89,103
    #test_comman_total_file=[3, 4, 65, 66]
    
    
    #comman_total_file=[1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 27, 28, 29, 30, 31, 34, 35, 36, 37, 39, 40, 42, 43, 44, 46, 48, 49, 51, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 95, 96, 97, 100, 102, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122]
    y_box = [0 if i<61 else 1 for i in total_comman_total_file] 
 
    if multi_M:
        total_file = DIY_Folder_3M(num_classes = opt.num_classes,data_path=data_path,train_form = opt.train_form,root_mask_radiomics = opt.root_bbx_path,use_radiomics=opt.use_radiomics,istest=False,sec_pair_orig=opt.sec_pair_orig,multiclass = opt.multiclass,vice_path = ""
                                   ,multi_M = multi_M,t1_path = opt.t1_path
                                   , t2_path=opt.t2_path, t1c_path = opt.t1cF_path
                                   , valid_sid_box=total_comman_total_file)
    else:
        total_file = DIY_Folder(num_classes = opt.num_classes,data_path=data_path,train_form = opt.train_form,root_mask_radiomics = opt.root_bbx_path,use_radiomics=opt.use_radiomics,istest=False,sec_pair_orig=opt.sec_pair_orig,multiclass = opt.multiclass,vice_path = ""
                                , valid_sid_box=total_comman_total_file)


    print("empty_radiomics_box: ",total_file.empty_radiomics_box)
    print("len(total_file): ",len(total_file))
    print("total_file.gety(): ",total_file.gety())
    #root_radiomics="",root_mask_radiomics="",use_radiomics=True, norm_radiomics=True
    if multi_M:
        test_file = DIY_Folder_3M(num_classes = opt.num_classes,data_path=test_data_path,train_form = opt.train_form,root_mask_radiomics = opt.test_root_bbx_path,use_radiomics=opt.use_radiomics,istest=True,sec_pair_orig= opt.sec_pair_orig,multiclass = opt.multiclass
                                  ,multi_M = multi_M,t1_path = opt.t1_test_path
                                   , t2_path=opt.t2_test_path, t1c_path = opt.t1c_test_path
                                   , valid_sid_box=test_comman_total_file)
    else:
        test_file = DIY_Folder(num_classes = opt.num_classes,data_path=test_data_path,train_form = opt.train_form,root_mask_radiomics = opt.test_root_bbx_path,use_radiomics=opt.use_radiomics,istest=True,sec_pair_orig= opt.sec_pair_orig,multiclass = opt.multiclass
                               , valid_sid_box=test_comman_total_file)
    test_data,test_sids = test_file.select_dataset(data_idx=[i for i in range(len(test_file))], aug=False,use_secondimg=opt.usesecond,noSameRM=opt.noSameRM, usethird = opt.usethird,comman_total_file=test_comman_total_file)

    test_loader= torch.utils.data.DataLoader(test_data
                                                , batch_size = opt.batch_size
                                                , num_workers = opt.num_workers
                                                , pin_memory = True
                                                , drop_last = False
                                                )
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
    usesecond = opt.usesecond
    num_classes = opt.num_classes
    input_channel = 2 if usesecond else 1
    input_channel = 3 if multi_M else input_channel
    input_channel = 4 if multi_M and usesecond else input_channel
    
    target_names =  ['class 0', 'class 1','class 2'] if num_classes==3  else ['class 0', 'class 1']
    labels = [0,1,2] if num_classes==3 else [0,1]
    loss_func_dict = { 'CE' : nn.CrossEntropyLoss().to(DEVICE)
                      , 'Weighted_CE' : Weighted_CE(classes_weight=opt.loss_weight,n_classes=num_classes)
                     , 'SelfKL' :  SelfKL(num_classes=num_classes,lambda_0=opt.lambda_0,lambda_1=opt.lambda_1,lambda_2=opt.lambda_2,lambda_3=opt.lambda_2, CE_or_KL=opt.CE_or_KL)                    
                     , 'Weighted_SelfKL': SelfKL(num_classes=num_classes,lambda_0=opt.lambda_0,lambda_1=opt.lambda_1,lambda_2=opt.lambda_2,lambda_3=opt.lambda_2, CE_or_KL=opt.CE_or_KL,classes_weight=opt.loss_weight)
                     , 'FLoss':Focal_Loss(num_classes=num_classes,gamma=opt.FL_gamma)
                     , 'BatchAvg': BatchCriterion(num_classes=num_classes,DEVICE=DEVICE)
                     ,'SingleBatchAvg': SingleBatchCriterion()
                     ,"AllBatchavg":SingleBatchCriterion()
                     ,"SingleBatchAvg_selfKL":SingleBatchCriterion()
                     }
    feature_align = True if opt.lossfunc=='BatchAvg'  else False
    criterion = loss_func_dict[opt.lossfunc]

    train_func_dict={"SelfKL":selfKL_weighted_train,
                     "":All_weighted_train,
                     "None":All_weighted_train,
                     "none":All_weighted_train
                    }
    weighted_train = train_func_dict[opt.train_form]
    valid_func_dict={"SelfKL":selfKL_weighted_validate,
                     "":All_weighted_validate,
                     "None":All_weighted_validate,
                     "none":All_weighted_validate
                }
    weighted_validate = valid_func_dict[opt.train_form]
    
    if opt.train_form == "single_batchavg_train":
        criterion=SingleBatchCriterion()


    sum_writer = tensorboardX.SummaryWriter(opt.output_dir)

    
    print(" ==========> All settled. Training is getting started...")
    print(" ==========> Training takes {} epochs.".format(epochs))
    print(" ==========> output_dir: ",opt.output_dir)
    print(" ==========> task: ",num_classes)

    # split the training_data into K fold with StratifiedKFold(shuffle = True)
    k=opt.kfold
    splits=StratifiedKFold(n_splits=k,shuffle=True,random_state=opt.random_seed)

    # to record metrics: the best_acc of k folds, best_acc of each fold
    foldperf={}
    fold_best_auc,best_fold,best_epoch=-1,1,0
    fold_record_valid_metrics,fold_record_matched_test_metrics,fold_aucs,test_fold_aucs,fold_record_matched_train_metrics,train_fold_aucs=[],[],[],[],[],[]
    fold_best_statedict = None  

    fold_test_tpr,fold_test_fpr=[],[]
    chores_time = time.time()-begin_time
    print("[....Chores time....]: %dh, %dmin, %ds "%(int(chores_time/3600),int(chores_time/60),int(chores_time%60)))
    begin_time = time.time()

    #================= begin to train, choose 1 of k folds as validation =================================
    print("======================== start train ================================================ \n")

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(total_comman_total_file)),y_box)):
            
 
        begin_training_time = time.time()-begin_time
        print("[....Training time....]: %dh, %dmin, %ds "%(int(begin_training_time/3600),int(begin_training_time/60),int(begin_training_time%60)))
        print("[....Training time....]: %dh, %dmin, %ds "%(int(begin_training_time/3600),int(begin_training_time/60),int(begin_training_time%60)),file=record_file)
        begin_time = time.time()

        print("\n============ FOLD %d ============\n"%(fold),file=record_file)
        print('Fold {}'.format(fold))
        


        #valid_data(no augmentation: aug=False) & train_data(need augmentation:aug = True)
        print("Get valid set")
        vali_data,vali_sids=total_file.select_dataset(data_idx=val_idx, aug=False,use_secondimg=opt.usesecond,noSameRM=opt.noSameRM, usethird=opt.usethird,comman_total_file=total_comman_total_file)
        print("Got train set")
        train_data,train_sids=total_file.select_dataset(data_idx=train_idx, aug=True,aug_form=opt.aug_form,use_secondimg=opt.usesecond,noSameRM=opt.noSameRM, usethird=opt.usethird,comman_total_file=total_comman_total_file)
        print("train_data: ",train_data)
        print('train_sids: ',train_sids)
        print("empty_radiomics_box: ",total_file.empty_radiomics_box)

        if opt.use_radiomics:
            radio_mean, radio_std = train_data.calc_own_radiomics_mean()
            vali_data.inject_other_mean(radio_mean, radio_std)
            test_data.inject_other_mean(radio_mean, radio_std)
            
            print("train_data. radio_mean: ",train_data.radio_mean)
            
        weights = get_weights_for_balanced_classes(train_data,num_classes,multi_M)
        print("\nweights\n",weights,file=record_file) 
        weights = torch.Tensor(weights)
        sampler = CustomWeightedRandomSampler(weights,len(weights))
        
        train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers = opt.num_workers
                                                , pin_memory = True
                                                , drop_last = False
                                                , sampler = sampler)

        valid_loader = DataLoader(vali_data, batch_size=opt.batch_size, num_workers = opt.num_workers
                                                , shuffle = True
                                                , pin_memory = True
                                                , drop_last = False)
           
           
        # choose model
        TEST=False
        type_moda=opt.type_moda
        
        if opt.model == "ResNet18":
            model = ResNet18(num_classes=num_classes,input_channel=input_channel,use_radiomics=opt.use_radiomics,feature_align=feature_align,use_clinical=opt.use_clinical)
            
            if type_moda=="T1":
                
                best_pretrained_model_path = "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1_composed_CE/model_result/ResNet18_CE_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar"%fold
                if opt.use_clinical:
                    best_pretrained_model_path="/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_composed_RM_2/model_result/ResNet18_CE_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar"%fold
                    #best_pretrained_model_path="/home/chenxr/Pineal_region/after_12_08/Results/Single_batchavg/Two/T1_singlebatchavg_composed_ResNet18/model_result/ResNet18_SingleBatchAvg_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar"%fold
            #orig_path = "/home/chenxr/Pineal_region/after_12_08/Results/0630_all/Two/none_sampler_composed_ResNet18/model_result/"
            #orig_path ="/home/chenxr/Pineal_region/after_12_08/Results/0704_T1C/Two/new_data/bbx_seg_sampler_Composed_ResNet18/model_result/"
            #orig_path ="/home/chenxr/Pineal_region/after_12_08/Results/0704_T1C/Two/new_data/none_sampler_Composed_ResNet18/model_result/"
                
                checkpoint = torch.load(best_pretrained_model_path,map_location = DEVICE)
                if checkpoint['state_dict'].get( "fc_two.weight",None) !=None:
                     checkpoint['state_dict']['fc.weight']=checkpoint['state_dict']['fc_two.weight']
                     checkpoint['state_dict']['fc.bias']=checkpoint['state_dict']['fc_two.bias']
                     del checkpoint['state_dict']['fc_two.weight']
                     del checkpoint['state_dict']['fc_two.bias']
                model.load_state_dict(checkpoint['state_dict'])
            
            if type_moda=="T2":
                
                best_pretrained_model_path = "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T2_composed_CE/model_result/ResNet18_CE_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar"%fold
                if opt.use_clinical:
                    best_pretrained_model_path="/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T2_composed_ce/model_result/ResNet18_CE_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar"%fold
                
            #orig_path = "/home/chenxr/Pineal_region/after_12_08/Results/0630_all/Two/none_sampler_composed_ResNet18/model_result/"
            #orig_path ="/home/chenxr/Pineal_region/after_12_08/Results/0704_T1C/Two/new_data/bbx_seg_sampler_Composed_ResNet18/model_result/"
            #orig_path ="/home/chenxr/Pineal_region/after_12_08/Results/0704_T1C/Two/new_data/none_sampler_Composed_ResNet18/model_result/"
                
                checkpoint = torch.load(best_pretrained_model_path,map_location = DEVICE)
                if checkpoint['state_dict'].get( "fc_two.weight",None) !=None:
                     checkpoint['state_dict']['fc.weight']=checkpoint['state_dict']['fc_two.weight']
                     checkpoint['state_dict']['fc.bias']=checkpoint['state_dict']['fc_two.bias']
                     del checkpoint['state_dict']['fc_two.weight']
                     del checkpoint['state_dict']['fc_two.bias']
                
                model.load_state_dict(checkpoint['state_dict'])

            if type_moda=="T1C":
                
                best_pretrained_model_path = "/home/chenxr/Pineal_region/after_12_08/Results/Noclinical_PureMRI/Two/T1C_composed_CE/model_result/ResNet18_CE_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar"%fold
                if opt.use_clinical:
                    #best_pretrained_model_path="/home/chenxr/Pineal_region/after_12_08/Results/use_clinical/Two/T1C_clinical_ce/model_result/ResNet18_CE_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar"%fold
                    #best_pretrained_model_path="/home/chenxr/Pineal_region/after_12_08/Results/Use_Clinical/Two/T1C_composed_RM/model_result/ResNet18_CE_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar"%fold
                    best_pretrained_model_path="/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_Singlebatchavg_ce_new_37/model_result/ResNet18_CE_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar"%fold
            #orig_path = "/home/chenxr/Pineal_region/after_12_08/Results/0630_all/Two/none_sampler_composed_ResNet18/model_result/"
            #orig_path ="/home/chenxr/Pineal_region/after_12_08/Results/0704_T1C/Two/new_data/bbx_seg_sampler_Composed_ResNet18/model_result/"
            #orig_path ="/home/chenxr/Pineal_region/after_12_08/Results/0704_T1C/Two/new_data/none_sampler_Composed_ResNet18/model_result/"
                
                checkpoint = torch.load(best_pretrained_model_path,map_location = DEVICE)
                model.load_state_dict(checkpoint['state_dict'])
            
        elif opt.model == "ResNet34":
            model = ResNet34(num_classes=num_classes,input_channel=input_channel,use_radiomics=opt.use_radiomics,feature_align=feature_align)
        elif opt.model == "ResNet10":
            model = ResNet10(num_classes=num_classes,input_channel=input_channel,use_radiomics=opt.use_radiomics,feature_align=feature_align)
        elif opt.model == "ResNet24":
            model = ResNet24(num_classes=num_classes,input_channel=input_channel,use_radiomics=opt.use_radiomics,feature_align=feature_align)
        elif opt.model == "ResNet30":
            model = ResNet30(num_classes=num_classes,input_channel=input_channel,use_radiomics=opt.use_radiomics,feature_align=feature_align)
        else:
            print("[ERROR: ] Wrong model chosen\n")

        train_loss_box ,train_acc_box= [],[]
        valid_loss_box,valid_acc_box=[],[]
        t_acc_record,v_acc_record=[],[]
        test_acc_record,test_loss_box,test_acc_box=[],[],[]

        saved_metrics,saved_epoch=[],[]

        best_auc, sofar_valid_acc=-1,-1

        best_statedict = deepcopy(model.state_dict())
        best_epoch=0

        
        if TEST==True:
            model = model.to(DEVICE)
            model.eval()
            train_loss, train_losses, validating_train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3],train_mets= weighted_validate(valid_loader = train_loader, model = model, criterion = criterion, labels = labels, multi_M= multi_M)
            train_a,train_b,train_c = train_met.a, train_met.b, train_met.c
            train_pre, train_rec, train_F1, train_spe = train_met.pre,train_met.sen,train_met.F1, train_met.spe
            
            
            valid_loss, valid_losses, valid_acc, valid_accs, valid_met,[valid_loss_0,valid_loss_1,valid_loss_2,valid_loss_3],valid_mets= weighted_validate(valid_loader = valid_loader, model = model, criterion = criterion, labels = labels, multi_M= multi_M)
            valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c
            valid_pre, valid_rec, valid_F1, valid_spe = valid_met.pre,valid_met.sen, valid_met.F1, valid_met.spe
            valid_met_0,valid_met_1,valid_met_2 = valid_mets[0],valid_mets[1],valid_mets[2]

            test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3],test_mets= weighted_validate(valid_loader = test_loader, model = model, criterion = criterion, labels = labels, multi_M= multi_M)
            test_a,test_b,test_c = test_met.a, test_met.b, test_met.c
            test_pre,test_rec, test_F1,test_spe = test_met.pre,test_met.sen, test_met.F1, test_met.spe
            test_met_0,test_met_1,test_met_2 = test_mets[0],test_mets[1],test_mets[2]
            
            #print("[Fold-%d | Epoch : %d]: training_train_acc: "%(fold,epoch),train_acc, " ; validating_acc: ",validating_train_acc)
            if opt.num_classes==3:
                valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu() ,multi_class = 'ovo',labels=[0,1,2])
                test_auc = roc_auc_score(test_a.cpu(),test_b.cpu() ,multi_class = 'ovo',labels=[0,1,2])
                train_auc = roc_auc_score(train_a.cpu(),train_b.cpu() ,multi_class = 'ovo',labels=[0,1,2])
            else:
                valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu()[:,1])
                test_auc = roc_auc_score(test_a.cpu(),test_b.cpu()[:,1])
                train_auc = roc_auc_score(train_a.cpu(),train_b.cpu()[:,1])

            valid_loss_box.append(valid_loss.detach().cpu())
            valid_acc_box.append(valid_acc)
            test_loss_box.append(test_loss)       
            test_acc_box.append(test_acc)

            
            v_acc_record.append(valid_accs.list)       
            test_acc_record.append(test_accs.list)
            fold_aucs.append(valid_auc)
            fold_record_valid_metrics.append(valid_met)
            fold_record_matched_test_metrics.append(test_met)
            test_fold_aucs.append(test_auc)
            
            fold_record_matched_train_metrics.append(train_met)
            train_fold_aucs.append(train_auc)
            continue  
        
        model = model.to(DEVICE)
        n_iter = 0
        single_loss_writer = tensorboardX.SummaryWriter(opt.output_dir+'/loss_fold_'+str(fold))
        #pps="_k-fold-sub-fold-"+str(fold)+"_"
        #model = load_curr_best_checkpoint(model,out_dir = opt.output_dir,model_name = opt.model,pps = pps)
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
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)


        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=custom_lr_scheduler)

        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose = 1, patience=5, factor = 0.5) 
        # torch.optim.lr_scheduler.steplr
        
        early_stopping = EarlyStopping(patience = opt.patience, verbose=True)
        history = {'train_loss': [], 'valid_loss': [],'test_loss':[],'train_acc':[],'valid_acc':[], 'test_acc':[],'train_auc':[],'test_auc':[],'valid_auc':[],'lr':[]}


        
        


        for epoch in range(opt.epochs):

            train_loss, train_losses, train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3]= weighted_train(train_loader = train_loader
                                                , model = model
                                                , criterion = criterion
                                                , optimizer = optimizer
                                                , epoch = best_epoch
                                                ,labels = labels, multi_M= multi_M
                                                )

            train_loss_box.append(train_loss.detach().cpu())
            train_acc_box.append(train_acc)
            t_acc_record.append(train_accs.list)
            #================================== every epoch's metrics record =================================================
            """
            train_loss, train_losses, train_acc, train_accs,train_met =validate(valid_loader = train_loader
                                        , model = model
                                        , criterion = criterion)
            """
            train_loss, train_losses, validating_train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3],train_mets= weighted_validate(valid_loader = train_loader, model = model, criterion = criterion, 
                                                                                                                                                              labels = labels, multi_M= multi_M)
            train_a,train_b,train_c = train_met.a, train_met.b, train_met.c
            train_pre, train_rec, train_F1, train_spe = train_met.pre,train_met.sen,train_met.F1, train_met.spe
            print("[Fold-%d | Epoch : %d]: training_train_acc: "%(fold,epoch),train_acc, " ; validating_acc: ",validating_train_acc)
            
            valid_loss, valid_losses, valid_acc, valid_accs, valid_met,[valid_loss_0,valid_loss_1,valid_loss_2,valid_loss_3],valid_mets= weighted_validate(valid_loader = valid_loader, model = model, criterion = criterion,
                                                                                                                                                    labels = labels, multi_M= multi_M)
            valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c
            valid_pre, valid_rec, valid_F1, valid_spe = valid_met.pre,valid_met.sen, valid_met.F1, valid_met.spe
            valid_met_0,valid_met_1,valid_met_2 = valid_mets[0],valid_mets[1],valid_mets[2]

            test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3],test_mets= weighted_validate(valid_loader = test_loader, model = model, criterion = criterion, 
                                                                                                                                          labels = labels, multi_M= multi_M)
            test_a,test_b,test_c = test_met.a, test_met.b, test_met.c
            test_pre,test_rec, test_F1,test_spe = test_met.pre,test_met.sen, test_met.F1, test_met.spe
            test_met_0,test_met_1,test_met_2 = test_mets[0],test_mets[1],test_mets[2]
            
            if opt.num_classes==3:
                valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu() ,multi_class = 'ovo',labels=[0,1,2])
                test_auc = roc_auc_score(test_a.cpu(),test_b.cpu() ,multi_class = 'ovo',labels=[0,1,2])
                train_auc = roc_auc_score(train_a.cpu(),train_b.cpu() ,multi_class = 'ovo',labels=[0,1,2])
            else:
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

            sum_write_Mark=opt.sum_write_Mark

            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"loss", {'train':train_loss,'valid':valid_loss,'test':test_loss},epoch)
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"acc", {'train':train_acc,'valid':valid_acc,'test':test_acc},epoch)

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
                best_scale = []
                

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
            fold_best_scale = best_scale

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
        train_loss, train_losses, train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3],train_mets =weighted_validate(valid_loader = train_loader
                                    , model = model
                                    , criterion = criterion
                                    , labels = labels
                                    , multi_M= multi_M)
        train_a,train_b,train_c = train_met.a, train_met.b, train_met.c

        valid_loss, valid_losses, valid_acc, valid_accs, valid_met,[valid_loss_0,valid_loss_1,valid_loss_2,valid_loss_3],valid_mets= weighted_validate(valid_loader = valid_loader
                                    , model = model
                                    , criterion = criterion
                                    , labels = labels
                                    , multi_M= multi_M)
        valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c

        test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3] ,test_mets = weighted_validate(valid_loader = test_loader
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

        fpr1, tpr1, thersholds = roc_curve(test_a.cpu(), test_b.cpu()[:,1])#用1的预测  
        fold_test_tpr.append(tpr1)
        fold_test_fpr.append(fpr1)
         
        recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_epoch",record_file = None,ps_thing=[best_fold,fold,fold_best_auc,best_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets, labels = labels)
        recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_epoch",record_file = record_file,ps_thing=[best_fold,fold,fold_best_auc,best_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets, labels = labels)
        print("best_scale: ==========================================",file=record_file)
        print(best_scale,file=record_file)
        print("best_scale: ==========================================")
        print(best_scale)

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
        
        draw_heatmap(train_met.CM,valid_met.CM, test_met.CM,title="fold_"+str(fold)+"_",save_path=opt.pic_output_dir+'_k-fold_'+str(fold))


        foldperf['fold{}'.format(fold+1)] = history 
        
    print("fold_test_fpr: ",fold_test_fpr)
    print("fold_test_tpr: ",fold_test_tpr)
    print("test_fold_aucs: ",test_fold_aucs)
    testauc_record_file.write("\nfold_test_fpr:\n")
    testauc_record_file.write(str(fold_test_fpr))
    testauc_record_file.write("\nfold_test_tpr:\n")
    testauc_record_file.write(str(fold_test_tpr))
    testauc_record_file.write("\ntest_fold_aucs:\n")
    testauc_record_file.write(str(test_fold_aucs))
    
  
    if TEST==True:
        record_avg(fold_record_valid_metrics,fold_record_matched_test_metrics,record_file,fold_aucs,test_fold_aucs,fold_record_matched_train_metrics,train_fold_aucs)
        torch.cuda.empty_cache()
        sum_writer.close()
        record_file.close()
        return
    os.system('echo " === TRAIN mae mtc:{:.5f}" >> {}'.format(train_loss, output_path))

    # to get averaged metrics of k folds
    record_avg(fold_record_valid_metrics,fold_record_matched_test_metrics,record_file,fold_aucs,test_fold_aucs,fold_record_matched_train_metrics,train_fold_aucs)

    # to print & record the best result of total
    model.load_state_dict(fold_best_statedict)        
    train_loss, train_losses, train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3],train_mets =weighted_validate(valid_loader = train_loader
                                , model = model
                                , criterion = criterion
                                , labels = labels
                                , multi_M= multi_M)
    train_a,train_b,train_c = train_met.a, train_met.b, train_met.c

    valid_loss, valid_losses, valid_acc, valid_accs, valid_met,[valid_loss_0,valid_loss_1,valid_loss_2,valid_loss_3],valid_mets= weighted_validate(valid_loader = valid_loader
                                , model = model
                                , criterion = criterion
                                , labels = labels
                                , multi_M= multi_M)
    valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c

    test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3] ,test_mets= weighted_validate(valid_loader = test_loader
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

    recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_fold",record_file = None,ps_thing=[best_fold,best_fold,fold_best_auc,fold_best_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets, labels = labels)
    recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_fold",record_file = record_file,ps_thing=[best_fold,best_fold,fold_best_auc,fold_best_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets, labels = labels)
    print("fold_best_scale: ==========================================",file=record_file)
    print(fold_best_scale,file=record_file)
    print("fold_best_scale: ==========================================")
    print(fold_best_scale)
    draw_heatmap(train_met.CM,valid_met.CM, test_met.CM,title="fold_"+str(fold)+"_",save_path=opt.pic_output_dir+'_total_'+str(fold))



    torch.cuda.empty_cache()
    sum_writer.close()
    record_file.close()
    testauc_record_file.close()
    print("=======training end========")



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
