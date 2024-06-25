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
from utils2.config_3class import opt
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

from load_data_3class import DIY_Folder
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

from utils2.self_KL_L0 import Self_L0 
from utils2.self_KL_L1 import Self_L1 
from utils2.self_KL_L2 import Self_L2 
from utils2.self_KL_L3 import Self_L3 
from utils2.weighting.min_norm_solvers import MinNormSolver, gradient_normalizers
from utils2.FLoss import Focal_Loss
import seaborn as sns
from utils2.batchaverage import BatchCriterion
<<<<<<< HEAD

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_device(2)
=======
import timm

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_device(1)
>>>>>>> 3a4a4f2 (20240625-code)
    DEVICE=torch.device('cuda')
else:
    DEVICE=torch.device('cpu')
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
print('DEVICE: ',DEVICE)



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



def train(train_loader, model, criterion, optimizer, epoch,tasks=['L0','L2','L3'],loss_fn = {'L0':Self_L0(),'L2':Self_L2(),'L3':Self_L3()},use_MGDA=True,n_iter=0,writer = None):
    
    losses = AverageMeter()
    maes = AverageMeter()
    Loss_0 = AverageMeter()
    Loss_1 = AverageMeter()
    Loss_2 = AverageMeter()
    Loss_3 = AverageMeter()


    for i, (img,_,target, saffix, sradiomics) in enumerate(train_loader):
        n_iter+=1
        target = torch.from_numpy(np.expand_dims(target,axis=1))


        input_img = img.to(DEVICE)
        input_affix = saffix.to(DEVICE)
        input_radiomics = sradiomics.to(DEVICE)
        
        #print("input_affix: ",input_affix)
        

        # convert the input's shape: [8,91,109,91] → [8,1,91,109,91] ,to match the input of model
        #input = torch.reshape(input, [input.shape[0],1,input.shape[1],input.shape[2],input.shape[3]])
        #convert target's shape: [[1],[0]] → [1,0]
        target = convert(target)
        

        print("target: ",target)

        model.train()
        model.zero_grad()

        #out = model(input,input_affix)


        loss_data = {}
        grads = {}
        scale = {}
        """
        if use_MGDA:
            optimizer.zero_grad()
            # First compute representations (z)
            images_volatile = Variable(input, volatile=True)
            input_affix_vola = Variable(input_affix, volatile=True)
            rep = model(images_volatile, input_affix_vola)
            print("rep: ",rep)
            print("rep.grad: ",rep.grad)
            print("rep.requires_grad:",rep.requires_grad)
            # As an approximate solution we only need gradients for input
            #if isinstance(rep, list):
                # This is a hack to handle psp-net
                #rep = rep[0]
                #rep_variable = [Variable(rep.clone(), requires_grad=True)]
                #list_rep = True
            #else:
                #rep_variable = Variable(rep.clone(), requires_grad=True)
                #list_rep = False
            list_rep = False
            # Compute gradients of each loss function wrt z
            print("rep: ",rep)
            print("rep.grad: ",rep.grad)
            print("rep.requires_grad:",rep.requires_grad)
           
            c2 = nn.CrossEntropyLoss()
            print("CE loss: ",c2(rep, target))
            for t in tasks:
                out_t = rep
                print("out: ",out_t)
                print("out.grad: ",out_t.grad)
                print("out.requires_grad:",out_t.requires_grad)
                
                loss = loss_fn[t](out_t, target)
                print(t," ; val: ",loss)
                loss_data[t] = loss
                loss.backward()
                grads[t] = []
                if list_rep:
                    rep = rep
                    #grads[t].append(Variable(rep_variable[0].grad.data.clone(), requires_grad=False))
                    #rep_variable[0].grad.data.zero_()
                else:
                    print("rep: ",rep)
                    print("rep.grad: ",rep.grad)
                    print("rep.requires_grad:",rep.requires_grad)
                    grads[t].append(Variable(rep.grad.data.clone(), requires_grad=False))
                    rep.grad.data.zero_()

            # Normalize all gradients, this is optional and not included in the paper.
            gn = gradient_normalizers(grads, loss_data, 'loss+')
            for t in tasks:
                for gr_i in range(len(grads[t])):
                    grads[t][gr_i] = grads[t][gr_i] / gn[t]

            # Frank-Wolfe iteration to compute scales.
            sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
            for i, t in enumerate(tasks):
                scale[t] = float(sol[i])
        """
        if use_MGDA:
            for t in tasks:
                # Comptue gradients of each loss function wrt parameters
                optimizer.zero_grad()
                if(opt.lossfunc=="BatchAvg"):
                    out_t,feat = model(input_img,input_affix,input_radiomics)
                else:
                    out_t = model(input_img,input_affix,input_radiomics)
                    
                if(t=="BatchAvg"):
                    loss = loss_fn[t](feat, target)
                else:
                    loss = loss_fn[t](out_t, target)
                
                loss.backward()
                loss_data[t] = loss
                
                grads[t] = []
                for param in model.parameters():
                    if param.grad is not None:
                        #print("param.grad is not none")
                        grads[t].append(Variable(param.grad.data.clone(), requires_grad=False))
            rep= out_t
            #print("grads: ",grads['L1'])
            print("out_t: ",out_t)
            print("loss_data: ",loss_data)
            # Normalize all gradients, this is optional and not included in the paper.
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
        
        #input.size(0) = batch_size 
        # the CE's ouput is averaged, so 
        loss=0.

                
        if(opt.lossfunc == 'SelfKL' or opt.lossfunc == 'Weighted_SelfKL'):
                loss,loss_0 ,loss_1,loss_2,loss_3 = criterion(rep, target)
        elif(opt.lossfunc=="BatchAvg"):
            spare_criterion = nn.CrossEntropyLoss()
            spare_selfKL_criterion = SelfKL(num_classes=3,lambda_0=1.0,lambda_1=1.0,lambda_2=1.0,lambda_3=1.0, CE_or_KL=opt.CE_or_KL) 
            loss_a,loss_0 ,loss_1,loss_2,loss_3 = spare_selfKL_criterion(rep,target)
            loss_b = criterion(feat, target)
            loss = loss_a+loss_b
            
        else:
            loss= criterion(rep, target)
            loss_0 ,loss_1,loss_2,loss_3 =0.,0.,0.,0.

        losses.update(loss*input_img.size(0),input_img.size(0))
        Loss_0.update(loss_0*input_img.size(0),input_img.size(0))
        Loss_1.update(loss_1*input_img.size(0),input_img.size(0))
        Loss_2.update(loss_2*input_img.size(0),input_img.size(0))
        Loss_3.update(loss_3*input_img.size(0),input_img.size(0))


        pred, mae = get_corrects(output=rep, target=target)

        maes.update(mae, input_img.size(0))
        CM=confusion_matrix(target.cpu(), pred.cpu(),labels=[0,1,2])
        acc,met_0,met_1,met_2  = cal_metrics(CM)
        pre = (met_0.pre+met_1.pre+met_2.pre)/3
        sen = (met_0.sen+met_1.sen+met_2.sen)/3
        spe = (met_0.spe+met_1.spe+met_2.spe)/3
        F1 = (met_0.F1+met_1.F1+met_2.F1)/3 

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
        
        if(opt.lossfunc=="BatchAvg"):
            rep,feat = model(input_img,input_affix,input_radiomics)
        else:
            rep = model(input_img,input_affix,input_radiomics)

        for i, t in enumerate(tasks):
            if(t=="BatchAvg"):
                loss_t = loss_fn[t](feat, target)
            else:
                loss_t = loss_fn[t](rep, target)
            loss_data[t] = loss_t
            if i > 0:
                loss = loss + scale[t]*loss_t
                print("New scale and new loss val: ",scale[t], "; loss_"+t, " ;val: ",loss_t)
            else:
                loss = scale[t]*loss_t
                print("New scale and new loss val: ",scale[t], "; loss_"+t, " ;val: ",loss_t)
        print("New total loss: ",loss)
                
        writer.add_scalar('training_loss', loss, n_iter)
        for t in tasks:
            writer.add_scalar('training_loss_{}'.format(t), loss_data[t], n_iter)
            writer.add_scalar("scale_{}".format(t),scale[t],n_iter)

        print("loss:",loss," ; loss.grad: ",loss.grad,"; output is leaf? ",rep.is_leaf," ; out.grad:",rep.grad," ;scale: ",scale)
        loss.backward()
        optimizer.step()
    #total metrics'calcultion
    
    acc,met_0,met_1,met_2  = cal_metrics(CM)
    met= Metrics()
    avg_pre = (met_0.pre+met_1.pre+met_2.pre)/3
    avg_sen = (met_0.sen+met_1.sen+met_2.sen)/3
    avg_spe = (met_0.spe+met_1.spe+met_2.spe)/3
    avg_F1 = (met_0.F1+met_1.F1+met_2.F1)/3 
    
    met= Metrics()
    met.update(a=0,b=0,c=0, auc=0, acc=acc,sen=avg_sen,pre=avg_pre,F1=avg_F1,spe=avg_spe,CM=CM)

    return losses.avg,losses, maes.avg, maes, met,[Loss_0.avg,Loss_1.avg,Loss_2.avg,Loss_3.avg],n_iter,scale
    

def masked_constrained_train(train_loader, model, criterion, optimizer, epoch,tasks=['a','b'],loss_fn = {'a':nn.CrossEntropyLoss(),'b':nn.CrossEntropyLoss()},use_MGDA=True,n_iter=0,writer = None):
    
    Losses = AverageMeter()
    Loss_0 = AverageMeter()
    Loss_1 = AverageMeter()
    Loss_2 = AverageMeter()
    Loss_3 = AverageMeter()
    maes = AverageMeter()

    for i, (img,masked_img,_,target, saffix,sradiomics) in enumerate(train_loader):
        n_iter+=1
        target = torch.from_numpy(np.expand_dims(target,axis=1))
        input_img = img.to(DEVICE)
        masked_img = masked_img.to(DEVICE)
        input_affix = saffix.to(DEVICE)
        input_radiomics=sradiomics.to(DEVICE)
        #print("Intrainning, img.size: ",img.size)
        # convert the input_img's shape: [8,91,109,91] → [8,1,91,109,91] ,to match the input of model
        #input = torch.reshape(input, [input.shape[0],1,input.shape[1],input.shape[2],input.shape[3]])
        #convert target's shape: [[1],[0]] → [1,0]
        target = convert(target)
        print("target: ",target)
        model.train()
        
        loss_data = {}
        grads = {}
        scale = {}
        if use_MGDA:
                # Comptue gradients of each loss function wrt parameters
            optimizer.zero_grad()
            if(opt.lossfunc=="BatchAvg"):
                out_1,feat = model(input_img,input_affix,input_radiomics)
                out_2,masked_feat = model(masked_img,input_affix,input_radiomics)
            else:
                out_1 = model(input_img,input_affix,input_radiomics)
                out_2 = model(masked_img,input_affix,input_radiomics)  
                  
            print("This_out_1: ",out_1," ;out_2: ",out_2)  
            if(opt.lossfunc == 'SelfKL' or opt.lossfunc == 'Weighted_SelfKL'):
                loss_1,_ ,_,_,_ = criterion(out_1, target)
                loss_2,_ ,_,_,_ = criterion(out_2, target)
            elif(opt.lossfunc=="BatchAvg"):
                spare_criterion= nn.CrossEntropyLoss()
                loss_1 = criterion(feat, target) + spare_criterion(out_1, target)
                loss_2 = criterion(masked_feat, target) + spare_criterion(out_2, target)    
            else:
                loss_1 = loss_fn['a'](out_1, target)
                loss_2 = loss_fn['a'](out_2, target)
                 
            loss_a = loss_1+loss_2
            print("loss_1: ",loss_1," ;loss_2:",loss_2)
            loss_a.backward()
            loss_data['a'] = loss_a
            
            grads['a'] = []
            for param in model.parameters():
                if param.grad is not None:
                    #print("param.grad is not none")
                    grads['a'].append(Variable(param.grad.data.clone(), requires_grad=False))
            
                # Comptue gradients of each loss function wrt parameters
            optimizer.zero_grad()
            if(opt.lossfunc=="BatchAvg"):
                out_1,feat = model(input_img,input_affix,input_radiomics)
                out_2,masked_feat = model(masked_img,input_affix,input_radiomics)
            else:
                out_1 = model(input_img,input_affix,input_radiomics)
                out_2 = model(masked_img,input_affix,input_radiomics)   
                
            if(opt.lossfunc == 'SelfKL' or opt.lossfunc == 'Weighted_SelfKL' ):
                loss_3,_ ,_,_,_ = criterion(out_2, F.softmax(out_1,dim=1), False)        
            elif(opt.lossfunc == 'Weighted_CE'or opt.lossfunc == 'FLoss'  ):
                loss_3 = criterion(out_2, F.softmax(out_1,dim=1), False)          
            else:
                loss_3 = loss_fn['b'](out_2, F.softmax(out_1,dim=1))
                
            print("loss_3: ",loss_3)
            loss_b = loss_3
            loss_b.backward()
            loss_data['b'] = loss_b
            print("That_out_1: ",out_1," ;out_2: ",out_2)
            grads['b'] = []
            for param in model.parameters():
                if param.grad is not None:
                    #print("param.grad is not none")
                    grads['b'].append(Variable(param.grad.data.clone(), requires_grad=False))
                                               
            #print("grads: ",grads['L1'])
            print("out_1: ",out_1," ;out_2: ",out_2)
            print("loss_data: ",loss_data)
            # Normalize all gradients, this is optional and not included in the paper.
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
                
        model.zero_grad()     
        if(opt.lossfunc=="BatchAvg"):
            out_1,feat = model(input_img,input_affix,input_radiomics)
            out_2,masked_feat = model(masked_img,input_affix,input_radiomics)
        else:
            out_1 = model(input_img,input_affix,input_radiomics)
            out_2 = model(masked_img,input_affix,input_radiomics)  

        if(opt.lossfunc == 'SelfKL' or opt.lossfunc == 'Weighted_SelfKL' ):
            loss_1,_ ,_,_,_ = criterion(out_1, target)
            loss_2,_ ,_,_,_ = criterion(out_2, target)
            loss_3,_ ,_,_,_ = criterion(out_2, F.softmax(out_1,dim=1), False)        
        elif(opt.lossfunc=="BatchAvg"):
            spare_criterion= nn.CrossEntropyLoss()
            loss_1 = criterion(feat, target) + spare_criterion(out_1, target)
            loss_2 = criterion(masked_feat, target) + spare_criterion(out_2, target)   
            loss_3 = loss_fn['b'](out_2, F.softmax(out_1,dim=1))    
        elif(opt.lossfunc == 'Weighted_CE' or opt.lossfunc == 'FLoss' ):
                loss_1= criterion(out_1, target)
                loss_2= criterion(out_2, target)
                loss_b= criterion(out_2, F.softmax(out_1,dim=1),False)       
        else:
            loss_1 = loss_fn['a'](out_1, target)
            loss_2 = loss_fn['a'](out_2, target)
            loss_3 = loss_fn['b'](out_2, F.softmax(out_1,dim=1))
                

        loss_data['a'] = loss_1+loss_2
        loss_data['b'] = loss_3   
        loss = scale['a']*loss_data['a'] + scale['b']*loss_data['b'] 
        for i, t in enumerate(tasks):
            print("New scale and new loss val: ",scale[t], "; loss_"+t, " ;val: ",loss_data[t])
        print("New total loss: ",loss)        

        #input.size(0) = batch_size 
        # the CE's ouput is averaged, so 
        Losses.update(loss*input_img.size(0),input_img.size(0)) 
        Loss_0.update(0.,input_img.size(0))
        Loss_1.update(loss_1*input_img.size(0),input_img.size(0))
        Loss_2.update(loss_2*input_img.size(0),input_img.size(0))
        Loss_3.update(loss_3*input_img.size(0),input_img.size(0))

        pred, mae = get_corrects(output=out_1, target=target)

        maes.update(mae, out_1.size(0))
        
        CM=confusion_matrix(target.cpu(), pred.cpu(),labels=[0,1,2])
        
        acc,met_0,met_1,met_2 = cal_metrics(CM)
        pre = (met_0.pre+met_1.pre+met_2.pre)/3
        sen = (met_0.sen+met_1.sen+met_2.sen)/3
        spe = (met_0.spe+met_1.spe+met_2.spe)/3
        F1 = (met_0.F1+met_1.F1+met_2.F1)/3 

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
        writer.add_scalar('training_loss', loss, n_iter)
        for t in tasks:
            writer.add_scalar('training_loss_{}'.format(t), loss_data[t], n_iter)
            writer.add_scalar("scale_{}".format(t),scale[t],n_iter)
        loss.backward()
        optimizer.step()
        
    # total metrics'calcultion
    acc,met_0,met_1,met_2  = cal_metrics(CM)
    avg_pre = (met_0.pre+met_1.pre+met_2.pre)/3
    avg_sen = (met_0.sen+met_1.sen+met_2.sen)/3
    avg_spe = (met_0.spe+met_1.spe+met_2.spe)/3
    avg_F1 = (met_0.F1+met_1.F1+met_2.F1)/3 
    
    met= Metrics()
    met.update(a=0,b=0,c=0, auc=0, acc=acc,sen=avg_sen,pre=avg_pre,F1=avg_F1,spe=avg_spe,CM=CM)


    return Losses.avg,Losses, maes.avg, maes, met,[Loss_0.avg,Loss_1.avg,Loss_2.avg,Loss_3.avg],n_iter,scale
    


def validate(valid_loader, model, criterion,tasks=['L0','L2','L3'],loss_fn = None,n_iter=0,writer = None, ps='valid'):
    
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
        
        for i, (img,_,target, saffix,sradiomics) in enumerate(valid_loader):
            
            target = torch.from_numpy(np.expand_dims(target,axis=1))
            target = convert(target)


            input_img = img.to(DEVICE)
            input_affix = saffix.to(DEVICE)
            input_radiomics = sradiomics.to(DEVICE)
            input_target = target.to(DEVICE)
            #print("input_affix: ",input_affix)
            #input_img = torch.reshape(input_img, [input.shape[0],1,input.shape[1],input.shape[2],input.shape[3]])
            
            if(opt.lossfunc=="BatchAvg"):
                out,feat = model(input_img,input_affix,input_radiomics)

            else:
                out = model(input_img,input_affix,input_radiomics)
            
            if(opt.lossfunc == 'SelfKL' or opt.lossfunc == 'Weighted_SelfKL'):
                loss,loss_0 ,loss_1,loss_2,loss_3 = criterion(out, target)
            elif(opt.lossfunc=="BatchAvg"):
                spare_criterion = nn.CrossEntropyLoss()
                spare_selfKL_criterion = SelfKL(num_classes=3,lambda_0=1.0,lambda_1=1.0,lambda_2=1.0,lambda_3=1.0, CE_or_KL=opt.CE_or_KL) 
                loss_a,loss_0 ,loss_1,loss_2,loss_3 = spare_selfKL_criterion(out,target)
                loss_b = criterion(feat, target)
                loss = loss_a+loss_b
            else:
                loss= criterion(out, target)
                loss_0 ,loss_1,loss_2,loss_3 =0.,0.,0.,0.

            num_val_batches+=1
            num_val_items+=input_img.size(0)
            if not writer==None:
                for t in tasks:
                    loss_t = loss_fn[t](out, target)
                    tot_loss['all'] += loss_t
                    tot_loss[t] += loss_t
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
 
            CM+=confusion_matrix(target.cpu(), pred.cpu(),labels=[0,1,2])
        
        a=torch.tensor(total_target)
        b=torch.tensor(total_out)
        c=torch.tensor(total_pred)    
        # total metrics'calcultion
        print("a",a)
        print("b",b)
        print("c",c)
        auc = roc_auc_score(a.cpu(),b.cpu(),multi_class = 'ovo',labels=[0,1,2])
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
        auc = roc_auc_score(a.cpu(),b.cpu(),multi_class = 'ovo',labels=[0,1,2])
        
        avg_pre = (met_0.pre+met_1.pre+met_2.pre)/3
        avg_sen = (met_0.sen+met_1.sen+met_2.sen)/3
        avg_spe = (met_0.spe+met_1.spe+met_2.spe)/3
        avg_F1 = (met_0.F1+met_1.F1+met_2.F1)/3     
        print('Confusion Matirx : ',CM,'[Metrics]-Accuracy(mean): ' , acc,'- Sensitivity : ',avg_sen*100,'- Specificity : ',avg_spe*100,'- Precision: ',avg_pre*100,'- F1 : ',avg_F1*100,'- auc : ',auc*100)

        # Metrics is a DIY_package to store all metrics(acc, auc, F1,pre, recall, spe,CM, outpur, pred,target)
        met= Metrics()

        met.update(a=a,b=b,c=c, acc=acc,sen=recall,pre=pre,F1=new_F1,spe=avg_spe,auc=auc,CM=CM)

        mets=[met_0,met_1,met_2]
        
        if not writer==None:
            for t in tasks:
                writer.add_scalar(ps+'_loss_{}'.format(t), tot_loss[t]/num_val_batches, n_iter)
            writer.add_scalar(ps+'_loss', tot_loss['all']/num_val_items, n_iter)
              
        return losses.avg,losses, maes.avg, maes, met,[Loss_0.avg,Loss_1.avg,Loss_2.avg,Loss_3.avg],mets
    
def masked_constrained_validate(valid_loader, model, criterion,tasks=['a','b'],loss_fn =None,n_iter=0,writer = None, ps='valid'):
    
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
    tot_loss = {}
    tot_loss['all'] = 0.0
    for t in tasks:
            tot_loss[t] = 0.0
    num_val_batches=0
    num_val_items = 0
    model.eval() #because if allow model.eval, the bn wouldn't work, and the train_set'result would be different(model.train & model.eval)

    with torch.no_grad():
        for i, (img,masked_img,_,target, saffix,sradiomics) in enumerate(valid_loader):
            target = torch.from_numpy(np.expand_dims(target,axis=1))
            target = convert(target)

            input_img = img.to(DEVICE)
            input_affix = saffix.to(DEVICE)
            masked_img = masked_img.to(DEVICE)
            input_radiomics=sradiomics.to(DEVICE)
            #input = torch.reshape(input, [input.shape[0],1,input.shape[1],input.shape[2],input.shape[3]])

            num_val_batches+=1
            num_val_items+=input_img.size(0)

            if(opt.lossfunc=="BatchAvg"):
                img_out,feat = model(input_img,input_affix,input_radiomics)
                masked_img_out,masked_feat = model(masked_img,input_affix,input_radiomics)
            else:
                img_out = model(input_img,input_affix,input_radiomics)
                masked_img_out = model(masked_img,input_affix,input_radiomics)  

            if(opt.lossfunc == 'SelfKL' or opt.lossfunc == 'Weighted_SelfKL'):
                 loss_1,_ ,_,_,_ = criterion(img_out, target)
                 loss_2,_ ,_,_,_ = criterion(masked_img_out, target)
                 loss_b,_ ,_,_,_ = criterion(masked_img_out, F.softmax(img_out,dim=1),False)
            elif(opt.lossfunc == 'Weighted_CE' or opt.lossfunc == 'FLoss' ):
                 loss_1= criterion(img_out, target)
                 loss_2= criterion(masked_img_out, target)
                 loss_b= criterion(masked_img_out, F.softmax(img_out,dim=1),False)
            elif(opt.lossfunc=="BatchAvg"):
                spare_criterion= nn.CrossEntropyLoss()
                loss_1 = criterion(feat, target) + spare_criterion(img_out, target)
                loss_2 = criterion(masked_feat, target) + spare_criterion(masked_img_out, target)   
                loss_b = spare_criterion(masked_img_out, F.softmax(img_out,dim=1)) 
            else:        
                loss_1 = criterion(img_out, target) 
                loss_2 = criterion(masked_img_out, target)
                loss_b = criterion(masked_img_out, F.softmax(img_out,dim=1)) 
            loss_a = loss_1+loss_2                   
            loss = loss_a+loss_b
                
            Losses.update(loss*input_img.size(0),input_img.size(0)) 
            Loss_0.update(loss*input_img.size(0),input_img.size(0))
            Loss_1.update(loss_1*input_img.size(0),input_img.size(0))
            Loss_2.update(loss_2*input_img.size(0),input_img.size(0))
            Loss_3.update(loss_b*input_img.size(0),input_img.size(0))

            pred, mae = get_corrects(output=img_out, target=target)
            maes.update(mae, input_img.size(0))

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
        auc = roc_auc_score(a.cpu(),b.cpu(),multi_class = 'ovo',labels=[0,1,2])
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
        auc = roc_auc_score(a.cpu(),b.cpu(),multi_class = 'ovo',labels=[0,1,2])
        
        avg_pre = (met_0.pre+met_1.pre+met_2.pre)/3
        avg_sen = (met_0.sen+met_1.sen+met_2.sen )/3
        avg_spe = (met_0.spe+met_1.spe+met_2.spe)/3
        avg_F1 = (met_0.F1+met_1.F1+met_2.F1)/3     
        print('Confusion Matirx : ',CM,'[Metrics]-Accuracy(mean): ' , acc,'- Sensitivity : ',avg_sen*100,'- Specificity : ',avg_spe*100,'- Precision: ',avg_pre*100,'- F1 : ',avg_F1*100,'- auc : ',auc*100)

        # Metrics is a DIY_package to store all metrics(acc, auc, F1,pre, recall, spe,CM, outpur, pred,target)
        met= Metrics() 
        met.update(a=a,b=b,c=c, acc=acc,sen=recall,pre=pre,F1=avg_F1,spe=avg_spe,auc=auc,CM=CM)
       
        mets = [met_0,met_1,met_2]

        if not writer==None:
            for t in tasks:
                writer.add_scalar(ps+'_loss_{}'.format(t), tot_loss[t]/num_val_batches, n_iter)
            writer.add_scalar(ps+'_loss', tot_loss['all']/num_val_items, n_iter)
            
        return Losses.avg,Losses, maes.avg, maes, met,[Loss_0.avg,Loss_1.avg,Loss_2.avg,Loss_3.avg],mets

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

def get_weights_for_balanced_classes(dataset,nclasses):
    count=[0]*nclasses
    for i, item in enumerate(dataset):
        if opt.train_form =="masked_constrained_train":
            (data,auged_data,sid,target,saffix,sradiomics) = item
        else:
            (img,_,target, saffix,sradiomics) = item
        print("i: target: ",i,";",target)
        count[target]+=1
    print("count: ",count)
    w_per_class = [0.]*nclasses
    N=float(sum(count))
    for i in range(nclasses):
        w_per_class[i] = N/float(count[i])
    weight = [0]*len(dataset)
    if opt.train_form =="masked_constrained_train":
        for idx, (data,auged_data,sid,target,saffix,sradiomics)  in enumerate(dataset):
            weight[idx]=w_per_class[target]
                
    else:
        for idx, (img,_,target, saffix,sradiomics)  in enumerate(dataset):
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
    
def main(output_path):

    begin_time = time.time()
    json_path = os.path.join(opt.output_dir, 'hyperparameter.json')
    with open(json_path,'w') as jsf:
        jsf.write(json.dumps(vars(opt)
                                , indent=4
                                , separators=(',',':')))
    # record metrics into a txt
    record_file_path= opt.output_dir+opt.model+'_'+opt.lossfunc+opt.ps+'_record.txt'
    record_file = open(record_file_path, "a")
    
    data_path=opt.data_path
    test_data_path = opt.testdata_path

    print("=========== start train the brain age estimation model =========== \n")
    print(" ==========> Using {} processes for data loader.".format(opt.num_workers))

    #load the training_data  and test_data (training_data will be splited later for cross_validation)
    total_file = DIY_Folder(data_path=data_path,train_form = opt.train_form,root_mask_radiomics = opt.root_bbx_path,use_radiomics=opt.use_radiomics,istest=False,sec_pair_orig=opt.sec_pair_orig,multiclass = opt.multiclass)
    print("len(total_file): ",len(total_file))
    print("total_file.gety(): ",total_file.gety())
    #root_radiomics="",root_mask_radiomics="",use_radiomics=True, norm_radiomics=True
    test_file = DIY_Folder(data_path=test_data_path,train_form = opt.train_form,root_mask_radiomics = opt.test_root_bbx_path,use_radiomics=opt.use_radiomics,istest=True,sec_pair_orig= opt.sec_pair_orig,multiclass = opt.multiclass)
    test_data = test_file.select_dataset(data_idx=[i for i in range(len(test_file))], aug=False,use_secondimg=opt.usesecond,noSameRM=opt.noSameRM, usethird = opt.usethird)
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
    begin_time = time.time()

    epochs = opt.epochs
    input_channel = 2 if opt.usesecond else 1
    num_classes = 3 if opt.multiclass else 2
    target_names =  ['class 0', 'class 1','class 2']
    loss_func_dict = { 'CE' : nn.CrossEntropyLoss().to(DEVICE)
                      , 'Weighted_CE' : Weighted_CE(classes_weight=opt.loss_weight,n_classes=num_classes)
                     , 'SelfKL' :  SelfKL(num_classes=num_classes,lambda_0=1.0,lambda_1=1.0,lambda_2=1.0,lambda_3=1.0, CE_or_KL=opt.CE_or_KL)                    
                     , 'Weighted_SelfKL': SelfKL(num_classes=num_classes,lambda_0=1.0,lambda_1=1.0,lambda_2=1.0,lambda_3=1.0, CE_or_KL=opt.CE_or_KL,classes_weight=opt.loss_weight)
                     , 'FLoss':Focal_Loss(num_classes=num_classes,gamma=opt.FL_gamma)
                     , 'BatchAvg': BatchCriterion(num_classes=num_classes,DEVICE=DEVICE)
                     }
    criterion = loss_func_dict[opt.lossfunc]
    feature_align = feature_align = True if 'BatchAvg' in opt.lossfunc or opt.lossfunc=='CE' else False
    print("feature_align: ",feature_align)
    
    train_func_dict={"None": train
                    ,"masked_constrained_train": masked_constrained_train
                    }
    training = train_func_dict[opt.train_form]
    valid_func_dict={"None": validate
                ,"masked_constrained_train": masked_constrained_validate
                }
    validating = valid_func_dict[opt.train_form]
    
    if(opt.lossfunc == 'SelfKL'):
        tasks=['L0','L2','L3']
        loss_fn = {'L0':Self_L0(num_classes=num_classes),'L2':Self_L2(num_classes=num_classes),'L3':Self_L3(num_classes=num_classes)}
    if( opt.lossfunc == 'Weighted_SelfKL'):
        tasks=['L0','L2','L3']
        loss_fn = {'L0':Self_L0(num_classes=num_classes,classes_weight=opt.loss_weight),'L2':Self_L2(num_classes=num_classes,classes_weight=opt.loss_weight),'L3':Self_L3(num_classes=num_classes,classes_weight=opt.loss_weight)}
    if( opt.lossfunc == 'BatchAvg'):
        tasks=['BatchAvg','L0','L2','L3']
        loss_fn = {'BatchAvg':criterion,'L0':Self_L0(num_classes=num_classes),'L2':Self_L2(num_classes=num_classes),'L3':Self_L3(num_classes=num_classes)}
  
    if opt.train_form == "masked_constrained_train":
        tasks=['a','b']
        loss_fn = {'a':nn.CrossEntropyLoss(),'b':nn.CrossEntropyLoss()}
        if opt.lossfunc == "CE":
            loss_fn = {'a':nn.CrossEntropyLoss(),'b':nn.CrossEntropyLoss()}
        elif opt.lossfunc == "BatchAvg":
            loss_fn ={'a': criterion,'b': nn.CrossEntropyLoss()}
        else:
            loss_fn ={'a': criterion ,'b': criterion}


    sum_writer = tensorboardX.SummaryWriter(opt.output_dir)

    
    print(" ==========> All settled. Training is getting started...")
    print(" ==========> Training takes {} epochs.".format(epochs))
    print(" ==========> output_dir: ",opt.output_dir)
    print(" ==========> task: ",num_classes)

    # split the training_data into K fold with StratifiedKFold(shuffle = True)
    k=5
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

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(total_file)),total_file.gety())):
        if fold<opt.continue_train_fold :
            continue
        print("begin to train fold: ",fold)
            
 
        begin_training_time = time.time()-begin_time
        print("[....Training time....]: %dh, %dmin, %ds "%(int(begin_training_time/3600),int(begin_training_time/60),int(begin_training_time%60)))
        print("[....Training time....]: %dh, %dmin, %ds "%(int(begin_training_time/3600),int(begin_training_time/60),int(begin_training_time%60)),file=record_file)
        begin_time = time.time()

        print("\n============ FOLD %d ============\n"%(fold),file=record_file)
        print('Fold {}'.format(fold))
        


        #valid_data(no augmentation: aug=False) & train_data(need augmentation:aug = True)
        print("Get valid set: ",len(val_idx))
        vali_data=total_file.select_dataset(data_idx=val_idx, aug=False,use_secondimg=opt.usesecond,noSameRM=opt.noSameRM, usethird=opt.usethird)
        print("vali_data: ",vali_data)
        print("Got train set: ",len(train_idx))
        train_data=total_file.select_dataset(data_idx=train_idx, aug=True,aug_form=opt.aug_form,use_secondimg=opt.usesecond,noSameRM=opt.noSameRM, usethird=opt.usethird)
        print("train_data: ",len(train_data),";",train_data)
        

        if opt.use_radiomics:
            radio_mean, radio_std = train_data.calc_own_radiomics_mean()
            vali_data.inject_other_mean(radio_mean, radio_std)
            test_data.inject_other_mean(radio_mean, radio_std)
            
            print("train_data. radio_mean: ",train_data.radio_mean)
            
        weights = get_weights_for_balanced_classes(train_data,num_classes)
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

        if opt.model == "tencent_resnet10":
            model = tencent_resnet.resnet10(sample_input_W=75,
            sample_input_H=80,
            sample_input_D=75,
            shortcut_type='B',
            no_cuda=False,
            num_seg_classes=num_classes,
            input_channel=input_channel,
            feature_align=feature_align
            )
            model =  model.to(DEVICE)
            net_dict = model.state_dict()
            checkpoint = torch.load(opt.tencent_pth_rootdir + "resnet_10_23dataset.pth")     
            print("pretrained model exists? ",os.path.exists(opt.tencent_pth_rootdir + "resnet_10_23dataset.pth")) 
            print("net_dict.keys(): ",net_dict.keys())
            pretrain_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in net_dict.keys()}
            net_dict.update(pretrain_dict)
            model.load_state_dict(net_dict)
            print("check_point_path: ",opt.tencent_pth_rootdir + "resnet_10_23dataset.pth")
        elif opt.model == "tencent_resnet18":
            model = tencent_resnet.resnet18(sample_input_W=75,
            sample_input_H=80,
            sample_input_D=75,
            shortcut_type='B',
            no_cuda=False,
            num_seg_classes=num_classes,
            input_channel=input_channel,
            feature_align=feature_align
            )
            model =  model.to(DEVICE)
            net_dict = model.state_dict()
            checkpoint = torch.load(opt.tencent_pth_rootdir + "resnet_18_23dataset.pth")     
            print("pretrained model exists? ",os.path.exists(opt.tencent_pth_rootdir + "resnet_18_23dataset.pth")) 
            print("net_dict.keys(): ",net_dict.keys())
            pretrain_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in net_dict.keys()}
            net_dict.update(pretrain_dict)
            model.load_state_dict(net_dict)
            print("check_point_path: ",opt.tencent_pth_rootdir + "resnet_18_23dataset.pth")
            
        elif opt.model == "tencent_resnet34":
            model = tencent_resnet.resnet34(sample_input_W=75,
            sample_input_H=80,
            sample_input_D=75,
            shortcut_type='B',
            no_cuda=False,
            num_seg_classes=num_classes,
            input_channel=input_channel,
            feature_align=feature_align
            )
            model =  model.to(DEVICE)
            net_dict = model.state_dict()
            checkpoint = torch.load(opt.tencent_pth_rootdir + "resnet_34_23dataset.pth")     
            print("pretrained model exists? ",os.path.exists(opt.tencent_pth_rootdir + "resnet_34_23dataset.pth")) 
            print("net_dict.keys(): ",net_dict.keys())
            pretrain_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in net_dict.keys()}
            net_dict.update(pretrain_dict)
            model.load_state_dict(net_dict)
            print("check_point_path: ",opt.tencent_pth_rootdir + "resnet_34_23dataset.pth")
 
        if opt.model == "ResNet18":
            model = ResNet18(num_classes=num_classes,input_channel=input_channel,use_radiomics=opt.use_radiomics,feature_align=feature_align)
            orig_path = "/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/new_bbx_composed_sampler_ResNet18/model_result/"
            #orig_path = "/home/chenxr/Pineal_region/after_12_08/Results/0402_T1C/multiclass/no_MGDA/none_composed_sampler_ResNet18/model_result/"
            pps="_k-fold-sub-fold-"+str(fold)+"_"
            best_model_path = orig_path+"ResNet18"+'_'+"CE"+opt.ps+pps+'_best_model.pth.tar'
            checkpoint = torch.load(best_model_path)
            
        if opt.model == "ResNet34":
            model = ResNet34(num_classes=num_classes,input_channel=input_channel,use_radiomics=opt.use_radiomics,feature_align=feature_align)
        elif opt.model == "ResNet10":
            model = ResNet10(num_classes=num_classes,input_channel=input_channel,use_radiomics=opt.use_radiomics,feature_align=feature_align)
        elif opt.model == "ResNet24":
            model = ResNet24(num_classes=num_classes,input_channel=input_channel,use_radiomics=opt.use_radiomics,feature_align=feature_align)
        elif opt.model == "ResNet30":
            model = ResNet30(num_classes=num_classes,input_channel=input_channel,use_radiomics=opt.use_radiomics,feature_align=feature_align)
        elif opt.model == "DIY_ResNet18":
            model = DIY_ResNet18(num_classes=num_classes,input_channel=input_channel)
        elif opt.model == "DIY_ResNet10":
            model = DIY_ResNet10(num_classes=num_classes,input_channel=input_channel)
        else:
            print("[ERROR: ] Wrong model chosen\n")
        """            
        elif opt.model == "VGG11_bn":
            model = VGG11_bn()
        elif opt.model == "VGG13_bn":
            model = VGG13_bn()
        elif opt.model == "Inception2":
            model = Inception2()
        elif opt.model == "VGG16_bn":
            model = VGG16_bn()
        elif opt.model == "seresnet18":
            model = seresnet18()
        """
        
        model = model.to(DEVICE)
        n_iter = 0
        single_loss_writer = tensorboardX.SummaryWriter(opt.output_dir+'/loss_fold_'+str(fold))


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
<<<<<<< HEAD
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
=======
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
        scheduler = timm.scheduler.cosineLRscheduler(optimizer=optimizer,#调度器会通过这个引用修改优化器里的l#待训练总的epoch数量
                                                        t_initial=epoch,
                                                        lr_min = 1e-5,#余弦退火学习率衰减后最低的lr
                                                        warmup_t=20,
                                                        warmup_lr_init=1e-8, #学习率预热阶段的lr起始值
                                                        decay_rate=0.5
                                                        )
>>>>>>> 3a4a4f2 (20240625-code)


        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose = 1, patience=5, factor = 0.5) 
        # torch.optim.lr_scheduler.steplr
        
        early_stopping = EarlyStopping(patience = 50, verbose=True)
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

            train_loss, train_losses, train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3],n_iter,scale= training(train_loader = train_loader
                                                , model = model
                                                , criterion = criterion
                                                , optimizer = optimizer
                                                , epoch = epoch
                                                ,  tasks = tasks
                                                , loss_fn = loss_fn
                                                , n_iter = n_iter
                                                , writer = single_loss_writer)

            train_loss_box.append(train_loss.detach().cpu())
            train_acc_box.append(train_acc)
            t_acc_record.append(train_accs.list)
            #================================== every epoch's metrics record =================================================
            """
            train_loss, train_losses, train_acc, train_accs,train_met =validate(valid_loader = train_loader
                                        , model = model
                                        , criterion = criterion)
            """
            train_loss, train_losses, validating_train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3],train_mets= validating(valid_loader = train_loader, model = model, criterion = criterion
                                                                                                                                                    , loss_fn = loss_fn
                                                                                                                                                    , n_iter = n_iter
                                                                                                                                                    , writer = single_loss_writer
                                                                                                                                                    , ps='train')
            train_a,train_b,train_c = train_met.a, train_met.b, train_met.c
            train_pre, train_rec, train_F1, train_spe = train_met.pre,train_met.sen,train_met.F1, train_met.spe
            print("[Each Epoch]: training_train_acc: ",train_acc, " ; validating_acc: ",validating_train_acc)
            
            valid_loss, valid_losses, valid_acc, valid_accs, valid_met,[valid_loss_0,valid_loss_1,valid_loss_2,valid_loss_3],valid_mets= validating(valid_loader = valid_loader, model = model, criterion = criterion
                                                                                                                                                    , loss_fn = loss_fn
                                                                                                                                                    , n_iter = n_iter
                                                                                                                                                    , writer = single_loss_writer
                                                                                                                                                    , ps='valid')
            valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c
            valid_pre, valid_rec, valid_F1, valid_spe = valid_met.pre,valid_met.sen, valid_met.F1, valid_met.spe
            valid_met_0,valid_met_1,valid_met_2 = valid_mets[0],valid_mets[1],valid_mets[2]

            test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3],test_mets= validating(valid_loader = test_loader, model = model, criterion = criterion                                                                                                                                  , loss_fn = loss_fn
                                                                                                                                                   
                                                                                                                                                    , n_iter = n_iter
                                                                                                                                                    , writer = single_loss_writer
                                                                                                                                                    , ps='test')
            test_a,test_b,test_c = test_met.a, test_met.b, test_met.c
            test_pre,test_rec, test_F1,test_spe = test_met.pre,test_met.sen, test_met.F1, test_met.spe
            test_met_0,test_met_1,test_met_2 = test_mets[0],test_mets[1],test_mets[2]
            
            valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu() ,multi_class = 'ovo',labels=[0,1,2])
            test_auc = roc_auc_score(test_a.cpu(),test_b.cpu() ,multi_class = 'ovo',labels=[0,1,2])
            train_auc = roc_auc_score(train_a.cpu(),train_b.cpu() ,multi_class = 'ovo',labels=[0,1,2])


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
                best_scale = scale
                

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
            early_stopping(valid_loss)
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
        train_loss, train_losses, train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3],train_mets =validating(valid_loader = train_loader
                                    , model = model
                                    , criterion = criterion)
        train_a,train_b,train_c = train_met.a, train_met.b, train_met.c

        valid_loss, valid_losses, valid_acc, valid_accs, valid_met,[valid_loss_0,valid_loss_1,valid_loss_2,valid_loss_3],valid_mets= validating(valid_loader = valid_loader
                                    , model = model
                                    , criterion = criterion)
        valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c

        test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3] ,test_mets = validating(valid_loader = test_loader
                                    , model = model
                                    , criterion = criterion)
        test_a,test_b,test_c = test_met.a, test_met.b, test_met.c

        train_auc = roc_auc_score(train_a.cpu(),train_b.cpu() ,multi_class = 'ovo',labels=[0,1,2])
        valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu(),multi_class = 'ovo',labels=[0,1,2])
        test_auc = roc_auc_score(test_a.cpu(),test_b.cpu(),multi_class = 'ovo',labels=[0,1,2])

        recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_epoch",record_file = None,ps_thing=[best_fold,fold,fold_best_auc,best_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets)
        recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_epoch",record_file = record_file,ps_thing=[best_fold,fold,fold_best_auc,best_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets)

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
  

    os.system('echo " === TRAIN mae mtc:{:.5f}" >> {}'.format(train_loss, output_path))

    # to get averaged metrics of k folds
    record_avg(fold_record_valid_metrics,fold_record_matched_test_metrics,record_file,fold_aucs,test_fold_aucs,fold_record_matched_train_metrics,train_fold_aucs)

    # to print & record the best result of total
    model.load_state_dict(fold_best_statedict)        
    train_loss, train_losses, train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3],train_mets =validating(valid_loader = train_loader
                                , model = model
                                , criterion = criterion)
    train_a,train_b,train_c = train_met.a, train_met.b, train_met.c

    valid_loss, valid_losses, valid_acc, valid_accs, valid_met,[valid_loss_0,valid_loss_1,valid_loss_2,valid_loss_3],valid_mets= validating(valid_loader = valid_loader
                                , model = model
                                , criterion = criterion)
    valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c

    test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3] ,test_mets= validating(valid_loader = test_loader
                                , model = model
                                , criterion = criterion)
    test_a,test_b,test_c = test_met.a, test_met.b, test_met.c

    train_auc = roc_auc_score(train_a.cpu(),train_b.cpu() ,multi_class = 'ovo',labels=[0,1,2])
    valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu() ,multi_class = 'ovo',labels=[0,1,2])
    test_auc = roc_auc_score(test_a.cpu(),test_b.cpu() ,multi_class = 'ovo',labels=[0,1,2])

    recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_fold",record_file = None,ps_thing=[best_fold,best_fold,fold_best_auc,fold_best_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets)
    recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_fold",record_file = record_file,ps_thing=[best_fold,best_fold,fold_best_auc,fold_best_auc,best_epoch],target_names=target_names,train_mets=train_mets,valid_mets=valid_mets,test_mets=test_mets)
    print("fold_best_scale: ==========================================",file=record_file)
    print(fold_best_scale,file=record_file)
    print("fold_best_scale: ==========================================")
    print(fold_best_scale)
    
    draw_heatmap(train_met.CM,valid_met.CM, test_met.CM,title="fold_"+str(fold)+"_",save_path=opt.pic_output_dir+'_total_'+str(fold))

    torch.cuda.empty_cache()
    sum_writer.close()
    record_file.close()
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
        print("[train_avg_pre]: %f \t[valid_avg_pre]: %f \t[test_avg_pre]: %f \t"%((train_mets[0].pre+train_mets[1].pre+train_mets[2].pre)/3,(valid_mets[0].pre+valid_mets[1].pre+valid_mets[2].pre)/3,(test_mets[0].pre+test_mets[1].pre+test_mets[2].pre)/3))
 
        print("[train]: <recall_0>: %f \t<recall_1>: %f \t<recall_2>: %f \n"%(train_mets[0].sen,train_mets[1].sen,train_mets[2].sen))
        print("[valid]: <recall_0>: %f \t<recall_1>: %f \t<recall_2>: %f \n"%(valid_mets[0].sen,valid_mets[1].sen,valid_mets[2].sen))
        print("[test] : <recall_0>: %f \t<recall_1>: %f \t<recall_2>: %f \n"%(test_mets[0].sen,test_mets[1].sen,test_mets[2].sen))
        print("[train_avg_recall]: %f \t[valid_avg_recall]: %f \t[test_avg_recall]: %f \t"%((train_mets[0].sen+train_mets[1].sen+train_mets[2].sen)/3,(valid_mets[0].sen+valid_mets[1].sen+valid_mets[2].sen)/3,(test_mets[0].sen+test_mets[1].sen+test_mets[2].sen)/3))
 
        print("[train]: <F1_0>: %f \t<F1_1>: %f \t<F1_2>: %f \n"%(train_mets[0].F1,train_mets[1].F1,train_mets[2].F1))
        print("[valid]: <F1_0>: %f \t<F1_1>: %f \t<F1_2>: %f \n"%(valid_mets[0].F1,valid_mets[1].F1,valid_mets[2].F1))
        print("[test] : <F1_0>: %f \t<F1_1>: %f \t<F1_2>: %f \n"%(test_mets[0].F1,test_mets[1].F1,test_mets[2].F1))


        print("[train_avg_F1]: %f \t[valid_avg_F1]: %f \t[test_avg_F1]: %f \t"%((train_mets[0].F1+train_mets[1].F1+train_mets[2].F1)/3,(valid_mets[0].F1+valid_mets[1].F1+valid_mets[2].F1)/3,(test_mets[0].F1+test_mets[1].F1+test_mets[2].F1)/3))

        print("[train]: <spe_0>: %f \t<spe_1>: %f \t<spe_2>: %f \n"%(train_mets[0].spe,train_mets[1].spe,train_mets[2].spe))     
        print("[valid]: <spe_0>: %f \t<spe_1>: %f \t<spe_2>: %f \n"%(valid_mets[0].spe,valid_mets[1].spe,valid_mets[2].spe))
        print("[test] : <spe_0>: %f \t<spe_1>: %f \t<spe_0>: %f \n"%(test_mets[0].spe,test_mets[1].spe,test_mets[2].spe))
        print("[train_avg_spe]: %f \t[valid_avg_spe]: %f \t[test_avg_spe]: %f \t"%((train_mets[0].spe+train_mets[1].spe+train_mets[2].spe)/3,(valid_mets[0].spe+valid_mets[1].spe+valid_mets[2].spe)/3,(test_mets[0].spe+test_mets[1].spe+test_mets[2].spe)/3))
   
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
        print("[train_avg_pre]: %f \t[valid_avg_pre]: %f \t[test_avg_pre]: %f \n"%((train_mets[0].pre+train_mets[1].pre+train_mets[2].pre)/3,(valid_mets[0].pre+valid_mets[1].pre+valid_mets[2].pre)/3,(test_mets[0].pre+test_mets[1].pre+test_mets[2].pre)/3),file=record_file)
 
        print("[train]: <recall_0>: %f \t<recall_1>: %f \t<recall_2>: %f \n"%(train_mets[0].sen,train_mets[1].sen,train_mets[2].sen),file=record_file)
        print("[valid]: <recall_0>: %f \t<recall_1>: %f \t<recall_2>: %f \n"%(valid_mets[0].sen,valid_mets[1].sen,valid_mets[2].sen),file=record_file)
        print("[test] : <recall_0>: %f \t<recall_1>: %f \t<recall_2>: %f \n"%(test_mets[0].sen,test_mets[1].sen,test_mets[2].sen),file=record_file)
        print("[train_avg_recall]: %f \t[valid_avg_recall]: %f \t[test_avg_recall]: %f \n"%((train_mets[0].sen+train_mets[1].sen+train_mets[2].sen)/3,(valid_mets[0].sen+valid_mets[1].sen+valid_mets[2].sen)/3,(test_mets[0].sen+test_mets[1].sen+test_mets[2].sen)/3),file=record_file)
 
        print("[train]: <F1_0>: %f \t<F1_1>: %f \t<F1_2>: %f \n"%(train_mets[0].F1,train_mets[1].F1,train_mets[2].F1),file=record_file)
        print("[valid]: <F1_0>: %f \t<F1_1>: %f \t<F1_2>: %f \n"%(valid_mets[0].F1,valid_mets[1].F1,valid_mets[2].F1),file=record_file)
        print("[test] : <F1_0>: %f \t<F1_1>: %f \t<F1_2>: %f \n"%(test_mets[0].F1,test_mets[1].F1,test_mets[2].F1),file=record_file)


        print("[train_avg_F1]: %f \t[valid_avg_F1]: %f \t[test_avg_F1]: %f \n"%((train_mets[0].F1+train_mets[1].F1+train_mets[2].F1)/3,(valid_mets[0].F1+valid_mets[1].F1+valid_mets[2].F1)/3,(test_mets[0].F1+test_mets[1].F1+test_mets[2].F1)/3),file=record_file)

        print("[train]: <spe_0>: %f \t<spe_1>: %f \t<spe_2>: %f \n"%(train_mets[0].spe,train_mets[1].spe,train_mets[2].spe),file=record_file)     
        print("[valid]: <spe_0>: %f \t<spe_1>: %f \t<spe_2>: %f \n"%(valid_mets[0].spe,valid_mets[1].spe,valid_mets[2].spe),file=record_file)
        print("[test] : <spe_0>: %f \t<spe_1>: %f \t<spe_0>: %f \n"%(test_mets[0].spe,test_mets[1].spe,test_mets[2].spe),file=record_file)
        print("[train_avg_spe]: %f \t[valid_avg_spe]: %f \t[test_avg_spe]: %f \n"%((train_mets[0].spe+train_mets[1].spe+train_mets[2].spe)/3,(valid_mets[0].spe+valid_mets[1].spe+valid_mets[2].spe)/3,(test_mets[0].spe+test_mets[1].spe+test_mets[2].spe)/3),file=record_file)
   
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
