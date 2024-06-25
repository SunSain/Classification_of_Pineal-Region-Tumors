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

from utils2.self_KL_L0 import Self_L0 
from utils2.self_KL_L1 import Self_L1 
from utils2.self_KL_L2 import Self_L2 
from utils2.self_KL_L3 import Self_L3 

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

from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from statistics import mean
import time
import math

from utils2.norm_radiomics_dataset import  Unite_norm_radiomics
from utils2.weighting import MGDA
from utils2.weighting.abstract_arch import AbsArchitecture
from utils2.weighting.min_norm_solvers import MinNormSolver, gradient_normalizers
from utils2.self_KL_L0 import Self_L0 
from utils2.self_KL_L1 import Self_L1 
from utils2.self_KL_L2 import Self_L2 
from utils2.self_KL_L3 import Self_L3 

target_names = ['class 0', 'class 1']


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
    checkpoint_path = out_dir+model_name+'_'+opt.lossfunc+opt.ps+pps+"_fold_epoch_"+str(fold)+"_"+str(epoch)+'_checkpoint.pth.tar'
    best_model_path = out_dir+model_name+'_'+opt.lossfunc+opt.ps+pps+'_best_model.pth.tar'

    print("checkpoint_path: ",checkpoint_path)
    torch.save(state, checkpoint_path)
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


        input = Variable(img.to(DEVICE))
        input_affix = Variable(saffix.to(DEVICE))
        input_radiomics = sradiomics.to(DEVICE)
        
        #print("input_affix: ",input_affix)
        

        # convert the input's shape: [8,91,109,91] → [8,1,91,109,91] ,to match the input of model
        #input = torch.reshape(input, [input.shape[0],1,input.shape[1],input.shape[2],input.shape[3]])
        #convert target's shape: [[1],[0]] → [1,0]
        target = convert(target)
        
        input_target = target.to(DEVICE)
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
                out_t = model(input,input_affix)
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
        if(opt.lossfunc == 'SelfKL'):
                loss,loss_0 ,loss_1,loss_2,loss_3 = criterion(rep, target)
        else:
            loss= criterion(rep, target)
            loss_0 ,loss_1,loss_2,loss_3 =0.,0.,0.,0.
        
        
        losses.update(loss*input.size(0),input.size(0))
        Loss_0.update(loss_0*input.size(0),input.size(0))
        Loss_1.update(loss_1*input.size(0),input.size(0))
        Loss_2.update(loss_2*input.size(0),input.size(0))
        Loss_3.update(loss_3*input.size(0),input.size(0))


        pred, mae = get_corrects(output=rep, target=target)

        maes.update(mae, input.size(0))
        CM=confusion_matrix(target.cpu(), pred.cpu(),labels=[0,1])
        acc2, sen, pre, F1, spe = cal_metrics(CM)

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
                  , loss=losses, acc=maes, acc2=acc2, sen=sen, pre=pre, F1=F1, spe=spe )
                )
            print("CM: ",CM)
            
            
        optimizer.zero_grad()
        rep= model(input, input_affix)
        for i, t in enumerate(tasks):
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
    acc2, sen, pre, F1, spe = cal_metrics(CM)
    met= Metrics()
    met.update(a=0,b=0,c=0, auc=0, acc=acc2,sen=sen,pre=pre,F1=F1,spe=spe,CM=CM)

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
        img = img.to(DEVICE)
        masked_img = masked_img.to(DEVICE)
        input_affix = saffix.to(DEVICE)
        #print("Intrainning, img.size: ",img.size)
        # convert the input's shape: [8,91,109,91] → [8,1,91,109,91] ,to match the input of model
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
            out_1 = model(img,input_affix)
            out_2 = model(masked_img,input_affix)    
            print("This_out_1: ",out_1," ;out_2: ",out_2)  
            if(opt.lossfunc == 'SelfKL'):
                loss_1,_ ,_,_,_ = criterion(out_1, target)
                loss_2,_ ,_,_,_ = criterion(out_2, target)
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
            out_1 = model(img,input_affix)
            out_2 = model(masked_img,input_affix)
            if(opt.lossfunc == 'SelfKL'):
                loss_3,_ ,_,_,_ = criterion(out_2, F.softmax(out_1,dim=1), False)
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
        out_1 = model(img,input_affix)
        out_2 = model(masked_img,input_affix) 
        
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
        Losses.update(loss*img.size(0),img.size(0)) 
        Loss_0.update(0.,img.size(0))
        Loss_1.update(loss_1*img.size(0),img.size(0))
        Loss_2.update(loss_2*img.size(0),img.size(0))
        Loss_3.update(loss_3*img.size(0),img.size(0))

        pred, mae = get_corrects(output=out_1, target=target)

        maes.update(mae, out_1.size(0))
        CM=confusion_matrix(target.cpu(), pred.cpu(),labels=[0,1])
        
        acc2, sen, pre, F1, spe = cal_metrics(CM)

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
                  , loss=Losses,loss_0=Loss_0, loss_1=Loss_1,loss_2=Loss_2, acc=maes, acc2=acc2, sen=sen, pre=pre, F1=F1, spe=spe )
                )
            print("CM: ",CM)


                
        writer.add_scalar('training_loss', loss, n_iter)
        for t in tasks:
            writer.add_scalar('training_loss_{}'.format(t), loss_data[t], n_iter)
            writer.add_scalar("scale_{}".format(t),scale[t],n_iter)
            
        loss.backward()
        optimizer.step()
    # total metrics'calcultion
    acc2, sen, pre, F1, spe = cal_metrics(CM)
    met= Metrics()
    met.update(a=0,b=0,c=0, auc=0, acc=acc2,sen=sen,pre=pre,F1=F1,spe=spe,CM=CM)

    return Losses.avg,Losses, maes.avg, maes, met,[Loss_0.avg,Loss_1.avg,Loss_2.avg,Loss_3.avg],n_iter,scale
    

def validate(valid_loader, model, criterion,tasks=['L0','L2','L3'],loss_fn = {'L0':Self_L0(),'L2':Self_L2(),'L3':Self_L3()},use_MGDA=True,n_iter=0,writer = None, ps='valid'):
    
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


            input = img.to(DEVICE)
            input_affix = saffix.to(DEVICE)
            input_radiomics = sradiomics.to(DEVICE)
            input_target = target.to(DEVICE)
            #print("input_affix: ",input_affix)
            #input = torch.reshape(input, [input.shape[0],1,input.shape[1],input.shape[2],input.shape[3]])
            
            out = model(input,input_affix)
            
            if(opt.lossfunc == 'SelfKL'):
                 loss,loss_0 ,loss_1,loss_2,loss_3 = criterion(out, target)
            else:
                loss= criterion(out, target)
                loss_0 ,loss_1,loss_2,loss_3 =0.,0.,0.,0.

            num_val_batches+=1
            num_val_items+=input.size(0)
            for t in tasks:
                loss_t = loss_fn[t](out, target)
                tot_loss['all'] += loss_t
                tot_loss[t] += loss_t
            #input.size(0) = batch_size 
            # the CE's ouput is averaged, so 
            losses.update(loss*input.size(0),input.size(0))
            Loss_0.update(loss_0*input.size(0),input.size(0))
            Loss_1.update(loss_1*input.size(0),input.size(0))
            Loss_2.update(loss_2*input.size(0),input.size(0))
            Loss_3.update(loss_3*input.size(0),input.size(0))

            pred, mae = get_corrects(output=out, target=target)
            maes.update(mae, input.size(0))

            # collect every output/pred/target, combine them together for total metrics'calculation
            total_target.extend(target)
            total_out.extend(torch.softmax(out,dim=1).cpu().numpy())
            total_pred.extend(pred.cpu().numpy())
 
            CM+=confusion_matrix(target.cpu(), pred.cpu(),labels=[0,1])
            
        a=torch.tensor(total_target)
        b=torch.tensor(total_out)
        c=torch.tensor(total_pred)    
        # total metrics'calcultion
    
        acc, sen, pre, F1, spe = cal_metrics(CM)
        print("normal_valid : a: ",a)
        print("normal_valid :b: ",b)
        auc = roc_auc_score(a.cpu(),b.cpu()[:,1])
        print('Confusion Matirx : ',CM,'[Metrics]-Accuracy(mean): ' , acc,'- Sensitivity : ',sen*100,'- Specificity : ',spe*100,'- Precision: ',pre*100,'- F1 : ',F1*100,'- auc : ',auc*100)

        # Metrics is a DIY_package to store all metrics(acc, auc, F1,pre, recall, spe,CM, outpur, pred,target)
        met= Metrics()
        met.update(a=a,b=b,c=c, acc=acc,sen=sen,pre=pre,F1=F1,spe=spe,auc=auc,CM=CM)
        
        for t in tasks:
            writer.add_scalar(ps+'_loss_{}'.format(t), tot_loss[t]/num_val_batches, n_iter)
        writer.add_scalar(ps+'_loss', tot_loss['all']/num_val_items, n_iter)
        
        return losses.avg,losses, maes.avg, maes, met,[Loss_0.avg,Loss_1.avg,Loss_2.avg,Loss_3.avg]
    
def masked_constrained_validate(valid_loader, model, criterion,tasks=['a','b'],loss_fn = {'a':nn.CrossEntropyLoss(),'b':nn.CrossEntropyLoss()},use_MGDA=True,n_iter=0,writer = None, ps='valid'):
    
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

            img = img.to(DEVICE)
            input_affix = saffix.to(DEVICE)
            masked_img = masked_img.to(DEVICE)
            #input = torch.reshape(input, [input.shape[0],1,input.shape[1],input.shape[2],input.shape[3]])
            
            img_out = model(img,input_affix)
            masked_img_out = model(masked_img,input_affix)
            
            num_val_batches+=1
            num_val_items+=img.size(0)
            

            if(opt.lossfunc == 'SelfKL'):
                 loss_1,_ ,_,_,_ = criterion(img_out, target)
                 loss_2,_ ,_,_,_ = criterion(masked_img_out, target)
                 loss_b,_ ,_,_,_ = criterion(masked_img_out, F.softmax(img_out,dim=1),False)
            else:        
                loss_1 = criterion(img_out, target) 
                loss_2 = criterion(masked_img_out, target)
                loss_b = criterion(masked_img_out, F.softmax(img_out,dim=1),False) 
            loss_a = loss_1+loss_2                   
            loss = loss_a+loss_b
                
            Losses.update(loss*img.size(0),img.size(0)) 
            Loss_0.update(loss*img.size(0),img.size(0))
            Loss_1.update(loss_1*img.size(0),img.size(0))
            Loss_2.update(loss_2*img.size(0),img.size(0))
            Loss_3.update(loss_b*img.size(0),img.size(0))

            pred, mae = get_corrects(output=img_out, target=target)
            maes.update(mae, img.size(0))

            # collect every output/pred/target, combine them together for total metrics'calculation
            total_target.extend(target)
            print("vlidate: target: ",target)
            total_out.extend(torch.softmax(img_out,dim=1).cpu().numpy())
            total_pred.extend(pred.cpu().numpy())
 
            CM+=confusion_matrix(target.cpu(), pred.cpu(),labels=[0,1])
            
        a=torch.tensor(total_target)
        b=torch.tensor(total_out)
        c=torch.tensor(total_pred)    
        # total metrics'calcultion
    
        acc, sen, pre, F1, spe = cal_metrics(CM)
        print("masked_valid : a: ",a)
        print("masked_valid :b: ",b)
        auc = roc_auc_score(a.cpu(),b.cpu()[:,1],labels=[0,1])
        print('Confusion Matirx : ',CM,'[Metrics]-Accuracy(mean): ' , acc,'- Sensitivity : ',sen*100,'- Specificity : ',spe*100,'- Precision: ',pre*100,'- F1 : ',F1*100,'- auc : ',auc*100)

        # Metrics is a DIY_package to store all metrics(acc, auc, F1,pre, recall, spe,CM, outpur, pred,target)
        met= Metrics()
        met.update(a=a,b=b,c=c, acc=acc,sen=sen,pre=pre,F1=F1,spe=spe,auc=auc,CM=CM)
        for t in tasks:
            writer.add_scalar(ps+'_loss_{}'.format(t), tot_loss[t]/num_val_batches, n_iter)
        writer.add_scalar(ps+'_loss', tot_loss['all']/num_val_items, n_iter)
        
        return Losses.avg,Losses, maes.avg, maes, met,[Loss_0.avg,Loss_1.avg,Loss_2.avg,Loss_3.avg]


def normal_validate(valid_loader, model, criterion):
    
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

    model.eval() #because if allow model.eval, the bn wouldn't work, and the train_set'result would be different(model.train & model.eval)

    with torch.no_grad():
        
        for i, (img,_,target, saffix,sradiomics) in enumerate(valid_loader):
            
            target = torch.from_numpy(np.expand_dims(target,axis=1))
            target = convert(target)


            input = img.to(DEVICE)
            input_affix = saffix.to(DEVICE)
            input_radiomics = sradiomics.to(DEVICE)
            input_target = target.to(DEVICE)
            #print("input_affix: ",input_affix)
            #input = torch.reshape(input, [input.shape[0],1,input.shape[1],input.shape[2],input.shape[3]])
            
            out = model(input,input_affix)
            
            if(opt.lossfunc == 'SelfKL'):
                 loss,loss_0 ,loss_1,loss_2,loss_3 = criterion(out, target)
            else:
                loss= criterion(out, target)
                loss_0 ,loss_1,loss_2,loss_3 =0.,0.,0.,0.


            #input.size(0) = batch_size 
            # the CE's ouput is averaged, so 
            losses.update(loss*input.size(0),input.size(0))
            Loss_0.update(loss_0*input.size(0),input.size(0))
            Loss_1.update(loss_1*input.size(0),input.size(0))
            Loss_2.update(loss_2*input.size(0),input.size(0))
            Loss_3.update(loss_3*input.size(0),input.size(0))

            pred, mae = get_corrects(output=out, target=target)
            maes.update(mae, input.size(0))

            # collect every output/pred/target, combine them together for total metrics'calculation
            total_target.extend(target)
            total_out.extend(torch.softmax(out,dim=1).cpu().numpy())
            total_pred.extend(pred.cpu().numpy())
 
            CM+=confusion_matrix(target.cpu(), pred.cpu(),labels=[0,1])
            
        a=torch.tensor(total_target)
        b=torch.tensor(total_out)
        c=torch.tensor(total_pred)    
        # total metrics'calcultion
    
        acc, sen, pre, F1, spe = cal_metrics(CM)
        auc = roc_auc_score(a.cpu(),b.cpu()[:,1])
        print('Confusion Matirx : ',CM,'[Metrics]-Accuracy(mean): ' , acc,'- Sensitivity : ',sen*100,'- Specificity : ',spe*100,'- Precision: ',pre*100,'- F1 : ',F1*100,'- auc : ',auc*100)

        # Metrics is a DIY_package to store all metrics(acc, auc, F1,pre, recall, spe,CM, outpur, pred,target)
        met= Metrics()
        met.update(a=a,b=b,c=c, acc=acc,sen=sen,pre=pre,F1=F1,spe=spe,auc=auc,CM=CM)
        
        
        return losses.avg,losses, maes.avg, maes, met,[Loss_0.avg,Loss_1.avg,Loss_2.avg,Loss_3.avg]
    
def normal_masked_constrained_validate(valid_loader, model, criterion,tasks=['a','b'],loss_fn = {'a':nn.CrossEntropyLoss(),'b':nn.CrossEntropyLoss()},use_MGDA=True,n_iter=0,writer = None, ps='valid'):
    
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
        for i, (img,masked_img,_,target, saffix,sradiomics) in enumerate(valid_loader):
            target = torch.from_numpy(np.expand_dims(target,axis=1))
            target = convert(target)
<<<<<<< HEAD


=======
>>>>>>> 3a4a4f2 (20240625-code)
            img = img.to(DEVICE)
            input_affix = saffix.to(DEVICE)
            masked_img = masked_img.to(DEVICE)
            #input = torch.reshape(input, [input.shape[0],1,input.shape[1],input.shape[2],input.shape[3]])
            
            img_out = model(img,input_affix)
            masked_img_out = model(masked_img,input_affix)
            
            if(opt.lossfunc == 'SelfKL'):
                 loss_1,_ ,_,_,_ = criterion(img_out, target)
                 loss_2,_ ,_,_,_ = criterion(masked_img_out, target)
                 loss_b,_ ,_,_,_ = criterion(masked_img_out, F.softmax(img_out,dim=1),False)
            else:        
                loss_1 = criterion(img_out, target) 
                loss_2 = criterion(masked_img_out, target)
                loss_b = criterion(masked_img_out, F.softmax(img_out,dim=1),False) 
            loss_a = loss_1+loss_2                   
            loss = loss_a+loss_b
                
            Losses.update(loss*img.size(0),img.size(0)) 
            Loss_0.update(loss*img.size(0),img.size(0))
            Loss_1.update(loss_1*img.size(0),img.size(0))
            Loss_2.update(loss_2*img.size(0),img.size(0))
            Loss_3.update(loss_b*img.size(0),img.size(0))



            pred, mae = get_corrects(output=img_out, target=target)
            maes.update(mae, img.size(0))

            # collect every output/pred/target, combine them together for total metrics'calculation
            total_target.extend(target)
            print("vlidate: target: ",target)
            total_out.extend(torch.softmax(img_out,dim=1).cpu().numpy())
            total_pred.extend(pred.cpu().numpy())
 
            CM+=confusion_matrix(target.cpu(), pred.cpu(),labels=[0,1])
            
        a=torch.tensor(total_target)
        b=torch.tensor(total_out)
        c=torch.tensor(total_pred)    
        # total metrics'calcultion
    
        acc, sen, pre, F1, spe = cal_metrics(CM)
        auc = roc_auc_score(a.cpu(),b.cpu()[:,1],labels=[0,1])
        print('Confusion Matirx : ',CM,'[Metrics]-Accuracy(mean): ' , acc,'- Sensitivity : ',sen*100,'- Specificity : ',spe*100,'- Precision: ',pre*100,'- F1 : ',F1*100,'- auc : ',auc*100)

        # Metrics is a DIY_package to store all metrics(acc, auc, F1,pre, recall, spe,CM, outpur, pred,target)
        met= Metrics()
        met.update(a=a,b=b,c=c, acc=acc,sen=sen,pre=pre,F1=F1,spe=spe,auc=auc,CM=CM)

        
        return Losses.avg,Losses, maes.avg, maes, met,[Loss_0.avg,Loss_1.avg,Loss_2.avg,Loss_3.avg]


def cal_metrics(CM):
    tn=CM[0][0]
    tp=CM[1][1]
    fp=CM[0][1]
    fn=CM[1][0]
    acc=np.sum(np.diag(CM)/np.sum(CM))
    sen=tp/(tp+fn)
    pre=tp/(tp+fp)
    F1= (2*sen*pre)/(sen+pre)
    spe = tn/(tn+fp)
    return acc, sen, pre, F1, spe


def _compute_loss(self, preds, gts, task_name=None):
    train_losses = torch.zeros(self.task_num).to(self.device)
    for tn, task in enumerate(self.task_name):
        train_losses[tn] = self.meter.losses[task]._update_loss(preds[task], gts[task])

    return train_losses

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


    transform_dict = {
        tio.RandomNoise(),
        tio.RandomFlip(flip_probability=0.5, axes=('LR')),
        tio.RandomBlur(),
        #tio.RandomAffine(scales=0.75),#要设定范围（小范围）
        tio.RandomSpike(),#
        #tio.RandomSwap()
        }
    

    #load the training_data  and test_data (training_data will be splited later for cross_validation)
    total_file = DIY_Folder(data_path=data_path,train_form = opt.train_form,root_mask_radiomics = opt.root_bbx_path,use_radiomics=opt.use_radiomics,istest=False,sec_pair_orig=opt.sec_pair_orig)
    print("len(total_file): ",len(total_file))
    print("total_file.gety(): ",total_file.gety())
    #root_radiomics="",root_mask_radiomics="",use_radiomics=True, norm_radiomics=True
    test_file = DIY_Folder(data_path=test_data_path,train_form = opt.train_form,root_mask_radiomics = opt.test_root_bbx_path,use_radiomics=opt.use_radiomics,istest=True,sec_pair_orig=opt.sec_pair_orig)

    test_data = test_file.select_dataset(data_idx=[i for i in range(len(test_file))], aug=False,use_secondimg=opt.usesecond,noSameRM=opt.noSameRM,usethird=opt.usethird)



    test_loader= torch.utils.data.DataLoader(test_data
                                                , batch_size = opt.batch_size
                                                , num_workers = opt.num_workers
                                                , pin_memory = True
                                                , drop_last = False
                                                )

    print("opt.output_dir: ",opt.output_dir)
    load_data_time = time.time()-begin_time
    print("[....Loading data OK....]: %dh, %dmin, %ds "%(int(load_data_time/3600),int(load_data_time/60),int(load_data_time%60)))
    print("[....Loading data OK....]: %dh, %dmin, %ds "%(int(load_data_time/3600),int(load_data_time/60),int(load_data_time%60)),file=record_file)
    begin_time = time.time()
    loss_func_dict = { 'CE' : nn.CrossEntropyLoss().to(DEVICE)
                      , 'Weighted_CE' : Weighted_CE(classes_weight=opt.loss_weight,n_classes=opt.num_classes)
                     , 'SelfKL' :  SelfKL(num_classes=opt.num_classes,lambda_0=opt.lambda_0,lambda_1=opt.lambda_1,lambda_2=opt.lambda_2,lambda_3=opt.lambda_2, CE_or_KL=opt.CE_or_KL)                    
                     }
    criterion = loss_func_dict[opt.lossfunc]
    if opt.lossfunc=="SelfKL":
        tasks=['L0','L2','L3']
        loss_fn = {'L0':Self_L0(),'L2':Self_L2(),'L3':Self_L3()}
    if opt.train_form == "masked_constrained_train":
        tasks=['a','b']
        loss_fn = {'a':nn.CrossEntropyLoss(),'b':nn.CrossEntropyLoss()}

    train_func_dict={"None": train
                    ,"masked_constrained_train": masked_constrained_train
                    }
    training = train_func_dict[opt.train_form]
    valid_func_dict={"None": validate
                ,"masked_constrained_train": masked_constrained_validate
                }
    validating = valid_func_dict[opt.train_form]

    noemal_valid_func_dict={"None": normal_validate
                ,"masked_constrained_train": normal_masked_constrained_validate
                }
    normal_validating = noemal_valid_func_dict[opt.train_form]
    

    epochs = opt.epochs
    print("total epochs : ",epochs)
    sum_writer = tensorboardX.SummaryWriter(opt.output_dir)
    print(" ==========> All settled. Training is getting started...")
    print(" ==========> Training takes {} epochs.".format(epochs))

    # split the training_data into K fold with StratifiedKFold(shuffle = True)
    k=5
    splits=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)
    
    # just a record
    strafold_fp= open(opt.output_dir+opt.model+'_'+opt.lossfunc+opt.ps+"fold_split_case.txt", 'w')

    # to record metrics: the best_acc of k folds, best_acc of each fold
    foldperf={}
    fold_best_acc,best_fold,best_epoch=-1,1,0

    fold_record_valid_metrics,fold_record_matched_test_metrics,fold_aucs,test_fold_aucs=[],[],[],[]
    fold_best_scale={}


    # record the best_model's statedict
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
        print("Get valid set")
        vali_data=total_file.select_dataset(data_idx=val_idx, aug=False,use_secondimg=opt.usesecond,noSameRM=opt.noSameRM,usethird=opt.usethird)
        print("Got train set")
        train_data=total_file.select_dataset(data_idx=train_idx, aug=True,aug_form=opt.aug_form,use_secondimg=opt.usesecond,noSameRM=opt.noSameRM,usethird=opt.usethird)
        

        if opt.use_radiomics:
            radio_mean, radio_std = train_data.calc_own_radiomics_mean()
            vali_data.inject_other_mean(radio_mean, radio_std)
            test_data.inject_other_mean(radio_mean, radio_std)
            
            print("train_data. radio_mean: ",train_data.radio_mean)
            train_data.input_unit_radiomics_mean(radio_mean, radio_std)
            vali_data.input_unit_radiomics_mean(radio_mean, radio_std)        
            test_data.input_unit_radiomics_mean(radio_mean, radio_std)
            print("train_data. radio_mean: ",train_data.radio_mean)
                      
        train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers = opt.num_workers
                                                , shuffle = True
                                                , pin_memory = True
                                                , drop_last = False)

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
            num_seg_classes=2)
            model =  model.to(DEVICE)
            net_dict = model.state_dict()
            checkpoint = torch.load(opt.tencent_pth_rootdir + "resnet_10_23dataset.pth")     
            print("pretrained model exists? ",os.path.exists(opt.tencent_pth_rootdir + "resnet_10_23dataset.pth")) 
            print("net_dict.keys(): ",net_dict.keys())
            pretrain_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in net_dict.keys()}
            net_dict.update(pretrain_dict)
            model.load_state_dict(net_dict)
            print("check_point_path: ",opt.tencent_pth_rootdir + "resnet_10_23dataset.pth")
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
        if opt.model == "ResNet18":
            model = ResNet18(num_classes=2)
        elif opt.model == "ResNet10":
            model = ResNet10(num_classes=2)
        elif opt.model == "DIY_ResNet18":
            model = DIY_ResNet18(num_classes=2)
        elif opt.model == "DIY_ResNet10":
            model = DIY_ResNet10(num_classes=2)
        else:
            print("[ERROR: ] Wrong model chosen\n")
        
        model = model.to(DEVICE)
        n_iter = 0
        single_loss_writer = tensorboardX.SummaryWriter(opt.output_dir+'/loss_fold_'+str(fold))


        optimizer = optim.Adam(model.parameters(), lr = opt.lr, weight_decay = opt.weight_decay)#decay=0

        # 设置warm up的轮次为20次
        warm_up_iter = opt.warm_up_iter
        lr_step = opt.lr_step
        T_max = opt.epochs	# 周期
        lr_max = opt.max_lr	# 最大值: 0.1/
        lr_min = opt.min_lr	# 最小值: 1e-5

        # 为param_groups[0] (即model.layer2) 设置学习率调整规则 - Warm up + Cosine Anneal
        lambda0 = lambda cur_iter: (cur_iter % 10+1 )*10**(cur_iter // 5)*1e-4 if  cur_iter < warm_up_iter else \
                                   (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.1

        print("step lr :",(lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (20-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.1)
        # LambdaLR
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)


        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose = 1, patience=5, factor = 0.5) 
        # torch.optim.lr_scheduler.steplr
        
        early_stopping = EarlyStopping(patience = 50, verbose=True)
        history = {'train_loss': [], 'valid_loss': [],'test_loss':[],'train_acc':[],'valid_acc':[], 'test_acc':[],'train_auc':[],'test_auc':[],'valid_auc':[],'lr':[]}


        
        train_loss_box ,train_acc_box= [],[]
        valid_loss_box,valid_acc_box=[],[]
        t_acc_record,v_acc_record=[],[]
        test_acc_record,test_loss_box,test_acc_box=[],[],[]

        saved_metrics,saved_epoch=[],[]

        best_acc, sofar_valid_acc, sofar_valid_auc=-1,-1,-1

        best_statedict = deepcopy(model.state_dict())
        best_epoch=0
        best_scale = {}
        
        for epoch in range(opt.epochs):

            train_loss, train_losses, train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3],n_iter,scale= training(train_loader = train_loader
                                                , model = model
                                                , criterion = criterion
                                                , optimizer = optimizer
                                                , epoch = epoch
                                                , tasks = tasks
                                                , loss_fn = loss_fn
                                                ,use_MGDA = opt.use_MGDA
                                                , n_iter = n_iter
                                                , writer = single_loss_writer)

            #================================== every epoch's metrics record =================================================
            """
            train_loss, train_losses, train_acc, train_accs,train_met =validate(valid_loader = train_loader
                                        , model = model
                                        , criterion = criterion)
            """
            train_a,train_b,train_c = train_met.a, train_met.b, train_met.c
            train_pre, train_rec, train_F1, train_spe = train_met.pre,train_met.sen,train_met.F1, train_met.spe

            valid_loss, valid_losses, valid_acc, valid_accs, valid_met,[valid_loss_0,valid_loss_1,valid_loss_2,valid_loss_3]= validating(valid_loader = valid_loader
                                                                                                                                         , model = model
                                                                                                                                         , criterion = criterion
                                                                                                                                         , tasks = tasks
                                                                                                                                        , loss_fn = loss_fn
                                                                                                                                        , use_MGDA = opt.use_MGDA
                                                                                                                                        , n_iter = n_iter
                                                                                                                                        , writer = single_loss_writer
                                                                                                                                        , ps='valid')
            valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c
            valid_pre, valid_rec, valid_F1, valid_spe = valid_met.pre,valid_met.sen, valid_met.F1, valid_met.spe

            test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3]= validating(valid_loader = test_loader
                                                                                                                                         , model = model
                                                                                                                                         , criterion = criterion
                                                                                                                                         , tasks = tasks
                                                                                                                                        , loss_fn = loss_fn
                                                                                                                                        , use_MGDA = opt.use_MGDA
                                                                                                                                        , n_iter = n_iter
                                                                                                                                        , writer = single_loss_writer
                                                                                                                                        , ps='test')

            test_a,test_b,test_c = test_met.a, test_met.b, test_met.c
            test_pre,test_rec, test_F1,test_spe = test_met.pre,test_met.sen, test_met.F1, test_met.spe
            

            valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu()[:,1])
            test_auc = roc_auc_score(test_a.cpu(),test_b.cpu()[:,1])

            train_loss_box.append(train_loss.detach().cpu())
            train_acc_box.append(train_acc)
            valid_loss_box.append(valid_loss.detach().cpu())
            valid_acc_box.append(valid_acc)
            test_loss_box.append(test_loss)       
            test_acc_box.append(test_acc)

            t_acc_record.append(train_accs.list)
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

            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"F1", {'train':train_F1,'valid':valid_F1,'test':test_F1},epoch)
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"recall", {'train':train_rec,'valid':valid_rec,'test':test_rec},epoch)
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"spe", {'train':train_spe,'valid':valid_spe,'test':test_spe},epoch)

            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"train_subloss", {'loss_0':train_loss_0,'loss_1':train_loss_1,'loss_2':train_loss_2,'loss_3':train_loss_3},epoch)
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"loss_0", {'train':train_loss_0,'valid':valid_loss_0,'test':test_loss_0},epoch)
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"loss_1", {'train':train_loss_1,'valid':valid_loss_1,'test':test_loss_1},epoch)
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"loss_2", {'train':train_loss_2,'valid':valid_loss_2,'test':test_loss_2},epoch)
            sum_writer.add_scalars(opt.model+str(fold)+sum_write_Mark+"loss_3", {'train':train_loss_3,'valid':valid_loss_3,'test':test_loss_3},epoch)
         
            is_best=False
            """
            if epoch%100 == 0:
                tmp_state = deepcopy(model.state_dict())
                #pps="_"+str(epoch/100)+"_"
                pps="_k-fold-sub-fold-"+str(fold)+"_"
                save_checkpoint({ 'epoch': epoch
                , 'arch': opt.model
                , 'state_dict': tmp_state
                , 'fold':fold}
                , is_best = False
                , out_dir = opt.output_dir
                , model_name = opt.model
                , pps = pps
                , fold = fold
                , epoch = epoch
                )
            """
            
            if valid_acc > best_acc:
                best_acc = valid_acc
                sofar_valid_acc = valid_acc
                sofar_valid_auc = valid_auc
                sofar_valid_metrics = valid_met
                sofar_test_metrics = test_met
                sofar_test_auc = test_auc
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
        if fold_best_acc < best_acc:
            print("fold: %d, change fold_best_acc from %f to %f "%(fold,fold_best_acc,best_acc))
            fold_best_acc = best_acc
            best_fold = fold
            fold_best_statedict=best_statedict
            fold_best_scale = best_scale
            

            is_best=True
            saved_metrics.append(fold_best_acc)
            saved_epoch.append(best_epoch)
            pps="_total_best_"
            print("【Total best fresh!】====> Best at fold %d, epoch %d, valid acc: %f  auc: %f\n"%(fold,epoch, best_acc, sofar_valid_auc))
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
         
    
        # to print & record the best result of this fold
        pps="_k-fold-sub-fold-"+str(fold)+"_"
        model = load_curr_best_checkpoint(model,out_dir = opt.output_dir,model_name = opt.model,pps = pps)
        #model.load_state_dict(best_statedict)        
        train_loss, train_losses, train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3] =normal_validating(valid_loader = train_loader
                                    , model = model
                                    , criterion = criterion)
        train_a,train_b,train_c = train_met.a, train_met.b, train_met.c

        valid_loss, valid_losses, valid_acc, valid_accs, valid_met,[valid_loss_0,valid_loss_1,valid_loss_2,valid_loss_3]= normal_validating(valid_loader = valid_loader
                                    , model = model
                                    , criterion = criterion)
        valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c

        test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3] = normal_validating(valid_loader = test_loader
                                    , model = model
                                    , criterion = criterion)
        test_a,test_b,test_c = test_met.a, test_met.b, test_met.c

        train_auc = roc_auc_score(train_a.cpu(),train_b.cpu()[:,1])
        valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu()[:,1])
        test_auc = roc_auc_score(test_a.cpu(),test_b.cpu()[:,1])

        recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_epoch",record_file = None,ps_thing=[best_fold,fold,fold_best_acc,best_acc,best_epoch],scale = best_scale)
        recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_epoch",record_file = record_file,ps_thing=[best_fold,fold,fold_best_acc,best_acc,best_epoch],scale=best_scale)


        plt.subplot(2,3,1)
        plt.plot(history['train_loss'],'r--',label='t_loss')
        plt.plot(history['valid_loss'],'b--',label='v_loss')
        plt.title("train_valid_loss")


        plt.subplot(2,3,3)
        plt.plot(history['train_acc'],'r--',label='t_acc')
        plt.plot(history['valid_acc'],'b--',label='v_acc')
        plt.title("train_valid_acc")

        """
        plt.subplot(2,3,2)
        plt.plot(history['train_loss'],'g--',label='t_loss')
        plt.plot(history['train_acc'],'y--',label='t_acc')
        plt.title("train_loss_acc")
        
        plt.subplot(2,3,4)
        plt.plot(history['valid_loss'],'g--',label='v_loss')
        plt.plot(history['valid_acc'],'y--',label='v_acc')
        plt.title("valid_acc_loss")
        """

        plt.subplot(2,3,5)
        plt.plot(history['lr'],'b--',label='learning_rate')
        plt.title("learning_rate")

        plt.savefig(opt.pic_output_dir+opt.model+'_'+opt.lossfunc+opt.ps+'_k-fold-'+str(fold)+'_loss_acc.png')

        #plt.show()
        plt.close()


        foldperf['fold{}'.format(fold+1)] = history  
  

    os.system('echo " === TRAIN mae mtc:{:.5f}" >> {}'.format(train_loss, output_path))

    # to get averaged metrics of k folds
    record_avg(fold_record_valid_metrics,fold_record_matched_test_metrics,record_file,fold_aucs,test_fold_aucs)

    # to print & record the best result of total
    model.load_state_dict(fold_best_statedict)        
    train_loss, train_losses, train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3] =normal_validating(valid_loader = train_loader
                                , model = model
                                , criterion = criterion)
    train_a,train_b,train_c = train_met.a, train_met.b, train_met.c

    valid_loss, valid_losses, valid_acc, valid_accs, valid_met,[valid_loss_0,valid_loss_1,valid_loss_2,valid_loss_3]= normal_validating(valid_loader = valid_loader
                                , model = model
                                , criterion = criterion)
    valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c

    test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3] = normal_validating(valid_loader = test_loader
                                , model = model
                                , criterion = criterion)
    test_a,test_b,test_c = test_met.a, test_met.b, test_met.c

    train_auc = roc_auc_score(train_a.cpu(),train_b.cpu()[:,1])
    valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu()[:,1])
    test_auc = roc_auc_score(test_a.cpu(),test_b.cpu()[:,1])

    recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_fold",record_file = None,ps_thing=[best_fold,best_fold,fold_best_acc,fold_best_acc,best_epoch],scale = fold_best_scale)
    recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_fold",record_file = record_file,ps_thing=[best_fold,best_fold,fold_best_acc,fold_best_acc,best_epoch],scale = fold_best_scale)


    torch.cuda.empty_cache()
    sum_writer.close()
    strafold_fp.close()
    record_file.close()
    print("=======training end========")



def record_avg(fold_record_valid_metrics,fold_record_matched_test_metrics,record_file,fold_aucs,test_fold_aucs):
    

    fold_accs,fold_f1s,fold_pres,fold_recalls,fold_spes=[],[],[],[],[]
    test_fold_accs,test_fold_f1s,test_fold_pres,test_fold_recalls,test_fold_spes=[],[],[],[],[]

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
        
    record_file.write("vALID<== acc records , avg: %f ==>"%(mean(fold_accs)))
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
    print("[valid]=%.3f-|-%.3f-|-%.3f-|-%.3f-|-%.3f-|-%.3f"%(mean(fold_accs),mean(fold_aucs),mean(fold_f1s),mean(fold_pres),mean(fold_recalls),mean(fold_spes)))
    print("[test]==%.3f-|-%.3f-|-%.3f-|-%.3f-|-%.3f-|-%.3f"%(mean(test_fold_accs),mean(test_fold_aucs),mean(test_fold_f1s),mean(test_fold_pres),mean(test_fold_recalls),mean(test_fold_spes)))


def  recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="",record_file = None,ps_thing=[],scale = {}):
    
    train_a, train_c,train_acc,train_pre, train_rec, train_F1, train_spe = train_met.a, train_met.c,train_met.acc,train_met.pre,train_met.sen,train_met.F1, train_met.spe
    valid_a,valid_c,valid_acc,valid_pre, valid_rec, valid_F1, valid_spe = valid_met.a,  valid_met.c, valid_met.acc,valid_met.pre,valid_met.sen, valid_met.F1, valid_met.spe
    test_a, test_c, test_acc, test_pre,  test_rec,   test_F1,test_spe   =  test_met.a,   test_met.c, test_met.acc,      test_met.pre,test_met.sen, test_met.F1, test_met.spe

    [best_fold,fold,fold_best_acc,best_acc,best_epoch] = ps_thing
    if not record_file == None:
        print("scale: ==========================================",file=record_file)
        print(scale,file=record_file)
    print("scale: ==========================================")
    print(scale)

    if record_form == "best_fold":
        print("\n============ Total End ============\n")
        if record_file!=None: print("\n============ Total End ============\n",file=record_file)

    head_str = "\n[Fold end]" if record_form=="best_epoch" else "\n[Total end]"
    if record_file == None:
        print(head_str+"best_fold %d & now_fold %d  |||| best_vali_acc %f  & now_vali_acc %f ==========================================\n"%(best_fold, fold, fold_best_acc,best_acc ))
        print("\n================================== [Fold: %d [Train] best at epoch %d ] ==========================================\n"%(fold, best_epoch))                                                                                                    
        print(classification_report(train_a.cpu(), train_c.cpu(), target_names=target_names, labels=[0,1]))
        print("\n================================== [Fold: %d [Valid] best at epoch %d ] ==========================================\n"%(fold, best_epoch)) 
        print(classification_report(valid_a.cpu(), valid_c.cpu(), target_names=target_names, labels=[0,1]))
        print("\n================================== [Fold: %d [Test] best at epoch %d ] ==========================================\n"%(fold, best_epoch)) 
        print(classification_report(test_a.cpu(), test_c.cpu(), target_names=target_names, labels=[0,1]))

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
    #===============================================================
    else:

        print(head_str+"best_fold %d & now_fold %d  |||| best_vali_acc %f  & now_vali_acc %f ==========================================\n"%(best_fold, fold, fold_best_acc,best_acc),file=record_file)
        print("\n================================== [Fold: %d [Train] best at epoch %d loss: %f ] ==========================================\n"%(fold, best_epoch,train_loss),file=record_file)                                                                                                    
        print(classification_report(train_a.cpu(), train_c.cpu(), target_names=target_names, labels=[0,1]),file=record_file)
        print("\n================================== [Fold: %d [Valid] best at epoch %d loss: %f ] ==========================================\n"%(fold, best_epoch, valid_loss),file=record_file) 
        print(classification_report(valid_a.cpu(), valid_c.cpu(), target_names=target_names, labels=[0,1]),file=record_file)
        print("\n================================== [Fold: %d [Test] best at epoch %d loss: %f ] ==========================================\n"%(fold, best_epoch, test_loss),file=record_file) 
        print(classification_report(test_a.cpu(), test_c.cpu(), target_names=target_names, labels=[0,1]),file=record_file)

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


if __name__ == "__main__":
    output_path = os.path.join(opt.output_dir, 'result')
    print("output_path: ",output_path)

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    if not os.path.exists(opt.pic_output_dir):
        os.makedirs(opt.pic_output_dir)      
    print("=> training beigin. \n")
    os.system('echo "train {}"  >>  {}'.format(datetime.datetime.now(),output_path))

    main(output_path)

    