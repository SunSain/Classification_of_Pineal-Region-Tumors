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

from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from statistics import mean
import time
import math
from model import tencent_resnet

target_names = ['class 0', 'class 1']


if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_device(2)
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
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    best_epoch = checkpoint['epoch']
    return model,best_epoch

def convert(original): #[[0],[1],[0],[0]]→ [0,1,0,0]
    target=torch.Tensor(len(original))
    for i in range(len(original)):
        target[i]=original[i][0]
    target=target.type(torch.LongTensor).to(DEVICE)
    return target



def normal_validate(valid_loader, model, criterion, lossfunc_name,hyf):
        
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
            
            if(lossfunc_name == 'SelfKL'):
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
    
def normal_masked_constrained_validate(valid_loader, model, criterion,lossfunc_name,hyf):
    
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


            img = img.to(DEVICE)
            input_affix = saffix.to(DEVICE)
            masked_img = masked_img.to(DEVICE)
            #input = torch.reshape(input, [input.shape[0],1,input.shape[1],input.shape[2],input.shape[3]])
            
            img_out = model(img,input_affix)
            masked_img_out = model(masked_img,input_affix)
            
            if(lossfunc_name == 'SelfKL'):
                 loss_1,_ ,_,_,_ = criterion(img_out, target)
                 loss_2,_ ,_,_,_ = criterion(masked_img_out, target)
                 loss_b,_ ,_,_,_ = criterion(masked_img_out, F.softmax(img_out,dim=1),False)
            else:        
                loss_1 = criterion(img_out, target) 
                loss_2 = criterion(masked_img_out, target)
                loss_b = criterion(masked_img_out, F.softmax(img_out,dim=1)) 
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

"""
def validate(valid_loader, model, criterion,lossfunc_name, hyf):
        
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
            
            if(lossfunc_name == 'SelfKL'):
                print('SelfKL')
                loss,loss_0 ,loss_1,loss_2,loss_3 = criterion(out, target)
            else:
                print("Not selfkl, criterion ")
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
    
def masked_constrained_validate(valid_loader, model, criterion,lossfunc_name,hyf):
    
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


            img = img.to(DEVICE)
            input_affix = saffix.to(DEVICE)
            masked_img = masked_img.to(DEVICE)
            #input = torch.reshape(input, [input.shape[0],1,input.shape[1],input.shape[2],input.shape[3]])
            
            img_out = model(img,input_affix)
            masked_img_out = model(masked_img,input_affix)

            
            if(lossfunc_name == 'SelfKL'):
                 loss_0,_ ,_,_,_ = criterion(img_out, target)
                 loss_1,_ ,_,_,_ = criterion(masked_img_out, target)
                 loss_2,_ ,_,_,_ = criterion(masked_img_out, F.softmax(img_out,dim=1),False)
            else:        
                loss_0 = criterion(img_out, target) 
                loss_1 = criterion(masked_img_out, target)
                loss_2 = criterion(masked_img_out, F.softmax(img_out,dim=1),False) 
            
            loss = hyf["constrain_lambd"]  * (loss_0 + loss_1) + (1.0-hyf["constrain_lambd"] )*loss_2
                
            Losses.update(loss*img.size(0),img.size(0)) 
            
            Loss_0.update(loss_0*img.size(0),img.size(0))
            Loss_1.update(loss_1*img.size(0),img.size(0))
            Loss_2.update(loss_2*img.size(0),img.size(0))
            Loss_3.update( hyf["constrain_lambd"] * (loss_0 + loss_1)*img.size(0),img.size(0))



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
"""
"""

def main(output_path):
    
    begin_time = time.time()
    json_path = os.path.join(opt.output_dir, 'hyperparameter.json')

    # record metrics into a txt
    record_file_path= opt.output_dir+opt.model+'_'+opt.lossfunc+opt.ps+'_test_record.txt'
    record_file = open(record_file_path, "w")
    
    data_path=opt.data_path
    test_data_path = opt.testdata_path

    print("=========== start train the brain age estimation model =========== \n")
    print(" ==========> Using {} processes for data loader.".format(opt.num_workers))


    #load the training_data  and test_data (training_data will be splited later for cross_validation)
    total_file = DIY_Folder(data_path=data_path,train_form = opt.train_form,root_mask_radiomics = opt.root_bbx_path,use_radiomics=opt.use_radiomics,istest=False)
    print("len(total_file): ",len(total_file))
    print("total_file.gety(): ",total_file.gety())
    #root_radiomics="",root_mask_radiomics="",use_radiomics=True, norm_radiomics=True
    test_file = DIY_Folder(data_path=test_data_path,train_form = opt.train_form,root_mask_radiomics = opt.test_bbx_path,use_radiomics=opt.use_radiomics,istest=True)

    test_data = test_file.select_dataset(data_idx=[i for i in range(len(test_file))], aug=False,use_secondimg=True,noSameRM=opt.noSameRM,usethird=opt.usethird)



    test_loader= torch.utils.data.DataLoader(test_data
                                                , batch_size = opt.batch_size
                                                , num_workers = opt.num_workers
                                                , pin_memory = True
                                                , drop_last = False
                                                )


    load_data_time = time.time()-begin_time
    print("[....Loading data OK....]: %dh, %dmin, %ds "%(int(load_data_time/3600),int(load_data_time/60),int(load_data_time%60)))
    print("[....Loading data OK....]: %dh, %dmin, %ds "%(int(load_data_time/3600),int(load_data_time/60),int(load_data_time%60)),file=record_file)
    begin_time = time.time()
    loss_func_dict = { 'CE' : nn.CrossEntropyLoss().to(DEVICE)
                      , 'Weighted_CE' : Weighted_CE(classes_weight=opt.loss_weight,n_classes=opt.num_classes)
                      , 'SelfKL' :  SelfKL(num_classes=opt.num_classes,lambda_0=opt.lambda_0,lambda_1=opt.lambda_1,lambda_2=opt.lambda_2,lambda_3=opt.lambda_2, CE_or_KL=opt.CE_or_KL)                    
                     }

    criterion = loss_func_dict[opt.lossfunc]

    valid_func_dict={"None": validate
                ,"masked_constrained_train": masked_constrained_validate
                }
    #validating = valid_func_dict[opt.train_form]
    
    noemal_valid_func_dict={"None": normal_validate
                ,"masked_constrained_train": normal_masked_constrained_validate
                }
    validating = noemal_valid_func_dict[opt.train_form]
    

    print(" ==========> All settled. Training is getting started...")


    # split the training_data into K fold with StratifiedKFold(shuffle = True)
    k=5
    splits=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)
    
    # just a record
    strafold_fp= open(opt.output_dir+opt.model+'_'+opt.lossfunc+opt.ps+"fold_split_case.txt", 'w')

    # to record metrics: the best_acc of k folds, best_acc of each fold
    foldperf={}
    fold_best_acc=-1
    best_fold=1
    best_epoch=0
    
    fold_record_valid_metrics,fold_record_matched_test_metrics,fold_aucs,test_fold_aucs=[],[],[],[]
    fold_accs,fold_f1s,fold_pres,fold_recalls,fold_spes=[],[],[],[],[]
    test_fold_accs,test_fold_f1s,test_fold_pres,test_fold_recalls,test_fold_spes=[],[],[],[],[]


    # record the best_model's statedict
    fold_best_statedict = None  

    chores_time = time.time()-begin_time
    print("[....Chores time....]: %dh, %dmin, %ds "%(int(chores_time/3600),int(chores_time/60),int(chores_time%60)))
    begin_time = time.time()

    #================= begin to train, choose 1 of k folds as validation =================================
    print("======================== start train ================================================ \n")

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(total_file)),total_file.gety())):
 
        begin_training_time = time.time()-begin_time
        print("[....Training time....]: %dh, %dmin, %ds "%(int(begin_training_time/3600),int(begin_training_time/60),int(begin_training_time%60)))
        print("[....Training time....]: %dh, %dmin, %ds "%(int(begin_training_time/3600),int(begin_training_time/60),int(begin_training_time%60)),file=record_file)
        begin_time = time.time()

        print("\n============ FOLD %d ============\n"%(fold),file=record_file)
        print('Fold {}'.format(fold))
        


        #valid_data(no augmentation: aug=False) & train_data(need augmentation:aug = True)
        print("Get valid set")
        vali_data=total_file.select_dataset(data_idx=val_idx, aug=False,use_secondimg=True,noSameRM=opt.noSameRM,usethird=opt.usethird)
        print("Got train set")
        train_data=total_file.select_dataset(data_idx=train_idx, aug=True,aug_form=opt.aug_form,use_secondimg=True,noSameRM=opt.noSameRM,usethird=opt.usethird)
        

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
            model = tencent_resnet.resnet10(sample_input_W=400,
            sample_input_H=400,
            sample_input_D=23,
            shortcut_type='B',
            no_cuda=False,
            num_seg_classes=2)
            model =  model.to(DEVICE)
            net_dict = model.state_dict()
            checkpoint = torch.load(opt.tencent_pth_rootdir + "resnet_10_23dataset.pth")      
            print("net_dict.keys(): ",net_dict.keys())
            pretrain_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in net_dict.keys()}
            net_dict.update(pretrain_dict)
            model.load_state_dict(net_dict)
            print("check_point_path: ",opt.tencent_pth_rootdir + "resnet_10_23dataset.pth")
        elif opt.model == "ResNet18":
            model = ResNet18()
        elif opt.model == "ResNet10":
            model = ResNet10()
        elif opt.model == "DIY_ResNet18":
            model = DIY_ResNet18()
        elif opt.model == "DIY_ResNet10":
            model = DIY_ResNet10()
        else:
            print("[ERROR: ] Wrong model chosen\n")
        
        model = model.to(DEVICE)

        pps="_k-fold-sub-fold-"+str(fold)+"_"
        pps="_total_best_"
        model, best_epoch = load_curr_best_checkpoint(model,out_dir = opt.output_dir,model_name = opt.model,pps = pps)
        #model.load_state_dict(best_statedict)        
        train_loss, train_losses, train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3]  =validating(valid_loader = train_loader
                                    , model = model
                                    , criterion = criterion)
        train_a,train_b,train_c = train_met.a, train_met.b, train_met.c
        train_acc,train_pre, train_rec, train_F1, train_spe =train_met.acc, train_met.pre,train_met.sen,train_met.F1, train_met.spe

        valid_loss, valid_losses, valid_acc, valid_accs, valid_met,[valid_loss_0,valid_loss_1,valid_loss_2,valid_loss_3]= validating(valid_loader = valid_loader, model = model, criterion = criterion)
        valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c
        valid_pre, valid_rec, valid_F1, valid_spe = valid_met.pre,valid_met.sen, valid_met.F1, valid_met.spe

        test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3]= validating(valid_loader = test_loader, model = model, criterion = criterion)
        test_a,test_b,test_c = test_met.a, test_met.b, test_met.c
        test_pre,test_rec, test_F1,test_spe = test_met.pre,test_met.sen, test_met.F1, test_met.spe
        
        train_auc = roc_auc_score(train_a.cpu(),train_b.cpu()[:,1])
        valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu()[:,1])
        test_auc = roc_auc_score(test_a.cpu(),test_b.cpu()[:,1])
        fold_aucs.append(valid_auc)
        test_fold_aucs.append(test_auc)
        sofar_valid_metrics = valid_met
        fold_record_valid_metrics.append(sofar_valid_metrics)
        fold_record_matched_test_metrics.append(test_met)

        recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_epoch",record_file = None,ps_thing=[best_fold,fold,valid_acc,valid_acc,best_epoch])
        recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_epoch",record_file = record_file,ps_thing=[best_fold,fold,valid_acc,valid_acc,best_epoch])



    os.system('echo " ================================= "')


    os.system('echo " === TRAIN mae mtc:{:.5f}" >> {}'.format(train_loss, output_path))

    # to get averaged metrics of k folds
    record_avg(fold_record_valid_metrics,fold_record_matched_test_metrics,record_file,fold_aucs,test_fold_aucs)

    # to print & record the best result of total  
    pps="_total_best_"
    model, best_epoch = load_curr_best_checkpoint(model,out_dir = opt.output_dir,model_name = opt.model,pps = pps)  
    train_loss, train_losses, train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3]  =validating(valid_loader = train_loader
                                , model = model
                                , criterion = criterion)
    train_a,train_b,train_c = train_met.a, train_met.b, train_met.c
    train_acc,train_pre, train_rec, train_F1, train_spe =train_met.acc, train_met.pre,train_met.sen,train_met.F1, train_met.spe

    valid_loss, valid_losses, valid_acc, valid_accs, valid_met,[valid_loss_0,valid_loss_1,valid_loss_2,valid_loss_3]= validating(valid_loader = valid_loader, model = model, criterion = criterion)
    valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c
    valid_pre, valid_rec, valid_F1, valid_spe = valid_met.pre,valid_met.sen, valid_met.F1, valid_met.spe

    test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3]= validating(valid_loader = test_loader, model = model, criterion = criterion)
    test_a,test_b,test_c = test_met.a, test_met.b, test_met.c
    test_pre,test_rec, test_F1,test_spe = test_met.pre,test_met.sen, test_met.F1, test_met.spe

    train_auc = roc_auc_score(train_a.cpu(),train_b.cpu()[:,1])
    valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu()[:,1])
    test_auc = roc_auc_score(test_a.cpu(),test_b.cpu()[:,1])

    recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_fold",record_file = None,ps_thing=[best_fold,best_fold,fold_best_acc,fold_best_acc,best_epoch])
    recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_fold",record_file = record_file,ps_thing=[best_fold,best_fold,fold_best_acc,fold_best_acc,best_epoch])


    torch.cuda.empty_cache()
    record_file.close()
    strafold_fp.close()
"""

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


def  recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="",record_file = None,ps_thing=[]):
    
    train_a, train_c,train_acc,train_pre, train_rec, train_F1, train_spe = train_met.a, train_met.c,train_met.acc,train_met.pre,train_met.sen,train_met.F1, train_met.spe
    valid_a,valid_c,valid_acc,valid_pre, valid_rec, valid_F1, valid_spe = valid_met.a,  valid_met.c, valid_met.acc,valid_met.pre,valid_met.sen, valid_met.F1, valid_met.spe
    test_a, test_c, test_acc, test_pre,  test_rec,   test_F1,test_spe   =  test_met.a,   test_met.c, test_met.acc,      test_met.pre,test_met.sen, test_met.F1, test_met.spe

    [best_fold,fold,fold_best_acc,best_acc,best_epoch] = ps_thing

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



def main(output_path):
    
    # bbx/bbx_pair_orig, constrain part
    data_path=opt.data_path
    test_data_path = opt.testdata_path   
    root_mask_radiomics =  opt.root_bbx_path
    test_root_mask_radiomics = opt.test_root_bbx_path
    train_form = "masked_constrained_train"
    
    train_form = "None"
    root_mask_radiomics =  opt.root_mask_radiomics
    test_root_mask_radiomics = opt.test_root_mask_radiomics
    
    use_radiomics=False
    usethird= False
    sec_pair_orig=False

    total_file = DIY_Folder(data_path=data_path,train_form = train_form,root_mask_radiomics = root_mask_radiomics,use_radiomics=use_radiomics,istest=False,sec_pair_orig=sec_pair_orig)
    print("len(total_file): ",len(total_file))
    print("total_file.gety(): ",total_file.gety())
    #root_radiomics="",root_mask_radiomics="",use_radiomics=True, norm_radiomics=True
    test_file = DIY_Folder(data_path=test_data_path,train_form = train_form,root_mask_radiomics = test_root_mask_radiomics,use_radiomics=use_radiomics,istest=True,sec_pair_orig=sec_pair_orig)

    seg_NoneTrain_dirs=["/opt/chenxingru/Pineal_region/after_12_08/Results/0402_T1C/no_MGDA/secondimg_tencentMed_composed_selfKL/model_result/",
                        "/opt/chenxingru/Pineal_region/after_12_08/Results/0402_T1C/no_MGDA/secondimg_tencentMed_composed_RM_selfKL_5021_RMafter/model_result/",
                        "/opt/chenxingru/Pineal_region/after_12_08/Results/0402_T1C/no_MGDA/secondimg_tencentMed_composed_CE/model_result/",
                        "/opt/chenxingru/Pineal_region/after_12_08/Results/0402_T1C/MGDA/secondimg_tencentMed_composed_RM_selfKL/model_result/"]
    
    notsecond_output_dirs=["/opt/chenxingru/Pineal_region/after_12_08/Results/0402_T1C/MGDA/none_masked_constrained_train_CE/model_result/",
                           "/opt/chenxingru/Pineal_region/after_12_08/Results/0402_T1C/no_MGDA/tencentMed_composed_masked_constrained_train_CE/model_result/",
                           "/opt/chenxingru/Pineal_region/after_12_08/Results/0402_T1C/no_MGDA/none_masked_constrained_train_CE/model_result/",
                           "/opt/chenxingru/Pineal_region/after_12_08/Results/0402_T1C/no_MGDA/no_tencentMed_composed_RM_masked_constrained_train_CE_2/model_result/"]
    
    seg_output_dirs=["/opt/chenxingru/Pineal_region/after_12_08/Results/0402_T1C/no_MGDA/secondimg_noSameRM_composed_constrained_trian_CE_55/model_result/",
                     "/opt/chenxingru/Pineal_region/after_12_08/Results/0402_T1C/no_MGDA/secondimg_composed_constrained_trian_CE_55/model_result/",
                    "/opt/chenxingru/Pineal_region/after_12_08/Results/0402_T1C/no_MGDA/secondimg_tencentMed_composed_constrained_trian_CE_55/model_result/",
                    "/opt/chenxingru/Pineal_region/after_12_08/Results/0402_T1C/MGDA/secondimg_composed_constrained_trian_CE_55/model_result/"
                    
    ]
    pair_output_dirs=["/opt/chenxingru/Pineal_region/after_12_08/Results/0402_T1C/MGDA/secondimg_bbx_pair_orig_composed_constrained_trian_selfKL_5121/model_result/",
                 "/opt/chenxingru/Pineal_region/after_12_08/Results/0402_T1C/no_MGDA/secondimg_bbx_pair_orig_noSameRM_composed_constrained_trian_CE_55/model_result/",
                 "/opt/chenxingru/Pineal_region/after_12_08/Results/0402_T1C/no_MGDA/secondimg_bbx_pair_orig_composed_constrained_trian_CE_55/model_result/",
                 "/opt/chenxingru/Pineal_region/after_12_08/Results/0402_T1C/no_MGDA/secondimg_bbx_pair_orig_composed_constrained_trian_selfKL_5121/model_result/"
                 ]
    
    output_dirs=["/opt/chenxingru/Pineal_region/after_12_08/Results/0402_T1C/no_MGDA/secondimg_bbx_noSameRM_composed_constrained_trian_CE_55/model_result/",
                 "/opt/chenxingru/Pineal_region/after_12_08/Results/0402_T1C/no_MGDA/secondimg_bbx_composed_constrained_trian_CE_55/model_result/",
                 "/opt/chenxingru/Pineal_region/after_12_08/Results/0402_T1C/MGDA/secondimg_bbx_composed_constrained_trian_CE/model_result/"
                 ]
    Med_dir=["/opt/chenxingru/Pineal_region/after_12_08/Results/0402_T1C/no_MGDA/secondimg_tencentMed_composed_constrained_trian_CE_55/model_result/"  
    ]
    
    special=["/opt/chenxingru/Pineal_region/after_12_08/Results/0402_T1C/no_MGDA/noSameRM/secondimg_bbx_noSameRM_composed_constrained_trian_selfKL_5221/model_result/"]
    
    output_dirs=seg_NoneTrain_dirs
    use_secondimg=True
    
    print("output_dirs: ",output_dirs)
    for i, output_dir in enumerate(output_dirs):
    
        begin_time = time.time()
        json_path = os.path.join(output_dir, 'hyperparameter.json')
        jsf=open(json_path,'r')
        hyf=json.load(jsf)
        output_dir = hyf["output_dir"]
        lossfunc= hyf["lossfunc"]
        ps = hyf["ps"]
        model_name = hyf["model"]
        
        #use_radiomics=hyf["use_radiomics"]
        #noSameRM= hyf["noSameRM"]
        noSameRM=False
        use_radiomics=False
        

        batch_size = hyf["batch_size"]
        lambda_0,lambda_1,lambda_2 = hyf["lambda_0"],hyf["lambda_1"],hyf["lambda_2"]
        aug_form = hyf["aug_form"]
        
        
            
        # record metrics into a txt
        record_file_path= output_dir+model_name+'_'+lossfunc+ps+'_test_record.txt'
        record_file = open(record_file_path, "w")

        #load the training_data  and test_data (training_data will be splited later for cross_validation)

        test_data = test_file.select_dataset(data_idx=[i for i in range(len(test_file))], aug=False,use_secondimg=use_secondimg,noSameRM=noSameRM,usethird=usethird)



        test_loader= torch.utils.data.DataLoader(test_data
                                                    , batch_size = batch_size
                                                    , num_workers = opt.num_workers
                                                    , pin_memory = True
                                                    , drop_last = False
                                                    )


        
        loss_func_dict = { 'CE' : nn.CrossEntropyLoss().to(DEVICE)
                        , 'SelfKL' :  SelfKL(num_classes=opt.num_classes,lambda_0=lambda_0,lambda_1=lambda_1,lambda_2=lambda_2,lambda_3=lambda_2, CE_or_KL=opt.CE_or_KL)                    
                        }

        criterion = loss_func_dict[lossfunc]

        
        noemal_valid_func_dict={"None": normal_validate
                    ,"masked_constrained_train": normal_masked_constrained_validate
                    }
        validating = noemal_valid_func_dict[train_form]
        

        print(" ==========> All settled. testing is getting started...")


        # split the training_data into K fold with StratifiedKFold(shuffle = True)
        k=5
        splits=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)
        
        # just a record
        strafold_fp= open(output_dir+model_name+'_'+lossfunc+ps+"fold_split_case.txt", 'w')

        # to record metrics: the best_acc of k folds, best_acc of each fold
        fold_best_acc=-1
        best_fold=1
        best_epoch=0
        
        fold_record_valid_metrics,fold_record_matched_test_metrics,fold_aucs,test_fold_aucs=[],[],[],[]
        #================= begin to train, choose 1 of k folds as validation =================================
        print("======================== start test ================================================ \n")

        for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(total_file)),total_file.gety())):


            print("\n============ FOLD %d ============\n"%(fold),file=record_file)
            print('Fold {}'.format(fold))
            


            #valid_data(no augmentation: aug=False) & train_data(need augmentation:aug = True)
            print("Get valid set")
            vali_data=total_file.select_dataset(data_idx=val_idx, aug=False,use_secondimg=use_secondimg,noSameRM=noSameRM,usethird=usethird)
            print("Got train set")
            train_data=total_file.select_dataset(data_idx=train_idx, aug=True,aug_form=aug_form,use_secondimg=use_secondimg,noSameRM=noSameRM,usethird=usethird)

    
            if use_radiomics:
                radio_mean, radio_std = train_data.calc_own_radiomics_mean()
                vali_data.inject_other_mean(radio_mean, radio_std)
                test_data.inject_other_mean(radio_mean, radio_std)
                
                print("train_data. radio_mean: ",train_data.radio_mean)
                train_data.input_unit_radiomics_mean(radio_mean, radio_std)
                vali_data.input_unit_radiomics_mean(radio_mean, radio_std)        
                test_data.input_unit_radiomics_mean(radio_mean, radio_std)
                print("train_data. radio_mean: ",train_data.radio_mean)
                        
            train_loader = DataLoader(train_data, batch_size=batch_size, num_workers = opt.num_workers
                                                    , shuffle = True
                                                    , pin_memory = True
                                                    , drop_last = False)

            valid_loader = DataLoader(vali_data, batch_size=batch_size, num_workers = opt.num_workers
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
                num_seg_classes=2)
            elif model_name == "ResNet18":
                model = ResNet18()
            elif model_name == "ResNet10":
                model = ResNet10()
            elif model_name == "DIY_ResNet18":
                model = DIY_ResNet18()
            elif model_name == "DIY_ResNet10":
                model = DIY_ResNet10()
            else:
                print("[ERROR: ] Wrong model chosen\n")
            
            model = model.to(DEVICE)

            pps="_k-fold-sub-fold-"+str(fold)+"_"
            model, best_epoch = load_curr_best_checkpoint(model,out_dir = output_dir,model_name = model_name,pps = pps,lossfunc=lossfunc,ps = ps)
            #model.load_state_dict(best_statedict)        
            train_loss, train_losses, train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3]  =validating(valid_loader = train_loader
                                        , model = model
                                        , criterion = criterion
                                        , lossfunc_name=lossfunc
                                        , hyf=hyf)
            train_a,train_b,train_c = train_met.a, train_met.b, train_met.c
            train_acc,train_pre, train_rec, train_F1, train_spe =train_met.acc, train_met.pre,train_met.sen,train_met.F1, train_met.spe

            valid_loss, valid_losses, valid_acc, valid_accs, valid_met, [valid_loss_0,valid_loss_1,valid_loss_2,valid_loss_3]= validating(valid_loader = valid_loader
                                        , model = model
                                        , criterion = criterion
                                        , lossfunc_name=lossfunc
                                        , hyf=hyf)
            valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c
            valid_acc,valid_pre, valid_rec, valid_F1, valid_spe = valid_met.acc,valid_met.pre,valid_met.sen, valid_met.F1, valid_met.spe

            test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3] = validating(valid_loader = test_loader
                                        , model = model
                                        , criterion = criterion
                                        , lossfunc_name=lossfunc
                                        , hyf=hyf)
            test_a,test_b,test_c = test_met.a, test_met.b, test_met.c
            test_acc,test_pre,test_rec, test_F1,test_spe = test_met.acc,test_met.pre,test_met.sen, test_met.F1, test_met.spe

            train_auc = roc_auc_score(train_a.cpu(),train_b.cpu()[:,1])
            valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu()[:,1])
            test_auc = roc_auc_score(test_a.cpu(),test_b.cpu()[:,1])
            fold_aucs.append(valid_auc)
            test_fold_aucs.append(test_auc)
            sofar_valid_metrics = valid_met
            fold_record_valid_metrics.append(sofar_valid_metrics)
            fold_record_matched_test_metrics.append(test_met)

            recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_epoch",record_file = None,ps_thing=[best_fold,fold,valid_acc,valid_acc,best_epoch])
            recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_epoch",record_file = record_file,ps_thing=[best_fold,fold,valid_acc,valid_acc,best_epoch])



        os.system('echo " ================================= "')


        os.system('echo " === TRAIN mae mtc:{:.5f}" >> {}'.format(train_loss, output_path))

        # to get averaged metrics of k folds
        record_avg(fold_record_valid_metrics,fold_record_matched_test_metrics,record_file,fold_aucs,test_fold_aucs)

        # to print & record the best result of total  
        pps="_total_best_"
        
        model, best_epoch = load_curr_best_checkpoint(model,out_dir = output_dir,model_name = model_name,pps = pps,lossfunc=lossfunc,ps=ps)  
        train_loss, train_losses, train_acc, train_accs,train_met,[train_loss_0,train_loss_1,train_loss_2,train_loss_3]  =validating(valid_loader = train_loader
                                    , model = model
                                    , criterion = criterion
                                    , lossfunc_name=lossfunc
                                    , hyf=hyf)
        train_acc,train_pre, train_rec, train_F1, train_spe =train_met.acc, train_met.pre,train_met.sen,train_met.F1, train_met.spe

        valid_loss, valid_losses, valid_acc, valid_accs, valid_met,[valid_loss_0,valid_loss_1,valid_loss_2,valid_loss_3]= validating(valid_loader = valid_loader
                                    , model = model
                                    , criterion = criterion
                                    , lossfunc_name=lossfunc
                                    , hyf=hyf)
        valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c
        valid_acc,valid_pre, valid_rec, valid_F1, valid_spe = valid_met.acc,valid_met.pre,valid_met.sen, valid_met.F1, valid_met.spe

        test_loss, test_losses, test_acc, test_accs, test_met,[test_loss_0,test_loss_1,test_loss_2,test_loss_3] = validating(valid_loader = test_loader
                                    , model = model
                                    , criterion = criterion
                                    , lossfunc_name=lossfunc
                                    , hyf=hyf)
        test_a,test_b,test_c = test_met.a, test_met.b, test_met.c
        test_acc,test_pre,test_rec, test_F1,test_spe = test_met.acc,test_met.pre,test_met.sen, test_met.F1, test_met.spe


        train_auc = roc_auc_score(train_a.cpu(),train_b.cpu()[:,1])
        valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu()[:,1])
        test_auc = roc_auc_score(test_a.cpu(),test_b.cpu()[:,1])

        recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_fold",record_file = None,ps_thing=[best_fold,best_fold,fold_best_acc,fold_best_acc,best_epoch])
        recording(train_met,valid_met,test_met,train_loss,valid_loss, test_loss,train_auc,valid_auc,test_auc,record_form="best_fold",record_file = record_file,ps_thing=[best_fold,best_fold,fold_best_acc,fold_best_acc,best_epoch])


        torch.cuda.empty_cache()
        record_file.close()
        strafold_fp.close()

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

    