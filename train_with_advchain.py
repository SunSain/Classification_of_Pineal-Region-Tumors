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
from utils2.for_adv_config import opt
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
from advchain.common.utils import random_chain
from advchain.augmentor.adv_affine import AdvAffine
from advchain.augmentor.adv_bias import AdvBias
from advchain.augmentor.adv_noise import AdvNoise
from advchain.augmentor.adv_morph import AdvMorph
from advchain.augmentor.adv_compose_solver import ComposeAdversarialTransformSolver

target_names = ['class 0', 'class 1']


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


n,c,h,w,d = 2,1,23,400,400
spatial_dims=3 ## for 2D, change to 3 for 3D
debug  = False
augmentor_bias= AdvBias(
                 spatial_dims=spatial_dims,
                 config_dict={'epsilon':0.3,
                 'control_point_spacing':[h//2,w//2,d//2],
                 'downscale':4,
                 'data_size':(n,c,h,w,d),
                 'interpolation_order':3,
                 'init_mode':'random',
                 'space':'log'},debug=debug)

augmentor_noise= AdvNoise( 
                spatial_dims=spatial_dims,
                config_dict={'epsilon':1,
                'xi':1e-6,
                 'data_size':(n,c,h,w,d)},
                 debug=debug)

augmentor_affine= AdvAffine(
                spatial_dims=spatial_dims,
                config_dict={
                'rot_x':10.0/180,
                'rot_y':10.0/180,
                 'rot_z':10.0/180,
                 'scale_x':0.1,
                 'scale_y':0.1,
                 'scale_z':0.1,
                 'shift_x':0.1,
                 'shift_y':0.1,
                 'shift_z':0.1,
                 'data_size':(n,c,h,w,d),
                 'forward_interp':'bilinear',
                 'backward_interp':'bilinear'},
                 debug=debug)

augmentor_morph= AdvMorph(
                spatial_dims=spatial_dims,
                config_dict=
                {'epsilon':1.5,
                 'data_size':(n,c,h,w,d),
                 'vector_size':[h//2,w//16,d//16],
                  'forward_interp':'bilinear',
                 'backward_interp':'bilinear'
                 }, 
                 image_padding_mode="zeros",
                 debug=debug)

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


def train(train_loader, model, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    maes = AverageMeter()


    for i, (img,_,target, _) in enumerate(train_loader):
        target = torch.from_numpy(np.expand_dims(target,axis=1))


        input = img.to(DEVICE)
        target = convert(target)


        transformation_family = [augmentor_noise,augmentor_bias,augmentor_morph,augmentor_affine]
        one_chain = random_chain(transformation_family.copy(),max_length=len(transformation_family))

        ## 2. set up a solver to optimize this chain
        solver = ComposeAdversarialTransformSolver(
            chain_of_transforms=one_chain,
            divergence_types = ['mse','contour'], ### you can also change it to 'kl'
            divergence_weights=[1.0,0.5],
            use_gpu= True,
            debug=False,
            if_norm_image=True, ## turn it on when intensity range needs to be preserved. Otherwise, turn it off. 
        )
        solver.init_random_transformation()
        ## 3. optimize transformation parameters to augment data and compute regularization loss
        adv_consistency_reg_loss = solver.adversarial_training(
        data = input,
        model = model,
        n_iter = 1, ## set up  the number of iterations for updating the transformation model.
        lazy_load = [False]*len(one_chain), 
        optimize_flags = [True]*len(one_chain),  ## you can also turn off adversarial training for one particular transformation
        step_sizes = 1) ## set up step size, you can also change it to a list of step sizes, so that different transformation have different step size

        ## 4. standard training 


        model.train()
        model.zero_grad()

        out = model(input)

        loss = criterion(out, target)
        total_loss = loss + opt.adv_w * adv_consistency_reg_loss ## for semi-supervised learning, it is better to perform linear ramp-up to schedule the weight w.
   

        #input.size(0) = batch_size 
        # the CE's ouput is averaged, so 
        losses.update(total_loss*input.size(0),input.size(0))

        pred, mae = get_corrects(output=out, target=target)

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
        total_loss.backward()
        print("loss:",loss," ; loss.grad: ",loss.grad,"; output is leaf? ",out.is_leaf," ; out.grad:",out.grad)

        optimizer.step()

 
    # total metrics'calcultion

    acc2, sen, pre, F1, spe = cal_metrics(CM)
  
    met= Metrics()
    met.update(a=0,b=0,c=0, auc=0, acc=acc2,sen=sen,pre=pre,F1=F1,spe=spe,CM=CM)

    return losses.avg,losses, maes.avg, maes, met
    

def validate(valid_loader, model, criterion):
    
    losses = AverageMeter()
    maes = AverageMeter()
    CM=0
    total_target=[]
    total_out=[]
    total_pred=[]

    model.eval() #because if allow model.eval, the bn wouldn't work, and the train_set'result would be different(model.train & model.eval)

    with torch.no_grad():
        for i, (img,_,target, _) in enumerate(valid_loader):
            target = torch.from_numpy(np.expand_dims(target,axis=1))
            target = convert(target)


            input = img.to(DEVICE)
            #input = torch.reshape(input, [input.shape[0],1,input.shape[1],input.shape[2],input.shape[3]])
            
            out = model(input)

            loss = criterion(out,target)
            losses.update(loss*input.size(0), input.size(0))

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

        
        return losses.avg,losses, maes.avg, maes, met

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


def main(output_path):

    begin_time = time.time()
    json_path = os.path.join(opt.output_dir, 'hyperparameter.json')
    with open(json_path,'w') as jsf:
        jsf.write(json.dumps(vars(opt)
                                , indent=4
                                , separators=(',',':')))
    # record metrics into a txt
    record_file_path= opt.output_dir+opt.model+'_'+opt.lossfunc+opt.ps+'_record.txt'
    record_file = open(record_file_path, "w")
    
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
    total_file = DIY_Folder(data_path=data_path)
    test_file = DIY_Folder(data_path=test_data_path,transform_dict=None)

    test_data = test_file.prepro_aug(data_idx=[i for i in range(len(test_file))], aug=False, transform_dict=None)



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
    fold_best_acc=-1
    best_fold=1
    best_epoch=0
    
    fold_record_valid_metrics=[]
    fold_record_matched_test_metrics=[]
    fold_accs=[]
    fold_f1s=[]
    fold_pres=[]
    fold_recalls=[]
    fold_aucs=[]
    fold_spes=[]
    test_fold_accs=[]
    test_fold_aucs=[]
    test_fold_f1s=[]
    test_fold_pres=[]
    test_fold_recalls=[]
    test_fold_aucs=[]
    test_fold_spes=[]

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
        vali_data=total_file.prepro_aug(data_idx=val_idx, aug=False, transform_dict=None)
        train_data=total_file.prepro_aug(data_idx=train_idx, aug=True, transform_dict=transform_dict,aug_form=opt.aug_form)
        
      
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


        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose = 1, patience=5, factor = 0.5) 
        # torch.optim.lr_scheduler.steplr
        
        early_stopping = EarlyStopping(patience = 100, verbose=True)
        history = {'train_loss': [], 'valid_loss': [],'test_loss':[],'train_acc':[],'valid_acc':[], 'test_acc':[],'train_auc':[],'test_auc':[],'valid_auc':[],'lr':[]}


        
        train_loss_box = []
        train_acc_box=[]

        valid_loss_box=[]
        valid_acc_box=[]


        t_acc_record=[]
        v_acc_record=[]
        test_acc_record=[]

        test_loss_box=[]
        test_acc_box=[]


        saved_metrics=[]
        saved_epoch=[]

        best_acc=-1
        sofar_valid_acc=-1
        sofar_valid_auc=-1
        best_statedict = model.state_dict()
        best_epoch=0


        for epoch in range(opt.epochs):

            train_loss, train_losses, train_acc, train_accs,train_met= train(train_loader = train_loader
                                                , model = model
                                                , criterion = criterion
                                                , optimizer = optimizer
                                                , epoch = epoch)

            #================================== every epoch's metrics record =================================================
            """
            train_loss, train_losses, train_acc, train_accs,train_met =validate(valid_loader = train_loader
                                        , model = model
                                        , criterion = criterion)
            """
            train_a,train_b,train_c = train_met.a, train_met.b, train_met.c
            train_pre, train_rec, train_F1, train_spe = train_met.pre,train_met.sen,train_met.F1, train_met.spe

            valid_loss, valid_losses, valid_acc, valid_accs, valid_met= validate(valid_loader = valid_loader
                                        , model = model
                                        , criterion = criterion)
            valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c
            valid_pre, valid_rec, valid_F1, valid_spe = valid_met.pre,valid_met.sen, valid_met.F1, valid_met.spe

            test_loss, test_losses, test_acc, test_accs, test_met = validate(valid_loader = test_loader
                                        , model = model
                                        , criterion = criterion)
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
            sum_writer.add_scalar(opt.model+str(fold)+sum_write_Mark+"train/loss", train_loss,epoch)
            sum_writer.add_scalar(opt.model+str(fold)+sum_write_Mark+"valid/loss", valid_loss,epoch)   
            sum_writer.add_scalar(opt.model+str(fold)+sum_write_Mark+"test/loss", test_loss,epoch)

            sum_writer.add_scalar(opt.model+str(fold)+sum_write_Mark+"train/acc", train_acc,epoch)
            sum_writer.add_scalar(opt.model+str(fold)+sum_write_Mark+"valid/acc", valid_acc,epoch)   
            sum_writer.add_scalar(opt.model+str(fold)+sum_write_Mark+"test/acc", test_acc,epoch)

            #sum_writer.add_scalar(opt.model+str(fold)+sum_write_Mark+"train/auc", train_auc,epoch)
            sum_writer.add_scalar(opt.model+str(fold)+sum_write_Mark+"valid/auc", valid_auc,epoch)   
            sum_writer.add_scalar(opt.model+str(fold)+sum_write_Mark+"test/auc", test_auc,epoch)      

            sum_writer.add_scalar(opt.model+str(fold)+sum_write_Mark+"train/F1", train_F1,epoch)
            sum_writer.add_scalar(opt.model+str(fold)+sum_write_Mark+"valid/F1", valid_F1,epoch)   
            sum_writer.add_scalar(opt.model+str(fold)+sum_write_Mark+"test/F1", test_F1,epoch)  

            sum_writer.add_scalar(opt.model+str(fold)+sum_write_Mark+"train/recall", train_rec,epoch)
            sum_writer.add_scalar(opt.model+str(fold)+sum_write_Mark+"valid/recall", valid_rec,epoch)   
            sum_writer.add_scalar(opt.model+str(fold)+sum_write_Mark+"test/recall", test_rec,epoch)  

            sum_writer.add_scalar(opt.model+str(fold)+sum_write_Mark+"train/spe", train_spe,epoch)
            sum_writer.add_scalar(opt.model+str(fold)+sum_write_Mark+"valid/spe", valid_spe,epoch)   
            sum_writer.add_scalar(opt.model+str(fold)+sum_write_Mark+"test/spe", test_spe,epoch)  

            is_best=False
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

            
            if valid_acc > best_acc:
                best_acc = valid_acc
                sofar_valid_acc = valid_acc
                sofar_valid_auc = valid_auc
                sofar_valid_metrics = valid_met
                sofar_test_metrics = test_met
                sofar_test_auc = test_auc

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
        fold_accs.append(best_acc)
        fold_aucs.append(sofar_valid_auc)
        fold_record_valid_metrics.append(sofar_valid_metrics)
        fold_record_matched_test_metrics.append(sofar_test_metrics)
        test_fold_aucs.append(sofar_test_auc)
         
    
        # to print & record the best result of this fold
        pps="_k-fold-sub-fold-"+str(fold)+"_"
        model = load_curr_best_checkpoint(model,out_dir = opt.output_dir,model_name = opt.model,pps = pps)
        #model.load_state_dict(best_statedict)        
        train_loss, train_losses, train_acc, train_accs,train_met =validate(valid_loader = train_loader
                                    , model = model
                                    , criterion = criterion)
        train_a,train_b,train_c = train_met.a, train_met.b, train_met.c
        train_pre, train_rec, train_F1, train_spe = train_met.pre,train_met.sen,train_met.F1, train_met.spe

        valid_loss, valid_losses, valid_acc, valid_accs, valid_met= validate(valid_loader = valid_loader
                                    , model = model
                                    , criterion = criterion)
        valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c
        valid_pre, valid_rec, valid_F1, valid_spe = valid_met.pre,valid_met.sen, valid_met.F1, valid_met.spe

        test_loss, test_losses, test_acc, test_accs, test_met = validate(valid_loader = test_loader
                                    , model = model
                                    , criterion = criterion)
        test_a,test_b,test_c = test_met.a, test_met.b, test_met.c
        test_pre,test_rec, test_F1,test_spe = test_met.pre,test_met.sen, test_met.F1, test_met.spe

        train_auc = roc_auc_score(train_a.cpu(),train_b.cpu()[:,1])
        valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu()[:,1])
        test_auc = roc_auc_score(test_a.cpu(),test_b.cpu()[:,1])

        print("\n[Fold end] best_fold %d & now_fold %d  |||| best_vali_acc %f  & now_vali_acc %f ==========================================\n"%(best_fold, fold, fold_best_acc,best_acc ))
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


#====================================================================================================================================================
        print("\n[Fold end] best_fold %d & now_fold %d  |||| best_vali_acc %f  & now_vali_acc %f ==========================================\n"%(best_fold, fold, fold_best_acc,best_acc ),file=record_file)
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

        plt.subplot(2,3,1)
        plt.plot(history['train_loss'],'r--',label='t_loss')
        plt.plot(history['valid_loss'],'b--',label='v_loss')
        plt.title("train_valid_loss")

        plt.subplot(2,3,2)
        plt.plot(history['train_loss'],'g--',label='t_loss')
        plt.plot(history['train_acc'],'y--',label='t_acc')
        plt.title("train_loss_acc")

        plt.subplot(2,3,3)
        plt.plot(history['train_acc'],'r--',label='t_acc')
        plt.plot(history['valid_acc'],'b--',label='v_acc')
        plt.title("train_valid_acc")


        plt.subplot(2,3,4)
        plt.plot(history['valid_loss'],'g--',label='v_loss')
        plt.plot(history['valid_acc'],'y--',label='v_acc')
        plt.title("valid_acc_loss")

        plt.subplot(2,3,5)
        plt.plot(history['lr'],'b--',label='learning_rate')
        plt.title("learning_rate")

        plt.savefig(opt.pic_output_dir+opt.model+'_'+opt.lossfunc+opt.ps+'_k-fold-'+str(fold)+'_loss_acc.png')

        #plt.show()
        plt.close()


        foldperf['fold{}'.format(fold+1)] = history  

    """
    model_path = best_model_path = opt.output_dir+opt.model+'_'+opt.lossfunc+opt.ps+'_best_model.pth.tar'
    checkpoint_path = opt.output_dir+opt.model+'_'+opt.lossfunc+opt.ps+'_checkpoint.pth.tar'
    print("save_model_path: ",best_model_path)
    print("checkpoint_path: ",checkpoint_path)
    torch.save(model.state_dict(), checkpoint_path)
    torch.save(model,model_path)
    """

  

    os.system('echo " ================================= "')


    os.system('echo " === TRAIN mae mtc:{:.5f}" >> {}'.format(train_loss, output_path))

    # to get averaged metrics of k folds
    for i, metrics in enumerate(fold_record_valid_metrics):
        F1, pre, rec,spe = metrics.F1, metrics.pre, metrics.sen, metrics.spe
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

    # to print & record the best result of total
    model.load_state_dict(fold_best_statedict)        
    train_loss, train_losses, train_acc, train_accs,train_met =validate(valid_loader = train_loader
                                , model = model
                                , criterion = criterion)
    train_a,train_b,train_c = train_met.a, train_met.b, train_met.c
    train_pre, train_rec, train_F1, train_spe = train_met.pre,train_met.sen,train_met.F1, train_met.spe

    valid_loss, valid_losses, valid_acc, valid_accs, valid_met= validate(valid_loader = valid_loader
                                , model = model
                                , criterion = criterion)
    valid_a,valid_b,valid_c = valid_met.a,valid_met.b,valid_met.c
    valid_pre, valid_rec, valid_F1, valid_spe = valid_met.pre,valid_met.sen, valid_met.F1, valid_met.spe

    test_loss, test_losses, test_acc, test_accs, test_met = validate(valid_loader = test_loader
                                , model = model
                                , criterion = criterion)
    test_a,test_b,test_c = test_met.a, test_met.b, test_met.c
    test_pre,test_rec, test_F1,test_spe = test_met.pre,test_met.sen, test_met.F1, test_met.spe

    train_auc = roc_auc_score(train_a.cpu(),train_b.cpu()[:,1])
    valid_auc = roc_auc_score(valid_a.cpu(),valid_b.cpu()[:,1])
    test_auc = roc_auc_score(test_a.cpu(),test_b.cpu()[:,1])
    print("\n============ Total End ============\n",file=record_file)
    print("\n[Total end] best_fold %d & best_vali_acc %f ==========================================\n"%(best_fold, fold_best_acc ),file=record_file)
    print("\n================================== [Fold: %d [Train] loss: %f] ==========================================\n"%(best_fold, train_loss),file=record_file)                                                                                                    
    print(classification_report(train_a.cpu(), train_c.cpu(), target_names=target_names, labels=[0,1]),file=record_file)
    print("\n================================== [Fold: %d [Valid] loss:%f ] ==========================================\n"%(best_fold, valid_loss),file=record_file) 
    print(classification_report(valid_a.cpu(), valid_c.cpu(), target_names=target_names, labels=[0,1]),file=record_file)
    print("\n================================== [Fold: %d [Test] loss: %f] ==========================================\n"%(best_fold, test_loss),file=record_file) 
    print(classification_report(test_a.cpu(), test_c.cpu(), target_names=target_names, labels=[0,1]),file=record_file)

    print("\n================================== [Fold: %d] ==========================================\n"%(best_fold),file=record_file)
    print("[train]: %f \t[valid]: %f \t[test]: %f \n"%(train_auc, valid_auc, test_auc),file=record_file)


    print("\n================================== [Fold: %d acc at epoch %d ] ==========================================\n"%(fold,best_epoch),file=record_file)
    print("[train]: %f \t[valid]: %f \t[test]: %f \n"%(train_acc, valid_acc, test_acc),file=record_file)
    print("\n================================== [Fold: %d F1 best at epoch %d ] ==========================================\n"%(fold,best_epoch),file=record_file)
    print("[train]: %f \t[valid]: %f \t[test]: %f \n"%(train_F1, valid_F1, test_F1),file=record_file)
    print("\n================================== [Fold: %d spe best at epoch %d ] ==========================================\n"%(fold,best_epoch),file=record_file)
    print("[train]: %f \t[valid]: %f \t[test]: %f \n"%(train_spe, valid_spe, test_spe),file=record_file)       
    print("\n================================== [Fold: %d sen best at epoch %d ] ==========================================\n"%(fold,best_epoch),file=record_file)
    print("[train]: %f \t[valid]: %f \t[test]: %f \n"%(train_rec, valid_rec, test_rec),file=record_file)

#==================================================================================================================================
    print("\n[Total end] best_fold %d & best_vali_acc %f ==========================================\n"%(best_fold, fold_best_acc ))
    print("\n================================== [Fold: %d [Train]] ==========================================\n"%(best_fold))                                                                                                    
    print(classification_report(train_a.cpu(), train_c.cpu(), target_names=target_names, labels=[0,1]))
    print("\n================================== [Fold: %d [Valid] ] ==========================================\n"%(best_fold)) 
    print(classification_report(valid_a.cpu(), valid_c.cpu(), target_names=target_names, labels=[0,1]))
    print("\n================================== [Fold: %d [Test] ] ==========================================\n"%(best_fold)) 
    print(classification_report(test_a.cpu(), test_c.cpu(), target_names=target_names, labels=[0,1]))

    print("\n================================== [Fold: %d] ==========================================\n"%(best_fold))
    print("[train]: %f \t[valid]: %f \t[test]: %f \n"%(train_auc, valid_auc, test_auc))
    print("\n================================== [Fold: %d acc at epoch %d ] ==========================================\n"%(fold,best_epoch))
    print("[train]: %f \t[valid]: %f \t[test]: %f \n"%(train_acc, valid_acc, test_acc))
    print("\n================================== [Fold: %d F1 best at epoch %d ] ==========================================\n"%(fold,best_epoch))
    print("[train]: %f \t[valid]: %f \t[test]: %f \n"%(train_F1, valid_F1, test_F1))
    print("\n================================== [Fold: %d spe best at epoch %d ] ==========================================\n"%(fold,best_epoch))
    print("[train]: %f \t[valid]: %f \t[test]: %f \n"%(train_spe, valid_spe, test_spe))       
    print("\n================================== [Fold: %d sen best at epoch %d ] ==========================================\n"%(fold,best_epoch))
    print("[train]: %f \t[valid]: %f \t[test]: %f \n"%(train_rec, valid_rec, test_rec))



    torch.cuda.empty_cache()
    sum_writer.close()
    strafold_fp.close()
    record_file.close()
    print("training end========")




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

    