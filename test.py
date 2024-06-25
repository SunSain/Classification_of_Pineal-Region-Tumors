<<<<<<< HEAD
from genericpath import exists
import os,torch,json
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

import torch.optim as optim

from load_data_binary import DIY_Folder
from utils2.avgmeter import AverageMeter
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix,precision_recall_fscore_support

import matplotlib.pyplot as plt
from model.resnet34_3d import ResNet34
from model.resnet18_3d import ResNet18
from model.resnet10_3d import ResNet10
from model.SEResnet import seresnet18
from model.vgg11 import VGG11_bn
from model.vgg13 import VGG13_bn
from model.Inception2 import Inception2
from model.vgg16 import VGG16_bn

from model.diy_resnet18 import DIY_ResNet18
from model.diy_resnet10 import DIY_ResNet10

from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from sklearn.model_selection import StratifiedKFold


from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']

import torchio as tio
from model import tencent_resnet


if torch.cuda.is_available():
    torch.cuda.set_device(3)
    DEVICE=torch.device('cuda')
else:
    DEVICE=torch.device('cpu')
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
print('DEVICE: ',DEVICE)
torch.cuda.empty_cache()



def metric(output, target):
    target = target.data.cpu()
    pred = output.cpu()  
    #print("pred0: ",pred," ; target: ",target)
    pred=pred.max(1,keepdim=True)[1]
    pred=convert_target(pred).cpu()
    correct=pred.eq(target).sum().item()
    #print("output: ",output)
    #print("pred: ",pred)
    #print("target: ",target)
    #print("corret: ",correct)
    return pred,correct

def convert_target(original):
    target=torch.Tensor(len(original))
    for i in range(len(original)):
        target[i]=original[i][0]
    target=target.type(torch.LongTensor).to(DEVICE)
    return target

def metric_1(output, target):
    target = target.data.cpu()
    pred = output.cpu()  

    correct=0
    pred=pred.max(1,keepdim=True)[1]
    pred=convert_target(pred).cpu()

    #print("pred0: ",pred)
    #print("target: ",target)
    for i ,t in enumerate(target):
        #print("i: ",i, " ; t: ",t," ;pred[i]: ",pred[i]," ;correct: ",correct)
        if t==0: continue
        if pred[i]==1:
            correct+=1
    #print("output: ",output)
    #print("pred: ",pred)
    #print("target: ",target)
    #print("corret: ",correct)
    return pred,correct


def get_every_class_result(total_target, total_out,n_class=3):

    a1=[]
    a2=[]
    a3=[]
    b1=[]
    b2=[]
    b3=[]
    c1=[]
    c2=[]
    c3=[]
    target_number_1=0
    target_number_2=0
    target_number_3=0
    for i, target in enumerate(total_target):
        outlist=total_out[i]
        out=outlist[target]
        if target==0:
            target_number_1+=1
            a1.append(1)
            b1.append([1-out, out])

            a2.append(0)
            b2.append([out,1-out])

            a3.append(0)
            b3.append([out,1-out])

        elif target ==1:
            target_number_2+=1
            a2.append(1)
            b2.append([1-out, out])

            a1.append(0)
            b1.append([out,1-out])

            a3.append(0)
            b3.append([out,1-out])

        else:
            target_number_3+=1
            a3.append(1)
            b3.append([1-out, out])

            a2.append(0)
            b2.append([out,1-out])

            a1.append(0)
            b1.append([out,1-out])
    return a1,a2,a3,b1,b2,b3,target_number_1,target_number_2,target_number_3

def draw_confusion_matrix(cf_matrix, save_path):
    
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cf_matrix.shape[0]):
        for j in range(cf_matrix.shape[1]):
            ax.text(x=j, y=i,s=cf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    plt.savefig(save_path)
    plt.close()



def validate(valid_loader, model, criterion):
    
    losses = AverageMeter()
    maes = AverageMeter()

    #model.eval()

    total_target=[]
    total_out=[]
    total_pred=[]
    number=0
    with torch.no_grad():
        for i, (img,_,target, _) in enumerate(valid_loader):
            target = torch.from_numpy(np.expand_dims(target,axis=1))
            target = convert_target(target)
            input = img.to(DEVICE)
            input = torch.reshape(input, [input.shape[0],1,input.shape[1],input.shape[2],input.shape[3]])
            
            #print("input.new_size: ",input.size())
            out = model(input)
            total_target.extend(target)
            loss = criterion(out,target)

            losses.update(loss, input.size(0))
            pred,mae = metric(output=out, target=target)
            maes.update(mae, input.size(0)) 
            number+=input.size(0)
            #print("number: ",number)

            out=torch.softmax(out,dim=1)

            total_out.extend(out.cpu().numpy())

            total_pred.extend(pred.cpu().numpy())

            #auc = roc_auc_score(target.cpu(),out.cpu(),multi_class='ovr')
            #print("auc: ",auc)



        a=torch.tensor(total_target)
        b=torch.tensor(total_out)
        c=torch.tensor(total_pred)
        #print("target: ",a)
        #print("out: ",b)
        #print("pred: ",c)

        cf_matrix=confusion_matrix(a, c,labels=[0, 1])

        p_r_f_matrix=precision_recall_fscore_support(a, c,labels=[0, 1])
        total_p_r_f_matrix=precision_recall_fscore_support(a, c,labels=[0, 1])

        positive_class=1
       
        _,mae = metric(output=b, target=a)

        """
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr[positive_class], tpr[positive_class], _ = roc_curve(a.cpu(), b.cpu()[:, positive_class])
        roc_auc[positive_class] = auc(fpr[positive_class], tpr[positive_class])
        print(roc_auc)
        """


        auc = roc_auc_score(a.cpu(),b.cpu()[:,positive_class])
        #print(" ; auc2: ",auc2)  
        #print("mae: ",mae," ; number: ",number) 
        acc= mae/number    

        print(classification_report(a.cpu(), c.cpu(), target_names=target_names))

        return total_p_r_f_matrix,p_r_f_matrix,cf_matrix,losses.avg,losses, maes.avg,maes,auc,acc,a,c

def main(output_path):

    json_path = os.path.join(opt.output_dir, 'hyperparameter.json')
    with open(json_path,'w') as jsf:
        jsf.write(json.dumps(vars(opt)
                                , indent=4
                                , separators=(',',':')))
    excel_path="/opt/chenxingru/Pineal_region/Pineal_0410.xlsx/"
    data_path=opt.data_path

    print("=========== start testing the brain age estimation model =========== \n")
    print(" ==========> Using {} processes for data loader.".format(opt.num_workers))

    transform_dict = {
        tio.RandomElasticDeformation(),
        tio.RandomNoise(),
        tio.RandomFlip(),
        tio.RandomBlur(),
        tio.RandomAffine(),
        tio.RandomMotion(),
        tio.RandomSwap()
        }
    
    data = DIY_Folder(data_path=data_path,transform_dict=transform_dict)
    print("======================== start train the brain age estimation model ================================================ \n")
   

    test_data_path = opt.testdata_path
    test_data = DIY_Folder(data_path=test_data_path,transform_dict=None)

    test_loader= torch.utils.data.DataLoader(test_data
                                                , batch_size = opt.batch_size
                                                , num_workers = opt.num_workers
                                                , pin_memory = True
                                                , drop_last = False
                                                )
    print("len__data: ",len(data))
    print("len_test_data: ",len(test_data))
    print("len_test_data2: ",len(test_loader))

    count=0
    for i,(img,_,target, _) in enumerate(test_loader):
        count+=1
    print("len_test_data3: ",count)

    pps1="_k-fold-sub-fold-"+str(0)+"_"
    pps2="_total_best_"
    pps3=""
    pps=pps2
    model_file_path = opt.output_dir + opt.model +'_'+opt.lossfunc+opt.ps+pps+"_best_model.pth.tar"
    print("model_file_path: ",model_file_path)


    if opt.model == "ResNet34":
        model = ResNet34()
    elif opt.model == "ResNet18":
        model = ResNet18()
    elif opt.model == "ResNet10":
        model = ResNet10()
    elif opt.model == "VGG11_bn":
        model = VGG11_bn()
    elif opt.model == "Inception2":
        model = Inception2()
    elif opt.model == "VGG16_bn":
        model = VGG16_bn()
    elif opt.model == "seresnet18":
        model = seresnet18()
    elif opt.model == "DIY_ResNet18":
        model = DIY_ResNet18()
    elif opt.model == "DIY_ResNet10":
        model = DIY_ResNet10()
    elif opt.model == "tencent_resnet34":
        model = tencent_resnet.resnet34(sample_input_W=75,
                sample_input_H=80,
                sample_input_D=75,
                shortcut_type='A',
                no_cuda=False,
                num_seg_classes=2)
        
        net_dict = model.state_dict()
        checkpoint =  torch.load(opt.output_dir + opt.model +'_'+opt.lossfunc+opt.ps+pps+"_checkpoint.pth.tar")    
        print("net_dict.keys(): ",net_dict.keys())
        pretrain_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in net_dict.keys()}
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)
        print("check_point_path: ",opt.output_dir + opt.model +'_'+opt.lossfunc+opt.ps+pps+"_checkpoint.pth.tar")
  
    elif opt.model == "tencent_resnet18":
        model = tencent_resnet.resnet18(sample_input_W=75,
            sample_input_H=80,
            sample_input_D=75,
            shortcut_type='A',
            no_cuda=False,
            num_seg_classes=2)

    elif opt.model == "tencent_resnet50":
        model = tencent_resnet.resnet50(sample_input_W=75,
            sample_input_H=80,
            sample_input_D=75,
            shortcut_type='B',
            no_cuda=False,
            num_seg_classes=2)

    elif opt.model == "tencent_resnet10":
        model = tencent_resnet.resnet10(sample_input_W=75,
            sample_input_H=80,
            sample_input_D=75,
            shortcut_type='B',
            no_cuda=False,
            num_seg_classes=2)
        
        
    else:
        print("[ERROR: ] Wrong model chosen\n")

    model=model.to(DEVICE)

    net_dict = model.state_dict()
    checkpoint = torch.load(opt.output_dir + opt.model +'_'+opt.lossfunc+opt.ps+pps+"_checkpoint.pth.tar")
    print("net_dict.keys(): ",net_dict.keys())
    pretrain_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in net_dict.keys()}
    net_dict.update(pretrain_dict)
    model.load_state_dict(net_dict)


    print("best_fold: ",checkpoint['fold'])

    loss_func_dict = { 'CE' : nn.CrossEntropyLoss().to(DEVICE)
                      , 'KL' : nn.KLDivLoss().to(DEVICE)
                     }

    criterion = loss_func_dict['CE']
    print("Chosen criterion: ",criterion)

    print(" ==========> All settled. Testing is getting started...")


    k=5
    splits=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)
    foldperf={}
    fold_best_acc=-1

    print("len(data): ",len(data))
    print("np.arange(len(data)): ",np.arange(len(data)))
    print("len(y): ",len(data.gety()))


    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(data)),data.gety())):

        print('Fold {}'.format(fold + 1))
        #print("(train_idx,val_idx): ",train_idx," \n================\n",val_idx)
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(data, batch_size=opt.batch_size, sampler=train_sampler, num_workers = opt.num_workers
                                                , pin_memory = True
                                                , drop_last = False)
        valid_loader = DataLoader(data, batch_size=opt.batch_size, sampler=valid_sampler, num_workers = opt.num_workers
                                                , pin_memory = True
                                                , drop_last = False)
           
         
        print("len_train_data: ",len(train_loader))
        print("len_valid_data: ",len(valid_loader))

        count =0                
        for i,(img,_,target, _) in enumerate(train_loader):
            count+=1
        print("train_item: ",count)

        count=0
        for i,(img,_,target, _) in enumerate(valid_loader):
             count+=1
        print("valid_item: ",count)

        count=0
        for i,(img,_,target, _) in enumerate(test_loader):
            count+=1
        print("test_item: ",count)
        
        history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
        


        train_total_prf,train_p_r_f_matrix,train_confusion_matrix,train_loss, train_losses, train_acc, train_accs,train_auc,train_cc, train_target,train_out = validate(valid_loader = train_loader
                                            , model = model
                                            , criterion = criterion)

        valid_total_prf,valid_p_r_f_matrix,valid_confusion_matrix,valid_loss, valid_losses, valid_acc, valid_accs, valid_auc,valid_cc, valid_target, valid_out= validate(valid_loader = valid_loader
                                    , model = model
                                    , criterion = criterion)

        test_total_prf,test_p_r_f_matrix,test_confusion_matrix,test_loss, test_losses, test_acc, test_accs,test_auc, test_cc, test_target, test_out= validate(valid_loader = test_loader
                                    , model = model
                                    , criterion = criterion)




        print("Fold:{}/{} AVG Training Loss:{:.3f} AVG Valid Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Valid Acc {:.2f} % AVG Test Acc {:.2f} % AVG Train AUC {:.2f} % AVG Valid AUC {:.2f} % AVG Test AUC {:.2f} %".format(fold + 1,
                                                                                                            opt.epochs,
                                                                                                            train_loss,
                                                                                                            valid_loss,
                                                                                                            test_loss,
                                                                                                            train_acc,
                                                                                                            valid_acc,
                                                                                                            test_acc,
                                                                                                            train_auc,
                                                                                                            valid_auc,
                                                                                                            test_auc))
        print("\n================================== [Train] ==========================================\n")                                                                                                    
        print(classification_report(train_target.cpu(), train_out.cpu(), target_names=target_names, labels=[0,1]))
        print("\n================================== [Valid] ==========================================\n") 
        print(classification_report(valid_target.cpu(), valid_out.cpu(), target_names=target_names, labels=[0,1]))
        print("\n================================== [Test] ==========================================\n") 
        print(classification_report(test_target.cpu(), test_out.cpu(), target_names=target_names, labels=[0,1]))
        b=classification_report(test_target.cpu(), test_out.cpu(), target_names=target_names, labels=[0,1])

        train_save_path=opt.pic_output_dir+opt.model+'_'+opt.lossfunc+opt.ps+'_k-fold-'+str(fold)+'_Train_Confusion_matrix.png'
        valid_save_path=opt.pic_output_dir+opt.model+'_'+opt.lossfunc+opt.ps+'_k-fold-'+str(fold)+'_Valid_Confusion_matrix.png'
        test_save_path=opt.pic_output_dir+opt.model+'_'+opt.lossfunc+opt.ps+'_k-fold-'+str(fold)+'_Test_Confusion_matrix.png'

        draw_confusion_matrix(train_confusion_matrix,train_save_path)
        draw_confusion_matrix(valid_confusion_matrix,valid_save_path)
        draw_confusion_matrix(test_confusion_matrix,test_save_path)


        print("f1-score: ",b[2])

        print("train_auc:",train_auc)
        print("valid_auc",valid_auc)
        print("test_auc",test_auc)

        print("train_loss: ",train_loss)
        print("valid_loss: ",valid_loss)
        print("test_loss: ",test_loss)

        print("train_acc: ",train_acc)
        print("valid_acc: ",valid_acc)
        print("test_acc: ",test_acc)

        print("train_acc2: ",train_cc)
        print("valid_acc2: ",valid_cc)
        print("test_acc2: ",test_cc)

        torch.cuda.empty_cache()
        history['train_loss'].append(train_loss.cpu().detach().numpy())
        history['test_loss'].append(test_loss.cpu().detach().numpy())
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

    """
    print("======================== VALIDATION ================================================ \n")

    valid_total_prf,valid_p_r_f_matrix,valid_confusion_matrix,valid_loss, valid_losses, valid_acc, valid_accs, valid_auc,valid_cc, valid_mea1,valid_mea2,valid_mea3,valid_auc1, valid_auc2,valid_auc3= validate(valid_loader = valid_loader
                                     , model = model
                                     , criterion = criterion)
    print("======================== TEST ================================================ \n")

    test_total_prf,test_p_r_f_matrix,test_confusion_matrix,test_loss, test_losses, test_acc, test_accs,test_auc, test_cc,test_mea1, test_mea2,test_mea3,test_auc1,test_auc2, test_auc3= validate(valid_loader = test_loader
                                     , model = model
                                     , criterion = criterion)
    """
                                    


    print("train_confusion_matrix: ",train_confusion_matrix)
    print("valid_confusion_matrix: ",valid_confusion_matrix)
    print("test_confusion_matrix: ",test_confusion_matrix)
    print("train_p_r_f_matrix: ",train_p_r_f_matrix)
    print("valid_p_r_f_matrix: ",valid_p_r_f_matrix)
    print("test_p_r_f_matrix: ",test_p_r_f_matrix)
    print("train_total_prf: ",train_total_prf)
    print("valid_total_prf: ",valid_total_prf)
    print("test_total_prf: ",test_total_prf)


    os.system('echo " ================================= "')

    os.system('echo " === TEST mae mtc:{:.5f}" >> {}'.format(train_loss, output_path))


    print("train_auc:",train_auc)
    print("valid_auc",valid_auc)
    print("test_auc",test_auc)

    print("train_loss: ",train_loss)
    print("valid_loss: ",valid_loss)
    print("test_loss: ",test_loss)

    print("train_acc: ",train_acc)
    print("valid_acc: ",valid_acc)
    print("test_acc: ",test_acc)

    print("train_acc2: ",train_cc)
    print("valid_acc2: ",valid_cc)
    print("test_acc2: ",test_cc)

    torch.cuda.empty_cache()




if __name__ == "__main__":
    output_path = os.path.join(opt.output_dir, 'result')
    print("output_path: ",output_path)

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    
    print("=> testing beigin. \n")
    os.system('echo "test {}"  >>  {}'.format(datetime.datetime.now(),output_path))

    main(output_path)

    
=======

import numpy as np
fold_test_fpr= [[0., 0., 0., 1.], [0., 0., 1., 1.], [0. , 0. , 0.5, 0.5, 1. ]]
fold_test_tpr= [[0. , 0.5, 1. , 1. ], [0. , 0.5, 0.5, 1. ], [0. , 0.5, 0.5, 1. , 1. ]]
test_fold_aucs=  [1.0, 0.5, 0.75]

print("mean(fold_test_fpr): ",np.mean(fold_test_fpr,axis=1))
print("mean(fold_test_tpr): ",np.mean(fold_test_tpr,axis=0))
print("mean(test_fold_aucs): ",np.mean(test_fold_aucs))
>>>>>>> 3a4a4f2 (20240625-code)
