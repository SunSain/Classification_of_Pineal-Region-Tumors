import numpy as np
from itertools import combinations
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from math import log
#-----------------
import torch
import torch.nn as nn
#------------------

   
class Focal_Loss(nn.Module):
    def __init__(self,num_classes,weight=1.0,gamma=2):
        super(Focal_Loss,self).__init__()
        self.gamma=gamma
        self.weight=weight
        self.num_classes=num_classes
        
    def __call__(self,source,target,need_new_target=True):
        """
        preds:softmax输出结果
        labels:真实值
        ps:相比selfKL的多分类CE,此处使用了BCE,即考虑了同一组里面分到0的概率的大小并进行惩罚
        """
        eps=1e-7
        new_source=F.softmax(source,dim=1)
        #new_source = source
        if need_new_target:
            new_target=(F.one_hot(target,num_classes=source.size(1))).float()  
          
            new_target=torch.clamp(new_target,min=0.0001,max=1.0)   
            
        else:
            new_target = target
        
        ce=-1*torch.log(new_source+eps)*new_target
        ce2=-1*torch.log(1-new_source+eps)*(1-new_target)
        ce=ce+ce2
        #print("ce: ",torch.sum(ce)/ce.size(0))
        #print(ce)
        floss=torch.pow((1-new_source),self.gamma)*ce
        #print("floss: ",floss)
        floss=torch.sum(floss,dim=1)
        #print("total loss: ",floss)
        #print("torch.mean(floss): ",torch.mean(floss))
        return torch.mean(floss)

if __name__ =="__main__":
    M=Focal_Loss(num_classes=3)
    y = torch.Tensor([1,0,2,1]).long()
    q=torch.Tensor([[0.1,0.8,0.1],
        [0.5,0.3,0.2],
        [0.8,0.1,0.1],
        [0.1,0.1,0.8]])
    CE=nn.CrossEntropyLoss()
    ce_loss= CE(q,y) 
    print("CE: ",ce_loss)
    loss0 = M(q,y)