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

    
      
class Self_L0(nn.Module):

    def __init__(self, weight=1.0,size_average=True,num_classes=2,classes_weight=None):
        """
        初始化参数，因为要实现 torch.nn.CrossEntropyLoss 的两个比较重要的参数

        :param weight: 给予每个类别不同的权重
        :param size_average: 是否要对 loss 求平均
        
        [!!!]#5.20将KL 改为CE 用于临时训练，其它的（L1,L2,L3）未变.尚未改回

        """
        self.weight=weight
        self.size_average = size_average
        if classes_weight==None:
                self.classes_weight = [1.0,1.0,1.0] if num_classes==3 else [1.0,1.0]
        else:
            self.classes_weight=classes_weight
        
        
    def __call__(self, source, target): #a=A

        if self.weight==0:
            return 0.0
        new_target=(F.one_hot(target,num_classes=source.size(1))).float()  
          
        new_target=torch.clamp(new_target,min=0.0001,max=1.0) 
        
        weight = torch.Tensor(self.classes_weight).repeat(new_target.size(0),1).cuda()
        new_target = torch.mul(weight, new_target)  
        
        new_source=F.softmax(source,dim=1)
        loss=torch.mean(-torch.sum(new_target * torch.log(new_source), axis = -1))
        

        
        #loss=torch.mean(-torch.sum(target * torch.log(source) - target * torch.log(target), axis = -1))
        # original loss calc
        #loss=torch.mean(-torch.sum(target * torch.log(source), axis = -1))

        #loss *= self.weight
            
        return loss
        
