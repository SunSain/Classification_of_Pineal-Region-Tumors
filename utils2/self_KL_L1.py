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
    
    
      
class Self_L1(nn.Module):

    def __init__(self, weight=1.0, size_average=True,num_classes=2,classes_weight=None):
        """
        初始化参数，因为要实现 torch.nn.CrossEntropyLoss 的两个比较重要的参数

        :param weight: 给予每个类别不同的权重
        :param size_average: 是否要对 loss 求平均

        [5/22]delete j, 张烁，刘子阳讨论
        """

        self.weight = weight
        self.size_average = size_average
        if classes_weight==None:
            self.classes_weight = [1.0,1.0,1.0] if num_classes==3 else [1.0,1.0]
        else:
            self.classes_weight=classes_weight

    def __call__(self, s, t): #a=A

        if self.weight==0:
            return 0.0
        
        new_target=(F.one_hot(t,num_classes=s.size(1))).float()  
        new_target=torch.clamp(new_target,min=1e-4,max=1.0) 
          
        weight = torch.Tensor(self.classes_weight).repeat(t.size(0),1).cuda()
        print("new_target: ",new_target)
        new_target = torch.mul(weight, new_target)
        print("weighted_new_target: ",new_target)
        
        new_source=F.softmax(s,dim=1)
        loss=1e-4
        for i in range(new_source.size(0)):
            for j in range(new_source.size(0)):
                a=-torch.sum(new_source[i] * torch.log(new_source[j]/new_source[i]), axis = -1)
                b=-torch.sum(new_target[i] * torch.log(new_target[j]/new_target[i]), axis = -1)
                #a=-torch.sum(new_source[i] * torch.log(new_source[j]), axis = -1)
                #b=-torch.sum(new_target[i] * torch.log(new_target[j]), axis = -1)
                #print("torch.abs(a-b): ",torch.abs(a-b))
                loss+=torch.abs(a-b)
                #print("loss_1: ",loss)
        loss/=(new_source.size(0) * new_source.size(0))


        """
        loss=0
        for i in range(source.size(0)):
            for j in range(source.size(0)):
                a=-torch.sum(source[i] * torch.log(source[j]/source[i]), axis = -1)
                b=-torch.sum(target[i] * torch.log(target[j]/target[i]), axis = -1)
                #a=-torch.sum(source[i] * torch.log(source[j]), axis = -1)
                #b=-torch.sum(target[i] * torch.log(target[j]), axis = -1)
                #print("torch.abs(a-b): ",torch.abs(a-b))
                loss+=torch.abs(a-b)
                #print("loss_1: ",loss)
        loss/=(source.size(0) * source.size(0))
        
        loss *= self.weight
        """
        return loss        
