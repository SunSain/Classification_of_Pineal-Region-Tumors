import torch
import torch.nn as nn
import torch.nn.functional as F
    
import numpy as np
from itertools import combinations
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from math import log
        

class Weighted_CE(nn.Module):

    def __init__(self, weight=None, size_average=True,classes_weight=[1,1,1], n_classes=3):
        """
        初始化参数，因为要实现 torch.nn.CrossEntropyLoss 的两个比较重要的参数

        :param weight: 给予每个类别不同的权重
        :param size_average: 是否要对 loss 求平均
        : loss = plog(q), 所以 p*weigt 即赋予了权重
        """
        self.weight = weight
        self.size_average=size_average
        self.classes_weight=classes_weight
        self.n_classes = n_classes

        
    def __call__(self, source, target,need_new_target=True):
        """
        计算损失
        这个方法让类的实例表现的像函数一样，像函数一样可以调用

        :param input: (batch_size, C)，C是类别的总数
        :param target: (batch_size, 1)
        :return: 损失
        :typenum==0: L0
        """
        if need_new_target:
            new_target=(F.one_hot(target,num_classes=source.size(1))).float()  
          
            new_target=torch.clamp(new_target,min=0.0001,max=1.0)   
            
        else:
            new_target = target
            
        new_source=F.softmax(source,dim=1)

        weight = torch.Tensor(self.classes_weight).repeat(target.size(0),1).cuda()

        weighted_target = torch.mul(weight, new_target)

        loss=torch.mean(-torch.sum(weighted_target * torch.log(new_source), axis = -1))

        cre2 = nn.CrossEntropyLoss()
        loss2 = cre2(source, target)

        print("loss1: ",loss, " ; loss2: ",loss2)

        return loss
        

