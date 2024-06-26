
from itertools import combinations
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
     
class SelfKL(nn.Module):

    def __init__(self, weight=None, size_average=True,num_classes=0,lambda_0=0.0,lambda_1=0.0,lambda_2=0.0,lambda_3=0.0,CE_or_KL=True,classes_weight=None):
        """
        初始化参数，因为要实现 torch.nn.CrossEntropyLoss 的两个比较重要的参数

        :param weight: 给予每个类别不同的权重
        :param size_average: 是否要对 loss 求平均
        """

        self.weight = weight
        self.size_average = size_average
        
        self.num_classes=num_classes
        self.lambda_0=lambda_0
        self.lambda_1=lambda_1
        self.lambda_2=lambda_2
        self.lambda_3=lambda_3
        if classes_weight==None:
            self.classes_weight = [1.0,1.0,1.0] if num_classes==3 else [1.0,1.0]
        else:
            self.classes_weight=classes_weight


        
    def __call__(self, source, target, need_new_target=True):
        """
        计算损失
        这个方法让类的实例表现的像函数一样，像函数一样可以调用

        :param input: (batch_size, C),C是类别的总数
        :param target: (batch_size, 1)
        :return: 损失
        :typenum==0: L0
        :typenum==1: L1
        :typenum==2: L2
        :typenum==3: L3
        """
        #print("target: ",target)
        if need_new_target:
            new_target=(F.one_hot(target,num_classes=source.size(1))).float()  
          
            new_target=torch.clamp(new_target,min=0.0001,max=1.0)   
            
        else:
            new_target = target
        
        weight = torch.Tensor(self.classes_weight).repeat(target.size(0),1).cuda()
        print("new_target: ",new_target)
        new_target = torch.mul(weight, new_target)
        print("weighted_new_target: ",new_target)
        
        new_source=F.softmax(source,dim=1)

        L0= Self_L0(self.lambda_0)
        L1= Self_L1(self.lambda_1)
        L2= Self_L2(self.lambda_2)
        L3= Self_L3(self.lambda_3)
        
        loss_0=0
        loss_1=0
        loss_2=0
        loss_3=0
        
        loss_0=L0(new_source,new_target)
        loss_1=L1(new_source,new_target)
        #print("[SKL: loss_1]: ",loss_1, " ;loss_0: ",loss_0)
        loss_2=L2(new_source,new_target)
        loss_3=L3(new_source,new_target)       

        #total_loss=Variable(torch.tensor(loss_0+loss_1+loss_2+loss_3),requires_grad=True)
        #total_loss=torch.add(loss_0,loss_1,loss_2,loss_3).requires_grad_(True)
        #print("total_loss: ",total_loss)     
        #print("add_loss: "+str(loss_0+loss_1+loss_2+loss_3))  
        return torch.add(torch.add(loss_0,loss_1),torch.add(loss_2,loss_3)).requires_grad_(True), loss_0,loss_1,loss_2,loss_3


class Self_L0(nn.Module):
    
    def __init__(self, weight=1.0,size_average=True):
        """
        初始化参数，因为要实现 torch.nn.CrossEntropyLoss 的两个比较重要的参数

        :param weight: 给予每个类别不同的权重
        :param size_average: 是否要对 loss 求平均
        
        [!!!]#5.20将KL 改为CE 用于临时训练，其它的（L1,L2,L3）未变.尚未改回

        """
        self.weight=weight
        self.size_average = size_average
        
        
    def __call__(self, source, target): #a=A

        if self.weight==0:
            return 0.0

        loss=torch.mean(-torch.sum(target * torch.log(source), axis = -1))
        

        
        #loss=torch.mean(-torch.sum(target * torch.log(source) - target * torch.log(target), axis = -1))
        # original loss calc
        #loss=torch.mean(-torch.sum(target * torch.log(source), axis = -1))

        #loss *= self.weight
            
        return loss
class Self_L1(nn.Module):
    
    def __init__(self, weight=1.0, size_average=True):
        """
        初始化参数，因为要实现 torch.nn.CrossEntropyLoss 的两个比较重要的参数

        :param weight: 给予每个类别不同的权重
        :param size_average: 是否要对 loss 求平均

        [5/22]delete j, 张烁，刘子阳讨论
        """

        self.weight = weight
        self.size_average = size_average

    def __call__(self, new_source, new_target): #a=A

        if self.weight==0:
            return 0.0
        
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
        

       
class Self_L2(nn.Module):

    def __init__(self, weight=1.0, size_average=True):
        """
        初始化参数，因为要实现 torch.nn.CrossEntropyLoss 的两个比较重要的参数

        :param weight: 给予每个类别不同的权重
        :param size_average: 是否要对 loss 求平均
        """

        self.weight = weight
        self.size_average = size_average
 
        
    
    
    def __call__(self, new_source, new_target): #a=A

        if self.weight==0:
            return 0.0

        loss=1e-4
        for i in range(new_source.size(0)):
            for j in range(new_source.size(0)):
                a= -torch.sum(new_target[j] * torch.log(new_source[i]/new_target[j]), axis = -1)
                b= -torch.sum(new_target[j] * torch.log(new_target[i]/new_target[j]), axis = -1)
                loss+=torch.abs(a-b)
        #print("loss1: ",loss)
        loss/=(new_source.size(0) * new_source.size(0))

        
        """

        loss=0
        for i in range(source.size(0)):
            for j in range(source.size(0)):
                a= -torch.sum(target[j] * torch.log(source[i]/target[j]), axis = -1)
                b= -torch.sum(target[j] * torch.log(target[i]/target[j]), axis = -1)
                loss+=torch.abs(a-b)
        #print("loss1: ",loss)
        loss/=(source.size(0) * source.size(0))
        #print("loss1_3: ",loss)
                
        #loss=torch.mean(-torch.sum(target * torch.log(source), axis = -1))
        
        loss *= self.weight
        """
            
        #print("loss0_3: ",loss)
        return loss   
       
class Self_L3(nn.Module):

    def __init__(self, weight=1.0, size_average=True):
        """
        初始化参数，因为要实现 torch.nn.CrossEntropyLoss 的两个比较重要的参数

        :param weight: 给予每个类别不同的权重
        :param size_average: 是否要对 loss 求平均
        """

        self.weight = weight
        self.size_average = size_average

    def __call__(self, new_source, new_target): #a=A

        if self.weight==0:
            return 0.0
        

        loss=1e-4
        for i in range(new_source.size(0)):
            for j in range(new_source.size(0)):
                #print("source[i]: ",source[i],"  \nsource[j]: ",source[j], " \n target[i]: ",target[i]," ; target[j]: ",target[j])
                a=-torch.sum(new_source[i] * torch.log(new_source[j]/new_source[i]), axis = -1)
                b=-torch.sum(new_target[i] * torch.log(new_source[j]/new_target[i]), axis = -1)
                #a= -torch.sum(target[j] * torch.log(source[i]), axis = -1)
                #b= -torch.sum(target[j] * torch.log(target[i]), axis = -1)
                #print("a : ",a, " ; b: ",b)
                loss+=torch.abs(a-b)
        #print("loss3: ",loss)
        loss/=(new_source.size(0) * new_source.size(0))

        
        """

        loss=0
        for i in range(source.size(0)):
            for j in range(source.size(0)):
                #print("source[i]: ",source[i],"  \nsource[j]: ",source[j], " \n target[i]: ",target[i]," ; target[j]: ",target[j])
                a=-torch.sum(source[i] * torch.log(source[j]/source[i]), axis = -1)
                b=-torch.sum(target[i] * torch.log(source[j]/target[i]), axis = -1)
                #a= -torch.sum(target[j] * torch.log(source[i]), axis = -1)
                #b= -torch.sum(target[j] * torch.log(target[i]), axis = -1)
                #print("a : ",a, " ; b: ",b)
                loss+=torch.abs(a-b)
        #print("loss3: ",loss)
        loss/=(source.size(0) * source.size(0))
        #print("loss3_v: ",loss)
                
        #loss=torch.mean(-torch.sum(target * torch.log(source), axis = -1))
        
        loss *= self.weight
        """
            
        #print("loss3_vv: ",loss)
        return loss   
 

    
if __name__=="__main__":
    lambda_0=1
    lambda_1=0
    lambda_2=0
    lambda_3=0
    CE_or_KL=False # use KL

    criterion=SelfKL(num_classes=10,lambda_0=lambda_0, lambda_1=lambda_1,lambda_2=lambda_2,lambda_3=lambda_3,CE_or_KL=CE_or_KL)

    #loss=criterion([],[])

    y=torch.Tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
    q=torch.Tensor([0.0979, 0.1032, 0.1052, 0.0768, 0.0915, 0.0699, 0.1226, 0.0883, 0.1011,
         0.1435])

    y=torch.Tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]])
    y = torch.Tensor([1,7,1,1]).long()
    q=torch.Tensor([[0.0979, 0.1032, 0.1052, 0.0768, 0.0915, 0.0699, 0.1226, 0.0883, 0.1011,
         0.1435],
        [0.1032, 0.0803, 0.1014, 0.0751, 0.0913, 0.0765, 0.1423, 0.0998, 0.1148,
         0.1153],
        [0.1023, 0.0907, 0.1138, 0.0793, 0.0855, 0.0808, 0.1353, 0.0999, 0.0894,
         0.1230],
        [0.1149, 0.1142, 0.0973, 0.0642, 0.0881, 0.0896, 0.1282, 0.1047, 0.1158,
         0.0831]])

    start=time.time()
    loss0=criterion(q,y)
    CE=nn.CrossEntropyLoss()
    ce_loss= CE(q,y) 
    
    print("ce_loss: ",ce_loss," ; loss0: ",loss0)

