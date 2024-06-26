import torch
from torch.autograd import Function
from torch import nn
import math
import numpy as np
import torch.nn.functional as F

class SingleBatchCriterion(nn.Module):
    ''' Compute the loss within each batch  
    '''
    def __init__(self):
        super(SingleBatchCriterion, self).__init__()
        self.negM = 1
        self.T = 1.0
    """
    #x:  tensor([[ 0.5991,  0.7510,  0.7194,  ...,  0.4499,  0.0000,  1.4940],                                                                                                   
        [ 0.6198,  0.6140,  0.9817,  ...,  0.8739,  0.0000,  0.5030],                                                                                                       
        [ 0.5070,  0.8152,  0.5782,  ...,  1.2632,  0.0000,  1.0976],                                                                                                       
        ...,                                                                                                                                                                
        [ 0.6198,  0.6140,  0.9817,  ...,  0.8739,  0.0000,  0.5030],                                                                                                       
        [ 0.5070,  0.8152,  0.5782,  ...,  1.2632,  0.0000,  1.0976],                                                                                                       
        [ 1.0304,  0.6559,  0.3158,  ...,  0.9487,  0.0000, -1.2809]],                                                                                                      
       device='cuda:2', grad_fn=<CatBackward0>) 
    #after x:  tensor([[0.0204, 0.0180, 0.0300,  ..., 0.0376, 0.0100, 0.0041],                                                                                                   
        [0.0337, 0.0521, 0.0311,  ..., 0.0666, 0.0206, 0.0230],                                                                                                             
        [0.0367, 0.0497, 0.0363,  ..., 0.0565, 0.0198, 0.0592],                                                                                                             
        ...,                                                                                                                                                                
        [0.0337, 0.0521, 0.0311,  ..., 0.0666, 0.0206, 0.0230],                                                                                                             
        [0.0367, 0.0497, 0.0363,  ..., 0.0565, 0.0198, 0.0592],                                                                                                             
        [0.0146, 0.0332, 0.0257,  ..., 0.0987, 0.0134, 0.0598]],                                                                                                            
       device='cuda:2')  
    """
       
    def forward(self, x):
        batchSize = x.size(0)
        print("x: ",x)
        #x = F.softmax(x)
        x = F.softmax(x)
        norm = x.pow(2).sum(1, keepdim=True).pow(1./2)
        print("norm: ",norm)
        x = x.div(norm)
        print("after x: ",x)
        self.diag_mat = 1 - torch.eye(batchSize).cuda()
        #get positive innerproduct
        reordered_x = torch.cat((x.narrow(0,batchSize//2,batchSize//2),\
                x.narrow(0,0,batchSize//2)), 0)
        #reordered_x = reordered_x.data
        pos = (x*reordered_x.data).sum(1).div_(self.T).exp_()
        print("pos: ",(x*reordered_x.data).sum(1).div_(self.T))
        #get all innerproduct, remove diag
        all_prob = torch.mm(x,x.t().data).div_(self.T).exp_()*self.diag_mat
        if self.negM==1:
            all_div = all_prob.sum(1)
        else:
            #remove pos for neg
            all_div = (all_prob.sum(1) - pos)*self.negM + pos

        lnPmt = torch.div(pos, all_div)

        # negative probability
        Pon_div = all_div.repeat(batchSize,1)
        lnPon = torch.div(all_prob, Pon_div.t())
        lnPon = -lnPon.add(-1)
        
        # equation 7 in ref. A (NCE paper)
        lnPon.log_()
        # also remove the pos term
        lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_()
        lnPmt.log_()

        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.sum(0)

        # negative multiply m
        lnPonsum = lnPonsum * self.negM
        loss = - (lnPmtsum + lnPonsum)/batchSize
        #assert 0==1
        return loss

def mul():
    a=torch.tensor([[1,2],[2,1]])
    b = torch.tensor([[3,4],[7,1]])
    print((a*b).sum(1))

if __name__ =="__main__":
    m = SingleBatchCriterion()
    x=torch.tensor([[ 0.8344,  0.9054,  0.8852,  0.6770,  1.0000, -1.4791],                              
        [ 0.8400,  0.9057,  0.8103,  0.7250,  1.0000, -1.4791]
    ])
    x=x*1.0
    x= F.softmax(x)
    print("x:",x)
    targets = torch.tensor([0,0,0,0,1,1])
    x=x.cuda()
    loss = m(x)
    print("loss: ",loss)
    c = nn.CrossEntropyLoss()
    loss = c(x,targets)
    print("CE_loss: ",loss)
    a=torch.tensor([1,2])
    print(-a.add(-1))