import torch
from torch.autograd import Function
from torch import nn
import math
import numpy as np
import torch.nn.functional as F

class BatchCriterion(nn.Module):
    ''' Compute the loss within each batch  
    '''
    def __init__(self,num_classes=3, negM=1, T=1, batchSize=4, DEVICE=torch.device('cpu')):
        super(BatchCriterion, self).__init__()
        self.negM = negM
        self.T = T
        self.num_classes=num_classes
        self.DEVICE =DEVICE
<<<<<<< HEAD
        
    def forward(self, x, targets):
=======
    def __call__(self, x, targets):
        batchSize = x.size(0)
        #print("x: ",x)
        #print("targets: ",targets)
        
        x = F.softmax(x)
        print("x: ",x)
        norm = x.pow(2).sum(1, keepdim=True).pow(1./2)
        x = x.div(norm)
        #print("x: ",x," \nx.size(): ",x.size())
        self.diag_mat= 1 - torch.eye(self.num_classes).to(self.DEVICE)
        items = []
        for i in range(self.num_classes):
            item = (targets==i).nonzero(as_tuple=True)[0]
            items.append(item)
        #print("items: ",items)
        feats = None
        new_x=x.clone().requires_grad_(True)
        for i in range(self.num_classes):
            feat = torch.tensor([0.]).repeat(x.size(1)) if  items[i].nelement()== 0 else 1/items[i].size(0) * (new_x[items[i]].sum(0)) 
            #print(i,feat)
            feat = feat.unsqueeze(0)
            feat = feat.to(self.DEVICE)
            if feats == None:
                feats = feat
                continue
            feats = torch.cat((feats, feat), dim=0)
        print("feats: ",feats)
        reordered_x = torch.cat((feats.narrow(0,2//2,2//2),\
                feats.narrow(0,0,2//2)), 0)
        print("reordered_x: ",reordered_x)
        pos = (reordered_x.data*reordered_x.data).sum(1)
        print("pos: ",pos)
        pos = pos.div_(self.T).exp_()
        all_prob = torch.mm(reordered_x,reordered_x.t().data).div_(self.T).exp_()
        #print("matrix: ",torch.mm(reordered_x,reordered_x.t().data))
        print("all_prob: ",all_prob)
        if self.negM==1:
            all_div = all_prob.sum(1)
        else:
            #remove pos for neg
            all_div = (all_prob.sum(1) - pos)*self.negM + pos

        print("all_div: ",all_div)
        lnPmt = torch.div(pos, all_div)
        #print("lnPmt: ",lnPmt)
        """
        # negative probability
        Pon_div = all_div.repeat(self.num_classes,1)
        print("Pon_div.t(): ",Pon_div.t())
        lnPon = torch.div(all_prob, Pon_div.t())
        print("lnPon_1: ",lnPon)
        lnPon = -lnPon.add(-1)
        print("lnPon_2: ",lnPon)
        # equation 7 in ref. A (NCE paper)
        lnPon.log_()        
        # also remove the pos term
        print("lnPon: ",lnPon)
        lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_() 
        print("lnPon_last: ",lnPon)
        lnPmt.log_()
        lnPonsum = lnPon.sum(0)
        print("lnPmtsum: ",lnPmtsum)
        print("lnPonsum: ",lnPonsum)
        lnPonsum = lnPonsum * self.negM
        """
        lnPmt.log_()
        lnPmtsum = lnPmt.sum(0)
        
        # negative multiply m
        
        #loss = - (lnPmtsum + lnPonsum)
        loss = -lnPmtsum
        return loss

    def forward_3class_only(self, x, targets):
>>>>>>> 3a4a4f2 (20240625-code)
        batchSize = x.size(0)
        #print("x: ",x," \nx.size(): ",x.size())
        
        norm = x.pow(2).sum(1, keepdim=True).pow(1./2)
        x = x.div(norm)
        #print("x: ",x," \nx.size(): ",x.size())
        self.diag_mat= 1 - torch.eye(self.num_classes)
        item_0 = (targets==0).nonzero(as_tuple =True)[0]
        item_1 = (targets==1).nonzero(as_tuple =True)[0]
        item_2 = (targets==2).nonzero(as_tuple =True)[0]
        #print("item_0 :",item_0, " ; item_0.size: ",item_0.nelement())
        #print("item_1 :",item_1, " ; item_0.size: ",item_1.nelement())
        #print("item_2 :",item_2, " ; item_0.size: ",item_2.nelement())
        #print("targets :",targets)
        #print("len(item_0.size()): ",len(item_0.size()))
        #print("len(item_1.size()): ",len(item_1.size()))
        #print("len(item_2.size()): ",len(item_2.size()))
        feat_0 = torch.tensor([0.]).repeat(x.size(1)) if  item_0.nelement()== 0 else 1/item_0.size(0) * (x[item_0].sum(0)) 
        feat_1 = torch.tensor([0.]).repeat(x.size(1)) if  item_1.nelement()== 0  else  1/item_1.size(0) * (x[item_1].sum(0))
        feat_2 = torch.tensor([0.]).repeat(x.size(1)) if  item_2.nelement()== 0  else 1/item_2.size(0) * (x[item_2].sum(0)) 
        feat_0=feat_0.unsqueeze(0)
        feat_0=feat_0.to(self.DEVICE)
        feat_1=feat_1.unsqueeze(0)
        feat_2=feat_2.unsqueeze(0)
        feat_1=feat_1.to(self.DEVICE)
        feat_2=feat_2.to(self.DEVICE)
        #print("feat_0: ",feat_0)
        #print("feat_1: ",feat_1)
        #print("feat_2: ",feat_2)
        
        
        #get positive innerproduct

        reordered_x = torch.cat((feat_0,feat_1,feat_2), 0)
<<<<<<< HEAD
        
        #print("reordered_x: ",reordered_x)
        #print("reordered_x*reordered_0.data: ",reordered_x*reordered_x)
        #print("sum(1): ",(reordered_x.data*reordered_x.data).sum(0))
        #reordered_x = reordered_x.data
        pos = (reordered_x.data*reordered_x.data).sum(1)
        pos = pos.div_(self.T).exp_()
        #print("pos-1: ",pos)
        #pos = pos.sum()
        #get all innerproduct, remove dag
        all_prob = torch.mm(reordered_x,reordered_x.t().data).div_(self.T).exp_()
        #print("all_prob: ",all_prob)
        if self.negM==1:
            all_div = all_prob.sum(1)
            #print("all_div: ",all_div)
=======
    
        pos = (reordered_x.data*reordered_x.data).sum(1)
        pos = pos.div_(self.T).exp_()
        all_prob = torch.mm(reordered_x,reordered_x.t().data).div_(self.T).exp_()*self.diag_mat
        if self.negM==1:
            all_div = all_prob.sum()
>>>>>>> 3a4a4f2 (20240625-code)
        else:
            #remove pos for neg
            all_div = (all_prob.sum(1) - pos)*self.negM + pos

        lnPmt = torch.div(pos, all_div)
        #print("lnPmt: ",lnPmt)

        # negative probability
        Pon_div = all_div.repeat(self.num_classes,1)
        lnPon = torch.div(all_prob, Pon_div.t())
        lnPon = -lnPon.add(-1)
        # equation 7 in ref. A (NCE paper)
        lnPon.log_()
        # also remove the pos term
        
        lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_() 
        #print("lnPon: ",lnPon)
        lnPmt.log_()


        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.sum(0)
<<<<<<< HEAD
        #print("lnPonsum: ",lnPonsum)
=======
        print("lnPmtsum: ",lnPmtsum)
        print("lnPonsum: ",lnPonsum)
>>>>>>> 3a4a4f2 (20240625-code)
        # negative multiply m
        lnPonsum = lnPonsum * self.negM
        loss = - (lnPmtsum + lnPonsum)
        return loss

if __name__ =="__main__":
<<<<<<< HEAD
    m = BatchCriterion()
    x=torch.tensor([[9,1,0,0,0,0],
       [9,4,0,0,0,0],
       [0,0,1,1,0,0],
       [0,0,1,5,0,0],
       [9,0,0,0,2,2],
       [9,0,0,0,4,5]
    ])
    x=x*1.0
    x= F.softmax(x)
    print("x:",x)
    targets = torch.tensor([1,1,1,1,2,2])
=======
    m = BatchCriterion(2)
    x=torch.tensor([[3,1,0,0,0,0],
       [6,2,0,0,0,0],
       [0,0,4,5,0,0],
       [0,0,8,10,0,0]
    ])
    x=x*1.0
    #x= F.softmax(x,1)
    print("x:",x)
    targets = torch.tensor([0,0,1,1])
>>>>>>> 3a4a4f2 (20240625-code)
    loss = m(x,targets)
    print("loss: ",loss)
    c = nn.CrossEntropyLoss()
    loss = c(x,targets)
<<<<<<< HEAD
    print("CE_loss: ",loss)
=======
    print("CE_loss: ",loss)
    a=torch.tensor([1,2])
    print(-a.add(-1))
>>>>>>> 3a4a4f2 (20240625-code)
