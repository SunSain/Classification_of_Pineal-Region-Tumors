import torch
from torch import nn, einsum
from einops import rearrange


class Concateation_MLP(nn.Module):
    def __init__(self,dim=1280,num_classes=2,head=1):
        super(Concateation_MLP, self).__init__()
        self.head = head
        self.dim = dim
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim*3), 
                                      nn.Linear(dim*3, num_classes))
        self.fc=nn.Linear(out_dim*3*2, self.mlp_outdim)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,self_feat,cross_feat):
        
        atten = torch.cat((self_feat,cross_feat), dim=-1)
        print("inner atten: ",atten)
        feat = atten
        out = self.mlp_head(atten)
        print("inner out1: ",out)
        #out = torch.squeeze(out,dim=0)
        print("inner out2: ",out)
        return out,feat
    
if __name__ =="__main__":
    t1 = torch.rand(1,514)
    #print("t1: ",t1)
    t2 = torch.rand(1,514)
    t1c = torch.rand(1,514)
    tans = Concateation_MLP(dim=514, num_classes=2)
    out = tans(t1,t2,t1c)
        

