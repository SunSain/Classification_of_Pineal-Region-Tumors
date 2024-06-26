from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary
from math import sqrt
import torch
import torch.nn


class Self_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self,input_dim=514,dim_k=64,dim_v=64):
        
        super(Self_Attention,self).__init__()
        self.q1 = nn.Linear(input_dim,dim_k)
        self.k1 = nn.Linear(input_dim,dim_k)
        self.v1 = nn.Linear(input_dim,dim_v)
        self.q2 = nn.Linear(input_dim,dim_k)
        self.k2 = nn.Linear(input_dim,dim_k)
        self.v2 = nn.Linear(input_dim,dim_v)
        self.qc = nn.Linear(input_dim,dim_k)
        self.kc = nn.Linear(input_dim,dim_k)
        self.vc = nn.Linear(input_dim,dim_v)
        self._norm_fact = 1 / sqrt(dim_k)
    
        self.fc = nn.Linear(64*6,2)
    
    def forward(self,t1,t2,t1c): #batch_size * 512
        Q1 = self.q1(t1) # Q: batch_size * seq_len * dim_k
        print("Q1.size: ",Q1.size())
        Q1 = Q1.unsqueeze(1)
        print("after Q1.size: ",Q1.size())
        K1 = self.k1(t1) # K: batch_size * seq_len * dim_k
        print("K1.size: ",K1.size())
        K1=K1.unsqueeze(1)
        print("after K1.size: ",K1.size())
        
        V1 = self.v1(t1) # V: batch_size * seq_len * dim_v
        V1 = V1.unsqueeze(1)

        Q2 = self.q2(t2) # Q: batch_size * seq_len * dim_k
        Q2 = Q2.unsqueeze(1)
        K2 = self.k2(t2) # K: batch_size * seq_len * dim_k
        K2=K2.unsqueeze(1)
        V2 = self.v2(t2) # V: batch_size * seq_len * dim_v
        V2 = V2.unsqueeze(1)
        
        Qc = self.qc(t1c) # Q: batch_size * seq_len * dim_k
        Qc = Qc.unsqueeze(1)
        Kc = self.kc(t1c) # K: batch_size * seq_len * dim_k
        Kc=Kc.unsqueeze(1)
        Vc = self.vc(t1c) # V: batch_size * seq_len * dim_v
        Vc = Vc.unsqueeze(1)
        print("size check: ",Q2.size(),K1.permute(0,2,1).size())
        #不重复写，换成函数
        atten_12 = nn.Softmax(dim=-1)(torch.bmm(Q2,K1.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        atten_21 = nn.Softmax(dim=-1)(torch.bmm(Q1,K2.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        atten_1c = nn.Softmax(dim=-1)(torch.bmm(Qc,K1.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        atten_c1 = nn.Softmax(dim=-1)(torch.bmm(Q1,Kc.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        atten_2c = nn.Softmax(dim=-1)(torch.bmm(Qc,K2.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        atten_c2 = nn.Softmax(dim=-1)(torch.bmm(Q2,Kc.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        print("atten_c2.size: ",atten_c2.size())
        output_12 = torch.bmm(atten_12,V1) # Q * K.T() * V # batch_size * seq_len * dim_v
        output_12 = output_12.squeeze(1)
        print("output_12: ",output_12)
        output_21 = torch.bmm(atten_21,V2) # Q * K.T() * V # batch_size * seq_len * dim_v
        output_21 = output_21.squeeze(1)
        output_1c = torch.bmm(atten_1c,V1) # Q * K.T() * V # batch_size * seq_len * dim_v
        output_1c = output_1c.squeeze(1)
        output_2c = torch.bmm(atten_2c,V2) # Q * K.T() * V # batch_size * seq_len * dim_v
        output_2c = output_2c.squeeze(1)
        output_c1 = torch.bmm(atten_c1,Vc) # Q * K.T() * V # batch_size * seq_len * dim_v
        output_c1 = output_c1.squeeze(1)
        output_c2 = torch.bmm(atten_c2,Vc) # Q * K.T() * V # batch_size * seq_len * dim_v
        output_c2 = output_c2.squeeze(1)
        
        T = torch.cat((output_12,output_1c,output_21,output_2c,output_c1,output_c2), -1)
        print("T.size: ",T.size())
        output= self.fc(T)
        print("output: ",output)
        return output