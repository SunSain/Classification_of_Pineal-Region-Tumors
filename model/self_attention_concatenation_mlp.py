import torch
from torch import nn, einsum
from einops import rearrange


class CrossAttention(nn.Module):
    def __init__(self,dim=514,out_dim=256,head=1):
        super(CrossAttention, self).__init__()
        self.head = head
        self.dim = dim
        self.out_dim = out_dim
        self.to_qkv1 = nn.Linear(dim, out_dim*3*head)
        self.to_qkv2 = nn.Linear(dim, out_dim*3*head)
        self.to_qkv3 = nn.Linear(dim, out_dim*3*head)
        self.norm = nn.Softmax(dim=-1)
        self.to_out = nn.Linear(out_dim,out_dim)

    
    def get_result(self,q,k,v):
        b, n, h = self.b, self.n, self.head
        dots = einsum('b h i w, b h j w -> b h i j',q,k)
        attn = self.norm(dots)
        out = einsum('b h i w, b h w j -> b h i j',attn, v)
        out = rearrange(out,'b h n w -> b n (h w)')
        return self.to_out(out)
    
    def forward(self,t1,t2,t1c):
        
        b, n, _ , h = *t1.shape, self.head
        self.b, self.n = b,n

        #print("t1.shape: ",t1.shape)
        #print("self.to_qkv1: ",self.to_qkv1)
        qkv = self.to_qkv1(t1).chunk(3,dim=-1)
        q1,k1,v1 = map(lambda t: rearrange(t, 'b n (h w) -> b h n w', h=h),qkv)
        q, k, v =  map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        
        qkv = self.to_qkv2(t2).chunk(3,dim=-1)
        q2,k2,v2 = map(lambda t: rearrange(t, 'b n (h w) -> b h n w', h=h),qkv)
        
        qkv = self.to_qkv3(t1c).chunk(3,dim=-1)
        q3,k3,v3 = map(lambda t: rearrange(t, 'b n (h w) -> b h n w', h=h),qkv)
        
        
        out12 = self.get_result(q2,k1,v1)
        out13 = self.get_result(q3,k1,v1)
        out21 = self.get_result(q1,k2,v2)
        out23 = self.get_result(q3,k2,v2)
        out31 = self.get_result(q1,k3,v3)
        out32 = self.get_result(q2,k3,v3)
        #b n (h w)

        
        k1 = rearrange(k1,'b h n w -> b n (h w)')
        k2 = rearrange(k2,'b h n w -> b n (h w)')
        k3 = rearrange(k3,'b h n w -> b n (h w)')
        
        t1_result = torch.cat((out21,out31),dim=-1)+torch.cat((k1,k1),dim=-1)
        t2_result = torch.cat((out12,out32),dim=-1)+torch.cat((k2,k2),dim=-1)
        t1c_result =torch.cat((out13,out23),dim=-1)+torch.cat((k3,k3),dim=-1)
        
        #print("t1_result.shape: ",t1_result.shape)
        return t1_result,t2_result,t1c_result

        T = torch.cat((out12,out13,out21,out23,out31,out32), -1)
        # T: (6 b) n (h w)
        
        

class Attention(nn.Module):
    def __init__(self,dim=514,out_dim=256,head=1):
        super(Attention, self).__init__()
        self.head = head
        self.dim = dim
        self.out_dim = out_dim
        self.to_qkv = nn.Linear(dim, out_dim*3*head)
        self.norm = nn.Softmax(dim=-1)
        self.to_out = nn.Linear(out_dim,dim)
    
    def forward(self,x):
        b, n, _ , h = *x.shape, self.head
        qkv = self.to_qkv(x).chunk(3,dim=-1)
        q,k,v = map(lambda t: rearrange(t, 'b n (h w) -> b h n w', h=h))
        
        dots = einsum('b h i w, b h j w -> b h i j',q,k)
        attn = self.norm(dots)
        out = einsum('b h i w, b h w j -> b h i j',attn, v)
        
        out = rearrange('b h n w -> b n (h w)',out)
        
        return self.to_out(out)

class SelfAttention(nn.Module):
    def __init__(self, dim=1024, heads=1, dim_head = 128, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), 
                                    nn.Dropout(dropout))

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t:rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)   
        dots = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)+x

class SelfAttention_MLP_Transformer(nn.Module):
    def __init__(self,dim=514, out_dim=256, new_arch=False,head=1,num_classes=2, depth=1):
        super(SelfAttention_MLP_Transformer, self).__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.head = head
        self.num_classes = num_classes
        self.mlp_outdim=out_dim*3
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim*3), 
                                      nn.Linear(dim*3, num_classes))
        inner_dim=out_dim
        self.t1_self_attention=SelfAttention(dim, inner_dim, head)
        self.t1c_self_attention=SelfAttention(dim, inner_dim, head)
        self.t2_self_attention=SelfAttention(dim, inner_dim, head)
        

    def forward(self,t1,t2,t1c):
        t1 =torch.unsqueeze(t1,dim=0)
        t2 =torch.unsqueeze(t2,dim=0)
        t1c =torch.unsqueeze(t1c,dim=0)
        atten_1 = self.t1_self_attention(t1)
        atten_2 = self.t1c_self_attention(t1c)
        atten_3 = self.t2_self_attention(t2)
        atten = torch.cat((atten_1,atten_2,atten_3), dim=-1)
        feat = atten
        out = self.mlp_head(atten)
        
        out = torch.squeeze(out,dim=0)
        return out,feat

        

