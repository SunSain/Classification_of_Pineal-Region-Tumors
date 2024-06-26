import torch
from torch import nn, einsum
from einops import rearrange


class MixAttention(nn.Module):
    def __init__(self,dim=514,out_dim=514,head=1,w1=0.,w2=0.,w3=0.):
        super(MixAttention, self).__init__()
        self.head = head
        self.dim = dim
        self.out_dim = out_dim
        self.to_qkv1 = nn.Linear(dim, out_dim*3*head)
        self.to_qkv2 = nn.Linear(dim, out_dim*3*head)
        self.to_qkv3 = nn.Linear(dim, out_dim*3*head)
        self.norm = nn.Softmax(dim=-1)
        self.to_out = nn.Linear(out_dim,out_dim)
        self.w1, self.w2, self.w3=w1,w2,w3

    
    def get_result(self,q,k,v):
        b, n, h = self.b, self.n, self.head
        dots = einsum('b h i w, b h j w -> b h i j',q,k)
        attn = self.norm(dots)
        out = einsum('b h i w, b h w j -> b h i j',attn, v)
        out = rearrange(out,'b h n w -> b n (h w)')
        return self.to_out(out)
    
    def forward(self,t1,t2,t1c):

        print("t1.shape: ",t1.shape)
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
        
        out11 = self.get_result(q1,k1,v1)
        out22 = self.get_result(q2,k2,v2)
        out33 = self.get_result(q3,k3,v3)     
        #b n (h w)

        #out11,out21,out31 =(1-self.w1)*out11, self.w1*out21,self.w1*out31
        #out22,out12,out32 =(1-self.w2)*out22, self.w2*out12,self.w2*out32
        #out33,out13,out23 =(1-self.w3)*out33, self.w3*out13,self.w3*out23
        
        #t1_result = torch.cat((out21,out11,out31),dim=-1)+torch.cat((k1,k1,k1),dim=-1)
        #t2_result = torch.cat((out12,out22,out32),dim=-1)+torch.cat((k2,k2,k2),dim=-1)
        #t1c_result =torch.cat((out13,out33,out23),dim=-1)+torch.cat((k3,k3,k3),dim=-1)

        #t1_result = torch.cat((out21,out11,out31),dim=-1)
        #t2_result = torch.cat((out12,out22,out32),dim=-1)
        #t1c_result =torch.cat((out13,out33,out23),dim=-1)
    
        #t1_result = torch.cat((out21,out11,out31),dim=-1)+torch.cat((t1,t1,t1),dim=-1)
        #t2_result = torch.cat((out12,out22,out32),dim=-1)+torch.cat((t2,t2,t2),dim=-1)
        #t1c_result =torch.cat((out13,out33,out23),dim=-1)+torch.cat((t1c,t1c,t1c),dim=-1)
        #print("t1_result.shape: ",t1_result.shape)


        out11,out21,out31 =(1-self.w1)*out11, 0.5*self.w1*out21,0.5*self.w1*out31
        out22,out12,out32 =(1-self.w2)*out22, 0.5*self.w2*out12,0.5*self.w2*out32
        out33,out13,out23 =(1-self.w3)*out33, 0.5*self.w3*out13,0.5*self.w3*out23

        t1_result = out21+out11+out31
        t2_result = out12+out22+out32
        t1c_result = out13+out33+out23
        
        #t1_result = torch.cat((out21,out11,out31),dim=-1)
        #t2_result = torch.cat((out12,out22,out32),dim=-1)
        #t1c_result = torch.cat((out13,out33,out23),dim=-1)

        
        return t1_result,t2_result,t1c_result

        #T = torch.cat((out12,out13,out21,out23,out31,out32), -1)
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
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), 
                                    nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t:rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)   
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)+x

class MixTransformer(nn.Module):
    def __init__(self,dim=514, out_dim=256,head=1,num_classes=2, depth=1,w1=0,w2=0,w3=0):
        super(MixTransformer, self).__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.head = head
        self.num_classes = num_classes
        self.mlp_outdim=out_dim*3
        self.mlp_head = nn.Sequential(nn.LayerNorm(out_dim), 
                                      nn.Linear(out_dim, num_classes))
        self.mix_attention = MixAttention(dim, out_dim, head,w1,w2,w3)


    def forward(self,t1,t2,t1c):
        t1 =torch.unsqueeze(t1,dim=0)
        t2 =torch.unsqueeze(t2,dim=0)
        t1c =torch.unsqueeze(t1c,dim=0)
        atten_1,atten_2,atten_3 = self.mix_attention(t1,t2,t1c)
        print("atten_1.shape: ",atten_1.shape)
        #atten = torch.cat((atten_1,atten_2,atten_3), dim=-1)
        atten =atten_1+atten_2+atten_3
        
        print("atten.shape: ",atten.shape)
        feat = atten
        out = self.mlp_head(atten)

        out = torch.squeeze(out,dim=0)
        return out,feat

if __name__ =="__main__":
    t1 = torch.rand(1,514) #[1,4,514] [4,514]
    #t1 = torch.unsqueeze(t1,0)
    #print("t1: ",t1)
    t2 = torch.rand(1,514)
    t1c = torch.rand(1,514)
    tans = MixTransformer(dim=514, out_dim=256)

    out = tans(t1,t2,t1c)
        

