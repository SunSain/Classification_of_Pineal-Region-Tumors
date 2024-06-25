

import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import cv2


from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary
from sklearn import decomposition
import numpy as np


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


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv3d:
    """3x3 convolution with padding"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class PCA(object):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        n = X.shape[0]
        self.mean = torch.mean(X, axis=0)
        X = X - self.mean
        covariance_matrix = 1 / n * torch.matmul(X.T, X)
        eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
        eigenvalues = torch.norm(eigenvalues, dim=1)
        idx = torch.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        self.proj_mat = eigenvectors[:, 0:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return X.matmul(self.proj_mat)

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 2,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        input_channel=1,
        use_radiomics=False,
        feature_align=False,
        Two_Three = False,
        use_clinical=True,
        clinicaltype=None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer
        self.use_radiomics=use_radiomics
        self.feature_align = feature_align
        self.Two_Three = Two_Three
        self.inplanes = 64
        self.dilation = 1
        self.num_classes = num_classes
        self.use_clinical=use_clinical
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(input_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1,1)) 
        
        self.pca= PCA(n_components=10)
        self.kernel_pca = decomposition.KernelPCA(n_components=None, kernel="rbf", gamma=10, fit_inverse_transform=True, alpha=0.1)
        self.radiomics_fc = nn.Linear(93* block.expansion, 128)
        if use_clinical and clinicaltype==None:
            fc_input_orig = 512+2
        else:
            fc_input_orig = 512
        fc_input_radiomics = 512+2+128
        if use_radiomics:
            self.fc = nn.Linear((fc_input_radiomics) * block.expansion, 2)
        else:
            self.fc = nn.Linear((fc_input_orig) * block.expansion, 2)
            
        
            
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor,input_affix:Tensor) -> Tensor:
        # See note [TorchScript super()]
        #print("x.size: ",x.size())
        #print("x: ",x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #print("x.size3: ",x.size())
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.use_clinical:
            x =  torch.cat((x,input_affix),1)
        #x = x.view(-1,514)

        if self.feature_align:
            feat = x
        feat = x
        
        
        x = self.fc(x)
        if self.feature_align:
            return x, feat
        #return x
        return x, feat

    def forward(self, x: Tensor,input_affix:Tensor) -> Tensor:
        return self._forward_impl(x,input_affix)

"""
print("input_radiomics: ",input_radiomics)
self.pca.fit(input_radiomics)
y_pca = self.pca.transform(input_radiomics)
y= self.radiomics_fc(y_pca)

y_kpca = self.kernel_pca.fit(input_radiomics).transform(input_target)
print("y.size: ",y_kpca)
y= self.radiomics_fc(y_kpca)

print("y.size: ",y)
x =  torch.cat((x, y),1)
"""

def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    num_classes,
    weightspath: None,
    **kwargs: Any,
) -> ResNet:
    if weightspath is not None:
        print("has weights, begin to load")

    model = ResNet(block, num_classes,layers, **kwargs)

    if weightspath is not None:
        state_dict= torch.load(weightspath)
        model.load_state_dict(state_dict)

    return model





def ResNet10(*, weightspath= None, num_classes=2, **kwargs: Any) -> ResNet:
    
    weightspath = weightspath
    
    return _resnet(BasicBlock,num_classes, [1,1,1,1], weightspath, **kwargs)


def ResNet18(*, weightspath= None, num_classes=2, **kwargs: Any) -> ResNet:

    weightspath = weightspath
    
    return _resnet(BasicBlock,num_classes, [2, 2, 2, 2], weightspath, **kwargs)

def ResNet22(*, weightspath= None, num_classes=2, **kwargs: Any) -> ResNet:
    
    weightspath = weightspath
    
    return _resnet(BasicBlock,num_classes, [2, 2, 4, 2], weightspath, **kwargs)

def ResNet24(*, weightspath= None, num_classes=2, **kwargs: Any) -> ResNet:
    
    weightspath = weightspath
    
    return _resnet(BasicBlock,num_classes, [2, 3, 4, 2], weightspath, **kwargs)


def ResNet30(*, weightspath= None, num_classes=2, **kwargs: Any) -> ResNet:
    
    weightspath = weightspath
    
    return _resnet(BasicBlock,num_classes, [3, 4, 5, 2], weightspath, **kwargs)


def ResNet34(*, weightspath= None,num_classes=2, **kwargs: Any) -> ResNet:
    
    weightspath = weightspath
    
    return _resnet(BasicBlock,num_classes, [3,4,6,3], weightspath, **kwargs)

def ResNet50(*, weightspath= None, num_classes=2, **kwargs: Any) -> ResNet:
    
    weightspath = weightspath
    return _resnet(Bottleneck, num_classes,[3, 4, 6, 3], weightspath, **kwargs)

import torch
import torch.nn.functional as F


class GradCAM(object):
    """Calculate GradCAM salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcam = GradCAM(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """
    def __init__(self, model, layer_name,input_size,verbose=False):
        layer_name = layer_name
        self.model = model

        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        target_layer = self.model.layer4

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        device =  'cpu'
        self.model(torch.zeros(1, 1, *(input_size), device=device),torch.tensor([[0,1]]))
        print('saliency_map size :', self.activations['value'].shape[2:])


    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        print("input: ",input)
        print("input.size: ",input.shape)
        b, d,c, h, w = input.size()

        logit,feat = self.model(input,torch.tensor([[0,1]]))
        print("logit: ",logit)
        
        score = logit[:, class_idx].squeeze()
        print("logit: ",logit)
        print("score: ",score)

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        print("gradients.size():",gradients.size())
        activations = self.activations['value']
        b, k, d,u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        print("alpha.size: ",alpha.shape)
        #alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b,k, 1,1)
        print("weights.size: ",weights.shape)
        

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        #saliency_map = saliency_map.expand(-1, c, -1, -1, -1)
        #saliency_map = saliency_map.repeat(1, 24, 1, 1, 1)
        saliency_map = saliency_map.squeeze(1).squeeze(3)
        print("saliency_map.shape: ",saliency_map.shape)# 1,512,1,1  ;1,512,16,16
        print("(d,c,h, w): ",d,c,h, w)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        print("saliency_map.size: ",saliency_map.shape)
        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)


class GradCAMpp(GradCAM):
    """Calculate GradCAM++ salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcampp
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcampp = GradCAMpp(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcampp(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """
    def __init__(self, model,target_layer,input_size, verbose=False):
        super(GradCAMpp, self).__init__(model,target_layer,input_size, verbose)

    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()

        logit,feat = self.model(input,torch.tensor([[0,1]]))
        print("logit: ",logit)
        
        score = logit[:, class_idx].squeeze() 
            
        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value'] # dS/dA
        activations = self.activations['value'] # A
        b, k, u, v = gradients.size()

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
                activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom+1e-7)
        positive_gradients = F.relu(score.exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(400,400,23), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map-saliency_map_min).div(saliency_map_max-saliency_map_min).data

        return saliency_map, logit

class MixGradCAM(object):
    """Calculate GradCAM salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcam = GradCAM(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """
    def __init__(self, model, layer_name,input_size,verbose=False):
        layer_name = layer_name
        self.model = model

        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            atten_1,atten_2,atten_3=output
            atten = torch.cat((atten_1,atten_2,atten_3), dim=-1)
            self.activations['value'] = atten
            return None

        target_layer = self.model.mix_attention

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        device =  'cpu'
        self.model(torch.zeros(1, 514, device=device),torch.zeros(1, 514, device=device),torch.zeros(1, 514, device=device))
        print('saliency_map size :', self.activations['value'].shape[2:])


    def forward(self, t1c_input,t1_input,t2_input,input_affix, class_idx=None, retain_graph=False,t1_model="",t2_model="",t1c_model=""):
        
        b, d,c, h, w = t1c_input.size()
        t1_model.eval()
        t1c_model.eval()
        t2_model.eval()
        
        t1_img_out,t1_feature = t1_model(t1_input,input_affix)
        t2_img_out,t2_feature = t2_model(t2_input,input_affix)
        t1c_img_out,t1c_feature = t1c_model(t1c_input,input_affix)

        input_t1_feature = t1_feature.to(DEVICE)
        input_t2_feature = t2_feature.to(DEVICE)
        input_t1c_feature = t1c_feature.to(DEVICE)

        logit,feat = self.model(input_t1_feature,input_t2_feature,input_t1c_feature)

        print("logit: ",logit)
        
        score = logit[:, class_idx].squeeze()
        print("logit: ",logit)
        print("score: ",score)

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        print("gradients.size():",gradients.size())
        activations = self.activations['value']
        
        gradients = gradients.unsqueeze(1)
        b, d,u,k = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        print("alpha.size: ",alpha.shape)
        #alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b,k, 1,1)
        print("weights.size: ",weights.shape)
        

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        #saliency_map = saliency_map.expand(-1, c, -1, -1, -1)
        #saliency_map = saliency_map.repeat(1, 24, 1, 1, 1)
        saliency_map = saliency_map
        print("saliency_map.shape: ",saliency_map.shape)# 1,512,1,1  ;1,512,16,16
        print("(d,c,h, w): ",d,c,h, w)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        print("saliency_map.size: ",saliency_map.shape)
        return saliency_map, logit

    def __call__(self, t1c_input,t1_input,t2_input,input_affix, class_idx=None, retain_graph=False,t1_model="",t2_model="",t1c_model=""):
        return self.forward(t1c_input,t1_input,t2_input,input_affix, class_idx=class_idx, retain_graph=False,t1_model=t1_model,t2_model=t2_model,t1c_model=t1c_model)


class MixGradCAMpp(GradCAM):
    """Calculate GradCAM++ salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcampp
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcampp = GradCAMpp(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcampp(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """
    def __init__(self, model,target_layer,input_size, verbose=False):
        super(GradCAMpp, self).__init__(model,target_layer,input_size, verbose)

    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()

        logit,feat = self.model(input,torch.tensor([[0,1]]))
        print("logit: ",logit)
        
        score = logit[:, class_idx].squeeze() 
            
        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value'] # dS/dA
        activations = self.activations['value'] # A
        b, k, u, v = gradients.size()

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
                activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom+1e-7)
        positive_gradients = F.relu(score.exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(400,400,23), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map-saliency_map_min).div(saliency_map_max-saliency_map_min).data

        return saliency_map, logit


class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        self.activations = None
        self.model.eval()

    def save_gradient(self, grad):
        self.gradient = grad

    def forward_pass(self, x):
        self.activations = []
        self.gradient = None

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        #print("x.size3: ",x.size())
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x =  torch.cat((x,torch.tensor([[0,1]])),1)

        self.activations.append(x)
        x = self.model.fc(x)

        return x

    def backward_pass(self, indices):
        one_hot = torch.zeros_like(indices, dtype=torch.float)
        one_hot.scatter_(1, indices, 1.0)
        self.model.zero_grad()
        one_hot.backward(torch.ones_like(indices), retain_graph=True)

    def generate_heatmap(self, input_tensor, class_index):
        self.model.eval()
        hook_handle = self.target_layer.register_backward_hook(self.save_gradient)
        activations = self.forward_pass(input_tensor)
        one_hot = torch.zeros((1, self.model.fc.out_features), dtype=torch.float)
        one_hot[0][class_index] = 1
        self.backward_pass(one_hot)
        hook_handle.remove()

        pooled_gradients = torch.mean(self.gradient, dim=[2, 3, 4], keepdim=True)
        weighted_activations = torch.mean(torch.mul(pooled_gradients, self.activations[-1]), dim=1, keepdim=True)
        heatmap = nn.functional.relu(weighted_activations)
        
        heatmap = heatmap.squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap

def append_dict(sub_fns,key_name,root,filedict):
        for i, f in enumerate(sub_fns):
            fn = f.split("_")
            fn = fn[0].split(".")
            try:
                    sid=int(fn[0])
            except:
                print("it's the_select_file")
                continue
            # no more new data
            if sid>122:
                continue
            # no more new data
            if filedict.get(sid,None)==None :
                filedict[sid]={}
            filedict[sid][key_name]=os.path.join(root,f)
            print("self.filedict[sid][key_name] : ",filedict[sid][key_name])
        return filedict
  

def get_pathlist():
        t1_path='/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1' 
        t2_path='/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T2' 
        t1c_path='/opt/chenxingru/Pineal_region/after_12_08/0704_data/test_T1C'

        filedict={}
        t1_sub_fns = sorted(os.listdir(t1_path))
        t2_sub_fns = sorted(os.listdir(t2_path))
        t1c_sub_fns = sorted(os.listdir(t1c_path))
        filedict=append_dict(t1_sub_fns,"T1",t1_path,filedict)
        filedict=append_dict(t2_sub_fns,"T2",t2_path,filedict)
        filedict=append_dict(t1c_sub_fns,"T1C",t1c_path,filedict)
        
        delete_box=[]
        for i, f in enumerate(t1c_sub_fns): 
            fn = f.split("_")
            fn = fn[0].split(".")
            try:
                    sid=int(fn[0])
            except:
                print("select file")
                continue
            # no more new data
            if sid>122:continue
            # no more new data            
            item = filedict[sid]
            t1_path=t1_path
            t2_path=t2_path
            t1c_path=t1c_path
            try:
                t1_path =  item['T1']
                t2_path =  item['T2']
                t1c_path =  item['T1C']   
                if  t1c_path==None or t1_path==None or t2_path==None :
                    delete_box.append(sid) 
            except:
                print("Got file missing!")
                delete_box.append(sid)
                continue 
        """ 
        delete_box.append(32)
        delete_box.append(33)
        delete_box.append(50)
        delete_box.append(83)
        """
        print("Missing files [ T1/T2/T1C ]: ",delete_box) 
        print(filedict.keys())
        for i,sid in enumerate(delete_box):
            del filedict[sid]
        print(filedict.keys())
        return  filedict



def augment( origin_img,aug_form=None,saveroot="",filename="",aug=False):
        # ori_img is must be T1C

        composed_RM = False
        tradition_token = aug_form
        
        croporpad = tio.CropOrPad(
        (400,400,23),
        mask_name='brain_mask',
        )
        img = croporpad(origin_img)
        img_orig=img

        
        if aug_form == "Composed_RM" or aug_form == "composed_RM":
            composed_RM=True
            tradition_token = "Composed"   
        if aug== False or aug_form=="" or aug_form=="None" or aug_form=="none":
            img = sitk.GetArrayFromImage(img_orig).astype(np.float32)
            img = np.expand_dims(img, axis=0)
            img = np.ascontiguousarray(img, dtype= np.float32)   
            
            return img,img
        
        if not aug_form == "Random_Mask": #not zero, not just RM, then must be traditional aug or aug_RM
            transform_dict={
                tio.RandomAffine(degrees=15),
                tio.RandomFlip(flip_probability=0.5, axes=('LR'))
                }
            trans_dict = {
                    'Affine':tio.RandomAffine(degrees=15),
                    'Flip':tio.RandomFlip(flip_probability=0.5, axes=('LR')),
                    'Composed':tio.Compose(transform_dict)
                    }
            transform = trans_dict[tradition_token]
            print("transform: ",transform)
            org_img = sitk.GetArrayFromImage(origin_img).astype(np.float32)
            org_img = np.expand_dims(org_img, axis=0)
            org_img = np.ascontiguousarray(org_img, dtype= np.float32)   
            
            image = transform(img) 
            img = sitk.GetArrayFromImage(image).astype(np.float32)
            img = np.expand_dims(img, axis=0)
            img = np.ascontiguousarray(img, dtype= np.float32)
            #img = torch.from_numpy(img).type(torch.FloatTensor)
            auged_img = img 
            
            img = array2image(auged_img[0],origin_img)
            save_path = os.path.join(saveroot,filename)
            sitk.WriteImage(img,save_path)
            
            if not composed_RM:
                return 
            else:
                img = array2image(auged_img[0],origin_img)
                if use_secondimg:
                    secondimg = array2image(maskimg[0],submaskimg)
                if usethird:
                    thirdimg = array2image(third_maskimg[0],thirdimg)
                if  attachment!=None:
                    img_t1 = array2image(auged_img_t1[0],img_t1)
                    img_t2 = array2image(auged_img_t2[0],img_t2)
                else:
                    img_t1,img_t2 =img, img
                
        

        if aug_form == "Random_Mask" or composed_RM == True : # constrained_loss_part don't need RM, only RM or aug_RM(and normal_train)
            print("transform: Random_Mask_Aug ")
            img = sitk.GetArrayFromImage(img).astype(np.float32)
            img = np.expand_dims(img, axis=0)
            img = np.ascontiguousarray(img, dtype= np.float32)
            org_img = img
            #img = torch.from_numpy(img).type(torch.FloatTensor)

            maskimg=img
            instance_img = np.append(img,maskimg,axis=0)
                
            print("instance_img.shape: ",instance_img.shape)
            instance_img = torch.from_numpy(instance_img).type(torch.FloatTensor)
            #instance_img={"img":torch.from_numpy(img), "secondimg":torch.from_numpy(maskimg),"noSameRM":noSameRM}
            #instance_img={0:torch.from_numpy(img), 1:torch.from_numpy(secondimg),2:noSameRM}
            #print("instance_img: ",instance_img)
            mask_instance_img = tio.Lambda(ratio_cube_mask)(instance_img).numpy()
            auged_img,auged_maskimg = img[0],mask_instance_img[1]
            auged_img = np.expand_dims(auged_img, axis=0)
            auged_maskimg = np.expand_dims(auged_maskimg, axis=0)   
            
            return auged_img

import os
import torch
import nibabel as nib
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import SimpleITK as sitk
from scipy.ndimage import rotate
import nibabel as nib
import numpy as np
import os
from utils2.config import opt
from sklearn.model_selection import train_test_split
import torchio as tio
from utils2.preprocessor import Preprocessor,array2image
from utils2.cube_mask import cube_mask,ratio_cube_mask
from utils2.diy_single_radiomics import DiySingleRadiomics
from torchvision.utils import make_grid, save_image

from grad_cam_utils import visualize_cam, Normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# Example usage:
# Assuming `model` is your trained 3D CNN model and `target_layer` is the last convolutional layer
# Initialize GradCAM
def singlemodality():
    best_pretrained_model_path="/home/chenxr/Pineal_region/after_12_08/Results/New_attention_2/Two/three_modality_best_attention/model_result/ResNet18_BatchAvg_0401_T1__k-fold-sub-fold-1__best_model.pth.tar"
    best_pretrained_model_path="/home/chenxr/Pineal_region/after_12_08/Results/New_attention/Two/three_modality_best_attention/model_result/ResNet18_BatchAvg_0401_T1__k-fold-sub-fold-2__best_model.pth.tar"
    best_pretrained_model_path="/home/chenxr/Pineal_region/after_12_08/Results/New_attention_2/Two/three_modality_best_attention/model_result/ResNet18_BatchAvg_0401_T1__k-fold-sub-fold-1__best_model.pth.tar"
    best_pretrained_model_path="/home/chenxr/Pineal_region/after_12_08/Results/attention/Two/three_modality_best_attention/model_result/ResNet18_BatchAvg_0401_T1__k-fold-sub-fold-2__best_model.pth.tar"
    best_pretrained_model_path="/home/chenxr/Pineal_region/after_12_08/Results/tmp_test/Two/T1C_SingleBatchAvg_selfKL_Composed_RM_composed_ResNet18/model_result/ResNet18_SingleBatchAvg_selfKL_0401_T1__k-fold-sub-fold-2__best_model.pth.tar"
    best_pretrained_model_path="/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_Singlebatchavg_ce_new_37/model_result/ResNet18_CE_0401_T1__k-fold-sub-fold-2__best_model.pth.tar"
    model = ResNet18(num_classes=2,input_channel=1,use_radiomics=False,feature_align=True,use_clinical=True)
                

    checkpoint = torch.load(best_pretrained_model_path,map_location = 'cpu')
    if checkpoint['state_dict'].get( "fc_two.weight",None) !=None:
            checkpoint['state_dict']['fc.weight']=checkpoint['state_dict']['fc_two.weight']
            checkpoint['state_dict']['fc.bias']=checkpoint['state_dict']['fc_two.bias']
            del checkpoint['state_dict']['fc_two.weight']
            del checkpoint['state_dict']['fc_two.bias']
    model.load_state_dict(checkpoint['state_dict'])
    target_layer=model.fc

    grad_cam = GradCAM3D(model, target_layer)

    # Forward pass your input MRI image through the model
    #input_tensor = torch.randn(1, 1, depth, height, width)  # Assuming depth, height, and width are the dimensions of your 3D MRI image
    class_index = 0  # Index of the target class for which you want to generate the heatmap
    root="/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1C/"
    for filep in sorted(os.listdir(root)):
        t1c_path = os.path.join(root,filep)
        sid = filep.split("_")[0]
        #t1c_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1C/034_7_4A2FBEA7174F4ED6BE896ADB8D202353_t1_tirm_tra_dark-fluid_Turn_0000.nii.gz"
        #preprocessor = Preprocessor(target_spacing=[0.45,0.45,6.5])
        #origin_img, brain_mask_img = preprocessor.run(img_path=t1c_path)

        origin_img = sitk.ReadImage(t1c_path)
        input_tensor=torch.Tensor(augment( origin_img))
        input_tensor= torch.unsqueeze(input_tensor,dim=0)
        print("input_tensor: ",input_tensor)
        print("input_tensor.size: ",input_tensor.shape)
        # Generate heatmap
        #heatmap = grad_cam.generate_heatmap(input_tensor, class_index)
        #print("heatmap: ",heatmap)

        target_layer='fc'
        input_size=[22,477,473]
        alexnet_gradcam = GradCAM(model,target_layer,input_size, True)
        alexnet_gradcampp = GradCAMpp(model,target_layer,input_size, True)


        mask, _ = alexnet_gradcam(input_tensor,class_idx=0)
        #b,n,h,c = mask.shape
        #mask = mask.view(n,1,h,c)
        new_mask=[]
        new_result=[]
        heatmap, result = visualize_cam(mask[0][0], input_tensor[0][0][12])
        new_result=result
        new_mask=heatmap
        """
        for i in range(mask.shape[0]):
            submask=mask[i]
            print("submask.size: ",submask.shape)
            heatmap, result = visualize_cam(submask, input_tensor[0][0][i])
            print("heatmap.size: ",heatmap.shape)
            new_mask.append(heatmap.tolist())
            new_result.append(result.tolist())
        """


        nifti_file = nib.load(t1c_path)
        data = nifti_file.get_fdata()

        visualization=new_mask
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(input_tensor[0,0, 12,:, :], cmap='gray') 
        axes[0].set_title('Input')

        selected_image = np.array(visualization)
        print("visualization.shape: ",np.array(visualization).shape)
        # 确保选择的图像是 float 类型，并且归一化到 [0, 1] 范围内
        gray_image = np.mean(selected_image, axis=0)

        # 应用颜色映射
        cmap_image = cm.jet(gray_image)



        #original = np.uint8(cm.gray(data[12])[..., :3] * 255)
        selected_image = np.array(new_result)
        print("visualization.shape: ",np.array(visualization).shape)
        # 确保选择的图像是 float 类型，并且归一化到 [0, 1] 范围内
        gray_image = np.mean(selected_image, axis=0)

        # 应用颜色映射
        cmap_image = cm.jet(gray_image)


        axes[1].imshow(cmap_image)
        axes[1].set_title('Grad-CAM')
        fig.suptitle(f'target = {0}')
        plt.savefig("/home/chenxr/"+sid+"_mix_attention.png")
        
        #print("new_mask: ",new_mask)
        new_mask=torch.tensor(new_mask)
        print("new_mask.shape: ",new_mask.shape)
        #mask_pp, _ = alexnet_gradcampp(input_tensor,class_idx=0)
        #heatmap_pp, result_pp = visualize_cam(mask_pp, input_tensor)
        images=[]
        #images.append(torch.stack([input_tensor.squeeze().cpu(), new_mask,result], 0))
            
        #images = make_grid(torch.cat(images, 0), nrow=5)

        #print("images:",images)
        #print("images.size: ",images.size)


        # Visualize the heatmap
        # You can use libraries like matplotlib or OpenCV to visualize the heatmap
        # For example, you can overlay the heatmap on the original MRI image to visualize the regions of interest
        # You may need to adjust the dimensions and format of the heatmap to match your MRI image



def load_cnn_feature_model(best_model_path,input_channel=1,feature_align=True):
    model = ResNet18(num_classes=2,input_channel=1,use_radiomics=False,feature_align=True,use_clinical = True)
    try:
        print("\n1")
        checkpoint = torch.load(best_model_path,map_location = DEVICE)
        print("\2")
        if checkpoint['state_dict'].get( "fc_two.weight",None) !=None:
                     checkpoint['state_dict']['fc.weight']=checkpoint['state_dict']['fc_two.weight']
                     checkpoint['state_dict']['fc.bias']=checkpoint['state_dict']['fc_two.bias']
                     del checkpoint['state_dict']['fc_two.weight']
                     del checkpoint['state_dict']['fc_two.bias']
        model.load_state_dict(checkpoint['state_dict'])
        print("\n3")
        for param in model.parameters():
            #print("\n4")
            param.requires_grad = False
        #print("\n5")
    except:
        print("no cnn pretrained path: ",best_model_path)
        assert 0==1
    model.eval()
    return model

DEVICE=torch.device('cpu')
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def multimodality():
    fold=2
    t1c_best_path="/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_Singlebatchavg_ce_new_37/model_result/ResNet18_CE_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar"%fold
    t1_best_path="/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1_mask1_composed_RM_Singlebatchavg_37/model_result/ResNet18_CE_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar"%fold
    t2_best_path="/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T2_mask1_composed_RM_Singlebatchavg_37/model_result/ResNet18_CE_0401_T1__k-fold-sub-fold-%d__best_model.pth.tar"%fold
    model=MixTransformer(514,256,w1=0.5,w2=0.5,w3=0.5)
    
    t1c_best_path="/home/chenxr/Pineal_region/after_12_08/Results/weighted_clinical/Two/T1C_composed_RM_Sing_Batchavg_13/model_result/ResNet18_CE_0401_T1__k-fold-sub-fold-2__best_model.pth.tar"
    bestpath="/home/chenxr/Pineal_region/after_12_08/Results/test_mixattention/Two/self_cross_best_CE_19_newPretrained/model_result/ResNet18_CE_0401_T1__k-fold-sub-fold-2__best_model.pth.tar"
    bestpath="/home/chenxr/Pineal_region/after_12_08/Results/new_mixattention/Two/arch1_111_CE/model_result/ResNet18_CE_0401_T1__k-fold-sub-fold-2__best_model.pth.tar"
    check=torch.load(bestpath)
    model.load_state_dict(check['state_dict'])
    model.eval()
    with torch.no_grad():
        torch.cuda.empty_cache()
    #print("Initial gpu: ")
    #print_gpu_usage()
    t1model=load_cnn_feature_model(t1_best_path,input_channel=1,feature_align=True)
    t1_model = t1model.to(DEVICE)
    #print("after T1 gpu: ")
    #print_gpu_usage()
    t2model=load_cnn_feature_model(t2_best_path,input_channel=1,feature_align=True)
    t2_model = t2model.to(DEVICE)
    #print("after T2 gpu: ")
    #print_gpu_usage()
    t1cmodel=load_cnn_feature_model(t1c_best_path,input_channel=1,feature_align=True)
    t1c_model = t1cmodel.to(DEVICE)
    t1c_model.eval()
    t1_model.eval()
    t2_model.eval()
    pathlist=get_pathlist()
    for key, stuff  in pathlist.items():
        #if key10: continue
        t1c_path,t1_path,t2_path=stuff.get("T1C"),stuff.get("T1"),stuff.get("T2")
        print("t1c_path,t1_path,t2_path: ",t1c_path,t1_path,t2_path)
        #t1c_path = os.path.join(root,filep)
        sid = key
        #t1c_path="/opt/chenxingru/Pineal_region/after_12_08/0704_data/train_T1C/034_7_4A2FBEA7174F4ED6BE896ADB8D202353_t1_tirm_tra_dark-fluid_Turn_0000.nii.gz"
        #preprocessor = Preprocessor(target_spacing=[0.45,0.45,6.5])
        #origin_img, brain_mask_img = preprocessor.run(img_path=t1c_path)
        print("sid: ",sid)
        t1c_origin_img = sitk.ReadImage(t1c_path)
        t1c_input_tensor,t1c_origin_img=augment( t1c_origin_img)
        t1c_input_tensor=torch.Tensor(t1c_input_tensor)
        t1c_input_tensor= torch.unsqueeze(t1c_input_tensor,dim=0)

        t1_origin_img = sitk.ReadImage(t1_path)
        t1_input_tensor,t1_origin_img=augment( t1_origin_img)
        t1_input_tensor=torch.Tensor(t1_input_tensor)
        
        t1_input_tensor= torch.unsqueeze(t1_input_tensor,dim=0)
        
        t2_origin_img = sitk.ReadImage(t2_path)
        t2_input_tensor,t2_origin_img=augment( t2_origin_img)
        t2_input_tensor=torch.Tensor(t2_input_tensor)
        
        t2_input_tensor= torch.unsqueeze(t2_input_tensor,dim=0)
        
        t2_input_tensor=t2_input_tensor.to(DEVICE)
        t1_input_tensor=t1_input_tensor.to(DEVICE)
        
        t1c_input_tensor=t1c_input_tensor.to(DEVICE)
        
        target_layer=""
        input_size=[23,400,400]
        alexnet_gradcam = MixGradCAM(model,target_layer,input_size, True)
        mask, _ = alexnet_gradcam(t1c_input_tensor,t1_input_tensor,t2_input_tensor,input_affix=torch.tensor([[0,1]]).to(DEVICE), class_idx=0, retain_graph=False,t1c_model=t1c_model,t2_model=t2_model,t1_model=t1_model)
        #b,n,h,c = mask.shape
        #mask = mask.view(n,1,h,c)
        print("t1c_input_tensor.shape: ",t1c_input_tensor.shape)
        print("mask: ",mask.shape)
        new_mask=[]
        new_result=[]
        heatmap, result = visualize_cam(mask[0][0], t1c_input_tensor[0][0][12])
        new_result=result
        new_mask=heatmap
        visualization=new_mask
        fig, axes = plt.subplots(1, 2)
        
        axes[0].imshow(t1c_input_tensor[0,0, 12,:, :], cmap='gray') 
        axes[0].set_title('Input')
        selected_image = np.array(visualization)
        gray_image = np.mean(selected_image, axis=0)
        cmap_image = cm.jet(gray_image)
        selected_image = np.array(new_result)
        gray_image = np.mean(selected_image, axis=0)
        cmap_image = cm.jet(gray_image)
        axes[1].imshow(cmap_image)
        axes[1].set_title('Grad-CAM')
        fig.suptitle(f'target = {0}')
        plt.savefig("/home/chenxr/"+str(sid)+"_t1c_mix_attention2.png")


        heatmap, result = visualize_cam(mask[0][0], t1c_input_tensor[0][0][13])
        new_result=result
        new_mask=heatmap
        visualization=new_mask
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(t1c_input_tensor[0,0, 13,:, :], cmap='gray') 
        axes[0].set_title('Input')
        selected_image = np.array(visualization)
        gray_image = np.mean(selected_image, axis=0)
        cmap_image = cm.jet(gray_image)
        selected_image = np.array(new_result)
        gray_image = np.mean(selected_image, axis=0)
        cmap_image = cm.jet(gray_image)
        axes[1].imshow(cmap_image)
        axes[1].set_title('Grad-CAM')
        fig.suptitle(f'target = {0}')
        plt.savefig("/home/chenxr/"+str(sid)+"_t1c_mix_attention3.png")

        heatmap, result = visualize_cam(mask[0][0], t1c_input_tensor[0][0][14])
        new_result=result
        new_mask=heatmap
        visualization=new_mask
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(t1c_input_tensor[0,0, 14,:, :], cmap='gray') 
        axes[0].set_title('Input')
        selected_image = np.array(visualization)
        gray_image = np.mean(selected_image, axis=0)
        cmap_image = cm.jet(gray_image)
        selected_image = np.array(new_result)
        gray_image = np.mean(selected_image, axis=0)
        cmap_image = cm.jet(gray_image)
        axes[1].imshow(cmap_image)
        axes[1].set_title('Grad-CAM')
        fig.suptitle(f'target = {0}')
        plt.savefig("/home/chenxr/"+str(sid)+"_t1c_mix_attention4.png")

        heatmap, result = visualize_cam(mask[0][0], t1c_input_tensor[0][0][15])
        new_result=result
        new_mask=heatmap
        visualization=new_mask
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(t1c_input_tensor[0,0, 15,:, :], cmap='gray') 
        axes[0].set_title('Input')
        selected_image = np.array(visualization)
        gray_image = np.mean(selected_image, axis=0)
        cmap_image = cm.jet(gray_image)
        selected_image = np.array(new_result)
        gray_image = np.mean(selected_image, axis=0)
        cmap_image = cm.jet(gray_image)
        axes[1].imshow(cmap_image)
        axes[1].set_title('Grad-CAM')
        fig.suptitle(f'target = {0}')
        plt.savefig("/home/chenxr/"+str(sid)+"_t1c_mix_attention5.png")

        heatmap, result = visualize_cam(mask[0][0], t1c_input_tensor[0][0][16])
        new_result=result
        new_mask=heatmap
        visualization=new_mask
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(t1c_input_tensor[0,0, 16,:, :], cmap='gray') 
        axes[0].set_title('Input')
        selected_image = np.array(visualization)
        gray_image = np.mean(selected_image, axis=0)
        cmap_image = cm.jet(gray_image)
        selected_image = np.array(new_result)
        gray_image = np.mean(selected_image, axis=0)
        cmap_image = cm.jet(gray_image)
        axes[1].imshow(cmap_image)
        axes[1].set_title('Grad-CAM')
        fig.suptitle(f'target = {0}')
        plt.savefig("/home/chenxr/"+str(sid)+"_t1c_mix_attention6.png")

        heatmap, result = visualize_cam(mask[0][0], t1c_input_tensor[0][0][17])
        new_result=result
        new_mask=heatmap
        visualization=new_mask
        fig, axes = plt.subplots(1, 2) 
        axes[0].imshow(t1c_input_tensor[0,0, 17,:, :], cmap='gray') 
        axes[0].set_title('Input')
        selected_image = np.array(visualization)
        gray_image = np.mean(selected_image, axis=0)
        cmap_image = cm.jet(gray_image)
        selected_image = np.array(new_result)
        gray_image = np.mean(selected_image, axis=0)
        cmap_image = cm.jet(gray_image)
        axes[1].imshow(cmap_image)
        axes[1].set_title('Grad-CAM')
        fig.suptitle(f'target = {0}')
        plt.savefig("/home/chenxr/"+str(sid)+"_t1c_mix_attention7.png")

        heatmap, result = visualize_cam(mask[0][0], t1c_input_tensor[0][0][11])
        new_result=result
        new_mask=heatmap
        visualization=new_mask
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(t1c_input_tensor[0,0, 11,:, :], cmap='gray') 
        axes[0].set_title('Input')
        selected_image = np.array(visualization)
        gray_image = np.mean(selected_image, axis=0)
        cmap_image = cm.jet(gray_image)
        selected_image = np.array(new_result)
        gray_image = np.mean(selected_image, axis=0)
        cmap_image = cm.jet(gray_image)
        axes[1].imshow(cmap_image)
        axes[1].set_title('Grad-CAM')
        fig.suptitle(f'target = {0}')
        plt.savefig("/home/chenxr/"+str(sid)+"_t1c_mix_attention8.png")
        
        heatmap, result = visualize_cam(mask[0][0], t1c_input_tensor[0][0][10])
        new_result=result
        new_mask=heatmap
        visualization=new_mask
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(t1c_input_tensor[0,0, 10,:, :], cmap='gray') 
        axes[0].set_title('Input')
        selected_image = np.array(visualization)
        gray_image = np.mean(selected_image, axis=0)
        cmap_image = cm.jet(gray_image)
        selected_image = np.array(new_result)
        gray_image = np.mean(selected_image, axis=0)
        cmap_image = cm.jet(gray_image)
        axes[1].imshow(cmap_image)
        axes[1].set_title('Grad-CAM')
        fig.suptitle(f'target = {0}')
        plt.savefig("/home/chenxr/"+str(sid)+"_t1c_mix_attention9.png")

        heatmap, result = visualize_cam(mask[0][0], t1c_input_tensor[0][0][9])
        new_result=result
        new_mask=heatmap
        visualization=new_mask
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(t1c_input_tensor[0,0, 9,:, :], cmap='gray') 
        axes[0].set_title('Input')
        selected_image = np.array(visualization)
        gray_image = np.mean(selected_image, axis=0)
        cmap_image = cm.jet(gray_image)
        selected_image = np.array(new_result)
        gray_image = np.mean(selected_image, axis=0)
        cmap_image = cm.jet(gray_image)
        axes[1].imshow(cmap_image)
        axes[1].set_title('Grad-CAM')
        fig.suptitle(f'target = {0}')
        plt.savefig("/home/chenxr/"+str(sid)+"_t1c_mix_attention10.png")
        
        new_mask=[]
        new_result=[]
        heatmap, result = visualize_cam(mask[0][0], t1_input_tensor[0][0][12])
        new_result=result
        new_mask=heatmap
        visualization=new_mask
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(t1_input_tensor[0,0, 12,:, :], cmap='gray') 
        axes[0].set_title('Input')
        selected_image = np.array(visualization)
        gray_image = np.mean(selected_image, axis=0)
        cmap_image = cm.jet(gray_image)
        selected_image = np.array(new_result)
        gray_image = np.mean(selected_image, axis=0)
        cmap_image = cm.jet(gray_image)
        axes[1].imshow(cmap_image)
        axes[1].set_title('Grad-CAM')
        fig.suptitle(f'target = {0}')
        plt.savefig("/home/chenxr/"+str(sid)+"_t1_mix_attention.png")
        
        new_mask=[]
        new_result=[]
        heatmap, result = visualize_cam(mask[0][0], t2_input_tensor[0][0][12])
        new_result=result
        new_mask=heatmap
        visualization=new_mask
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(t2_input_tensor[0,0, 12,:, :], cmap='gray') 
        axes[0].set_title('Input')
        selected_image = np.array(visualization)
        gray_image = np.mean(selected_image, axis=0)
        cmap_image = cm.jet(gray_image)
        selected_image = np.array(new_result)
        gray_image = np.mean(selected_image, axis=0)
        cmap_image = cm.jet(gray_image)
        axes[1].imshow(cmap_image)
        axes[1].set_title('Grad-CAM')
        fig.suptitle(f'target = {0}')
        plt.savefig("/home/chenxr/"+str(sid)+"_t2_mix_attention.png")
        

if __name__ =="__main__":
    multimodality()     
    #singlemodality()
    
    
    