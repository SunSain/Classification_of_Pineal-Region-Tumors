from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary
from sklearn import decomposition
import numpy as np



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
        Two_Three = False
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
        fc_input_orig = 512+2
        fc_input_radiomics = 512+2+128
        if use_radiomics:
            self.fc = nn.Linear((fc_input_radiomics) * block.expansion, num_classes)
            self.fc_two = nn.Linear((fc_input_radiomics) * block.expansion, 2)
        else:
            self.fc = nn.Linear((fc_input_orig) * block.expansion, num_classes)
            self.fc_two = nn.Linear((fc_input_orig) * block.expansion, 2)
            
        
            
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

    def _forward_impl(self, x: Tensor, input_affix: Tensor,radiomics:Tensor) -> Tensor:
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

        x =  torch.cat((x, input_affix),1)
        #x = x.view(-1,514)

        if self.use_radiomics:
            print("y.size: ",radiomics.size())
            y= self.radiomics_fc(radiomics)
            x =  torch.cat((x, y),1)
        
        
        if self.feature_align:
            feat = x
        feat = x
        if self.Two_Three:
            x1 = self.fc_two(x)
            x2 = self.fc(x)
            if self.feature_align:
                return x1,x2, feat
            return x1,x2
        if self.num_classes==2:
            x = self.fc_two(x)
            if self.feature_align:
                return x, feat
            #return x
            return x, feat
        
        x = self.fc(x)
        if self.feature_align:
            return x, feat
        #return x
        return x, feat

    def forward(self, x: Tensor, input_affix: Tensor,radiomics:Tensor) -> Tensor:
        return self._forward_impl(x,input_affix,radiomics)

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


if __name__=="__main__":
    model = ResNet10().cuda()
    summary(model,(1,512,512,23))