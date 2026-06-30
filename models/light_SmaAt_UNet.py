#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:36:55 2024


@author: luther

largely inspired from https://github.com/HansBambel/SmaAt-UNet/tree/master
"""

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                       Test  SmaAt-Unet                                | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
from torch.nn.modules.utils import _triple
import torch.nn.functional as F

import numpy as np
import math

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                        DSC Conv                                       | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

# Taken from https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/14
# padding circular has been added because earth is a circle (:p)
# class DepthToSpace(nn.Module):
#     def __init__(self, block_size):
#         super().__init__()
#         self.bs = block_size

#     def forward(self, x):
#         N, C, H, W = x.size()
#         x = x.view(N, self.bs, self.bs, C // (self.bs**2), H, W)  # (N, bs, bs, C//bs^2, H, W)
#         x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
#         x = x.view(N, C // (self.bs**2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
#         return x


# class SpaceToDepth(nn.Module):
#     # Expects the following shape: Batch, Channel, Height, Width
#     def __init__(self, block_size):
#         super().__init__()
#         self.bs = block_size

#     def forward(self, x):
#         N, C, H, W = x.size()
#         x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
#         x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
#         x = x.view(N, C * (self.bs**2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
#         return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=1):
        super(DepthwiseSeparableConv, self).__init__()
        # In Tensorflow DepthwiseConv2D has depth_multiplier instead of kernels_per_layer
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode='circular', 
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                      CBAM module                                      | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16, activation=nn.ReLU()):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #  https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
        #  uses Convolutions instead of Linear
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            activation,
            nn.Linear(input_channels // reduction_ratio, input_channels),
        )

    def forward(self, x):
        # Take the input and apply average and max pooling
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale


class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16, kernel_size=7, activation=nn.ReLU()):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio, activation=activation)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out
    
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                      Unet module                                      | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #


class DoubleConvDS(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernels_per_layer=1, activation=nn.ReLU()):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        if not activation.inplace :
            activation.inplace=True
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels,
                mid_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                padding=1,
            ),
            nn.BatchNorm2d(mid_channels),
            activation,
            DepthwiseSeparableConv(
                mid_channels,
                out_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            activation,
        )

    def forward(self, x):
        return self.double_conv(x)


class DownDS(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernels_per_layer=1, activation=nn.ReLU()):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer, activation=activation),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpDS(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, kernels_per_layer=1,activation=nn.ReLU()):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConvDS(
                in_channels,
                out_channels,
                in_channels // 2,
                kernels_per_layer=kernels_per_layer,
                activation=activation
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer, activation=activation)

    def forward(self, x1, x2):
              
        
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2], 'circular')
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
    
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                      SmaAt-UNet                                       | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #


class light_SmaAt_UNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        kernels_per_layer=2,
        bilinear=True,
        reduction_ratio=16,
        activation=nn.ReLU(),
        psc=False,
        chl=False,
        nb_layers=64,
        depth=5
    ):
        super(light_SmaAt_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        self.depth=depth
        self.psc=psc
        self.chl=chl
        self.nb_layers=nb_layers
        reduction_ratio = reduction_ratio
        
        self.depth = depth
        self.down_convs = []
        self.up_convs = []
        self.cbams=[]
        # create the encoder pathway and add to a list
        
        self.inc = DoubleConvDS(self.n_channels, self.nb_layers, kernels_per_layer=kernels_per_layer, activation=activation)
        self.cbam1 = CBAM(self.nb_layers, reduction_ratio=reduction_ratio, activation=activation)

        for i in range(1,depth-1):
            ins = self.nb_layers if i == 1 else outs
            outs = self.nb_layers*(2**i)

            cbam = CBAM(outs, reduction_ratio=reduction_ratio, activation=activation)
            self.cbams.append(cbam)
            down_conv = DownDS(ins, outs, kernels_per_layer=kernels_per_layer, activation=activation)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        ins = outs
        outs = self.nb_layers*(2**(depth-1))
        factor = 2 if self.bilinear else 1 #careful if bilinear is false
        cbam = CBAM(outs//factor, reduction_ratio=reduction_ratio, activation=activation)
        self.cbams.append(cbam)
        down_conv = DownDS(ins, outs//factor, kernels_per_layer=kernels_per_layer, activation=activation)
        self.down_convs.append(down_conv)
            
        for i in range(depth-2):
            ins = outs
            outs = ins // 2
            up_conv = UpDS(ins, outs//factor, self.bilinear, kernels_per_layer=kernels_per_layer, activation=activation)

            self.up_convs.append(up_conv)
            
        up_conv = UpDS(outs, outs//2, self.bilinear, kernels_per_layer=kernels_per_layer, activation=activation)

        self.up_convs.append(up_conv)
            
        self.outc = OutConv(self.nb_layers, self.n_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.cbams = nn.ModuleList(self.cbams)


    #     self.reset_params()

    # @staticmethod
    # def weight_init(m):
    #     if isinstance(m, nn.Conv2d):
    #         init.xavier_normal_(m.weight)
    #         init.constant_(m.bias, 0)


    # def reset_params(self):
    #     for i, m in enumerate(self.modules()):
    #         self.weight_init(m)


    def forward(self, x):
            
        x=self.inc(x)
        encoder_outs = [self.cbam1(x)]         
        # encoder pathway, save outputs for merging
        for i, (module1, module2) in enumerate(zip(self.down_convs,self.cbams)):
            x = module1(x)
            encoder_outs.append(module2(x))

        x=encoder_outs[-1]
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(x, before_pool)
        
        x = self.outc(x)
        
        #just un petit module pour softmax les psc si nécessaire
        if self.psc:
            psc=nn.functional.softmax(x[:,:3],dim=1)
            chl=x[:,3]
            return (psc,torch.unsqueeze(chl,axis=1))
        elif self.chl:
            psc=nn.functional.softmax(x,dim=1)
            return tuple([psc])
        
        return x



if __name__=="__main__":
    

    model=light_SmaAt_UNet(8,4,kernels_per_layer=2,bilinear=True,reduction_ratio=16,
                     activation=nn.ReLU(),psc=True,chl=False, nb_layers=32, depth=4)
    model.to(device='cuda')
    a=torch.Tensor(np.random.random(((32,8,360,100)))).to(device='cuda')
    b=model(a)