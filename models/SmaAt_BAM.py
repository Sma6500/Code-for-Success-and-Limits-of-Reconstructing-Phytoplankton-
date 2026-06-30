#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:10:08 2025

@author: luther

largely inspired from https://github.com/HansBambel/SmaAt-UNet/tree/master
"""

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         SmaAt-Unet                                    | #
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

from t2vec import Time2Vec

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
# |                                      BAM module                                      | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #



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
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.spatial_att(x)
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

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
#normal OutConv
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
                
    def forward(self, x):

        return self.conv(x)
    


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         print("Outconv computes sign and magnitude")
#         #Out_channels currently only serves to check the good hyperparameters configuration.
#         if out_channels!=2 :
#             raise ValueError(f"n_classes is not correctly configured, should be 2 instead of {out_channels}")
        
#         # Separate output for magnitude
#         self.magnitude_output = DepthwiseSeparableConv(in_channels, 5, kernel_size=3, padding=1)
        
#         # Separate output for sign
#         self.sign_output = DepthwiseSeparableConv(in_channels, 1, kernel_size=3, padding=1)
        
#         # Activation functions
#         self.relu = nn.ReLU()  # Ensures non-negative output for magnitude
#         #self.sigmoid=nn.sigmoid()
#         self.tanh = nn.Tanh()  # Ensures output between -1 and 1 for sign
    
#     def forward(self, x):

#         # Separate outputs
#         magnitude = self.relu(self.magnitude_output(x))  # Non-negative magnitude
#         sign = self.tanh(self.sign_output(x))  # Output between -1 and 1
#         #sign = self.sigmoid(self.sign_output(x))  # Output between 0 and 1

        
#         #return magnitude*sign
#         return (torch.unsqueeze(magnitude[:,0],1),
#                 torch.unsqueeze(magnitude[:,1],1),
#                 torch.unsqueeze(magnitude[:,2],1),
#                 torch.unsqueeze(magnitude[:,3],1),
#                 torch.unsqueeze(magnitude[:,4],1),
#                                 sign)
# # # +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                      SmaAt-UNet                                       | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #


class SmaAt_BAM(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        kernels_per_layer=2,
        bilinear=True,
        reduction_ratio=16,
        activation=nn.ReLU(),
        chl=False,
        nb_layers=8,
        time_encoded=False
    ):
        super(SmaAt_BAM, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        self.chl=chl
        self.nb_layers=nb_layers
        reduction_ratio = reduction_ratio
        
        self.time_encoded=time_encoded
        if time_encoded:
            print(f"adding time2vec encoding with trend and {time_encoded} frequencies \nTime encoding scalar map is expected with input\nInput shape is currently (100,360)")
            self.t2v=Time2Vec(time_encoded, self.n_channels, (100,360))
            self.n_channels += time_encoded+1

        self.inc = DoubleConvDS(self.n_channels, self.nb_layers, kernels_per_layer=kernels_per_layer, activation=activation)
        self.cbam1 = CBAM(self.nb_layers, reduction_ratio=reduction_ratio, activation=activation)
        self.down1 = DownDS(self.nb_layers, self.nb_layers*2, kernels_per_layer=kernels_per_layer, activation=activation)
        self.cbam2 = CBAM(self.nb_layers*2, reduction_ratio=reduction_ratio, activation=activation)
        self.down2 = DownDS(self.nb_layers*2, self.nb_layers*4, kernels_per_layer=kernels_per_layer, activation=activation)
        self.cbam3 = CBAM(self.nb_layers*4, reduction_ratio=reduction_ratio, activation=activation)
        self.down3 = DownDS(self.nb_layers*4, self.nb_layers*8, kernels_per_layer=kernels_per_layer, activation=activation)
        self.cbam4 = CBAM(self.nb_layers*8, reduction_ratio=reduction_ratio, activation=activation)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(self.nb_layers*8, self.nb_layers*16 // factor, kernels_per_layer=kernels_per_layer, activation=activation)
        self.cbam5 = CBAM(self.nb_layers*16 // factor, reduction_ratio=reduction_ratio, activation=activation)
        self.up1 = UpDS(self.nb_layers*16, self.nb_layers*8 // factor, self.bilinear, kernels_per_layer=kernels_per_layer, activation=activation)
        self.up2 = UpDS(self.nb_layers*8, self.nb_layers*4 // factor, self.bilinear, kernels_per_layer=kernels_per_layer, activation=activation)
        self.up3 = UpDS(self.nb_layers*4, self.nb_layers*2 // factor, self.bilinear, kernels_per_layer=kernels_per_layer, activation=activation)
        self.up4 = UpDS(self.nb_layers*2, self.nb_layers, self.bilinear, kernels_per_layer=kernels_per_layer, activation=activation)

    
        self.outc = OutConv(self.nb_layers, self.n_classes)
    
        if self.chl==2 or self.chl==3:
            self.relu = nn.ReLU()  # Ensures non-negative output for magnitude
            #self.sigmoid=nn.sigmoid()
            #self.tanh = nn.Tanh()  # Ensures output between -1 and 1 for sign
            
    def forward(self, x):
        
        if self.time_encoded:
            x0=self.t2v(x)
        else :
            x0=x
        x1 = self.inc(x0)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        
        #just un petit module pour softmax les psc si nécessaire
        if self.chl==0:
            psc=nn.functional.softmax(logits[:,:3],dim=1)
            chl=logits[:,3:]
            return (psc,chl)
        elif self.chl==1:
            psc=nn.functional.softmax(logits,dim=1)
            return tuple([psc])
        elif self.chl==2:
            psc=nn.functional.softmax(logits[:,:3],dim=1)
            chl=logits[:,3,None]
            sign = logits[:,4,None]   # Output between -1 and 1
            magnitude = self.relu(logits[:,5:])  # Non-negative magnitude
            chl_anom=torch.concat([magnitude,sign],dim=1)
            return (psc,chl,chl_anom)
        elif self.chl==3:
            chl=logits[:,0,None]
            sign = logits[:,1,None]  # Output between -1 and 1
            magnitude = self.relu(logits[:,2:])  # Non-negative magnitude
            chl_anom=torch.concat([magnitude,sign],dim=1)
            return (chl,chl_anom)
        else :
            return tuple([logits])
        

if __name__=="__main__":
    

    model=SmaAt_BAM(5, 2, kernels_per_layer=2, nb_layers=64, time_encoded=False)
    model.to(device='cuda')
    a=torch.Tensor(np.random.random(((4,5,100,360)))).to(device='cuda')
    b=model(a)