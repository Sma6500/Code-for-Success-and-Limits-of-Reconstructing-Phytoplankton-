#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 17:17:51 2024


@author: luther
largely inspired from but with DSC convolutions : 

https://github.com/milesial/Pytorch-UNet/tree/master
https://github.com/jaxony/unet-pytorch

padding mode is replicate

"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         UNET                                          | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
import numpy as np

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

"""
UNet implementation
"""


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


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):    
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, 
                     padding=padding, padding_mode='circular', bias=bias, groups=groups)


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)

def upconv2x2(in_channels, out_channels, mode='bilinear', kernels_per_layer=1):
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=1, kernels_per_layer=kernels_per_layer))


def batch_norm(in_channels):
    return nn.BatchNorm2d(in_channels)



class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU/SiLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True, activation=nn.ReLU(), kernels_per_layer=1):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.norm = batch_norm(out_channels)
        self.activation=activation

        self.conv1 = DepthwiseSeparableConv(
                        self.in_channels,
                        self.out_channels,
                        kernel_size=3,
                        kernels_per_layer=kernels_per_layer,
                        padding=1)
        
        self.conv2 = DepthwiseSeparableConv(
                        self.out_channels,
                        self.out_channels,
                        kernel_size=3,
                        kernels_per_layer=kernels_per_layer,
                        padding=1)
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x=self.conv1(x)
        x=self.norm(x)
        x=self.activation(x)
        
        x=self.conv2(x)
        x=self.norm(x)
        x=self.activation(x)
        
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU/SiLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, 
                 merge_mode='concat', up_mode='transpose', activation=nn.ReLU(), kernels_per_layer=1):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.norm = batch_norm(out_channels)
        self.activation=activation


        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode, kernels_per_layer=kernels_per_layer)

        if self.merge_mode == 'concat':
            self.conv1 = DepthwiseSeparableConv(
                            2*self.out_channels,
                            self.out_channels,
                            kernel_size=3,
                            kernels_per_layer=kernels_per_layer,
                            padding=1)
        else:
            #https://distill.pub/2016/deconv-checkerboard/
            # num of input channels to conv2 is same

            self.conv1 = DepthwiseSeparableConv(
                            self.out_channels,
                            self.out_channels,
                            kernel_size=3,
                            kernels_per_layer=kernels_per_layer,
                            padding=1)
            
        self.conv2 = DepthwiseSeparableConv(
                        self.out_channels,
                        self.out_channels,
                        kernel_size=3,
                        kernels_per_layer=kernels_per_layer,
                        padding=1)
            
            
        #self.conv2 = conv3x3(self.out_channels, self.out_channels)


    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
      
        from_up = self.upconv(from_up)
        diffY = from_down.size()[2] - from_up.size()[2]
        diffX = from_down.size()[3] - from_up.size()[3]

        from_up = torch.nn.functional.pad(from_up, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2], 'circular')
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        
        x=self.conv1(x)
        x=self.activation(x)
        x=self.norm(x)
        x=self.conv2(x)
        x=self.activation(x)
        x=self.norm(x)
        
        return x
        
        
                                

class UNet_DSC(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding replicate is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    """

    def __init__(self, in_channels=8, out_channels=3, depth=5, start_filts=64, 
                 up_mode='bilinear', merge_mode='concat', activation='ReLU',chl=False, kernels_per_layer=2):

        super(UNet_DSC, self).__init__()

        if up_mode in ('bilinear'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only"
                             "\"upsample\" is allowed.".format(up_mode))
            
        if activation in ('ReLU', 'SiLU'):
            if activation=='ReLU' :
                self.activation = nn.ReLU()
            else : 
                self.activation = nn.SiLU()
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "activation. Only \"SiLU\" and "
                             "\"ReLU\" are allowed.".format(activation))
            
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")


        self.out_channels = out_channels #n_channels
        self.in_channels = in_channels #n_classes
        self.start_filts = start_filts #nb_layers
        self.depth = depth
        self.down_convs = []
        self.up_convs = []
        self.chl=chl




        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False

            down_conv = DownConv(ins, outs, pooling=pooling, activation=self.activation, kernels_per_layer=kernels_per_layer)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode, merge_mode=merge_mode, activation=self.activation, kernels_per_layer=kernels_per_layer)
            self.up_convs.append(up_conv)
            
        self.conv_final = conv1x1(self.start_filts, self.out_channels)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()
        if self.chl==2 or self.chl==3:
            self.relu = nn.ReLU()  # Ensures non-negative output for magnitude
            #self.sigmoid=nn.sigmoid()
            #self.tanh = nn.Tanh()  # Ensures output between -1 and 1 for sign
            
            
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, x):
            
        
        encoder_outs = []         
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)

            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
        
        logits = self.conv_final(x)
        
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

if __name__ == "__main__":
    """
    testing
    """
    #from torchsummary import summary
    model = UNet_DSC(depth=4, in_channels=24, out_channels=4, merge_mode='concat', start_filts=32, activation='ReLU',chl=4, kernels_per_layer=1)
    model.to(device='cuda')    
    
    #block_input=Input_Block(8, 32)
    #summary(model, input_size=(8,128,128))
    
    # input_names = ['Sentence']
    # output_names = ['yhat']
    a=torch.Tensor(np.random.random(((1,24,360,100)))).to(device='cuda')
    # torch.onnx.export(model, a, 'test0.onnx', input_names=input_names, output_names=output_names)
    b=model(a)