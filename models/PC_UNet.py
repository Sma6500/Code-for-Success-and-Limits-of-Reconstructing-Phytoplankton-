#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:30:36 2023

@author: luther
largely inspired from : 

https://github.com/milesial/Pytorch-UNet/tree/master
https://github.com/jaxony/unet-pytorch

padding mode is replicate

"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                               Partial convolution UNET                                | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import numpy as np
from Partialconv_2d import PartialConv2d

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                       UNET PART                                       | #
# |                                                                                       | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

def pconv3x3(in_channels, out_channels, kernel_size=3):
    return PartialConv2d(in_channels, out_channels, kernel_size=kernel_size, 
                         stride=1, padding=1, bias=False, multi_channel=True, return_mask=True)

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(in_channels,out_channels, kernel_size=1, groups=groups, stride=1)

def upconv2x2(in_channels, out_channels, mode='bilinear'):
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def bn(in_channels):
    return nn.BatchNorm2d(in_channels)

class DownPConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    Mask is maxpooled too but I didn't find any informations on what's the right way to downscale it.
    Not explained in the paper and others implementation put a strange kernel/stride/padding size to convolution to 
    succeed it.
    A ReLU/SiLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True, batch_norm=True, activation=nn.ReLU()):
        super(DownPConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.batch_norm=batch_norm
        if self.batch_norm:
            self.norm = bn(out_channels)
        self.activation=activation

        self.conv1 = pconv3x3(self.in_channels, self.out_channels)
        self.conv2 = pconv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, mask):

        x, mask=self.conv1(x, mask)
        if self.batch_norm :
            x=self.norm(x)
        x=self.activation(x)

        x, mask=self.conv2(x, mask)
        if self.batch_norm :
            x=self.norm(x)
        x = self.activation(x)

        before_pool = x
        mask_before_pool = mask
        if self.pooling:
            x = self.pool(x)
            #bon aucune idée de si ça marche et j'aime pas l'idée d'utiliser la convolution pour downscaler, à voir 
            #avec Anastase à l'occaz
            mask=self.pool(mask)
        return x, mask, before_pool, mask_before_pool


    
class UpPConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU/SiLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, 
                 merge_mode='concat', up_mode='bilinear',batch_norm=True,activation=nn.ReLU()):
        super(UpPConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.batch_norm=batch_norm
        if self.batch_norm :
            self.norm = bn(out_channels)
        self.activation=activation


        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)
		
            
        if self.merge_mode=='concat':
            self.conv1=pconv3x3(2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = pconv3x3(self.out_channels, self.out_channels)

        self.conv2 = pconv3x3(self.out_channels, self.out_channels)


    def forward(self, from_down, from_up, mask_from_down):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        diffY = from_down.size()[2] - from_up.size()[2]
        diffX = from_down.size()[3] - from_up.size()[3]

        from_up = F.pad(from_up, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2], 'replicate')
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        if self.merge_mode=='concat':
            x = torch.cat((from_up, from_down), 1)
            mask_from_down=torch.cat([mask_from_down,mask_from_down],1)
        else:
            x = from_up + from_down
            
        x, mask=self.conv1(x, mask_from_down) #on upsample pas le mask from up, mais on pourrait, à voir, on peut aussi combiner les 2 masques
        if self.batch_norm :
            x=self.norm(x)
        x=self.activation(x)

        x, mask=self.conv2(x, mask) 
        if self.batch_norm :
            x=self.norm(x)
        x=self.activation(x)
        
        return x, mask
        
    
        
                                

class UNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    """

    def __init__(self, in_channels=8, out_channels=3, depth=5, start_filts=64, 
                 up_mode='bilinear', merge_mode='concat', activation='ReLU', batch_norm=True):

        super(UNet, self).__init__()

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

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.batch_norm=batch_norm
        self.down_convs = []
        self.up_convs = []


        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False

            down_conv = DownPConv(ins, outs, pooling=pooling, batch_norm=self.batch_norm, activation=self.activation)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks

        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpPConv(ins, outs, up_mode=up_mode,
                merge_mode=merge_mode, activation=self.activation, batch_norm=self.batch_norm)
            self.up_convs.append(up_conv)
            
        self.conv_final = conv1x1(self.start_filts, self.out_channels)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        #self.reset_params()

    # @staticmethod
    # def weight_init(m):
    #     if isinstance(m, nn.Conv2d):
    #         init.xavier_normal_(m.weight)
    #         init.constant_(m.bias, 0)


    # def reset_params(self):
    #     for i, m in enumerate(self.modules()):
    #         self.weight_init(m)


    def forward(self, inputs):
            
        x,mask=inputs
        encoder_outs = []         
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, mask, before_pool, mask_before_pool = module(x, mask)

            encoder_outs.append((mask_before_pool, before_pool))

        for i, module in enumerate(self.up_convs):
            mask, before_pool = encoder_outs[-(i+2)]
            x, new_mask = module(before_pool, x, mask)
        
        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        
        x = self.conv_final(x)
        
        return x
    
    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()


if __name__ == "__main__":
    """
    testing
    """
    #from torchsummary import summary
    device='cuda'
    model = UNet(out_channels=1, depth=5, in_channels=7, merge_mode='concat', activation='ReLU', batch_norm=True)
    model.to(device=device)
    
    #block_input=Input_Block(8, 32)
    #summary(model, input_size=(8,128,128))
    
    # input_names = ['Sentence']
    # output_names = ['yhat']
    a=torch.Tensor(np.random.random(((1,7,100,360)))).to(device='cuda')
    mask=np.squeeze(np.load("/home/luther/Documents/npy_data/bath.npy"))
    masks=np.stack([mask]*7)
    masks=torch.Tensor(np.expand_dims(masks,axis=0))
    
    b=model((a.to(device=device), masks.to(device=device)))
    # torch.onnx.export(model, a, 'test0.onnx', input_names=input_names, output_names=output_names)
