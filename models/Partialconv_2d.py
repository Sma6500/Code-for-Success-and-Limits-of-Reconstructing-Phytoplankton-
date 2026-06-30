#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 13:52:57 2023

@author: luther
"""

###############################################################################
#largerly inspired from : https://github.com/NVIDIA/partialconv/tree/master
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################


# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         README                                        | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
"""
partial convolution for inpainting (using multiple channels and updating mask)
How to instantiate : 
PartialConv2d(in channel, output channel, kernel_size=3, stride=1, padding=1, bias=False, multi_channel=True, return_mask=True)

requires to provide the multi-channel mask, for global ocean image, don't forget to put the Lands mask at each step 
except for the non oceanagraphic inputs (such as winds or short-wave radiation)
"""

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                              Partial convolution 2D                                   | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False  

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in
                        
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)


        if self.return_mask:
            return output, self.update_mask
        else:
            return output
        
        
        
if __name__=="__main__":
    
    import numpy as np
    import matplotlib.pyplot as plt
    data=np.load("/home/luther/Documents/npy_data/chl/chl_avw_1998_2020_100k_8d_lat50.npy")
    bathy=np.load("/home/luther/Documents/npy_data/physics/physics_1998_2020_8d_100_lat50.npy")
    mask_continent=torch.unsqueeze(torch.Tensor(~np.isnan(bathy[:32,-1])),axis=1)
    b=torch.unsqueeze(torch.Tensor(np.nan_to_num(data[:32],nan=0.0)),axis=1)
    b_mask=torch.unsqueeze(torch.Tensor(~np.isnan(data[:32])),axis=1)
    padding=1
    net=[PartialConv2d(1 , 32, kernel_size=3, stride=1, padding=padding, return_mask=True, multi_channel=True,bias=False),
         PartialConv2d(32 , 64, kernel_size=3, stride=1, padding=padding, return_mask=True, multi_channel=True,bias=False),
         PartialConv2d(64 , 64, kernel_size=3, stride=1, padding=padding, return_mask=True, multi_channel=True,bias=False),
         PartialConv2d(64 , 32, kernel_size=3, stride=1, padding=padding, return_mask=True, multi_channel=True,bias=False),
         PartialConv2d(32 , 1, kernel_size=3, stride=1, padding=padding, return_mask=True, multi_channel=True,bias=False)]
                      
    device='cuda'
    b.to(device=device)
    b_mask.to(device=device)
    for layer in net:
        print(layer)
        layer.to(device=device)
        b,b_mask=layer(b.to(device),b_mask.to(device))
        plt.imshow(b.cpu().detach().numpy()[0,0])
        plt.colorbar()
        plt.show()
        b_mask=torch.mul(b_mask, mask_continent.to(device))
        
    pool=nn.MaxPool2d(kernel_size=2,stride=2)
    
    
    
    
    
    
    