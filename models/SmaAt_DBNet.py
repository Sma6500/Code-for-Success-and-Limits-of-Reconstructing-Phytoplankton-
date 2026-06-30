#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:27:17 2023

@author: luther

Pareil que le DualBranchNet mais avec un SmaAt-UNet à la place de l'UNet simple
"""


# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                        SmaAt-DBNET                                    | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
import numpy as np
import torch.nn.functional as F
from SmaAt_UNet import SmaAt_UNet  
from UNet import batch_norm, conv1x1, conv3x3
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #


class Output_Block(nn.Module):
    """
    A postprocessing module that performs :
        - two convolution 
        - a batch norm
        - an activation 
    """
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(), n_hidden=32):
        
        super(Output_Block, self).__init__()
        self.norm=batch_norm(out_channels)
        self.conv=nn.Sequential(conv3x3(in_channels, n_hidden),
                                conv3x3(n_hidden, out_channels))
        self.activation=activation
        
    def forward(self, x):
        
        x=self.conv(x)
        x=self.norm(x)
        x=self.activation(x)
        
        return x
        
    

class SmaAt_DualBranchNet(nn.Module):
    """ 
    VERY SPECIFIC DESIGN
    SmaAt-UNet followed by two output block, one with softmax activation to produce %PSC
    one with Relu+conv1*1 activation to produce CHL
    """

    def __init__(self, in_channels=8, end_filts=32, up_mode='bilinear',
                 activation='ReLU', freeze_key_id='0', kernels_per_layer=2):
        
        
        super(SmaAt_DualBranchNet, self).__init__()
        
        if activation in ('ReLU', 'SiLU'):
            if activation=='ReLU' :
                self.activation = nn.ReLU()
            else : 
                self.activation = nn.SiLU()
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "activation. Only \"SiLU\" and "
                             "\"ReLU\" are allowed.".format(activation))
            
        if up_mode in ('bilinear'):
            self.up_mode = True
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "up mode. Only \"bilinear\" is allowed.".format(up_mode))
            
        #https://jimmy-shen.medium.com/pytorch-freeze-part-of-the-layers-4554105e03a6
        # This is for a Graph Neural Network based on the GIN paper
        self._FREEZE_KEY = {'0': [],
                            '1': ['main_block','chl_block'], #if freeze_key_id is 1 freeze the Unet block and the Chl
                            }
        
        #depth is always 5 and starting channels 64, upmode bilinear
        self.main_block=SmaAt_UNet(n_channels=in_channels, n_classes=end_filts,  
                                            kernels_per_layer=kernels_per_layer, bilinear=True, reduction_ratio=16,
                                            activation=self.activation)
        
        self.psc_block=Output_Block(in_channels=end_filts, out_channels=3, 
                                    activation=nn.Softmax(1))
        
        # if activation in ('ReLU', 'SiLU'):
        #     if activation=='ReLU' :
        #         self.chl_activation = nn.ReLU()
        #     else : 
        #         self.chl_activation = nn.SiLU()

        self.chl_block=Output_Block(in_channels=end_filts, out_channels=16, 
                                    # activation=nn.Sequential(self.chl_activation,
                                    #                          conv1x1(16, 1)))
                                     activation=nn.Sequential(conv1x1(16, 1)))
        
        self.freeze_model_weights(freeze_key_id=freeze_key_id)
        
    def freeze_model_weights(self, freeze_key_id="0"):
        """
        freeze the model weights based on the layer names
        for example, if the layers contain '0', then if the name of the parameter contains _FREEZE_KEY['0'] will be frozen
        """
        print('Going to apply weight frozen')
        print('before frozen, require grad parameter names:')
        for name, param in self.named_parameters():
            if param.requires_grad:print(name)
        freeze_keys = self._FREEZE_KEY[freeze_key_id]
        print('freeze_keys', freeze_keys)
        for name, para in self.named_parameters():
            if para.requires_grad and any(key in name for key in freeze_keys):
                para.requires_grad = False
        print('after frozen, require grad parameter names:')
        for name, para in self.named_parameters():
            if para.requires_grad:print(name)
        print("\n________________Done___________________\n")
        
    def parameters(self):
        for name, param in self.named_parameters():
            if not(param.requires_grad):
                continue
            yield param
            
    def forward(self, x):
        
        out=self.main_block(x)
        
        chl=self.chl_block(out)
        psc=self.psc_block(out)
        
        return (psc, chl)
    
    


if __name__ == "__main__":
    """
    testing
    """
    model = SmaAt_DualBranchNet()
    model.to(device='cuda')

    a=torch.Tensor(np.random.random(((2,8,100,360)))).to(device='cuda')

    b=model(a)