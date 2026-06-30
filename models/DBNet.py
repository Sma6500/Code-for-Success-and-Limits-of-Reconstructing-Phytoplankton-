#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 11:06:32 2022

@author: lollier

"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                        DBNET                                          | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
import numpy as np
import torch.nn.functional as F
from UNet import UNet, batch_norm, conv1x1, conv3x3
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
        
    

class DualBranchNet(nn.Module):
    """ 
    VERY SPECIFIC DESIGN
    UNet followed by two output block, one with softmax activation to produce %PSC
    one with Relu+conv1*1 activation to produce CHL
    """

    def __init__(self, in_channels=8, depth=5, start_filts=64, end_filts=32, up_mode='bilinear', 
                 merge_mode='concat', activation='ReLU', freeze_key_id='0'):
        
        super(DualBranchNet, self).__init__()
        
        if activation in ('ReLU', 'SiLU'):
            if activation=='ReLU' :
                self.activation = nn.ReLU()
            else : 
                self.activation = nn.SiLU()
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "activation. Only \"SiLU\" and "
                             "\"ReLU\" are allowed.".format(activation))
            
        #https://jimmy-shen.medium.com/pytorch-freeze-part-of-the-layers-4554105e03a6
        # This is for a Graph Neural Network based on the GIN paper
        self._FREEZE_KEY = {'0': [],
                            '1': ['main_block','chl_block'], #if freeze_key_id is 1 freeze the Unet block and the Chl
                            }
        
        self.main_block=UNet(in_channels=in_channels, out_channels=end_filts, depth=depth, 
                                   start_filts=start_filts, up_mode=up_mode, 
                                   merge_mode=merge_mode, activation=activation)
        
        
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
    model = DualBranchNet(depth=5, in_channels=8, merge_mode='concat', activation='ReLU', freeze_key_id='0')
    model.to(device='cuda')

    a=torch.Tensor(np.random.random(((16,8,100,360)))).to(device='cuda')

    b=model(a)