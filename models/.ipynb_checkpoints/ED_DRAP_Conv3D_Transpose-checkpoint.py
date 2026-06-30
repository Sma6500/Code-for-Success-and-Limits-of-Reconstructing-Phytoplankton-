#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:08:54 2022

@author: Luther Ollier from paper :
    H. Che, D. Niu, Z. Zang, Y. Cao and X. Chen, "ED-DRAP: Encoder–Decoder Deep Residual Attention Prediction Network for Radar Echoes," in IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1-5, 2022, Art no. 1004705, doi: 10.1109/LGRS.2022.3141498.

- Unet related modules
- Attentions modules
- ED-DRAP

"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         ED-DRAP                                       | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
import numpy as np
from torch.nn.modules.utils import _triple
import math
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                        Conv 2D+1                                      | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #


class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input 
    planes with distinct spatial and time axes, by performing a 2D convolution over the 
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time 
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), stride=1, padding=1, bias=True
                 , bn=True, activation='ReLU'):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula 
        # from the paper section 3.5
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                            (kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        # the spatial conv is effectively a 2D conv due to the 
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        
        if bn : 
            self.bn = nn.BatchNorm3d(intermed_channels)
        if activation=='ReLU':
            self.relu = nn.ReLU()

        # the temporal conv is effectively a 1D conv, but has batch norm 
        # and ReLU added inside the model constructor, not here. This is an 
        # intentional design choice, to allow this module to externally act 
        # identical to a standard Conv3D, so it can be reused easily in any 
        # other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)
    
    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x
    

                              
        
def spatial_conv(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True):
    # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
    kernel_size = _triple(kernel_size)
    stride = _triple(stride)
    padding = _triple(padding)
       
    spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
    spatial_stride =  [1, stride[1], stride[2]]
    spatial_padding =  [0, padding[1], padding[2]]
    


    # the spatial conv is effectively a 2D conv due to the 
    # spatial_kernel_size, followed by batch_norm and ReLU
    spatial_conv = nn.Conv3d(in_channels, out_channels, spatial_kernel_size,
                                stride=spatial_stride, padding=spatial_padding, bias=bias)
    return spatial_conv

def temporal_conv(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True):
    
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)


        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]
        
        temporal_conv = nn.Conv3d(in_channels, out_channels, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)
        return temporal_conv
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                        U-net modules                                  | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #



def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        padding_mode='replicate',
        bias=bias,
        groups=groups)

def upsample(in_channels, out_channels, mode='bicubic'):
        # out_channels is always going to be the same
        # as in_channels
        return nn.Upsample(mode='trilinear', scale_factor=(1,2,2))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

def batch_norm(in_channels):
    return nn.BatchNorm3d(in_channels)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU/SiLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True, activation=nn.ReLU()):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.activation=activation

        self.intermed_channels = int(math.floor((27 * in_channels * out_channels)/(9 * in_channels + 3 * out_channels)))
        self.norm = batch_norm(self.intermed_channels)

        self.conv1 = spatial_conv(self.in_channels, self.intermed_channels, (3,3,3))
        self.conv2 = temporal_conv(self.intermed_channels, self.out_channels, (3,3,3))

        if self.pooling:
            self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

    def forward(self, x):

        x = self.activation(self.norm(self.conv1(x)))
        x = self.activation(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x#, before_pool
    
    
class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU/SiLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, up_mode='transpose', activation=nn.ReLU()):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_mode = up_mode
        self.norm = batch_norm(out_channels)
        self.activation=activation

        self.intermed_channels = int(math.floor((27 * self.in_channels * out_channels)/(9 * self.in_channels + 3 * out_channels)))
                    
        self.upsample = upsample(self.in_channels, self.out_channels, mode=self.up_mode)
        self.conv1 = spatial_conv(self.in_channels, self.intermed_channels, (3,3,3))
        self.conv2 = temporal_conv(self.intermed_channels, self.out_channels, (3,3,3))




    def forward(self, x):
        x = self.upsample(x)
        x= self.conv1(x)
        x = self.activation(x)
        x = self.activation(self.norm(self.conv2(x)))
        #x = self.activation(self.conv2(x))
        return x
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                        Attention modules                              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

class SEA(nn.Module):
    """
    3-D Sequence attention module SEA : 
    performs 3-D average pooling to contract spatial info and keep temporal, two conv 3D and sigmoid 
    join a weight to each time step in order to capture the most important time related features
    """
    
    def __init__(self, channels, image_size):
        super(SEA, self).__init__()
        
        self.channels=channels
        self.kernel_size=(1,image_size, image_size)
        self.average_pool=nn.AvgPool3d(kernel_size=self.kernel_size) #1,ikmage size ?
        self.sigmoid=nn.Sigmoid()
        
        self.intermed_channels = int(math.floor((27 * self.channels * self.channels)/(9 * self.channels + 3 * self.channels)))
        self.conv1 = nn.Sequential(spatial_conv(self.channels, self.intermed_channels, (3,3,3)),
                                   nn.ReLU(),
                                   temporal_conv(self.intermed_channels, self.channels, (3,3,3))
                                   )
        self.conv2 = nn.Sequential(spatial_conv(self.channels, self.intermed_channels, (3,3,3)),
                                   nn.ReLU(),
                                   temporal_conv(self.intermed_channels, self.channels, (3,3,3))
                                   )
        
        
    def forward(self, x):
        
        out=self.average_pool(x)
        out=self.conv1(out)
        out=self.conv2(out)
        out=self.sigmoid(out)
        
        return out+x
              
            
class SPA(nn.Module):
    """
    3-D Spatial attention module SPA : 
    performs 3-D conv to contract time related information and sigmoid to join a 
    weight to each pixel to capture the most important spatial features
    """
    def __init__(self, channels, time_size):
        super(SPA, self).__init__()
        self.channels=channels
        self.kernel=(time_size,1,1)
        self.intermed_channels = int(math.floor((27 * self.channels * self.channels)/(9 * self.channels + 3 * self.channels)))

        self.conv1 = nn.Sequential(spatial_conv(self.channels, self.intermed_channels, (3,3,3)),
                                   nn.ReLU(),
                                   temporal_conv(self.intermed_channels, self.channels, (3,3,3))
                                   )  
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,x):
        
        out=self.sigmoid(self.conv1(x))
        
        return out+x
        
class SSAB(nn.Module):
    """
    Sequence and Spatial Attention Block
    """
    def __init__(self, channels, input_size):
        super(SSAB, self).__init__()
        self.channels=channels
        self.time_size=input_size[0]
        
        if input_size[1]!=input_size[2]:
            raise ValueError(f"image size is not squared anymore {input_size}")
            
        self.image_size=input_size[1]
        self.intermed_channels = int(math.floor((27 * self.channels * self.channels)/(9 * self.channels + 3 * self.channels)))

        
        self.conv1 = nn.Sequential(spatial_conv(self.channels, self.intermed_channels, (3,3,3)),
                                   nn.ReLU(),
                                   temporal_conv(self.intermed_channels, self.channels, (3,3,3))
                                   ) 
        self.conv2 = nn.Sequential(spatial_conv(self.channels, self.intermed_channels, (3,3,3)),
                                   nn.ReLU(),
                                   temporal_conv(self.intermed_channels, self.channels, (3,3,3))
                                   )
        
        self.sea = SEA(self.channels, self.image_size)
        self.spa = SPA(self.channels, self.time_size)
        
    def forward(self, x):
        
        out=self.conv1(x)
        out=self.conv2(out)
        out=self.sea(out)
        out=self.spa(out)
        
        return out+x
    

class RSSAB(nn.Module):
    """
    performs M-SSAB
    the original paper doesnt mention any number so it's basically a parameter to tune
    """
    def __init__(self, n_ssab, channels, input_size):
        super(RSSAB, self).__init__()
        self.input_size=input_size #note : à voir si c'est pas mieux de séparer directement ici
        self.n_ssab=n_ssab
        self.channels=channels
        
        self.ssabs=[]
        for i in range(self.n_ssab):
            ssab=SSAB(self.channels, self.input_size)
            self.ssabs.append(ssab)
        self.intermed_channels = int(math.floor((27 * self.channels * self.channels)/(9 * self.channels + 3 * self.channels)))

        self.conv=nn.Sequential(spatial_conv(self.channels, self.intermed_channels, (3,3,3)),
                                   nn.ReLU(),
                                   temporal_conv(self.intermed_channels, self.channels, (3,3,3))
                                   ) 
        self.ssabs=nn.ModuleList(self.ssabs)
        
    def forward(self, x):
        
        for i, module in enumerate(self.ssabs):
            out=module(x) if i==0 else module(out)
        
        out=self.conv(out)
        
        return out+x

class N_RSSAB(nn.Module):
    """
    performs N RSSAB
    """
    def __init__(self, n_rssab, channels, input_size, n_ssab=1):#ya marqué nulle part dans le papier combien de ssab ils mettent
        super(N_RSSAB, self).__init__()
        self.input_size=input_size #note : à voir si c'est pas mieux de séparer directement ici
        self.n_rssab=n_rssab
        self.channels=channels
        self.n_ssab=n_ssab
        
        self.rssabs=[]
        for i in range(self.n_rssab):
            rssab=RSSAB(self.n_ssab, self.channels, self.input_size)
            self.rssabs.append(rssab)
            
        self.intermed_channels = int(math.floor((27 * self.channels * self.channels)/(9 * self.channels + 3 * self.channels)))
 
        self.conv=nn.Sequential(spatial_conv(self.channels, self.intermed_channels, (3,3,3)),
                                   nn.ReLU(),
                                   temporal_conv(self.intermed_channels, self.channels, (3,3,3))
                                   )
        self.rssabs=nn.ModuleList(self.rssabs)
        
    def forward(self, x):
        
        for i, module in enumerate(self.rssabs):
            x=module(x) 
        
        x=self.conv(x)
        
        return x

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                        ED-DRAP                                        | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
    
class ED_DRAP(nn.Module):
    """ `ED-DRAP` class is based on https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9674896&isnumber=9651998

    The ED-DRAP is a convolutional encoder-decoder neural network 
    with deep attention modules.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Deep spatial and temporal attentions modules are added 
    in the decoding pathway to gather the essential information.
    
    Implementaion of the original paper is not available yet 
    but I give here some details on the present file :
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels --> padding mode = replicate
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        (merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    (5) 3 attention block are present, the first one before the first up convolution
        the others after each up convolution
    (6) As the network is residual (meaning that it adds the last input image to the output)
        we absolutely need to have more or same numbers of inputs channels than outputs.
        If there is less outputs channels, it adds the first input channels corresponding to
        the correct number
    """

    def __init__(self, size_input=(10,128,128), activation='ReLU',
                 channel_input=1, channel_output=1, filters_list=[8,16,32]):

        super(ED_DRAP, self).__init__()
            
        if activation in ('ReLU', 'SiLU'):
            if activation=='ReLU' :
                self.activation = nn.ReLU()
            else : 
                self.activation = nn.SiLU()
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "activation. Only \"SiLU\" and "
                             "\"ReLU\" are allowed.".format(activation))
        if channel_output>channel_input:
            raise ValueError("can't add residul frame on different numbers of channels"
                             "please check that number of inputs channels is greater than or equal to"
                             "the number of output channels")

        self.num_classes=channel_output #attributes needed in the the model class
        self.nb_inputs=channel_input #same idea, all the code needs to be cleaned
        self.channel_input = channel_input
        self.channel_output = channel_output
        self.filters_list=filters_list
        self.size_input=size_input
        
        
        #list module :
            #Down way
        self.norm=batch_norm(self.channel_input)
        self.intermed_channels_0 = int(math.floor((27 * self.channel_input * self.filters_list[0])/(9 * self.channel_input + 3 * self.filters_list[0])))

        self.first_conv=nn.Sequential(spatial_conv(self.channel_input, self.intermed_channels_0, (3,3,3)),
                                   nn.ReLU(),
                                   temporal_conv(self.intermed_channels_0, self.filters_list[0], (3,3,3))
                                   ) 
        self.Down_conv1=DownConv(self.filters_list[0], self.filters_list[1])
        self.Down_conv2=DownConv(self.filters_list[1], self.filters_list[2])

            #upway
        self.Up_conv1=UpConv(self.filters_list[2], self.filters_list[1])
        self.Up_conv2=UpConv(self.filters_list[1], self.filters_list[0])
        self.intermed_channels_1 = int(math.floor((27 * self.channel_output * self.filters_list[0])/(9 * self.filters_list[0] + 3 * self.channel_output)))

        self.final_conv=nn.Sequential(spatial_conv(self.filters_list[0], self.intermed_channels_1, (3,3,3)),
                                   nn.ReLU(),
                                   temporal_conv(self.intermed_channels_1, self.channel_output, (3,3,3))
                                   ) 

            #Attention 
            
        self.attention1=RSSAB(4, self.filters_list[2], (self.size_input[0], self.size_input[1]//4, self.size_input[2]//4))
        self.attention2=RSSAB(2, self.filters_list[1], (self.size_input[0], self.size_input[1]//2, self.size_input[2]//2))
        self.attention3=RSSAB(1, self.filters_list[0], self.size_input)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, x):
        
        skip_connection=[]
        last_frame=torch.stack([x[:,:self.channel_output,-1]]*x.size()[2], axis=2)
                               
        x=self.norm(x)

        x=self.first_conv(x)

        skip_connection.append(x)

        x=self.Down_conv1(x)

        skip_connection.append(x)
        
        x=self.Down_conv2(x)

        skip_connection.append(x)

        x=self.attention1(x)
        
        x=self.Up_conv1(x+skip_connection[-1])

        x=self.attention2(x)

        x=self.Up_conv2(x+skip_connection[-2])

        x=self.attention3(x)

        x=self.final_conv(x+skip_connection[-3])

    
        return x+last_frame
    
    
if __name__=="__main__":
    
    #from torchsummary import summary
    # model = N_RSSAB(4, 8, (10,75,75))
    # model.to(device='cuda')

    model=ED_DRAP(size_input=(10,128,128), activation='ReLU',
                 channel_input=2, channel_output=2, filters_list=[8,16,32])
    model.to(device=0)
    # input_names = ['Sentence']
    # output_names = ['yhat']
    a=torch.Tensor(np.random.random(((1,2,10,32,32)))).to(device='cuda')
    # torch.onnx.export(model, a, 'test0.onnx', input_names=input_names, output_names=output_names)
    
    

    
    
    #177 266
