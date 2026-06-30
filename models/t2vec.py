#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 10:24:07 2024

@author: luther

inspired from https://github.com/andrewrgarcia/time2vec/blob/main/time2vec/torch/time2vec_torch.py

changed channel last  for channel first (or second with batch)

modified trend to make a trend component for each pixel of inputed maps.
same for periodic weights

Removed the Matmul from the original implementation, idk why it has been put in first place, to investigate..
in the original implementation, initialization is done in the first forward, but it creates a conflict
with the .to(device).
"""

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         Time 2 Vec                                    | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

import torch
import torch.nn as nn
import torch.nn.functional as F

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #


class Time2Vec(nn.Module):
    def __init__(self, num_frequency, num_vars, input_shape):
        """
        Custom PyTorch object for Time2Vec Transformation.

        Parameters
        -------------
        num_frequency: int
            The number of periodic components to model.
        num_vars: int
            The number of variables to consider from the input.

        """
        super(Time2Vec, self).__init__()
        self.num_frequency = num_frequency
        self.num_vars = num_vars
        
        # Initialize periodic/trend weights and bias based on input shape        
        #periodic
        self.periodic_weight = nn.Parameter(torch.Tensor(1,self.num_frequency, *input_shape[-2:]))
        self.periodic_bias = nn.Parameter(torch.Tensor(1,self.num_frequency, *input_shape[-2:]))
        
        # first periodic weight constraints
        # if self.num_frequency>=1:
        #     self.first_periodic_weight_constraint=(-torch.pi*2*0.15,torch.pi*2*0.15)
        #     print("weight constraint for first periodic weight corresponds to +/- 0.15 so 1 year and half if 0.1 is one year")
            
        # second periodic weight constraints
        # if self.num_frequency>=2:
        #     self.second_periodic_weight_constraint=(torch.pi*2*0.2,torch.pi*2*1.)
        #     print("weight constraint for second periodic weight corresponds to 0.15/1. with 0.1 for one year")
            
        nn.init.uniform_(self.periodic_weight)
        nn.init.uniform_(self.periodic_bias)
    
        #trend
        self.trend_weight = nn.Parameter(torch.Tensor(1,1, *input_shape[-2:]))
        self.trend_bias = nn.Parameter(torch.Tensor(1,1, *input_shape[-2:]))
        
        #self.trend_weight_constraints=(-1,1)
        
        nn.init.uniform_(self.trend_weight,-1.,1.)
        nn.init.uniform_(self.trend_bias,-1,1.)
        

    # def periodic_constraints(self,omega,constraints):
        
    #     a=constraints[0]
    #     b=constraints[1]
        
    #     return (torch.sigmoid(omega)*(b-a))-a
        

    def forward(self, inputs):
        # Split inputs into x and t
        x = inputs[:, :self.num_vars-1, :, :]
        t = inputs[:, self.num_vars-1:, :,:]
        
        #self.trend_weight=self.periodic_constraints(self.trend_weight, self.trend_weight_constraints)
        trend_component = self.trend_weight * t + self.trend_bias
        
        # if self.num_frequency>=1:
        #     self.periodic_weight[:,0]=self.periodic_constraints(self.periodic_weight[:,0], self.first_periodic_weight_constraint)
            
        # if self.num_frequency>=2:
        #     self.periodic_weight[:,1]=self.periodic_constraints(self.periodic_weight[:,1], self.second_periodic_weight_constraint)

        # Periodic component
        periodic_component = torch.sin(t*self.periodic_weight + self.periodic_bias)

        # Concatenate trend and periodic components
        t_encoded = torch.cat([trend_component, periodic_component], dim=1)

        # Concatenate x and t_encoded
        output = torch.cat([x, t_encoded], dim=1)
        
        return output
    
    

if __name__ == '__main__':
    
    import numpy as np
    
    mask=np.load("/home/luther/Documents/npy_data/bath.npy")
    mask=~np.isnan(mask)
    
    physics=np.load("/home/luther/Documents/npy_data/physics/processed_physics_1993_2023_8d_lat50_100.npy")
    physics=torch.Tensor(physics[:,:4])
    
    time=np.ones(physics.shape)[:,0]*mask
    
    print('40 years used for time encoding')

    for t in range(time.shape[0]):
        
        time_encoder=(t+1)/(40*46)
        time[t]=time[t]*time_encoder
    
    
    t2v=Time2Vec(2, 5, (100,360))
    t2v.to('cuda')
    
    inputs=torch.concat((physics,torch.unsqueeze(torch.Tensor(time),axis=1)),dim=1).to('cuda')
    
    outputs=t2v(inputs[:32])
    