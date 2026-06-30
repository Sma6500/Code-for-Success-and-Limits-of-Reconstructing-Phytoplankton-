#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:38:51 2025

@author: luther
"""

### transformations
import torch
from torchvision.transforms import functional as F
import numpy as np

class CustomCrop:
    def __call__(self, tensor):
        # Slice the required parts (Pacific)
        slice1 = tensor[..., 20:80, 330:] #popopo trop stylé les ...
        slice2 = tensor[..., 20:80, 0:90]
        
        # Concatenate along the longitude axis
        cropped_tensor = torch.cat((slice1, slice2), dim=-1)
        
        return cropped_tensor


class ReverseCustomCropNumpy:
    def __call__(self, cropped_array, original_shape=(100,360)):
        # Initialize the array with np.nan
        restored_array = np.full((*cropped_array.shape[:-2],*original_shape), np.nan)
        
        # Place the cropped values back in their original positions
        restored_array[:, :, 20:80, 0:90] = cropped_array[..., 30:]
        restored_array[:, :, 20:80, 330:] = cropped_array[..., :30]
        
        return restored_array
    
class CustomMask:
    def __call__(self, tensor):
        # Slice the required parts (Pacific)
        tensor[..., 60:] = 0.#torch.nan
        tensor[..., :30] = 0.#torch.nan
        
        return tensor
    
class CustomMask1:
    def __call__(self, tensor):
        # Slice the required parts (Pacific)
        tensor[..., 50:] = 0.#torch.nan
        tensor[..., :40] = 0.#torch.nan
        
        return tensor
