#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:48:36 2022

@author: lollier



"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         Cost functions                                | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import torch
import numpy as np
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

"""
généralement, index désigne l'index dans le tuple sorti par le réseau (psc,chl) par exemple 0 ou 1 ou 2 si (psc, chl, chl_anom)
index_chl désigne l'index du channel sur lequel on veut contrôler, par exemple (psc, chl) : index=1 et index_chl=None
mais si chl a plusieurs channel, on pourrait avoir index_chl=0 ou 1 (anomalie) ou -1  (signe)

"""
class MSE_Loss_masked(torch.nn.Module):

    def __init__(self,index,index_chl=None,take_abs=False):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction='none')
        self.index=index
        self.index_chl=index_chl
        self.take_abs=take_abs
        
    def forward(self, prediction, target, reduction='mean'):
        """
        Parameters
        ----------
        prediction : torch tensor
        target : torch tensor
        reduction : none or mean, optional
            if none, return the mse on each pixel,
            else return the mean over all pixels.
            The default is 'mean'..
        """
        prediction=torch.squeeze(prediction[self.index][:,self.index_chl])
        target=torch.squeeze(target[self.index][:,self.index_chl])
        
        #dans le cas où on contrôle la valeur de l'anomalie
        if self.take_abs:
            target=torch.abs(target)
        
        # Mask out the NaN values in target
        valid_mask = ~torch.isnan(target)
        
        # If all target values are NaN, return 0.0
        if torch.sum(valid_mask) == 0:
            if reduction == 'none':
                return torch.zeros_like(prediction)
            return torch.tensor(0.0, device=prediction.device)

        # Apply mask to the prediction and target
        pred_masked = prediction * valid_mask
        target_masked = torch.nan_to_num(target)

        if reduction == 'none':
            return self.mse(pred_masked, target_masked)

        # Compute the mean over non-NaN values
        mse_loss = self.mse(pred_masked, target_masked)
        return torch.sum(mse_loss) / valid_mask.sum()


class HingeLoss_masked(torch.nn.Module):
    def __init__(self, index_sign=-1, index_chl=0):
        super(HingeLoss_masked, self).__init__()
        self.index_sign=index_sign
        self.index_chl=index_chl

        
    def forward(self, prediction, target):
        # Ensure the targets are -1 or 1 for hinge loss
        
        prediction=prediction[self.index_chl][:,self.index_sign]
        target=target[self.index_chl]
        
        # Mask out the NaN values in target
        valid_mask = ~torch.isnan(target)
        
        # If all target values are NaN, return 0.0
        if torch.sum(valid_mask) == 0:
            return torch.tensor(0.0, device=prediction.device)
        
        target = torch.sign(target)
        loss = torch.clamp(1 - prediction * target, min=0)
        return torch.sum(loss)/valid_mask.sum()
    
class BCELoss_masked(torch.nn.Module):
    def __init__(self, index, index_sign=-1):
        super(BCELoss_masked, self).__init__()
        self.index=index
        self.index_sign = index_sign
        #self.weights=np.load("/home/luther/Documents/npy_data/chl_snr.npy")
        
    def forward(self, prediction, target):
        # Select the specified indices for prediction and target
        prediction=torch.squeeze(prediction[self.index][:,self.index_sign])

        target=torch.squeeze(target[self.index])
        # Mask out NaN values in the target
        valid_mask = ~torch.isnan(target)
        
        # If all target values are NaN, return 0.0 loss
        if torch.sum(valid_mask) == 0:
            return torch.tensor(0.0, device=prediction.device)
        #print(torch.sum(target>0),torch.sum(target<0))

        # Convert targets to binary (0 or 1) based on sign
        # Assume positive values are 1, and non-positive values (including -1) are 0
        target_binary = torch.where(target > 0, torch.tensor(1.0, device=target.device), torch.tensor(0.0, device=target.device))
        #prediction_binary = torch.where(prediction > 0, torch.tensor(1.0, device=prediction.device), torch.tensor(0.0, device=prediction.device))

        # Apply valid_mask to the prediction and target
        prediction_binary = prediction[valid_mask]
        target_binary = target_binary[valid_mask]
        
        #if len(target.shape)>2 : 
            #weights=np.stack([self.weights]*target.shape[0],axis=0)
        #else :
        #    weights=self.weights
        #weights_flattened=torch.tensor(weights,device=target.device)[valid_mask]

        # Calculate BCE loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction_binary, target_binary, reduction='none')#/weights_flattened
        
        return loss.mean()
    
class RMSE_Loss_masked(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = MSE_Loss_masked()

    def forward(self, prediction, target, reduction='mean'):
        """
        Parameters
        ----------
        prediction : torch tensor
        target : torch tensor
        reduction : none or mean, optional
            if none, return the rmse on each pixel,
            else return the mean over all pixels.
            The default is 'mean'..
        """
        return torch.sqrt(self.mse(prediction, target, reduction=reduction))

class CHL_underestimation(torch.nn.Module):

    """
    Sert à réduire la sous-estimation des fortes valeurs de chl, 
    prend la sous-estimation maximum comme loss,
    il faut réfléchir à un terme à mettre en facteur pour caractériser le nombre
    de valeur sous-estimée "fortes"
    """
    def __init__(self,index=1,bias=0):
        super().__init__()
        self.bias = bias
        self.index=index
    def forward(self, prediction, target):
        """
        Parameters
        ----------
        prediction : torch tensor
        target : torch tensor
        """
        prediction=prediction[self.index]
        target=target[self.index]
        
        diff=target-prediction
        overestimates=torch.nan_to_num(torch.maximum(torch.zeros(diff.size()).to(device=diff.device)+self.bias,diff))
        loss = torch.max(overestimates)

        return loss

class quantile_loss(torch.nn.Module):
    
    def __init__(self, index=1, index_quantile=1, quantile=0.95):
        super().__init__()
        self.quantile=quantile
        self.index=index
        self.index_quantile=index_quantile
        
    def forward(self, prediction ,target):
        """
        Compute the quantile loss between the true targets and predicted quantiles.
    
        Parameters:
            y_true (torch.Tensor): True target values (ground truth).
            y_pred (torch.Tensor): Predicted quantiles.
            quantile (float): Desired quantile level.
    
        Returns:
            torch.Tensor: Quantile loss.
        """
        
        if torch.isnan(target[self.index]).all():
            return torch.tensor(0.0, device=target[self.index].device)  # Return 0 loss if target is full of NaN
        
        
        residual = target[self.index]-prediction[self.index][:,self.index_quantile] 
        loss = torch.max(self.quantile * residual, (self.quantile - 1) * residual)
        return torch.nanmean(loss)




class KL_divergence(torch.nn.Module):
    """
    KL div
    """
    def __init__(self):
        super().__init__()
        self.kl=torch.nn.KLDivLoss(reduction="none")
        
    def forward(self, prediction, target):
        
        prediction=torch.log(prediction[0])
        target=target[0]
        
        if torch.isnan(target).all():
            return torch.tensor(0.0, device=target.device)  # Return 0 loss if target is full of NaN
        
        pred_masked = torch.mul(prediction, ~torch.isnan(target))
        
        
        return 3*torch.sum(self.kl(pred_masked,torch.nan_to_num(target)))/(torch.flatten(prediction).size()[0]-torch.count_nonzero(torch.isnan(target)))

    
if __name__ == '__main__':

    import torch.nn.functional as F
    pred = tuple([torch.ones((16,3, 360, 100)).to(device='cuda')])
    target = tuple([torch.zeros((16, 360, 100)).to(device='cuda')])

    loss = BCELoss_masked(index=0, index_sign=-1)
    t=torch.nn.BCELoss()
    # pred_s=F.softmax(pred,1)
    # target_s=F.softmax(target,1)
    
    b = loss(pred, target)


    print(b.shape,)
