#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:00:59 2022

@author: lollier*

model class to pass to the trainer class

- instancie un modèle d'architecture sur device
- instancie la/les fonctions de couts et l'optimizer
- run le forward à partir des inputs et targets et renvoie la prediction ou la loss
"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         MODEL                                         | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import torch
from utils.losses import MSE_Loss_masked, CHL_underestimation, KL_divergence, quantile_loss, HingeLoss_masked, BCELoss_masked
from torch.cuda.amp import GradScaler

#from lion_pytorch import Lion
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
class model(): 
    """
    TO DO :
        - handle parallelized training
        - think on a nicer way to implement criterion and __init_criterion__ *
        
        
        *c'est un peu mieux mais rien de dingue, on boucle sur pred et target pour des loss sur plusieures sorties
        et si jamais on veut mettre plusieurs loss sur une seule sortie il faut la pimper dans losses.py
    """
    
    def __init__(self, net, device, criterion_config, optimizer_config):
        
        """
        Parameters
        ----------
        net : nn torch network
            torch network from the model folder.
        device : 'cpu','cuda',0 or 1
            device for the net, data and forward loop
        criterion_config : dictionnary
            see config.py.
        optimizer_config : dictionnary
            see config.py.

        """
        
        self.device = device #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = net.to(self.device)
                
        #init optim and losses
        self.optimizer= self.__init_optimizer__(optimizer_config)
        self.losses, self.losses_weights, self.losses_names=self.__init_criterion__(criterion_config)
        
        #parameter for Automated mixed precision (see https://pytorch.org/docs/stable/notes/amp_examples.html)
        #for float16
        self.scaler=GradScaler()
        
    
    def __init_criterion__(self, criterion_config):
        """
        Parameters
        ----------
        criterion_config : dict
        configure de losses required for training.
        loop on preds and targets (it must be tuples) and apply at each index the corresponding losses
        therefor it needs to be ordered
        Returns
        -------
        """
        if "details" in criterion_config.keys():
            self.details=criterion_config['details']
        else : self.details=False
            
        losses=[]
        weights=[]
        losses_names=[]
        
        #MSE
        if 'MSE_psc' in criterion_config.keys() and criterion_config['MSE_psc']!=0.:

            losses.append(MSE_Loss_masked(0))
            weights.append(criterion_config['MSE_psc'])
            losses_names.append('MSE_psc')
                    
        #self.loss_chl=False 

        if 'MSE_chl' in criterion_config.keys() and criterion_config['MSE_chl']!=0.:
            
            index_loss=1 if len(losses)>=1 else 0
            print(f'index loss mse chl :{index_loss}')
            losses.append(MSE_Loss_masked(index_loss))
            weights.append(criterion_config['MSE_chl'])
            losses_names.append('MSE_chl')
            #self.loss_chl=True 
        
        if 'MSE_chl_anom' in criterion_config.keys() and criterion_config['MSE_chl_anom']!=0.:
            
            index_loss=2 if len(losses)>=2 else 1
            print(f'index loss mse chl anom :{index_loss}')
            losses.append(MSE_Loss_masked(index_loss,index_chl=0,take_abs=True))
            weights.append(criterion_config['MSE_chl_anom'])
            losses_names.append('MSE_chl_anom')
            #self.loss_chl=True 
    
            
        if 'quantile_loss' in criterion_config.keys() and criterion_config['quantile_loss']!=0.:
            
            index_loss=1 if len(losses)>=1 else 0
            if isinstance(criterion_config['quantile_loss'], tuple) :
                for i,(quantile,quantile_weight) in enumerate(criterion_config['quantile_loss']):

                    losses.append(quantile_loss(index=index_loss,index_quantile=i,quantile=quantile))
                    weights.append(quantile_weight)
                    losses_names.append(f'quantile_loss q{quantile}')
                    
        if 'BCELoss' in criterion_config.keys() and criterion_config['BCELoss']!=0.:
            index_loss=2 if len(losses)>=3 else 1
            print(f'index loss sign chl anom :{index_loss}')
            losses.append(BCELoss_masked(index=index_loss, index_sign=-1))
            weights.append(criterion_config['BCELoss'])
            losses_names.append('BCELoss')

        if 'HingeLoss' in criterion_config.keys() and criterion_config['HingeLoss']!=0.:
            
            print("Hingeloss configured with classic index (-1 for sign and 0 for chl)")
            losses.append(HingeLoss_masked(index_sign=-1,index_chl=-1))
            weights.append(criterion_config['HingeLoss'])
            losses_names.append('HingeLoss')
                
        return losses, weights, losses_names
            
    def __init_optimizer__(self, optimizer_config):
        """
        Parameters
        ----------
        optimizer_config : dict
            contains the optimizer you want to use for training from the following list :
                'Adam', 'AdamW', 'SparseAdam', 'SGD'.
        Returns
        -------
        """
        
        if optimizer_config['optimizer']=='Adam':
            optimizer=torch.optim.Adam(self.net.parameters(), optimizer_config['learning_rate'])
            
        elif optimizer_config['optimizer']=='SGD':
            optimizer=torch.optim.SGD(self.net.parameters(), optimizer_config['learning_rate'])
            
        elif optimizer_config['optimizer']=='AdamW':
            optimizer=torch.optim.AdamW(self.net.parameters(), optimizer_config['learning_rate'])

        elif optimizer_config['optimizer']=='SparseAdam':
            optimizer=torch.optim.SparseAdam(self.net.parameters(), optimizer_config['learning_rate'])
            
        #elif optimizer_config['optimizer']=='Lion':
        #    #https://github.com/lucidrains/lion-pytorch carefully read the doc for tuning
        #    optimizer=Lion(self.net.parameters(), optimizer_config['learning_rate'])
           
        else : 
            print("optimizer badly configured")
            
        return optimizer
    
    def criterion(self, predictions, targets):
        
        self.loss_dict={}
        
        res=0
        for i, (loss, weight, name) in enumerate(zip(self.losses, self.losses_weights, self.losses_names)):
            

            temp=loss(predictions, targets)  
            self.loss_dict[name]=temp.cpu().detach().item()
            res+=temp*weight
        
        return res
        

        
    def forward(self, inputs, targets, train=True, test=False):
        
        if test and train :
            raise ValueError("In forward function, train and test cannot be both True")
        
        if isinstance(targets, tuple) or isinstance(targets, list):
           targets=tuple([target.to(self.device) for target in targets])
        else :
            targets=targets.to(self.device)
        if train:
            self.optimizer.zero_grad()
            
            
        #Pour les partials conv il faut gérer le mask directement dans l'architecture ou le dataloader
        # if 'Partial_conv' in self.config.keys() and self.config['Partial_conv']:
        #     mask=torch.ones(item.size())*(~torch.isnan(item))
        #     predictions,_ = self.net(item.to(self.device), mask.to(self.device))
        # else :
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            predictions=self.net(inputs.to(self.device))
            
            loss = self.criterion(predictions, targets)

        if train :
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        if test : return predictions,loss.item()
        
        return loss.item()
    
    
    
    
if __name__=='__main__':
     
    import sys
    sys.path.append("/home/luther/Documents/scripts_training/models/")
    from models.UNet import UNet
    from models.SmaAt_UNet import SmaAt_UNet
    import numpy as np
    import torch.nn as nn
    
    criterion_config = {
        'MSE_psc' : 0., # eventual features and weights of losses (put None instead of 0 if you don't want the loss to be compute)
        'MSE_chl':1.,
        'KL_div':1.
    }
    
    optimizer_config = {
        'optimizer' : 'Adam', # Adam, AdamW, SGD, Lion, SparseAdam
        'learning_rate' : 0.001
    }
    
    

            
            
    net=SmaAt_UNet(n_channels=8, 
                   n_classes=4,
                   activation=nn.ReLU(),
                   kernels_per_layer=2,
                   psc=True)
                       
    net.to(device='cuda')
    
    test_model=model(net, 'cuda', criterion_config, optimizer_config)
    mask=torch.Tensor(np.where(np.eye(360)[:100]==1,1,np.nan))

    test_inp=torch.Tensor(np.random.random(((1,8,100,360))))
    test_targ=(mask*torch.Tensor(np.random.random(((1,3,100,360)))),torch.Tensor(np.random.random(((1,1,100,360)))))
     
    
    print(test_model.forward(test_inp, test_targ))
    
 
        # if 'Under_chl' in criterion_config.keys() and criterion_config['Under_chl']!=0.:
                        
        #     losses.append(CHL_underestimation(index=index_loss))
        #     weights.append(criterion_config['Under_chl'])
        #     losses_names.append('Under_chl')
            
                
        # if 'KL_div' in criterion_config.keys() and criterion_config['KL_div']!=0.:
            
        #     losses.append(KL_divergence())
        #     weights.append(criterion_config['KL_div'])
        #     losses_names.append('KL_div')
