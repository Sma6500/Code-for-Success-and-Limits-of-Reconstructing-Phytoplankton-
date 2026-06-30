#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:26:13 2022

@author: lollier

Class Trainer that instanciate the classes dataloaders and model in order to achieve training and validation step
"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         TRAINER                                       | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
from tqdm import tqdm

import torch

from dataloaders import get_dataloaders
from model import model
from utils.early_stopping import EarlyStopping
from utils.functions import timer
import torch
# default `log_dir` is "runs" - we'll be more specific here
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
class Trainer():
    
    """
    TO DO :
        - handle parallelized training
        
    """
     
    def __init__(self, net, 
                 train_config, 
                 dataloader_config, 
                 criterion_config, 
                 optimizer_config, 
                 scheduler_config):
        
        self.model = model(net, train_config['device'], criterion_config, optimizer_config) 
        self.config = train_config 
        self.scheduler=self.__init_scheduler__(scheduler_config)
        self.dataloader_config=dataloader_config
        self.trainloader, self.validloader, self.testloader, self.scaler_chl = self.__init_dataloaders__(dataloader_config)
        if isinstance(self.scaler_chl, tuple):
            self.scaler_chl_anom=self.scaler_chl[1]
            self.scaler_chl=self.scaler_chl[0]
            
        self.early_stopping = self.__init_early_stopping__()
        
        
        self.state = {'train_loss': 0,  
                      'valid_loss': 0, 
                      'best_valid_loss': np.Inf,
                      'epoch': 0,
                      'epoch_time':0}
        

    def __str__(self): 
        title = 'Training settings  :' + '\n' + '\n'
        net         = 'Net.......................:  \n' + self.config['name'] + '\n' 
        optimizer   = 'Optimizer.................:  ' + f'{self.model.optimizer}' + '\n'
        scheduler   = 'Learning Rate Scheduler...:  ' + self.scheduler.name + '\n'
        nb_epochs   = 'Number of epochs..........:  ' + str(self.config['nb_epochs']) + '\n'
        summary = title + net + optimizer + scheduler + nb_epochs
        return (80*'_' + '\n' + summary + 80*'_')
        
    
    def __init_dataloaders__(self, dataloader_config):
        return get_dataloaders(dataloader_config)

    def __init_early_stopping__(self):
        early_stopping = EarlyStopping(patience=self.config['patience_early_stopping'], 
                                       delta = self.config['delta_early_stopping'], 
                                       verbose=True if self.config['verbose']>=0 else False)
        early_stopping.set_checkpoints(self.config)
        return early_stopping

    def __init_scheduler__(self, scheduler_config):
        """
        Parameters
        ----------
        scheduler_config : dict
            contains the scheduler you want to use for training from the following list :
                'ROP' : Reduce LR when valid loss stabilize,
                'ELR' : Multiply LR each epoch
                'OC' : OneCycleLR, increase and then decrease lr.
        Returns
        -------
        """        
        if scheduler_config['scheduler']=='ROP':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.model.optimizer, 
                                                                   mode=scheduler_config['mode'],
                                                                  factor=scheduler_config['factor'],
                                                                  patience=scheduler_config['patience'],
                                                                  threshold=scheduler_config['threshold'],
                                                                  verbose=scheduler_config['verbose'])
        elif scheduler_config['scheduler']=='ELR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(self.model.optimizer,
                                                     gamma=scheduler_config['factor'],
                                                     verbose=scheduler_config['verbose'])
        
        #notice that optimizer.step() needs to be apply each batch for OneCycle LR
        elif scheduler_config['scheduler']=='OC':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(self.model.optimizer, 
                                                            max_lr=scheduler_config['max_lr'],
                                                            epochs=self.config['nb_epochs'],
                                                            steps_per_epoch=scheduler_config['steps_per_epoch']) #nb batch
        
        
        else : 
            raise ValueError("scheduler badly configured")
            return None
        
        #future tentative d'inclure un warmup, peut être voir warmup-pytorch
        # if 'warmup' in scheduler_config.keys() and scheduler_config['warmup']:
            
        #     temp_scheduler=torch.optim.lr_scheduler.ConstantLR()
        
        
        scheduler.name=scheduler_config['scheduler']

        return scheduler

    def training_step(self):
        """
        Action
        -------
        Make a training step for an epoch, 
        -------
        train_loss : torch tensor
        """
        
        self.model.net.train()
        #self.model.net.eval()

        
        #if self.config['verbose']==1:
        if self.model.details:
            self.train_loss_details={losses_name:0 for losses_name in self.model.losses_names}
        train_loss = 0.

        for inputs,targets in tqdm(self.trainloader):
            

            t_l = self.model.forward(inputs, targets)  
            train_loss+=t_l
            
            if self.model.details:
                for losses_name in self.train_loss_details.keys():
                    self.train_loss_details[losses_name]+=self.model.loss_dict[losses_name]/len(self.trainloader)
                    
            if self.scheduler.name=='OC':
                self.model.scheduler.step()

                
        return train_loss/len(self.trainloader)
    
    
    def validation_step(self):
        """
        Action
        -------
        Make a validation step for an epoch, 
        -------
        valid_loss : torch tensor
        """
        
        self.model.net.eval()
        
        if self.model.details:
            self.valid_loss_details={losses_name:0 for losses_name in self.model.losses_names}
            
        valid_loss = []
        
        
        
        with torch.no_grad():
            for inputs,target in tqdm(self.validloader):

                v_l = self.model.forward(inputs, target, train=False)
                valid_loss.append(v_l)
                       
                if self.model.details:
                    for losses_name in self.valid_loss_details.keys():
                        self.valid_loss_details[losses_name]+=self.model.loss_dict[losses_name]/len(self.validloader)
                    

        if self.config['optim']:
            return np.mean(valid_loss),np.std(valid_loss)
        
        return np.sum(valid_loss)/len(self.validloader)
    
        
    def __chl_post_processing__(self, chl_array):
        if self.dataloader_config['norm_chl'] : #and dataloader_config['norm_mode']=='standard': #actually no min max experiences has been conducted
            if np.squeeze(chl_array).shape[0]==3 and len(self.scaler_chl.mean)==3:
                for i in range(3):
                    chl_array[:,i]=self.scaler_chl.mean[i]+(chl_array[:,i]*(self.scaler_chl.std[i]+self.scaler_chl.epsilon))
            else :
                chl_array=self.scaler_chl.mean+(chl_array*(self.scaler_chl.std+self.scaler_chl.epsilon))
        if self.dataloader_config['log_chl']:
            chl_array=10**chl_array
                    
        return chl_array
    
    def __chl_anom_post_processing__(self, chl_anom_array):
        chl_anom=chl_anom_array[:,0,None]*np.sign(chl_anom_array[:,-1,None])

        chl_anom_array_rescaled=self.scaler_chl_anom.mean+(chl_anom*(self.scaler_chl_anom.std+self.scaler_chl_anom.epsilon))
        chl_anom_array=np.sign(chl_anom_array_rescaled)*10**(np.abs(chl_anom_array_rescaled)-5)
                    
        return chl_anom_array
    
    def test_step(self):
        """
        Action
        -------
        Make all predictions on the test set 
        -------
        """

    
        self.model.net.eval()
            
        test_loss = 0.
        chl_predictions=[]
        psc_predictions=[]
        type_output=self.model.net.chl #se référer à SmaAt-Unet pour plus de détails
        
        with torch.no_grad():
            
            for inputs,target in tqdm(self.testloader):
                preds,t_l = self.model.forward(inputs, target, train=False, test=True)
                test_loss+=t_l     
                
                
                if type_output==0:
                    chl_predictions.append(self.__chl_post_processing__(preds[1].detach().cpu().numpy()))
                    psc_predictions.append(preds[0].detach().cpu().numpy())
                    
                elif type_output==1:
                    psc_predictions.append(preds[0].detach().cpu().numpy())

                elif type_output==2:
                    chl_predictions.append([self.__chl_post_processing__(preds[1].detach().cpu().numpy()),
                                            self.__chl_anom_post_processing__(preds[2].detach().cpu().numpy())])
                    psc_predictions.append(preds[0].detach().cpu().numpy())
                elif type_output==3:
                    chl_predictions.append([self.__chl_post_processing__(preds[0].detach().cpu().numpy()),
                                            self.__chl_anom_post_processing__(preds[1].detach().cpu().numpy())])
                    
                else : 
                    chl_predictions.append(self.__chl_post_processing__(preds[0].detach().cpu().numpy()))

            del inputs,target
        return test_loss/len(self.testloader), np.array(psc_predictions), np.array(chl_predictions)
    
    

    def verbose(self):
        print()
        print('Train Loss................: {}'.format(self.state['train_loss']))
        print('Validation Loss.................: {}'.format(self.state['valid_loss']))
        print()
        # lr can't be read easily when scheduler is used
        print('Current Learning Rate.....: {}'.format(self.model.optimizer.param_groups[0]['lr']))
        print('Best Validation Loss........: {}'.format(self.state['best_valid_loss']))
        
        if self.config['verbose']==1:
            for name in self.train_loss_details.keys():
                print('%d Train : %d,  Valid : %d' % (name,self.train_loss_details[name],self.valid_loss_details[name]))

    def save_out(self, epoch):
        """
        Input : 
            epoch, int
            current epoch
            
        Action
        -------
        save loss data (with details if required) and learning rate for each epoch
        """
        with open(os.path.join(self.config['checkpoints_path'],self.config['name']+"_training.txt"), "a+") as file :
            file.write('\n\n'+80*'_')
            file.write('\nEPOCH %d / %d' % (epoch+1, self.config['nb_epochs']))
            file.write('\nTrain Loss................: {}'.format(self.state['train_loss']))
            file.write('\nValidation Loss...........: {}'.format(self.state['valid_loss']))
            file.write('\nNext Learning Rate........: {}'.format(self.model.optimizer.param_groups[0]['lr']))
            file.write('\nStep time ................: {}'.format(self.state['epoch_time']))
            
            if self.model.details:
                file.write('\n Details : ')
                for name in self.train_loss_details.keys():
                    file.write('\n Train  '+name+'................: {}'.format(self.train_loss_details[name]))
                    file.write('\n Valid  '+name+'................: {}'.format(self.valid_loss_details[name]))

    @timer
    def update_state(self):
        """
        Action
        -------
        run training loop for one epoch.
        """
        train_loss = self.training_step()
        if self.config['optim']:
            valid_loss,std_valid_loss=self.validation_step()
        else:
            valid_loss = self.validation_step()
            
        #scheduler time
        if self.scheduler.name=='ROP':
            self.scheduler.step(valid_loss)
        elif self.scheduler.name=='ELR' :
            self.scheduler.step()
            
        self.state['train_loss'] = train_loss
        self.state['valid_loss'] = valid_loss
        if self.config['optim']:
            self.state['valid_std'] = std_valid_loss

        
        if valid_loss < self.state['best_valid_loss']:
            self.state['best_valid_loss'] = valid_loss

            
    def run(self):
        """
        Action
        -------
        run training loop 
        if optim True will return std and mean of valid loss
        """
        
        for epoch in range(self.config['nb_epochs']):
            print(80*'_')
            print('EPOCH %d / %d' % (epoch+1, self.config['nb_epochs']))
            self.state['epoch']=epoch
            self.state['epoch_time']=self.update_state()
            if self.config['verbose']>=0:
                self.verbose()
            self.save_out(epoch)
            
            if isinstance(self.config['checkpoint'],int) and epoch%self.config['checkpoint']==0:
                filename = 'checkpoint_' + str(epoch)+'_'+self.config['name']+'_.pt'
                path = os.path.join(self.config['checkpoints_path'],filename)
                torch.save(self.model.net.state_dict(), path)
                print()
                print(80*'_')
                print('Current State saved.')
            self.early_stopping(self.state['valid_loss'], self.model.net)
            
            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                print("Validation loss didn't improve since epoch {}".format(epoch-self.early_stopping.patience))
                break
        
        if self.config['optim']:
            return self.state['best_valid_loss'],self.state['valid_std']

if __name__=='__main__':
    
    import sys
    path="/home/luther/Documents/npy_data/"
    #path="/datatmp/home/lollier/npy_data/"
    
    dataloader_config={'dataset_path_inputs':path+"physics/physics_completion/reduced_data.nc",
                       'dataset_path_chl_psc':path+"completion/avw_roy/reduced_dataset.nc",
                       'hplc_path':path+"insitu/hplc_merged_new_glob_100km.csv",
                       'transform':None,
                       'batch_size': 4,
                       'norm_chl':True,
                       'norm_mode':'standard',
                       'log_chl':True,
                       'completion':'zeros',
                       'stations_coord':{'SWAtl':(-55,-43),# stations withdrawed of the dataset for control
                                         'NAtl':(-37,36),
                                         'NPac':(156,24),
                                         'SIO':(60,-32),
                                         'SCTR':(80,-3)}}
    
    
    sys.path.append("/home/luther/Documents/scripts_training/models/")
    from models.SmaAt_UNet import SmaAt_UNet
    import torch.nn as nn

    

    
    train_config = {
        'nb_epochs' : 1, # arbitrary high
        'checkpoints_path': '/home/luther/Documents/result/completion/', 
        'verbose': 0.,
        'checkpoint':None, #save the weights every x epochs
        'name':'test',
        'patience_early_stopping':50, #needs to be >>> patience of lr scheduler
        'delta_early_stopping':0.00000001,
        'device':0,
        'nb_training':1,
        'optim':False, #if True, trainer will return mean valid loss and std valid loss for optimization
        'comment':'test'
    }
    
    criterion_config = {
        'MSE_psc' : 1., # eventual features and weights of losses (put None instead of 0 if you don't want the loss to be compute)
        'MSE_chl':1.,
        'Under_chl':0.,
        'KL_div':0.,
        'quantile_loss':0.,
        'HingeLoss':0.,
        'details':True
    }
        
    optimizer_config = {
        'optimizer' : 'AdamW', # Adam, AdamW, SGD, Lion, SparseAdam
        'learning_rate' : 0.001
    }
        
    
    scheduler_config = {
        'scheduler': 'ROP', # ROP or ELR, OC not available currently
        'mode': 'min', # we want to detect a decrease and not an increase. 
        'factor': 0.75, # when loss has stagnated for too long, new_lr = factor*lr
        'patience': 10,# how long to wait before updating lr
        'threshold': 0.000000001, 
        'max_lr':0.001,
        'verbose':True,
        'steps_per_epoch':None,#1+(int((dataloader_config['split_index']['train'][1]-dataloader_config['split_index']['train'][0]+1)*46/dataloader_config['batch_size']))
    }
    
    
    net=SmaAt_UNet(n_channels=12, 
                   n_classes=4,
                   activation=nn.SiLU(),
                   kernels_per_layer=2,
                   chl=0,
                   nb_layers=64,
                   time_encoded=False)
                   
    net.to(device='cuda')
    

    net.load_state_dict(torch.load('/home/luther/Documents/result/completion/SmaAt_completion_avw_roy_0/SmaAt_completion_avw_roy_0best_valid_loss.pt'))
    
    trainer = Trainer(net, train_config, dataloader_config, criterion_config, optimizer_config, scheduler_config)
    
    #trainer.run()
    
    
