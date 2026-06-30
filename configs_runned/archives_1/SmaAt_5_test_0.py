#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:13:27 2022

@author: lollier

config file
"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         CONFIG                                        | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             ONLY THIS FILE SHOULD BE MODIFIED                         | #
# |                           ALL HYPER PARAMETERS SHOULD BE CONFIGURED                   | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #



######################################### DATALOADER ##########################################


path="/home/luther/Documents/npy_data/"
#path="/datatmp/home/lollier/npy_data/"

dataloader_config={'dataset_path_inputs':path+"physics/processed_physics_1998_2020_8d_100_lat50.npy",
                   'dataset_path_psc':path+"PSC/New_PSC_Roy_8d_1998_2020_lat50_100km.npy",
                   'dataset_path_chl':path+"chl/chl_avw_1998_2020_100k_8d_lat50.npy",
                   'transform':None,
                   'batch_size': 16,
                   'norm_chl':True,
                   'norm_mode':'standard',
                   'log_chl':True,
                   'completion':'zeros',
                   'stations_coord':{'SWAtl':(-55,-43),# stations withdrawed of the dataset for control
                                     'NAtl':(-37,36),
                                     'NPac':(156,24),
                                     'SIO':(60,-32),
                                     'SCTR':(80,-3)}}


########################################### TRAIN #############################################


train_config = {
    'nb_epochs' : 2000, # arbitrary high
    'checkpoints_path': '/home/luther/Documents/result/', 
    'verbose': 0.,
    'checkpoint':None, #save the weights every x epochs
    'name':'SmaAt_5_test_0',
    'patience_early_stopping':100, #needs to be >>> patience of lr scheduler
    'delta_early_stopping':0.00000001,
    'device':0,
    'nb_training':1,
    'optim':False, #if True, trainer will return mean valid loss and std valid loss for optimization
    'comment':'SmaAt-UNet avec 32 layers de départ et 1 kernel par layer donc bcp moins de poids et un seul pas de temps en entrée et dataloaders correct gros bouffon'
}

########################################### Model #############################################

        
model_config = {
    'in_channels' : 8,
    'model':'SmaAt-UNet',
    'depth':5,
    'merge_mode':'concat',
    'activation':'ReLU',
    'weight_load':None,
    'freeze_key':'0',
    'kernels_per_layer':1,
    'chl':False,
    'nb_layers':32,
}


########################################### criterion config #############################################


criterion_config = {
    'MSE_psc' : 1., # eventual features and weights of losses (put None instead of 0 if you don't want the loss to be compute)
    'MSE_chl':1.,
    'Under_chl':0.,
    'KL_div':0.,
    'details':True
}

########################################### criterion config #############################################

optimizer_config = {
    'optimizer' : 'Adam', # Adam, AdamW, SGD, Lion, SparseAdam
    'learning_rate' : 0.001
}

######################################### SCHEDULER ###########################################


scheduler_config = {
    'scheduler': 'ROP', # ROP or ELR, OC not available currently
    'mode': 'min', # we want to detect a decrease and not an increase. 
    'factor': 0.5, # when loss has stagnated for too long, new_lr = factor*lr
    'patience': 10,# how long to wait before updating lr
    'threshold': 0.000000001, 
    'max_lr':0.001,
    'verbose':True,
    'steps_per_epoch':None,#1+(int((dataloader_config['split_index']['train'][1]-dataloader_config['split_index']['train'][0]+1)*46/dataloader_config['batch_size']))
}
    
