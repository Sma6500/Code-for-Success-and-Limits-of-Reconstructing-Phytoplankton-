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

dataloader_config={'dataset_path_inputs':path+"physics/physics_completion/processed_1998_2023_reanalysis_100km_8d_seaice_32.npy",
                   'dataset_path_psc':path+"completion/avw_cmems/psc_p_avw_cmems_8d_glob_100km_1998_2023.npy",
                   'dataset_path_chl':path+"completion/avw_cmems/chl_avw_cmems_8d_glob_100km_1998_2023.npy",
                   'transform':None,
                   'batch_size': 16,
                   'norm_chl':True,
                   'norm_mode':'standard',
                   'log_chl':True,
                   'completion':'zeros',
                   'mask_ratio':(0.5,0.5,0.5),
                   'stations_coord':{'SWAtl':(-55,-43),# stations withdrawed of the dataset for control
                                     'NAtl':(-37,36),
                                     'NPac':(156,24),
                                     'SIO':(60,-32),
                                     'SCTR':(80,-3)}}


########################################### TRAIN #############################################


train_config = {
    'nb_epochs' : 500, # arbitrary high
    'checkpoints_path': '/home/luther/Documents/result/completion/', 
    'verbose': 0.,
    'checkpoint':None, #save the weights every x epochs
    'name':'SmaAt_avwcmems_completion_test',
    'patience_early_stopping':30, #needs to be >>> patience of lr scheduler
    'delta_early_stopping':0.00000001,
    'device':0,
    'nb_training':1,
    'optim':False, #if True, trainer will return mean valid loss and std valid loss for optimization
    'comment':'SMaat pour compléter les données avwcmems avec psc'
}

########################################### Model #############################################

        
model_config = {
    'in_channels' : 13,
    'model':'SmaAt-UNet',
    'depth':5,
    'merge_mode':'concat',
    'activation':'SiLU',
    'weight_load':None,
    'freeze_key':'0',
    'kernels_per_layer':2,
    'chl':0, # 0 pour prédire CHL et PSC (maintient la compatibilité avec les configs passées).
                 # 1 pour prédire uniquement les PSC (avec éventuellement la chl en entrée, en fonction du dataloader)
                 # 2 pour prédire uniquement la CHL (étude Marina/Stéphane).
    'nb_layers':64,
}


########################################### criterion config #############################################


criterion_config = {
    'MSE_psc' : 10., # eventual features and weights of losses (put None instead of 0 if you don't want the loss to be compute)
    'MSE_chl':1.,
    'Under_chl':0.,
    'KL_div':0.,
    'quantile_loss':0.1,
    'details':True
}

########################################### criterion config #############################################

optimizer_config = {
    'optimizer' : 'AdamW', # Adam, AdamW, SGD, Lion, SparseAdam
    'learning_rate' : 0.0001
}

######################################### SCHEDULER ###########################################


scheduler_config = {
    'scheduler': 'ROP', # ROP or ELR, OC not available currently
    'mode': 'min', # we want to detect a decrease and not an increase. 
    'factor': 0.75, # when loss has stagnated for too long, new_lr = factor*lr
    'patience': 25,# how long to wait before updating lr
    'threshold': 0.000000001, 
    'max_lr':0.001,
    'verbose':True,
    'steps_per_epoch':None,#1+(int((dataloader_config['split_index']['train'][1]-dataloader_config['split_index']['train'][0]+1)*46/dataloader_config['batch_size']))
}
    
