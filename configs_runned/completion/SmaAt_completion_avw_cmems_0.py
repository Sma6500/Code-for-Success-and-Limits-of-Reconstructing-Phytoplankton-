#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:13:27 2022

@author: lollier

model config file
"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         CONFIG                                        | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NEVER BE MODIFIED                        | #
# |                   It details how to configured the various hyperparameters            | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #



######################################### DATALOADER ##########################################


path="/home/luther/Documents/npy_data/"
#path="/datatmp/home/lollier/npy_data/"

dataloader_config={'dataset_path_inputs':path+"physics/physics_completion/processed_1998_2023_reanalysis_100km_daily_seaice.nc",
                   'dataset_path_psc':path+"PSC/PSC_Roy_8d_1998_2021_lat50_100km.npy",
                   'dataset_path_chl':path+"chl/chl_avw_lat50_100km_8d_1998_2023.npy",
                   'dataset_path_chl_psc':path+"completion/avw_cmems/chl_psc_pcorrected_avw_cmems_1d_glob_100km_1998_2023.nc",
                   'hplc_path':False, #path+"insitu/hplc_merged_new_glob_100km.csv",
                   'transform':None,
                   'batch_size': 16,
                   'norm_chl':True,
                   'norm_mode':'standard',
                   'log_chl':True,
                   'anomalies':0, #0 or False or None for classic behavior, 1 for chl anomalies, 2 for physics anomalies 
                   'completion':'zeros',
                   'stations_coord':{'SWAtl':(-55,-43),# stations withdrawed of the dataset for control
                                     'NAtl':(-37,36),
                                     'NPac':(156,24),
                                     'SIO':(60,-32),
                                     'SCTR':(80,-3)}}


########################################### TRAIN #############################################


train_config = {
    'nb_epochs' : 300, # arbitrary high
    'checkpoints_path': '/home/luther/Documents/result/completion/', 
    'verbose': 0.,
    'checkpoint':None, #save the weights every x epochs
    'name':'SmaAt_completion_avw_cmems_0',
    'patience_early_stopping':50, #needs to be >>> patience of lr scheduler
    'delta_early_stopping':0.00000001,
    'device':0,
    'nb_training':1,
    'optim':False, #if True, trainer will return mean valid loss and std valid loss for optimization
    'comment':'SMaat completion MSE classique'
}

########################################### Model #############################################

        
model_config = {
    'in_channels' : 12,
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
                 # 3 pour prédire uniquement la CHL avec n_classes=2 pour que la sortie soit le signe et l'amplitude
    'nb_layers':64,
    'time2vec':False,
}


########################################### criterion config #############################################


criterion_config = {
    'MSE_psc' : 1., # eventual features and weights of losses (put None instead of 0 if you don't want the loss to be compute)
    'MSE_chl':1.,
    'Under_chl':0.,
    'KL_div':0.,
    'quantile_loss':0.,
    'HingeLoss':0.,
    'details':True
}

########################################### criterion config #############################################

optimizer_config = {
    'optimizer' : 'AdamW', # Adam, AdamW, SGD, Lion, SparseAdam
    'learning_rate' : 0.001
}

######################################### SCHEDULER ###########################################


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
    
