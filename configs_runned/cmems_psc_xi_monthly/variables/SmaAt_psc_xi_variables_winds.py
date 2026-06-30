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

dataloader_config={'dataset_path_inputs':path+"physics/processed_physics_1998_2024_monthly_lat50_100.npy",
                   'dataset_path_inputs_test':path+"physics/processed_physics_1993_2024_monthly_lat50_100.npy",
                   'dataset_path_psc':path+"PSC/cmems_xi/process/2002_2023_xi_cmems_psc_mo_lat50.nc",
                   'dataset_path_chl':None,
                   'hplc_path':None,#path+"insitu/hplc_merged_new_glob_100km.csv",
                   'transform':None,
                   'batch_size': 16,
                   'norm_chl':True,
                   'norm_mode':'standard',
                   'log_chl':True,
                   'lenght_year':12,
                   'completion':'zeros',
                   'stations_coord':{'SWAtl':(-55,-43),# stations withdrawed of the dataset for control
                                     'NAtl':(-37,36),
                                     'NPac':(156,24),
                                     'SIO':(60,-32),
                                     'SCTR':(80,-3)},
                   'variables_physiques':[6],
                   'phy_anomalies':False, #[0,1,2,3...] list of which physical fields anomalies should be added
                   'chl_anomalies':False, 
                   } 
                   #weekly
                   #0 mld, 1 SST, 2 Salinité, 3 SSH, 4 Solar, 5 bathy, 6 currents, 7 winds
                   #monthly
                   #0 mld, 1 SST, 2 Salinité, 3 SSH, 4 Solar, 5 currents, 6 winds

########################################### TRAIN #############################################


train_config = {
    'nb_epochs' : 300, # arbitrary high
    'checkpoints_path': '/home/luther/Documents/result/cmems_psc_xi_monthly/variables/', 
    'verbose': 0.,
    'checkpoint':None, #save the weights every x epochs
    'name':'SmaAt_psc_xi_variables_winds',
    'patience_early_stopping':100, #needs to be >>> patience of lr scheduler
    'delta_early_stopping':0.00000001,
    'device':0,
    'nb_training':1,
    'optim':False, #if True, trainer will return mean valid loss and std valid loss for optimization
    'comment':'variables testing for article'
}

########################################### Model #############################################

        
model_config = {
    'in_channels' : 1,
    'model':'SmaAt-UNet',
    'depth':5,
    'merge_mode':'concat',
    'activation':'SiLU',
    'weight_load':None,
    'freeze_key':'0',
    'kernels_per_layer':1,
    'n_classes':3, #à adapter en fonction d'une éventuelle pinball loss
    'chl':4, # 0 pour prédire CHL et PSC (maintient la compatibilité avec les configs passées).
                 # 1 pour prédire uniquement les PSC (avec éventuellement la chl en entrée, en fonction du dataloader)
                 # 2 pour prédire psc (3) + chl (1) + sign anomalie (1) + valeur anomalie (1) + eventuelle quantile loss
                 #           (psc,chl,sign,magnitude)
                 # 3 pour prédire chl (1) + sign anomalie (1) + valeur anomalie (1) + eventuelle quantile loss
                 # autres pour sortir direct l'output de la dernière 1x1 convolution (nb de channel = nclasses),  ie seulement pour la chloro par exemple
    'nb_layers':16,
    'time2vec':False, #outdated
}


########################################### criterion config #############################################


criterion_config = {
    'MSE_psc' : 1., # eventual features and weights of losses (put None instead of 0 if you don't want the loss to be compute)
    'MSE_chl':0.,
    'MSE_chl_anom':0.,
    'quantile_loss':0., #à remplir en mode : ((quantile, weight)), ex : ((0.5,0.5),(0.05,0.05),(0.25,0.20),(0.75,0.20),(0.95,0.05)) 
    'BCELoss':0.,
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
    'factor': 0.1, # when loss has stagnated for too long, new_lr = factor*lr
    'patience': 25,# how long to wait before updating lr
    'threshold': 0.000000001, 
    'max_lr':0.001,
    'verbose':True,
    'steps_per_epoch':None,#1+(int((dataloader_config['split_index']['train'][1]-dataloader_config['split_index']['train'][0]+1)*46/dataloader_config['batch_size']))
}
    
