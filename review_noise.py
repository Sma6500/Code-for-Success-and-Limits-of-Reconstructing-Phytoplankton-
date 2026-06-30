#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:09:36 2026

@author: luther

Tester bruit review 1
"""

from torch import load, save 
import os
from Trainer import Trainer
import json

import numpy as np 
import energyusage


import sys
sys.path.append("/home/luther/Documents/scripts_training/models/")
sys.path.append("/usr/home/lollier/Documents/scripts_training/models/")


from models.SmaAt_UNet import SmaAt_UNet
from utils.functions import timer, saving_tool#, process_results

import torch
import torch.nn as nn
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

with open("/home/luther/Documents/result/avw_psc_chl_monthly/variables/SmaAt_chlm_avw_variables_all/SmaAt_chlm_avw_variables_all_config.json", "r") as f:
    configs = json.loads(json.load(f))
    
(
    model_config,
    dataloader_config,
    train_config,
    criterion_config,
    scheduler_config,
    optimizer_config,
) = configs
    
dataloader_config['transform']=None

######################parameters####################################
if 'nb_layers' in model_config.keys():
    nb_layers=model_config['nb_layers']
else : nb_layers=64

if model_config['activation'] in ('ReLU', 'SiLU'):
    if model_config['activation']=='ReLU' :
        activation = nn.ReLU()
    else : 
        activation = nn.SiLU()
else:
    raise ValueError("\"{}\" is not a valid mode for "
                     "activation. Only \"SiLU\" and "
                     "\"ReLU\" are allowed.".format(model_config['activation']))

if 'time2vec' in model_config.keys() and model_config['time2vec']:
    time_encoded=model_config['time2vec']
else :
    time_encoded=False
        

n_classes=model_config.get('n_classes', 4)
            

        
if model_config['model'] != 'SmaAt-UNet':
    raise ValueError('wrong model')
    
net=SmaAt_UNet(n_channels=model_config['in_channels'], 
               n_classes=n_classes,
               activation=activation,
               kernels_per_layer=model_config['kernels_per_layer'],
               chl=model_config['chl'],
               nb_layers=nb_layers,
               time_encoded=time_encoded)

for sigma in [0.05,0.1,0.2,0.5,1]:
             
    dataloader_config['sigma']=sigma
    
    trainer = Trainer(net, train_config, dataloader_config, criterion_config, optimizer_config, scheduler_config)
    trainer.model.net.load_state_dict(load(os.path.join(train_config['checkpoints_path'], 
                                                            train_config['name'] + 'best_valid_loss.pt')))
       
    loss, psc_predictions, chl_predictions=trainer.test_step()
    
    save_path="/home/luther/Documents/result/avw_psc_chl_monthly/variables_noise/"
    
    np.save(save_path+f"/all_{sigma}_{train_config['name']}chl_pred_glob.npy",chl_predictions)
    np.save(save_path+f"/all_{sigma}_{train_config['name']}psc_pred_glob.npy",psc_predictions)


