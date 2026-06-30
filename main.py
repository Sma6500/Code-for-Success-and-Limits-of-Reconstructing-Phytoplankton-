
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:42:54 2022

@author: lollier
"""



# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         MAIN                                          | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

from torch import load, save 
import os
from Trainer import Trainer
from config import model_config, dataloader_config, train_config, criterion_config, scheduler_config, optimizer_config


import numpy as np 
import energyusage


import sys
sys.path.append("/home/luther/Documents/scripts_training/models/")
sys.path.append("/usr/home/lollier/Documents/scripts_training/models/")

from models.DBNet import DualBranchNet
from models.SmaAt_DBNet import SmaAt_DualBranchNet
from models.light_SmaAt_UNet import light_SmaAt_UNet
from models.UNet import UNet
from models.UNet_DSC import UNet_DSC
from models.UNet_CBAM import UNet_CBAM
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

"""
This function takes hyperparameters from config.py. 
It creates an object from the Model class and then uses it to define an object 
from the Trainer class.
The training is launched by the call to the run() method from the trainer object. 
This call is inside a try: block in order to handle exceptions.
For now, the only exception handled is a KeyboardInterrupt: 
the current network will be saved.
"""

def main(model_config, dataloader_config, train_config, criterion_config, optimizer_config, scheduler_config):

    print('Building Model...')
    
    """
    Architecture configuration
    """
    
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
        
    #c'est pas très propre mais ça a le mérite d'être compatible avec toutes les configs y compris les anciennes
    # psc=True,
    # chl=False
    # n_classes=4
    # if 'chl' in model_config.keys() and model_config['chl']:
    #     if model_config['chl']==1:
    #         psc=False
    #         chl=True
    #         n_classes=3
    #     elif model_config['chl']==2:
    #         psc=False
    #         chl=False
    #         n_classes=1
    #     elif model_config['chl']==3:
    #         psc=False
    #         chl=False
    #         n_classes=2

    #     else :
    #         print("Attention, configuration manuelle de l'argument chl de model_config")
            
    #obsolète
    n_classes=model_config.get('n_classes', 4)
            
########################################################################
    if model_config['model']=='UNet':
               
        net=UNet(in_channels=model_config['in_channels'],
                 out_channels=model_config['n_classes'],
                  depth=model_config['depth'],
                  start_filts=nb_layers,
                  merge_mode=model_config['merge_mode'], 
                  activation=model_config['activation'],
                  chl=model_config['chl'])
        
    elif model_config['model']=='UNet_DSC':
        
        net=UNet_DSC(in_channels=model_config['in_channels'],
                     out_channels=model_config['n_classes'],
                     depth=model_config['depth'],
                     start_filts=nb_layers,
                     merge_mode=model_config['merge_mode'], 
                     activation=model_config['activation'],
                     chl=model_config['chl'],
                     kernels_per_layer=model_config['kernels_per_layer'])
        
    elif model_config['model']=='UNet_CBAM':
                
        net=UNet_CBAM(n_channels=model_config['in_channels'],
                     n_classes=model_config['n_classes'],#tjrs 4 pour PSC et CHL
                     depth=model_config['depth'],
                     activation=activation,
                     nb_layers=nb_layers,
                     chl=model_config['chl'],)
        
    elif model_config['model']=='SmaAt-UNet':

        net=SmaAt_UNet(n_channels=model_config['in_channels'], 
                       n_classes=n_classes,
                       activation=activation,
                       kernels_per_layer=model_config['kernels_per_layer'],
                       chl=model_config['chl'],
                       nb_layers=nb_layers,
                       time_encoded=time_encoded)
        
    elif model_config['model']=='SmaAt-BAM':

        net=SmaAt_UNet(n_channels=model_config['in_channels'], 
                       n_classes=n_classes,
                       activation=activation,
                       kernels_per_layer=model_config['kernels_per_layer'],
                       chl=model_config['chl'],
                       nb_layers=nb_layers,
                       time_encoded=time_encoded)
        
    #maintenant qu'on peut avoir des grosses tailles de batchs, la light Smaat est plus très pertinent    
    elif model_config['model']=='light_SmaAt-UNet':
        
        print("attention, config outdated, pas de mise à jour pour l'option model_config[chl]")
            
        net=light_SmaAt_UNet(n_channels=model_config['in_channels'], 
                            n_classes=n_classes,
                            activation=activation,
                            kernels_per_layer=model_config['kernels_per_layer'],
                            psc=psc,
                            chl=chl,
                            nb_layers=nb_layers,
                            depth=model_config['depth'])                 
    
    # elif model_config['model']=='PC_UNet':
    #     #net=UNet(model_config['num_classes'], model_config['in_channels'], model_config['depth'], model_config['start_filts'], model_config['activation'])
    #     net=PConvUNet(layer_size=7, input_channels=model_config['in_channels'], output_channels=model_config['num_classes'], upsampling_mode='bicubic')
    #     model_config['Partial_conv']=True
    #     dataloader_config['completion']=False
    
    # elif model_config['model']=='DualBranchNet':
    #     net = DualBranchNet(depth=model_config['depth'], in_channels=model_config['in_channels'],
    #                         merge_mode=model_config['merge_mode'], activation=model_config['activation'],
    #                         freeze_key_id=model_config['freeze_key'])
        
    # elif model_config['model']=='SmaAt_DualBranchNet':
    #     if not('kernels_per_layer' in model_config.keys()):
    #         model_config['kernels_per_layer']=2
    #     net=SmaAt_DualBranchNet(in_channels=model_config['in_channels'], 
    #                             freeze_key_id=model_config['freeze_key'],
    #                             activation=model_config['activation'],
    #                             kernels_per_layer=model_config['kernels_per_layer'])

    else :
        raise ValueError("\"{}\" is not a valid mode for model.".format(model_config['model']))
    

    if model_config['weight_load'] is not(None):
        net.load_state_dict(load(model_config['weight_load']))
        print("\nweights loaded \n")
    
    print('\nSet up the saving folder')
    #initiate the saver and create a directory to store results, copy dataloader and architecture in the directory
    saver=saving_tool(train_config['checkpoints_path'], train_config['name'], model_config['model'])
    train_config['checkpoints_path']=saver.result_dir #update the saving directory
    trainer = Trainer(net, train_config, dataloader_config, criterion_config, optimizer_config, scheduler_config)
    
    try:
        print(trainer)
        trainer.run()
    except KeyboardInterrupt:

        filename = 'interrupted_'+ train_config['name']
        path = os.path.join(train_config['checkpoints_path'],filename)
        save(trainer.model.net.state_dict(), path)
        print()
        print(80*'_')
        print('Training Interrupted')
        print('Current State saved.')
    
    path = os.path.join(train_config['checkpoints_path'],'training_finished_'+train_config['name']+'_.pt')
    save(trainer.model.net.state_dict(),path)
    
    print('\nTraining finished, running prediction on test set..')
    trainer.model.net.load_state_dict(load(os.path.join(train_config['checkpoints_path'], 
                                                        train_config['name'] + 'best_valid_loss.pt')))
    print("\nweights loaded \n")
    loss, psc_predictions, chl_predictions=trainer.test_step()
    saver.save_pred(chl_predictions, psc_predictions, train_config['checkpoints_path'])

    print(f'\nTest loss : {loss}')
    print('\nSaving config and results .....')    
    
    saver.save_glob(loss, model_config['model'], trainer.state['epoch'],
                    trainer.state['epoch']-trainer.early_stopping.counter,
                    (optimizer_config['optimizer'],scheduler_config['scheduler'],optimizer_config['learning_rate']),
                    sum(p.numel() for p in net.parameters()),train_config['comment'])
                    
    dataloader_config.pop('transform', None)
    
    saver.save_config([model_config, dataloader_config, train_config, criterion_config, scheduler_config, optimizer_config],
                train_config['checkpoints_path'],
                train_config['name']+'_config')

        

    

if __name__ == '__main__':


    if 'nb_training' in train_config.keys():
        nb_training=train_config['nb_training']-1
    else :
        nb_training=0
    
    print(f'\nTraining {nb_training+1}\n')

    main(model_config, dataloader_config, train_config, criterion_config, optimizer_config, scheduler_config)

    
    #process_results(train_config['checkpoints_path'], train_config['name'])
    train_config['checkpoints_path']=train_config['checkpoints_path'][:-len(train_config['name'])]

########################## entrainement double branche (hasbeen) #########################################################################
    # if model_config['model']=='DualBranchNet' or model_config['model']=='SmaAt_DualBranchNet':
    #     print('training chl finished, starting training psc')
        
    #     model_config['weight_load']=os.path.join(train_config['checkpoints_path'], train_config['name'], train_config['name']+'best_valid_loss.pt')
    #     train_config['name']+='_psc'
    #     train_config['comment']+=' psc'
    #     model_config['freeze_key']='1'
        
    #     if 'KL_div' in criterion_config.keys() and criterion_config['KL_div']!=0.:   
            
    #         for key in criterion_config.keys():
    #             criterion_config[key]=0.
    #         criterion_config['KL_div']=1.  
            
    #     else :
    #         for key in criterion_config.keys():
    #             criterion_config[key]=0.
    #         criterion_config['MSE_psc']=1.
            
    #     dataloader_config['transform']=None
        
        
    #     main(model_config, dataloader_config, train_config, criterion_config, optimizer_config, scheduler_config)
    #     # #process_results(train_config['checkpoints_path'], train_config['name'])

    #     train_config['checkpoints_path']=train_config['checkpoints_path'][:-len(train_config['name'])]

########################## entrainement double branche (hasbeen) #########################################################################

    
    for i in range(nb_training):
        print(f'\nTraining {i+1}\n')

        train_config['name']+='_'+str(i)
        dataloader_config['transform']=None
        main(model_config, dataloader_config, train_config, criterion_config, optimizer_config, scheduler_config)
        train_config['checkpoints_path']=train_config['checkpoints_path'][:-len(train_config['name'])]
        #process_results(train_config['checkpoints_path'], train_config['name'])
        
        # if model_config['model']=='DualBranchNet' or model_config['model']=='SmaAt_DualBranchNet':
        #     print('training chl finished, starting training psc')
        #     train_config['name']+='_psc'
        #     train_config['comment']+=' psc'
        #     for key in criterion_config.keys():
        #         criterion_config[key]=0.
        #     criterion_config['MSE_psc']=1.
        #     dataloader_config['transform']=None
            
            
        #     main(model_config, dataloader_config, train_config, criterion_config, optimizer_config, scheduler_config)
            
        #     #process_results(train_config['checkpoints_path'], train_config['name'])

        #     train_config['checkpoints_path']=train_config['checkpoints_path'][:-len(train_config['name'])]
    
                
