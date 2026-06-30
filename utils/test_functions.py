#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:33:15 2023

@author: luther

Notes : en fait ici j'importe  le dataloader correspondant mais en vrai c'est pas 
utile parce que ya pas besoin de la chl ni des psc dans le dataloader.

"""

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         Functions                                     | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import numpy as np
import json
import os 
import torch
import subprocess

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   UTILS FUNCTIONS TO PROCESS DATA, SAVE AND LOAD RESULT               | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

#############################Load config######################################



class config():
    
    def __init__(self, path_config, transformation=None):
        
        self.model_config, self.dataloader_config, self.train_config, self.criterion_config, self.scheduler_config, self.optimizer_config=self.load_config(path_config,transformation=transformation)
        self.completion=False
        
    def __str__(self):
        print("Oceano infos\n")
        self.infos()
        print("#"*80)
        print("Deep learning infos\n")
        self.hyperparameters()
        return('')

    def load_config(self,path_config, transformation=None):

        with open(path_config, "r") as config_file:
            configs=json.loads(json.load(config_file))

            model_config=configs[0]

            dataloader_config=configs[1]
            dataloader_config['transform']=transformation

            train_config=configs[2]

            criterion_config=configs[3]

            scheduler_config=configs[4]

            optimizer_config=configs[5]

        return model_config, dataloader_config, train_config, criterion_config, scheduler_config, optimizer_config
    
    def read_dataset_infos(self, path):
        with open(path, 'r') as f:
                data=f.readlines()
                for line in data:
                    if 'completion' in line :
                        self.completion=True
                    print(line)

    def infos(self):
        try :
            print('#'*80)
            self.read_dataset_infos(self.dataloader_config['dataset_path_inputs'][:-3]+'txt')
            if not(self.completion):
                try :
                    print(f"completion before dataloader : {self.dataloader_config['completion']}")
                except : print('completions infos are not available')

            print('#'*80)
            self.read_dataset_infos(self.dataloader_config['dataset_path_psc'][:-3]+'txt')
            print('#'*80)
            self.read_dataset_infos(self.dataloader_config['dataset_path_chl'][:-3]+'txt')
        except :
            print('Infos are not available for this dataset')
        

    def hyperparameters(self):
        try :
            print('#'*80)
            print('\nHyperparameters configs :\n')
            print(f"normalization method : {self.dataloader_config['norm_mode']}\n")
            print(f"train, valid, test in {self.dataloader_config['split_mode']} : {self.dataloader_config['split_index']}\n")
            print(f"model : {self.model_config['model']} \n")
            
            cost_function=""
            for key in self.criterion_config.keys():
                if key!='details':
                    cost_function+=f" + {key} * {self.criterion_config[key]}"
            print(f"Cost function : {cost_function[2:]} \n")
            print(f"Optimizer / base lr : {self.optimizer_config['optimizer']} / {self.optimizer_config['learning_rate']}\n")
            print(f"Scheduler : {self.scheduler_config['scheduler']}\n")
            print('#'*80)
        except : print("hyperparameters not available")
             
#############################test data generator################################

# def build_data_gen(physics, psc, chl, path_result, dataloader_config, scaler):
    
#     import sys
#     sys.path.append(path_result)
#     from dataloaders import Dataset_psc_chl
#     from torch.utils.data import DataLoader
    
#     if dataloader_config['log_chl']:
#         chl=np.log10(chl)
    
#     physics=np.swapaxes(scaler.fit_transform(np.swapaxes(physics, 1, -1)),-1,1)     

#     if dataloader_config['norm_chl']:
#         chl=np.squeeze(scaler.fit_transform(np.expand_dims(chl, -1)))     
    
    
#     total_timesteps = physics.shape[0]

#     # Number of timesteps per year
#     timesteps_per_year = 46
    
#     # Calculate the total number of years in the dataset
#     total_years = total_timesteps // timesteps_per_year
    
#     # Initialize indices

#     test_indices = []
    
#     # Iterate over years and append indices to sets
#     for i in range(total_years):
#         start_idx = i * timesteps_per_year
#         end_idx = (i + 1) * timesteps_per_year
        
#         if i==9 or i==16:
#             test_indices.append(slice(start_idx, end_idx))

        
#     #test_indices=slice(-46*2,None) #on prend les 2 dernières années pour le test

#     physics_testset=np.concatenate([physics[index] for index in test_indices])
#     psc_testset=np.concatenate([psc[index] for index in test_indices])
#     chl_testset=np.concatenate([chl[index] for index in test_indices])    
     
#     test_dataset=Dataset_psc_chl(physics_testset,chl_testset, psc_testset, transform=dataloader_config['transform'], completion=dataloader_config['completion'])
#     test_generator = DataLoader(test_dataset, batch_size=1, shuffle=False)

#     glob_dataset=Dataset_psc_chl(physics, chl, psc, transform=dataloader_config['transform'], completion=dataloader_config['completion'])
#     glob_generator = DataLoader(glob_dataset, batch_size=1, shuffle=False)
    
#     return glob_generator, test_generator


#################Load Net, predictions and test predictions#####################

#bon là je vais load la config sauvegardée dans les résultats mais en soit
#vu que j'execute le notebook directement dans le script training je pourrais
#probablement charger directement le fichier config.py
from torch import load
import os
import numpy as np 
import sys
from tqdm import tqdm
sys.path.append("/home/luther/Documents/scripts_training/models/")
sys.path.append("/usr/home/lollier/Documents/scripts_training/models/")

from models.DBNet import DualBranchNet
from models.SmaAt_DBNet import SmaAt_DualBranchNet
from models.UNet import UNet
from models.UNet_DSC import UNet_DSC
from models.UNet_CBAM import UNet_CBAM
from models.light_SmaAt_UNet import light_SmaAt_UNet
from models.SmaAt_UNet import SmaAt_UNet

import torch
import torch.nn as nn

def load_results(dataloader_config, model_config, train_config, glob_generator, test_generator, scaler):
    
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
    psc=True,
    chl_b=False
    n_classes=4
    if 'chl' in model_config.keys() and model_config['chl']:
        if model_config['chl']==1:
            psc=False
            chl_b=True
            n_classes=3
        elif model_config['chl']==2:
            psc=False
            chl_b=False
            n_classes=1
        elif model_config['chl']==3:
            psc=False
            chl_b=False
            n_classes=2
        else :
            raise ValueError("Attention, mauvaise configuration de l'argument chl de model_config")
            
    
    if model_config['model']=='UNet':
        
        net=UNet(in_channels=model_config['in_channels'],
                 out_channels=4,#tjrs 4 pour PSC et CHL
                  depth=model_config['depth'],
                  merge_mode=model_config['merge_mode'], 
                  activation=model_config['activation'],
                  psc=True)
        
                
    elif model_config['model']=='UNet_DSC':        
        
        net=UNet_DSC(in_channels=model_config['in_channels'],
                     out_channels=4,#tjrs 4 pour PSC et CHL
                     depth=model_config['depth'],
                     merge_mode=model_config['merge_mode'], 
                     activation=model_config['activation'],
                     start_filts=nb_layers,
                     psc=True,
                     kernels_per_layer=model_config['kernels_per_layer'])
        
    elif model_config['model']=='UNet_CBAM':
        
        net=UNet_CBAM(n_channels=model_config['in_channels'],
                     n_classes=4,#tjrs 4 pour PSC et CHL
                     depth=model_config['depth'],
                     nb_layers=nb_layers,
                     psc=True,)

    elif model_config['model']=='SmaAt-UNet':
        
        net=SmaAt_UNet(n_channels=model_config['in_channels'], 
                       n_classes=n_classes,
                       activation=activation,
                       kernels_per_layer=model_config['kernels_per_layer'],
                       psc=psc,chl=chl_b,
                       nb_layers=nb_layers,
                       time_encoded=time_encoded)
        
    elif model_config['model']=='light_SmaAt-UNet':
        print("attention, config outdated, pas de mise à jour pour l'option model_config[chl]")

        net=light_SmaAt_UNet(n_channels=model_config['in_channels'], 
                            n_classes=n_classes,
                            activation=activation,
                            kernels_per_layer=model_config['kernels_per_layer'],
                            psc=psc,
                            chl=chl_b,
                            nb_layers=nb_layers,
                            depth=model_config['depth'])      

    elif model_config['model']=='DualBranchNet':
        net = DualBranchNet(depth=model_config['depth'], in_channels=model_config['in_channels'],
                            merge_mode=model_config['merge_mode'], activation=model_config['activation'],
                            freeze_key_id=model_config['freeze_key'])
        
    elif model_config['model']=='SmaAt_DualBranchNet':
        if not('kernels_per_layer' in model_config.keys()):
            model_config['kernels_per_layer']=2
        net=SmaAt_DualBranchNet(in_channels=model_config['in_channels'], 
                                freeze_key_id=model_config['freeze_key'],
                                activation=model_config['activation'],
                                kernels_per_layer=model_config['kernels_per_layer'])
    else :
        raise ValueError("\"{}\" is not a valid mode for"
                              "merging up and down paths. "
                              "Only \"DualBranchNet\" and \"SmaAt_DualBranchNet\" are allowed.".format(model_config['model']))
    
    net.to(device=train_config['device'])
    net.load_state_dict(load(os.path.join(train_config['checkpoints_path'], 
                                          train_config['name'] + 'best_valid_loss.pt'),map_location='cuda:0'))


    glob_predictions=[]
    test_predictions=[]
    net.eval()

    with torch.no_grad():
        
        #glob
        for inputs,targets in tqdm(glob_generator):
            
            preds=net(inputs.to(device=train_config['device']))
            
            if (isinstance(preds, tuple) or isinstance(preds, list)) and len(preds)>1:
                preds=[pred.detach().cpu().numpy() for pred in preds]
                
                if 'chl' in model_config.keys() and model_config['chl']==3:
                    sign=preds[-1]
                    preds=[pred*np.sign(sign) for pred in preds[:-1]]
                    
                    if dataloader_config['norm_chl'] : #and dataloader_config['norm_mode']=='standard': #actually no min max experiences has been conducted
                        preds=[scaler.mean+(pred*(scaler.std+scaler.epsilon)) for pred in preds]
                    
                    preds.append(sign)
                    
                elif dataloader_config['norm_chl'] : #and dataloader_config['norm_mode']=='standard': #actually no min max experiences has been conducted
                    preds[1]=scaler.mean+(preds[1]*(scaler.std+scaler.epsilon))
                if dataloader_config['log_chl']:
                        preds[1]=10**preds[1]
            else :
                preds=preds[0].detach().cpu().numpy()
                
                if not(psc) and not(chl_b):
                    if dataloader_config['norm_chl'] : #and dataloader_config['norm_mode']=='standard': #actually no min max experiences has been conducted
                        preds=scaler.mean+(preds*(scaler.std+scaler.epsilon))
                    if dataloader_config['log_chl']:
                            preds=10**preds
                    
            glob_predictions.append(preds)
        
        #test
        for inputs,targets in tqdm(test_generator):
            
            preds=net(inputs.to(device=train_config['device']))
            
            if (isinstance(preds, tuple) or isinstance(preds, list)) and len(preds)>1:
                preds=[pred.detach().cpu().numpy() for pred in preds]
                
                if 'chl' in model_config.keys() and model_config['chl']==3:
                    sign=preds[-1]
                    preds=[pred*np.sign(sign) for pred in preds[:-1]]
                    
                    if dataloader_config['norm_chl'] : #and dataloader_config['norm_mode']=='standard': #actually no min max experiences has been conducted
                        preds=[scaler.mean+(pred*(scaler.std+scaler.epsilon)) for pred in preds]
                    
                    preds.append(sign)

                    
                elif dataloader_config['norm_chl'] :#and dataloader_config['norm_mode']=='standard': #actually no min max experiences has been conducted
                    preds[1]=scaler.mean+(preds[1]*(scaler.std+scaler.epsilon))
                if dataloader_config['log_chl']:
                        preds[1]=10**preds[1]
            else :
                preds=preds[0].detach().cpu().numpy()
                
                if not(psc) and not(chl_b):
                    if dataloader_config['norm_chl'] : #and dataloader_config['norm_mode']=='standard': #actually no min max experiences has been conducted
                        preds=scaler.mean+(preds*(scaler.std+scaler.epsilon))
                    if dataloader_config['log_chl']:
                            preds=10**preds
                            
            test_predictions.append(preds)
            
    return net, glob_predictions, test_predictions
            


#############################Run Jupyter Notebook######################################


import os
import glob
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError

path_notebook_model="/home/luther/Documents/scripts_training/notebook_model.ipynb"

def process_results(path_result, net_name, path_notebook_model=path_notebook_model):
    
    notebook='notebook_model'
    notebook_out=os.path.join(path_result,net_name+'_out')
    with open(path_notebook_model) as n:
        nb = nbformat.read(n, as_version=4)
        ep = ExecutePreprocessor(timeout=int(600), kernel_name='python3')
        try:
            print('Running :', notebook)
            out = ep.preprocess(nb, {'metadata': {'path': path_result}})
        except CellExecutionError:
            out = None
            msg = 'Error executing the notebook "%s".\n' % notebook_out
            msg += 'See notebook for the traceback.'
            print(msg)
        except TimeoutError:
            msg = 'Timeout executing the notebook .\n' 
            print(msg)
        finally:
            # Write output file
            with open(notebook_out + '.ipynb', mode='wt') as f:
                nbformat.write(nb, f)
                

    


    
    
    
    
    
    
