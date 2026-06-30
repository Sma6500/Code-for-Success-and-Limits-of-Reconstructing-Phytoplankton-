#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:59:27 2024

@author: luther

"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         DATALOADERS                                   | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, RandomSampler
import torch
from utils.scaler import MinMaxScaler, StandardScaler
#hplc_points=np.load("/datatmp/home/lollier/npy_data/insitu/Hourany/hplc_correspondings.npy")
hplc_points=pd.read_csv("/home/luther/Documents/npy_data/insitu/hplc_merged_new.csv")
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

class Dataset_psc_chl(Dataset):

    def __init__(self, physics, chl, transform=None, completion='inf value'):

        super(Dataset_psc_chl, self).__init__()
        
        if completion in ('inf value', 'zeros'):
            self.completion=completion
        else:
            raise ValueError("\"{}\" is not valid for "
                             "completion. Only \"inf value\" and "
                             "\"zeros\" are allowed.".format(completion))
            
        self.dataset_physics = physics
        #self.dataset_psc=psc
        self.dataset_chl = chl

        
        self.transform = transform

    def __len__(self):
        return self.dataset_physics.shape[0] 

    def __getitem__(self, index):
        """
        Main function of the CustomDataset class. 
        """
        
        inputs=self.dataset_physics[index]

        if self.completion=='inf value' : 
            inputs[np.where(np.isnan(inputs))]=-1000
        elif self.completion=='zeros' :
            inputs[np.where(np.isnan(inputs))]=0.
        
        inputs=torch.Tensor(inputs)

            
        chl=torch.Tensor(self.dataset_chl[index])
        #psc=torch.Tensor(self.dataset_psc[index])
        
        if self.transform is not None:
            
            inputs=self.transform(inputs)   
            #psc=self.transform(psc)
            chl=self.transform(chl)
                
            
        return inputs, tuple([torch.unsqueeze(chl, dim=0)])
 

def coord_to_index(lon, lat):
    #retourne un tuple lat, lon index correspondant aux index associés au coordonnées dans le dataset
    idx_lon=lon+180
    idx_lat=50-lat    
    return (idx_lat, idx_lon)

    
    
    
    

def get_dataloaders(dataloader_config):
    """
    Parameters
    ----------
    dataloader_config : dict
        dataloader configuration (see config.py).

    Returns
    -------
    Pytorch Dataloader or dict of Dataloader 
    """
        
    #je pourrais faire une liste mais pas sur que ça améliore la lisibilité
    physics=np.load(dataloader_config['dataset_path_inputs'])
    #psc=np.load(dataloader_config['dataset_path_psc'])
    chl=np.load(dataloader_config['dataset_path_chl'])
    
    
    #withdraw test stations from dataset
    stations_array_idx={}
    for station in dataloader_config['stations_coord'].keys():
        stations_array_idx[station]=coord_to_index(*dataloader_config['stations_coord'][station])
        #psc[:,:,stations_array_idx[station][0],stations_array_idx[station][1]]=np.nan
        chl[:,stations_array_idx[station][0],stations_array_idx[station][1]]=np.nan
        
        
    #withdraw hplc stations from dataset
    df=hplc_points[hplc_points['Year'] > 1997]
    df.loc[:, 'time_idx'] = df['time_idx'] - 230 #on réajuste à 1998
    df = df.dropna(subset=['time_idx','grid_lat','grid_lon'])
    for i,row in df[['time_idx','grid_lat','grid_lon']].iterrows():
        
        #psc[int(row['time_idx']),:,int(row['grid_lat']),int(row['grid_lon'])]=np.nan
        chl[int(row['time_idx']),int(row['grid_lat']),int(row['grid_lon'])]=np.nan
    
    #on convertit les inf en nan  (c'est surtout pour le rotationnel du vent)
    physics = np.where(np.isinf(physics), np.nan, physics)

    if dataloader_config['log_chl']:
        epsilon=1e-5
        chl=np.clip(chl,epsilon,5)
        chl=np.log10(chl)
            
    #################################Train_valid_test_splitter#################################
    # Total number of timesteps
    total_timesteps = physics.shape[0]
    
    # Number of timesteps per year
    timesteps_per_year = 46
    
    # Calculate the total number of years in the dataset
    total_years = total_timesteps // timesteps_per_year
    
    # Initialize indices
    train_indices = []
    valid_indices = []
    test_indices = []
    
    # Iterate over years and append indices to sets
    for i in range(total_years):
        start_idx = i * timesteps_per_year
        end_idx = (i + 1) * timesteps_per_year
        
        if i>7 and i<13 :
            test_indices.append(slice(start_idx, end_idx))
        elif i >=20 :
            valid_indices.append(slice(start_idx, end_idx))
        else :
            train_indices.append(slice(start_idx, end_idx))


        
    
    # #by year
    #test_index=slice(-46*2,-1) #on prend les 2 dernières années pour le test faut ajouter None à la place de -1
    # valid_index=[slice(i,i+46) for i in range(46*4,physics.shape[0]-46*2,46*5)] 
    # train_index=[slice(i,i+(46*4)) for i in range(0,physics.shape[0]-46*2,46*5)]
    
    # #trainset
    physics_trainset=np.concatenate([physics[index] for index in train_indices])
    #psc_trainset=np.concatenate([psc[index] for index in train_indices])
    chl_trainset=np.concatenate([chl[index] for index in train_indices])

    
    #adding random pixel mask in validset from trainset
    #ça doit allonger un peu la phase de validation
#     index_to_mask=np.argwhere(~np.isnan(psc_trainset[:,0])) #O parce qu'on regarde pour les Micro et on retire pour les 3
#     np.random.shuffle(index_to_mask)
#     psc_masked_validset=np.copy(psc_trainset)
#     chl_masked_validset=np.copy(chl_trainset)

# #bon avec un break ya probablement moyen de s'en sortir mieux
#     for i in range(len(index_to_mask)):
#         if i/len(index_to_mask)<0.15: #as we already have year, i only mask 10% pixels
#             psc_trainset[index_to_mask[i][0],:,index_to_mask[i][1],index_to_mask[i][2]]=np.nan
#             chl_trainset[tuple(index_to_mask[i])]=np.nan
#         else : 
#             psc_masked_validset[index_to_mask[i][0],:,index_to_mask[i][1],index_to_mask[i][2]]=np.nan
#             chl_masked_validset[tuple(index_to_mask[i])]=np.nan
              
    # #validset
    physics_validset=np.concatenate([physics[index] for index in valid_indices])
    #psc_validset=np.concatenate([psc[index] for index in valid_indices])
    chl_validset=np.concatenate([chl[index] for index in valid_indices])            
            
    physics_testset=np.concatenate([physics[index] for index in test_indices])
    #psc_testset=np.concatenate([psc[index] for index in test_indices])
    chl_testset=np.concatenate([chl[index] for index in test_indices])      
    
    ################################# scaler #################################


    if dataloader_config['norm_mode']: 
        
        if not(dataloader_config['norm_mode'] in ('min_max', 'standard')):
            raise ValueError("\"{}\" is not a valid mode for "
                             "splitting. Only \"min_max\" and \"standard\" are allowed.".format(dataloader_config['norm_mode']))   
        if dataloader_config['norm_mode']=='standard':
            scaler=StandardScaler()
        elif dataloader_config['norm_mode']=='min_max':
            scaler=MinMaxScaler()

        #on swap les axes de manière à ce que le channel soit la dernière dimension puis on reswap dans le bon sens
        #un jour il faudra probablement réécrire le code avec les channels de features en dernier quoiqu'il arrive
        physics_trainset=np.swapaxes(scaler.fit_transform(np.swapaxes(physics_trainset, 1, -1)),-1,1)            
        physics_validset=np.swapaxes(scaler.transform(np.swapaxes(physics_validset, 1, -1)),-1,1)            
        physics_testset=np.swapaxes(scaler.transform(np.swapaxes(physics_testset, 1, -1)),-1,1)            

        chl_trainset=np.squeeze(scaler.fit_transform(np.expand_dims(chl_trainset, -1)))            
        chl_validset=np.squeeze(scaler.transform(np.expand_dims(chl_validset, -1)))          
        chl_testset=np.squeeze(scaler.transform(np.expand_dims(chl_testset, -1)))         



    training_set=Dataset_psc_chl(physics_trainset,chl_trainset,#psc_trainset, 
                                 transform=dataloader_config['transform'], 
                                 completion=dataloader_config['completion'])
    
    validation_set=Dataset_psc_chl(physics_validset,chl_validset,#psc_validset, 
                                 transform=dataloader_config['transform'], 
                                 completion=dataloader_config['completion'])

    
    test_set=Dataset_psc_chl(physics_testset,chl_testset,#psc_testset, 
                                 transform=dataloader_config['transform'], 
                                 completion=dataloader_config['completion'])
    
        
    training_generator   = DataLoader(training_set,
                                      batch_size=dataloader_config['batch_size'],
                                      shuffle=True)
    
    validation_generator = DataLoader(validation_set,
                                      batch_size=dataloader_config['batch_size'],
                                      shuffle=True)
    
    test_generator = DataLoader(test_set, batch_size=1, shuffle=False)
        
    return training_generator, validation_generator, test_generator
    


if __name__=='__main__': 
    

    path="/home/luther/Documents/npy_data/"
    
    dataloader_config={'dataset_path_inputs':path+"physics/processed_physics_1998_2020_8d_100_lat50.npy",
                       'dataset_path_psc':path+"PSC/PSC_Roy_8d_1998_2020_lat50_100km.npy",
                       'dataset_path_chl':path+"chl/chl_avw_1998_2020_100k_8d_lat50.npy",
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
    
    train_g, valid_g, test_g=get_dataloaders(dataloader_config)
    

            
        
        
        
        
        
    
    
    
    

