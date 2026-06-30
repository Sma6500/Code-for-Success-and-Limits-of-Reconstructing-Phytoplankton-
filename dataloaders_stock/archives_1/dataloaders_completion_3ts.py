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
from sklearn.model_selection import train_test_split
import torch
from utils.scaler import MinMaxScaler, StandardScaler
#hplc_points=np.load("/datatmp/home/lollier/npy_data/insitu/Hourany/hplc_correspondings.npy")
hplc_points=pd.read_csv("/home/luther/Documents/npy_data/insitu/hplc_merged_new_glob_100km.csv")
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

class Dataset_psc_chl(Dataset):

    def __init__(self, physics, chl, psc, transform=None, completion='inf value', mask_ratio=0.5):

        super(Dataset_psc_chl, self).__init__()
        
        if completion in ('inf value', 'zeros'):
            self.completion=completion
        else:
            raise ValueError("\"{}\" is not valid for "
                             "completion. Only \"inf value\" and "
                             "\"zeros\" are allowed.".format(completion))
            
        self.mask_ratio=mask_ratio
        self.dataset_physics = physics
        self.dataset_psc=psc
        self.dataset_chl = chl
        self.transform = transform

        self.dataset_chl_masked, self.dataset_psc_masked = self.build_masked_dataset()
            
    def build_masked_dataset(self):
        
        print(str(self.mask_ratio) + " has been removed from inputs for completion")
            #adding random pixel mask in validset from trainset
            #ça doit allonger un peu la phase de validation
        index_to_mask_psc=np.argwhere(~np.isnan(self.dataset_psc[:,0])) #O parce qu'on regarde pour les Micro et on retire pour les 3
        index_to_mask_chl=np.argwhere(~np.isnan(self.dataset_chl))
        np.random.shuffle(index_to_mask_psc)
        np.random.shuffle(index_to_mask_chl)

        dataset_psc_masked=np.copy(self.dataset_psc)
        dataset_chl_masked=np.copy(self.dataset_chl)

    #bon avec un break ya probablement moyen de s'en sortir mieux
        for i in range(len(index_to_mask_psc)):
            if i/len(index_to_mask_psc)<self.mask_ratio: 
                dataset_psc_masked[index_to_mask_psc[i][0],:,index_to_mask_psc[i][1],index_to_mask_psc[i][2]]=np.nan
        #chl
        for i in range(len(index_to_mask_chl)):
            if i/len(index_to_mask_chl)<self.mask_ratio: 
                dataset_chl_masked[tuple(index_to_mask_chl[i])]=np.nan #ici on peut remplacer par une valeur autre
                        

        return dataset_chl_masked, dataset_psc_masked


    def __len__(self):
        return self.dataset_physics.shape[0]-2

    def __getitem__(self, index):
        """
        Main function of the CustomDataset class. 
        """
        
        index+=1
        inputs=np.concatenate((self.dataset_physics[index-1],np.expand_dims(self.dataset_chl_masked[index-1],axis=0),self.dataset_psc_masked[index-1],
                               self.dataset_physics[index],np.expand_dims(self.dataset_chl_masked[index],axis=0),self.dataset_psc_masked[index],
                               self.dataset_physics[index+1],np.expand_dims(self.dataset_chl_masked[index+1],axis=0),self.dataset_psc_masked[index+1]))

        if self.completion=='inf value' : 
            inputs[np.where(np.isnan(inputs))]=-1000
        elif self.completion=='zeros' :
            inputs[np.where(np.isnan(inputs))]=0.
        
        inputs=torch.Tensor(inputs)

            
        chl=torch.Tensor(self.dataset_chl[index])
        psc=torch.Tensor(self.dataset_psc[index])
        
        if self.transform is not None:
            
            inputs=self.transform(inputs)   
            psc=self.transform(psc)
            chl=self.transform(chl)
                
            
        return inputs, (psc, torch.unsqueeze(chl, dim=0))
 

def coord_to_index(lon, lat):
    #retourne un tuple lat, lon index correspondant aux index associés au coordonnées dans le dataset
    idx_lon=lon+180
    idx_lat=100-lat    
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
    psc=np.load(dataloader_config['dataset_path_psc'])
    chl=np.load(dataloader_config['dataset_path_chl'])
    
    
    #withdraw test stations from dataset
    stations_array_idx={}
    for station in dataloader_config['stations_coord'].keys():
        stations_array_idx[station]=coord_to_index(*dataloader_config['stations_coord'][station])
        psc[:,:,stations_array_idx[station][0],stations_array_idx[station][1]]=np.nan
        chl[:,stations_array_idx[station][0],stations_array_idx[station][1]]=np.nan
        
        
    #withdraw hplc stations from dataset
    df=hplc_points[hplc_points['Year'] > 1997]
    df.loc[:, 'time_idx'] = df['time_idx'] - 230 #on réajuste à 1998
    df = df.dropna(subset=['time_idx','grid_lat','grid_lon'])
    for i,row in df[['time_idx','grid_lat','grid_lon']].iterrows():
        
        psc[int(row['time_idx']),:,int(row['grid_lat']),int(row['grid_lon'])]=np.nan
        chl[int(row['time_idx']),int(row['grid_lat']),int(row['grid_lon'])]=np.nan
    
    #on convertit les inf en nan  (c'est surtout pour le rotationnel du vent)
    physics = np.where(np.isinf(physics), np.nan, physics)
    
    if dataloader_config['log_chl']:
        epsilon=1e-5
        chl=np.clip(chl,epsilon,10)
        chl=np.log10(chl)
            
    #################################Train_valid_test_splitter#################################

    
    
    # Generate train/test indices
    total_size = len(physics)
    indices = np.arange(total_size)
    
    # First split: Train/Test (10% for the test set)
    train_indices, test_indices = train_test_split(indices, test_size=0.10, random_state=42)
    
    # Second split: Train/Validation (10% of remaining 90%, i.e., ~10% of total dataset for validation)
    train_indices, valid_indices = train_test_split(train_indices, test_size=0.1111, random_state=42)

    # Now use these indices to create your train/valid/test datasets

    # Physics data
    physics_trainset = physics[train_indices]
    physics_validset = physics[valid_indices]
    physics_testset = physics[test_indices]
    
    chl_trainset=chl[train_indices]
    chl_validset=chl[valid_indices]
    chl_testset=chl[test_indices]
    
    psc_trainset=psc[train_indices]
    psc_validset=psc[valid_indices]
    psc_testset=psc[test_indices]


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

    try :
        train_ratio,valid_ratio,test_ratio=dataloader_config['mask_ratio']
    except KeyError :
        print('aucune configuration du mask_ratio')
        
    #################################deleting variable to clear ram #################################

    del physics
    del chl
    del psc
    
    
    training_set=Dataset_psc_chl(physics_trainset,chl_trainset,psc_trainset, 
                                 transform=dataloader_config['transform'], 
                                 completion=dataloader_config['completion'],
                                 mask_ratio=train_ratio)
    
    validation_set=Dataset_psc_chl(physics_validset,chl_validset,psc_validset, 
                                 transform=dataloader_config['transform'], 
                                 completion=dataloader_config['completion'],
                                 mask_ratio=valid_ratio)

    
    test_set=Dataset_psc_chl(physics_testset,chl_testset,psc_testset, 
                                 transform=dataloader_config['transform'], 
                                 completion=dataloader_config['completion'],
                                 mask_ratio=test_ratio)
    
        
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
    #path="/datatmp/home/lollier/npy_data/"
    
    dataloader_config={'dataset_path_inputs':path+"physics/physics_completion/partial_1998_2023_reanalysis_100km_daily_16.npy",
                       'dataset_path_psc':path+"completion/occci_cmems/psc_p_occci_glob_100_1998_2023_daily.npy",
                       'dataset_path_chl':path+"completion/occci_cmems/chl_occci_glob_100_1998_2023_daily.npy",
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

    
    train_g, valid_g, test_g=get_dataloaders(dataloader_config)
    

            
        
        
        
        
        
    
    
    
    

