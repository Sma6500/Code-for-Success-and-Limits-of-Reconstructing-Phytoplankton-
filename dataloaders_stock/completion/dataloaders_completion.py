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
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, RandomSampler
from sklearn.model_selection import train_test_split
from utils.scaler import MinMaxScaler, StandardScaler, StandardScaler_xarray
import xarray as xr
import dask.array as da
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import sys
from utils.plot import imshow_area


# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

#not mandatory bot sometimes useful with dask arrays, 10000 is still an acceptable number but do not go much higher
sys.setrecursionlimit(10000)

class Dataset_psc_chl(Dataset):

    def __init__(self, mask, indexs, scaler_phy, scaler_chl, dataloader_config, transform=None, completion='inf value', test=False):

        super(Dataset_psc_chl, self).__init__()
        
        if completion in ('inf value', 'zeros'):
            self.completion=completion
        else:
            raise ValueError("\"{}\" is not valid for "
                             "completion. Only \"inf value\" and "
                             "\"zeros\" are allowed.".format(completion))
            
        self.dataloader_config=dataloader_config
        self.test=test
        print('initilisation du dataset')

        if not(test):
            self.physics_dataset=xr.open_dataset(self.dataloader_config['dataset_path_inputs'], chunks={'time': 100}).to_array(dim='variables').astype('float16').data[:,indexs].compute()
            print('données physics chargée')

            self.chl_dataset = xr.open_dataset(self.dataloader_config['dataset_path_chl_psc'], chunks={'time': 100}).to_array(dim='variables').astype('float16').sel(variables='CHL').data[indexs].compute()
            print('données chl chargée')

            self.psc_dataset = xr.open_dataset(self.dataloader_config['dataset_path_chl_psc'], chunks={'time': 100}).to_array(dim='variables').astype('float16').sel(variables=['Micro','Nano','Pico']).data[:,indexs].compute()
            print('données psc chargée')

        else : 
            self.physics_dataset=xr.open_dataset(self.dataloader_config['dataset_path_inputs'], chunks={'time': 100}).to_array(dim='variables').data
            self.chl_dataset = xr.open_dataset(self.dataloader_config['dataset_path_chl_psc'], chunks={'time': 100}).to_array(dim='variables').sel(variables='CHL').data
            self.psc_dataset = xr.open_dataset(self.dataloader_config['dataset_path_chl_psc'], chunks={'time': 100}).to_array(dim='variables').sel(variables=['Micro','Nano','Pico']).data
        
        #self.path_physics = physics_path
        #self.path_chl_psc=chl_psc_path
        self.scaler_phy, self.scaler_chl = scaler_phy, scaler_chl
        self.transform = transform
        self.mask= mask
        self.indexs=indexs
            
        
            
    def __len__(self):
        return len(self.indexs) 

    def __getitem__(self, index):
        """
        Main function of the CustomDataset class. 
        """
        
        if self.test :
            
            physics=self.physics_dataset[:,index].compute()
            chl=self.chl_dataset[index].compute()
            psc=self.psc_dataset[:,index].compute()

        else : 
            physics=self.physics_dataset[:,index]
            chl=self.chl_dataset[index]
            psc=self.psc_dataset[:,index]

        
        # on convertit les inf en nan  (c'est surtout pour le rotationnel du vent)
        physics = np.where(np.isinf(physics,), np.nan, physics)
        if self.dataloader_config['log_chl']:
            epsilon=1e-5
            chl=np.clip(chl,epsilon,10)
            chl=np.log10(chl)
        
        chl = np.where(self.mask[index]==2,np.nan, chl)
        psc = np.where(np.repeat(self.mask[np.newaxis,index],3,axis=0)==2,np.nan, psc)
        
        chl_masked=np.where(self.mask[index]==1,np.nan, chl)
        psc_masked=np.where(np.repeat(self.mask[np.newaxis,index],3,axis=0)==1,np.nan, psc)
        

        # on normalise
        physics=(physics - self.scaler_phy.mean[:,*[np.newaxis for i in range(len(physics.shape)-1)]]) / (self.scaler_phy.std[:,*[np.newaxis for i in range(len(physics.shape)-1)]] + self.scaler_phy.epsilon)
        
        chl=(chl-self.scaler_chl.mean)/self.scaler_chl.std
        chl_masked=(chl_masked-self.scaler_chl.mean)/self.scaler_chl.std

        inputs=np.concatenate((physics, chl_masked[np.newaxis], psc_masked))
        #inputs=np.concatenate((physics, chl[np.newaxis], psc))

        
        if self.completion=='inf value' : 
            inputs=np.where(da.isnan(inputs),-1000,inputs)
        elif self.completion=='zeros' :
            inputs=np.nan_to_num(inputs)
        
        inputs = torch.tensor(inputs, dtype=torch.float32)

        chl = torch.tensor(chl[np.newaxis], dtype=torch.float32)
        psc = torch.tensor(psc, dtype=torch.float32)


        if self.transform is not None:
            
            inputs=self.transform(inputs)   
            psc=self.transform(psc)
            chl=self.transform(chl)
                
            
        return inputs, (psc, chl)
 

def coord_to_index(lon, lat):
    return 100 - lat, lon + 180




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
    
    physics = xr.open_dataset(dataloader_config['dataset_path_inputs'], chunks='auto').to_array(dim='variables').data
    chl = xr.open_dataset(dataloader_config['dataset_path_chl_psc'], chunks='auto').to_array(dim='variables').sel(variables=['CHL']).data.squeeze()
    
    #on convertit les inf en nan  (c'est surtout pour le rotationnel du vent)
    physics = da.where(da.isinf(physics), np.nan, physics)
    
    if dataloader_config['log_chl']:
        epsilon=1e-5
        chl=da.clip(chl,epsilon,10)
        chl=da.log10(chl)
            
    #################################Masking with cloud like mask#################################

    #completion part, we create a masked database from cloud mask
    cloud_data=np.load("/home/luther/Documents/npy_data/cloud_mask_avw_1d_100km_glob.npy",  mmap_mode='r').astype(np.float32)

    
    random_index=np.random.randint(0,cloud_data.shape[0],size=(chl.shape[0])) #we've got a problem here folks, we cannot retrieve the exact mean and std used for training. 
    random_mask=cloud_data[random_index]
    
    # Mask out specified stations
    # Build a mask to add to the random_mask
        
    for station, coord in dataloader_config['stations_coord'].items():
        idx_lat, idx_lon = coord_to_index(*coord)
        random_mask[..., idx_lat, idx_lon] = 2
        
############ plus de vérification avec les données hplc, il faut refaire l'index pour le daily ##############
 #########  et probablement refaire la base de donnée au passage #############################
    # Adjust HPLC data masking
    # hplc_df = pd.read_csv(dataloader_config['hplc_path'])
    # hplc_df = hplc_df[hplc_df['Year'] > 1997]
    # hplc_df['time_idx'] -= 230
    # hplc_df = hplc_df.dropna(subset=['time_idx', 'grid_lat', 'grid_lon'])

    # for _, row in hplc_df.iterrows():
    #     random_mask[int(row['time_idx']), int(row['grid_lat']), int(row['grid_lon'])] = 2
########################################################################################################
    chl=da.where(random_mask>=1,chl,np.nan) #pour le calcul de la normalisation

    
    #################################Train_valid_test_splitter#################################
    
    # Split dataset into train/validation/test
    indices = np.arange(physics.shape[1])
    train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
    train_idx, valid_idx = train_test_split(train_idx, test_size=0.1111, random_state=42)

    physics_train= physics[:,train_idx]
    chl = chl[train_idx]
    
    random_mask_train, random_mask_valid, random_mask_test = random_mask[train_idx], random_mask[valid_idx], np.zeros(random_mask.shape)


    ################################# scaler #################################
    # Normalize data if required
    if dataloader_config['norm_mode']: 
        
        if not(dataloader_config['norm_mode'] in ('standard')):
            raise ValueError("\"{}\" is not a valid mode for "
                             "splitting. Only \"standard\" is hanlde with xarray loading.".format(dataloader_config['norm_mode']))   
        if dataloader_config['norm_mode']=='standard':
            scaler_phy=StandardScaler_xarray()
            scaler_chl=StandardScaler_xarray()
        
            
    scaler_phy.fit(physics_train)
    scaler_chl.fit(chl,dims_first=False)
        
    print('scaler_done')
    
    
    training_set=Dataset_psc_chl(mask=random_mask_train,
                                 indexs=train_idx,
                                 scaler_phy=scaler_phy,
                                 scaler_chl=scaler_chl,
                                 dataloader_config=dataloader_config,
                                 transform=dataloader_config['transform'], 
                                 completion=dataloader_config['completion'])
    print('train done')

    validation_set=Dataset_psc_chl(mask=random_mask_valid,
                                   indexs=valid_idx,
                                   scaler_phy=scaler_phy,
                                   scaler_chl=scaler_chl,
                                   dataloader_config=dataloader_config,
                                   transform=dataloader_config['transform'], 
                                   completion=dataloader_config['completion'])
    print('train and valid done')
    
    test_set=Dataset_psc_chl(mask=random_mask_test,
                             indexs=indices,
                             scaler_phy=scaler_phy,
                             scaler_chl=scaler_chl,
                             dataloader_config=dataloader_config,
                             transform=dataloader_config['transform'], 
                             completion=dataloader_config['completion'],
                             test=True)
    
    print('dataset_done')

    training_generator   = DataLoader(training_set,
                                      batch_size=dataloader_config['batch_size'],
                                      shuffle=True,num_workers=4, prefetch_factor=4)
                                      
    
    validation_generator = DataLoader(validation_set,
                                      batch_size=dataloader_config['batch_size'],
                                      shuffle=True,num_workers=4, prefetch_factor=4)
    
    test_generator = DataLoader(test_set, batch_size=1, shuffle=False)#, num_workers=4, prefetch_factor=4)#,pin_memory=True,num_workers=4)
    print('generator')

    return training_generator, validation_generator, test_generator, scaler_chl


if __name__=='__main__': 
    

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

        
    train_g, valid_g, test_g, scaler =get_dataloaders(dataloader_config)
    

            
            # def build_masked_dataset(self):
                
            #     print(str(self.mask_ratio) + " has been removed from inputs for completion")
            #         #adding random pixel mask in validset from trainset
            #         #ça doit allonger un peu la phase de validation
            #     index_to_mask_psc=np.argwhere(~np.isnan(self.dataset_psc[:,0])) #O parce qu'on regarde pour les Micro et on retire pour les 3
            #     index_to_mask_chl=np.argwhere(~np.isnan(self.dataset_chl))
            #     np.random.shuffle(index_to_mask_psc)
            #     np.random.shuffle(index_to_mask_chl)

            #     dataset_psc_masked=np.copy(self.dataset_psc)
            #     dataset_chl_masked=np.copy(self.dataset_chl)

            # #bon avec un break ya probablement moyen de s'en sortir mieux
            #     for i in range(len(index_to_mask_psc)):
            #         if i/len(index_to_mask_psc)<self.mask_ratio: 
            #             dataset_psc_masked[index_to_mask_psc[i][0],:,index_to_mask_psc[i][1],index_to_mask_psc[i][2]]=np.nan
            #     #chl
            #     for i in range(len(index_to_mask_chl)):
            #         if i/len(index_to_mask_chl)<self.mask_ratio: 
            #             dataset_chl_masked[tuple(index_to_mask_chl[i])]=np.nan #ici on peut remplacer par une valeur autre
                                

            #     return dataset_chl_masked, dataset_psc_masked
        
        
        
        
    
    #withdraw test stations from dataset
    # stations_array_idx={}
    # for station in dataloader_config['stations_coord'].keys():
    #     stations_array_idx[station]=coord_to_index(*dataloader_config['stations_coord'][station])
    #     psc[:,:,stations_array_idx[station][0],stations_array_idx[station][1]]=np.nan
    #     chl[:,stations_array_idx[station][0],stations_array_idx[station][1]]=np.nan
    
    

