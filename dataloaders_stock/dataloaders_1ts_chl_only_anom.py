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
from utils.functions import extend_nan_both_dimensions
#hplc_points=np.load("/datatmp/home/lollier/npy_data/insitu/Hourany/hplc_correspondings.npy")
hplc_points=pd.read_csv("/home/luther/Documents/npy_data/insitu/hplc_merged_new.csv")
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

class Dataset_psc_chl(Dataset):

    def __init__(self, physics, chl, psc, chl_anom, transform=None, completion='inf value'):

        super(Dataset_psc_chl, self).__init__()
        
        if completion in ('inf value', 'zeros'):
            self.completion=completion
        else:
            raise ValueError("\"{}\" is not valid for "
                             "completion. Only \"inf value\" and "
                             "\"zeros\" are allowed.".format(completion))
        
        self.dataset_physics = physics
        self.dataset_psc=psc
        self.dataset_chl = chl
        self.dataset_chl_anom=chl_anom
        
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
        chl_anom=torch.Tensor(self.dataset_chl_anom[index])
        psc=torch.Tensor(self.dataset_psc[index])
        
        if self.transform is not None:
            
            inputs=self.transform(inputs)   
            psc=self.transform(psc)
            chl=self.transform(chl)
            chl_anom=self.transform(chl_anom)
            
        return inputs, (torch.unsqueeze(chl, dim=0), torch.unsqueeze(chl_anom, dim=0))
 
#(psc,chl,sign,magnitude)

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
    print("Côtes masquées à 2 pixels")
    physics=np.load(dataloader_config['dataset_path_inputs'], mmap_mode='r').astype(np.float16)
    mask_coast = np.where(np.isnan(extend_nan_both_dimensions(physics[0,0], steps=2)), np.nan,1)
    physics*=mask_coast

    physics_tot=np.load(dataloader_config.get("dataset_path_inputs_test","/home/luther/Documents/npy_data/physics/processed_physics_1993_2023_8d_lat50_100.npy"), mmap_mode='r').astype(np.float16)*mask_coast
    psc=np.load(dataloader_config['dataset_path_psc'])*mask_coast
    chl=np.load(dataloader_config['dataset_path_chl'])*mask_coast
    

    #anomalies computing
    if dataloader_config.get('phy_anomalies',False):
        
        #except change, 1426/46=31 years, we compute the mean over the 31 available years because we suppose it does not change a lot between 30 and 26 years.
        phy_mean=np.concatenate([np.nanmean(physics_tot.reshape(31,46,8,100,360),axis=0)]*31,axis=0)
        phy_anom_tot=physics_tot-phy_mean
        phy_anom_tot=phy_anom_tot[:,dataloader_config['phy_anomalies']]
        phy_anom=phy_anom_tot[230:]
    
    if dataloader_config.get('chl_anomalies', False):
        
        chl_clim=np.concatenate([np.nanmean(chl.reshape(26,46,100,360),axis=0)]*26,axis=0)
        chl_anom=chl-chl_clim
        #on recalcul un peu l'anomalie 
        chl_anom=np.sign(chl_anom) * (np.log10(np.clip(np.abs(chl_anom),1e-5,10))+5)
        
    else :
        ValueError("chl_anomalies set to false, wrong dataloader")
        
    #select variables
    variables_physiques=dataloader_config.get('variables_physiques', [0,1,2,3,4,5,6,7]) #ou slice(0,None)
    
    physics=physics[:,variables_physiques]
    physics_tot=physics_tot[:,variables_physiques]
    
    #merge 
    if dataloader_config.get('phy_anomalies',False):
        physics=np.concatenate([physics,phy_anom],axis=1)
        physics_tot=np.concatenate([physics_tot,phy_anom_tot],axis=1)

    
    #withdraw test stations from dataset
    stations_array_idx={}
    for station in dataloader_config['stations_coord'].keys():
        stations_array_idx[station]=coord_to_index(*dataloader_config['stations_coord'][station])
        psc[:,:,stations_array_idx[station][0],stations_array_idx[station][1]]=np.nan
        chl[:,stations_array_idx[station][0],stations_array_idx[station][1]]=np.nan
        chl_anom[:,stations_array_idx[station][0],stations_array_idx[station][1]]=np.nan
        
    #withdraw hplc stations from dataset
    df=hplc_points[hplc_points['Year'] > 1997]
    df.loc[:, 'time_idx'] = df['time_idx'] - 230 #on réajuste à 1998
    df = df.dropna(subset=['time_idx','grid_lat','grid_lon'])
    for i,row in df[['time_idx','grid_lat','grid_lon']].iterrows():
        
        psc[int(row['time_idx']),:,int(row['grid_lat']),int(row['grid_lon'])]=np.nan
        chl[int(row['time_idx']),int(row['grid_lat']),int(row['grid_lon'])]=np.nan
        chl_anom[int(row['time_idx']),int(row['grid_lat']),int(row['grid_lon'])]=np.nan

    #on convertit les inf en nan  (c'est surtout pour le rotationnel du vent)
    physics = np.where(np.isinf(physics), np.nan, physics)
    physics_tot = np.where(np.isinf(physics_tot), np.nan, physics_tot)

    if dataloader_config['log_chl']:
        epsilon=1e-5
        chl=np.clip(chl,epsilon,10)
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
    # print('test set up differently take care')
    # for i in range(total_years):
    #     start_idx = i * timesteps_per_year
    #     end_idx = (i + 1) * timesteps_per_year
        
    #     if i<=12 :
    #         test_indices.append(slice(start_idx, end_idx))
    #     elif i >=22:
    #         valid_indices.append(slice(start_idx, end_idx))
    #     else :
    #         train_indices.append(slice(start_idx, end_idx))
            
    for i in range(total_years):
        start_idx = i * timesteps_per_year
        end_idx = (i + 1) * timesteps_per_year
        
        if i==0 or i>=16 :
            test_indices.append(slice(start_idx, end_idx))
        elif i >=12 and i <16 :
            valid_indices.append(slice(start_idx, end_idx))
        else :
            train_indices.append(slice(start_idx, end_idx))

    
        # Use generators and in-memory data management to reduce footprint
    def create_subset(slices, data):
        return np.concatenate([data[s] for s in slices], axis=0)
    #trainset
    physics_trainset = create_subset(train_indices, physics)
    psc_trainset = create_subset(train_indices, psc)
    chl_trainset = create_subset(train_indices, chl)
    chl_anom_trainset = create_subset(train_indices, chl_anom)
    
    #validset
    physics_validset = create_subset(valid_indices, physics)
    psc_validset = create_subset(valid_indices, psc)
    chl_validset = create_subset(valid_indices, chl)
    chl_anom_validset = create_subset(valid_indices, chl_anom)

          

    # a réfléchir
    physics_testset=physics_tot
    psc_testset=np.concatenate((np.zeros(psc.shape)[:230],psc),axis=0)
    chl_testset=np.concatenate((np.zeros(chl.shape)[:230],chl),axis=0)   
    chl_anom_testset=np.concatenate((np.zeros(chl.shape)[:230],chl_anom),axis=0)   

    
    ################################# scaler #################################


    if dataloader_config['norm_mode']: 
        
        if not(dataloader_config['norm_mode'] in ('min_max', 'standard')):
            raise ValueError("\"{}\" is not a valid mode for "
                             "splitting. Only \"min_max\" and \"standard\" are allowed.".format(dataloader_config['norm_mode']))   
        if dataloader_config['norm_mode']=='standard':
            scaler_phy=StandardScaler()
            scaler_chl=StandardScaler()
            scaler_chl_anom=StandardScaler()

        elif dataloader_config['norm_mode']=='min_max':
            scaler_phy=MinMaxScaler()
            scaler_chl=MinMaxScaler()
            scaler_chl_anom=MinMaxScaler()


    
        #on swap les axes de manière à ce que le channel soit la dernière dimension puis on reswap dans le bon sens
        #un jour il faudra probablement réécrire le code avec les channels de features en dernier quoiqu'il arrive
        physics_trainset=np.swapaxes(scaler_phy.fit_transform(np.swapaxes(physics_trainset, 1, -1)),-1,1)            
        physics_validset=np.swapaxes(scaler_phy.transform(np.swapaxes(physics_validset, 1, -1)),-1,1)            
        physics_testset=np.swapaxes(scaler_phy.transform(np.swapaxes(physics_testset, 1, -1)),-1,1)            

        chl_trainset=np.squeeze(scaler_chl.fit_transform(np.expand_dims(chl_trainset, -1)))            
        chl_validset=np.squeeze(scaler_chl.transform(np.expand_dims(chl_validset, -1)))          
        #chl_testset=np.squeeze(scaler_chl.transform(np.expand_dims(chl_testset, -1)))      #en vrai cette étape du calcul est pas nécessaire   
        
        chl_anom_trainset=np.squeeze(scaler_chl_anom.fit_transform(np.expand_dims(chl_anom_trainset, -1)))            
        chl_anom_validset=np.squeeze(scaler_chl_anom.transform(np.expand_dims(chl_anom_validset, -1)))          
        #chl_anom_testset=np.squeeze(scaler_chl_anom.transform(np.expand_dims(chl_anom_testset, -1)))   #en vrai cette étape du calcul est pas nécessaire  



    training_set=Dataset_psc_chl(np.float16(physics_trainset),np.float16(chl_trainset),np.float16(psc_trainset), np.float16(chl_anom_trainset),
                                 transform=dataloader_config['transform'], 
                                 completion=dataloader_config['completion'])

    validation_set=Dataset_psc_chl(np.float16(physics_validset),np.float16(chl_validset),np.float16(psc_validset),np.float16(chl_anom_validset),
                                 transform=dataloader_config['transform'], 
                                 completion=dataloader_config['completion'])

    test_set=Dataset_psc_chl(np.float16(physics_testset),np.float16(chl_testset),np.float16(psc_testset), np.float16(chl_anom_testset),
                                 transform=dataloader_config['transform'], 
                                 completion=dataloader_config['completion'])

    
    training_generator   = DataLoader(training_set,
                                      batch_size=dataloader_config['batch_size'],
                                      shuffle=True)
    
    validation_generator = DataLoader(validation_set,
                                      batch_size=dataloader_config['batch_size'],
                                      shuffle=True)
    
    test_generator = DataLoader(test_set, batch_size=1, shuffle=False)

    return training_generator, validation_generator, test_generator, (scaler_chl, scaler_chl_anom)
    


if __name__=='__main__': 
    

    path="/home/luther/Documents/npy_data/"
    
    dataloader_config={'dataset_path_inputs':path+"physics/processed_physics_1998_2023_8d_lat50_100.npy",
                       'dataset_path_psc':path+"PSC/1998_2023_psc4_8d_100km_lat50.npy",
                       'dataset_path_chl':path+"chl/chl_avw_lat50_100km_8d_1998_2023.npy",
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
                                         'SCTR':(80,-3)},
                       'variables_physiques':[],
                       'phy_anomalies':[0,1], #[0,1,2,3...] list of which physical fields anomalies should be added
                       'chl_anomalies':True, #not set up yet
                       } 
    
    train_g, valid_g, test_g, scaler=get_dataloaders(dataloader_config)
    

            
        
        
        
        # dataloader_config={'dataset_path_inputs':path+"physics/processed_physics_1998_2023_8d_lat50_100.npy",
        #            'dataset_path_psc':path+"PSC/1998_2023_psc4_8d_100km_lat50.npy",
        #            'dataset_path_chl':path+"chl/chl_avw_lat50_100km_8d_1998_2023.npy",
        #            'hplc_path':path+"insitu/hplc_merged_new_glob_100km.csv",
        #            'transform':None,
        #            'batch_size': 32,
        #            'norm_chl':True,
        #            'norm_mode':'standard',
        #            'log_chl':True,
        #            'anomalies':0, #0 or False or None for classic behavior, 1 for chl anomalies, 2 for physics anomalies 
        #            'completion':'zeros',
        #            'stations_coord':{'SWAtl':(-55,-43),# stations withdrawed of the dataset for control
        #                              'NAtl':(-37,36),
        #                              'NPac':(156,24),
        #                              'SIO':(60,-32),
        #                              'SCTR':(80,-3)},
        #            'variables_physiques':[0,1,2,3,4,5,6,7]} 
        
    
    
    
    

