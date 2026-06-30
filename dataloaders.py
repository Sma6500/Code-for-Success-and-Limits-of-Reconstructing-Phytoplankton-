#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:34:48 2025

@author: luther

"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         DATALOADERS                                   | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, RandomSampler
import torch
from utils.scaler import MinMaxScaler, StandardScaler
from utils.functions import extend_nan_both_dimensions
from utils.transform import CustomCrop,CustomMask

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #



class Dataset_psc_chl(Dataset):

    def __init__(self, physics, chl, psc, chl_anom=None, transform=None, completion='inf value', sigma=0.):
        super().__init__()

        if completion not in ('inf value', 'zeros'):
            raise ValueError(f"\"{completion}\" in dataset settings is not a valid completion set up. Only \"inf value\" and \"zeros\" are allowed.")

        self.completion = completion
        self.dataset_physics = physics
        self.dataset_chl = chl
        self.dataset_psc = psc
        self.dataset_chl_anom = chl_anom
        self.transform = transform
        
        self.sigma=sigma

    def __len__(self):
        return len(self.dataset_physics)
    
    def add_gaussian_noise(self, x):
        noise = np.random.normal(
            loc=0.0,
            scale=self.sigma,
            size=x.shape
        )   
        return x + noise
    
    def __getitem__(self, index):
        """
        Main function of the CustomDataset class.
        """
        inputs = self.dataset_physics[index]
        
        fill_value = -1000 if self.completion == 'inf value' else 0.
        inputs = np.nan_to_num(inputs, nan=fill_value)
        #inputs=self.add_gaussian_noise(inputs)
        
        # Convert to tensors
        inputs = torch.as_tensor(inputs, dtype=torch.float32)
        chl = torch.as_tensor(self.dataset_chl[index], dtype=torch.float32)
        psc = torch.as_tensor(self.dataset_psc[index], dtype=torch.float32)
        chl_anom = torch.as_tensor(self.dataset_chl_anom[index], dtype=torch.float32) if self.dataset_chl_anom is not None else None

        # Apply transform if defined
        if self.transform:
            inputs = self.transform(inputs)
            print("care, transform only applied on input")
            #chl = self.transform(chl)
            #psc = self.transform(psc)

            if chl_anom is not None:
                chl_anom = self.transform(chl_anom)

        # Return tuple
        if chl_anom is not None :
            return inputs, (psc, chl.unsqueeze(0), chl_anom.unsqueeze(0)) 
        else :
            return  inputs, (psc,chl.unsqueeze(0))
        #return inputs, (chl.unsqueeze(0), chl_anom.unsqueeze(0)) if chl_anom is not None else inputs, chl.unsqueeze(0))


def load_data(path, variables=None):
    """ Load dataset based on file type (.nc or .npy). """
    if path.endswith(".nc"):
        return xr.open_dataset(path)[variables].to_array(dim='variables').data if variables else xr.open_dataset(path)['CHL1_mean'].values
    elif path.endswith(".npy"):
        return np.load(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")
        
        
def coord_to_index(lon, lat):
    #retourne un tuple lat, lon index correspondant aux index associés au coordonnées dans le dataset
    idx_lon=lon+180
    idx_lat=50-lat    
    return (idx_lat, idx_lon)

def compute_anomalies(array, lenght_year=12):
    timestep=len(array)#we suppose that t is the first dim
    shape=array.shape
    weekly_mean=np.concatenate([np.nanmean(array.reshape(timestep//lenght_year,lenght_year,*shape[1:]),axis=0)]*(timestep//lenght_year))
    return array-weekly_mean

def get_dataloaders(dataloader_config):
    """
    Loads datasets, applies preprocessing, and returns PyTorch dataloaders.
    """

    print("Côtes masquées à 2 pixels")

    # Load physics data
    physics = load_data(dataloader_config['dataset_path_inputs'], ['mld', 'sst', 'sss', 'ssh', 'solar', 'norm_currents', 'vort_winds','ssr']) #ssr instead of solar take care
    print('ssr replace solar')
    physics = np.swapaxes(physics, 0, 1) if len(physics) < 10 else physics  # Swap axes only for .nc files, in nc files the variables are on the first axis instead of the second one

    # Apply coastal mask
    mask_coast = np.where(np.isnan(extend_nan_both_dimensions(physics[0, 0], steps=2)), np.nan, 1)
    physics *= mask_coast
    import copy
    physics_tot = copy.deepcopy(physics)#np.load(dataloader_config["dataset_path_inputs_test"])
    physics_tot *= mask_coast

    # Load chlorophyll data
    chl = load_data(dataloader_config['dataset_path_chl']) * mask_coast
    
    # Load PSC data
    print('care, psc global took in 40:-40')
    psc = load_data(dataloader_config['dataset_path_psc'],variables=['Micro','Nano','Pico'])[...,40:-40,:] * mask_coast
    psc = np.swapaxes(psc, 0, 1) if len(psc) < 10 else psc  # Swap axes only for .nc files, in nc files the variables are on the first axis instead of the second one
    psc=np.clip(psc, 0, 1)
    
    # Compute anomalies if enabled
    phy_anom, chl_anom = None, None
    if dataloader_config.get('phy_anomalies', False):
        phy_anom = compute_anomalies(physics,dataloader_config.get('lenght_year', 12))[:, dataloader_config['phy_anomalies']]
        physics_anom_tot = compute_anomalies(physics_tot,dataloader_config.get('lenght_year', 12))[:, dataloader_config['phy_anomalies']]

    if dataloader_config.get('chl_anomalies', False):
        chl_anom = np.sign(compute_anomalies(chl,dataloader_config.get('lenght_year', 12))) * (np.log10(np.clip(np.abs(compute_anomalies(chl,dataloader_config.get('lenght_year', 12))), 1e-5, 10)) + 5)

    # Select physical variables
    variables_physiques = dataloader_config.get('variables_physiques', slice(None))
    physics = physics[:, variables_physiques]
    physics_tot = physics_tot[:, variables_physiques]
    
    if dataloader_config.get('phy_anomalies', False):
        physics = np.concatenate([physics, phy_anom], axis=1)
        physics_tot=np.concatenate([physics_tot, physics_anom_tot], axis=1)
        
    # Remove test stations
    for station, coords in dataloader_config['stations_coord'].items():
        lat_idx, lon_idx = coord_to_index(*coords)
        chl[:, lat_idx, lon_idx] = np.nan
        psc[..., lat_idx, lon_idx]=np.nan
        if chl_anom is not None:
            chl_anom[:, lat_idx, lon_idx] = np.nan

    # Convert infinities to NaN
    physics = np.where(np.isinf(physics), np.nan, physics)
    physics_tot = np.where(np.isinf(physics_tot), np.nan, physics_tot)

    # Apply log transformation to chl
    if dataloader_config['log_chl']:
        print('chl log')
        chl = np.log10(np.clip(chl, 1e-5, 10))

    # Split dataset into train, validation, and test
    total_timesteps, timesteps_per_year = physics.shape[0], dataloader_config.get('lenght_year', 12)
    total_years = total_timesteps // timesteps_per_year

    train_idx, valid_idx, test_idx = [], [], []
    
    
    for i in range(total_years):
        indices = slice(i * timesteps_per_year, (i + 1) * timesteps_per_year)
        if i>=13 : #Bsplit i<=11:#A split i<=9:#Normal split i >= 13 :
            test_idx.append(indices)
        elif i<=4: # B split i>=24 or i<=2: #A split i>=22:#Normal split i <= 4:
            valid_idx.append(indices)
        else : #A split else : #Normal split else:
            train_idx.append(indices)

    def create_subset(slices, data):
        return np.concatenate([data[s] for s in slices], axis=0)

    # Create train/valid/test sets
    physics_trainset = create_subset(train_idx, physics)
    physics_validset = create_subset(valid_idx, physics)
    physics_testset = np.copy(physics_tot)

    chl_trainset = create_subset(train_idx, chl)
    chl_validset = create_subset(valid_idx, chl)
    chl_testset = np.zeros_like(physics_tot[:,0])
    
    psc_trainset = create_subset(train_idx, psc)
    psc_validset = create_subset(valid_idx, psc)
    psc_testset = np.zeros((physics_tot.shape[0],3,100,360))

    chl_anom_trainset = create_subset(train_idx, chl_anom) if chl_anom is not None else None
    chl_anom_validset = create_subset(valid_idx, chl_anom) if chl_anom is not None else None
    chl_anom_testset = np.zeros_like(physics_tot[:,0]) if chl_anom is not None else None

    # Normalize datasets
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

    physics_trainset = np.swapaxes(scaler_phy.fit_transform(np.swapaxes(physics_trainset, 1, -1)), -1, 1)
    physics_validset = np.swapaxes(scaler_phy.transform(np.swapaxes(physics_validset, 1, -1)), -1, 1)
    physics_testset = np.swapaxes(scaler_phy.transform(np.swapaxes(physics_testset, 1, -1)), -1, 1)

    chl_trainset = np.squeeze(scaler_chl.fit_transform(np.expand_dims(chl_trainset, -1)))
    chl_validset = np.squeeze(scaler_chl.transform(np.expand_dims(chl_validset, -1)))

    if chl_anom is not None:
        chl_anom_trainset = np.squeeze(scaler_chl_anom.fit_transform(np.expand_dims(chl_anom_trainset, -1)))
        chl_anom_validset = np.squeeze(scaler_chl_anom.transform(np.expand_dims(chl_anom_validset, -1)))
        
    # Create datasets
    training_set = Dataset_psc_chl(physics_trainset, chl_trainset, psc_trainset, chl_anom_trainset, transform=dataloader_config['transform'], completion=dataloader_config['completion'])
    validation_set = Dataset_psc_chl(physics_validset, chl_validset, psc_validset, chl_anom_validset, transform=dataloader_config['transform'], completion=dataloader_config['completion'])
    test_set = Dataset_psc_chl(physics_testset, chl_testset, psc_testset, chl_anom_testset, transform=dataloader_config['transform'], completion=dataloader_config['completion'])

    # Create dataloaders
    training_generator = DataLoader(training_set, batch_size=dataloader_config['batch_size'], shuffle=True)
    validation_generator = DataLoader(validation_set, batch_size=dataloader_config['batch_size'], shuffle=True)
    test_generator = DataLoader(test_set, batch_size=1, shuffle=False)

    if dataloader_config.get('chl_anomalies', False):
        return training_generator, validation_generator, test_generator, (scaler_chl, scaler_chl_anom)
    return training_generator, validation_generator, test_generator, scaler_chl

 

if __name__=='__main__': 
    
    path="/home/luther/Documents/npy_data/"
    #path="/datatmp/home/lollier/npy_data/"
    
    dataloader_config={'dataset_path_inputs':path+"physics/physics_completion/processed_1998_2023_reanalysis_100km_monthly_lat50_seaice.nc",
                       'dataset_path_inputs_test':path+"physics/processed_physics_1993_2023_monthly_lat50_100.npy",
                       'dataset_path_psc':path+"PSC/chl_psc_avw_roy_monthly_glob_100km_1998_2023_1.nc",
                       'dataset_path_chl':path+"chl/monthly_avw/chl_avw_m_glob_lat50_1998_2023.nc",
                       'hplc_path':None,#path+"insitu/hplc_merged_new_glob_100km.csv",
                       'transform':CustomMask(),
                       'batch_size': 1,
                       'norm_chl':True,
                       'norm_mode':'standard',
                       'log_chl':True,
                       'completion':'zeros',
                       'stations_coord':{'SWAtl':(-55,-43),# stations withdrawed of the dataset for control
                                         'NAtl':(-37,36),
                                         'NPac':(156,24),
                                         'SIO':(60,-32),
                                         'SCTR':(80,-3)},
                       'variables_physiques':[0,1,2],
                       'phy_anomalies':[0,1,2,3,4,5,6], #[0,1,2,3...] list of which physical fields anomalies should be added
                       'chl_anomalies':False, 
                       } 
        
    train_g, valid_g, test_g, scaler =get_dataloaders(dataloader_config)
    

    #a, b =get_dataloaders(dataloader_config)

        
        
        
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
        
    
    
    
    

