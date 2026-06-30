#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 10:35:23 2022

@author: lollier

Dataloader that loads physics as inputs (regardless the number of channels) and loads chl and psc as outputs
chl is log10 and normalized 

"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         DATALOADERS                                   | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, RandomSampler
import torch
from utils.scaler import scaler
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #


class Dataset_psc_chl(Dataset):

    def __init__(self, physics, chl, psc, transform=None, completion='inf value'):

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
        psc=torch.Tensor(self.dataset_psc[index])
        
        if self.transform is not None:
            
            inputs=self.transform(inputs)   
            psc=self.transform(psc)
            chl=self.transform(chl)
                
            
        return inputs, (psc, torch.unsqueeze(chl,0))
 

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
    psc=np.load(dataloader_config['dataset_path_psc'])
    chl=np.load(dataloader_config['dataset_path_chl'])
    
    
    #withdraw test stations from dataset
    stations_array_idx={}
    for station in dataloader_config['stations_coord'].keys():
        stations_array_idx[station]=coord_to_index(*dataloader_config['stations_coord'][station])
        psc[:,:,stations_array_idx[station][0],stations_array_idx[station][1]]=np.nan
        chl[:,stations_array_idx[station][0],stations_array_idx[station][1]]=np.nan

    
    if dataloader_config['log_chl']:
        chl=np.log10(chl)
            
    #################################Train_valid_test_splitter#################################

    #by year
    test_index=slice(-46*2,-1) #on prend les 2 dernières années pour le test
    valid_index=[slice(i,i+46) for i in range(46*4,physics.shape[0]-46*2,46*5)]
    train_index=[slice(i,i+(46*4)) for i in range(0,physics.shape[0]-46*2,46*5)]
    
    #trainset
    physics_trainset=np.concatenate([physics[index] for index in train_index])
    psc_trainset=np.concatenate([psc[index] for index in train_index])
    chl_trainset=np.concatenate([chl[index] for index in train_index])

    
    #adding random pixel mask in validset from trainset
    #ça doit allonger un peu la phase de validation
    index_to_mask=np.argwhere(~np.isnan(psc_trainset[:,0])) #O parce qu'on regarde pour les Micro et on retire pour les 3
    np.random.shuffle(index_to_mask)
    psc_masked_validset=np.copy(psc_trainset)
    chl_masked_validset=np.copy(chl_trainset)

#bon avec un break ya probablement moyen de s'en sortir mieux
    for i in range(len(index_to_mask)):
        if i/len(index_to_mask)<0.05: #as we already have year, i only mask 5% pixels
            psc_trainset[index_to_mask[i][0],:,index_to_mask[i][1],index_to_mask[i][2]]=np.nan
            chl_trainset[tuple(index_to_mask[i])]=np.nan
        else : 
            psc_masked_validset[index_to_mask[i][0],:,index_to_mask[i][1],index_to_mask[i][2]]=np.nan
            chl_masked_validset[tuple(index_to_mask[i])]=np.nan
              
    #validset
    physics_validset=np.concatenate([physics[index] for index in valid_index])
    psc_validset=np.concatenate([psc[index] for index in valid_index]+[psc_masked_validset])
    chl_validset=np.concatenate([chl[index] for index in valid_index]+[chl_masked_validset])            
            
    physics_testset=physics[test_index]
    psc_testset=psc[test_index]
    chl_testset=chl[test_index]
    
    ################################# scaler #################################

    sc_physics={}
    sc_chl={}
        
    if dataloader_config['norm_mode']: 
        
            
        sc_physics['train']=scaler(mode=dataloader_config['norm_mode'])
        sc_chl['train']=scaler(mode=dataloader_config['norm_mode'])

        sc_physics['valid']=scaler(mode=dataloader_config['norm_mode'])
        sc_chl['valid']=scaler(mode=dataloader_config['norm_mode'])
        
        sc_physics['test']=scaler(mode=dataloader_config['norm_mode'])
        sc_chl['test']=scaler(mode=dataloader_config['norm_mode'])
        
        
        physics_trainset=sc_physics['train'].norm(physics_trainset, axis=1)
        physics_validset=sc_physics['valid'].norm(physics_validset, axis=1)
        physics_testset=sc_physics['test'].norm(physics_testset, axis=1)

        chl_trainset=sc_chl['train'].norm(chl_trainset)
        chl_validset=sc_chl['valid'].norm(chl_validset)
        chl_testset=sc_chl['test'].norm(chl_testset)
                        
        #norm
        #chl[key]=sc_chl[key].norm(chl[key], axis=1)
            
                
            # if not(dataloader_config['psc_%']):
            #     #chequer cette multplication
            #     psc[key]=psc[key]*chl[key]


    training_set=Dataset_psc_chl(physics_trainset,chl_trainset,psc_trainset, 
                                 transform=dataloader_config['transform'], 
                                 completion=dataloader_config['completion'])
    
    validation_set=Dataset_psc_chl(physics_validset,chl_validset,psc_validset, 
                                 transform=dataloader_config['transform'], 
                                 completion=dataloader_config['completion'])

    
    test_set=Dataset_psc_chl(physics_testset,chl_testset,psc_testset, 
                                 transform=dataloader_config['transform'], 
                                 completion=dataloader_config['completion'])
    
        
    training_generator   = DataLoader(training_set,
                                      batch_size=dataloader_config['batch_size'],
                                      shuffle=True)
    
    validation_generator = DataLoader(validation_set,
                                      batch_size=dataloader_config['batch_size'],
                                      shuffle=True)
    
    test_generator = DataLoader(test_set, batch_size=1, shuffle=False)
        
    return training_generator, validation_generator, test_generator, (sc_physics, sc_chl)
    


if __name__=='__main__': 
    
    path="/home/luther/Documents/npy_data/"
    
    dataloader_config={'dataset_path_inputs':path+"physics/processed_physics_1998_2020_8d_100_lat50.npy",
                       'dataset_path_psc':path+"PSC/PSC_Roy_8d_1998_2020_lat50_100km.npy",
                       'dataset_path_chl':path+"chl/chl_avw_1998_2020_100k_8d_lat50.npy",
                       'transform':None,
                       'batch_size': 4,
                       'norm_mode':'standard',
                       'log_chl':True,
                       'completion':'zeros',
                       'stations_coord':{'SWAtl':(-55,-43),# stations withdrawed of the dataset for control
                                         'NAtl':(-37,36),
                                         'NPac':(156,24),
                                         'SIO':(60,-32),
                                         'SCTR':(80,-3)}}
    
    train_g, valid_g, test_g, scalers=get_dataloaders(dataloader_config)
    

            
        
        
        
        
        
    
    
    
    