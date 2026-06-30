#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 11:38:56 2022

@author: lollier

- Timer

"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         Functions                                     | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import numpy as np
import xarray as xr
import netCDF4
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

#####################################TIMER######################################

from time import time

def timer(function):
    def wrapper(*args, **kwargs):
        start=time()
        output=function(*args, **kwargs)
        end=time()
        run_time=end-start
        print('\nFunction mesured...: '+ str(function))
        print('Time taken.........: %dh %dm %.2fs\n' % (run_time//3600, (run_time%3600)//60, run_time%60))
        if output is None:
            return run_time
        return output, run_time

    return wrapper
    
def load_config(path_config, transformation):
    """
    Parameters
    ----------
    path_config : str
        path config.
    transformation : torch transform
        transformation to apply to tensor in order to go through the network.

    Returns
    -------
    model_config : dict
        DESCRIPTION.
    dataloader_config : dict
        DESCRIPTION.
    train_config : dict
        DESCRIPTION.
    criterion_config : dict
        DESCRIPTION.
    scheduler_config : dict
        DESCRIPTION.
    optimizer_config : dict
        DESCRIPTION.

    """
    
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
            self.read_dataset_infos(self.dataloader_config['dataset_path_predictors'][:-3]+'txt')
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
             
# def load_testset(dataloader_config): 
    
#     predictors=np.load(dataloader_config['dataset_path_predictors'])
#     psc=np.load(dataloader_config['dataset_path_psc'])
#     chl=np.load(dataloader_config['dataset_path_chl'])
    

#     if not(dataloader_config['psc_%']):
#         psc=chl*psc
    
    
#     #on garde la chloro normale en test
#     # if dataloader_config['log_chl']:
#     #     chl=np.log10(chl)
        
#     split=train_test_split(mode=dataloader_config['split_mode'],split_index=dataloader_config['split_index'])
    
#     predictors=split.split(predictors)
#     psc=split.split(psc)
#     chl=split.split(chl)
    
#     if dataloader_config['withdraw_seasonality']:
#         for key in chl.keys():
#             chl_log=np.log10(chl[key])
#             seasonal_mean=np.expand_dims(np.repeat(np.nanmean(np.concatenate([chl_log[i::46] for i in range(46)],axis=1),axis=0),len(chl_log)//46,axis=0),axis=1)

#             #withdraw seasonality mean
#             chl[key]=10**(chl_log-seasonal_mean)
                    
#     return predictors['test'],psc['test'],chl['test']

class saving_tool():
    
    def __init__(self, checkpoint_path, net_name, architecture_name):
        
        self.checkpoint_path=checkpoint_path
        self.net_name=net_name
        self.result_dir=os.path.join(checkpoint_path,net_name)
        self.architecture_name=architecture_name
        
        self.init_result_directory()
        
    def init_result_directory(self):
        
        if not(self.net_name in os.listdir(self.checkpoint_path)):
            subprocess.run(["mkdir",f"{self.result_dir}"])
            subprocess.run(["cp","./dataloaders.py", f"{self.result_dir}/"])
            #subprocess.run(["cp",f"/home/luther/Documents/scripts_training/models/{self.architecture_name}.py", f"{self.result_dir}/"])

        else :
            raise ValueError(f"{self.net_name} already exist or has been runned before/"
                            "please change net name")
        '''
    w  write mode
    r  read mode
    a  append mode

    w+  create file if it doesn't exist and open it in write mode
    r+  open for reading and writing. Does not create file.
    a+  create file if it doesn't exist and open it in append mode
    '''


    def save_dict(self, dictionary, path, filename):
        """
        Parameters
        ----------
        dictionary : type : guess :p
            config dictionary.
        path : str
            save config path.
        filename : str
            name of config saving file.

        Returns
        -------
        None.

        """
    
        file = open(os.path.join(path, filename) +".json", "a+")
        json.dump(dictionary, file)
        file.close()

    def save_config(self,config_list, path, filename):
        """
        Parameters
        ----------
        config_list : list of dictionary
            all config dictionary.
        path : str
            save config path.
        filename : str
            name of config saving file.

        Returns
        -------
        None.

        """

        self.save_dict(json.dumps([config for config in config_list]), path, filename)
            
    def save_glob(self, loss, model, epoch,
                    last_improved_epoch,scheduler_info,
                    model_parameters, comment):
        """
        add the mains informations of training to a file 'runs.txt'
        """
        with open(os.path.join(self.checkpoint_path,"runs.txt"), "a+") as file :
            file.write('\n\n'+80*'_')
            file.write('\nNAME .....................: {}'.format(self.net_name))
            file.write('\nModel ....................: {}'.format(model))
            file.write('\nModel parameters..........: {}'.format(model_parameters))
            file.write('\noptim, scheduler \& starting lr....: {} {} {}'.format(*scheduler_info))
            file.write('\nTest Loss ................: {}'.format(loss))
            file.write('\nEpoch ....................: {}'.format(epoch))
            file.write('\nBest valid loss Epoch.....: {}\n'.format(last_improved_epoch))
            file.write('\nComment ..................: \n'+comment+'\n')
            
    def save_pred(self, chl_pred, psc_pred, path, mode='numpy'):
        if len(chl_pred)>0:
            np.save(path+'/chl_pred_glob.npy',chl_pred)
        if len(psc_pred)>0:
            np.save(path+'/psc_pred_glob.npy',psc_pred)


        
#############################Run Jupyter Notebook######################################


import os
import glob
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError


def process_results(glob_path, net_name):
    """
    Parameters
    ----------
    glob_path : str
        path to the folder containing runs.txt and notebook_model.ipynb.
    net_name : str
        name of the model in order to access to the result folder.
    """
    
    folder_path=os.path.join(glob_path,net_name)
    notebook='notebook_model'
    notebook_out=os.path.join(glob_path,net_name,net_name+'_out')
    with open(os.path.join(glob_path,notebook) + '.ipynb') as n:
        nb = nbformat.read(n, as_version=4)
        ep = ExecutePreprocessor(timeout=int(600), kernel_name='python3')
        try:
            print('Running :', notebook)
            out = ep.preprocess(nb, {'metadata': {'path': folder_path}})
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
  
                
import numpy as np
import pymannkendall as mk
from tqdm import tqdm
from scipy.sparse import coo_matrix
from joblib import Parallel, delayed


def compute_trend_single_point(data_slice, threshold):
    """
    Compute trend for a single (lat, lon) point, considering NaN handling and Mann-Kendall test.
    """
    if np.isnan(data_slice).sum() / data_slice.size >= threshold:
        return np.nan
    else:
        trend_test = mk.hamed_rao_modification_test(data_slice, alpha=0.05)
        return trend_test.slope if trend_test.h else np.nan

def compute_trend_iav(data, threshold=0.25, n_jobs=-1):
    """
    Perform Mann-Kendall modified test on each point of data.
    `data` needs to be a 3D array (time, lat, lon).
    `threshold` controls the number of NaNs tolerated in a time series.
    `n_jobs` sets the number of parallel jobs to run (-1 uses all available CPUs).
    """
    if len(data.shape) != 3:
        raise ValueError(f"Only 3D arrays are accepted, {len(data.shape)}D not valid")

    # Prepare to store the result
    lat_size, lon_size = data.shape[1], data.shape[2]
    trend = np.empty((lat_size, lon_size))

    # Parallelize over latitudes and longitudes
    trend_results = Parallel(n_jobs=n_jobs)(
        delayed(compute_trend_single_point)(data[:, lat, lon], threshold)
        for lat in range(lat_size) for lon in range(lon_size)
    )

    # Reshape the flat results back into (lat, lon) form
    trend = np.array(trend_results).reshape(lat_size, lon_size)

    return trend


def onehot_sparse(b):
    'one hot encoding of matrix a'
    a=np.array(np.nan_to_num(b),dtype=int)
    N = a.size
    L = a.max()+1
    data = np.ones(N,dtype=int)
    return coo_matrix((data,(np.arange(N),a.ravel())), shape=(N,L)).toarray().reshape(*b.shape,L)


def extend_nan_both_dimensions(arr, steps=2):
    # Create a copy of the original array
    extended_arr = np.copy(arr)
    
    # Get the shape of the array
    rows, cols = arr.shape
    
    # Iterate through the array to find NaNs and extend them
    for i in range(rows):
        for j in range(cols):
            if np.isnan(arr[i, j]):
                # Extend NaN to the right/left
                for k in range(1, steps + 1):
                    if j + k < cols:
                        extended_arr[i, j + k] = np.nan
                    if j-k>=0:
                        extended_arr[i, j - k] = np.nan

                # Extend NaN downward/upward
                for k in range(1, steps + 1):
                    if i + k < rows:
                        extended_arr[i + k, j] = np.nan
                    if i-k>=0:
                        extended_arr[i - k, j] = np.nan


    return extended_arr
    
if __name__=="__main__":
    
    path="/datatmp/home/lollier/npy_emergence/"
    
    dataloader_config={'dataset_path_predictors':path+"dyn.npy",
                       'dataset_path_psc':path+"psc.npy",
                       'dataset_path_chl':path+"chl.npy",
                       'transform':None,
                       'split_mode':'year',
                       'split_index':{'train':(1998,2017),'valid':(1998,2017),'test':(2018,2019)},
                       'random_mask':True,
                       'batch_size': 4,
                       'norm_mode':'standard',
                       'psc_%':True,
                       'log_chl':True,
                       'concat':False,
                       'timestep':False,
                       'withdraw_seasonality':False,
                       'completion':False,
                       'stations_coord':{'SWAtl':(-55,-43),# stations withdrawed of the dataset for control
                                         'NAtl':(-37,36),
                                         'NPac':(156,24),
                                         'SIO':(60,-32),
                                         'SCTR':(80,-3)}}
    
    predictors=np.load(dataloader_config['dataset_path_predictors'])
    psc=np.load(dataloader_config['dataset_path_psc'])
    chl=np.load(dataloader_config['dataset_path_chl'])
    
    
  
    
    
    
