#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 11:38:56 2022

@author: lollier
"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         Functions                                     | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import numpy as np
import json
import os 
from torchvision.transforms.functional import crop
import torch
import subprocess

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   UTILS FUNCTIONS TO PROCESS DATA, SAVE AND LOAD RESULT               | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

    
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


class saving_tool():
    
    def __init__(self, checkpoint_path, net_name):
        
        self.checkpoint_path=checkpoint_path
        self.net_name=net_name
        self.result_dir=os.path.join(checkpoint_path,net_name)
        
        self.init_result_directory()
        
    def init_result_directory(self):
        
        if not(self.net_name in os.listdir(self.checkpoint_path)):
            subprocess.run([f"mkdir {self.result_dir}"])
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
                    last_improved_epoch):

        with open(os.path.join(self.checkpoint_path,"runs.txt"), "a+") as file :
            file.write('\n\n'+80*'_')
            file.write('\nNAME .....................: {}'.format(self.net_name)
            file.write('\nModel ....................: {}'.format(model))
            file.write('\nTest Loss ................: {}'.format(loss))
            file.write('\nEpoch ...................: {}'.format(epoch))
            file.write('\nBest valid loss Epoch.....: {}'.format(last_improved_epoch))

        
#############################Run Jupyter Notebook######################################
# ! python
# coding: utf-8

import os
import glob

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError

# Parse args
parser = argparse.ArgumentParser(description="Runs a set of Jupyter \
                                              notebooks.")
file_text = """ Notebook file(s) to be run, e.g. '*.ipynb' (default),
'my_nb1.ipynb', 'my_nb1.ipynb my_nb2.ipynb', 'my_dir/*.ipynb'
"""
parser.add_argument('file_list', metavar='F', type=str, nargs='*', 
    help=file_text)
parser.add_argument('-t', '--timeout', help='Length of time (in secs) a cell \
    can run before raising TimeoutError (default 600).', default=600, 
    required=False)
parser.add_argument('-p', '--run-path', help='The path the notebook will be \
    run from (default pwd).', default='.', required=False)
args = parser.parse_args()
print('Args:', args)
if not args.file_list: # Default file_list
    args.file_list = glob.glob('*.ipynb')

# Check list of notebooks
notebooks = []
print('Notebooks to run:')
for f in args.file_list:
    # Find notebooks but not notebooks previously output from this script
    if f.endswith('.ipynb') and not f.endswith('_out.ipynb'):
        print(f[:-6])
        notebooks.append(f[:-6]) # Want the filename without '.ipynb'

# Execute notebooks and output
num_notebooks = len(notebooks)
print('*****')
for i, n in enumerate(notebooks):
    n_out = n + '_out'
    with open(n + '.ipynb') as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=int(args.timeout), kernel_name='python3')
        try:
            print('Running', n, ':', i, '/', num_notebooks)
            out = ep.preprocess(nb, {'metadata': {'path': args.run_path}})
        except CellExecutionError:
            out = None
            msg = 'Error executing the notebook "%s".\n' % n
            msg += 'See notebook "%s" for the traceback.' % n_out
            print(msg)
        except TimeoutError:
            msg = 'Timeout executing the notebook "%s".\n' % n
            print(msg)
        finally:
            # Write output file
            with open(n_out + '.ipynb', mode='wt') as f:
                nbformat.write(nb, f)
                
#####################################TIMER######################################

from time import time

def timed(function):
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

#########################################Scaler######################################

#c'est clean sauf le clipping
class scaler():
    """
    scaler for datasets, can apply 3 differents scalers clipping, min_max and z-score(standard)
    TO DO :
        - clipping is not implemented yet in the norm denorm module
    """
    
    def __init__(self, mode='min_max'):
        """
        Parameters
        ----------
        mode : str, optional
            min_max, standard or clipping. The default is 'min_max'.
            if clipping, you need to use directly the implemented function
            axis handling are on your expense also
        """
        
        if mode in ('min_max', 'standard', 'clipping'):
            self.mode = mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "splitting. Only \"min_max\" , \"standard\"and "
                             "\"clipping\" are allowed.".format(mode))   
        
        if mode=='min_max':
            self.max_list=[]
            self.min_list=[]
            self.normalizer=self.min_max
            self.denormalizer=self.de_min_max
            
        elif mode=='standard':
            self.mean_list=[]
            self.std_list=[]
            self.normalizer=self.standard
            self.denormalizer=self.de_standard
            
        else :
            print("use directly the intern clipping, de_clipping function "
                  "parameters are respectively (data, v_min, v_max) and data")

        
    def norm(self, data, axis=None):
        """
        Parameters
        ----------
        data : numpy array
        axis : int, optional
            axis where normalization is applied. The default is None.

        Returns
        -------
        res : numpy array
            normalized array.
        """
        
        res=np.empty(data.shape)

        if isinstance(axis, int):
            
            shape=data.shape


            for index in range(data.shape[axis]):
                slices=[]
                for i in range(len(shape)):
                    if i !=axis : 
                        slices.append(slice(0,shape[i],1))
                    else :
                        slices.append(slice(index,index+1,1))
                res[tuple(slices)]=self.normalizer(data[tuple(slices)])
                
        else :
            res=self.normalizer(data)
        
        return res
    
    def denorm(self, data, axis=None):
        """
        Parameters
        ----------
        data : numpy array
        axis : int, optional
            axis where denormalization is applied. The default is None.

        Returns
        -------
        res : numpy array
            denormalized array. 
        You need to use the same instance of scaler for normalization and denormalization
        """        
        res=np.empty(data.shape)

        if isinstance(axis, int):
            
            shape=data.shape


            for index in range(data.shape[axis]):
                slices=[]
                for i in range(len(shape)):
                    if i !=axis : 
                        slices.append(slice(0,shape[i],1))
                    else :
                        slices.append(slice(index,index+1,1))
                res[tuple(slices)]=self.denormalizer(data[tuple(slices)], index)
                
        else :
            res=self.denormalizer(data,0)
        
        return res

    
    def min_max(self, data):
        
        self.maximum=np.nanmax(data)
        self.minimum=np.nanmin(data)
        self.max_list.append(self.maximum)
        self.min_list.append(self.minimum)
        ecart=self.maximum-self.minimum
        data=(data-self.minimum)/ecart
        
        return data.astype('float32')
    
    def de_min_max(self, data, i):
        return (data*(self.max_list[i]-self.min_list[i]))+self.min_list[i]
    
    def standard(self, data):
        
        self.mean=np.nanmean(data)
        self.std=np.nanstd(data)
        self.mean_list.append(self.mean)
        self.std_list.append(self.std)
        data=(data-self.mean)/self.std
        
        return data.astype('float32')
    
    def de_standard(self, data, i):
        return (data*self.std_list[i])+self.mean_list[i]
        
    def clipping(self, data, v_min, v_max):
        
        self.low=np.where(data<v_min, data-v_min, 0)
        self.high=np.where(data>v_max, data-v_max, 0)
        
        data=np.where(data<v_min, v_min, data)
        data=np.where(data>v_max, v_max, data)
        
        return data.astype('float32')
    
    def de_clipping(self, data):
        return data+self.low+self.high
        
##########################Train_test_splitter####################################################
class train_test_split():
    """
    class for dataset splitting (either by index or year)
    """
    def __init__(self, mode='index', split_index={'train':(0,828),'valid':(828,920),'test':(920,1012)}):
        """
        Parameters
        ----------
        mode : str, optional
            index or year according to your split_index arg, 
            if 'year' you need to provide the corresponding split_index. The default is 'index'.
        split_index : dict, optional
            dictionnary with the splitting indices for train, test and valid. 
            The default is {'train':(0,828),'valid':(828,920),'test':(920,1012)}.
        """
        
        if mode in ('year', 'index'):
            self.mode = mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "splitting. Only \"year\" and "
                             "\"index\" are allowed.".format(mode))
        self.mode=mode
        
        if self.mode=='year':
            if split_index is self.__init__.__defaults__[1]: #https://stackoverflow.com/questions/14749328/how-to-check-whether-optional-function-parameter-is-set
                raise ValueError("you need to provide a new split_index dictionnary "
                                 "with mode 'year'")
            
            self.split_index={}
            for key in split_index.keys():
                self.split_index[key]=((split_index[key][0]-1998)*46,(split_index[key][1]-1997)*46) #1997 because we take the end of the year as closing index
        else:
            self.split_index=split_index #care because dict are non mutuable object 

    def split(self, dataset):
        """
        Parameters
        ----------
        dataset : indexable object with the splitting dim in first

        Returns
        -------
        splitted_dataset : dict
            dict with splitted dataset.
        """
        
        splitted_dataset={}
        
        for key in self.split_index.keys():
            
            splitted_dataset[key]=dataset[slice(*self.split_index[key])]
            
        return splitted_dataset
    

    
# if __name__=="__main__":
    # path='/usr/home/lollier/datatmp/weight_save/Unet'
    # filename='test'
    # train_config = {
    #     'nb_epochs' : 1000, # arbitrary high
    #     'checkpoints_path': '/usr/home/lollier/datatmp/weight_save/Unet', 
    #     'verbose': True,
    #     'checkpoint':50, #save the weights every 50 epochs
    #     'name':'transfo_field',
    #     'patience_early_stopping':50, #needs to be > patience of lr scheduler
    #     'delta_early_stopping':0.00001
    # }
    
    # save_dict(train_config, path, filename)