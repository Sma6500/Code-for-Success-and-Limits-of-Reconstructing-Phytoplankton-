#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 09:25:30 2025

@author: luther
"""

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from scipy.stats import linregress
from utils.functions import extend_nan_both_dimensions
from utils.plot import imshow_area


path_phy="/home/luther/Documents/npy_data/physics/processed_physics_1998_2024_monthly_lat50_100.npy"
path_chl="/home/luther/Documents/npy_data/chl/monthly_avw/chl_avw_m_glob_lat50_1998_2023.nc"

sst = np.load(path_phy)[:,1]
chl = xr.open_dataset(path_chl)['CHL1_mean'].values 

mask_coast = np.where(np.isnan(extend_nan_both_dimensions(sst[0], steps=2)), np.nan, 1)

sst=np.nan_to_num(sst)*mask_coast
chl=np.nan_to_num(np.log10(np.clip(chl, 1e-5, 10)))*mask_coast

sst_train=sst[:5*46]
chl_train=chl[:5*46]

coef_map=np.zeros((100,360))
bias_map=np.zeros((100,360))

pred_chl=np.zeros_like(chl)

for lat in range(100):
    for lon in range(360):
        if not(np.isnan(mask_coast[lat,lon])):
            result = linregress(sst_train[:,lat,lon], chl_train[:,lat,lon])
            coef_map[lat,lon],bias_map[lat,lon]=result.slope,result.intercept

            pred_chl[:,lat,lon]=result.slope*sst[:-12,lat,lon]+result.intercept
            
rmse=np.sqrt(np.nanmean((10**chl-10**pred_chl)**2))