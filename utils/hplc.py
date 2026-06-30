#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:09:15 2024

@author: luther

renvoie les point hplc pour 100km de résolution 
ATTENTION A UPDATE SI JAMAIS ON VEUT 25KM
"""

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         HPLC process                                  | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import pandas as pd
import numpy as np
import xarray as xr
import datetime as dt


# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   UTILS FUNCTIONS TO PROCESS DATA, SAVE AND LOAD RESULT               | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #



def find_clos_pix(lat,lon):
    
    dist=np.inf
    new_lat,new_lon=0,0
    for lat_grid in bath['lat'].values:
        for lon_grid in bath['lon'].values:
            temp=np.sqrt((lon-lon_grid)**2+(lat-lat_grid)**2)
            if temp<dist:
                dist=temp
                new_lat=lat_grid
                new_lon=lon_grid
    return (new_lat,new_lon)

def find_clos_pixlat(row):
    
    lat,lon=row[0],row[1]

    return find_clos_pix(lat, lon)[0]

def find_clos_pixlon(row):
    
    lat,lon=row[0],row[1]

    return find_clos_pix(lat, lon)[1]


#8 janvier parce que dans nos prédictions on enlève la 1ere date
def find_clos_timestep(row,first_ts=dt.date(1998,1,8),extent=1056):
    
    
    time_list=[]
    ts=first_ts
    current_year=first_ts.year
    
    for i in range(1+extent//46):
        while ts.year==current_year:
            time_list.append(ts)
            ts+=dt.timedelta(days=8)
        current_year+=1
        ts=dt.date(current_year,1,1)
        
    if len(time_list)==extent+1:
        time_list.pop(-1)
    elif len(time_list)!=extent:
        raise ValueError('erreur dans la construction de liste de temps')
   
    # This function will return the datetime in items which is the closest to the date pivot.
    def nearest(items, pivot):
        return items.index(min(items, key=lambda x: abs(x - pivot)))

    m,d,y=row[2],row[3],row[4]
    date=dt.date(int(y),int(m),int(d))
    near_date=nearest(time_list, date)   
    
    if abs(time_list[near_date]-date)>dt.timedelta(days=8):
        near_date=np.nan
    
    return near_date


def coord_to_grid(lat, lon):
    
    #array needs to be 2d lat, lon ((100,360) for instance)
    axes_label=[-179.5,
                179.5,
                49.5,
                -49.5]   
    
    grid_lon, grid_lat = np.linspace(axes_label[0],axes_label[1],360),  np.linspace(axes_label[2],axes_label[3],100)

    idx_lon=np.where(grid_lon==lon)
    idx_lat=np.where(grid_lat==lat)
    
    if len(idx_lon[0])+len(idx_lat[0]) != 2:
        return (np.nan, np.nan)
        
    return (idx_lat[0][0], idx_lon[0][0])


def grid_idxlat(row):
    
    lat,lon=row[17],row[18]

    return coord_to_grid(lat, lon)[0]

def grid_idxlon(row):
    
    lat,lon=row[17],row[18]

    return coord_to_grid(lat, lon)[1]

import scipy
import sklearn 
from sklearn.linear_model import HuberRegressor, LinearRegression, RANSACRegressor, Lasso, TheilSenRegressor

import datetime as dt

from scipy.stats import linregress
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

def linear_reg(x,y, mode='affine', scale=False, sample_weight=None):

    x_m=0
    y_m=0
    x_std=1
    y_std=1
    
    
    if scale :    
        x_m = np.nanmean(x)
        y_m = np.nanmean(y)
        x_std=np.sqrt(np.nanvar(x))
        y_std=np.sqrt(np.nanvar(y))
    
    new_x=(x-x_m)/x_std
    new_y=(y-y_m)/y_std
 
    if mode=='OLS':
        model = OLS(new_y,sm.add_constant(new_x))
    
        results = model.fit()
    
        a=(y_std*results.params[1])/x_std
        b=y_m+y_std*(results.params[0]-((results.params[1]*x_m)/x_std))
        rsquared=results.rsquared
        
    elif mode=='Huber':
        
        model=HuberRegressor()
        results=model.fit(sm.add_constant(new_x),new_y,sample_weight=sample_weight)
        
        
        a=(y_std*results.coef_[1])/x_std
        b=y_m+y_std*(results.coef_[0]-((results.coef_[1]*x_m)/x_std))
        rsquared=results.score(sm.add_constant(new_x),new_y)
    
    elif mode=='Ransac':
        
        model=RANSACRegressor()#estimator=Lasso(alpha=0.1),min_samples=0.5,max_trials=500)
        results=model.fit(sm.add_constant(new_x),new_y,sample_weight=sample_weight)
        
        
        a=(y_std*results.estimator_.coef_[1])/x_std
        b=y_m+y_std*(results.estimator_.coef_[0]-((results.estimator_.coef_[1]*x_m)/x_std))
        rsquared=results.score(sm.add_constant(new_x),new_y)   
        
    elif mode=='Lasso':
        
        model=Lasso(alpha=0.1)
        results=model.fit(sm.add_constant(new_x),new_y,sample_weight=sample_weight)
        
        
        a=(y_std*results.coef_[1])/x_std
        b=y_m+y_std*(results.coef_[0]-((results.coef_[1]*x_m)/x_std))
        rsquared=results.score(sm.add_constant(new_x),new_y)           
    
    elif mode=='TSR':
        
        model=TheilSenRegressor()
        results=model.fit(sm.add_constant(new_x),new_y,sample_weight=sample_weight)
        
        
        a=(y_std*results.coef_[1])/x_std
        b=y_m+y_std*(results.coef_[0]-((results.coef_[1]*x_m)/x_std))
        rsquared=results.score(sm.add_constant(new_x),new_y)           
    
    elif mode=='affine':
        
        model=OLS(new_y,new_x.reshape(-1,1))
        results=model.fit()
        
        
        a=(y_std*results.params[0])/x_std
        b=0#y_m+y_std*(results.coef_[0]-((results.coef_[1]*x_m)/x_std))
        #rsquared=results.score(new_x.reshape(-1,1),new_y)      
        rsquared=results.rsquared
    
    return rsquared, a, b

if __name__ ==  "__main__":
    
    df=pd.read_excel('/home/luther/Documents/npy_data/insitu/Hourany/Assets_HPLC_SOMChlF.xlsx',sheet_name='HPLC DPA-based Chla PG')
    df['Micro']=df['Chl_Dino']+df['Chla_Diat']
    df['Nano']=df['Chla_Hapto']+df['Chla_Crypto']+df['Chla_Pelago']+df['Chla_Green']
    df['Pico']=df['Chla_Prok']
    df['Chla']=df['Micro']+df['Nano']+df['Pico']
    
    
    #useful for getting lon lat grid on 100km
    bath=xr.open_dataset('/home/luther/Documents/npy_data/bath_100.nc')
    
    
    
    
    #df['grid_idx']=df.apply(find_clos_pix, axis=1, raw=True, result_type='expand')
    df['new_lat']=df.apply(find_clos_pixlat, axis=1, raw=True, result_type='expand')
    df['new_lon']=df.apply(find_clos_pixlon, axis=1, raw=True, result_type='expand')
        
    df['time_idx']=df.apply(find_clos_timestep, axis=1, raw=True, result_type='expand')
    
    df['grid_lat']=df.apply(grid_idxlat, axis=1, raw=True, result_type='expand')#, args=(,0))
    df['grid_lon']=df.apply(grid_idxlon, axis=1, raw=True, result_type='expand')#, args=(,1))
    
    
    
    lat=np.array(df['grid_lat'].values,dtype=np.int64)
    lon=np.array(df['grid_lon'].values,dtype=np.int64)
    ts=df['time_idx'].values
    Chl=np.array(df['Chla'].values)
    Micro=np.array(df['Micro'].values)
    Nano=np.array(df['Nano'].values)
    Pico=np.array(df['Pico'].values)
    
    
    df['count_pix_stp']=df.groupby(['time_idx','new_lat','new_lon']).transform('size')
    df_1=df.groupby(['time_idx','new_lat','new_lon']).mean()
    
    hplc_points=[]
    
    for t,la,lo, chl, micro, nano, pico in np.stack((ts,lat,lon, Chl, Micro, Nano, Pico),axis=1):
        if not(np.isnan(la+lo+t) or abs(la)>100 or abs(lo)>360):
            hplc_points.append((int(t),int(la),int(lo),chl,micro,nano,pico))
    
