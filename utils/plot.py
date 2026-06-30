#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:34:22 2023

@author: lollier
"""

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         Plot                                          | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from matplotlib.cm import ScalarMappable

import numpy as np
import os
import xarray as xr
import cmocean.cm as cm
import torch

import cartopy
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import pymannkendall as mk
import matplotlib.pyplot as plt
import scipy as sc
from tqdm import tqdm
from scipy import signal
from matplotlib import cm as mcm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.sparse import coo_matrix
import scipy as sc
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                           PLOT FUNCTIONS TO SHOW DATA AND RESULT                      | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #


"""
TO DO
--> ajouter un affichage en log, ie un param qui prend le log de la carte d'entrée mais sans changer la colorbar
--> pour les PSC gérer un triple affichage ?
         
"""

def check_shape(data):
    """
    Parameters
    ----------
    tensor : tensor or array
        check the size and the format of inputs and return the axes for plot.

    Returns
    -------
    tensor : TYPE
        DESCRIPTION.
    axes_label : TYPE
        DESCRIPTION.

    """
    if type(data) not in [np.ndarray, np.array, torch.Tensor]:
        raise ValueError('type not valid')
    
    if type(data)==torch.Tensor:
        data=data.numpy()
        
    if data.shape==(120,360):
        axes_label=[-179.5,
                    179.5,
                    59.5,
                    -59.5]   
        return data, axes_label
    
    if data.shape==(100,360):
        axes_label=[-179.5,
                    179.5,
                    49.5,
                    -49.5]   
        return data, axes_label
    
    if data.shape==(180,360):
        axes_label=[-179.5,
                    179.5,
                    89.5,
                    -89.5]   
        return data, axes_label

        
    if data.shape==(4320,8640):
        axes_label=[-179.97,
                    179.97,
                    89.97,
                    -89.97]
        return data, axes_label
    
    if data.shape==(481,1440):
        axes_label=[0,
                   360,
                   59.97,
                   -59.97]
        return data, axes_label
    
    if data.shape==(720,1440):
        axes_label=[-179.97,
                    179.97,
                    89.97,
                    -89.97]
        return data, axes_label
    
    
    else :
        raise ValueError('shape not valid')
        


def imshow_area(array, cmap='jet', fig=None, ax=None, 
                vmin=None, vmax=None, log=False, title=False,
                colorbar=True, symlog=False, save=None,
                contour=None):

    if (fig is None and not(ax is None)) or (ax is None and not(fig is None)):
        raise ValueError("You need to specify both ax and fig params")
        
    array, axes_label = check_shape(array)
    
    lon = np.linspace(axes_label[0], axes_label[1], array.shape[1])
    lat = np.linspace(axes_label[2], axes_label[3], array.shape[0])
    proj = ccrs.Robinson(central_longitude=200)

    if fig is None and ax is None:
        fig = plt.figure(figsize=(20, 15), dpi=200) 
        ax = plt.subplot(1, 1, 1, projection=proj)
        show = True
    else:
        fig, ax = fig, ax
        show = False
        
    if log:
        array = np.log10(array)
        if vmin is None and vmax is None:
            vmax, vmin = np.nanmax(array), np.nanmin(array)
        cb = LogNorm(10**vmin, 10**vmax)
        
    elif symlog:
        if vmin is None and vmax is None:
            vmax, vmin = np.nanmax(array), np.nanmin(array)
        cb = SymLogNorm(0.001, vmin=vmin, vmax=vmax)

    img = ax.pcolormesh(lon, lat, 
                        np.roll(array, int(array.shape[1] / 2.25), axis=1), 
                        transform=ccrs.PlateCarree(central_longitude=200), 
                        cmap=cmap, vmin=vmin, vmax=vmax)
        
    ax.coastlines(alpha=0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(),
                      draw_labels=True,
                      linewidth=0.5, 
                      color='gray', 
                      alpha=0.2, 
                      linestyle='solid')
    gl.xlabels_top=False
    
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='grey', zorder=1)  
    ax.set_extent(axes_label, crs=ccrs.PlateCarree(central_longitude=200))
    
    if colorbar:
        if isinstance(colorbar, str):
            colorbar_label = colorbar
        else:
            colorbar_label = None
        if log:
            fig.colorbar(ScalarMappable(cb, cmap=cmap), ax=ax, 
                         orientation='vertical', shrink=0.4, label=colorbar_label)
        elif symlog:
            fig.colorbar(ScalarMappable(cb, cmap=cmap), ax=ax, 
                         orientation='vertical', shrink=0.4, label=colorbar_label)
        else:
            if vmin is None and vmax is None:
                vmax, vmin = np.nanmax(array), np.nanmin(array)
            cb = Normalize(vmin, vmax)
            fig.colorbar(ScalarMappable(cb, cmap=cmap), ax=ax, 
                         orientation='vertical', shrink=0.4, label=colorbar_label)
    
    if contour:
        # Extract contour levels and additional parameters from the contour argument
        levels = contour.get('levels', None)
        colors = contour.get('colors', 'k')
        linewidths = contour.get('linewidths', 1)
        linestyles = contour.get('linestyles', 'solid')
        contour_array=contour.get('contour_data',array)

        # Add contour lines
        cs = ax.contour(lon, lat, 
                        np.roll(contour_array, int(contour_array.shape[1] / 2.25), axis=1), 
                        levels=levels, colors=colors, linewidths=linewidths, 
                        linestyles=linestyles, transform=ccrs.PlateCarree(central_longitude=200))
        ax.clabel(cs, inline=True, fontsize=8, fmt='%1.1f')  # Add labels to contours
    
    ax.set(xlabel='Longitude', ylabel='Latitude')
    
    if title:
        ax.set_title(title)
    if save:
        plt.savefig(save,bbox_inches='tight')
    if show:
        plt.show()




###################################################################"
"""
Period Plotting
"""
###################################################################"

import pandas as pd

def plot_mei_index(train_period, validation_period, test_period, data=None):
    """
    Plots the MEI Index over time with shaded regions for training, validation, and test periods.

    Parameters:
    - data: pd.DataFrame with a datetime index and a single column for the MEI Index values.
    - train_period: tuple of (start_date, end_date) for the training period as strings in 'YYYY-MM-DD' format.
    - validation_period: tuple of (start_date, end_date) for the validation period as strings in 'YYYY-MM-DD' format.
    - test_period: tuple of (start_date, end_date) for the test period as strings in 'YYYY-MM-DD' format.
    """
    # Plot the MEI index data
    fig = plt.figure(figsize=(12, 4), dpi=200)#, frameon=False)
    ax = plt.subplot(111)
    if data is not(None):
        ax.plot(data.index, data.iloc[:, 0], color="brown", alpha=0.6, label="MEI Index")
        # Formatting the plot
        ax.set_ylabel("MEI Index")
        ax.ylim(-3, 3)
        ax.set_xlim(data.index.min(), data.index.max())

    else :
        dates = pd.date_range(start="1993-01-01", end="2023-12-31", freq="M")
        ax.plot(dates,np.concatenate((np.zeros(12*5)*np.nan,1+np.zeros(dates.shape[0]-12*5))),alpha=0.75,color="green",label="Chlorophylle Satellite spanning period")
        ax.plot(dates,np.zeros(dates.shape),alpha=0.75,color="blue",label="Glorysv12 spanning period")
        ax.set_ylabel("Training, validation and test periods")
        ax.get_yaxis().set_visible(False)
        #ax.set_xlim(dates[0], dates[-1])

    # Shade the validation, train, and test periods
    ax.axvspan(*validation_period, color="orange", alpha=0.3, label="Validation")
    ax.axvspan(*train_period, color="brown", alpha=0.2, label="Train")
    
    if isinstance(test_period[0],tuple):
        for per in test_period:
            ax.axvspan(*per, color="green", alpha=0.2, label="Test")
            ax.text(pd.Timestamp(per[0]) + (pd.Timestamp(per[1]) - pd.Timestamp(per[0])) / 2,
                     0.85, "Test", color="green", fontsize=12, fontweight="bold", ha="center")
    else : 
            ax.axvspan(*test_period, color="green", alpha=0.2, label="Test")
            ax.text(pd.Timestamp(test_period[0]) + (pd.Timestamp(test_period[1]) - pd.Timestamp(test_period[0])) / 2,
                     0.85, "Test", color="green", fontsize=12, fontweight="bold", ha="center")

    # Add labels in the shaded regions
    ax.text(pd.Timestamp(validation_period[0]) + (pd.Timestamp(validation_period[1]) - pd.Timestamp(validation_period[0])) / 2,
             0.85, "Validation", color="orange", fontsize=12, fontweight="bold", ha="center")
    ax.text(pd.Timestamp(train_period[0]) + (pd.Timestamp(train_period[1]) - pd.Timestamp(train_period[0])) / 2,
             0.85, "Train", color="brown", fontsize=12, fontweight="bold", ha="center")



    ax.legend(loc="center")
    plt.show()






# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                   Palette de couleurs                                 | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

m = -1
M = 1
s = (M-m)/24
levs = np.arange(m, M + s, s)
# Choix des couleurs de la palette :
negatives = mcm.get_cmap('Blues', int(256*2))(range(int(256*2)))
positives = mcm.get_cmap('autumn', 256)(range(256))
positives[:, -1] = np.linspace(1, 0.2, 256)
negatives = negatives[int(256):]
negatives[:, -1] = np.linspace(0.2, 1, 256)
colorlist = np.zeros((512, 4))
colorlist[256:] = negatives
colorlist[:256] = positives
colorlist = np.flip(colorlist, axis = 0)
mycmp = LinearSegmentedColormap.from_list('Mycmp', colors = colorlist, N = 256)
    
def loss_plot(path, criterion_config, details=True, title=None):

    losses={'Train Loss':[], 
            'Validation Loss':[], 
            'Learning Rate':[],
            'Learning Rate Epoch':[],
            'Step time':[]}
    if details :
        for specific_loss in criterion_config.keys():
            if specific_loss!='details' and criterion_config[specific_loss]!=0.:
                losses['Train  '+specific_loss]=[]
                losses['Valid  '+specific_loss]=[]
    with open(path, 'r') as f:
        data=f.readlines()
        for line in data:
            for key in losses.keys():
                if key in line : 
                    losses[key].append(float(line.split()[-1]))

        for i in range(len(losses['Learning Rate'])-1):
            if losses['Learning Rate'][i]!=losses['Learning Rate'][i+1]:
                losses['Learning Rate Epoch'].append(i)
        
    fig, ax=plt.subplots(1,1, figsize=(15,7))
    
    for key in losses.keys():
        if 'Loss' in key:
            ax.plot((np.log(losses[key])),label=key)
            
    ax.vlines(losses['Learning Rate Epoch'], 0, 1, transform=ax.get_xaxis_transform(), label='lr decreased', colors='r', alpha=0.1)
    
    if details : 
        for key in criterion_config.keys():
            if key!='details' and criterion_config[key]!=0.:
                ax.plot(np.log(losses['Train  '+ key]),label='Train  '+ key)
                ax.plot(np.log(losses['Valid  '+ key]),label='Valid  '+ key)
                #legend+=['Train '+ key, 'Valid '+key]
    ax.legend()
    if title is not(None): plt.title(title)
    plt.show()
    
    
import numpy as np
import datetime as dt

def generate_dates(start_date, delta, num_dates):
    """
    Generate a list of dates starting from start_date with a given delta.
    
    Parameters:
        start_date (datetime.date): The initial date.
        delta (datetime.timedelta): The interval between dates.
        num_dates (int): The number of dates to generate.
        
    Returns:
        list: List of datetime.date objects.
    """
    dates = []
    current_date = start_date
    current_year=start_date.year
    for _ in range(num_dates):
        dates.append(current_date)
        current_date += delta
        if current_date.year!=current_year:
            current_year+=1
            current_date=dt.date(current_year,1,1)
    return dates

def monthly_averaged_from_weekly(data, start_date=dt.date(1993, 1, 1), delta=dt.timedelta(days=8)):
    """
    Compute the monthly averaged data from weekly data.
    
    Parameters:
        data (np.ndarray): The input data array with time as the first dimension.
        start_date (datetime.date): The initial date for the data.
        delta (datetime.timedelta): The interval between data points.
        
    Returns:
        np.ndarray: The monthly averaged data.
    """
    # Generate the dates corresponding to the data points
    num_dates = data.shape[0]
    dates = generate_dates(start_date, delta, num_dates)
    
    # Group data by month
    monthly_data = {}
    for i, date in enumerate(dates):
        month = (date.year, date.month)
        if month not in monthly_data:
            monthly_data[month] = []
        monthly_data[month].append(data[i])
    
    # Compute the monthly averages
    monthly_averages = []
    for month in sorted(monthly_data.keys()):
        monthly_values = np.array(monthly_data[month])
        monthly_mean = np.nanmean(monthly_values, axis=0)
        monthly_averages.append(monthly_mean)
    
    return np.array(monthly_averages)

if __name__=="__main__": 
    
    #path_save="/home/luther/Documents/presentations/CSI_1/completion/physics/"
    
    plt.rcParams.update({'font.size': 12})
    import xarray as xr
    
    




    #physics=np.load("/home/luther/Documents/npy_data/physics/processed_physics_1993_2023_8d_lat50_100.npy")
    # physics=np.load("/home/luther/Documents/npy_data/physics/processed_physics_1998_2023_8d_lat50_100.npy")

    # imshow_area(np.nanmean(physics[:,0],axis=0))    
    chl_8d_100=np.load("/home/luther/Documents/npy_data/chl/chl_avw_glob_100km_8d_1997_2023.npy")
    
    
    
    imshow_area(np.nanmean(chl_8d_100,axis=0), cmap='viridis', vmin=-3,vmax=1, colorbar=False, log=True, title='Chl : Mean(mg/m3)')
    # path="/home/luther/Documents/npy_data/completion/daily/chl_psc_avw_roy_1d_glob_100km_1998_2023_psc100.nc"
    # chl_1d_100=xr.open_dataset(path, chunks={'time': 100}).to_array(dim='variables').astype('float16').sel(variables='CHL').data.compute()
    
    # imshow_area(np.sum(np.isnan(chl_8d_100),axis=0)/len(chl_8d_100),cmap='terrain', title="Missing values Chl Avw weekly 100km", save="/home/luther/Documents/plot/prep/miss_values_chlavw_8d_100km.png")
    # imshow_area(np.sum(np.isnan(chl_1d_100),axis=0)/len(chl_1d_100),cmap='terrain', title="Missing values Chl Avw daily 100km", save="/home/luther/Documents/plot/prep/miss_values_chlavw_1d_100km.png")

    # weekly_var=np.nanmean(np.nanstd(chl_1d_100.reshape(1187,8,180,360),axis=1),axis=0)
    # imshow_area(weekly_var+1e-5, log=True, vmin=-2, vmax=1.5,
    #             cmap='jet', title="Mean weekly variability of Chl Avw 100km", save="/home/luther/Documents/plot/prep/var_dailyweekly_chlavw.png")
    
    # mean_1d=np.nanmean(chl_1d_100,axis=0)
    # mean_1d = np.where(np.isinf(mean_1d), np.nan, mean_1d)

    # imshow_area(mean_1d, log=True, vmin=-2, vmax=1.5,
    #             cmap='jet', title="Mean  Chl Avw 100km 1998-2023", save="/home/luther/Documents/plot/prep/mean_chlavw.png")
    # imshow_area(np.nanmean(chl,axis=0),cmap="jet", log=True,
    # imshow_area(np.nanmean(chl,axis=0),cmap="jet", log=True,
    #             title="Averaged chl-a 1998-2023", 
    #             colorbar='mg/m3',save=path_save+"chl_avw_mean.png")
    
    # imshow_area(np.nanmean(physics[:,5],axis=0),cmap=cm.deep, 
    #             title="Averaged mixed layer depth 1998-2023", 
    #             colorbar='m',save=path_save+"mld_averaged.png")
    
    # imshow_area(np.nanmean(physics[:,5],axis=0),cmap='jet', 
    #             title="Averaged sea surface temperature 1998-2023", 
    #             colorbar='°C')#,save=path_save+"sst_averaged.png")
    
    # imshow_area(np.nanmean(physics[:,0],axis=0),cmap=cm.haline, vmin=30, vmax=37, 
    #             title="Averaged sea surface salinity 1998-2023", 
    #             colorbar='ppt (1e-4)',save=path_save+"sss_averaged.png")
    
    # imshow_area(np.nanmean(physics[:,4],axis=0),cmap='seismic' ,vmin=-2, vmax=2, 
    #             title="Averaged sea surface height 1998-2023", 
    #             colorbar='m',save=path_save+"ssh_averaged.png")
    
    # imshow_area(np.nanmean(physics[:,8],axis=0),cmap=cm.solar,# vmin=0.65e6,vmax=0.95e6 ,
    #             title="Averaged shortwave radiation 1998-2023", 
    #             colorbar='J/m2',save=path_save+"solar_averaged.png")
    
    # imshow_area(np.nanmean(physics[:,6],axis=0),cmap=cm.topo, 
    #             title="Bathymetry", 
    #             colorbar='m',save=path_save+"bathy_averaged.png")
    
    # imshow_area(np.nanmean(physics[:,2],axis=0),cmap=cm.turbid, vmin=0.1, vmax=0.8,
    #             title="Averaged currents norm 1998-2023", 
    #             colorbar='m/s',save=path_save+"currents_norm_averaged.png")
    
    # imshow_area(np.nanmean(physics[:,2],axis=0),cmap=cm.curl, vmin=-0.02, vmax=0.02,
    #             title="Averaged vorticity of surface winds 1998-2023", 
    #             colorbar='m/s',save=path_save+"winds_vorticity_averaged.png")
    

#%%
    
        
    # import matplotlib.pyplot as plt
    # import matplotlib.dates as mdates
    # import datetime
    
    # # Define the data for each satellite mission or dataset
    # data = [
    #     ("SeaWiFS", datetime.datetime(1997, 9, 1), datetime.datetime(2010, 12, 31), 'purple'),
    #     ("MERIS", datetime.datetime(2002, 5, 1), datetime.datetime(2012, 4, 30), 'cyan'),
    #     ("MODIS-Aqua", datetime.datetime(2002, 7, 1), datetime.datetime(2023, 1, 1), 'yellowgreen'),
    #     ("VIIRS-SNPP", datetime.datetime(2012, 10, 1), datetime.datetime(2023, 1, 1), 'orange'),
    #     ("OLCI", datetime.datetime(2016, 2, 16), datetime.datetime(2023, 1, 1), 'pink'),
    #     ("Merged Chla Datasets", datetime.datetime(1997, 9, 1), datetime.datetime(2023, 1, 1), 'lightblue'),
    #     ("OC-CCI", datetime.datetime(1997, 9, 1), datetime.datetime(2023, 1, 1), 'red'),
    #     ("GlobColour", datetime.datetime(1997, 9, 1), datetime.datetime(2023, 1, 1), 'lightgreen')
    # ]
    
    # # Convert start and end dates to matplotlib dates
    # for i in range(len(data)):
    #     start_date = mdates.date2num(data[i][1])
    #     end_date = mdates.date2num(data[i][2])
    #     data[i] = (data[i][0], start_date, end_date - start_date, data[i][3])
    
    # # Set up the plot
    # fig, axs = plt.subplots(2,1,figsize=(17, 12))
    # ax=axs[1]
    # ax.set_title("Concurrent Global Ocean Color Satellite Missions")
    # ax.set_xlabel("Year")
    # ax.set_yticks(range(len(data)))
    # ax.set_yticklabels([item[0] for item in data])
    
    # # Plot each mission/dataset as a horizontal bar
    # for i, (label, start, duration, color) in enumerate(data):
    #     ax.broken_barh([(start, duration)], (i - 0.4, 0.8), color=color)
    
    # # Set up x-axis with year format
    # ax.xaxis.set_major_locator(mdates.YearLocator(5))
    # ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # plt.xticks(rotation=45)
    
    # # Show grid and plot
    # plt.grid(axis='x', linestyle='--', color='gray')
    # plt.tight_layout()
    # plt.show()
    
    # chl_avw=np.load("/home/luther/Documents/npy_data/chl/chl_avw_lat50_100km_8d_1997_2023.npy")
    # chl_occi=np.load("/home/luther/Documents/npy_data/chl/chl_occi_lat50_100km_8d_1997_2023.npy")
    
    # chl_avw_monthly=monthly_averaged_from_weekly(chl_avw[16:],start_date=dt.date(1998, 1, 1), delta=dt.timedelta(days=8))
    # chl_occi_monthly=monthly_averaged_from_weekly(chl_occi[16:],start_date=dt.date(1998, 1, 1), delta=dt.timedelta(days=8))
    

    # #mei_index = pd.DataFrame({"MEI Index": np, index=dates)
    # #dates = pd.date_range(start="1998-01-01", end="2016-12-31", freq="M")
    # #mei_index = pd.DataFrame({"MEI Index": np.random.normal(loc=0, scale=1, size=len(dates))}, index=dates)
    # dates = pd.date_range(start="1993-01-01", end="2023-12-31", freq="M")
    # chl_month_avw=pd.DataFrame({"Chl avw": np.concatenate((np.zeros((60))*np.nan,np.nanmean(chl_avw_monthly,axis=(1,2))),axis=0)}, index=dates)
    # chl_month_occi=pd.DataFrame({"Chl avw": np.concatenate((np.zeros((60))*np.nan,np.nanmean(chl_occi_monthly,axis=(1,2))),axis=0)}, index=dates)
    
    
    #   # Define the periods
    # physique_coverage = ("1993-01-01", "2023-12-31")
    #   #chl_coverage = ("1998-01-01", "2023-12-31")
    
    #   # Plot the MEI index data
    # #fig = plt.figure(figsize=(12, 4), dpi=200)#, frameon=False)
    # ax = axs[0]
    # ax.plot(chl_month_avw.index, chl_month_avw.iloc[:, 0], color="blue", alpha=0.6, label="Chl avw")
    # ax.plot(chl_month_occi.index, chl_month_occi.iloc[:, 0], color="red", alpha=0.6, label="Chl occi")
    
    # # Formatting the plot
    # ax.set_ylabel("Averaged Chl mg/m3")
    # #ax.ylim(-3, 3)
    # ax.set_xlim(chl_month_avw.index.min(), chl_month_avw.index.max())
    
    #   # else :
    #   #     dates = pd.date_range(start="1993-01-01", end="2023-12-31", freq="M")
    #   #     ax.plot(dates,np.concatenate((np.zeros(12*5)*np.nan,1+np.zeros(dates.shape[0]-12*5))),alpha=0.75,color="green",label="Chlorophylle Satellite spanning period")
    #   #     ax.plot(dates,np.zeros(dates.shape),alpha=0.75,color="blue",label="Glorysv12 spanning period")
    #   #     ax.set_ylabel("Training, validation and test periods")
    #   #     ax.get_yaxis().set_visible(False)
    #   #ax.set_xlim(dates[0], dates[-1])
    
    #   # Shade the validation, train, and test periods
    #   #ax.axvspan(*chl_coverage, color="orange", alpha=0.4, label="Validation")
    # ax.axvspan(*physique_coverage, color="blue", alpha=0.1, label="Available physical data")
    
    
    #   # Add labels in the shaded regions
    #   # ax.text(pd.Timestamp(chl_coverage[0]) + (pd.Timestamp(chl_coverage[1]) - pd.Timestamp(chl_coverage[0])) / 2,
    #   #          0.85, "Validation", color="orange", fontsize=12, fontweight="bold", ha="center")
    #   # ax.text(pd.Timestamp(physique_coverage[0]) + (pd.Timestamp(physique_coverage[1]) - pd.Timestamp(physique_coverage[0])) / 2,
    #   #          0.20, "Available physical data", color="blue", fontsize=12, fontweight="bold", alpha=0.6,ha="center")
    
    
    
    # ax.legend(loc="upper left")
    # #plt.savefig("/home/luther/Documents/presentations/CSI_1/satellite_period_chl_occi_avw.png")
    # plt.show()

#%%
# import numpy as np
# import datetime as dt
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import pandas as pd

# # Load the chlorophyll data
# chl_avw = np.load("/home/luther/Documents/npy_data/chl/chl_avw_lat50_100km_8d_1997_2023.npy")
# chl_occi = np.load("/home/luther/Documents/npy_data/chl/chl_occi_lat50_100km_8d_1997_2023.npy")

# # Assuming `monthly_averaged_from_weekly` is defined elsewhere
# chl_avw_monthly = monthly_averaged_from_weekly(chl_avw[16:], start_date=dt.date(1998, 1, 1), delta=dt.timedelta(days=8))
# chl_occi_monthly = monthly_averaged_from_weekly(chl_occi[16:], start_date=dt.date(1998, 1, 1), delta=dt.timedelta(days=8))

# chlnan_avw_monthly=monthly_averaged_from_weekly(~np.isnan(chl_avw[16:]),start_date=dt.date(1998, 1, 1), delta=dt.timedelta(days=8))
# chlnan_occi_monthly=monthly_averaged_from_weekly(~np.isnan(chl_occi[16:]),start_date=dt.date(1998, 1, 1), delta=dt.timedelta(days=8))

# # Generate the dates and create dataframes for plotting
# dates = pd.date_range(start="1993-01-01", end="2023-12-31", freq="M")
# chl_month_avw = pd.DataFrame({"Chl avw": np.concatenate((np.full(60, np.nan), np.nanmean(chl_avw_monthly, axis=(1, 2))))}, index=dates)
# chl_month_occi = pd.DataFrame({"Chl occi": np.concatenate((np.full(60, np.nan), np.nanmean(chl_occi_monthly, axis=(1, 2))))}, index=dates)

# chlnan_month_avw = pd.DataFrame({"Chlnan avw": np.concatenate((np.full(60, np.nan), np.nanmean(chlnan_avw_monthly, axis=(1, 2))))}, index=dates)
# chlnan_month_occi = pd.DataFrame({"Chlnan occi": np.concatenate((np.full(60, np.nan), np.nanmean(chlnan_occi_monthly, axis=(1, 2))))}, index=dates)

# # Define the period for shading
# physique_coverage = ("1993-01-01", "2023-12-31")

# # Data for the timeline plot
# timeline_data = [
#     ("SeaWiFS", dt.datetime(1997, 9, 1), dt.datetime(2010, 12, 31), 'purple'),
#     ("MERIS", dt.datetime(2002, 5, 1), dt.datetime(2012, 4, 30), 'cyan'),
#     ("MODIS-Aqua", dt.datetime(2002, 7, 1), dt.datetime(2023, 12,31), 'yellowgreen'),
#     ("VIIRS-SNPP", dt.datetime(2012, 10, 1), dt.datetime(2023, 12,31), 'orange'),
#     ("OLCI", dt.datetime(2016, 2, 16), dt.datetime(2023, 12,31), 'pink'),
#     ("Glorysv12 data", dt.datetime(1993,1,1), dt.datetime(2023,12,31), 'lightblue'),
#     ("OC-CCI", dt.datetime(1997, 9, 1), dt.datetime(2023, 12,31), 'red'),
#     ("GlobColour", dt.datetime(1997, 9, 1), dt.datetime(2023, 12,31), 'blue')
# ]

# # Convert dates to matplotlib format for the timeline plot
# for i in range(len(timeline_data)):
#     start_date = mdates.date2num(timeline_data[i][1])
#     end_date = mdates.date2num(timeline_data[i][2])
#     timeline_data[i] = (timeline_data[i][0], start_date, end_date - start_date, timeline_data[i][3])

# # Set up the figure with two subplots
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), gridspec_kw={'height_ratios': [1, 2]}, dpi=200)
# fig.subplots_adjust(hspace=0.4)

# # First subplot: Timeline plot
# ax1.set_title("Time span of sensors and merged datasets")
# ax1.set_xlabel("Year")
# ax1.set_yticks(range(len(timeline_data)))
# ax1.set_yticklabels([item[0] for item in timeline_data])

# # Add each timeline bar to ax1
# for i, (label, start, duration, color) in enumerate(timeline_data):
#     ax1.broken_barh([(start, duration)], (i - 0.4, 0.8), color=color)

# ax1.xaxis.set_major_locator(mdates.YearLocator(4))
# ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# ax1.grid(axis='x', linestyle='--', color='gray')
# ax1.set_xlim(chl_month_avw.index.min(), chl_month_avw.index.max())
# # Second subplot: Chlorophyll time series plot
# ax2.plot(chlnan_month_avw.index, chl_month_avw.iloc[:, 0], color="blue", alpha=0.6, label="GlobColour avw")
# ax2.plot(chlnan_month_occi.index, chl_month_occi.iloc[:, 0], color="red", alpha=0.6, label="OC-CCI")
# ax2.set_ylabel("Averaged Chl mg/m3")
# ax2.set_xlim(chl_month_avw.index.min(), chl_month_avw.index.max())

# # Shade the physical data availability period
# #ax2.axvspan(pd.to_datetime(physique_coverage[0]), pd.to_datetime(physique_coverage[1]), color="blue", alpha=0.1, label="Available physical data")

# # Add legend to the chlorophyll plot
# ax2.legend(loc="upper left")
# plt.savefig("/home/luther/Documents/presentations/CSI_1/temporal_problem_chl.png")
# # Show the combined plot
# plt.show()


#     path_loss="/home/luther/Documents/result/SmaAt_2_/SmaAt_2__training.txt"
#     criterion_config = {
#     'MSE_psc' : 1., # eventual features and weights of losses (put None instead of 0 if you don't want the loss to be compute)
#     'MSE_chl':0.,
#     'Under_chl':0.,
#     'details':True,
# }
#     loss_plot(path_loss, criterion_config, details=True)
    
    # PSC=np.load("/datatmp/home/lollier/npy_emergence/psc.npy")
    #chl_avw=np.load("/home/luther/Documents/npy_data/chl/chl_avw_1998_2020_100k_8d_lat50.npy")
    #imshow_area(np.nanmean(chl_avw,axis=0),log=True)
    # CHL_occi=np.load("/datatmp/home/lollier/chl/chl_occi_100k_8d_lat50.npy")

    # CHL_occi=CHL_occi[16:1028]
    # CHL_avw=np.squeeze(CHL_avw)
    
    # # plt.plot(np.nanmedian(CHL_avw, axis=(1,2)))
    # # plt.plot(np.nanmedian(CHL_occi, axis=(1,2)))
    # # plt.show()
    
    # premiere_periode=100*np.sum(np.isnan(CHL_avw[184:598]),axis=0)/414 #2002-2010
    # deuxieme_periode=100*np.sum(np.isnan(CHL_avw[828:]),axis=0)/184

    # #imshow_area(100*(deuxieme_periode-premiere_periode)/premiere_periode, vmin=-100,vmax=100,cmap='seismic',title='evolution % données manquantes chl entre 2002-2010 et 2016-2019 pour avw')



    
    # proj=ccrs.Mercator()
    # fig = plt.figure(figsize=(10,10))
    # gs = fig.add_gridspec(nrows=2, ncols=1,
    #                       hspace=0.01, wspace=0.025, right=0.9)
    # axs=[]
    # axs.append(fig.add_subplot(gs[0],projection=proj))
    # axs.append(fig.add_subplot(gs[1],projection=proj))
    
    # imshow_area(premiere_periode, cmap=cm.algae, fig=fig, ax=axs[0], title='% données manquantes chl entre 2002-2010 pour avw', colorbar=False, vmin=0, vmax=100)
    # imshow_area(deuxieme_periode, cmap=cm.algae, fig=fig, ax=axs[1],title='% données manquantes chl entre 2016-2019 pour avw', colorbar=False, vmin=0, vmax=100)
    

    # cb=Normalize(0,100)

    # cax_pred = plt.axes([1, 0.25, 0.02, 0.5])

    # fig.colorbar(ScalarMappable(cb,cmap=cm.algae),cax=cax_pred, label='%')
    
    # ###
    # premiere_periode=100*np.sum(np.isnan(CHL_occi[184:598]),axis=0)/414 #2002-2010
    # deuxieme_periode=100*np.sum(np.isnan(CHL_occi[828:]),axis=0)/184

    # #imshow_area(100*(deuxieme_periode-premiere_periode)/premiere_periode, vmin=-100,vmax=100,cmap='seismic',title='evolution % données manquantes chl entre 2002-2010 et 2016-2019 pour occci')
    
    
    # proj=ccrs.Mercator()
    # fig = plt.figure(figsize=(10,10))
    # gs = fig.add_gridspec(nrows=2, ncols=1,
    #                       hspace=0.01, wspace=0.025, right=0.9)
    # axs=[]
    # axs.append(fig.add_subplot(gs[0],projection=proj))
    # axs.append(fig.add_subplot(gs[1],projection=proj))
    
    # imshow_area(premiere_periode, cmap=cm.algae, fig=fig, ax=axs[0], title='% données manquantes chl entre 2002-2010 pour occci', colorbar=False, vmin=0, vmax=100)
    # imshow_area(deuxieme_periode, cmap=cm.algae, fig=fig, ax=axs[1],title='% données manquantes chl entre 2016-2019 pour occci', colorbar=False, vmin=0, vmax=100)
    

    # cb=Normalize(0,100)

    # cax_pred = plt.axes([1, 0.25, 0.02, 0.5])

    # fig.colorbar(ScalarMappable(cb,cmap=cm.algae),cax=cax_pred, label='%')
    # # plt.figure(figsize=(20,5))
    # year=np.linspace(1998,2019,1012)

    # plt.plot(year, np.sum(np.isnan(CHL_avw),axis=(1,2))/36000)
    # plt.plot(year, np.sum(np.isnan(CHL_occi),axis=(1,2))/36000)
    # plt.legend(['avw','occci'])
    # plt.grid()
    # plt.xlabel('year')
    # plt.title("% données manquantes chl pour avw et occci")
    # plt.show()

    #CHL par année
    
    # for i in range(0,CHL.shape[0],46):
    #     imshow_area(np.nanmean(CHL[i:i+46,0],axis=0), log=True,title=f'chl year {1998+i/46}')
    
    #CHL par mois
    
    # for i in range(12):
    #     imshow_area(np.nanmean(CHL[i::46,0],axis=0), log=True,title=f'chl month {i}')

    #%PSC par année
    
    # proj=ccrs.Mercator()

    # # for i in range(0,PSC.shape[0],46):
    #     fig = plt.figure(figsize=(10,15))
    #     gs = fig.add_gridspec(nrows=3, ncols=1,
    #                           hspace=0.05, wspace=0.025, right=1)
    #     axs=[]
    #     axs.append(fig.add_subplot(gs[0],projection=proj))
    #     axs.append(fig.add_subplot(gs[1],projection=proj))
    #     axs.append(fig.add_subplot(gs[2],projection=proj))
        
    #     imshow_area(np.nanmean(PSC[i:i+46,0],axis=0), cmap='BuGn', fig=fig, ax=axs[0], log=False,title=f'Micro year {1998+i/46}', colorbar=False)
    #     imshow_area(np.nanmean(PSC[i:i+46,1],axis=0), cmap='BuGn', fig=fig, ax=axs[1], log=False,title=f'Nano year {1998+i/46}', colorbar=False)
    #     imshow_area(np.nanmean(PSC[i:i+46,2],axis=0), cmap='BuGn', fig=fig, ax=axs[2], log=False,title=f'Pico year {1998+i/46}', colorbar=False)
        
    #     cb=Normalize(0,100)
    #     cax = plt.axes([1.1, 0.15, 0.01, 0.7])
    #     fig.colorbar(ScalarMappable(cb,cmap='BuGn'),cax=cax, label='%')
    #     plt.show()


    #PSC*CHL par année
    
    
    # proj=ccrs.Mercator()

    # for i in range(0,PSC.shape[0],46):
        
    #     fig = plt.figure(figsize=(10,15))
    #     gs = fig.add_gridspec(nrows=3, ncols=1,
    #                           hspace=0.05, wspace=0.025, right=1)
    #     axs=[]
    #     axs.append(fig.add_subplot(gs[0],projection=proj))
    #     axs.append(fig.add_subplot(gs[1],projection=proj))
    #     axs.append(fig.add_subplot(gs[2],projection=proj))
        
    #     chl=np.nanmean(CHL[i:i+46,0],axis=0)
        
    #     imshow_area(chl*np.nanmean(PSC[i:i+46,0],axis=0), cmap='hsv', fig=fig, ax=axs[0], log=True,title=f'Micro year {1998+i/46}', colorbar=False)
    #     imshow_area(chl*np.nanmean(PSC[i:i+46,1],axis=0), cmap='hsv', fig=fig, ax=axs[1], log=True,title=f'Nano year {1998+i/46}', colorbar=False)
    #     imshow_area(chl*np.nanmean(PSC[i:i+46,2],axis=0), cmap='hsv', fig=fig, ax=axs[2], log=True,title=f'Pico year {1998+i/46}', colorbar=False)
        
    #     cb=LogNorm(10**-3,10**2)
    #     cax = plt.axes([1.1, 0.15, 0.01, 0.7])
    #     fig.colorbar(ScalarMappable(cb,cmap='hsv'),cax=cax, label='concentration PSC')
    #     plt.show()
    
    
    
    # #PSC*CHL par mois
    
    
    # proj=ccrs.Mercator()

    # for i in range(12):
        
    #     fig = plt.figure(figsize=(10,15))
    #     gs = fig.add_gridspec(nrows=3, ncols=1,
    #                           hspace=0.05, wspace=0.025, right=1)
    #     axs=[]
    #     axs.append(fig.add_subplot(gs[0],projection=proj))
    #     axs.append(fig.add_subplot(gs[1],projection=proj))
    #     axs.append(fig.add_subplot(gs[2],projection=proj))
        
    #     chl=np.nanmean(CHL[i:i+46,0],axis=0)
        
    #     imshow_area(chl*np.nanmean(PSC[i::46,0],axis=0), cmap='hsv', fig=fig, ax=axs[0], log=True,title=f'Micro month {i}', colorbar=False)
    #     imshow_area(chl*np.nanmean(PSC[i::46,1],axis=0), cmap='hsv', fig=fig, ax=axs[1], log=True,title=f'Nano month {i}', colorbar=False)
    #     imshow_area(chl*np.nanmean(PSC[i::46,2],axis=0), cmap='hsv', fig=fig, ax=axs[2], log=True,title=f'Pico month {i}', colorbar=False)
        
    #     cb=LogNorm(10**-3,10**2)
    #     cax = plt.axes([1.1, 0.15, 0.01, 0.7])
    #     fig.colorbar(ScalarMappable(cb,cmap='hsv'),cax=cax, label='concentration PSC')
    #     plt.show()

    #%nan par pixel pour la CHL et pour les PSC
    
    # spatial
    #imshow_area(np.nanmean(CHL[:,0],axis=0), cmap='jet',title='% données manquantes chl', log=True)
    # imshow_area(np.sum(np.isnan(PSC[:,0]),axis=0)/1012, cmap='viridis_r',title='% données manquantes PSC résolution 100km')
    # imshow_area(np.sum(np.isnan(PSC[:,1]),axis=0)/1012, cmap='BuGn',title='% nan Nano')
    # imshow_area(np.sum(np.isnan(PSC[:,2]),axis=0)/1012, cmap='BuGn',title='% nan Pico')

    # #temporel

    # plt.plot(np.sum(np.isnan(CHL[:,0]),axis=(1,2))/36000)
    # plt.plot(np.sum(np.isnan(PSC[:,0]),axis=(1,2))/36000)
    # plt.plot(np.sum(np.isnan(PSC[:,1]),axis=(1,2))/36000)
    # plt.plot(np.sum(np.isnan(PSC[:,2]),axis=(1,2))/36000)
    
    # plt.legend(['chl','Micro','Nano','Pico'])
    # plt.show()
    
    

    
    
