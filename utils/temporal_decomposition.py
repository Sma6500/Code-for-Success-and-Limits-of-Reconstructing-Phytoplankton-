#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: lollier

Keerthi detrending 
Contrasted Contribution of Intraseasonal Time Scales to Surface Chlorophyll Variations in a Bloom and an Oligotrophic Regime
https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019JC015701

"""

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                               Temporal decomposition                                  | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import xarray as xr
from tqdm import tqdm
from scipy import signal

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                       Implement a derived X11 method for detrending                   | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #




class temporal_decomposition():
    """
    Create an object from a temporal series with Trend, Seasonality and IntraSeasonnal components.
    The frequency sample needs to be 8 days. Any kind of shape is allowed but the time component needs to be the first dimension.
    If the dataset is sufficiently long, it's better to exclude the first and last year from further analysis after decomposition,
    the method is less robust at the boundaries because of intrinsic convolutions properties. 
    
    TODO:
        - Separate T into Trend and Interannual
        - Verify with Keerthi the use of S3 or S2 at step 5/6
        - Linear interpolation for now, consider adding cubic or other methods
    """
    
    def __init__(self,time_serie):
        """
        Parameters
        ----------
        time_serie : np.array 
            data that will be decompose, time axis needs to be the first dim
            
        -------
        """
        self.time_serie=time_serie
        
        self.ts_shape=self.time_serie.shape
        self.len=time_serie.shape[0]
        
        self.T=np.empty(self.ts_shape)
        self.S=np.empty(self.ts_shape)
        self.I=np.empty(self.ts_shape)
        
        self.build_decomposition()
        
    def build_decomposition(self):
        
        #loop over all ts
        for ts_index,_ in (np.ndenumerate(self.time_serie[0])):
            
            #building the slice for the right index
            ts_slice=[]
            for dim,size in enumerate(self.ts_shape):
                if dim==0: #temporal axis
                    ts_slice.append(slice(0,size))
                else :
                    ts_slice.append(ts_index[dim-1])
            
            ts_slice=tuple(ts_slice)
            if np.isnan(self.time_serie[ts_slice]).sum()==self.len:
                
                self.T[ts_slice]=self.time_serie[ts_slice]
                self.S[ts_slice]=self.time_serie[ts_slice]
                self.I[ts_slice]=self.time_serie[ts_slice]
                
            else :
                t,s,i=self.decompose(self.interpolate(self.time_serie[ts_slice]))
                self.T[ts_slice]=t*(1+(np.isnan(self.time_serie[ts_slice])*self.time_serie[ts_slice]))
                self.S[ts_slice]=s*(1+(np.isnan(self.time_serie[ts_slice])*self.time_serie[ts_slice]))
                self.I[ts_slice]=i*(1+(np.isnan(self.time_serie[ts_slice])*self.time_serie[ts_slice]))
            
    #linear interoplation
    def interpolate(self, ts):
        axis=np.arange(0,self.len)
        return np.interp(axis, axis[~np.isnan(ts)], ts[~np.isnan(ts)])

    def moving_average(self, x):

        
        w=np.ones(47)*2
        w[0],w[-1]=1,1
        
        #padding same instead of duplication of series' boundaries.
        res=np.convolve(x, w, 'same') / (46*2)

        return res

    def seasonal_running_mean(self, x):
        res=np.zeros((x.shape[0]))

        #no duplications anymore, the seasonal running mean for the boundaries is only computed with the available years.
        for i in range(46,x.shape[0]-46):
            res[i]=0.25*(x[i-46]+(2*x[i])+x[i+46])
        
        for i in range(0,46):
            res[i]=((2*x[i])+x[i+46])/3
            
        for i in range(x.shape[0]-46,x.shape[0]):
            res[i]=(x[i-46]+(2*x[i]))/3

        return res
            
        
    def henderson_filter_51(self, x):

        H_wts=np.array([-0.0003, -0.0011, -0.0022, -0.0036, -0.0051, -0.0064, -0.0072, -0.0075, -0.0069, -0.0055, -0.0032, 0.0002, 0.0045,  0.0096,  0.0154,  0.0217,  0.0284,  0.0352,  0.0418,  0.0482,  0.0539,  0.0590,  0.0631, 0.0661, 0.0680,  0.0686, 0.0680,  0.0661,  0.0631, 0.0590,  0.0539,  0.0482,  0.0418,  0.0352,  0.0284,  0.0217,  0.0154,  0.0096,  0.0045,  0.0002, -0.0032, -0.0055, -0.0069, -0.0075, -0.0072, -0.0064, -0.0051, -0.0036, -0.0022, -0.0011, -0.0003])
        
        res=np.convolve(x, H_wts, 'same')#/H_wts.sum() ? la somme des H_wts fait pas exactement 1 probablement du à un pb sur la gestion des float
        return res
    
    def band_pass(self, x):
        
        # filt=np.zeros(x.shape)
        # filt[:12]=np.ones((12)) #12*8=88
        
        # fft_x=sc.fft.fft(x)
        
        # return sc.fft.ifft(fft_x*filt)    
        Te=8 
        # Fréquence d'échantillonnage
        fe = 1/Te  # Hz
        
        # Fréquence de nyquist
        f_nyq = fe / 2.  # Hz
        
        # Fréquence de coupure
        fc = 1/88  # Hz
        
        # Préparation du filtre de Butterworth en passe bande
        b, a = signal.butter(4, fc/f_nyq, 'high', analog=False)
        
        return signal.filtfilt(b, a, x)

    def decompose(self, ts):
        
        #1        
        T1=self.moving_average(ts)
        Z=ts-T1
        
        #2
        S0=self.seasonal_running_mean(Z)
        MS0=self.moving_average(S0)
        S1=S0-MS0
        Z1=ts-S1
        
        #3
        T2=self.henderson_filter_51(Z1)
        Z2=ts-T2
        
        #4
        S2=self.seasonal_running_mean(Z2)
        MS2=self.moving_average(S2)
        S3=S2-MS2
        Z3=ts-S3
        
        
        #5
        I1=self.band_pass(S2) #keert utilise S2 mais c'est peut être S3 ? ça change un peu I et S mais c'est très léger

        #6
        T=self.henderson_filter_51(Z3)

        #7
        S=S2-I1 #see above
        
        #8
        I=I1+(ts-S-T)
        
        return T,S,I


if __name__=="__main__": 
    
    import tol_colors as tc

    import matplotlib
    matplotlib.rcParams['axes.prop_cycle']=plt.cycler('color', tc.colorsets['bright'])
    
    print('')
    chl=np.load("/home/luther/Documents/npy_data/chl/chl_avw_glob_100km_8d_1997_2023.npy")
    chl_decomposed=temporal_decomposition(np.squeeze(chl))
    
    year=np.linspace(1997, 2023, 1212)
    plt.plot(year,np.nanmean(chl_decomposed.time_serie,axis=(1,2)))
    
    plt.plot(year,np.nanmean(chl_decomposed.T,axis=(1,2)))
    plt.plot(year,np.nanmean(chl_decomposed.S,axis=(1,2)))
    plt.plot(year,np.nanmean(chl_decomposed.I,axis=(1,2)))
    
    plt.legend(['chl','T','S','I'])
    plt.show()
    
    plt.figure(figsize=(15,4),dpi=200)
    year=np.linspace(1998, 2023, 1196)
    plt.plot(year,chl[16:,50,50], label='Chl avw', color=tc.colorsets['bright'][0], alpha=0.5)
    
    plt.plot(year,chl_decomposed.T[16:,50,50], label='Interannual',  color=tc.colorsets['bright'][1])
    plt.plot(year,chl_decomposed.S[16:,50,50], label='Seasonnal',  color=tc.colorsets['bright'][2])
    plt.plot(year,chl_decomposed.I[16:,50,50], label='SubSesonnal', color=tc.colorsets['bright'][3], alpha=0.5)
    plt.title("Decomposed time serie of a random chlorophyll-a avw pixel")
    plt.legend()
    plt.savefig("/home/luther/Documents/presentations/CSI_1/ts_decompose_ex.png")
    plt.show()
    # plt.plot(chl_decomposed.time_serie[500:592,30,45])
    
    # plt.plot(chl_decomposed.T[500:592,30,45])
    # plt.plot(chl_decomposed.S[500:592,30,45])
    # plt.plot(chl_decomposed.I[500:592,30,45])
    
    # plt.legend(['chl','T','S','I'])
    
    # plt.show()
    
    # plt.plot(chl_decomposed.time_serie[:,35,55]-chl_decomposed.T[:,35,55]-chl_decomposed.S[:,35,55]-chl_decomposed.I[:,35,55])
    # plt.show()
    # ts=np.array([np.mean(test_time_serie[i:i+8,150,45]) for i in range(0,9853,8)])

    # ts_dec=temporal_decomposition(ts)

    # plt.plot(ts_dec.time_serie)

    # plt.plot(ts_dec.T)
    # plt.plot(ts_dec.S)
    # plt.plot(ts_dec.I,alpha=0.5)

    # plt.legend(['chl','T','S','I'])
# chl_dataset=np.load("/datatmp/home/lollier/npy_emergence/chl.npy")[:,0]
# bath=np.load("/datatmp/home/lollier/npy_emergence/dyn.npy")[:,-1]
# np.putmask(chl_dataset, bath<1000, np.nan)

# plt.figure(figsize=(20,10))
# year=np.linspace(1998,2019,1012)

# plt.plot(year,np.nanmedian(chl_dataset,axis=(1,2)))
# #plt.plot(year,np.nanmean(chl_dataset,axis=(1,2)))

# plt.title("Mean CHL for Bath>1000")
# #plt.legend(['Median','Mean'])
# plt.xlabel('year')
# plt.grid()
# plt.legend(['chl'])#,'T','S','I'])
# plt.show()


# plt.plot(year,np.sum(np.isnan(chl_dataset),axis=(1,2))/36000)
# plt.plot(year,np.nanmean(chl_dataset,axis=(1,2)))
# plt.legend(["% valeur manquante CHL","Moyenne CHL globale"])
# plt.title("%nan CHL vs moyenne CHL")
# plt.xlabel('year')
# plt.grid()
# plt.show()


# % OC satellites operating periods
# % Sat Start End
# % SeaWiFS 04/09/1997 11/12/2010
# % MERIS 28/04/2002 04/05/2012
# % MODIS 03/07/2002 -
# % VIIRSN 02/01/2012 -
# % OLA 25/04/2016 -
# % VIIRSJ1 29/11/2017 -
# % OLB 14/05/2018 -

#    #from cmocean import cm
# premiere_periode=np.sum(np.isnan(chl_dataset[184:644]),axis=0)/460
# deuxieme_periode=np.sum(np.isnan(chl_dataset[736:]),axis=0)/276
# imshow_area((deuxieme_periode-premiere_periode)/premiere_periode, cmap='viridis_r',title='evolution % données manquantes chl entre 2002-2012 et 2014-2019')

# imshow_area(premiere_periode, cmap='viridis_r',title='% données manquantes chl entre 2002-2012')
# imshow_area(deuxieme_periode, cmap='viridis_r',title='% données manquantes chl entre 2014-2019')

# imshow_area((premiere_periode-deuxieme_periode), vmin=-0.2,vmax=0.2,cmap='seismic',title='evolution % données manquantes chl entre 2002-2012 et 2014-2019')


















