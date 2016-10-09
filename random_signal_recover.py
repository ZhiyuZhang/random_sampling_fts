###### 
#    Example: 
#-------------------------------------
#     from subtract_self import *
#     subtract_all()
# ------------------------------------

from random import randrange, uniform
import matplotlib.pyplot as plt
from scipy.constants import *
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
from   astropy.io import fits
from   numpy import random
import scipy.optimize
import astropy.units as u
import sys, copy
from   astropy.io import ascii
from astropy.convolution import Gaussian1DKernel, convolve

def textlines(ax_f,z,y,freq_list ):
    for freq, name in freq_list: 
        plt.axvline(x=freq,alpha=0.2,linewidth=4,drawstyle='steps-mid') 
        ax_f.text(freq, y, name,rotation=90,size='6')
    return 



def subtract_self(frand, FTSobsid,  redshift,  width_of_sm_kernel,apod_width, Name, window):
    freq_Cp      = 1900.55 /(1+redshift) 
#   freq_Oiii88  = 3393.006/(1+redshift)
    freq_Oiii88  = frand #800.
    freq_Oiii51  = 5785.88 /(1+redshift)
    freq_NII122  = 2459.38 /(1+redshift)
    freq_OI63    = 4744.75 /(1+redshift)
    freq_H2O1    = 1661.   /(1+redshift)
    freq_H2O2    = 1670.   /(1+redshift)
    freq_OH      = 2512.30 /(1+redshift) 
    freq_HF10    = 1232.48 /(1+redshift)
    freq_OI145   = 2060.07 /(1+redshift) 
#   freq_H2O3    = 2024.456550071783  /(1+redshift) 


    freq_list = [[freq_Cp,           "C$^+$ 158"], 
                 [freq_Oiii88,    'O$_{III} 88$'],
                 [freq_Oiii51,    'O$_{III} 51$'],
                 [freq_NII122,     'N$_{II}$122'],   
                 [freq_OI63         ,'O$_{I}$63'],
                 [freq_H2O1          ,'H$_2$O-1'],
                 [freq_H2O2          ,'H$_2$O-2'],
                 [freq_OH,                  'OH'],
                 [freq_OI145,         'O$_I$145'],
#                [freq_H2O3,              'H2O?'],
#                [freq_HF10,             'HF1-0'],
#                [freq_HF21,             'HF2-1']]
                 ]

## read in the target spectra 
    spec     = fits.open(str(FTSobsid)+'_HR_spectrum_point_apod.fits')
#   spec     = fits.open(str(FTSobsid)+'_HR_spectrum_point.fits')
    spec_ori = fits.open(str(FTSobsid)+'_HR_spectrum_point.fits')

# define detectors of SLW and SSW
    centreDetectors = ["SLWC3","SSWD4"]


# define the output spectra (using the central pixel) 
    cent_spec_SLW  = np.full((3,1905),1.0) 
    cent_spec_SSW  = np.full((3,2082),1.0) 

# assignment of the output spectra  
    for k in range(2, 24):
        if (spec[k].header['EXTNAME'] == centreDetectors[0]):
            cent_spec_SLW[0,]      =     spec[k].data.wave 
            cent_spec_SLW[1,]      =     spec[k].data.flux 
            cent_spec_SLW[2,]      = spec_ori[k].data.error/np.sqrt(apod_width/1.2)
    
        if (spec[k].header['EXTNAME'] == centreDetectors[1]):
            cent_spec_SSW[0,]      =     spec[k].data.wave 
            cent_spec_SSW[1,]      =     spec[k].data.flux 
            cent_spec_SSW[2,]      = spec_ori[k].data.error/np.sqrt(apod_width/1.2)

    ## -------------define different Gaussian Kernels 
    g    = Gaussian1DKernel(stddev=width_of_sm_kernel)


## --------------- make random frequency of sinc functions and add in 

    p0                = 0.8       # peak 
    p1                = frand #800      # central freq
    p2                = 0.37733*2  # Delta sigma / pi; FWHM = 1.20671 * \Delta sigma   
    x                 = cent_spec_SLW[0,]
    sinconly          = p0 * np.sinc(( x - p1) / p2)
    spec_n_sinc       = sinconly + cent_spec_SLW[1,] 
    cent_spec_SLW[1,] = spec_n_sinc

#------------------------------------------------------------
# directly derive the local baseline using the spectra themselves, without dark sky subtraction 
## -  mask/flag the channels with signals 
#------------------------------------------------------------
    cent_SLW_nan                      =  copy.copy(cent_spec_SLW[1]) 
    for freq_obs, name in freq_list:
         cent_SLW_nan[np.where(np.abs(cent_spec_SLW[0] - freq_obs) < window/2 )] = np.nan

    cent_SSW_nan                      =  copy.copy(cent_spec_SSW[1]) 
    for freq_obs, name in freq_list:
         cent_SSW_nan[np.where(np.abs(cent_spec_SSW[0] - freq_obs) < window/2  )] = np.nan

#------------------------------------------------------------


## ----- convolve the central spectra (after blanking) to lower resolutions to derive the 'local' baseline shape. 
    cent_SLW_nan_sm                =  convolve(cent_SLW_nan, g , boundary='extend')
    cent_SLW_subtract_self_base    =  cent_spec_SLW[1] - cent_SLW_nan_sm 
    cent_SSW_nan_sm                =  convolve(cent_SSW_nan, g , boundary='extend')
    cent_SSW_subtract_self_base    =  cent_spec_SSW[1] - cent_SSW_nan_sm 
    cent_SLW_err                   =  cent_spec_SLW[2,]
    cent_SSW_err                   =  cent_spec_SSW[2,]


    edge_cut          = 0 #120
    edge_SLW_low_cut  = 0 #30
    edge_SLW_high_cut = 0 #180
    edge_SSW_low_cut  = 0 #30
    edge_SSW_high_cut = 0 #30
    SLW_size         = len(cent_spec_SLW[0])
    SSW_size         = len(cent_spec_SSW[0]) 

    cent_SLW_subtract_self_base =  copy.copy(cent_SLW_subtract_self_base[edge_SLW_low_cut:SLW_size-edge_SLW_high_cut])
    cent_SSW_subtract_self_base =  copy.copy(cent_SSW_subtract_self_base[edge_SSW_low_cut:SSW_size-edge_SSW_high_cut])
    cent_SLW_wave               =  copy.copy(cent_spec_SLW[0][edge_SLW_low_cut:SLW_size-edge_SLW_high_cut])
    cent_SSW_wave               =  copy.copy(cent_spec_SSW[0][edge_SSW_low_cut:SSW_size-edge_SSW_high_cut])






    plt.clf()
    fig, ax_f = plt.subplots()
    ax_f.plot(cent_spec_SLW[0], sinconly ,       label='SLW'   , linewidth=0.1 )
#   ax_f.plot(cent_spec_SSW[0], spec_n_sinc ,    label='SSW'   , linewidth=0.1 )
    ymin, ymax = ax_f.get_ylim()

    ax_f.set_xlim(400,1600) 
    ax_f.annotate(Name, (1200, ymax*0.9 ), size=12) 
    ax_f.annotate('obsid '+str(FTSobsid), (1200, ymax*0.8 ), size=12) 
    ax_f.text(250, 0.2, "Flux density (Jy)",rotation=90,size='12')
    ax_f.text(800, ymin-0.2, "Frequency (GHz)",size='12')

    textlines(ax_f,redshift,0.3,freq_list)
    plt.legend( loc=2, borderaxespad=0.)
    plt.savefig('plots/'+str(FTSobsid)+'_sinc_only.pdf')
#-------------- plot the baseline and original spectra ----------------






#-------------- plot the baseline and original spectra ----------------
    plt.clf()
    fig, ax_f = plt.subplots()
    ax_f.plot(cent_spec_SLW[0], cent_spec_SLW[1],    label='SLW'   , linewidth=0.1 )
    ax_f.plot(cent_spec_SSW[0], cent_spec_SSW[1],    label='SSW'   , linewidth=0.1 )
    ax_f.plot(cent_spec_SLW[0], cent_SLW_nan_sm,     label='BL SLW', linewidth=0.1 )
    ax_f.plot(cent_spec_SSW[0], cent_SSW_nan_sm,     label='BL SSW', linewidth=0.1 )
    print(cent_SLW_nan_sm)
    ymin, ymax = ax_f.get_ylim()

#   ax_f.set_ylim(-0.5,0.7) 
    ax_f.set_xlim(400,1600) 
    ax_f.annotate(Name, (1200, ymax*0.9 ), size=12) 
    ax_f.annotate('obsid '+str(FTSobsid), (1200, ymax*0.8 ), size=12) 

    ax_f.text(250, 0.2, "Flux density (Jy)",rotation=90,size='12')
    ax_f.text(800, ymin-0.2, "Frequency (GHz)",size='12')

    textlines(ax_f,redshift,0.3,freq_list)
    plt.legend( loc=2, borderaxespad=0.)
    plt.savefig('plots/'+str(FTSobsid)+'_subtracted.pdf')
#-------------- plot the baseline and original spectra ----------------



#-------------- plot the baseline subtracted spectra ----------------
    plt.clf()
    fig, ax_f = plt.subplots()
    ax_f.plot(cent_SLW_wave, cent_SLW_subtract_self_base,     label='SLW', linewidth=0.1 )
#   ax_f.plot(cent_spec_SLW[0], cent_SLW_err,                    label='SLW err')
#   ax_f.plot(cent_spec_SLW[0], cent_SLW_err*(-1))
    ax_f.plot(cent_SSW_wave, cent_SSW_subtract_self_base,     label='SSW', linewidth=0.1 )
#   ax_f.plot(cent_spec_SSW[0], cent_SSW_subtract_self_base* 0,  label=str(Name))
#   ax_f.plot(cent_spec_SSW[0], cent_SLW_err,                    label='SSW err')
#   ax_f.plot(cent_spec_SSW[0], cent_SLW_err*(-1))

    ax_f.set_ylim(-0.5,0.7) 
    ax_f.set_xlim(400,1600) 
    ax_f.annotate(Name, ( 1400, 0.6 ), size=12) 
    ax_f.text(250, 0.2, "Flux density (Jy)",rotation=90,size='12')
    ax_f.text(800, -0.6, "Frequency (GHz)",size='12')

    textlines(ax_f,redshift,0.3,freq_list)
    plt.legend( loc=2, borderaxespad=0.)
    plt.savefig('plots/all'+str(FTSobsid)+'_subtracted.pdf')

    ax_f.set_ylim(-0.5,0.7)
    ax_f.set_xlim(freq_Cp -20, freq_Cp +20)
#   ax_f.set_xlim(freq_Oiii88-20, freq_Oiii88+20)
    textlines(ax_f,redshift,0.3,freq_list)
    plt.legend( loc=2, borderaxespad=0.)
    plt.savefig('plots/cplus'+str(FTSobsid)+'_subtracted.pdf')
    ax_f.set_xlim(freq_Oiii88-20, freq_Oiii88+20)
    plt.savefig('plots/oiii88'+str(FTSobsid)+'_subtracted.pdf')

#-----------------------------------------------------------------



#----------------- output fits files --------------------
# SLW 
    hdu = fits.PrimaryHDU()
    table_hdu = fits.new_table(fits.ColDefs(
         [fits.Column(name='wave',  format='D', unit='um', array=cent_spec_SLW[0]), 
          fits.Column(name='flux',  format='D', unit='Jy', array=cent_SLW_subtract_self_base),
          fits.Column(name='error', format='D', unit='Jy', array=cent_spec_SLW[2,]),
          ]))
    hdulist = fits.HDUList([hdu, table_hdu])
    hdulist.writeto('baselined/'+str(FTSobsid)+'_HR_spectrum_point_SLW_apod_baselined.fits', clobber=True)
    # SSW 
    hdu = fits.PrimaryHDU()
    table_hdu = fits.new_table(fits.ColDefs(
         [fits.Column(name='wave',  format='D', unit='um', array=cent_spec_SSW[0]), 
          fits.Column(name='flux',  format='D', unit='Jy', array=cent_SSW_subtract_self_base),
          fits.Column(name='error', format='D', unit='Jy', array=cent_spec_SSW[2,]),
          ]))
    hdulist = fits.HDUList([hdu, table_hdu])
    hdulist.writeto('baselined/'+str(FTSobsid)+'_HR_spectrum_point_SSW_apod_baselined.fits', clobber=True)
#   return

#----------------- output fits files --------------------



#all_sources = ascii.read('targets_updated_oct1.cat')
#all_sources = all_sources[all_sources['FTSobsid'] != 'N/A']
#for i in range(0, len(all_sources)):
#        Name = all_sources['Name'][i]
#        subtract_self(all_sources['FTSobsid'][i], all_sources['zp'][i], 7, 1.5, Name)
#       #subtract_self(FTSobsid,   redshift,  width_of_sm_kernel,apod_width, Name) 

frand    = uniform(400, 1000)
FTSobsid = 1342238709
z        = 1.325
Name     = 'HBootes03'
window   = 6
subtract_self(frand, FTSobsid, z, 13, 1.5, Name, window)

