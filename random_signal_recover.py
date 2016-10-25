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
from   astropy.modeling import models, fitting
from   astropy.modeling.models import Gaussian1D
from scipy.stats import norm



def fit_line_flux(filename, line_freq, sub_channels, fix_x):
    spec           = fits.open(filename)
    xaxis          = spec[1].data.wave

    flux           = spec[1].data.flux
    if (line_freq > np.max(xaxis) or line_freq < np.min(xaxis)):
        print("line is not covered")
        return
    print("now line is covered")
    line_pixel     = np.where(abs(xaxis- line_freq) == abs(xaxis- line_freq).min())[0][0]
    # Find the position of the desired line (element number)
    xaxis          = xaxis[line_pixel - sub_channels // 2 : line_pixel + sub_channels // 2]
    flux           =  flux[line_pixel - sub_channels // 2 : line_pixel + sub_channels // 2]

    # Fit the data using a Gaussian
    g_init   = models.Gaussian1D(amplitude=0.1, mean=line_freq, stddev=0.1)   ##
    g_init.fixed['mean'] = fix_x       ## here to fix parameters  or not
    fit_g     =  fitting.LevMarLSQFitter()
    g         =  fit_g(g_init, xaxis, flux)
    a,b,c     =  g.parameters
    # a -- amplitude b -- mean c -- stddev
    # http://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Gaussian1D.html
    if fix_x == False:
        peak_err, centre_err, width_err = fit_g.fit_info['param_cov'].diagonal()**0.5
    else:
        print(fit_g.fit_info)
        if fit_g.fit_info['param_cov'].diagonal() is not None:
            peak_err, width_err = fit_g.fit_info['param_cov'].diagonal()**0.5
        else:
            peak_err, width_err = 100, 100

    peak_err2   =  np.std(flux-g(xaxis))
    # Area calculation : http://physics.stackexchange.com/questions/196957/error-propagation-with-an-integral
    area      =  np.sqrt(2) * a * abs(c) * np.sqrt(np.pi) *1e-26 * 1e9 * u.W / u.m**2
    area_err  =  np.sqrt(np.pi * (2*c**2*peak_err**2 + a**2*width_err**2)) * 1e-26 * 1e9 * u.W / u.m**2


    print("peak       :", g.parameters[0], "Jy" )
    print("peak error1:", peak_err , "Jy" )
    print("peak error2:", peak_err2 , "Jy" )
    print("peak S/N1  :", g.parameters[0]/peak_err  )
    print("peak S/N2  :", g.parameters[0]/peak_err2 )
    print("area       :", area )
    print("area error :", area_err)
    print("area S/N   :", "%8.4f"%(area/area_err))
    return a        #,area.value, 


# --------- line plotting  
#     -first, plot the spectrum with the lower axis in GHz 
#    define the figure and the lower axis 
    fig, ax_lower = plt.subplots()

# ---  plot xaxis with flux 
    ax_lower.plot(xaxis, flux)
# ---  plot axis with Gaussian fitting results 
    ax_lower.plot(xaxis, g(xaxis),color='green')
# --- set xlimit for the frequency xaxis in GHz 
    ax_lower.set_xlim(np.min(xaxis), np.max(xaxis))

    #-- copy the X-axis to the top
    ax_upper      = ax_lower.twiny()
    # -- define function for the conversion
    def tick_function(X):
        V = 1 / X
        return ["%.3f" % z for z in V]

    #  --    Calculate the wavelength ticks with  interval in um (from the max to the min with -5 interval in um)
    wavelength          = np.arange(int(np.max(constants.c/xaxis/1e3)), int(np.min(constants.c/xaxis/1e3)),-5)
    #  --    Calculate the frequencies corresponding to the wavelength ticks
    frequency           = constants.c / wavelength/ 1E3
    #  --    Calculate the location of tick in the coordinate from 0 to 1 (relative X coordinate): (0 .. x1 .. X2 .. 1)
    new_tick_locations  = (frequency - np.min(xaxis) ) / (np.max(xaxis)-np.min(xaxis))
    #   Move twinned axis ticks and label from top to bottom
    ax_upper.xaxis.set_ticks_position("top")
    ax_upper.xaxis.set_label_position("top")
    ax_upper.spines["top"].set_position(("axes", 1.))
    #   Turn on the frame for the twin axis, but then hide all but the bottom spine
    ax_upper.set_frame_on(True)
    ax_upper.patch.set_visible(False)

    for sp in ax_upper.spines.values():
        sp.set_visible(False)

    ax_upper.spines["bottom"].set_visible(True)
    ax_upper.set_xticklabels(tick_function(new_tick_locations) )
    ax_upper.set_xticks(new_tick_locations)
    ax_upper.set_xticklabels(wavelength.astype(str))
    ax_upper.set_xlabel(r"Rest-frame Wavelength ($\mu$m)")
    ax_lower.set_xlabel(r"Rest-frame Frequency  (GHz) ")
    plt.savefig('plots/random_fitting/'+str(line_freq)+'plot.pdf')
    return





def textlines(ax_f,z,y,freq_list ):
    for freq, name in freq_list: 
        plt.axvline(x=freq,alpha=0.2,linewidth=4,drawstyle='steps-mid') 
        ax_f.text(freq, y, name,rotation=90,size='6')
    return 



def subtract_self(frand, index, FTSobsid,  redshift,  width_of_sm_kernel,apod_width, Name, window):
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

    p0                = 0.3       # peak 
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


#    #-------------- plot the baseline and original spectra ----------------
#        plt.clf()
#        fig, ax_f = plt.subplots()
#        ax_f.plot(cent_spec_SLW[0], sinconly ,       label='SLW'   , linewidth=0.1 )
#    #   ax_f.plot(cent_spec_SSW[0], spec_n_sinc ,    label='SSW'   , linewidth=0.1 )
#        ymin, ymax = ax_f.get_ylim()
#    
#        ax_f.set_xlim(400,1600) 
#        ax_f.annotate(Name, (1200, ymax*0.9 ), size=12) 
#        ax_f.annotate('obsid '+str(FTSobsid), (1200, ymax*0.8 ), size=12) 
#        ax_f.text(250, 0.2, "Flux density (Jy)",rotation=90,size='12')
#        ax_f.text(800, ymin-0.2, "Frequency (GHz)",size='12')
#    
#        textlines(ax_f,redshift,0.3,freq_list)
#        plt.legend( loc=2, borderaxespad=0.)
#        plt.savefig('plots/'+str(FTSobsid)+str(index)+'_sinc_only.pdf')
#    #-------------- plot the baseline and original spectra ----------------


#   #-------------- plot the baseline and original spectra ----------------
#       plt.clf()
#       fig, ax_f = plt.subplots()
#       ax_f.plot(cent_spec_SLW[0], cent_spec_SLW[1],    label='SLW'   , linewidth=0.1 )
#       ax_f.plot(cent_spec_SSW[0], cent_spec_SSW[1],    label='SSW'   , linewidth=0.1 )
#       ax_f.plot(cent_spec_SLW[0], cent_SLW_nan_sm,     label='BL SLW', linewidth=0.1 )
#       ax_f.plot(cent_spec_SSW[0], cent_SSW_nan_sm,     label='BL SSW', linewidth=0.1 )
#       print(cent_SLW_nan_sm)
#       ymin, ymax = ax_f.get_ylim()
#   
#   #   ax_f.set_ylim(-0.5,0.7) 
#       ax_f.set_xlim(400,1600) 
#       ax_f.annotate(Name, (1200, ymax*0.9 ), size=12) 
#       ax_f.annotate('obsid '+str(FTSobsid), (1200, ymax*0.8 ), size=12) 
#   
#       ax_f.text(250, 0.2, "Flux density (Jy)",rotation=90,size='12')
#       ax_f.text(800, ymin-0.2, "Frequency (GHz)",size='12')
#   
#       textlines(ax_f,redshift,0.3,freq_list)
#       plt.legend( loc=2, borderaxespad=0.)
#       plt.savefig('plots/'+str(FTSobsid)+str(index)+'_subtracted.pdf')
#   #-------------- plot the baseline and original spectra ----------------



#  #-------------- plot the baseline subtracted spectra ----------------
#      plt.clf()
#      fig, ax_f = plt.subplots()
#      ax_f.plot(cent_SLW_wave, cent_SLW_subtract_self_base,     label='SLW', linewidth=0.1 )
#  #   ax_f.plot(cent_spec_SLW[0], cent_SLW_err,                    label='SLW err')
#  #   ax_f.plot(cent_spec_SLW[0], cent_SLW_err*(-1))
#      ax_f.plot(cent_SSW_wave, cent_SSW_subtract_self_base,     label='SSW', linewidth=0.1 )
#  #   ax_f.plot(cent_spec_SSW[0], cent_SSW_subtract_self_base* 0,  label=str(Name))
#  #   ax_f.plot(cent_spec_SSW[0], cent_SLW_err,                    label='SSW err')
#  #   ax_f.plot(cent_spec_SSW[0], cent_SLW_err*(-1))
#  
#      ax_f.set_ylim(-0.5,0.7) 
#      ax_f.set_xlim(400,1600) 
#      ax_f.annotate(Name, ( 1400, 0.6 ), size=12) 
#      ax_f.text(250, 0.2, "Flux density (Jy)",rotation=90,size='12')
#      ax_f.text(800, -0.6, "Frequency (GHz)",size='12')
#  
#      ax_f.set_xlim(freq_Oiii88-20, freq_Oiii88+20)
#  
#      textlines(ax_f,redshift,0.3,freq_list)
#      plt.legend( loc=2, borderaxespad=0.)
#      plt.savefig('plots/all'+str(FTSobsid)+str(index)+'_subtracted.pdf')
#  
#      ax_f.set_ylim(-0.5,0.7)
#      plt.legend( loc=2, borderaxespad=0.)
#      textlines(ax_f,redshift,0.3,freq_list)
#  
#      ax_f.set_xlim(freq_Cp -20, freq_Cp +20)
#      plt.savefig('plots/cplus'+str(FTSobsid)+str(index)+'_subtracted.pdf')
#  
#      ax_f.set_xlim(freq_Oiii88-20, freq_Oiii88+20)
#      plt.savefig('plots/oiii88'+str(FTSobsid)+str(index)+'_subtracted.pdf')
#  
#  #-----------------------------------------------------------------



#----------------- output fits files --------------------
# SLW 
    hdu = fits.PrimaryHDU()
    table_hdu = fits.new_table(fits.ColDefs(
         [fits.Column(name='wave',  format='D', unit='GHz', array=cent_spec_SLW[0]), 
          fits.Column(name='flux',  format='D', unit='Jy', array=cent_SLW_subtract_self_base),
          fits.Column(name='error', format='D', unit='Jy', array=cent_spec_SLW[2,]),
          ]))
    hdulist = fits.HDUList([hdu, table_hdu])
    hdulist.writeto('baselined/'+str(FTSobsid)+'_HR_spectrum_point_SLW_apod_baselined.fits', clobber=True)
    # SSW 
    hdu = fits.PrimaryHDU()
    table_hdu = fits.new_table(fits.ColDefs(
         [fits.Column(name='wave',  format='D', unit='GHz', array=cent_spec_SSW[0]), 
          fits.Column(name='flux',  format='D', unit='Jy', array=cent_SSW_subtract_self_base),
          fits.Column(name='error', format='D', unit='Jy', array=cent_spec_SSW[2,]),
          ]))
    hdulist = fits.HDUList([hdu, table_hdu])
    hdulist.writeto('baselined/'+str(FTSobsid)+'_HR_spectrum_point_SSW_apod_baselined.fits', clobber=True)
#   return 

#----------------- output fits files --------------------

fluxes = np.array([]) 
for i in range(0, 1000):
    frand    = uniform(510, 940 )
    FTSobsid = 1342238709
    z        = 1.325
    Name     = 'HBootes03'
    window   = 6
    sub_channels = 100 
    fix_x = True 
    subtract_self(frand, i, FTSobsid, z, 7, 1.5, Name, window)
#   subtract_self(frand, index, FTSobsid,  redshift,  width_of_sm_kernel,apod_width, Name, window):
#              random freq, index, FTSobsid, redshift, width of sm, apod_width,  target name, set window width 
    print(frand)
    flux = fit_line_flux('baselined/'+str(FTSobsid)+'_HR_spectrum_point_SSW_apod_baselined.fits', frand , sub_channels, fix_x) 
    if flux is not None:
        print('test OK')
        fluxes = np.append(fluxes,flux)
    flux = fit_line_flux('baselined/'+str(FTSobsid)+'_HR_spectrum_point_SLW_apod_baselined.fits', frand , sub_channels, fix_x)  
    if flux is not None:
        print('test OK')
        fluxes = np.append(fluxes,flux)
#                  

# -------- mask data larger than 0.8 which are outerliers
fluxes           = fluxes[np.where(fluxes < 0.8)]

plt.clf()
f, ax1           = plt.subplots(1, sharex=True, sharey=True)
bins             = np.linspace(0, 1, 40)
x                = (bins+0.015)[0:(40-1)] # calculate the x-axis (each point is in the middle of the bin, instead of the starting point of the bin)
g_init           = models.Gaussian1D(amplitude=1, mean=0.3, stddev=0.1)   ## initialise parameters for the Gaussian fitting
fit_g            = fitting.LevMarLSQFitter()
g                = fit_g(g_init, x, n)
n, bins, patches = ax1.hist(fluxes, bins, normed=1, facecolor='green', alpha=0.75)
l                = ax1.plot(bins, g(bins), 'r--', linewidth=2,label='user fitting')
ax1.legend(loc=7, borderaxespad=0.)
plt.savefig('test_hist_Gaussian_fit.pdf')

# - make statistics p-value test again to the masked fluxes array
# - KS test D-statistics table:  www.mathematik.uni-kl.de/~schwaar/Exercises/Tabellen/table_kolmogorov.pdf
# - p value less than 0.1, can not assume normal distribution

ptest  = scipy.stats.mstats.normaltest(fluxes)
kstest = stats.kstest(fluxes, 'norm')

print(ptest)
print(kstest)


