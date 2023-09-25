#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:56:25 2022

@author: pblack
"""

DATA_PATH = '/mirror/scratch/pblack'


import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import numpy as np
from astropy.time import Time 
from astropy.time import TimeDelta
import datetime
import time
import os
from math import nan
import gc

os.chdir('/scratch/nas_lbass/raw_data/')

# GOOD FREQ ARE 112-568  BUT remember channels start at 3 in a1p1 etc
# A1P1 - P(l,pi)
# A2P2 - P(r,0)
# A1P2 - P(l,0)
# A2P1 - P(r,pi)

#load background files and data tables
user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
frequency = np.load(DATA_PATH+'/temp/freq.npy') #its in hertz
#frequency = frequency * (10**-9)


####################################################################################

def corrections (a1p1,a1p2,a2p1,a2p2): 

    parameters = np.load(DATA_PATH+'/temp/parameters.npy')
    profiles = np.load(DATA_PATH+'/bp_profiles/profiles.npy')
    if parameters[1] == 'True':
        flatten = True
    else:
        flatten = False
    np.save(DATA_PATH+'/temp/flatten.npy', flatten)   

    # A1P1 - P(l,pi)
    # A1P2 - P(l,0)
    # A2P1 - P(r,pi)
    # A2P2 - P(r,0)
    #power_ratios = [1.03,1.03,1,1] #Peter specified corrections - 21/12/2022
    power_ratios = [1,1,1,1] #no correction
    power_ratios = np.load(DATA_PATH+'/bp_profiles/PRC.npy')
  #  power_ratios = [1.03,1.0309,1,1.03] #Pips attempt at corrections - 21/12/2022

    if flatten:
    
        for i in range (0, np.size(profiles[:,0])):
            try:
                if profiles[i,8] == 'True':
                    print('')
                    print('\033[0;0m Normalising against bandpass profile:',profiles[i,0],' \033[0;32m')
                    bandpass_norms = [profiles[i,4],profiles[i,5],profiles[i,6],profiles[i,7]]
                else:
                    pass
            except:
                print('')
                print('\033[1;31m Bandpass profile failed to load. \033[1;32m')

    if flatten:
                normload = np.load(bandpass_norms[0]+'.npy')
                a1p1 = a1p1 / normload #flatten bandpass
                normload = np.load(bandpass_norms[1]+'.npy')
                a1p2 = a1p2 / normload #flatten bandpass
                normload = np.load(bandpass_norms[2]+'.npy')
                a2p1 = a2p1 / normload #flatten bandpass
                normload = np.load(bandpass_norms[3]+'.npy')
                a2p2 = a2p2 / normload #flatten bandpass
  
    if power_ratios[0] == 1 and power_ratios[1] == 1 and power_ratios[2] == 1 and power_ratios[3] == 1:
        pass
    else:
        print('\033[0;0m Applying input power ratio corrections. \033[1;32m')
        a1p1 = a1p1 * power_ratios[0] #adjust power ratios
        a1p2 = a1p2 * power_ratios[1] #adjust power ratios
        a2p1 = a2p1 * power_ratios[2] #adjust power ratios
        a2p2 = a2p2 * power_ratios[3] #adjust power ratios

    return a1p1, a1p2, a2p1, a2p2



################################################################################

def get_minmax(band):

    ind = np.argmin(band[:,112:569])
    indx = np.argmax(band[:,112:569])
    band = band[:,112:569].flatten()
    bandmin = band[ind] 
    bandmax = band[indx]   

    return bandmin, bandmax

#--------------------------------------------------------------------

def waterfallPERmin(duration_actual,save_it=False, first_loop=True):

    flatten = np.load(DATA_PATH+'/temp/flatten.npy')
    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    del p11

    cmap = np.load(DATA_PATH+'/temp/cmap.npy')
    if cmap:
        cstr='viridis'
    else:
       # cstr='gist_heat' 
        cstr='turbo' 

    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')

    a1p1B = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
    mins,first_tick, sample_to_datetime, frequency = yaxis(a1p1B, duration_actual)
    del a1p1B
    frequency = frequency[112:569]

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False


    if parameters[8]=='True' or parameters[1]=='True':
        if parameters[1]=='True' and parameters[8]=='True':
            title_string = 'Spectrogram, All Inputs\n'+user_inputs[0]+' Global Normalised, Normalised'
        elif parameters[1]=='True':
            title_string =  'Spectrogram, All Inputs\n'+user_inputs[0]+' Global Normalised'
        else:  
            title_string = 'Spectrogram, All Inputs\n'+user_inputs[0]+' Normalised'

    else:
        title_string = 'Spectrogram, All Inputs\n'+user_inputs[0]

    if save_it:
        f, axarr = plt.subplots(2,2, figsize=(12,8), dpi=300)
    else:
        f, axarr = plt.subplots(2,2)
       
    plt.setp(axarr, yticks=first_tick, yticklabels=mins)
    
  #  if duration_actual > 12 and duration_actual < 48:

   #     plt.setp(axarr, yticks=np.arange(first_tick[0],sample_to_datetime,60))
    #elif duration_actual > 48:
     #   pass
    #else:
     #   plt.setp(axarr, yticks=np.arange(first_tick[0],sample_to_datetime,15))
    
    #for i in range (0,np.size(frequency)):
        

    plt.setp(axarr, xticks=[0,45,91,136,182,227,273,318,364,409,456], xticklabels=['1400','','1405','','1410','','1415','','1420','','1425'])
    plt.setp(axarr, xlabel='Frequency / MHz', ylabel='Time')
    
    band11 = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
    band12 = np.load (DATA_PATH+'/temp/a1p2_bandpass.npy')
    band21 = np.load (DATA_PATH+'/temp/a2p1_bandpass.npy')
    band22 = np.load (DATA_PATH+'/temp/a2p2_bandpass.npy')

    band11,band12,band21,band22 = corrections(band11,band12,band21,band22)

    if parameters[8]=='True':
        band11 = band11/np.mean(band11[:,112:569])
        band12 = band12/np.mean(band12[:,112:569])
        band21 = band21/np.mean(band21[:,112:569])
        band22 = band22/np.mean(band22[:,112:569])


    Pmax = []  #find the minimum and maximum values to use across all 4 plots
    Pmin = []
    bandmin, bandmax = get_minmax(band11)
    Pmax.append(bandmax)
    Pmin.append(bandmin)
    bandmin, bandmax = get_minmax(band12)
    Pmax.append(bandmax)
    Pmin.append(bandmin)
    bandmin, bandmax = get_minmax(band21)
    Pmax.append(bandmax)
    Pmin.append(bandmin)
    bandmin, bandmax = get_minmax(band22)
    Pmax.append(bandmax)
    Pmin.append(bandmin)
    Pmax = np.asarray(Pmax)
    Pmin = np.asarray(Pmin)
    Lmin = np.argmin(Pmin)
    Pmin = Pmin[Lmin]
    Lmax = np.argmax(Pmax)
    Pmax = Pmax[Lmax]
    
    mask_width = parameters[4].astype(np.float)
    mask_width = 1+int(mask_width/2)
    if parameters[6] == 'True':
        axarr[0,1].axvline(x=(371),linestyle='--', color='k', linewidth=1, label='1420.405 MHz')
        axarr[0,0].axvline(x=(371),linestyle='--', color='k', linewidth=1, label='1420.405 MHz')
        axarr[1,1].axvline(x=(371),linestyle='--', color='k', linewidth=1, label='1420.405 MHz')
        axarr[1,0].axvline(x=(371),linestyle='--', color='k', linewidth=1, label='1420.405 MHz')
    
    if parameters[2]=='True':
        band11[:,(484-mask_width):(483+mask_width+1)] = 'nan' #21cm
    if duration_actual < 8:
        axarr[0,1].imshow(band11[:,112:569], cmap=cstr,vmin=Pmin,vmax=Pmax,aspect='auto', interpolation='none')#112:569
        d1 = axarr[0,1].imshow(band11[:,112:569], cmap=cstr,vmin=Pmin,vmax=Pmax, aspect='auto', interpolation='none')
    else:
        axarr[0,1].imshow(band11[:,112:569], cmap=cstr,vmin=Pmin,vmax=Pmax, interpolation='none')
        d1 = axarr[0,1].imshow(band11[:,112:569], cmap=cstr,vmin=Pmin,vmax=Pmax, interpolation='none')
    axarr[0,1].set_title('P(L,$\pi$) W')
    f.colorbar(d1, ax=axarr[0, 1])
    del band11
       
    if parameters[2]=='True':
        band12[:,(484-mask_width):(483+mask_width)+1] = 'nan' #21cm
    if duration_actual < 8:
        axarr[0,0].imshow(band12[:,112:569], cmap=cstr,vmin=Pmin,vmax=Pmax, aspect='auto', interpolation='none')
        d2 = axarr[0,0].imshow(band12[:,112:569], cmap=cstr,vmin=Pmin,vmax=Pmax, aspect='auto', interpolation='none')
    else:
        axarr[0,0].imshow(band12[:,112:569], cmap=cstr,vmin=Pmin,vmax=Pmax, interpolation='none')
        d2 = axarr[0,0].imshow(band12[:,112:569], cmap=cstr,vmin=Pmin,vmax=Pmax, interpolation='none')
    axarr[0,0].set_title('P(L,0) E')
    f.colorbar(d2, ax=axarr[0, 0])
    del band12
    
    if parameters[2]=='True':
        band21[:,(484-mask_width):(483+mask_width)+1] = 'nan' #21cm
    if duration_actual < 8:
        axarr[1,0].imshow(band21[:,112:569], cmap=cstr,vmin=Pmin,vmax=Pmax,aspect='auto', interpolation='none')
        d3 = axarr[1,0].imshow(band21[:,112:569], cmap=cstr,vmin=Pmin,vmax=Pmax, aspect='auto', interpolation='none')
    else:
        axarr[1,0].imshow(band21[:,112:569], cmap=cstr,vmin=Pmin,vmax=Pmax, interpolation='none')
        d3 = axarr[1,0].imshow(band21[:,112:569], cmap=cstr,vmin=Pmin,vmax=Pmax, interpolation='none')
    axarr[1,0].set_title('P(R,$\pi$) E')
    f.colorbar(d3, ax=axarr[1, 0])
    del band21

    if parameters[2]=='True':
        band22[:,(484-mask_width):(483+mask_width)+1] = 'nan' #21cm
    if duration_actual < 8:
        axarr[1,1].imshow(band22[:,112:569], cmap=cstr,vmin=Pmin,vmax=Pmax,aspect='auto', interpolation='none')
        d4 = axarr[1,1].imshow(band22[:,112:569], cmap=cstr,vmin=Pmin,vmax=Pmax, aspect='auto', interpolation='none')
    else:
        axarr[1,1].imshow(band22[:,112:569], cmap=cstr,vmin=Pmin,vmax=Pmax, interpolation='none')
        d4 = axarr[1,1].imshow(band22[:,112:569], cmap=cstr,vmin=Pmin,vmax=Pmax, interpolation='none')
    axarr[1,1].set_title('P(R,0) W')
    f.colorbar(d4, ax=axarr[1, 1])
    del band22
    plt.suptitle(title_string)
    plt.tight_layout()
 
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_4xWF.png', bbox_inches="tight")
        print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
    else:
        plt.show()
    plt.close()
    
    if first_loop:
        print ('')
        save = input ('Do you want to save a printer friendly copy of this plot? (Y/N): \033[0;m')
        if str(save) == 'N' or str(save) == 'n':
            print('\033[1;32m')
            pass
        elif str(save) == 'Y' or str(save) == 'y':
            save_it=True
            first_loop = False
            waterfallPERmin(duration_actual,save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass
    
#--------------------------------------------------------------------

def waterfallSINGLE(duration_actual,save_it=False, first_loop=True):

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    flatten = np.load(DATA_PATH+'/temp/flatten.npy')
    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    del p11

    cmap = np.load(DATA_PATH+'/temp/cmap.npy')
    if cmap:
        cstr='viridis'
    else:
        #cstr='gist_heat'
        cstr='turbo' 

    a1p1B = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
    mins,first_tick, sample_to_datetime, frequency = yaxis(a1p1B, duration_actual)

    a1p2B = np.load (DATA_PATH+'/temp/a1p2_bandpass.npy')
    a2p1B = np.load (DATA_PATH+'/temp/a2p1_bandpass.npy')
    a2p2B = np.load (DATA_PATH+'/temp/a2p2_bandpass.npy')

    a1p1B,a1p2B,a2p1B,a2p2B = corrections(a1p1B,a1p2B,a2p1B,a2p2B)

    a1p1B = a1p1B[:,112:569]
    a1p2B = a1p2B[:,112:569]
    a2p1B = a2p1B[:,112:569]
    a2p2B = a2p2B[:,112:569]

    if parameters[8] == 'True':
        a1p1B = a1p1B/np.mean(a1p1B)
        a1p2B = a1p2B/np.mean(a1p2B)
        a2p1B = a2p1B/np.mean(a2p1B)
        a2p2B = a2p2B/np.mean(a2p2B)

    d_a = a1p1B - a2p1B # l0 - lpi 
    d_b = a1p2B - a2p2B #r0 - rpi 
    d_c = a1p2B - a1p1B  # l0 - r0 
    d_d = a2p2B - a2p1B  #   lpi - rpi 

    del a1p2B, a2p1B, a2p2B, a1p1B

    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')

    if parameters[8]=='True' or parameters[1]=='True':
        if parameters[1]=='True' and parameters[8]=='True':
            title_string = 'Spectrogram, Single Differences\n'+user_inputs[0]+' Global Normalised, Normalised'
        elif parameters[1]=='True':
            title_string =  'Spectrogram, Single Differences\n'+user_inputs[0]+' Global Normalised'
        else:  
            title_string = 'Spectrogram, Single Differences\n'+user_inputs[0]+' Normalised'

    else:
        title_string = 'Spectrogram, Single Differences\n'+user_inputs[0]

    if save_it:
        f, axarr = plt.subplots(2,2, figsize=(12,8), dpi=300)
    else:
        f, axarr = plt.subplots(2,2)
       
    plt.setp(axarr, yticks=first_tick, yticklabels=mins)
    
 #   if duration_actual > 12 and duration_actual < 48:
  #      plt.setp(axarr, yticks=np.arange(first_tick[0],sample_to_datetime,60))
   # elif duration_actual > 48:
    #    pass
  #  else:
   #     plt.setp(axarr, yticks=np.arange(first_tick[0],sample_to_datetime,15))
    
    plt.setp(axarr, xticks=[0,45,91,136,182,227,273,318,364,409,456], xticklabels=['1400','','1405','','1410','','1415','','1420','','1425'])
    plt.setp(axarr, xlabel='Frequency / MHz', ylabel='Time')

 
    if duration_actual < 8:
        axarr[1,1].imshow(d_d, cmap=cstr, aspect='auto', interpolation='none')
        d4 = axarr[1,1].imshow(d_d, cmap=cstr, aspect='auto', interpolation='none')
    else:
        axarr[1,1].imshow(d_d, cmap=cstr, interpolation='none')
        d4 = axarr[1,1].imshow(d_d, cmap=cstr, interpolation='none')
    axarr[1,1].set_title('$\delta$d = P(L,$\pi$) W - P(R,$\pi$) E')
    f.colorbar(d4, ax=axarr[1, 1])
    del d_d
       
    if duration_actual <8:
        axarr[1,0].imshow(d_c, cmap=cstr,aspect='auto', interpolation='none')
        d3 = axarr[1,1].imshow(d_c, cmap=cstr,  aspect='auto', interpolation='none')
    else:
        axarr[1,0].imshow(d_c, cmap=cstr, interpolation='none')
        d3 = axarr[1,1].imshow(d_c, cmap=cstr, interpolation='none')
    axarr[1,0].set_title('$\delta$c = P(L,0) E - P(R,0) W')
    f.colorbar(d3, ax=axarr[1, 0])
    del d_c
   

    if duration_actual <8:
        axarr[0,0].imshow(d_a, cmap=cstr,aspect='auto', interpolation='none')
        d1=axarr[0,0].imshow(d_a, cmap=cstr, aspect='auto', interpolation='none')
    else:
        axarr[0,0].imshow(d_a, cmap=cstr, interpolation='none')
        d1=axarr[0,0].imshow(d_a, cmap=cstr, interpolation='none')
    axarr[0,0].set_title('$\delta$a = P(L,0) E - P(L,$\pi$) W')
    f.colorbar(d1, ax=axarr[0, 0])
    del d_a
   
    if duration_actual <8:
        axarr[0,1].imshow(d_b, cmap=cstr, aspect='auto', interpolation='none')
        d2 = axarr[0,1].imshow(d_b, cmap=cstr, aspect='auto', interpolation='none')
    else:
        axarr[0,1].imshow(d_b, cmap=cstr, interpolation='none')
        d2 = axarr[0,1].imshow(d_b, cmap=cstr, interpolation='none')
    axarr[0,1].set_title('$\delta$b = P(R,0) W - P(R,$\pi$) E')
    f.colorbar(d2, ax=axarr[0, 1])
    del d_b
    
    
    plt.suptitle(title_string)
    plt.tight_layout()
  
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_4xWFSD.png', bbox_inches="tight")
        print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
    else:
        plt.show()
    plt.close()
    
    if first_loop:
        print ('')
        save = input ('Do you want to save a printer friendly copy of this plot? (Y/N): \033[0;m')
        if str(save) == 'N' or str(save) == 'n':
            print('\033[1;32m')
            pass
        elif str(save) == 'Y' or str(save) == 'y':
            save_it=True
            first_loop = False
            waterfallSINGLE(duration_actual,save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass
    

#-----------------------------------------------------------------------

def waterfallDOUBLE(duration_actual,save_it=False, first_loop=True):

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    flatten = np.load(DATA_PATH+'/temp/flatten.npy')
    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    del p11

    cmap = np.load(DATA_PATH+'/temp/cmap.npy')
    if cmap:
        cstr='viridis'
    else:
        #cstr='gist_heat'
        cstr='turbo' 

    a1p1B = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
    mins,first_tick, sample_to_datetime, frequency = yaxis(a1p1B, duration_actual)

    a1p2B = np.load (DATA_PATH+'/temp/a1p2_bandpass.npy')
    a2p1B = np.load (DATA_PATH+'/temp/a2p1_bandpass.npy')
    a2p2B = np.load (DATA_PATH+'/temp/a2p2_bandpass.npy')

    a1p1B,a1p2B,a2p1B,a2p2B = corrections(a1p1B,a1p2B,a2p1B,a2p2B)

    a1p1B = a1p1B[:,112:569]
    a1p2B = a1p2B[:,112:569]
    a2p1B = a2p1B[:,112:569]
    a2p2B = a2p2B[:,112:569]

    if parameters[8] == 'True':
        a1p1B = a1p1B/np.mean(a1p1B)
        a1p2B = a1p2B/np.mean(a1p2B)
        a2p1B = a2p1B/np.mean(a2p1B)
        a2p2B = a2p2B/np.mean(a2p2B)

    d_a = a1p1B - a2p1B # l0 - lpi 
    d_b = a1p2B - a2p2B #r0 - rpi 
    d_c = a1p2B - a1p1B  # l0 - r0 
    d_d = a2p2B - a2p1B  #   lpi - rpi 

    DD = (d_a - d_b)/2 # WMAP (ocra would be c-d)

    del d_a, d_b, d_c, d_d
    del a1p2B, a2p1B, a2p2B, a1p1B
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    if parameters[8]=='True' or parameters[1]=='True':
        if parameters[1]=='True' and parameters[8]=='True':
            title_string = 'Spectrogram, Double Difference\n'+user_inputs[0]+' Global Normalised, Normalised'
        elif parameters[1]=='True':
            title_string =  'Spectrogram, Double Difference\n'+user_inputs[0]+' Global Normalised'
        else:  
            title_string = 'Spectrogram, Double Difference\n'+user_inputs[0]+' Normalised'

    else:
        title_string = 'Spectrogram, Double Difference\n'+user_inputs[0]

    if save_it:
        plt.figure(figsize=(12, 8), dpi=300)
          
    plt.yticks(ticks=first_tick, labels=mins)
    plt.xticks(ticks=[0,45,91,136,182,227,273,318,364,409,456], labels=['1400','','1405','','1410','','1415','','1420','','1425'])

    if duration_actual <8:
        plt.imshow(DD, cmap=cstr,aspect='auto', interpolation='none') 
    else:
        plt.imshow(DD, cmap=cstr, interpolation='none') 
    plt.colorbar()
    plt.xlabel('Frequency / MHz')
    plt.ylabel('Time')

    plt.title(title_string)
    plt.tight_layout()
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_4xWFDD.png', bbox_inches="tight")
        print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
    else:
        plt.show()
    plt.close()
    
    if first_loop:
        print ('')
        save = input ('Do you want to save a printer friendly copy of this plot? (Y/N): \033[0;m')
        if str(save) == 'N' or str(save) == 'n':
            print('\033[1;32m')
            pass
        elif str(save) == 'Y' or str(save) == 'y':
            save_it=True
            first_loop = False
            waterfallDOUBLE(duration_actual,save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass

#----------------------------------------------------------------------------

def waterfallNULLS(duration_actual,save_it=False, first_loop=True):

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    flatten = np.load(DATA_PATH+'/temp/flatten.npy')
    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    del p11

    cmap = np.load(DATA_PATH+'/temp/cmap.npy')
    if cmap:
        cstr='viridis'
    else:
        #cstr='gist_heat'
        cstr='turbo' 

    a1p1B = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
    mins,first_tick, sample_to_datetime, frequency = yaxis(a1p1B, duration_actual)

    a1p2B = np.load (DATA_PATH+'/temp/a1p2_bandpass.npy')
    a2p1B = np.load (DATA_PATH+'/temp/a2p1_bandpass.npy')
    a2p2B = np.load (DATA_PATH+'/temp/a2p2_bandpass.npy')

    a1p1B,a1p2B,a2p1B,a2p2B = corrections(a1p1B,a1p2B,a2p1B,a2p2B)

    a1p1B = a1p1B[:,112:569]
    a1p2B = a1p2B[:,112:569]
    a2p1B = a2p1B[:,112:569]
    a2p2B = a2p2B[:,112:569]

    if parameters[8] == 'True':
        a1p1B = a1p1B/np.mean(a1p1B)
        a1p2B = a1p2B/np.mean(a1p2B)
        a2p1B = a2p1B/np.mean(a2p1B)
        a2p2B = a2p2B/np.mean(a2p2B)

    JPL_d1 = a1p1B - a1p2B # lpi - l0 W-E
    JPL_d2 = a2p2B - a2p1B # r0 - rpi W-E
    NULL_IB2_a = a1p2B - a2p1B # l0 - rpi  E-E
    NULL_IB2_b = a2p2B - a1p1B # r0 - lpi   W-W
    NULL_IBX = (NULL_IB2_a + NULL_IB2_b) /2

    NULL_JPL = (JPL_d1 - JPL_d2)/2  
   
    del a1p2B, a2p1B, a2p2B, a1p1B
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    if parameters[8]=='True' or parameters[1]=='True':
        if parameters[1]=='True' and parameters[8]=='True':
            title_string = 'Spectrogram, Nulls\n'+user_inputs[0]+' Global Normalised, Normalised'
        elif parameters[1]=='True':
            title_string =  'Spectrogram, Nulls\n'+user_inputs[0]+' Global Normalised'
        else:  
            title_string = 'Spectrogram, Nulls\n'+user_inputs[0]+' Normalised'

    else:
        title_string = 'Spectrogram, Nulls\n'+user_inputs[0]

    frequency = frequency[112:569]

    if save_it:
        f, axarr = plt.subplots(2,2, figsize=(12,8), dpi=300)
    else:
        f, axarr = plt.subplots(2,2)

    plt.setp(axarr, yticks=first_tick, yticklabels=mins)
    
  #  if duration_actual > 12 and duration_actual < 48:
   #     plt.setp(axarr, yticks=np.arange(first_tick[0],sample_to_datetime,60))
    #elif duration_actual > 48:
     #   pass
   # else:
    #    plt.setp(axarr, yticks=np.arange(first_tick[0],sample_to_datetime,15))
    
    plt.setp(axarr, xticks=[0,45,91,136,182,227,273,318,364,409,456], xticklabels=['1400','','1405','','1410','','1415','','1420','','1425'])
    plt.setp(axarr, xlabel='Frequency / MHz', ylabel='Time')

    if duration_actual <8:
        axarr[0,0].imshow(NULL_JPL, cmap=cstr, aspect='auto', interpolation='none')
        d1 = axarr[0,0].imshow(NULL_JPL, cmap=cstr, aspect='auto', interpolation='none')
    else:
        axarr[0,0].imshow(NULL_JPL, cmap=cstr, interpolation='none')
        d1 = axarr[0,0].imshow(NULL_JPL, cmap=cstr, interpolation='none')
    axarr[0,0].set_title('JPL Null [(L,$\pi$-L,0)-(R,0-R,$\pi$)]/2')
    f.colorbar(d1, ax=axarr[0,0])
       
    if duration_actual <8:
        axarr[0,1].imshow(NULL_IB2_a, cmap=cstr, aspect='auto', interpolation='none')
        d2 = axarr[0,1].imshow(NULL_IB2_a, cmap=cstr, aspect='auto', interpolation='none')
    else:
        axarr[0,1].imshow(NULL_IB2_a, cmap=cstr, interpolation='none')
        d2 = axarr[0,1].imshow(NULL_IB2_a, cmap=cstr, interpolation='none')
    axarr[0,1].set_title('IB Null [(L,0)E - (R,$\pi$)E]')
    f.colorbar(d2, ax=axarr[0,1])
  
    if duration_actual <8:
        axarr[1,1].imshow(NULL_IB2_b, cmap=cstr, aspect='auto', interpolation='none')
        d3 = axarr[1,1].imshow(NULL_IB2_b, cmap=cstr, aspect='auto', interpolation='none')
    else:
       axarr[1,1].imshow(NULL_IB2_b, cmap=cstr, interpolation='none')
       d3 = axarr[1,1].imshow(NULL_IB2_b, cmap=cstr, interpolation='none')
    axarr[1,1].set_title('IB Null [(R,0)W - (L,$\pi$)W]')
    f.colorbar(d3, ax=axarr[1,1])

    if duration_actual <8:
        axarr[1,0].imshow(NULL_IBX, cmap=cstr, aspect='auto', interpolation='none')
        d4 = axarr[1,0].imshow(NULL_IBX, cmap=cstr, aspect='auto', interpolation='none')
    else:
        axarr[1,0].imshow(NULL_IBX, cmap=cstr, interpolation='none')
        d4 = axarr[1,0].imshow(NULL_IBX, cmap=cstr, interpolation='none')
    axarr[1,0].set_title('Null [(L,0-R,$\pi$)+(R,0-L,$\pi$)]/2')
    f.colorbar(d4, ax=axarr[1,0])
    
    plt.suptitle(title_string)
    plt.tight_layout()

    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_WFNULLS.png', bbox_inches="tight")
        print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
    else:
        plt.show()
    plt.close()
    
    if first_loop:
        print ('')
        save = input ('Do you want to save a printer friendly copy of this plot? (Y/N): \033[0;m')
        if str(save) == 'N' or str(save) == 'n':
            print('\033[1;32m')
            pass
        elif str(save) == 'Y' or str(save) == 'y':
            save_it=True
            first_loop = False
            waterfallNULLS(duration_actual,save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass

#-----------------------------------------------------------------------

def yaxis(range_example, duration_actual): #range example being a1p1B etc


    binnable = np.load(DATA_PATH+'/temp/rebinnable.npy')
    if binnable:
        range_example = np.load(DATA_PATH+'/temp/one_wire.npy') 
    freq = np.load(DATA_PATH+'/temp/freq.npy')
    obsheader = np.load(DATA_PATH+'/temp/obshdr.npy')
    MJD = Time(obsheader[0,8],format='mjd', scale='utc', precision=4)

    sample_to_datetime = np.size(range_example[:,0])
    time_p1 = MJD + TimeDelta(range_example[:,0].astype(float), format='sec') 
    time_p1.format = 'iso'
    time_p1.out_subfmt = 'date_hm'
    time_p2 = np.asarray(time_p1.strftime('%H:%M'))   #array of HH:MM
    time_p2 = np.where(time_p2 != '00:00', time_p2, time_p1.strftime('%d %H:%M')) #at midnight add day number
    time_p1 = np.asarray(time_p1.strftime('%M'))
    time_p1 = np.where(time_p1 == '00') #these two lines find round hours
 
    mins = time_p2[time_p1].tolist() #values of hours for the tick labels
    first_tick = time_p1[0][:].tolist() #list of bin numbers for the hours tick labels

    quickload = np.load(DATA_PATH+'/temp/quickload.npy')

    if quickload:
        time_p1 = np.load(DATA_PATH+'/temp/time_array.npy') 
        time_p1 = Time(time_p1,format='mjd',scale='utc',precision=9)
        time_p1.format = 'iso'
        time_p1.out_subfmt = 'date_hm'
        time_p2 = np.asarray(time_p1.strftime('%H:%M'))   #array of HH:MM
        time_p2 = np.where(time_p2 != '00:00', time_p2, time_p1.strftime('%d %H:%M')) #at midnight add day number
        time_p1 = np.asarray(time_p1.strftime('%M'))
        time_p1 = np.where(time_p1 == '00') #these two lines find round hours
 
        mins = time_p2[time_p1].tolist() #values of hours for the tick labels
        first_tick = time_p1[0][:].tolist() #list of bin numbers for the hours tick labels
    
    del time_p1, time_p2, range_example

    return mins, first_tick, sample_to_datetime, freq


#-----------------------------------------------------------------------

def waterfall_singleInput(duration_actual,save_it=False, first_loop=True):

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    flatten = np.load(DATA_PATH+'/temp/flatten.npy')
    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    del p11

    cmap = np.load(DATA_PATH+'/temp/cmap.npy')
    if cmap:
        cstr='viridis'
    else:
        #cstr='gist_heat'
        cstr='turbo' 

    band11 = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
    mins,first_tick, sample_to_datetime, frequency = yaxis(band11, duration_actual)
    band12 = np.load (DATA_PATH+'/temp/a1p2_binned.npy')
    band21 = np.load (DATA_PATH+'/temp/a2p1_binned.npy')
    band22 = np.load (DATA_PATH+'/temp/a2p2_binned.npy')

    band11,band12,band21,band22 = corrections(band11,band12,band21,band22)

    if parameters[8] == 'True':
        band11 = band11/np.mean(band11)
        band12 = band12/np.mean(band12)
        band21 = band21/np.mean(band21)
        band22 = band22/np.mean(band22)

    
    if first_loop:
        print('')
        print('  1 - P(L,\u03C0) W')
        print('  2 - P(L,0) E')
        print('  3 - P(R,\u03C0) E')
        print('  4 - P(R,0) W')
        print('')
        which_input = input('Please select an input: ')
        try:
            which_input = int(which_input)
        except:
            pass
        print('')

    mask_width = parameters[4].astype(np.float)
    mask_width = 1+int(mask_width/2)
    if parameters[6] == 'True':
        plt.axvline(x=(371),linestyle='--', color='salmon', linewidth=2, label='1420.405 MHz')
    
    if which_input == 1:
        if parameters[2]=='True':
            band11[:,(484-mask_width):(483+mask_width)+1] = 'nan' #21cm
        if duration_actual < 8:
            plt.imshow(band11[:,112:569], cmap=cstr,aspect='auto', interpolation='none')#112:569
        else:
            plt.imshow(band11[:,112:569], cmap=cstr, interpolation='none')
        plt.title('Spectrogram, P(L,$\pi$) W\n'+user_inputs[0])
        del band11
       
    if which_input == 2:
        if parameters[2]=='True':
            band12[:,(484-mask_width):(483+mask_width)+1] = 'nan' #21cm
        if duration_actual < 8:
            plt.imshow(band12[:,112:569], cmap=cstr,aspect='auto', interpolation='none')
        else:
            plt.imshow(band12[:,112:569], cmap=cstr, interpolation='none')
        plt.title('Spectrogram, P(L,0) E\n'+user_inputs[0])
        del band12
    
    if which_input == 3:
        if parameters[2]=='True':
            band21[:,(484-mask_width):(483+mask_width)+1] = 'nan' #21cm
        if duration_actual < 8:
            plt.imshow(band21[:,112:569], cmap=cstr,aspect='auto', interpolation='none')
        else:
            axarr[1,0].imshow(band21[:,112:569], cmap=cstr, interpolation='none')
        plt.title('Spectrogram, P(R,$\pi$) E\n'+user_inputs[0])
        del band21

    if which_input == 4:
        if parameters[2]=='True':
            band22[:,(484-mask_width):(483+mask_width)+1] = 'nan' #21cm
        if duration_actual < 8:
            plt.imshow(band22[:,112:569], cmap=cstr,aspect='auto', interpolation='none')
        else:
            plt.imshow(band22[:,112:569], cmap=cstr, interpolation='none')
     
        plt.title('Spectrogram, P(R,0) W\n'+user_inputs[0])
        del band22
    
  #  user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
   # if parameters[8]=='True' or parameters[1]=='True':
    #    if parameters[1]=='True' and parameters[8]=='True':
     #       title_string = 'Spectrogram, Double Difference\n'+user_inputs[0]+' Global Normalised, Normalised'
      #  elif parameters[1]=='True':
       #     title_string =  'Spectrogram, Double Difference\n'+user_inputs[0]+' Global Normalised'
       # else:  
        #    title_string = 'Spectrogram, Double Difference\n'+user_inputs[0]+' Normalised'

   # else:
    #    title_string = 'Spectrogram, Double Difference\n'+user_inputs[0]

    if save_it:
        plt.figure(figsize=(12, 8), dpi=300)
          
    plt.yticks(ticks=first_tick, labels=mins)
    plt.xticks(ticks=[0,45,91,136,182,227,273,318,364,409,456], labels=['1400','','1405','','1410','','1415','','1420','','1425'])

    plt.colorbar()
    plt.xlabel('Frequency / MHz')
    plt.ylabel('Time')

    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+str(which_input)+'_WF.png', bbox_inches="tight")
        print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
    else:
        plt.show()
    plt.close()
    
    if first_loop:
        print ('')
        save = input ('Do you want to save a printer friendly copy of this plot? (Y/N): \033[0;m')
        if str(save) == 'N' or str(save) == 'n':
            print('\033[1;32m')
            pass
        elif str(save) == 'Y' or str(save) == 'y':
            save_it=True
            first_loop = False
            waterfall_singleInput(duration_actual,save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass



#----------------------------------------------------------------------------
def waterfall_oneDIFF(duration_actual,save_it=False, first_loop=True):

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    flatten = np.load(DATA_PATH+'/temp/flatten.npy')
    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    del p11

    cmap = np.load(DATA_PATH+'/temp/cmap.npy')
    if cmap:
        cstr='viridis'
    else:
        #cstr='gist_heat'
        cstr='turbo' 

    band11 = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
    mins,first_tick, sample_to_datetime, frequency = yaxis(band11, duration_actual)
    band12 = np.load (DATA_PATH+'/temp/a1p2_binned.npy')
    band21 = np.load (DATA_PATH+'/temp/a2p1_binned.npy')
    band22 = np.load (DATA_PATH+'/temp/a2p2_binned.npy')

    band11,band12,band21,band22 = corrections(band11,band12,band21,band22)

    if parameters[8]=='True':
        band11 = band11/np.mean(band11[:,112:569])
        band12 = band12/np.mean(band12[:,112:569])
        band21 = band21/np.mean(band21[:,112:569])
        band22 = band22/np.mean(band22[:,112:569])

    d_a = band12 - band11 # l0 - lpi E-W
    d_b = band22 - band21 #r0 - rpi W-E
    d_c = band12 - band22  # l0 - r0  E-W
    d_d = band11 - band21  #   lpi - rpi  W-E

    if first_loop:
        print('')
        print('  1 - \u03B4a  P(L,0) E - P(L,\u03C0) W')
        print('  2 - \u03B4b  P(R,0) W - P(R,\u03C0) E')
        print('  3 - \u03B4c  P(L,0) E - P(R,0) W')
        print('  4 - \u03B4d  P(L,\u03C0) W - P(R,\u03C0) E')
        print('')
        which_input = input('Please select an input: ')
        try:
            which_input = int(which_input)
        except:
            pass
        print('')

    mask_width = parameters[4].astype(np.float)
    mask_width = 1+int(mask_width/2)
    if parameters[6] == 'True':
        plt.axvline(x=(371),linestyle='--', color='salmon', linewidth=2, label='1420.405 MHz')
    
    if which_input == 1:
        if parameters[2]=='True':
            d_a[:,(484-mask_width):(483+mask_width)+1] = 'nan' #21cm
        if duration_actual < 8:
            plt.imshow(d_a[:,112:569], cmap=cstr,aspect='auto', interpolation='none')#112:569
        else:
            plt.imshow(d_a[:,112:569], cmap=cstr, interpolation='none')
        plt.title('Spectrogram Single Difference, $\delta$a = P(L,0) E - P(L,$\pi$) W\n'+user_inputs[0])
        del band11,d_a
       
    if which_input == 2:
        if parameters[2]=='True':
            d_b[:,(484-mask_width):(483+mask_width)] = 'nan' #21cm
        if duration_actual < 8:
            plt.imshow(d_b[:,112:569], cmap=cstr,aspect='auto', interpolation='none')
        else:
            plt.imshow(d_b[:,112:569], cmap=cstr, interpolation='none')
        plt.title('Spectrogram Single Difference, $\delta$b = P(R,0) W - P(R,$\pi$) E\n'+user_inputs[0])
        del band12, d_b
    
    if which_input == 3:
        if parameters[2]=='True':
            d_c[:,(484-mask_width):(483+mask_width)+1] = 'nan' #21cm
        if duration_actual < 8:
            plt.imshow(d_c[:,112:569], cmap=cstr,aspect='auto', interpolation='none')
        else:
            axarr[1,0].imshow(d_c[:,112:569], cmap=cstr, interpolation='none')
        plt.title('Spectrogram Single Difference, $\delta$c = P(L,0) E - P(R,0) W\n'+user_inputs[0])
        del band21, d_c

    if which_input == 4:
        if parameters[2]=='True':
            d_d[:,(484-mask_width):(483+mask_width)+1] = 'nan' #21cm
        if duration_actual < 8:
            plt.imshow(d_d[:,112:569], cmap=cstr,aspect='auto', interpolation='none')
        else:
            plt.imshow(d_d[:,112:569], cmap=cstr, interpolation='none')
     
        plt.title('Spectrogram Single Difference, $\delta$d = P(L,$\pi$) W - P(R,$\pi$) E\n'+user_inputs[0])
        del band22
    
  #  user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
   # if parameters[8]=='True' or parameters[1]=='True':
    #    if parameters[1]=='True' and parameters[8]=='True':
     #       title_string = 'Spectrogram, Double Difference\n'+user_inputs[0]+' Global Normalised, Normalised'
      #  elif parameters[1]=='True':
       #     title_string =  'Spectrogram, Double Difference\n'+user_inputs[0]+' Global Normalised'
       # else:  
        #    title_string = 'Spectrogram, Double Difference\n'+user_inputs[0]+' Normalised'

   # else:
    #    title_string = 'Spectrogram, Double Difference\n'+user_inputs[0]

    if save_it:
        plt.figure(figsize=(12, 8), dpi=300)
          
    plt.yticks(ticks=first_tick, labels=mins)
    plt.xticks(ticks=[0,45,91,136,182,227,273,318,364,409,456], labels=['1400','','1405','','1410','','1415','','1420','','1425'])

    plt.colorbar()
    plt.xlabel('Frequency / MHz')
    plt.ylabel('Time')

    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+str(which_input)+'_WF.png', bbox_inches="tight")
        print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
    else:
        plt.show()
    plt.close()
    
    if first_loop:
        print ('')
        save = input ('Do you want to save a printer friendly copy of this plot? (Y/N): \033[0;m')
        if str(save) == 'N' or str(save) == 'n':
            print('\033[1;32m')
            pass
        elif str(save) == 'Y' or str(save) == 'y':
            save_it=True
            first_loop = False
            waterfall_oneDIFF(duration_actual,save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass

#----------------------------------------------------------------------------


def WaterfallMenu(warnlen):

    looper = True
    duration_actual = np.load(DATA_PATH+'/temp/duration_actual.npy')

    print ('')
    print ('   -------------------------------------')
    print ('   >>>     WATERFALL PLOTS MENU      <<<')
    print ('   -------------------------------------')
    print ('')
    print ('   1 - All Inputs',warnlen)
    print ('')
    print ('   2 - Single Differences',warnlen)
    print ('')
    print ('   3 - Double Difference',warnlen)
    print ('')
    print ('   4 - Nulls',warnlen)
    print ('')
    print ('   5 - One Input')
    print ('')
    print ('   6 - One Single Difference')
    print ('')
    print ('')
    print ('   0 - Return to Quick-look menu')
    print ('')
    choice = input('Select menu option (number): \033[0;m')
    print('\033[1;32m')
    if choice.isdigit():        
        if int(choice) ==1:
            waterfallPERmin(duration_actual)

        elif int(choice) ==2:
            waterfallSINGLE(duration_actual)

        elif int(choice) ==3:
            waterfallDOUBLE(duration_actual)

        elif int(choice) ==4:
            waterfallNULLS(duration_actual)

        elif int(choice) == 5:
            waterfall_singleInput(duration_actual)

        elif int(choice)==6:
            waterfall_oneDIFF(duration_actual)

        elif int(choice) == 0:
            looper = False
            pass

         
    else:
        print('\033[1;31m Invalid selection by user. Please try again. \033[1;32m')
        WaterfallMenu(warnlen)

    return looper

#########################################################################

duration_actual = np.load(DATA_PATH+'/temp/duration_actual.npy')
if float(duration_actual) > 24:
    warnlen='\033[1;31m (dataset too large) \033[1;32m'
else:
    warnlen=''

looper = True

while looper:
    looper = WaterfallMenu(warnlen)

print ('\033[1;32m ')

os.system('chmod -R -f 0777 /mirror/scratch/pblack || true')

