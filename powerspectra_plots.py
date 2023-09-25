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
import scipy
from scipy.optimize import curve_fit
import numpy as np
import astropy.io
from astropy.io import fits
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time 
from astropy.time import TimeDelta
import datetime
import time
from tqdm import tqdm #progress bars
import glob
import os
from math import nan
import gc
from matplotlib import colors
from matplotlib.colors import LogNorm
from scipy.fft import rfft, rfftfreq
from scipy import stats

os.chdir('/scratch/nas_lbass/raw_data/')

# GOOD FREQ ARE 112-568  BUT remember channels start at 3 in a1p1 etc
# A1P1 - P(l,pi)
# A2P2 - P(r,0)
# A1P2 - P(l,0)
# A2P1 - P(r,pi)



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

#~~~~~~~~~~~~~~~~~~~~#~~~~~~~~JORDAN NORRIS~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~#

def getSamplingTime(time_array):
    """
    Takes differences in times between starts and ends of arrays to measure the
    average time between measurements for use in Fourier Transforming.
    Parameters
    ----------
    time_array : array
        Numpy array of timedate objects specifying time at each observation.

    Returns
    -------
    timestep : float
        Average time between measurements

    """
    starttime = time_array[0]
    latertime = time_array[-1]
    
    diff = latertime - starttime
    
    timestep = diff / np.size(time_array)
    
    return timestep

def UTCDateTime(range_example, duration_actual):
    """
    Based on code developed by Phillip Black. This code generates arrays of 
    sample times by using the astropy time backage for use in plotting data.

    Parameters
    ----------
    range_example : Array
        Array that may end up being used to test the length of the array
        corresponding to the time
    duration_actual : Float
        Float of the observation period that is being looked at in Hours

    Returns
    -------
    sample_to_datetime : Array
        Array of strings specifing the time in UTC that each line in the plotted
        arrays corresponds to.
    """ 
    obsheader = np.load(DATA_PATH+'/temp/obshdr.npy')
    
    mid_obs = Time(obsheader[0,7],format='isot', scale='utc', precision=0, 
                   out_subfmt='date')
    
    obs_date = Time(str(mid_obs),format='iso', scale='utc', precision=0, 
                    out_subfmt='date_hms')
    
    #bin length assumed to be one mnute - make that adjustable, PIPs
    sample_to_datetime = []
    i=0
    
    for i in range (0,np.size(range_example[:,0])): #this means sample time is days long
            sb = obs_date + TimeDelta(float(range_example[i,0]), format='sec') #changed run_start_date to obs_date
            r = sb.strftime("%Y-%m-%d %H:%M:%S")
            sample_to_datetime.append(r)
            
    return sample_to_datetime


#~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



#load background files and data tables
user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
frequency = np.load(DATA_PATH+'/temp/freq.npy') #its in hertz
#frequency = frequency * (10**-9)
####################################################################################


def xaxis(range_example, duration_actual): #range example being a1p1B etc
    binnable = np.load(DATA_PATH+'/temp/rebinnable.npy')
    if binnable:
        range_example = np.load(DATA_PATH+'/temp/one_wire.npy') 
    p = np.load(DATA_PATH+'/temp/run_start_date.npy', allow_pickle=True)
    run_start_date = Time(str(p), out_subfmt='date_hms')
    re_date = np.load(DATA_PATH+'/temp/re_date.npy')
    obsheader = np.load(DATA_PATH+'/temp/obshdr.npy')
    mid_obs = Time(obsheader[0,7],format='isot', scale='utc', precision=0, out_subfmt='date')
    obs_date = Time(str(mid_obs),format='iso', scale='utc', precision=0, out_subfmt='date_hms')
    freq = np.load(DATA_PATH+'/temp/freq.npy')
    jeff = (freq * 10**-9)
    frequency = np.around(jeff, decimals =3)
    
    #bin length assumed to be one mnute - make that adjustable

    sample_to_datetime = []
    i=0
    
    if re_date: #this means sample time restarts at 0
        for i in range (0,np.size(range_example[:,0])):
            sb = obs_date + TimeDelta(float(range_example[i,0]), format='sec')
            r = sb.strftime("%H:%M")   
            sample_to_datetime.append(r)
        
    else:
        for i in range (0,np.size(range_example[:,0])): #this means sample time is days long
            sb = run_start_date + TimeDelta(float(range_example[i,0]), format='sec')
            r = sb.strftime("%H:%M")
            sample_to_datetime.append(r)
        
    mins =[]
    first_tick=[]
    for i in range(0,np.size(sample_to_datetime)): #there is only 24 hours in a day!
        x = str(sample_to_datetime[i])
        bb = x.strip("'")
        h,m = bb.split(':')
        if m == '00': 
            mins.append(h+':'+m)
            first_tick.append(i)
        else:
            pass
        
    if duration_actual > 48:
        i=0
        j=0
        for j in range(0,np.size(mins)):
            for i in range (1,10):
                try:
                    mins[j] = str(mins[j]).replace('0'+str(i)+':00','')
                except:
                    pass
            for i in range (10,12):
                mins[j] = str(mins[j]).replace(str(i)+':00','')
            for i in range (13,24):
                mins[j] = str(mins[j]).replace(str(i)+':00','')

    return mins, first_tick, sample_to_datetime, frequency
    
#--------------------------------------------------------------------------------
#-----------------------------------------------------------------------

def yaxis(range_example, duration_actual): #range example being a1p1B etc
    binnable = np.load(DATA_PATH+'/temp/rebinnable.npy')
    if binnable:
        range_example = np.load(DATA_PATH+'/temp/one_wire.npy') 
    p = np.load(DATA_PATH+'/temp/run_start_date.npy', allow_pickle=True)
    run_start_date = Time(str(p), out_subfmt='date_hms')
    re_date = np.load(DATA_PATH+'/temp/re_date.npy')
    obsheader = np.load(DATA_PATH+'/temp/obshdr.npy')
    mid_obs = Time(obsheader[0,7],format='isot', scale='utc', precision=0, out_subfmt='date')
    obs_date = Time(str(mid_obs),format='iso', scale='utc', precision=0, out_subfmt='date_hms')
    freq = np.load(DATA_PATH+'/temp/freq.npy')
    jeff = (freq * 10**-9)
    frequency = np.around(jeff, decimals =3)
    
    #bin length assumed to be one mnute - make that adjustable
    
    sample_to_datetime = []
    i=0
    
    if re_date: #this means sample time restarts at 0
        for i in range (0,np.size(range_example[:,0])):
            sb = obs_date + TimeDelta(float(range_example[i,0]), format='sec')
            r = sb.strftime("%H:%M")   
            sample_to_datetime.append(r)
        
    else:
        for i in range (0,np.size(range_example[:,0])): #this means sample time is days long
            sb = run_start_date + TimeDelta(float(range_example[i,0]), format='sec')
            r = sb.strftime("%H:%M")
            sample_to_datetime.append(r)
        
    mins =[]
    first_tick=[]
    for i in range(0,np.size(sample_to_datetime)): #there is only 24 hours in a day!
        x = str(sample_to_datetime[i])
        bb = x.strip("'")
        h,m = bb.split(':')
        if m == '00': 
            mins.append(h+':'+m)
            first_tick.append(i)
        else:
            pass
        
    if duration_actual > 24:
        i=0
        j=0
        for j in range(0,np.size(mins)):
            for i in range (1,10):
                try:
                    mins[j] = str(mins[j]).replace('0'+str(i)+':00','')
                except:
                    pass
            for i in range (10,12):
                mins[j] = str(mins[j]).replace(str(i)+':00','')
            for i in range (13,24):
                mins[j] = str(mins[j]).replace(str(i)+':00','')

    return mins, first_tick, sample_to_datetime, frequency

#---------------------------------------------------------------------------------

def PowerSpectra(save_it=False, first_loop=True):

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    signal1 =[]
    signal2=[]
    signal3=[]
    signal4=[]
    a1p1 = np.load(DATA_PATH+'/temp/a1p1.npy')
    #a1p1 = a1p1[:,3:]
    a2p2 = np.load(DATA_PATH+'/temp/a2p2.npy')
    #a2p2 = a2p2[:,3:]
    a1p2 = np.load(DATA_PATH+'/temp/a1p2.npy')
    #a1p2 = a1p2[:,3:]
    a2p1 = np.load(DATA_PATH+'/temp/a2p1.npy')
    #a2p1 = a2p1[:,3:]

    sampletime1 = getSamplingTime(a1p1[:,0])
    sampletime2 = getSamplingTime(a2p2[:,0])
    sampletime3 = getSamplingTime(a1p2[:,0])
    sampletime4 = getSamplingTime(a2p1[:,0])

    for line in a1p1:
        #signal = np.sum(line[112:569])
        signal = np.sum(line[115:572])
        signal1.append(signal)
    signal1 = np.array(signal1)

    for line in a2p2:
       # signal = np.sum(line[112:569])
        signal = np.sum(line[115:572])
        signal2.append(signal)
    signal2 = np.array(signal2)

    for line in a1p2:
        #signal = np.sum(line[112:569])
        signal = np.sum(line[115:572])
        signal3.append(signal)
    signal3 = np.array(signal3)

    for line in a2p1:
       # signal = np.sum(line[112:569])
        signal = np.sum(line[115:572])
        signal4.append(signal)
    signal4 = np.array(signal4)

    numberofsamples1 = np.size(signal1)
    numberofsamples2 = np.size(signal2)
    numberofsamples3 = np.size(signal3)
    numberofsamples4 = np.size(signal4)
 
    sig_fft1 = rfft(signal1)
    power1 = sig_fft1 * np.conjugate(sig_fft1)
    freq1 = rfftfreq(numberofsamples1, sampletime1)
    sig_fft2 = rfft(signal2)
    freq2 = rfftfreq(numberofsamples2, sampletime2)
    power2 = sig_fft2 * np.conjugate(sig_fft2)
    sig_fft3 = rfft(signal3)
    freq3 = rfftfreq(numberofsamples3, sampletime3)
    power3 = sig_fft3 * np.conjugate(sig_fft3)
    sig_fft4 = rfft(signal4)
    freq4 = rfftfreq(numberofsamples4, sampletime4)
    power4 = sig_fft4 * np.conjugate(sig_fft4)

    nbin = 100
    nzf1 = np.nonzero(freq1)
    freq1 = (freq1[nzf1])
    bins = np.geomspace(np.min(freq1),np.max(freq1),nbin)
    amps = np.zeros(nbin)
    i=0
    for i in range(0,nbin):
		#m = nonzero(freq >= bins[line] && find the indices in logfreq where bins[line] <= freq < bins[line+1]
        if i < (nbin-1):
            m = np.where(freq1 >= bins[i])
            n = np.where(freq1 < bins[i+1])
        else:
            m = np.where(freq1 >= bins[i])
            n = m
        if len(power1[m[0][0]:n[0][-1]]) == 0:
            amps[i] = 'nan'
        else:
            amps[i] = np.mean(np.real(power1[m[0][0]:n[0][-1]]))

    nzf2 = np.nonzero(freq2)
    freq2 = (freq2[nzf2])
    bins2 = np.geomspace(np.min(freq2),np.max(freq2),nbin)
    amps2 = np.zeros(nbin)
    i=0
    for i in range(0,nbin):
		#m = nonzero(freq >= bins[line] && find the indices in logfreq where bins[line] <= freq < bins[line+1]
        if i < (nbin-1):
            m = np.where(freq2 >= bins2[i])
            n = np.where(freq2 < bins2[i+1])
        else:
            m = np.where(freq2 >= bins2[i])
            n = m
        if len(power2[m[0][0]:n[0][-1]]) == 0:
            amps2[i] = 'nan'
        else:
            amps2[i] = np.mean(np.real(power2[m[0][0]:n[0][-1]]))

    nzf3 = np.nonzero(freq3)
    freq3 = (freq3[nzf3])
    bins3 = np.geomspace(np.min(freq3),np.max(freq3),nbin)
    amps3 = np.zeros(nbin)
    i=0
    for i in range(0,nbin):
		#m = nonzero(freq >= bins[line] && find the indices in logfreq where bins[line] <= freq < bins[line+1]
        if i < (nbin-1):
            m = np.where(freq3 >= bins3[i])
            n = np.where(freq3 < bins3[i+1])
        else:
            m = np.where(freq3 >= bins3[i])
            n = m
        if len(power3[m[0][0]:n[0][-1]]) == 0:
            amps3[i] = 'nan'
        else:
            amps3[i] = np.mean(np.real(power3[m[0][0]:n[0][-1]]))

    nzf4 = np.nonzero(freq4)
    freq4 = (freq4[nzf4])
    bins4 = np.geomspace(np.min(freq4),np.max(freq4),nbin)
    amps4 = np.zeros(nbin)
    i=0
    for i in range(0,nbin):
		#m = nonzero(freq >= bins[line] && find the indices in logfreq where bins[line] <= freq < bins[line+1]
        if i < (nbin-1):
            m = np.where(freq4 >= bins4[i])
            n = np.where(freq4 < bins4[i+1])
        else:
            m = np.where(freq4 >= bins4[i])
            n = m
        if len(power4[m[0][0]:n[0][-1]]) == 0:
            amps4[i] = 'nan'
        else:
            amps4[i] = np.mean(np.real(power4[m[0][0]:n[0][-1]]))

 
    titlestring = 'Power Spectra \n'+user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours.'  
    
    if save_it:
        fig, axs = plt.subplots(2, 2, figsize=(12,8), dpi=300)
    else:
        fig, axs = plt.subplots(2, 2)
    
    axs[1,0].plot(bins4, amps4, c='firebrick', linewidth=2)
    axs[1,0].set_title('P(R,$\pi$) E')

    axs[0,0].plot(bins3, amps3, c='deeppink', linewidth=1)
    axs[0,0].set_title('P(L,0) E')

    axs[0,1].plot(bins, amps, c='b', linewidth=1)
    axs[0,1].set_title('P(L,$\pi$) W')

    axs[1,1].plot(bins2, amps2, c='c', linewidth=2)
    axs[1,1].set_title('P(R,0) W')

    for ax in axs.flat:
        ax.set(xlabel='Frequency /Hz', ylabel='Fourier Amplitude')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.margins(x=0)
        ax.axis('scaled')
        ax.grid(c='darkgrey', which='major')
        ax.grid(c='gainsboro', which='minor')

    plt.suptitle(titlestring)
    plt.tight_layout()
    
    
  
 
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_PowerSpectra.png', bbox_inches="tight")
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
            PowerSpectra(duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 

    del p11,p12,p21,p22
    



#----------------------------------------------------------------------

def PowerSpectraONE(save_it=False, first_loop=True):
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')

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

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    for_title = ['P(L,$\pi$) W','P(L,0) E','P(R,$\pi$) E','P(R,0) W']
    

    signal1=[]
    titlestring = 'Power Spectra, '+for_title[which_input-1]+' \n'+user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours.' 

    nbin = 100

    if which_input == 1:
        data = np.load(DATA_PATH+'/temp/a1p1.npy')

        sampletime1 = getSamplingTime(data[:,0])
        for line in data:
            signal = np.sum(line[115:572])
            signal1.append(signal)
        signal1 = np.array(signal1)
        numberofsamples1 = np.size(signal1)
        sig_fft1 = rfft(signal1)
        freq1 = rfftfreq(numberofsamples1, sampletime1)
        power1 = sig_fft1 * np.conjugate(sig_fft1)
        nzf1 = np.nonzero(freq1)
        freq1 = (freq1[nzf1])
        bins = np.geomspace(np.min(freq1),np.max(freq1),nbin)
        amps = np.zeros(nbin)
        i=0
        for i in range(0,nbin):
		#m = nonzero(freq >= bins[line] && find the indices in logfreq where bins[line] <= freq < bins[line+1]
            if i < (nbin-1):
                m = np.where(freq1 >= bins[i])
                n = np.where(freq1 < bins[i+1])
            else:
                m = np.where(freq1 >= bins[i])
                n = m
            if len(power1[m[0][0]:n[0][-1]]) == 0:
                amps[i] = 'nan'
            else:
                amps[i] = np.mean(np.real(power1[m[0][0]:n[0][-1]])) 
        plt.plot(bins, amps)
        plt.title(titlestring)

    elif which_input == 2:
        data = np.load(DATA_PATH+'/temp/a1p2.npy')

        sampletime1 = getSamplingTime(data[:,0])
        for line in data:
            signal = np.sum(line[115:572])
            signal1.append(signal)
        signal1 = np.array(signal1)
        numberofsamples1 = np.size(signal1)
        sig_fft1 = rfft(signal1)
        freq1 = rfftfreq(numberofsamples1, sampletime1)
        power1 = sig_fft1 * np.conjugate(sig_fft1)
        nzf1 = np.nonzero(freq1)
        freq1 = (freq1[nzf1])
        bins = np.geomspace(np.min(freq1),np.max(freq1),nbin)
        amps = np.zeros(nbin)
        i=0
        for i in range(0,nbin):
		#m = nonzero(freq >= bins[line] && find the indices in logfreq where bins[line] <= freq < bins[line+1]
            if i < (nbin-1):
                m = np.where(freq1 >= bins[i])
                n = np.where(freq1 < bins[i+1])
            else:
                m = np.where(freq1 >= bins[i])
                n = m
            if len(power1[m[0][0]:n[0][-1]]) == 0:
                amps[i] = 'nan'
            else:
                amps[i] = np.mean(np.real(power1[m[0][0]:n[0][-1]])) 
        plt.plot(bins, amps)
        plt.title(titlestring)

    elif which_input == 3:
        data = np.load(DATA_PATH+'/temp/a2p1.npy')

        sampletime1 = getSamplingTime(data[:,0])
        for line in data:
            signal = np.sum(line[115:572])
            signal1.append(signal)
        signal1 = np.array(signal1)
        numberofsamples1 = np.size(signal1)
        sig_fft1 = rfft(signal1)
        freq1 = rfftfreq(numberofsamples1, sampletime1)
        power1 = sig_fft1 * np.conjugate(sig_fft1)
        nzf1 = np.nonzero(freq1)
        freq1 = (freq1[nzf1])
        bins = np.geomspace(np.min(freq1),np.max(freq1),nbin)
        amps = np.zeros(nbin)
        i=0
        for i in range(0,nbin):
		#m = nonzero(freq >= bins[line] && find the indices in logfreq where bins[line] <= freq < bins[line+1]
            if i < (nbin-1):
                m = np.where(freq1 >= bins[i])
                n = np.where(freq1 < bins[i+1])
            else:
                m = np.where(freq1 >= bins[i])
                n = m
            if len(power1[m[0][0]:n[0][-1]]) == 0:
                amps[i] = 'nan'
            else:
                amps[i] = np.mean(np.real(power1[m[0][0]:n[0][-1]])) 
        plt.plot(bins, amps)
        plt.title(titlestring) 

    elif which_input == 4:
        data = np.load(DATA_PATH+'/temp/a2p2.npy')

        sampletime1 = getSamplingTime(data[:,0])
        for line in data:
            signal = np.sum(line[115:572])
            signal1.append(signal)
        signal1 = np.array(signal1)
        numberofsamples1 = np.size(signal1)
        sig_fft1 = rfft(signal1)
        freq1 = rfftfreq(numberofsamples1, sampletime1)
        power1 = sig_fft1 * np.conjugate(sig_fft1)
        nzf1 = np.nonzero(freq1)
        freq1 = (freq1[nzf1])
        bins = np.geomspace(np.min(freq1),np.max(freq1),nbin)
        amps = np.zeros(nbin)
        i=0
        for i in range(0,nbin):
		#m = nonzero(freq >= bins[line] && find the indices in logfreq where bins[line] <= freq < bins[line+1]
            if i < (nbin-1):
                m = np.where(freq1 >= bins[i])
                n = np.where(freq1 < bins[i+1])
            else:
                m = np.where(freq1 >= bins[i])
                n = m
            if len(power1[m[0][0]:n[0][-1]]) == 0:
                amps[i] = 'nan'
            else:
                amps[i] = np.mean(np.real(power1[m[0][0]:n[0][-1]])) 
        plt.plot(bins, amps)
        plt.title(titlestring)

   
    else:
        print('\033[1;31m Input not recognised, please try again. \033[1;32m')
        PowerSpectraONE()

    if save_it:
        plt.figure(figsize=(12, 8), dpi=300)

    plt.xlabel('Frequency /Hz')
    plt.ylabel('Fourier Amplitude')
    plt.xscale('log')
    plt.yscale('log')
    plt.margins(x=0)
    plt.axis('scaled')
    plt.grid(c='darkgrey', which='major')
    plt.grid(c='gainsboro', which='minor')
    plt.tight_layout()
 
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_PowerSpectra1.png', bbox_inches="tight")
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
            PowerSpectraONE(save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 

#------------------------------------------------------------------

def PowerSpectra4onONE(save_it=False, first_loop=True):
    
    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    signal1 =[]
    signal2=[]
    signal3=[]
    signal4=[]
    a1p1 = np.load(DATA_PATH+'/temp/a1p1.npy')
   # a1p1 = a1p1[:,3:]
    a2p2 = np.load(DATA_PATH+'/temp/a2p2.npy')
    #a2p2 = a2p2[:,3:]
    a1p2 = np.load(DATA_PATH+'/temp/a1p2.npy')
   # a1p2 = a1p2[:,3:]
    a2p1 = np.load(DATA_PATH+'/temp/a2p1.npy')
    #a2p1 = a2p1[:,3:]

    sampletime1 = getSamplingTime(a1p1[:,0])
    sampletime2 = getSamplingTime(a2p2[:,0])
    sampletime3 = getSamplingTime(a1p2[:,0])
    sampletime4 = getSamplingTime(a2p1[:,0])

    for line in a1p1:
        #signal = np.sum(line[112:569])
        signal = np.sum(line[115:572])
        signal1.append(signal)
    signal1 = np.array(signal1)

    for line in a2p2:
     #   signal = np.sum(line[112:569])
        signal = np.sum(line[115:572])
        signal2.append(signal)
    signal2 = np.array(signal2)

    for line in a1p2:
       # signal = np.sum(line[112:569])
        signal = np.sum(line[115:572])
        signal3.append(signal)
    signal3 = np.array(signal3)

    for line in a2p1:
      #  signal = np.sum(line[112:569])
        signal = np.sum(line[115:572])
        signal4.append(signal)
    signal4 = np.array(signal4)

    numberofsamples1 = np.size(signal1)
    numberofsamples2 = np.size(signal2)
    numberofsamples3 = np.size(signal3)
    numberofsamples4 = np.size(signal4)
 
    sig_fft1 = rfft(signal1)
    power1 = sig_fft1 * np.conjugate(sig_fft1)
    freq1 = rfftfreq(numberofsamples1, sampletime1)
    sig_fft2 = rfft(signal2)
    freq2 = rfftfreq(numberofsamples2, sampletime2)
    power2 = sig_fft2 * np.conjugate(sig_fft2)
    sig_fft3 = rfft(signal3)
    freq3 = rfftfreq(numberofsamples3, sampletime3)
    power3 = sig_fft3 * np.conjugate(sig_fft3)
    sig_fft4 = rfft(signal4)
    freq4 = rfftfreq(numberofsamples4, sampletime4)
    power4 = sig_fft4 * np.conjugate(sig_fft4)

    nbin = 100
    nzf1 = np.nonzero(freq1)
    freq1 = (freq1[nzf1])
    bins = np.geomspace(np.min(freq1),np.max(freq1),nbin)
    amps = np.zeros(nbin)
    i=0
    for i in range(0,nbin):
		#m = nonzero(freq >= bins[line] && find the indices in logfreq where bins[line] <= freq < bins[line+1]
        if i < (nbin-1):
            m = np.where(freq1 >= bins[i])
            n = np.where(freq1 < bins[i+1])
        else:
            m = np.where(freq1 >= bins[i])
            n = m
        if len(power1[m[0][0]:n[0][-1]]) == 0:
            amps[i] = 'nan'
        else:
            amps[i] = np.mean(np.real(power1[m[0][0]:n[0][-1]]))

    nzf2 = np.nonzero(freq2)
    freq2 = (freq2[nzf2])
    bins2 = np.geomspace(np.min(freq2),np.max(freq2),nbin)
    amps2 = np.zeros(nbin)
    i=0
    for i in range(0,nbin):
		#m = nonzero(freq >= bins[line] && find the indices in logfreq where bins[line] <= freq < bins[line+1]
        if i < (nbin-1):
            m = np.where(freq2 >= bins2[i])
            n = np.where(freq2 < bins2[i+1])
        else:
            m = np.where(freq2 >= bins2[i])
            n = m
        if len(power2[m[0][0]:n[0][-1]]) == 0:
            amps2[i] = 'nan'
        else:
            amps2[i] = np.mean(np.real(power2[m[0][0]:n[0][-1]]))

    nzf3 = np.nonzero(freq3)
    freq3 = (freq3[nzf3])
    bins3 = np.geomspace(np.min(freq3),np.max(freq3),nbin)
    amps3 = np.zeros(nbin)
    i=0
    for i in range(0,nbin):
		#m = nonzero(freq >= bins[line] && find the indices in logfreq where bins[line] <= freq < bins[line+1]
        if i < (nbin-1):
            m = np.where(freq3 >= bins3[i])
            n = np.where(freq3 < bins3[i+1])
        else:
            m = np.where(freq3 >= bins3[i])
            n = m
        if len(power3[m[0][0]:n[0][-1]]) == 0:
            amps3[i] = 'nan'
        else:
            amps3[i] = np.mean(np.real(power3[m[0][0]:n[0][-1]]))

    nzf4 = np.nonzero(freq4)
    freq4 = (freq4[nzf4])
    bins4 = np.geomspace(np.min(freq4),np.max(freq4),nbin)
    amps4 = np.zeros(nbin)
    i=0
    for i in range(0,nbin):
		#m = nonzero(freq >= bins[line] && find the indices in logfreq where bins[line] <= freq < bins[line+1]
        if i < (nbin-1):
            m = np.where(freq4 >= bins4[i])
            n = np.where(freq4 < bins4[i+1])
        else:
            m = np.where(freq4 >= bins4[i])
            n = m
        if len(power4[m[0][0]:n[0][-1]]) == 0:
            amps4[i] = 'nan'
        else:
            amps4[i] = np.mean(np.real(power4[m[0][0]:n[0][-1]]))

 
    titlestring = 'Power Spectra \n'+user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours.'  
    
    if save_it:
        plt.figure(figsize=(12, 8), dpi=300)
    
    plt.plot(bins4, amps4, c='firebrick', label='P(R,$\pi$) E', linewidth=2)
    plt.plot(bins3, amps3, c='deeppink', label='P(L,0) E', linewidth=1)
    plt.plot(bins, amps, c='b', label='P(L,$\pi$) W', linewidth=1)
    plt.plot(bins2, amps2, c='c', label='P(R,0) W', linewidth=2)

    plt.xlabel('Frequency /Hz')
    plt.ylabel('Fourier Amplitude')
    plt.xscale('log')
    plt.yscale('log')
    plt.margins(x=0)
    plt.axis('scaled')
    plt.legend()
    plt.grid(c='darkgrey', which='major')
    plt.grid(c='gainsboro', which='minor')
    plt.title(titlestring)
    plt.tight_layout()
    
    
  
 
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_PowerSpectras.png', bbox_inches="tight")
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
            PowerSpectra4onONE(save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 

#------------------------------------------------------------------

def PowerSpectraFIRSTDIFF(save_it=False, first_loop=True):

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    signal1 =[]
    signal2=[]
    signal3=[]
    signal4=[]
    a1p1 = np.load(DATA_PATH+'/temp/a1p1.npy')
    a1p1 = a1p1[:,3:]
    a2p2 = np.load(DATA_PATH+'/temp/a2p2.npy')
    a2p2 = a2p2[:,3:]
    a1p2 = np.load(DATA_PATH+'/temp/a1p2.npy')
    a1p2 = a1p2[:,3:]
    a2p1 = np.load(DATA_PATH+'/temp/a2p1.npy')
    a2p1 = a2p1[:,3:]

    if np.size(a1p1[:,0]) == np.size(a1p2[:,0]):
        pass

    elif np.size(a1p1[:,0]) > np.size(a1p2[:,0]):
        a1p1 = a1p1[:-1,:]
        a2p1 = a2p1[:-1,:]
    elif np.size(a1p1[:,0]) < np.size(a1p2[:,0]):
        a1p2 = a1p2[:-1,:]
        a2p2 = a2p2[:-1,:]

    d_a = a1p2 - a1p1 # l0 - lpi 
    d_b = a2p2 - a2p1 #r0 - rpi 
    d_c = a1p2 - a2p2  # l0 - r0 
    d_d = a1p1 - a2p1  #   lpi - rpi 

    sampletime1 = getSamplingTime(d_a[:,0])
    sampletime2 = getSamplingTime(d_b[:,0])
    sampletime3 = getSamplingTime(d_c[:,0])
    sampletime4 = getSamplingTime(d_d[:,0])

    for line in d_a:
        signal = np.sum(line[112:569])
       # signal = np.sum(line[115:572])
        signal1.append(signal)
    signal1 = np.array(signal1)

    for line in d_b:
        signal = np.sum(line[112:569])
        #signal = np.sum(line[115:572])
        signal2.append(signal)
    signal2 = np.array(signal2)

    for line in d_c:
        signal = np.sum(line[112:569])
        #signal = np.sum(line[115:572])
        signal3.append(signal)
    signal3 = np.array(signal3)

    for line in d_d:
        signal = np.sum(line[112:569])
        #signal = np.sum(line[115:572])
        signal4.append(signal)
    signal4 = np.array(signal4)

    numberofsamples1 = np.size(signal1)
    numberofsamples2 = np.size(signal2)
    numberofsamples3 = np.size(signal3)
    numberofsamples4 = np.size(signal4)
 
    sig_fft1 = rfft(signal1)
    power1 = sig_fft1 * np.conjugate(sig_fft1)
    freq1 = rfftfreq(numberofsamples1, sampletime1)
    sig_fft2 = rfft(signal2)
    freq2 = rfftfreq(numberofsamples2, sampletime2)
    power2 = sig_fft2 * np.conjugate(sig_fft2)
    sig_fft3 = rfft(signal3)
    freq3 = rfftfreq(numberofsamples3, sampletime3)
    power3 = sig_fft3 * np.conjugate(sig_fft3)
    sig_fft4 = rfft(signal4)
    freq4 = rfftfreq(numberofsamples4, sampletime4)
    power4 = sig_fft4 * np.conjugate(sig_fft4)

    nbin = 100
    nzf1 = np.nonzero(freq1)
    freq1 = (freq1[nzf1])
    bins = np.geomspace(np.min(freq1),np.max(freq1),nbin)
    amps = np.zeros(nbin)
    i=0
    for i in range(0,nbin):
		#m = nonzero(freq >= bins[line] && find the indices in logfreq where bins[line] <= freq < bins[line+1]
        if i < (nbin-1):
            m = np.where(freq1 >= bins[i])
            n = np.where(freq1 < bins[i+1])
        else:
            m = np.where(freq1 >= bins[i])
            n = m
        try:
            if len(power1[m[0][0]:n[0][-1]]) == 0:
                amps[i] = 'nan'
            else:
                amps[i] = np.mean(np.real(power1[m[0][0]:n[0][-1]]))
        except:
            pass
 

    nzf2 = np.nonzero(freq2)
    freq2 = (freq2[nzf2])
    bins2 = np.geomspace(np.min(freq2),np.max(freq2),nbin)
    amps2 = np.zeros(nbin)
    i=0
    for i in range(0,nbin):
		#m = nonzero(freq >= bins[line] && find the indices in logfreq where bins[line] <= freq < bins[line+1]
        if i < (nbin-1):
            m = np.where(freq2 >= bins2[i])
            n = np.where(freq2 < bins2[i+1])
        else:
            m = np.where(freq2 >= bins2[i])
            n = m

        try:
            if len(power2[m[0][0]:n[0][-1]]) == 0:
                amps2[i] = 'nan'
            else:
                amps2[i] = np.mean(np.real(power2[m[0][0]:n[0][-1]]))
        except:
            pass
      
    nzf3 = np.nonzero(freq3)
    freq3 = (freq3[nzf3])
    bins3 = np.geomspace(np.min(freq3),np.max(freq3),nbin)
    amps3 = np.zeros(nbin)
    i=0
    for i in range(0,nbin):
		#m = nonzero(freq >= bins[line] && find the indices in logfreq where bins[line] <= freq < bins[line+1]
        if i < (nbin-1):
            m = np.where(freq3 >= bins3[i])
            n = np.where(freq3 < bins3[i+1])
        else:
            m = np.where(freq3 >= bins3[i])
            n = m
        try:
            if len(power3[m[0][0]:n[0][-1]]) == 0:
                amps3[i] = 'nan'
            else:
                amps3[i] = np.mean(np.real(power3[m[0][0]:n[0][-1]]))
        except:
            pass

    nzf4 = np.nonzero(freq4)
    freq4 = (freq4[nzf4])
    bins4 = np.geomspace(np.min(freq4),np.max(freq4),nbin)
    amps4 = np.zeros(nbin)
    i=0
    for i in range(0,nbin):
		#m = nonzero(freq >= bins[line] && find the indices in logfreq where bins[line] <= freq < bins[line+1]
        if i < (nbin-1):
            m = np.where(freq4 >= bins4[i])
            n = np.where(freq4 < bins4[i+1])
        else:
            m = np.where(freq4 >= bins4[i])
            n = m
        try:
            if len(power4[m[0][0]:n[0][-1]]) == 0:
                amps4[i] = 'nan'
            else:
                amps4[i] = np.mean(np.real(power4[m[0][0]:n[0][-1]]))
        except:
            pass


    titlestring = 'Power Spectra, First Differences \n'+user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours.'  
    
    if save_it:
        fig, axs = plt.subplots(2, 2, figsize=(12,8), dpi=300)
    else:
        fig, axs = plt.subplots(2, 2)
    
    axs[1,1].plot(bins4, amps4, c='firebrick')
    axs[1,1].set_title('$\delta$d = P(L,$\pi$)W - P(R,$\pi$)E') #OCRAb

    axs[0,1].plot(bins3, amps3, c='deeppink')
    axs[0,1].set_title('$\delta$c = P(L,0)E - P(R,0)W') #OCRAa

    axs[0,0].plot(bins, amps, c='b')
    axs[0,0].set_title('$\delta$a = P(L,0)E - P(L,$\pi$)W') #WMAPa

    axs[1,0].plot(bins2, amps2, c='orange')
    axs[1,0].set_title('$\delta$b = P(R,0)W - P(R,$\pi$)E') #WMAPb


    for ax in axs.flat:
        ax.set(xlabel='Frequency /Hz', ylabel='Fourier Amplitude')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.margins(x=0)
        ax.axis('scaled')
        ax.grid(c='darkgrey', which='major')
        ax.grid(c='gainsboro', which='minor')

    plt.suptitle(titlestring)
    plt.tight_layout()
    
    
  
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_PowerSpectra_first_differences.png', bbox_inches="tight")
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
            PowerSpectraFIRSTDIFF(duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 
    

#----------------------------------------------------------------------


def PowerSpectraDBLDIFF(save_it=False, first_loop=True):

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    signal1 =[]
    signal2=[]
    signal3=[]
    signal4=[]
    a1p1 = np.load(DATA_PATH+'/temp/a1p1.npy')
    a1p1 = a1p1[:,3:]
    a2p2 = np.load(DATA_PATH+'/temp/a2p2.npy')
    a2p2 = a2p2[:,3:]
    a1p2 = np.load(DATA_PATH+'/temp/a1p2.npy')
    a1p2 = a1p2[:,3:]
    a2p1 = np.load(DATA_PATH+'/temp/a2p1.npy')
    a2p1 = a2p1[:,3:]

    if np.size(a1p1[:,0]) == np.size(a1p2[:,0]):
        pass

    elif np.size(a1p1[:,0]) > np.size(a1p2[:,0]):
        a1p1 = a1p1[:-1,:]
        a2p1 = a2p1[:-1,:]
    elif np.size(a1p1[:,0]) < np.size(a1p2[:,0]):
        a1p2 = a1p2[:-1,:]
        a2p2 = a2p2[:-1,:]

    d_a = a1p2 - a1p1 # l0 - lpi 
    d_b = a2p2 - a2p1 #r0 - rpi 
    d_c = a1p2 - a2p2  # l0 - r0 
    d_d = a1p1 - a2p1  #   lpi - rpi 

    DD = (d_a - d_b)/2 # WMAP (ocra would be c-d)

   
    sampletime1 = getSamplingTime(DD[:,0])

    for line in DD:
        signal = np.sum(line[112:569])
       # signal = np.sum(line[115:572])
        signal1.append(signal)
    signal1 = np.array(signal1)

    numberofsamples1 = np.size(signal1)
 
    sig_fft1 = rfft(signal1)
    power1 = sig_fft1 * np.conjugate(sig_fft1)
    freq1 = rfftfreq(numberofsamples1, sampletime1)

    nbin = 100
    nzf1 = np.nonzero(freq1)
    freq1 = (freq1[nzf1])
    bins = np.geomspace(np.min(freq1),np.max(freq1),nbin)
    amps = np.zeros(nbin)
    i=0
    for i in range(0,nbin):
		#m = nonzero(freq >= bins[line] && find the indices in logfreq where bins[line] <= freq < bins[line+1]
        if i < (nbin-1):
            m = np.where(freq1 >= bins[i])
            n = np.where(freq1 < bins[i+1])
        else:
            m = np.where(freq1 >= bins[i])
            n = m
        try:
            if len(power1[m[0][0]:n[0][-1]]) == 0:
                amps[i] = 'nan'
            else:
                amps[i] = np.mean(np.real(power1[m[0][0]:n[0][-1]]))
        except:
            pass


    titlestring = 'Power Spectra, Double Differences \n'+user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours.'  
    
    if save_it:
        plt.figure(figsize=(12, 8), dpi=300)
    plt.plot(bins, amps, c='firebrick')
    plt.title(titlestring)
    plt.xlabel('Frequency /Hz')
    plt.ylabel('Fourier Amplitude')
    plt.xscale('log')
    plt.yscale('log')
    plt.margins(x=0)
    plt.axis('scaled')
    plt.grid(c='darkgrey', which='major')
    plt.grid(c='gainsboro', which='minor')
    plt.tight_layout()

    
    
  
 
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_PowerSpectra_double_differences.png', bbox_inches="tight")
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
            PowerSpectraDBLDIFF(duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 


#------------------------------------------------------------------


def PowerSpectraMenu():
    looper = True

    print ('')
    print ('   -------------------------------------')
    print ('   >>>    POWER SPECTRA PLOTS MENU   <<<')
    print ('   -------------------------------------')
    print ('')
    print ('   1 - All Inputs')
    print ('')
    print ('   2 - Single Input')
    print ('')
    print ('   3 - Single Differences')
    print ('')
    print ('   4 - Double Differences')
    print ('')
    print ('')
    print ('   0 - Return to Quick-look menu')
    print ('')
    choice = input('Select menu option (number): \033[0;m')
    print('\033[1;32m')
    if choice.isdigit():        
        if int(choice) ==1:
            PowerSpectra4onONE()

        elif int(choice) ==2:
            PowerSpectraONE()

        elif int(choice) ==3:
            PowerSpectraFIRSTDIFF()

        elif int(choice) ==4:
            PowerSpectraDBLDIFF()

        elif int(choice) == 0:
            looper = False
            pass

        else:
            print('\033[1;31m No such option. Please try again.\033[1;32m')
            PowerSpectraMenu()
         
    else:
        print('\033[1;31m Invalid selection by user. Please try again. \033[1;32m')
        PowerSpectraMenu()

    return looper



#########################################################################

duration_actual = np.load(DATA_PATH+'/temp/duration_actual.npy')

looper =True
while looper:
    looper = PowerSpectraMenu()

print ('\033[1;32m ')

os.system('chmod -R -f 0777 /mirror/scratch/pblack || true')

