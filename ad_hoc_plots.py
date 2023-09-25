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
import matplotlib.dates as mdates

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

def get_minmax(band):

    ind = np.argmin(band[:,483-20:483+21])
    indx = np.argmax(band[:,483-20:483+21])
    band = band[:,483-20:483+21].flatten()
    bandmin = band[ind] 
    bandmax = band[indx]   

    return bandmin, bandmax

#--------------------------------------------------------------------


def time_series():

    quickload = np.load(DATA_PATH+'/temp/quickload.npy')

    if quickload:
        array_time = np.load(DATA_PATH+'/temp/time_array.npy') 
        array_time = Time(array_time,format='mjd',scale='utc',precision=9)
        array_time.format = 'iso'
        time2 = array_time.tt.datetime


    else:
        try:
            a1p1B = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
            obsheader = np.load(DATA_PATH+'/temp/obshdr.npy')
            MJD = Time(obsheader[0,8],format='mjd', scale='utc', precision=9)
            time_p1 = MJD + TimeDelta(a1p1B[:,0].astype(float), format='sec') 
            time_p1.format = 'iso'
            time2 = time_p1.tt.datetime
        except:   #possibly a nan in the a1p1 array causing a problem, so try a1p2 instead
            a1p1B = np.load (DATA_PATH+'/temp/a1p2_binned.npy')
            obsheader = np.load(DATA_PATH+'/temp/obshdr.npy')
            MJD = Time(obsheader[0,8],format='mjd', scale='utc', precision=9)
            time_p1 = MJD + TimeDelta(a1p1B[:,0].astype(float), format='sec') 
            time_p1.format = 'iso'
            time2 = time_p1.tt.datetime


    return time2


##################################################################################

#~~~~~~~~~~~~~~~~~~~~#~~~~~~~~JORDAN NORRIS~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~#

REFPIX = 359
REFFREQ = 1413500976.5625 # CENTRAL FREQUENCY
CHNSPACE = 54931.640625 #SPACING BETWEEN BINS
LOC = EarthLocation(lat=53.234338*u.deg, lon=-2.305018*u.deg, height=77*u.m)

def getBinFromFreq(freq, refFREQ, refPIX, chnSPACE):
    """
    INPUTS FREQUENCY AND RETURNS CORRESPONDING BIN NUMBER (MAY NEED ADDITION
    TO BE INLINE WITH FULL DATA). FOR THE a1p1_binned.npy arrays
    
    Parameters
    ----------
    freq : FLOAT
        FLOAT OF FREQUENCY IN Hz
    refFREQ : FLOAT
        FLOAT OF REFERENCE FREQUENCY WHICH IS THE CENTRAL FREQUENCY IN THE BINS.
    refPIX : INTEGER
        INDEX OF THE CENTRAL BIN (IN JUST THE FREQ ARRAY)
    chnSPACE : TYPE
        DESCRIPTION.

    Returns
    -------
    integer corresponding to the index where the power data for said frequency
    is located.

    """
    index = ((freq - refFREQ) / chnSPACE) + refPIX - 1
    index = int(index) + 4
    
    return index
# print(getBinFromFreq(1421116000.0000, REFFREQ, REFPIX, CHNSPACE))
# print(getBinFromFreq(1422764000.0000, REFFREQ, REFPIX, CHNSPACE))

def getFreqFromBin(index, refFREQ, refPIX, chnSPACE):
    """
    Converse of above as well but in this case bin number applies to a1p1_binned
    numpy array
    """
    index = index - 3
    freq = refFREQ + (chnSPACE * (index + 1 - refPIX))
    return freq


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

#-------------------------------------------------------------------------
#(date, user_hour, user_minute, user_sample, user_duration)
#file_table(fits_name, begs, ends, start_times, end_times, samFIRSTs, samLASTs, multi_day, same_day, same_run, corrupts)
# the frequency channels within the digital bandpass are 112 to 568

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

####################################################################################


def xaxis(range_example, duration_actual): #range example being a1p1B etc

    binnable = np.load(DATA_PATH+'/temp/rebinnable.npy')
    if binnable:
        range_example = np.load(DATA_PATH+'/temp/one_wire.npy') 
    freq = np.load(DATA_PATH+'/temp/freq.npy')
    obsheader = np.load(DATA_PATH+'/temp/obshdr.npy')
    MJD = Time(obsheader[0,8],format='mjd', scale='utc', precision=4)
    range_example = np.array(range_example)
    try:
        sample_to_datetime = np.size(range_example[:,0])
        time_p1 = MJD + TimeDelta(range_example[:,0].astype(float), format='sec') 

    except:
        sample_to_datetime = np.size(range_example[:])
        time_p1 = MJD + TimeDelta(range_example[:].astype(float), format='sec') 
    time_p1.format = 'iso'
    time_p1.out_subfmt = 'date_hm'
    time_p2 = np.asarray(time_p1.strftime('%H:%M'))   #array of HH:MM
    time_p2 = np.where(time_p2 != '00:00', time_p2, time_p1.strftime('%H:%M %d')) #at midnight add day number
    time_p1 = np.asarray(time_p1.strftime('%M'))
    time_p1 = np.where(time_p1 == '00') #these two lines find round hours
 
    mins = time_p2[time_p1].tolist() #values of hours for the tick labels
    first_tick = time_p1[0][:].tolist() #list of bin numbers for the hours tick labels
    
    del time_p1, time_p2, range_example

    return mins, first_tick, sample_to_datetime, freq
    
#--------------------------------------------------------------------------------

def FourFreqVTime(frequency, save_it=False, first_loop=True):
    

    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    band11 = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
    band12 = np.load (DATA_PATH+'/temp/a1p2_bandpass.npy')
    band21 = np.load (DATA_PATH+'/temp/a2p1_bandpass.npy')
    band22 = np.load (DATA_PATH+'/temp/a2p2_bandpass.npy')

    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours. Average Bandpass'
    
    if save_it:
        fig, axs = plt.subplots(2, 2, figsize=(12,8), dpi=300)
    else:
        fig, axs = plt.subplots(2, 2)

    Xtime = np.arange(0,np.size(p11),1)
    ave = np.mean(band22[:,463:504])
    avf = np.mean(band22[:,483])
    axs[0, 0].plot(Xtime, band11[:,463:504]/ave, color ='g', linewidth=1)
    axs[0, 0].plot(Xtime, band11[:,483]/avf, color ='b', linewidth=1)
    #axs[0, 0].plot(frequency[0:112], band11[0:112], color ='tab:gray', linewidth=0.5)
    #axs[0, 0].plot(frequency[569:-1], band11[569:-1], color ='tab:gray', linewidth=0.5)
    #axs[0,0].set_ylim((0.9,None))
    #axs[0,0].set_xlim((frequency[75],frequency[615]))
    axs[0, 0].set_title('P(l,$\pi$) W')
    #axs[0,0].axvline(x=frequency[112],linestyle='--', color='tab:gray', linewidth=1)
    #axs[0,0].axvline(x=frequency[568],linestyle='--', color='tab:gray', linewidth=1)
    axs[0,0].margins(x=0)
    axs[0, 1].plot(Xtime, band12[:,463:504]/ave, color ='g', linewidth=1)
    axs[0, 1].plot(Xtime, band12[:,483]/avf, color ='b', linewidth=1)
    #axs[0, 1].plot(frequency[0:112], band12[0:112], color ='tab:gray', linewidth=0.5)
    #axs[0, 1].plot(frequency[569:-1], band12[569:-1], color ='tab:gray', linewidth=0.5)
    #axs[0,1].set_ylim((0.9,None))
    #axs[0,1].set_xlim((frequency[75],frequency[615]))
    axs[0, 1].set_title('P(l,0) E')
    #axs[0,1].axvline(x=frequency[112],linestyle='--', color='tab:gray', linewidth=1)
    #axs[0,1].axvline(x=frequency[568],linestyle='--', color='tab:gray', linewidth=1)
    axs[0,1].margins(x=0)
    axs[1, 0].plot(Xtime, band22[:,463:504]/ave, color ='g', linewidth=1)
    axs[1, 0].plot(Xtime, band22[:,483]/avf, color ='b', linewidth=1)
    #axs[1, 0].plot(frequency[0:112], band21[0:112], color ='tab:gray', linewidth=0.5)
    #axs[1, 0].plot(frequency[569:-1], band21[569:-1], color ='tab:gray', linewidth=0.5)
    #axs[1,0].set_ylim((0.9,None))
    #axs[1,0].set_xlim((frequency[75],frequency[615]))
    axs[1, 0].set_title('P(r,0) W')
    #axs[1,0].axvline(x=frequency[112],linestyle='--', color='tab:gray', linewidth=1)
    #axs[1,0].axvline(x=frequency[568],linestyle='--', color='tab:gray', linewidth=1)
    axs[1,0].margins(x=0)
    axs[1, 1].plot(Xtime, band21[:,463:504]/ave, color ='g', linewidth=1)
    axs[1, 1].plot(Xtime, band21[:,483]/avf, color ='b', linewidth=1)
    #axs[1, 1].plot(frequency[0:112], band22[0:112], color ='tab:gray', linewidth=0.5)
    #axs[1, 1].plot(frequency[569:-1], band22[569:-1], color ='tab:gray', linewidth=0.5)
    #axs[1,1].set_ylim((0.9,None))
    #axs[1,1].set_xlim((frequency[75],frequency[615]))
    axs[1, 1].set_title('P(r,$\pi$) E')
    #axs[1,1].axvline(x=frequency[112],linestyle='--', color='tab:gray', linewidth=1)
    #axs[1,1].axvline(x=frequency[568],linestyle='--', color='tab:gray', linewidth=1)
    axs[1,1].margins(x=0)
    #Axis [1, 1]
    for ax in axs.flat:
        ax.set(xlabel='Time / Mins', ylabel='Normalised Power')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.close()

    a1p1B = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
    mins,first_tick, sample_to_datetime, frequency = xaxis(a1p1B, duration_actual)
    del a1p1B

    plt.xticks(ticks=first_tick, labels=mins, rotation=270)
    if duration_actual > 48 and duration_actual < 150:
        plt.xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),60))
    elif duration_actual > 150:
        pass
    else:
        plt.xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),15))

 #   plt.plot(Xtime, band11[:,463:504]/ave, color ='r', linewidth=1, label='HI Full Width')
    plt.plot(Xtime,band11[:,483]/avf, color ='b', linewidth=1, label='1420 MHz')

    plt.margins(x=0)
    plt.legend()
    plt.title('P(l,$\pi$) W Frequency vs Time')
    plt.xlabel('Time')
    plt.ylabel('Normalised Power')
  

    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_freq-vs-time.png', bbox_inches="tight")
        print('')
        print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
    else:
        plt.show()
    plt.close()
    
    if first_loop:
        print ('')
        save = input ('Do you want to save a printer friendly copy of this plot? (Y/N): ')
        if str(save) == 'N' or str(save) == 'n':
            pass
        elif str(save) == 'Y' or str(save) == 'y':
            save_it=True
            first_loop = False
            FourFreqVTime(frequency, save_it, first_loop)
 
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass




#-----------------------------------------------------------------------

def bandpassANI(frequency):

    #animation of bandpass by bin

    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    del p11
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    date_time = Time(user_inputs[0]+'T'+user_inputs[3], format='isot', scale='utc', precision=4)

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
    print('Please specify run-speed of animation, recommend < 1 second per bin')
    print('')
    which_speed = input('Seconds per bin (e.g. 0.5): ')
    try:
       which_speed = float(which_speed)
    except:
       which_input = 'eggs'

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    mask_width = parameters[4].astype(np.float)
    mask_width = 1+int(mask_width/2)
    
    if which_input == 1:
        band11 = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
        if parameters[2]=='True':
            band11[:,(485-mask_width):(484+mask_width)] = 'nan'
        for i in range (5,(np.size(band11[:,0]))):
            plt.plot(frequency[112:569], (band11[i,112:569]/band11[i-5,112:569]), color ='k' , linewidth=1)
            plt.xlabel('Frequency')
            plt.ylabel('Power')
            #if parameters[1]=='True':
            #    plt.ylim((0.9,1.10))
           # else:
             #   plt.ylim((0.75,1.5))
            plt.ylim((0.98,1.01))
            if parameters[6] == 'True':
                plt.axvline(x=frequency[484-mask_width],linestyle=':', color='salmon', linewidth=1)
                plt.axvline(x=frequency[484+mask_width],linestyle=':', color='c', linewidth=1)
                plt.axvline(x=frequency[484],linestyle='-', color='navy', linewidth=2, label='1402.405 MHz')
            plt.margins(x=0)
            roll_time = date_time + TimeDelta(i*60, format='sec')
            time_str = roll_time.strftime("%H:%M") 
            plt.title(user_inputs[0]+' '+time_str+' P(l,$\pi$) W'+str(i))
            plt.pause(which_speed)
            plt.clf()
        del band11
    
    elif which_input ==2:
        band12 = np.load (DATA_PATH+'/temp/a1p2_bandpass.npy')
        if parameters[2]=='True':
            band12[:,(485-mask_width):(484+mask_width)] = 'nan'
        for i in range (0,np.size(band12[:,0])):
            plt.plot(frequency[112:569], band12[i,112:569], color ='k' , linewidth=1)
            plt.xlabel('Frequency')
            plt.ylabel('Power')
            if parameters[1]=='True':
                plt.ylim((0.9,1.10))
            else:
                plt.ylim((0.75,1.5))
            if parameters[6] == 'True':
                plt.axvline(x=frequency[484-mask_width],linestyle=':', color='salmon', linewidth=1)
                plt.axvline(x=frequency[484+mask_width],linestyle=':', color='c', linewidth=1)
                plt.axvline(x=frequency[484],linestyle='-', color='navy', linewidth=2, label='1402.405 MHz')
            plt.margins(x=0)
            roll_time = date_time + TimeDelta(i*60, format='sec')
            time_str = roll_time.strftime("%H:%M") 
            plt.title(user_inputs[0]+' '+time_str+' P(l,0) E W')
            plt.pause(which_speed)
            plt.clf()
        del band12
    
    elif which_input ==3:
        band21 = np.load (DATA_PATH+'/temp/a2p1_bandpass.npy')
        if parameters[2]=='True':
            band21[:,(485-mask_width):(484+mask_width)] = 'nan'
        for i in range (0,np.size(band21[:,0])):
            plt.plot(frequency[112:569], band21[i,112:569], color ='k' , linewidth=1)
            plt.xlabel('Frequency')
            plt.ylabel('Power')
            if parameters[1]=='True':
                plt.ylim((0.9,1.10))
            else:
                plt.ylim((0.75,1.5))
            if parameters[6] == 'True':
                plt.axvline(x=frequency[484-mask_width],linestyle=':', color='salmon', linewidth=1)
                plt.axvline(x=frequency[484+mask_width],linestyle=':', color='c', linewidth=1)
                plt.axvline(x=frequency[484],linestyle='-', color='navy', linewidth=2, label='1402.405 MHz')
            plt.margins(x=0)
            roll_time = date_time + TimeDelta(i*60, format='sec')
            time_str = roll_time.strftime("%H:%M") 
            plt.title(user_inputs[0]+' '+time_str+' P(r,$\pi$) E')
            plt.pause(which_speed)
            plt.clf()
        del band21
    
    elif which_input ==4:
        band22 = np.load (DATA_PATH+'/temp/a2p2_bandpass.npy')
        if parameters[2]=='True':
            band22[:,(485-mask_width):(484+mask_width)] = 'nan'
        for i in range (0,np.size(band22[:,0])):
            plt.plot(frequency[112:569], band22[i,112:569], color ='k' , linewidth=1)
            plt.xlabel('Frequency')
            plt.ylabel('Power')
            if parameters[1]=='True':
                plt.ylim((0.9,1.10))
            else:
                plt.ylim((0.75,1.5))
            if parameters[6] == 'True':
                plt.axvline(x=frequency[484-mask_width],linestyle=':', color='salmon', linewidth=1)
                plt.axvline(x=frequency[484+mask_width],linestyle=':', color='c', linewidth=1)
                plt.axvline(x=frequency[484],linestyle='-', color='navy', linewidth=2, label='1402.405 MHz')
            plt.margins(x=0)
            roll_time = date_time + TimeDelta(i*60, format='sec')
            time_str = roll_time.strftime("%H:%M") 
            plt.title(user_inputs[0]+' '+time_str+' P(r,0) W')
            plt.pause(which_speed)
            plt.clf()
        del band22

    else:
        print('\033[1;31m Input not recognised, please try again. \033[1;32m')
        bandpassANI(frequency)

    plt.close()


#-----------------------------------------------------------------------------------

def show_Arshi():

    a1p1B = np.load(DATA_PATH+'/temp/a1p1_binned.npy')

    print(np.shape(a1p1B))
    input()
    print(a1p1B[0,0])
    input()
    print(a1p1B[0,1])
    input()
    print(a1p1B[0,2])
    input()
    print(a1p1B[0,3:7])
    input()

def exportArshi():

    duration_actual = np.load(DATA_PATH+'/temp/duration_actual.npy')
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    folderstring = user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)

    print('')
    print ('Exporting files to /mirror/scratch/arastogi/exports/')

    try:
        os.system('mkdir /mirror/scratch/arastogi/exports/'+folderstring)
        os.system('cp /mirror/scratch/pblack/temp/obshdr.npy /mirror/scratch/arastogi/exports/'+folderstring)
        os.system('cp /mirror/scratch/pblack/temp/a1p1_binned.npy /mirror/scratch/arastogi/exports/'+folderstring)
        os.system('cp /mirror/scratch/pblack/temp/a1p2_binned.npy /mirror/scratch/arastogi/exports/'+folderstring)
        os.system('cp /mirror/scratch/pblack/temp/a2p1_binned.npy /mirror/scratch/arastogi/exports/'+folderstring)
        os.system('cp /mirror/scratch/pblack/temp/a2p2_binned.npy /mirror/scratch/arastogi/exports/'+folderstring)
        os.system('cp /mirror/scratch/pblack/temp/freq.npy /mirror/scratch/arastogi/exports/'+folderstring)
        print('')
        print('a1p1_binned etc:')
        print('There are 721 columns in each array.  The first is sample time (in seconds past midnight from the day the observing run began), the second is switch phase (0 or 1) and the third is the elevation of the horns in degrees.  The remaining 718 columns correspond to the bandpass frequency-bins - each containing the power recorded at the given sample time.  Each row in the array is another 1-min averaged time-bin.')
        print('')
        print('freq:')
        print('A list of the frequencies accociated with each of the 718 frequency bins in MHz')
        print('')
        print('obshdr:')
        print('There are 9 columns corresponding to, observatory, observer, observation mode, longitude, latitude, altitude (above sea-level), azimuth, obs_date, MJD on which observations began.') 
        print('')
        print('Frequency bin containing HI line at 1420.405 MHz is 483.')
        print('')

    except:
        print ('\033[1;31m Exporting failed \033[0;0m')

    os.system('chmod -R -f 0777 /mirror/scratch/arastogi/ || true')



def rawData(duration_actual,save_it=False, first_loop=True):


    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    date_time = Time(user_inputs[0]+'T'+user_inputs[3], format='isot', scale='utc', precision=4)

    p11 = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
    p12 = np.load (DATA_PATH+'/temp/a1p2_bandpass.npy')
    p21 = np.load (DATA_PATH+'/temp/a2p1_bandpass.npy')
    p22 = np.load (DATA_PATH+'/temp/a2p2_bandpass.npy')

  #  p11 = np.load(DATA_PATH+'/temp/a1p1.npy')
   # p12 = np.load(DATA_PATH+'/temp/concats/breaky_a1p2_1.npy')
    #p21 = np.load(DATA_PATH+'/temp/concats/breaky_a2p1_1.npy')
   # p22 = np.load(DATA_PATH+'/temp/concats/breaky_a2p2_1.npy')


    if parameters[8] == 'True':
        p11 = p11/np.mean(p11)
        p12 = p12/np.mean(p12)
        p21 = p21/np.mean(p21)
        p22 = p22/np.mean(p22)

    #pX = ((p11[33,:] + p11[34,:])/2)  /  ((p11[31,:]+p11[36,:])/2)
    a1p1B = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
    mins,first_tick, sample_to_datetime, frequency = xaxis(a1p1B, duration_actual)


  #  for i in range(6,(np.size(p11))):
   #     pX = p11[i,:] / p11[i-5,:]
    #    peaks_ratio = (np.mean(pX[380:390]) / np.mean(pX[322:332]))
     #   if peaks_ratio > 1.0008:
      #      plt.plot(pX[112:569], color='r', linewidth=1, label='Modulation P(L,$\pi$) W') #frequency[112:569],
       #     roll_time = date_time + TimeDelta(i*60, format='sec')
        #    time_str = roll_time.strftime("%H:%M") 
         #   plt.title(user_inputs[0]+' '+time_str+' P(l,$\pi$) W'+str(i)+' '+str(peaks_ratio))
          #  plt.show()
        #    plt.close()
   #     else:
    #        pass

    dev =[]
    time = []
    w_input = 'L_0'
    p11 = p12

    for i in range(1,(np.size(p11[:,0])-1)):
        this_line = p11[i,:]
        last_line = p11[i-1,:]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(p11[0,:])#####
        rooted = math.sqrt(summed)
        dev.append(float(rooted))
        time.append(a1p1B[i,0])
    #    time.append(p11[i,0])

    p11 = p11
    plt.figure(figsize=(12, 8), dpi=300)
    mins,first_tick, sample_to_datetime, frequency = xaxis(time, duration_actual)
    plt.xticks(ticks=first_tick, labels=mins, rotation=270)
    plt.xticks(ticks=np.arange(first_tick[0],sample_to_datetime,15))

    plt.plot(dev, c='r')   
   # plt.plot(dev, c='r')     
    plt.ylabel('RMS')
    plt.xlabel('Time')
    plt.title(user_inputs[0]+' Time-bin to Time-bin RMS Deviations')
    plt.margins(x=0)
    plt.ylim(((0),(0.15)))
  #  plt.ylim(((0.0),(1.5)))
    plt.grid(c='gainsboro')
    plt.savefig(user_inputs[6]+user_inputs[0]+'_RMSdeviations_'+w_input+'.png', bbox_inches="tight")
    plt.show()
    plt.close()
    input('Break here for RMs only.')
    common_value = stats.mode(dev)
    pp_times = []
    pp_bins = []
    pwr_ratio = []
    pwr_ratio.append('Full Bandpass Power Ratio')
    pwr_ratio_goods = []
    pwr_ratio_goods.append('1400-1425 MHz Power Ratio')
    pwr_ratio_HI = []
    pwr_ratio_HI.append('HI-bin Power Ratio')
    pwr_ratio_CW = []
    pwr_ratio_CW.append('CW-bin Power Ratio')



    for i in range (0, np.size(dev)):  #find the common power level (not the right approach)
        if dev[i] == common_value[0]:
            base = p11[i,:]
            base_pwr = np.sum(p11[i,:])
            base_good = np.sum(p11[i,112:569])
            base_HI = np.sum(p11[i,483])
            base_CW = np.sum(p11[i,586])
            break
        else:
            pass

    mod_pwr = []
    mod_pwr.append('PP Full Bandpass Power')
    mod_good =  []
    mod_good.append('PP 1400-1425MHz Power')
    mod_HI =  []
    mod_HI.append('PP HI-bin Power')
    mod_CW =  []
    mod_CW.append('PP CW-bin Power')
    mod_HI_plus = []
    mod_HI_plus.append('PP HI-bin + 10bins Power')
    mod_HI_minus = []
    mod_HI_minus.append('PP HI_bin - 10 bins Power')
    base_pwr = []
    base_pwr.append('Full Bandpass Power (Normal)')
    base_good =  []
    base_good.append('1400-1425MHz Power (Normal)')
    base_HI =  []
    base_HI.append('HI-bin Power (Normal)')
    base_CW =  []
    base_CW.append('CW-bin Power (Normal)')
    base_HI_plus = []
    base_HI_plus.append('HI-bin + 10 bins Power (Normal)')
    base_HI_minus = []
    base_HI_minus.append('HI-bin - 10 bins Power (Normal)')
    pwr_ratio_HI_plus = []
    pwr_ratio_HI_plus.append('HI-bin + 10 bins Power Ratio')
    pwr_ratio_HI_minus = []
    pwr_ratio_HI_minus.append('HI-bin - 10 bins Power Ratio')
    

    for i in range (0, np.size(dev)):  #find power levels of mod state and ratios to base
        if dev[i] > (common_value[0] *1.4):
            mod_pwr.append(np.sum(p11[i,:]))
            mod_good.append(np.sum(p11[i,112:569]))
            mod_HI.append(np.sum(p11[i,483]))
            mod_HI_minus.append(np.sum(p11[i,483:494]))
            mod_HI_plus.append(np.sum(p11[i,473:484]))
            mod_CW.append(np.sum(p11[i,586]))

            base_pwr.append(np.sum(p11[i-5,:]))
            base_good.append(np.sum(p11[i-5,112:569]))
            base_HI.append(np.sum(p11[i-5,483]))
            base_CW.append(np.sum(p11[i-5,586]))
            base_HI_minus.append(np.sum(p11[i-5,483:494]))
            base_HI_plus.append(np.sum(p11[i-5,473:484]))

            pwr_ratio.append(mod_pwr[-1]/base_pwr[-1])
            pwr_ratio_goods.append(mod_good[-1]/base_good[-1])
            pwr_ratio_HI.append(mod_HI[-1]/base_HI[-1])
            pwr_ratio_CW.append(mod_CW[-1]/base_CW[-1])
            pwr_ratio_HI_plus.append(mod_HI_plus[-1]/base_HI_plus[-1])
            pwr_ratio_HI_minus.append(mod_HI_minus[-1]/base_HI_minus[-1])

            pp_times.append(time[i])
            pp_bins.append(i)

    
   # input('wait')
    obsheader = np.load(DATA_PATH+'/temp/obshdr.npy')
    MJD = Time(obsheader[0,8],format='mjd', scale='utc', precision=4)

    time_p1 = MJD + TimeDelta(np.asarray(pp_times), format='sec') 
    time_p1.format = 'iso'
    time_p1.out_subfmt = 'date_hm'
    time_p2 = np.asarray(time_p1.strftime('%H:%M'))   #array of HH:MM

    pptt = []
    pptt.append('Time')
    all_ratio_array = []
    all_diff_array = []
    f, axarr = plt.subplots(2,1, figsize=(12,8), dpi=300) 
    for i in range (0,np.size(pp_bins)):
        try:
            pX = p11[pp_bins[i],:] /  p11[(pp_bins[i])-5,:]   #base
            pY = p11[pp_bins[i],:] -  p11[(pp_bins[i])-5,:]
            all_ratio_array.append(pX)
            all_diff_array.append(pY)
            axarr[0].plot(frequency, pX, c='r')
            axarr[0].set_title('RATIO PP/normal')
            axarr[0].margins(x=0)
            axarr[0].set_xlabel('Frequency / MHz')
            axarr[1].set_xlabel('Frequency / MHz')
            axarr[0].set_ylabel('Ratio')
            axarr[1].set_ylabel('Power / a.u.')
            axarr[1].plot(frequency, pY, c='b')
            axarr[1].set_title('DIFFERENCE PP - normal')
            axarr[1].margins(x=0)
            plt.suptitle('Potential PP '+user_inputs[0]+'_'+time_p2[i]+'_'+w_input)
            plt.tight_layout()

            #plt.xlabel('Frequency / MHz')
            #plt.axvline(x=1400, c='c')
           # plt.axvline(x=1425, c='c')
           # plt.axvline(x=1426.03, c='m') #586 perhaps
            #plt.axvline(x=1420.405, c='g')  #483
            #plt.ylabel('Ratio (modulation state / standard state')
            plt.savefig(user_inputs[6]+user_inputs[0]+'_'+time_p2[i]+'_PP_'+w_input+'.png', bbox_inches="tight")
            
            #plt.close()
            axarr[0].cla()
            axarr[1].cla()
            axarr[0].cla()
            axarr[1].cla()
            pptt.append(time_p2[i])
        except:
            pptt.append(time_p2[i])

    np.save(user_inputs[6]+user_inputs[0]+'_'+w_input+'_RATIO.npy', all_ratio_array)
    np.save(user_inputs[6]+user_inputs[0]+'_'+w_input+'_DIFF.npy', all_diff_array)
    plt.close()
    PP_table = np.column_stack((pptt, base_pwr, mod_pwr, pwr_ratio, base_good, mod_good, pwr_ratio_goods, base_HI, mod_HI, pwr_ratio_HI, base_HI_minus, mod_HI_minus, pwr_ratio_HI_minus, base_HI_plus, mod_HI_plus, pwr_ratio_HI_plus, base_CW, mod_CW, pwr_ratio_CW))

    np.savetxt(user_inputs[6]+user_inputs[0]+"_PP_modulation_datatable_"+w_input+".csv", PP_table, delimiter=",", fmt='%s')

    #input('stop')  
 
      #  peaks_ratio = (np.mean(pX[380:390]) / np.mean(pX[322:332]))
       # if peaks_ratio > 1.0008:
        #    plt.plot(pX[112:569], color='r', linewidth=1, label='Modulation P(L,$\pi$) W') #frequency[112:569],
         #   roll_time = date_time + TimeDelta(i*60, format='sec')
          #  time_str = roll_time.strftime("%H:%M") 
           # plt.title(user_inputs[0]+' '+time_str+' P(l,$\pi$) W'+str(i)+' '+str(peaks_ratio))
           # plt.show()
        #    plt.close()
       # else:
        #    pass

   # input('loop done')

    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours. Raw Data (Power Signal)'
    a1p1B = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
    mins,first_tick, sample_to_datetime, frequency = xaxis(a1p1B, duration_actual)
    del a1p1B
    #load first

    if save_it:
        plt.figure(figsize=(12, 8), dpi=300)
   

    plt.plot(frequency[112:569], pX[112:569], color='r', linewidth=1, label='Modulation P(L,$\pi$) W') # linestyle=(0,(5,1)),


    plt.xlabel('Frequency')
    if parameters[8] == 'True':
        plt.ylabel('Normalised Power')
    else:
        plt.ylabel('Power / a.u.')

    if save_it:
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        plt.legend(loc="upper right", fontsize=11)
    #plt.title(title_string)
    plt.tight_layout()
    plt.margins(x=0)
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_modulation.png', bbox_inches="tight")
    else:
        plt.show()
    plt.close()
    
    if first_loop:
        print ('')
        save = input ('Do you want to save a printer friendly copy of this plot? (Y/N): ')
        if str(save) == 'N' or str(save) == 'n':
            pass
        elif str(save) == 'Y' or str(save) == 'y':
            save_it=True
            first_loop = False
            rawData(duration_actual, save_it, first_loop)
 
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 

    del p11,p12,p21,p22



#------------------------------------------------------------------------------------


def gather_PP():

    freq = np.load(DATA_PATH+'/temp/freq.npy')

    Lpi_DIFFs = sorted(glob.glob('/scratch/nas_lbass/analysis/2023_01/PP arrays/Lpi/*DIFF*.npy'))
    Rpi_DIFFs = sorted(glob.glob('/scratch/nas_lbass/analysis/2023_01/PP arrays/Rpi/*DIFF*.npy'))
    L0_DIFFs = sorted(glob.glob('/scratch/nas_lbass/analysis/2023_01/PP arrays/L0/*DIFF*.npy'))
    R0_DIFFs = sorted(glob.glob('/scratch/nas_lbass/analysis/2023_01/PP arrays/R0/*DIFF*.npy'))

    Lpi_RATIOs = sorted(glob.glob('/scratch/nas_lbass/analysis/2023_01/PP arrays/Lpi/*RATIO*.npy'))
    Rpi_RATIOs = sorted(glob.glob('/scratch/nas_lbass/analysis/2023_01/PP arrays/Rpi/*RATIO*.npy'))
    L0_RATIOs = sorted(glob.glob('/scratch/nas_lbass/analysis/2023_01/PP arrays/L0/*RATIO*.npy'))
    R0_RATIOs = sorted(glob.glob('/scratch/nas_lbass/analysis/2023_01/PP arrays/R0/*RATIO*.npy'))


    for i in range (0,np.size(Lpi_DIFFs)):
        if i == 0:
            Lpi_DIFF_array = np.load(Lpi_DIFFs[i])
            Rpi_DIFF_array = np.load(Rpi_DIFFs[i])
            L0_DIFF_array = np.load(L0_DIFFs[i])
            R0_DIFF_array = np.load(R0_DIFFs[i])
            Lpi_RATIO_array = np.load(Lpi_RATIOs[i])
            Rpi_RATIO_array = np.load(Rpi_RATIOs[i])
            L0_RATIO_array = np.load(L0_RATIOs[i])
            R0_RATIO_array = np.load(R0_RATIOs[i])

        else:
             Lpi_DIFF_array = np.row_stack((Lpi_DIFF_array, np.load(Lpi_DIFFs[i])))
             Rpi_DIFF_array = np.row_stack((Rpi_DIFF_array, np.load(Rpi_DIFFs[i])))
             L0_DIFF_array = np.row_stack((L0_DIFF_array, np.load(L0_DIFFs[i])))
             R0_DIFF_array = np.row_stack((R0_DIFF_array, np.load(R0_DIFFs[i])))
             Lpi_RATIO_array = np.row_stack((Lpi_RATIO_array, np.load(Lpi_RATIOs[i])))
             Rpi_RATIO_array = np.row_stack((Rpi_RATIO_array, np.load(Rpi_RATIOs[i])))
             L0_RATIO_array = np.row_stack((L0_RATIO_array, np.load(L0_RATIOs[i])))
             R0_RATIO_array = np.row_stack((R0_RATIO_array, np.load(R0_RATIOs[i])))

        
    Lpi_DIFF_array = np.mean(Lpi_DIFF_array, axis=0)
    Rpi_DIFF_array = np.mean(Rpi_DIFF_array, axis=0)
    L0_DIFF_array = np.mean(L0_DIFF_array, axis=0)
    R0_DIFF_array = np.mean(R0_DIFF_array, axis=0)

    Lpi_RATIO_array = np.mean(Lpi_RATIO_array, axis=0)
    Rpi_RATIO_array = np.mean(Rpi_RATIO_array, axis=0)
    L0_RATIO_array = np.mean(L0_RATIO_array, axis=0)
    R0_RATIO_array = np.mean(R0_RATIO_array, axis=0)

    L_mean_DIFF = Lpi_DIFF_array + L0_DIFF_array / 2
    R_mean_DIFF = Rpi_DIFF_array + R0_DIFF_array / 2
    L_mean_RATIO = Lpi_RATIO_array + L0_RATIO_array / 2
    R_mean_RATIO = Rpi_RATIO_array + R0_RATIO_array / 2

    f, axarr = plt.subplots(2,1) 

    #plt.plot(freq, L_mean_DIFF)
    #plt.show()
    #plt.close()
   # plt.plot(freq, R_mean_DIFF, c='m', label='RPG Right mean')
   # plt.plot(freq, L_mean_DIFF, c='r', label='RPG Left mean')

  #  np.save('/scratch/nas_lbass/analysis/2022_12/PPdec_avg_diff_Lpi.npy', Lpi_DIFF_array)
   # np.save('/scratch/nas_lbass/analysis/2022_12/PPdec_avg_diff_L0.npy', L0_DIFF_array)  
    #np.save('/scratch/nas_lbass/analysis/2022_12/PPdec_avg_diff_Rpi.npy', Rpi_DIFF_array)  
#    np.save('/scratch/nas_lbass/analysis/2022_12/PPdec_avg_diff_R0.npy', R0_DIFF_array)  
 #   np.save('/scratch/nas_lbass/analysis/2022_12/PPdec_avg_ratio_Lpi.npy', Lpi_RATIO_array)
  #  np.save('/scratch/nas_lbass/analysis/2022_12/PPdec_avg_ratio_L0.npy', L0_RATIO_array)  
   # np.save('/scratch/nas_lbass/analysis/2022_12/PPdec_avg_ratio_Rpi.npy', Rpi_RATIO_array)  
    #np.save('/scratch/nas_lbass/analysis/2022_12/PPdec_avg_ratio_R0.npy', R0_RATIO_array) 

    axarr[1].plot(freq, Rpi_DIFF_array, label='P(R,$\pi$) E', c='firebrick')
    axarr[1].plot(freq, L0_DIFF_array, label='P(L,0) E', c='deeppink')
    axarr[1].plot(freq, R0_DIFF_array, label='P(R,0) W', c='c')
    axarr[1].plot(freq, Lpi_DIFF_array, label='P(L,$\pi$) W', c='b')

    axarr[1].set_title('DIFFERENCE PP - normal')
    axarr[1].legend()
    axarr[1].margins(x=0)


    axarr[0].plot(freq, Rpi_RATIO_array, label='P(R,$\pi$) E', c='firebrick')
    axarr[0].plot(freq, L0_RATIO_array, label='P(L,0) E', c='deeppink')
    axarr[0].plot(freq, R0_RATIO_array, label='P(R,0) W', c='c')
    axarr[0].plot(freq, Lpi_RATIO_array, label='P(L,$\pi$) W', c='b')

    axarr[0].set_title('RATIO  PP/normal')
    axarr[0].legend()
    axarr[0].margins(x=0)
    #plt.plot(freq, Lpi_)
    axarr[0].set_xlabel('Frequency / MHz')
    axarr[1].set_xlabel('Frequency / MHz')
    axarr[0].set_ylabel('Ratio')
    axarr[1].set_ylabel('Power / a.u.')
    plt.tight_layout()
    plt.suptitle('10th-13th January Average PP')
    plt.show()
    plt.close()


#-----------------------------------------------------------------


def HI_waterfall(save_it=False, first_loop=True):

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

    Ind = int(483) #483 for HI bin
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')

    a1p1B = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
    mins,first_tick, sample_to_datetime, frequency = yaxis(a1p1B, duration_actual)
    del a1p1B

    frequency = np.round(frequency, decimals=3)

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False


    if parameters[8]=='True' or parameters[1]=='True':
        if parameters[1]=='True' and parameters[8]=='True':
            title_string = 'Spectrogram, HI, All Inputs. '+user_inputs[0]+' Global Normalised, Normalised \n HI Indicators at frequency-bin centres for 1420.367 MHz & 1420.422 MHz'
        elif parameters[1]=='True':
            title_string =  'Spectrogram, HI, All Inputs. '+user_inputs[0]+' Global Normalised \n HI Indicators at frequency-bin centres for 1420.367 MHz & 1420.422 MHz'
        else:  
            title_string = 'Spectrogram, HI, All Inputs. '+user_inputs[0]+' Normalised \n HI Indicators at frequency-bin centres for 1420.367 MHz & 1420.422 MHz'

    else:
        title_string = 'Spectrogram, HI, All Inputs. '+user_inputs[0]+'\n HI Indicators at frequency-bin centres for 1420.367 MHz & 1420.422 MHz'

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
     

    plt.setp(axarr, xticks=[0,4,8,12,16,20,24,28,32,36,40], xticklabels=[str(frequency[Ind-20]),'','','','',str(frequency[Ind]),'','','','',str(frequency[Ind+20])])
    plt.setp(axarr, xlabel='Frequency / MHz', ylabel='Time')



    a1p1_cw = np.load(DATA_PATH+'/temp/a1p1_bandpass.npy')
    a1p2_cw = np.load(DATA_PATH+'/temp/a1p2_bandpass.npy')
    a2p1_cw = np.load(DATA_PATH+'/temp/a2p1_bandpass.npy')
    a2p2_cw = np.load(DATA_PATH+'/temp/a2p2_bandpass.npy') 

    a1p1_cw, a1p2_cw,a2p1_cw,a2p2_cw = corrections(a1p1_cw,a1p2_cw,a2p1_cw,a2p2_cw)

    if parameters[8] == 'True':
        a1p1_cw = a1p1_cw / np.mean(a1p1_cw)
        a2p2_cw = a2p2_cw / np.mean(a2p2_cw)
        a1p2_cw = a1p2_cw / np.mean(a1p2_cw)
        a2p1_cw = a2p1_cw / np.mean(a2p1_cw)

    Pmax = []  #find the minimum and maximum values to use across all 4 plots
    Pmin = []
    bandmin, bandmax = get_minmax(a1p1_cw)
    Pmax.append(bandmax)
    Pmin.append(bandmin)
    bandmin, bandmax = get_minmax(a1p2_cw)
    Pmax.append(bandmax)
    Pmin.append(bandmin)
    bandmin, bandmax = get_minmax(a2p1_cw)
    Pmax.append(bandmax)
    Pmin.append(bandmin)
    bandmin, bandmax = get_minmax(a2p2_cw)
    Pmax.append(bandmax)
    Pmin.append(bandmin)
    Pmax = np.asarray(Pmax)
    Pmin = np.asarray(Pmin)
    Lmin = np.argmin(Pmin)
    Pmin = Pmin[Lmin]
    Lmax = np.argmax(Pmax)
    Pmax = Pmax[Lmax]

    axarr[0,1].imshow(a1p1_cw[:,(Ind-20):(Ind+21)], cmap=cstr,aspect='auto',vmin=Pmin,vmax=Pmax)#112:569
    d1 = axarr[0,1].imshow(a1p1_cw[:,(Ind-20):(Ind+21)], cmap=cstr, aspect='auto',vmin=Pmin,vmax=Pmax)
    axarr[0,1].set_title('P(L,$\pi$) W')
    f.colorbar(d1, ax=axarr[0, 1])
       
    axarr[0,0].imshow(a1p2_cw[:,(Ind-20):(Ind+21)], cmap=cstr, aspect='auto',vmin=Pmin,vmax=Pmax)
    d2 = axarr[0,0].imshow(a1p2_cw[:,(Ind-20):(Ind+21)], cmap=cstr,aspect='auto',vmin=Pmin,vmax=Pmax)
    axarr[0,0].set_title('P(L,0) E')
    f.colorbar(d2, ax=axarr[0, 0])
    
    axarr[1,0].imshow(a2p1_cw[:,(Ind-20):(Ind+21)], cmap=cstr,aspect='auto',vmin=Pmin,vmax=Pmax)
    d3 = axarr[1,0].imshow(a2p1_cw[:,(Ind-20):(Ind+21)], cmap=cstr,aspect='auto',vmin=Pmin,vmax=Pmax)
    axarr[1,0].set_title('P(R,$\pi$) E')
    f.colorbar(d3, ax=axarr[1, 0])

    axarr[1,1].imshow(a2p2_cw[:,(Ind-20):(Ind+21)], cmap=cstr,aspect='auto',vmin=Pmin,vmax=Pmax)
    d4 = axarr[1,1].imshow(a2p2_cw[:,(Ind-20):(Ind+21)], cmap=cstr,aspect='auto',vmin=Pmin,vmax=Pmax)
    axarr[1,1].set_title('P(R,0) W')
    f.colorbar(d4, ax=axarr[1, 1])

    axarr[0,0].axvline(x=20, c='k', linewidth=0.5)
    axarr[0,0].axvline(x=21, c='k', linewidth=0.5)
    axarr[1,0].axvline(x=20, c='k', linewidth=0.5)
    axarr[1,0].axvline(x=21, c='k', linewidth=0.5)
    axarr[1,1].axvline(x=20, c='k', linewidth=0.5)
    axarr[1,1].axvline(x=21, c='k', linewidth=0.5)
    axarr[0,1].axvline(x=20, c='k', linewidth=0.5)
    axarr[0,1].axvline(x=21, c='k', linewidth=0.5)
    plt.suptitle(title_string)
    plt.tight_layout()
 
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_4xHI_WF.png', bbox_inches="tight")
    else:
        plt.show()
    plt.close()
    
    if first_loop:
        print ('')
        save = input ('Do you want to save a printer friendly copy of this plot? (Y/N): ')
        if str(save) == 'N' or str(save) == 'n':
            pass
        elif str(save) == 'Y' or str(save) == 'y':
            save_it=True
            first_loop = False
            waterfallPERmin(duration_actual,save_it, first_loop)
 
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass



#------------------------------------------------------------------------------------

def HI_waterfallDD(save_it=False, first_loop=True):

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

    Ind = int(483) #483 for HI bin
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')

    a1p1B = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
    mins,first_tick, sample_to_datetime, frequency = yaxis(a1p1B, duration_actual)
    del a1p1B

    frequency = np.round(frequency, decimals=3)

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False


    if parameters[8]=='True' or parameters[1]=='True':
        if parameters[1]=='True' and parameters[8]=='True':
            title_string = 'Spectrogram, HI, All Inputs. '+user_inputs[0]+' Global Normalised, Normalised \n HI Indicators at frequency-bin centres for 1420.367 MHz & 1420.422 MHz'
        elif parameters[1]=='True':
            title_string =  'Spectrogram, HI, All Inputs. '+user_inputs[0]+' Global Normalised \n HI Indicators at frequency-bin centres for 1420.367 MHz & 1420.422 MHz'
        else:  
            title_string = 'Spectrogram, HI, All Inputs. '+user_inputs[0]+' Normalised \n HI Indicators at frequency-bin centres for 1420.367 MHz & 1420.422 MHz'

    else:
        title_string = 'Spectrogram, HI, Double Difference. '+user_inputs[0]+'\n HI Indicators at frequency-bin centres for 1420.367 MHz & 1420.422 MHz'

    if save_it:
        f, axarr = plt.subplots(figsize=(12,8), dpi=300)
    else:
        f, axarr = plt.subplots()
       
    plt.setp(axarr, yticks=first_tick, yticklabels=mins)
    
  #  if duration_actual > 12 and duration_actual < 48:

   #     plt.setp(axarr, yticks=np.arange(first_tick[0],sample_to_datetime,60))
    #elif duration_actual > 48:
     #   pass
    #else:
     #   plt.setp(axarr, yticks=np.arange(first_tick[0],sample_to_datetime,15))
    
    #for i in range (0,np.size(frequency)):
     

    plt.setp(axarr, xticks=[0,4,8,12,16,20,24,28,32,36,40], xticklabels=[str(frequency[Ind-20]),'','','','',str(frequency[Ind]),'','','','',str(frequency[Ind+20])])
    plt.setp(axarr, xlabel='Frequency / MHz', ylabel='Time')



    a1p1_cw = np.load(DATA_PATH+'/temp/a1p1_bandpass.npy')
    a1p2_cw = np.load(DATA_PATH+'/temp/a1p2_bandpass.npy')
    a2p1_cw = np.load(DATA_PATH+'/temp/a2p1_bandpass.npy')
    a2p2_cw = np.load(DATA_PATH+'/temp/a2p2_bandpass.npy') 

    a1p1_cw, a1p2_cw,a2p1_cw,a2p2_cw = corrections(a1p1_cw,a1p2_cw,a2p1_cw,a2p2_cw)

    d_a = a1p2_cw - a1p1_cw # l0 - lpi E-W
    d_b = a2p2_cw - a2p1_cw #r0 - rpi W-E
    d_c = a1p2_cw - a2p2_cw  # l0 - r0  E-W
    d_d = a1p1_cw - a2p1_cw  #   lpi - rpi  W-E

    DD = (d_a - d_b)/2 # WMAP (ocra would be c-d)

    if parameters[8] == 'True':
        a1p1_cw = a1p1_cw / np.mean(a1p1_cw)
        a2p2_cw = a2p2_cw / np.mean(a2p2_cw)
        a1p2_cw = a1p2_cw / np.mean(a1p2_cw)
        a2p1_cw = a2p1_cw / np.mean(a2p1_cw)
 
    DD = DD 

    axarr.imshow(DD[:,(Ind-20):(Ind+21)], cmap=cstr,aspect='auto')#112:569
    d2 = axarr.imshow(DD[:,(Ind-20):(Ind+21)], cmap=cstr,aspect='auto')#112:569
    axarr.axvline(x=20, c='k', linewidth=0.5)
    axarr.axvline(x=21, c='k', linewidth=0.5)
    plt.suptitle(title_string)
    plt.tight_layout()
    f.colorbar(d2, ax=axarr)
 
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_HI_WF_DD.png', bbox_inches="tight")
    else:
        plt.show()
    plt.close()
    
    if first_loop:
        print ('')
        save = input ('Do you want to save a printer friendly copy of this plot? (Y/N): ')
        if str(save) == 'N' or str(save) == 'n':
            pass
        elif str(save) == 'Y' or str(save) == 'y':
            save_it=True
            first_loop = False
            HI_waterfallPERminDD(duration_actual,save_it, first_loop)
 
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass





#------------------------------------------------------------------------------------


def ADHOCMenu(frequency):

    duration_actual = np.load(DATA_PATH+'/temp/duration_actual.npy')
    looper = True

    print ('')
    print ('   -------------------------------------')
    print ('   >>>          AD HOC MENU          <<<')
    print ('   -------------------------------------')
    print ('')
    print ('   1 - HI vs Time')
    print ('   2 - Bandpass Ani, Microphonics')
    print ('   3 - Save all Microphonic Events')
    print ('   4 - Export Data - Arshi')
    print ('   5 - Average Microphonics over period')
    print ('   6 - HI Waterfall')
    print ('   7 - HI Waterfall Double Difference')
    print ('')
    print ('')
    print ('   0 - Return to Quick-look menu')
    print ('')
    choice = input('Select menu option (number): \033[0;m')
    print('\033[1;32m')
    if choice.isdigit():        
        if int(choice) ==1:
            FourFreqVTime(frequency)

        if int(choice) ==2:
            bandpassANI(frequency)

        if int(choice) ==3:
            rawData(duration_actual)

        if int(choice) == 4:
            show_Arshi()

        if int(choice) == 5:
            gather_PP()

        if int(choice) == 6:
            HI_waterfall()

        if int(choice) == 7:
            HI_waterfallDD()



        elif int(choice) == 0:
            looper = False
            pass

         
    else:
        print('\033[1;31m Invalid selection by user. Please try again. \033[1;32m')
        ADHOCMenu(frequency)

    return looper


#########################################################################





duration_actual = np.load(DATA_PATH+'/temp/duration_actual.npy')

print ('\033[1;32m ')

looper = True
while looper:
    looper = ADHOCMenu(frequency)


os.system('chmod -R -f 0777 /mirror/scratch/pblack || true')

