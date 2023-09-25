#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:56:25 2022

@author: pblack
"""

DATA_PATH = '/mirror/scratch/pblack'

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
import matplotlib.dates as mdates
from matplotlib.dates import AutoDateFormatter, AutoDateLocator, DateFormatter
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


def RMSall(duration_actual,save_it=False, first_loop=True):


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

    p11,p12,p21,p22 = corrections(p11,p12,p21,p22)

    if parameters[8] == 'True':
        p11 = p11/np.mean(p11)
        p12 = p12/np.mean(p12)
        p21 = p21/np.mean(p21)
        p22 = p22/np.mean(p22)

    time2 = time_series()

    dev1 =[]
    dev2=[]
    dev3 =[]
    dev4 =[]

    time = []

    for i in range(1,(np.size(p11[:,0]))):
        this_line = p11[i,112:569]
        last_line = p11[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(p11[0,112:569])#####
        rooted = math.sqrt(summed)
        dev1.append(float(rooted))
      #  time.append(a1p1B[i,0])

    for i in range(1,(np.size(p12[:,0]))):
        this_line = p12[i,112:569]
        last_line = p12[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(p12[0,112:569])#####
        rooted = math.sqrt(summed)
        dev2.append(float(rooted))

    for i in range(1,(np.size(p21[:,0]))):
        this_line = p21[i,112:569]
        last_line = p21[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(p21[0,112:569])#####
        rooted = math.sqrt(summed)
        dev3.append(float(rooted))

    for i in range(1,(np.size(p22[:,0]))):
        this_line = p22[i,112:569]
        last_line = p22[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(p22[0,112:569])#####
        rooted = math.sqrt(summed)
        dev4.append(float(rooted))

 #   dev_avg = (np.asarray(dev1) + np.asarray(dev2) + np.asarray(dev3) + np.asarray(dev4))/4
  #  dev_avg_left = (np.asarray(dev1) + np.asarray(dev2))/2
   # dev_avg_right = (np.asarray(dev3) + np.asarray(dev4))/2

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

    ax.plot(time2[1:],dev1, color='b', linewidth=1, label='P(L,$\pi$) W')  
    ax.plot(time2[1:],dev2, color='deeppink', linewidth=1, label='P(L,0) E')
    ax.plot(time2[1:],dev4, color='c', linewidth=1, label='P(R,0) W') #, linestyle=(0,(5,1))
    ax.plot(time2[1:],dev3, color='firebrick', linewidth=1, label='P(R,$\pi$) E') # linestyle=(0,(5,1)),

   # plt.plot(dev_avg_left, color='orangered', linewidth=1, label='Average RMS')  
   # plt.plot(dev_avg_right, color='mediumblue', linewidth=1, label='Average RMS')  
 
    plt.title('Bandpass RMS (Time-bin to Time-bin), All Inputs')
    ax.set(ylabel='RMS')
    ax.set(xlabel='Time')
    plt.margins(x=0)

    y_limits = np.load(DATA_PATH+'/temp/ylims.npy')
    if y_limits:
        print('')
        man_lims = input('\033[1;32m Would you like to specify the y-axis limits on this plot? (Y/N): \033[0;m')
        if man_lims == str('Y') or man_lims == str('y'):
            print('')
            lower_limit = float(input('\033[1;32m Please enter an integer or decimal Lower Limit: \033[0;m'))
            upper_limit = float(input('\033[1;32m Please enter an integer or decimal Upper Limit: \033[0;m'))
        else:
            lower_limit = 0
            upper_limit = 0.15
    else:
        lower_limit = 0
        upper_limit = 0.15

    print('\033[1;32m')
    ax.set_ylim([lower_limit,upper_limit])


    ax.grid(c='darkgrey', which='major')
    ax.grid(c='gainsboro', which='minor')

    minor_locator = ticker.AutoMinorLocator()
    ax.yaxis.set_minor_locator(minor_locator)

    xtick_locator = AutoDateLocator()
    xtick_formatter = mdates.ConciseDateFormatter(xtick_locator)
    ax.xaxis.set_major_locator(xtick_locator)
    ax.xaxis.set_major_formatter(xtick_formatter)

    if duration_actual <= 0.5:
        xminor = mdates.MinuteLocator(byminute=range(60))
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 0.5 and duration_actual <= 4:
        xminor = mdates.MinuteLocator(byminute=[5,10,15,20,25,30,35,40,45,50,55])
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 4 and duration_actual <= 12:
        xminor = mdates.MinuteLocator(byminute=[15,30,45])
        ax.xaxis.set_minor_locator(xminor)

    if duration_actual > 12 and duration_actual <= 48:
        xminor = mdates.HourLocator()
        ax.xaxis.set_minor_locator(xminor)

    if duration_actual > 48 and duration_actual <= 150:
        xminor = mdates.HourLocator(byhour=[6,12,18])
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 150:
        xminor = mdates.HourLocator(byhour=[12])
        ax.xaxis.set_minor_locator(xminor)

    if save_it:
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_RMS_all4.png', bbox_inches="tight")
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
            RMSall(duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 


#------------------------------------------------------------------------------------------

def RMSsingle(duration_actual,save_it=False, first_loop=True, which_input=None):


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

    p11,p12,p21,p22 = corrections(p11,p12,p21,p22)

    if parameters[8] == 'True':
        p11 = p11/np.mean(p11)
        p12 = p12/np.mean(p12)
        p21 = p21/np.mean(p21)
        p22 = p22/np.mean(p22)

    time2 = time_series()

    dev1 =[]
    dev2=[]
    dev3 =[]
    dev4 =[]

    time = []

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

    for i in range(1,(np.size(p11[:,0]))):
        this_line = p11[i,112:569]
        last_line = p11[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(p11[0,112:569])#####
        rooted = math.sqrt(summed)
        dev1.append(float(rooted))
       # time.append(a1p1B[i,0])

    for i in range(1,(np.size(p12[:,0]))):
        this_line = p12[i,112:569]
        last_line = p12[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(p12[0,112:569])#####
        rooted = math.sqrt(summed)
        dev2.append(float(rooted))

    for i in range(1,(np.size(p21[:,0]))):
        this_line = p21[i,112:569]
        last_line = p21[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(p21[0,112:569])#####
        rooted = math.sqrt(summed)
        dev3.append(float(rooted))

    for i in range(1,(np.size(p22[:,0]))):
        this_line = p22[i,112:569]
        last_line = p22[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(p22[0,112:569])#####
        rooted = math.sqrt(summed)
        dev4.append(float(rooted))

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

    if which_input == 1:
        ax.plot(time2[1:],dev1, color='b', linewidth=1, label='P(L,$\pi$) W')  
        which_one = 'P(L,$\pi$)'
    elif which_input == 2:
        ax.plot(time2[1:],dev2, color='deeppink', linewidth=1, label='P(L,0) E')
        which_one = 'P(L,0)'
    elif which_input == 3:
        ax.plot(time2[1:],dev3, color='firebrick', linewidth=1, label='P(R,$\pi$) E') # linestyle=(0,(5,1)),
        which_one = 'P(R,$\pi$)'
    elif which_input == 4:
        ax.plot(time2[1:],dev4, color='c', linewidth=1, label='P(R,0) W') #, linestyle=(0,(5,1))
        which_one = 'P(L,0)'

    plt.title('Bandpass RMS (Time-bin to Time-bin), '+str(which_one))
    ax.set(ylabel='RMS')
    ax.set(xlabel='Time')
    plt.margins(x=0)
    y_limits = np.load(DATA_PATH+'/temp/ylims.npy')
    if y_limits:
        print('')
        man_lims = input('\033[1;32m Would you like to specify the y-axis limits on this plot? (Y/N): \033[0;m')
        if man_lims == str('Y') or man_lims == str('y'):
            print('')
            lower_limit = float(input('\033[1;32m Please enter an integer or decimal Lower Limit: \033[0;m'))
            upper_limit = float(input('\033[1;32m Please enter an integer or decimal Upper Limit: \033[0;m'))
        else:
            lower_limit = 0
            upper_limit = 0.15
    else:
        lower_limit = 0
        upper_limit = 0.15

    print('\033[1;32m')
    ax.set_ylim([lower_limit,upper_limit])
    ax.grid(c='darkgrey', which='major')
    ax.grid(c='gainsboro', which='minor')

    minor_locator = ticker.AutoMinorLocator()
    ax.yaxis.set_minor_locator(minor_locator)

    xtick_locator = AutoDateLocator()
    xtick_formatter = mdates.ConciseDateFormatter(xtick_locator)
    ax.xaxis.set_major_locator(xtick_locator)
    ax.xaxis.set_major_formatter(xtick_formatter)

    if duration_actual <= 0.5:
        xminor = mdates.MinuteLocator(byminute=range(60))
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 0.5 and duration_actual <= 4:
        xminor = mdates.MinuteLocator(byminute=[5,10,15,20,25,30,35,40,45,50,55])
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 4 and duration_actual <= 12:
        xminor = mdates.MinuteLocator(byminute=[15,30,45])
        ax.xaxis.set_minor_locator(xminor)

    if duration_actual > 12 and duration_actual <= 48:
        xminor = mdates.HourLocator()
        ax.xaxis.set_minor_locator(xminor)

    if duration_actual > 48 and duration_actual <= 150:
        xminor = mdates.HourLocator(byhour=[6,12,18])
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 150:
        xminor = mdates.HourLocator(byhour=[12])
        ax.xaxis.set_minor_locator(xminor)

    if save_it:
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_RMS_'+str(which_one)+'.png', bbox_inches="tight")
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
            RMSsingle(duration_actual, save_it, first_loop, which_input)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 

#------------------------------------------------------------------------------------------


def RMSlravg(duration_actual,save_it=False, first_loop=True):


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

    p11,p12,p21,p22 = corrections(p11,p12,p21,p22)

    if parameters[8] == 'True':
        p11 = p11/np.mean(p11)
        p12 = p12/np.mean(p12)
        p21 = p21/np.mean(p21)
        p22 = p22/np.mean(p22)

    time2 = time_series()

    empty = np.empty((1,718))
    empty[0,:] = np.nan
    if np.size(p11[:,0]) > np.size (p12[:,0]): #common for unbinned samples to mismatch by a single phase switch count. Correct to plot.
        p12 = np.insert(p12, (np.size(p12[:,0])-1) , empty, axis=0)
        p22 = np.insert(p22, (np.size(p22[:,0])-1), empty, axis=0)
    if np.size(p11[:,0]) < np.size (p12[:,0]):
        p11 = np.insert(p11, (np.size(p11[:,0])-1), empty, axis=0)  
        p21 = np.insert(p21, (np.size(p21[:,0])-1), empty, axis=0)  

    dev1 =[]
    dev2=[]
    dev3 =[]
    dev4 =[]

    time = []

    for i in range(1,(np.size(p11[:,0]))):
        this_line = p11[i,112:569]
        last_line = p11[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(p11[0,112:569])#####
        rooted = math.sqrt(summed)
        dev1.append(float(rooted))
       # time.append(a1p1B[i,0])

    for i in range(1,(np.size(p12[:,0]))):
        this_line = p12[i,112:569]
        last_line = p12[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(p12[0,112:569])#####
        rooted = math.sqrt(summed)
        dev2.append(float(rooted))

    for i in range(1,(np.size(p21[:,0]))):
        this_line = p21[i,112:569]
        last_line = p21[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(p21[0,112:569])#####
        rooted = math.sqrt(summed)
        dev3.append(float(rooted))

    for i in range(1,(np.size(p22[:,0]))):
        this_line = p22[i,112:569]
        last_line = p22[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(p22[0,112:569])#####
        rooted = math.sqrt(summed)
        dev4.append(float(rooted))

    dev_avg_left = (np.asarray(dev1) + np.asarray(dev2))/2
    dev_avg_right = (np.asarray(dev3) + np.asarray(dev4))/2

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()


    ax.plot(time2[1:],dev_avg_left, color='mediumblue', linewidth=1, label='Left RPG')  
    ax.plot(time2[1:],dev_avg_right, color='orangered', linewidth=1, label='Right RPG')  
 
    ax.set(ylabel='RMS')
    ax.set(xlabel='Time')
    plt.title('Bandpass RMS (Time-bin to Time-bin), Left & Right RPG Input Averages.')
    plt.margins(x=0)
    y_limits = np.load(DATA_PATH+'/temp/ylims.npy')
    if y_limits:
        print('')
        man_lims = input('\033[1;32m Would you like to specify the y-axis limits on this plot? (Y/N): \033[0;m')
        if man_lims == str('Y') or man_lims == str('y'):
            print('')
            lower_limit = float(input('\033[1;32m Please enter an integer or decimal Lower Limit: \033[0;m'))
            upper_limit = float(input('\033[1;32m Please enter an integer or decimal Upper Limit: \033[0;m'))
        else:
            lower_limit = 0
            upper_limit = 0.15
    else:
        lower_limit = 0
        upper_limit = 0.15

    print('\033[1;32m')
    ax.set_ylim([lower_limit,upper_limit])
    ax.grid(c='darkgrey', which='major')
    ax.grid(c='gainsboro', which='minor')



    minor_locator = ticker.AutoMinorLocator()
    ax.yaxis.set_minor_locator(minor_locator)

    xtick_locator = AutoDateLocator()
    xtick_formatter = mdates.ConciseDateFormatter(xtick_locator)
    ax.xaxis.set_major_locator(xtick_locator)
    ax.xaxis.set_major_formatter(xtick_formatter)

    if duration_actual <= 0.5:
        xminor = mdates.MinuteLocator(byminute=range(60))
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 0.5 and duration_actual <= 4:
        xminor = mdates.MinuteLocator(byminute=[5,10,15,20,25,30,35,40,45,50,55])
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 4 and duration_actual <= 12:
        xminor = mdates.MinuteLocator(byminute=[15,30,45])
        ax.xaxis.set_minor_locator(xminor)

    if duration_actual > 12 and duration_actual <= 48:
        xminor = mdates.HourLocator()
        ax.xaxis.set_minor_locator(xminor)

    if duration_actual > 48 and duration_actual <= 150:
        xminor = mdates.HourLocator(byhour=[6,12,18])
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 150:
        xminor = mdates.HourLocator(byhour=[12])
        ax.xaxis.set_minor_locator(xminor)

    if save_it:
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_RMS_lravg.png', bbox_inches="tight")
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
            RMSlravg(duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 

#------------------------------------------------------------------------------------------


def RMSleft(duration_actual,save_it=False, first_loop=True):


    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    date_time = Time(user_inputs[0]+'T'+user_inputs[3], format='isot', scale='utc', precision=4)

    p11 = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
    p12 = np.load (DATA_PATH+'/temp/a1p2_bandpass.npy')

    p11,p12,p11,p12 = corrections(p11,p12,p11,p12)

    if parameters[8] == 'True':
        p11 = p11/np.mean(p11)
        p12 = p12/np.mean(p12)


    time2 = time_series()

    dev1 =[]
    dev2=[]

    time = []

    for i in range(1,(np.size(p11[:,0]))):
        this_line = p11[i,112:569]
        last_line = p11[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(p11[0,112:569])#####
        rooted = math.sqrt(summed)
        dev1.append(float(rooted))
       # time.append(a1p1B[i,0])

    for i in range(1,(np.size(p12[:,0]))):
        this_line = p12[i,112:569]
        last_line = p12[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(p12[0,112:569])#####
        rooted = math.sqrt(summed)
        dev2.append(float(rooted))


    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

    ax.plot(time2[1:],dev1, color='b', linewidth=1, label='P(L,$\pi$) W')  
    ax.plot(time2[1:],dev2, color='deeppink', linewidth=1, label='P(L,0) E')
 
    plt.title('Bandpass RMS (Time-bin to Time-bin), Left RPG Inputs.')
    ax.set(ylabel='RMS')
    ax.set(xlabel='Time')
    plt.margins(x=0)
    y_limits = np.load(DATA_PATH+'/temp/ylims.npy')
    if y_limits:
        print('')
        man_lims = input('\033[1;32m Would you like to specify the y-axis limits on this plot? (Y/N): \033[0;m')
        if man_lims == str('Y') or man_lims == str('y'):
            print('')
            lower_limit = float(input('\033[1;32m Please enter an integer or decimal Lower Limit: \033[0;m'))
            upper_limit = float(input('\033[1;32m Please enter an integer or decimal Upper Limit: \033[0;m'))
        else:
            lower_limit = 0
            upper_limit = 0.15
    else:
        lower_limit = 0
        upper_limit = 0.15

    print('\033[1;32m')
    ax.set_ylim([lower_limit,upper_limit])
    ax.grid(c='darkgrey', which='major')
    ax.grid(c='gainsboro', which='minor')

    minor_locator = ticker.AutoMinorLocator()
    ax.yaxis.set_minor_locator(minor_locator)

    xtick_locator = AutoDateLocator()
    xtick_formatter = mdates.ConciseDateFormatter(xtick_locator)
    ax.xaxis.set_major_locator(xtick_locator)
    ax.xaxis.set_major_formatter(xtick_formatter)

    if duration_actual <= 0.5:
        xminor = mdates.MinuteLocator(byminute=range(60))
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 0.5 and duration_actual <= 4:
        xminor = mdates.MinuteLocator(byminute=[5,10,15,20,25,30,35,40,45,50,55])
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 4 and duration_actual <= 12:
        xminor = mdates.MinuteLocator(byminute=[15,30,45])
        ax.xaxis.set_minor_locator(xminor)

    if duration_actual > 12 and duration_actual <= 48:
        xminor = mdates.HourLocator()
        ax.xaxis.set_minor_locator(xminor)

    if duration_actual > 48 and duration_actual <= 150:
        xminor = mdates.HourLocator(byhour=[6,12,18])
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 150:
        xminor = mdates.HourLocator(byhour=[12])
        ax.xaxis.set_minor_locator(xminor)

    if save_it:
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_RMS_left.png', bbox_inches="tight")
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
            RMSleft(duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 


#------------------------------------------------------------------------------------------


def RMSright(duration_actual,save_it=False, first_loop=True):


    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    date_time = Time(user_inputs[0]+'T'+user_inputs[3], format='isot', scale='utc', precision=4)

    p21 = np.load (DATA_PATH+'/temp/a2p1_bandpass.npy')
    p22 = np.load (DATA_PATH+'/temp/a2p2_bandpass.npy')

    p21,p22,p21,p22 = corrections(p21,p22,p21,p22)

    if parameters[8] == 'True':
        p21 = p21/np.mean(p21)
        p22 = p22/np.mean(p22)

    time2 = time_series()

    dev3 =[]
    dev4 =[]

    time = []

    for i in range(1,(np.size(p21[:,0]))):
        this_line = p21[i,112:569]
        last_line = p21[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(p21[0,112:569])#####
        rooted = math.sqrt(summed)
        dev3.append(float(rooted))
       # time.append(a1p1B[i,0])

    for i in range(1,(np.size(p22[:,0]))):
        this_line = p22[i,112:569]
        last_line = p22[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(p22[0,112:569])#####
        rooted = math.sqrt(summed)
        dev4.append(float(rooted))

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

    ax.plot(time2[1:],dev4, color='c', linewidth=1, label='P(R,0) W') #, linestyle=(0,(5,1))
    ax.plot(time2[1:],dev3, color='firebrick', linewidth=1, label='P(R,$\pi$) E') # linestyle=(0,(5,1)),

    ax.set(ylabel='RMS')
    ax.set(xlabel='Time')
    plt.title('Bandpass RMS (Time-bin to Time-bin), Right RPG Inputs.')
    plt.margins(x=0)
    y_limits = np.load(DATA_PATH+'/temp/ylims.npy')
    if y_limits:
        print('')
        man_lims = input('\033[1;32m Would you like to specify the y-axis limits on this plot? (Y/N): \033[0;m')
        if man_lims == str('Y') or man_lims == str('y'):
            print('')
            lower_limit = float(input('\033[1;32m Please enter an integer or decimal Lower Limit: \033[0;m'))
            upper_limit = float(input('\033[1;32m Please enter an integer or decimal Upper Limit: \033[0;m'))
        else:
            lower_limit = 0
            upper_limit = 0.15
    else:
        lower_limit = 0
        upper_limit = 0.15

    print('\033[1;32m')
    ax.set_ylim([lower_limit,upper_limit])
    ax.grid(c='darkgrey', which='major')
    ax.grid(c='gainsboro', which='minor')


    minor_locator = ticker.AutoMinorLocator()
    ax.yaxis.set_minor_locator(minor_locator)

    xtick_locator = AutoDateLocator()
    xtick_formatter = mdates.ConciseDateFormatter(xtick_locator)
    ax.xaxis.set_major_locator(xtick_locator)
    ax.xaxis.set_major_formatter(xtick_formatter)

    if duration_actual <= 0.5:
        xminor = mdates.MinuteLocator(byminute=range(60))
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 0.5 and duration_actual <= 4:
        xminor = mdates.MinuteLocator(byminute=[5,10,15,20,25,30,35,40,45,50,55])
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 4 and duration_actual <= 12:
        xminor = mdates.MinuteLocator(byminute=[15,30,45])
        ax.xaxis.set_minor_locator(xminor)

    if duration_actual > 12 and duration_actual <= 48:
        xminor = mdates.HourLocator()
        ax.xaxis.set_minor_locator(xminor)

    if duration_actual > 48 and duration_actual <= 150:
        xminor = mdates.HourLocator(byhour=[6,12,18])
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 150:
        xminor = mdates.HourLocator(byhour=[12])
        ax.xaxis.set_minor_locator(xminor)

    if save_it:
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_RMS_right.png', bbox_inches="tight")
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
            RMSright(duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 


#------------------------------------------------------------------------------------------

def RMSpi(duration_actual,save_it=False, first_loop=True):


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

    p11,p12,p21,p22 = corrections(p11,p12,p21,p22)

    if parameters[8] == 'True':
        p11 = p11/np.mean(p11)
        p12 = p12/np.mean(p12)
        p21 = p21/np.mean(p21)
        p22 = p22/np.mean(p22)

    time2 = time_series()

    d_a = p12 - p11 # l0 - lpi 
    d_b = p22 - p21 #r0 - rpi 
    d_c = p12 - p22  # l0 - r0 
    d_d = p11 - p21  #   lpi - rpi 

    dev1 =[]
    dev2=[]
    dev3 =[]
    dev4 =[]

    time = []

    for i in range(1,(np.size(d_a[:,0]))):
        this_line = d_a[i,112:569]
        last_line = d_a[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(d_a[0,112:569])#####
        rooted = math.sqrt(summed)
        dev1.append(float(rooted))
       # time.append(a1p1B[i,0])

    for i in range(1,(np.size(d_b[:,0]))):
        this_line = d_b[i,112:569]
        last_line = d_b[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(d_b[0,112:569])#####
        rooted = math.sqrt(summed)
        dev2.append(float(rooted))

    for i in range(1,(np.size(d_c[:,0]))):
        this_line = d_c[i,112:569]
        last_line = d_c[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(d_c[0,112:569])#####
        rooted = math.sqrt(summed)
        dev3.append(float(rooted))

    for i in range(1,(np.size(d_d[:,0]))):
        this_line = d_d[i,112:569]
        last_line = d_d[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(d_d[0,112:569])#####
        rooted = math.sqrt(summed)
        dev4.append(float(rooted))

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

    ax.plot(time2[1:],dev1, c='b', label='$\delta$a = P(L,0)E - P(L,$\pi$)W') #WMAPa scheme
    ax.plot(time2[1:],dev2, c='orange', label='$\delta$b = P(R,0)W - P(R,$\pi$)E') #WMAPb scheme
    ax.plot(time2[1:],dev3, c='deeppink', label='$\delta$c = P(L,0)E - P(R,0)W') #OCRAa scheme
    ax.plot(time2[1:],dev4, c='firebrick', label='$\delta$d = P(L,$\pi$)W - P(R,$\pi$)E') #OCRAb scheme
 
    plt.title('Bandpass RMS (Time-bin to Time-bin), Single Differences.')
    ax.set(ylabel='RMS')
    ax.set(xlabel='Time')
    plt.margins(x=0)
    y_limits = np.load(DATA_PATH+'/temp/ylims.npy')
    if y_limits:
        print('')
        man_lims = input('\033[1;32m Would you like to specify the y-axis limits on this plot? (Y/N): \033[0;m')
        if man_lims == str('Y') or man_lims == str('y'):
            print('')
            lower_limit = float(input('\033[1;32m Please enter an integer or decimal Lower Limit: \033[0;m'))
            upper_limit = float(input('\033[1;32m Please enter an integer or decimal Upper Limit: \033[0;m'))
        else:
            lower_limit = 0
            upper_limit = 0.15
    else:
        lower_limit = 0
        upper_limit = 0.15

    print('\033[1;32m')
    ax.set_ylim([lower_limit,upper_limit])
    ax.grid(c='darkgrey', which='major')
    ax.grid(c='gainsboro', which='minor')

    minor_locator = ticker.AutoMinorLocator()
    ax.yaxis.set_minor_locator(minor_locator)

    xtick_locator = AutoDateLocator()
    xtick_formatter = mdates.ConciseDateFormatter(xtick_locator)
    ax.xaxis.set_major_locator(xtick_locator)
    ax.xaxis.set_major_formatter(xtick_formatter)

    if duration_actual <= 0.5:
        xminor = mdates.MinuteLocator(byminute=range(60))
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 0.5 and duration_actual <= 4:
        xminor = mdates.MinuteLocator(byminute=[5,10,15,20,25,30,35,40,45,50,55])
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 4 and duration_actual <= 12:
        xminor = mdates.MinuteLocator(byminute=[15,30,45])
        ax.xaxis.set_minor_locator(xminor)

    if duration_actual > 12 and duration_actual <= 48:
        xminor = mdates.HourLocator()
        ax.xaxis.set_minor_locator(xminor)

    if duration_actual > 48 and duration_actual <= 150:
        xminor = mdates.HourLocator(byhour=[6,12,18])
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 150:
        xminor = mdates.HourLocator(byhour=[12])
        ax.xaxis.set_minor_locator(xminor)

    if save_it:
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_RMS_single_diff.png', bbox_inches="tight")
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
            RMSpi(duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 

#------------------------------------------------------------------------------------------

def RMSzero(duration_actual,save_it=False, first_loop=True):


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

    p11,p12,p21,p22 = corrections(p11,p12,p21,p22)

    if parameters[8] == 'True':
        p11 = p11/np.mean(p11)
        p12 = p12/np.mean(p12)
        p21 = p21/np.mean(p21)
        p22 = p22/np.mean(p22)

    time2 = time_series()

    d_a = p12 - p11 # l0 - lpi E-W
    d_b = p22 - p21 #r0 - rpi W-E
    d_c = p12 - p22  # l0 - r0  E-W
    d_d = p11 - p21  #   lpi - rpi  W-E
    JPL_d1 = p11 - p12 # lpi - l0 W-E
    JPL_d2 = p22 - p21 # r0 - rpi W-E
    #NULL_IB2_a = p12 - p21 # l0 - rpi  E-E
    #NULL_IB2_b = p22 - p11 # r0 - lpi   W-W
    del p11,p12,p21,p22

    DD = (d_a - d_b)/2 # WMAP (ocra would be c-d)
    NULL_JPL = (JPL_d1 - JPL_d2)/2

    dev2=[]

    dev4 =[]

    time = []

    for i in range(1,(np.size(DD[:,0]))):
        this_line = DD[i,112:569]
        last_line = DD[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(DD[0,112:569])#####
        rooted = math.sqrt(summed)
        dev2.append(float(rooted))
       # time.append(a1p1B[i,0])

    for i in range(1,(np.size(NULL_JPL[:,0]))):
        this_line = NULL_JPL[i,112:569]
        last_line = NULL_JPL[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(NULL_JPL[0,112:569])#####
        rooted = math.sqrt(summed)
        dev4.append(float(rooted))

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

   # ax.plot(time2[1:],dev4, color='gray', linewidth=1, label='JPL Null')
    ax.plot(time2[1:],dev2, color='r', linewidth=1, label='Signal')

    ax.set(ylabel='RMS')
    ax.set(xlabel='Time')
    plt.title('Bandpass RMS (Time-bin to Time-bin), Double Difference.')
    plt.margins(x=0)
    y_limits = np.load(DATA_PATH+'/temp/ylims.npy')
    if y_limits:
        print('')
        man_lims = input('\033[1;32m Would you like to specify the y-axis limits on this plot? (Y/N): \033[0;m')
        if man_lims == str('Y') or man_lims == str('y'):
            print('')
            lower_limit = float(input('\033[1;32m Please enter an integer or decimal Lower Limit: \033[0;m'))
            upper_limit = float(input('\033[1;32m Please enter an integer or decimal Upper Limit: \033[0;m'))
        else:
            lower_limit = 0
            upper_limit = 0.15
    else:
        lower_limit = 0
        upper_limit = 0.15

    print('\033[1;32m')
    ax.set_ylim([lower_limit,upper_limit])
    ax.grid(c='darkgrey', which='major')
    ax.grid(c='gainsboro', which='minor')

    minor_locator = ticker.AutoMinorLocator()
    ax.yaxis.set_minor_locator(minor_locator)

    xtick_locator = AutoDateLocator()
    xtick_formatter = mdates.ConciseDateFormatter(xtick_locator)
    ax.xaxis.set_major_locator(xtick_locator)
    ax.xaxis.set_major_formatter(xtick_formatter)

    if duration_actual <= 0.5:
        xminor = mdates.MinuteLocator(byminute=range(60))
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 0.5 and duration_actual <= 4:
        xminor = mdates.MinuteLocator(byminute=[5,10,15,20,25,30,35,40,45,50,55])
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 4 and duration_actual <= 12:
        xminor = mdates.MinuteLocator(byminute=[15,30,45])
        ax.xaxis.set_minor_locator(xminor)

    if duration_actual > 12 and duration_actual <= 48:
        xminor = mdates.HourLocator()
        ax.xaxis.set_minor_locator(xminor)

    if duration_actual > 48 and duration_actual <= 150:
        xminor = mdates.HourLocator(byhour=[6,12,18])
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 150:
        xminor = mdates.HourLocator(byhour=[12])
        ax.xaxis.set_minor_locator(xminor)

    if save_it:
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_RMS_dbldiff.png', bbox_inches="tight")
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
            RMSzero(duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 

#------------------------------------------------------------------------------------------

def RMSavg(duration_actual,save_it=False, first_loop=True):


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

    p11,p12,p21,p22 = corrections(p11,p12,p21,p22)

    if parameters[8] == 'True':
        p11 = p11/np.mean(p11)
        p12 = p12/np.mean(p12)
        p21 = p21/np.mean(p21)
        p22 = p22/np.mean(p22)

    time2 = time_series()

    d_a = p12 - p11 # l0 - lpi E-W
    d_b = p22 - p21 #r0 - rpi W-E
    d_c = p12 - p22  # l0 - r0  E-W
    d_d = p11 - p21  #   lpi - rpi  W-E
    JPL_d1 = p11 - p12 # lpi - l0 W-E
    JPL_d2 = p22 - p21 # r0 - rpi W-E
    NULL_IB2_a = p12 - p21 # l0 - rpi  E-E
    NULL_IB2_b = p22 - p11 # r0 - lpi   W-W
    del p11,p12,p21,p22

    DD = (d_a - d_b)/2 # WMAP (ocra would be c-d)
    NULL_JPL = (JPL_d1 - JPL_d2)/2

    dev2=[]
    dev3=[]
    dev4 =[]

    time = []

    for i in range(1,(np.size(NULL_IB2_a[:,0]))):
        this_line = NULL_IB2_a[i,112:569]
        last_line = NULL_IB2_a[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(NULL_IB2_a[0,112:569])#####
        rooted = math.sqrt(summed)
        dev2.append(float(rooted))
       # time.append(a1p1B[i,0])

    for i in range(1,(np.size(NULL_JPL[:,0]))):
        this_line = NULL_JPL[i,112:569]
        last_line = NULL_JPL[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(NULL_JPL[0,112:569])#####
        rooted = math.sqrt(summed)
        dev4.append(float(rooted))

    for i in range(1,(np.size(NULL_IB2_b[:,0]))):
        this_line = NULL_IB2_b[i,112:569]
        last_line = NULL_IB2_b[i-1,112:569]
    #    last_line = p11[i-1,3:] #added the 3:<<<<<<<<
        ratio_between_lines = this_line / last_line  #if the system is stable this should be ~1 in each channel
        difference = last_line - this_line   #if the system is stable this should be ~0 in each channel
        diff_sq = difference **2
        summed = np.sum(diff_sq)
        summed = summed / np.size(NULL_IB2_b[0,112:569])#####
        rooted = math.sqrt(summed)
        dev3.append(float(rooted))

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

    ax.plot(time2[1:],dev4, color='gray', linewidth=1, label='JPL Null')
    ax.plot(time2[1:],dev2, color='r', linewidth=1, label='IB Null E-E')
    ax.plot(time2[1:],dev3, color='b', linewidth=1, label='IB Null W-W')

    ax.set(ylabel='RMS')
    ax.set(xlabel='Time')
    plt.title('Bandpass RMS (Time-bin to Time-bin), Nulls.')
    plt.margins(x=0)
    y_limits = np.load(DATA_PATH+'/temp/ylims.npy')
    if y_limits:
        print('')
        man_lims = input('\033[1;32m Would you like to specify the y-axis limits on this plot? (Y/N): \033[0;m')
        if man_lims == str('Y') or man_lims == str('y'):
            print('')
            lower_limit = float(input('\033[1;32m Please enter an integer or decimal Lower Limit: \033[0;m'))
            upper_limit = float(input('\033[1;32m Please enter an integer or decimal Upper Limit: \033[0;m'))
        else:
            lower_limit = 0
            upper_limit = 0.15
    else:
        lower_limit = 0
        upper_limit = 0.15

    print('\033[1;32m')
    ax.set_ylim([lower_limit,upper_limit])
    ax.grid(c='darkgrey', which='major')
    ax.grid(c='gainsboro', which='minor')

    minor_locator = ticker.AutoMinorLocator()
    ax.yaxis.set_minor_locator(minor_locator)

    xtick_locator = AutoDateLocator()
    xtick_formatter = mdates.ConciseDateFormatter(xtick_locator)
    ax.xaxis.set_major_locator(xtick_locator)
    ax.xaxis.set_major_formatter(xtick_formatter)

    if duration_actual <= 0.5:
        xminor = mdates.MinuteLocator(byminute=range(60))
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 0.5 and duration_actual <= 4:
        xminor = mdates.MinuteLocator(byminute=[5,10,15,20,25,30,35,40,45,50,55])
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 4 and duration_actual <= 12:
        xminor = mdates.MinuteLocator(byminute=[15,30,45])
        ax.xaxis.set_minor_locator(xminor)

    if duration_actual > 12 and duration_actual <= 48:
        xminor = mdates.HourLocator()
        ax.xaxis.set_minor_locator(xminor)

    if duration_actual > 48 and duration_actual <= 150:
        xminor = mdates.HourLocator(byhour=[6,12,18])
        ax.xaxis.set_minor_locator(xminor)
    if duration_actual > 150:
        xminor = mdates.HourLocator(byhour=[12])
        ax.xaxis.set_minor_locator(xminor)

    if save_it:
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_RMS_null.png', bbox_inches="tight")
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
            RMSavg(duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 




####################################################################

def RMSMenu(duration_actual):

    looper = True

    print ('')
    print ('   -------------------------------------')
    print ('   >>>       RMS PLOTS MENU        <<<')
    print ('   -------------------------------------')
    print ('')
    print ('   1 - All Inputs')
    print ('   2 - Single Input')
    print ('   3 - Average Left & Right')
    print ('   4 - Left Inputs')
    print ('   5 - Right Inputs')
    print ('   6 - Single Differences')
    print ('   7 - Double Difference')
    print ('   8 - Nulls')
    print ('')
    print ('')
    print ('   0 - Return to Quick-look menu')
    print ('')
    choice = input('Select menu option (number): \033[0;m')
    print('\033[1;32m')
    if choice.isdigit():
        if int(choice) ==1:
            RMSall(duration_actual)
        
        elif int(choice) ==2:
            RMSsingle(duration_actual)
        
        elif int(choice) ==3:
            RMSlravg(duration_actual)

        elif int(choice) ==4:
            RMSleft(duration_actual)

        elif int(choice)  == 5:
            RMSright(duration_actual)

        elif int(choice)  == 6:
            RMSpi(duration_actual)

        elif int(choice)  == 7:
            RMSzero(duration_actual)

        elif int(choice) ==8:
            RMSavg(duration_actual)


        elif int(choice) == 0:
            looper = False
            pass

    else:
        print('\033[1;31m Invalid selection by user. Please try again. \033[1;32m')
        RMSMenu(duration_actual)

    return looper

#---------------------------------------------------------------------

duration_actual = np.load(DATA_PATH+'/temp/duration_actual.npy')

looper = True
while looper:
    looper = RMSMenu(duration_actual)

print ('\033[1;32m ')


os.system('chmod -R -f 0777 /mirror/scratch/pblack || true')

