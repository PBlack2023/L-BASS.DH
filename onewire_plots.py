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

os.chdir('/scratch/nas_lbass/raw_data/')

# GOOD FREQ ARE 112-568  BUT remember channels start at 3 in a1p1 etc
# A1P1 - P(l,pi)
# A2P2 - P(r,0)
# A1P2 - P(l,0)
# A2P1 - P(r,pi)


#################################################################################

def time_series():

    quickload = np.load(DATA_PATH+'/temp/quickload.npy')

    if quickload:
        array_time = np.load(DATA_PATH+'/temp/temp_time_array.npy') 
        array_time = Time(array_time,format='mjd',scale='utc',precision=9)
        array_time.format = 'iso'
        time2 = array_time.tt.datetime

    else:
    
        one_wire = np.load (DATA_PATH+'/temp/one_wire.npy')
        obsheader = np.load(DATA_PATH+'/temp/obshdr.npy')
        MJD = Time(obsheader[0,8],format='mjd', scale='utc', precision=9)
        time_p1 = MJD + TimeDelta(one_wire[:,0].astype(float), format='sec') 
        time_p1.format = 'iso'
        time2 = time_p1.tt.datetime


    return time2



##########################################################################


def groupSensors(duration_actual):
    
    temp_data = np.load(DATA_PATH+'/temp/one_wire.npy', allow_pickle=True)
    print('')
    print ('\033[0;m Sorting One-wire data by sensor location')

    
# Temperatures of receiver box: sequence is Magic T input, 1st, 2nd, 3rd P162 LNAs, 2534 LNA, Phase switch, 4th P162 LNA. 

    rc1 = np.column_stack((temp_data[:,1], temp_data[:,2], temp_data[:,6], temp_data[:,7], temp_data[:,10], temp_data[:,11])) 
    rc2 = np.column_stack((temp_data[:,3], temp_data[:,4], temp_data[:,5], temp_data[:,8], temp_data[:,9], temp_data[:,12])) 

    rc1x = rc1
    rc2x = rc2

    a = np.where(np.isnan(rc1x[:,0]))
    rc1x = np.delete(rc1x,a,0)
    a = np.where(np.isnan(rc2x[:,0]))
    rc2x = np.delete(rc2x,a,0)

    if np.size(rc1x[:,0]) == 0: #there is no temperature data
        no_data = True
    else:
        no_data = False    

    RPG = np.load(DATA_PATH+'/temp/RPG_exist.npy')
    if RPG:
        RPG_temps = np.load(DATA_PATH+'/temp/RPG_temps.npy')


    if no_data:
        print('\033[0;m There is no One-Wire temperature data available during this period. \033[1;32m')
        print('')
    else:
        if int(np.size(rc1x[:,0])) == int(duration_actual)*60:
            print('')
            print ('\033[1;32m During this',duration_actual,'hour period:')
        else:
            print('\033[0;m One-Wire temperature data available for',np.size(rc1x[:,0]),'of',np.size(temp_data[:,0]),'data points. \033[1;32m')
            print('')
            print('\033[1;32m During the available',np.round(np.size(rc1x[:,0])/60,decimals=2),'hour period:')
            duration_actual = np.round(np.size(rc1x[:,0])/60,decimals=2)

    if RPG:
        pass
    else:
        print('\033[0;m There is no RPG temperature data available during this period. \033[1;32m')
        print('')

    if no_data:
        rc1 = None
        rc2 = None
        mT1 = None
        mT2 = None
        hc1 = None
        hc2 = None

    else:

        avgT = (np.mean(rc1x) + np.mean(rc2x)) / 2
        avgT = np.round(avgT, decimals=2)
       
        if np.amax(rc1x) > np.amax(rc2x):
            print ('   Inner Box | Max',np.round(np.amax(rc1x), decimals=2), end='')
        else:
            print ('   Inner Box | Max',np.round(np.amax(rc2x), decimals=2), end='')
        if np.amin(rc1x) < np.amin(rc2x):
            print ('°C | Min',np.round(np.amin(rc1x), decimals=2), end='')
        else:
            print ('°C | Min',np.round(np.amin(rc2x), decimals=2), end='')
        print('°C | Avg', str(avgT)+'°C')

# temperatures of the MAGIC T and first cable
 
        mT1 = np.column_stack((temp_data[:,14], temp_data[:,27], temp_data[:,28]))
        mT2 = np.column_stack((temp_data[:,13], temp_data[:,15], temp_data[:,16]))

        mT1x = mT1
        mT2x = mT2
        a = np.where(np.isnan(mT1x[:,0]))
        mT1x = np.delete(mT1x,a,0)
        a = np.where(np.isnan(mT2x[:,0]))
        mT2x = np.delete(mT2x,a,0)

        avgT = (np.mean(mT1x) + np.mean(mT2x)) / 2
        avgT = np.round(avgT, decimals=2)
       
        if np.amax(mT1x) > np.amax(mT2x):
            print ('   Outer Box | Max',np.round(np.amax(mT1x), decimals=2), end='')
        else:
            print ('   Outer Box | Max',np.round(np.amax(mT2x), decimals=2), end='')
        if np.amin(mT1x) < np.amin(mT2x):
            print ('°C | Min',np.round(np.amin(mT1x), decimals=2), end='')
        else:
            print ('°C | Min',np.round(np.amin(mT2x), decimals=2), end='')
        print('°C | Avg', str(avgT)+'°C')

# Temperatures of horns and cables:

        hc1 = np.column_stack((temp_data[:,38], temp_data[:,37], temp_data[:,36], temp_data[:,35], temp_data[:,34], temp_data[:,33], temp_data[:,32], temp_data[:,31], temp_data[:,30], temp_data[:,29])) 
        hc2 = np.column_stack((temp_data[:,26], temp_data[:,25], temp_data[:,24], temp_data[:,23], temp_data[:,22], temp_data[:,21], temp_data[:,20], temp_data[:,19], temp_data[:,18], temp_data[:,17])) 
 
        hc1x = hc1
        hc2x = hc2

        a = np.where(np.isnan(hc1x[:,0]))
        hc1x = np.delete(hc1x,a,0)
        a = np.where(np.isnan(hc2x[:,0]))
        hc2x = np.delete(hc2x,a,0)

        avgT = (np.mean(hc1x) + np.mean(hc2x)) / 2
        avgT = np.round(avgT, decimals=2)
    
        if np.amax(hc1x) > np.amax(hc2x):
            print ('   Externals | Max',np.round(np.amax(hc1x), decimals=2), end='')
        else:
            print ('   Externals | Max',np.round(np.amax(hc2x), decimals=2), end='')
        if np.amin(hc1x) < np.amin(hc2x):
            print ('°C | Min',np.round(np.amin(hc1x), decimals=2), end='')
        else:
            print ('°C | Min',np.round(np.amin(hc2x), decimals=2), end='')
        print('°C | Avg', str(avgT)+'°C')
    
    RPG = np.load(DATA_PATH+'/temp/RPG_exist.npy')
    if RPG:
        RPG_temps = np.load(DATA_PATH+'/temp/RPG_temps.npy')
        RPGavg = np.mean(RPG_temps[:,1])
        RPGavg = np.round(RPGavg, decimals=2)
        print ('   RPG ADC   | Max',np.round(np.amax(RPG_temps[:,1]), decimals=2), end='')
        print ('°C | Min',np.round(np.amin(RPG_temps[:,1]), decimals=2), end='')
        print('°C | Avg', str(RPGavg)+'°C')

    time.sleep(0.5)
    gc.collect()

    return rc1, rc2, hc1, hc2, mT1, mT2, temp_data[:,0], duration_actual, no_data

#---------------------------------------------------------------------

def rcplot(rc1, rc2, duration_actual, save_it=False, first_loop=True):
    
    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    x = time_series()

    avgrc1 = rc1.mean(axis=1)
    avgrc2 = rc2.mean(axis=1)
    
    avgrc = (avgrc1+avgrc2)/2
    
    np.save(DATA_PATH+'/temp/4norm/rc-mean.npy', avgrc)
    
    title_string = 'Temperature Time Series, Receiver Board / Inner Box \n'+ 'Solid Lines East, Dashed Lines West.'

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

    ax.plot(x,rc1[:,0], linestyle='-', c='r', linewidth=1, label='LNA1')
    ax.plot(x,rc1[:,1], linestyle='-', c='g', linewidth=1, label='LNA2')
    ax.plot(x,rc1[:,2], linestyle='-', c='blueviolet', linewidth=1, label='LNA3')
    ax.plot(x,rc1[:,3], linestyle='-', c='b', linewidth=1,label='2534 LNA')
    ax.plot(x,rc1[:,4], linestyle='-', c='m', linewidth=1, label='Phase Switch')
    ax.plot(x,rc1[:,5], linestyle='-', c='orange', linewidth=1, label='LNA4')
    
    ax.plot(x,rc2[:,0], linestyle='--', c='r', linewidth=1)
    ax.plot(x,rc2[:,1], linestyle='--', c='g', linewidth=1)
    ax.plot(x,rc2[:,2], linestyle='--', c='blueviolet', linewidth=1)
    ax.plot(x,rc2[:,3], linestyle='--', c='b', linewidth=1)
    ax.plot(x,rc2[:,4], linestyle='--', c='m', linewidth=1)
    ax.plot(x,rc2[:,5], linestyle='--', c='orange', linewidth=1)
 #   ax.axhline(y=44,c='darkgrey',linewidth=3,linestyle='--')
  #  ax.axhline(y=45,c='darkgrey',linewidth=3,linestyle='--')

    ax.plot(x, avgrc, linestyle='-', c='k', linewidth=2, label='Mean Temperature')

  #  ax.set_ylim(39,49)
    ax.set(xlabel='Time')
    ax.set(ylabel='Temperature / °C')
    ax.grid(c='darkgrey', which='major')
    ax.grid(c='gainsboro', which='minor')

    ax.margins(x=0)

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
        ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        ax.legend(bbox_to_anchor=(1.1,1),loc="upper right", fontsize=9)
    ax.set_title(title_string)

    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_InnerBoxTemps.png', bbox_inches="tight")
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
            rcplot(rc1, rc2, duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 
     
#-------------------------------------------------------------------------

def mTplot(mT1, mT2, duration_actual, save_it=False, first_loop=True):

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False


    x = time_series()
   
    avgmT1 = mT1.mean(axis=1)
    avgmT2 = mT2.mean(axis=1)
    
    avgmT = (avgmT1+avgmT2)/2
    
    np.save(DATA_PATH+'/temp/4norm/mT-mean.npy', avgmT)
    
    title_string = "Temperatures Time Series, Magic T's / Outer Box. \n"+'Solid Lines East, Dashed Lines West.'

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()
    
    ax.plot(x,mT1[:,0], linestyle='-', c='r', linewidth=1, label='T Port4')
    ax.plot(x,mT1[:,1], linestyle='-', c='g', linewidth=1, label ='T Port2')
    ax.plot(x,mT1[:,2], linestyle='-', c='b', linewidth=1, label='Cable 1A S1')
    ax.plot(x,mT2[:,0], linestyle='--', c='r', linewidth=1, label='T Port3')
    ax.plot(x,mT2[:,1], linestyle='--', c='g', linewidth=1, label='T Port1')
    ax.plot(x,mT2[:,2], linestyle='--', c='b', linewidth=1, label='Cable 2A S1')
  #  ax.axhline(y=41.5,c='darkgrey',linewidth=3,linestyle='--')
   # ax.axhline(y=40.5,c='darkgrey',linewidth=3,linestyle='--')
    ax.plot(x,avgmT, linestyle='-', c='k', linewidth=2, label='Mean Temperature')

    
    ax.set(xlabel='Time')
    ax.set(ylabel='Temperature / °C')
   # ax.set_ylim(35,45)
    ax.grid(c='darkgrey', which='major')
    ax.grid(c='gainsboro', which='minor')

    ax.margins(x=0)

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
        ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        ax.legend(bbox_to_anchor=(1.1,1),loc="upper right", fontsize=9)
    ax.set_title(title_string)

    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_OuterBoxTemps.png', bbox_inches="tight")
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
            mTplot(mT1, mT2,  duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 

#-------------------------------------------------------------------------
    
def hcplot(hc1, hc2, duration_actual, save_it=False, first_loop=True):
    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    x = time_series()
    
    avgeast = hc1[:,4:].mean(axis=1)
    avgwest = hc2[:,4:].mean(axis=1)
    
    hcMean = (avgeast + avgwest) / 2
    
    np.save(DATA_PATH+'/temp/4norm/hc-mean.npy', hcMean)

    title_string = 'Temperatures Time Series, External Cables \n'+'Solid Lines East, Dashed Lines West'

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()
    
    ax.plot(x,hc1[:,4], linestyle='-', c='m', linewidth=1, label='Cable S7')
    ax.plot(x,hc1[:,5], linestyle='-', c='y', linewidth=1, label='Cable S6')
    ax.plot(x,hc1[:,6], linestyle='-', c='orange', linewidth=1, label='Cable S5')
    ax.plot(x,hc1[:,7], linestyle='-', c='lime', linewidth=1, label= 'Cable S4')
    ax.plot(x,hc1[:,8], linestyle='-', c='teal', linewidth=1, label='Cable S3')
    ax.plot(x,hc1[:,9], linestyle='-', c='c', linewidth=1, label='Cable S2')
    
    ax.plot(x,hc2[:,4], linestyle='--', c='m', linewidth=1)
    ax.plot(x,hc2[:,5], linestyle='--', c='y', linewidth=1)
    ax.plot(x,hc2[:,6], linestyle='--', c='orange', linewidth=1)
    ax.plot(x,hc2[:,7], linestyle='--', c='lime', linewidth=1)
    ax.plot(x,hc2[:,8], linestyle='--', c='teal', linewidth=1)
    ax.plot(x,hc2[:,9], linestyle='--', c='c', linewidth=1)

    ax.plot(x,hcMean, linestyle='-', c='k', linewidth=2, label='Mean Temperature')
    
    ax.set(xlabel='Time')
    ax.set(ylabel='Temperature / °C')
    ax.grid(c='darkgrey', which='major')
    ax.grid(c='gainsboro', which='minor')

    ax.margins(x=0)

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
        ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        ax.legend(bbox_to_anchor=(1.1,1),loc="upper right", fontsize=9)
    ax.set_title(title_string)

    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_cableTemps.png', bbox_inches="tight")
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
            hcplot(hc1, hc2, duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 



#-------------------------------------------------------------------------
    
def hornplot(hc1, hc2,  duration_actual, save_it=False, first_loop=True):
    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    x = time_series()
    
    avgeast = hc1[:,:3].mean(axis=1)
    avgwest = hc2[:,:3].mean(axis=1)
    
    hcMean = (avgeast + avgwest) / 2
    
    np.save(DATA_PATH+'/temp/4norm/hc-mean.npy', hcMean)

    title_string = 'Temperatures Time Series, Horns \n'+'Solid Lines East, Dashed Lines West.'

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()
    
    ax.plot(x,hc1[:,0], linestyle='-', c='r', linewidth=1, label='Horn')
    ax.plot(x,hc1[:,1], linestyle='-', c='g', linewidth=1, label='Throat')
    ax.plot(x,hc1[:,2], linestyle='-', c='blueviolet', linewidth=1, label='Polarizer')
    ax.plot(x,hc1[:,3], linestyle='-', c='b', linewidth=1, label='Pol Connect')
    
    ax.plot(x,hc2[:,0], linestyle='--', c='r', linewidth=1)
    ax.plot(x,hc2[:,1], linestyle='--', c='g', linewidth=1)
    ax.plot(x,hc2[:,2], linestyle='--', c='blueviolet', linewidth=1)
    ax.plot(x,hc2[:,3], linestyle='--', c='b', linewidth=1)

    ax.plot(x,hcMean, linestyle='-', c='k', linewidth=2, label='Mean Temperature')
    
    ax.set(xlabel='Time')
    ax.set(ylabel='Temperature / °C')
    ax.grid(c='darkgrey', which='major')
    ax.grid(c='gainsboro', which='minor')

    ax.margins(x=0)

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
        ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        ax.legend(bbox_to_anchor=(1.1,1),loc="upper right", fontsize=9)
    ax.set_title(title_string)

    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_HornTemps.png', bbox_inches="tight")
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
            hornplot(hc1, hc2, duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 
        
#---------------------------------------------------------------------

def threeINone(rc1, rc2, mT1, mT2, hc1, hc2, user_inputs, duration_actual):
    
    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False


    x = time_series()
  
    fig = plt.figure(figsize=(12, 8), dpi=300)
    
    gs = fig.add_gridspec(3, hspace=0) #fig.suptitle('Sharing both axes')
    axs = gs.subplots(sharex=True, sharey=False)
    
    
    #plt.style.use('default')
    #radiometer
    # Hide x labels and tick labels for all but bottom plot.
        
        #receiver board
        
    axs[0].plot(x,rc1[:,0], linestyle='-', c='r', linewidth=0.5, label='LNA1')
    axs[0].plot(x,rc1[:,1], linestyle='-', c='g', linewidth=0.5, label='LNA2')
    axs[0].plot(x,rc1[:,2], linestyle='-', c='blueviolet', linewidth=0.5, label='LNA3')
    axs[0].plot(x,rc1[:,3], linestyle='-', c='b', linewidth=0.5, label='2534 LNA')
    axs[0].plot(x,rc1[:,4], linestyle='-', c='m', linewidth=0.5, label='Ph/S')
    axs[0].plot(x,rc1[:,5], linestyle='-', c='orange', linewidth=0.5, label='LNA4)')
    
    axs[0].plot(x,rc2[:,0], linestyle='--', dashes=(5, 10), c='r', linewidth=0.5)
    axs[0].plot(x,rc2[:,1], linestyle='--', dashes=(5, 10), c='g', linewidth=0.5)
    axs[0].plot(x,rc2[:,2], linestyle='--', dashes=(5, 10), c='blueviolet', linewidth=0.5)
    axs[0].plot(x,rc2[:,3], linestyle='--', dashes=(5, 10), c='b', linewidth=0.5)
    axs[0].plot(x,rc2[:,4], linestyle='--', dashes=(5, 10), c='m', linewidth=0.5)
    axs[0].plot(x,rc2[:,5], linestyle='--', dashes=(5, 10), c='orange', linewidth=0.5)
    axs[0].legend(bbox_to_anchor=(1.04,1), loc="upper left")
    axs[0].grid(c='gainsboro', which='major')
    axs[0].grid(c='whitesmoke', which='minor')
 #   axs[0].set_ylim(bottom=40, top=50)
    
    # Magic T
    
    axs[1].plot(x,mT1[:,0], linestyle='-', c='r', linewidth=0.5, label='T Port4')
    axs[1].plot(x,mT1[:,1], linestyle='-', c='g', linewidth=0.5, label ='T Port2')
    axs[1].plot(x,mT1[:,2], linestyle='-', c='b', linewidth=0.5, label ='Cable 1A Sensor1')
    axs[1].plot(x,mT2[:,0], linestyle='--', dashes=(5, 10), c='r', linewidth=0.5, label='T Port3')
    axs[1].plot(x,mT2[:,1], linestyle='--', dashes=(5, 10), c='g', linewidth=0.5, label='T Port1')
    axs[1].plot(x,mT2[:,2], linestyle='--', dashes=(5, 10), c='b', linewidth=0.5, label='Cable 2A Sensor1')
    axs[1].legend(bbox_to_anchor=(1.04,0.5), loc="center left")
    axs[1].grid(c='gainsboro', which='major')
    axs[1].grid(c='whitesmoke', which='minor')
#    axs[1].set_ylim(bottom=35, top=45)
    axs[1].set(ylabel='Temperature / °C')
    
    # Horns
    
    axs[2].plot(x,hc1[:,0], linestyle='-', c='r', linewidth=0.5, label='Horn')
    axs[2].plot(x,hc1[:,1], linestyle='-', c='g', linewidth=0.5, label='Throat')
    axs[2].plot(x,hc1[:,2], linestyle='-', c='blueviolet', linewidth=0.5, label='Polarizer')
    axs[2].plot(x,hc1[:,3], linestyle='-', c='b', linewidth=0.5, label = 'Pol Connect')
    axs[2].plot(x,hc1[:,4], linestyle='-', c='m', linewidth=0.5, label='Cable S7')
    axs[2].plot(x,hc1[:,5], linestyle='-', c='y', linewidth=0.5, label='Cable S6')
    axs[2].plot(x,hc1[:,6], linestyle='-', c='orange', linewidth=0.5, label='Cable S5')
    axs[2].plot(x,hc1[:,7], linestyle='-', c='lime', linewidth=0.5, label='Cable S4')
    axs[2].plot(x,hc1[:,8], linestyle='-', c='teal', linewidth=0.5, label='Cable S3')
    axs[2].plot(x,hc1[:,9], linestyle='-', c='c', linewidth=0.5, label='Cable S2')
    
    axs[2].plot(x,hc2[:,0], linestyle='--', dashes=(5, 10), c='r', linewidth=0.5)
    axs[2].plot(x,hc2[:,1], linestyle='--', dashes=(5, 10), c='g', linewidth=0.5)
    axs[2].plot(x,hc2[:,2], linestyle='--', dashes=(5, 10), c='blueviolet', linewidth=0.5)
    axs[2].plot(x,hc2[:,3], linestyle='--', dashes=(5, 10), c='b', linewidth=0.5)
    axs[2].plot(x,hc2[:,4], linestyle='--', dashes=(5, 10), c='m', linewidth=0.5)
    axs[2].plot(x,hc2[:,5], linestyle='--', dashes=(5, 10), c='y', linewidth=0.5)
    axs[2].plot(x,hc2[:,6], linestyle='--', dashes=(5, 10), c='orange', linewidth=0.5)
    axs[2].plot(x,hc2[:,7], linestyle='--', dashes=(5, 10), c='lime', linewidth=0.5)
    axs[2].plot(x,hc2[:,8], linestyle='--', dashes=(5, 10), c='teal', linewidth=0.5)
    axs[2].plot(x,hc2[:,9], linestyle='--', dashes=(5, 10), c='c', linewidth=0.5)
    axs[2].legend(bbox_to_anchor=(1.04,0), loc="lower left")
    axs[2].grid(c='gainsboro', which='major')
    axs[2].grid(c='whitesmoke', which='minor')
    #axs[2].set_ylim(bottom=-5, top=15)
    plt.suptitle('Temperatures Time Series, All One-Wire Data \n'+'Solid Lines East, Dashed Lines West.')

    xtick_locator = AutoDateLocator()
    xtick_formatter = mdates.ConciseDateFormatter(xtick_locator)
  #  ax.xaxis.set_major_locator(xtick_locator)
   # ax.xaxis.set_major_formatter(xtick_formatter)

    for ax in axs:
        ax.label_outer()
        ax.margins(x=0)
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
    
    for ax in axs.flat:
        ax.set(xlabel='Time')




    plt.tight_layout()
    plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_1Wtemps.png', bbox_inches="tight")
    plt.close()

    print('')
    print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
    
#------------------------------------------------------------------------
    
def partDeux(rc1, rc2, mT1, mT2, hc1, hc2, user_inputs, duration_actual):
    


    x = time_series()

 
    fig = plt.figure()
    gs = fig.add_gridspec(3, hspace=0) #fig.suptitle('Sharing both axes')
    axs = gs.subplots(sharex=True, sharey=False)
    
    
    #plt.style.use('default')
    #radiometer
    # Hide x labels and tick labels for all but bottom plot.
        
        #receiver board
        
    axs[0].plot(x,rc1[:,0], linestyle='-', c='r', linewidth=0.5, label='LNA1')
    axs[0].plot(x,rc1[:,1], linestyle='-', c='g', linewidth=0.5, label='LNA2')
    axs[0].plot(x,rc1[:,2], linestyle='-', c='blueviolet', linewidth=0.5, label='LNA3')
    axs[0].plot(x,rc1[:,3], linestyle='-', c='b', linewidth=0.5, label='2534 LNA')
    axs[0].plot(x,rc1[:,4], linestyle='-', c='m', linewidth=0.5, label='Ph/S')
    axs[0].plot(x,rc1[:,5], linestyle='-', c='orange', linewidth=0.5, label='LNA4)')
    
    axs[0].plot(x,rc2[:,0], linestyle='--', dashes=(5, 10), c='r', linewidth=0.5)
    axs[0].plot(x,rc2[:,1], linestyle='--', dashes=(5, 10), c='g', linewidth=0.5)
    axs[0].plot(x,rc2[:,2], linestyle='--', dashes=(5, 10), c='blueviolet', linewidth=0.5)
    axs[0].plot(x,rc2[:,3], linestyle='--', dashes=(5, 10), c='b', linewidth=0.5)
    axs[0].plot(x,rc2[:,4], linestyle='--', dashes=(5, 10), c='m', linewidth=0.5)
    axs[0].plot(x,rc2[:,5], linestyle='--', dashes=(5, 10), c='orange', linewidth=0.5)
    axs[0].legend(bbox_to_anchor=(1.04,1), loc="upper left")
    axs[0].grid(c='gainsboro', which='major')
    axs[0].grid(c='whitesmoke', which='minor')
 #   axs[0].set_ylim(bottom=40, top=50)
    
    # Magic T
    
    axs[1].plot(x,mT1[:,0], linestyle='-', c='r', linewidth=0.5, label='T Port4')
    axs[1].plot(x,mT1[:,1], linestyle='-', c='g', linewidth=0.5, label ='T Port2')
    axs[1].plot(x,mT1[:,2], linestyle='-', c='b', linewidth=0.5, label ='Cable 1A Sensor1')
    axs[1].plot(x,mT2[:,0], linestyle='--', dashes=(5, 10), c='r', linewidth=0.5, label='T Port3')
    axs[1].plot(x,mT2[:,1], linestyle='--', dashes=(5, 10), c='g', linewidth=0.5, label='T Port1')
    axs[1].plot(x,mT2[:,2], linestyle='--', dashes=(5, 10), c='b', linewidth=0.5, label='Cable 2A Sensor1')
    axs[1].legend(bbox_to_anchor=(1.04,0.5), loc="center left")
    axs[1].grid(c='gainsboro', which='major')
    axs[1].grid(c='whitesmoke', which='minor')
#    axs[1].set_ylim(bottom=35, top=45)
    axs[1].set(ylabel='Temperature / °C')
    
    # Horns
    
    axs[2].plot(x,hc1[:,0], linestyle='-', c='r', linewidth=0.5, label='Horn')
    axs[2].plot(x,hc1[:,1], linestyle='-', c='g', linewidth=0.5, label='Throat')
    axs[2].plot(x,hc1[:,2], linestyle='-', c='blueviolet', linewidth=0.5, label='Polarizer')
    axs[2].plot(x,hc1[:,3], linestyle='-', c='b', linewidth=0.5, label = 'Pol Connect')
    axs[2].plot(x,hc1[:,4], linestyle='-', c='m', linewidth=0.5, label='Cable S7')
    axs[2].plot(x,hc1[:,5], linestyle='-', c='y', linewidth=0.5, label='Cable S6')
    axs[2].plot(x,hc1[:,6], linestyle='-', c='orange', linewidth=0.5, label='Cable S5')
    axs[2].plot(x,hc1[:,7], linestyle='-', c='lime', linewidth=0.5, label='Cable S4')
    axs[2].plot(x,hc1[:,8], linestyle='-', c='teal', linewidth=0.5, label='Cable S3')
    axs[2].plot(x,hc1[:,9], linestyle='-', c='c', linewidth=0.5, label='Cable S2')
    
    axs[2].plot(x,hc2[:,0], linestyle='--', dashes=(5, 10), c='r', linewidth=0.5)
    axs[2].plot(x,hc2[:,1], linestyle='--', dashes=(5, 10), c='g', linewidth=0.5)
    axs[2].plot(x,hc2[:,2], linestyle='--', dashes=(5, 10), c='blueviolet', linewidth=0.5)
    axs[2].plot(x,hc2[:,3], linestyle='--', dashes=(5, 10), c='b', linewidth=0.5)
    axs[2].plot(x,hc2[:,4], linestyle='--', dashes=(5, 10), c='m', linewidth=0.5)
    axs[2].plot(x,hc2[:,5], linestyle='--', dashes=(5, 10), c='y', linewidth=0.5)
    axs[2].plot(x,hc2[:,6], linestyle='--', dashes=(5, 10), c='orange', linewidth=0.5)
    axs[2].plot(x,hc2[:,7], linestyle='--', dashes=(5, 10), c='lime', linewidth=0.5)
    axs[2].plot(x,hc2[:,8], linestyle='--', dashes=(5, 10), c='teal', linewidth=0.5)
    axs[2].plot(x,hc2[:,9], linestyle='--', dashes=(5, 10), c='c', linewidth=0.5)
    axs[2].legend(bbox_to_anchor=(1.04,0), loc="lower left")
    axs[2].grid(c='gainsboro', which='major')
    axs[2].grid(c='whitesmoke', which='minor')
    #axs[2].set_ylim(bottom=-5, top=15)
    plt.suptitle('Temperatures Time Series, All One-Wire Data \n'+'Solid Lines East, Dashed Lines West.')

    xtick_locator = AutoDateLocator()
    xtick_formatter = mdates.ConciseDateFormatter(xtick_locator)
  #  ax.xaxis.set_major_locator(xtick_locator)
   # ax.xaxis.set_major_formatter(xtick_formatter)

    for ax in axs:
        ax.label_outer()
        ax.margins(x=0)
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
        ax.grid(c='whitesmoke', which='minor')
    
    for ax in axs.flat:
        ax.set(xlabel='Time')

 
    plt.tight_layout()
    
    plt.show()
    plt.close

#----------------------------------------------------------------------------

def avgsplot(hc1, hc2,mT1, mT2,rc1, rc2, duration_actual, save_it=False, first_loop=True):
    
    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    x = time_series()
    
    avgeast = hc1.mean(axis=1)
    avgwest = hc2.mean(axis=1)
    hcMean = (avgeast + avgwest) / 2

    avgrc1 = rc1.mean(axis=1)
    avgrc2 = rc2.mean(axis=1)
    avgrc = (avgrc1+avgrc2)/2

    avgmT1 = mT1.mean(axis=1)
    avgmT2 = mT2.mean(axis=1)
    avgmT = (avgmT1+avgmT2)/2

    hcMean = (hcMean / np.mean(hcMean))/25

    diff = 1 - np.mean(hcMean)

    hcMean = hcMean + diff
    avgrc = avgrc / np.mean(avgrc)
    avgmT = avgmT / np.mean(avgmT)

    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    p12 = np.load (DATA_PATH+'/temp/a1p2_power.npy')
    p21 = np.load (DATA_PATH+'/temp/a2p1_power.npy')
    p22 = np.load (DATA_PATH+'/temp/a2p2_power.npy')

    power = p11 + p12 + p21 + p22
    power = power / np.mean(power)

    title_string = user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours. Mean One-Wire Temperatures.'
    if save_it:
        plt.figure(figsize=(12, 8), dpi=300)
    try:
        plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60), labels=mins, rotation=270)
        if duration_actual > 48 and duration_actual < 300:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60))
        elif duration_actual > 300:
            pass
        else:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),15))
    
    except:
        print('')
        print(' \033[0;m Applying x-axis tick mark correction. \033[1;32m ')
        print('')
        mins = mins[1:]
        plt.xticks(ticks=np.arange(first_tick[0],(np.size(temp_times)),60), labels=mins, rotation=270)
        if duration_actual > 48 and duration_actual < 300:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60))
        elif duration_actual > 300:
            pass
        else:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),15))
    
    #plt.plot(x,avgeast, linestyle='-', c='c', linewidth=1, label='East Externals')
    #plt.plot(x,avgwest, linestyle='-', c='teal', linewidth=1, label='West Externals')
   # plt.plot(x,hcMean, linestyle='-', c='b', linewidth=1, label='All Externals')
    #plt.plot(x,avgmT1, linestyle='-', c='b', linewidth=1, label='Outer Box East Pathway')
    #plt.plot(x,avgmT2, linestyle='--', c='b', linewidth=1, label='West')
    plt.plot(x,avgmT, linestyle='-', c='orange', linewidth=1, label='All Outer Box')
    plt.plot(x,avgrc, linestyle='-', c='r', linewidth=1, label='All Inner Box')
    plt.plot(x,(power/(1/avgrc)), linestyle='-', c='k', linewidth=1, label='Powers / RC&mT')
    plt.plot(x,(power), linestyle='-', c='hotpink', linewidth=1, label='Powers')
    
    plt.xticks(rotation = 270)
    
    plt.xlabel('Time')
    plt.ylabel('Temperature / °C')
    plt.margins(x=0)
    plt.tight_layout(h_pad=0.2, w_pad=0.2, rect=[0.097,0.175,0.977,0.922]) #tuple (left, bottom, right, top),
    plt.title(title_string)
    if save_it:
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_avgTemps.png', bbox_inches="tight")
        print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
    else:
        plt.legend(loc="upper right", fontsize=8)
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
            avgsplot(hc1, hc2,mT1, mT2,rc1, rc2, duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 


#------------------------------------------------------------------------------------------

def RPG_TEMPS(save_it=False, first_loop=True):

    title_string = 'Temperatures Time Series, RPG Spectrometer.'

    RPG_temps = np.load(DATA_PATH+'/temp/RPG_temps.npy')

    duration_actual = np.size(RPG_temps[:,0])/60
    duration_actual = np.round(duration_actual, decimals=2)

    RPG_times = Time(RPG_temps[:,0],format='mjd',scale='utc',precision=9)
    RPG_times.format = 'iso'
    time = RPG_times.tt.datetime

    fig, ax = plt.subplots()

  #  RPG_temps[:,1] = RPG_temps[:,1] / np.mean(RPG_temps[:,1])
   # RPG_temps[:,2] = RPG_temps[:,2] / np.mean(RPG_temps[:,2])
   # RPG_temps[:,3] = RPG_temps[:,3] / np.mean(RPG_temps[:,3])
    
    ax.plot(time,RPG_temps[:,1], linestyle='-', c='firebrick', linewidth=1, label='ADC')
    ax.plot(time,RPG_temps[:,2], linestyle='-', c='g', linewidth=1, label='FPGA')
    ax.plot(time,RPG_temps[:,3], linestyle='-', c='b', linewidth=1, label='Board')
       
    ax.set(xlabel='Time')
    ax.set(ylabel='Temperature / °C')
    ax.grid(c='darkgrey', which='major')
    ax.grid(c='gainsboro', which='minor')

    ax.margins(x=0)

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


    ax.legend(bbox_to_anchor=(1.1,1),loc="upper right", fontsize=9)
    ax.set_title(title_string)

    fig.tight_layout()

    if save_it:
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_RPGTemps.png', bbox_inches="tight")
        print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
    else:
        plt.legend(loc="upper right", fontsize=8)
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
            RPG_TEMPS(save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 


#---------------------------------------------------------------------

def ADC_NORM(save_it=False, first_loop=True):

    title_string = 'Temperatures Time Series, ADC\n'+'Normalised against a target temperature of 58.3°C'

    RPG_temps = np.load(DATA_PATH+'/temp/RPG_temps.npy')

    duration_actual = np.size(RPG_temps[:,0])/60
    duration_actual = np.round(duration_actual, decimals=2)

    RPG_times = Time(RPG_temps[:,0],format='mjd',scale='utc',precision=9)
    RPG_times.format = 'iso'
    time = RPG_times.tt.datetime

    fig, ax = plt.subplots()

  #  RPG_temps[:,1] = RPG_temps[:,1] / np.mean(RPG_temps[:,1])
    RPG_temps[:,1] = RPG_temps[:,1] / 58.3
    
    ax.plot(time,RPG_temps[:,1], linestyle='-', c='firebrick', linewidth=1, label='ADC')
       
    ax.set(xlabel='Time')
    ax.set(ylabel='Ratio to Target Temperature')
    ax.grid(c='darkgrey', which='major')
    ax.grid(c='gainsboro', which='minor')

    ax.margins(x=0)

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


    ax.legend(bbox_to_anchor=(1.1,1),loc="upper right", fontsize=9)
    ax.set_title(title_string)

    fig.tight_layout()

    if save_it:
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_ADCTemp.png', bbox_inches="tight")
        print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
    else:
        plt.legend(loc="upper right", fontsize=8)
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
            ADC_NORM(save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 
    

#---------------------------------------------------------------------



def OnewireMenu(duration_actual, user_inputs):

    looper = True
    temp_data = np.load(DATA_PATH+'/temp/one_wire.npy', allow_pickle=True)
    rc1, rc2, hc1, hc2, mT1, mT2, temp_times, duration_actual,no_data = groupSensors(duration_actual)
    RPG = np.load(DATA_PATH+'/temp/RPG_exist.npy')

    print ('')
    print ('   -------------------------------------')
    print ('   >>>    TEMPERATURE PLOTS MENU     <<<')
    print ('   -------------------------------------')
    print ('')
    if no_data:
        print ('\033[1;90m   1 - Horns \033[1;32m')
        print ('\033[1;90m   2 - Cables \033[1;32m')
        print ('\033[1;90m   3 - Magic T \033[1;32m')
        print ('\033[1;90m   4 - Reciever \033[1;32m')
        print ('\033[1;90m   5 - All One-Wire \033[1;32m')
    else:
        print ('   1 - Horns')  
        print ('   2 - Cables')
        print ('   3 - Magic T')
        print ('   4 - Reciever')
        print ('   5 - All One-Wire')
    print ('')
    if RPG:
        print ('   6 - RPG')
    else:
        print ('\033[1;90m   6 - RPG \033[1;32m') 
    if RPG:
        print ('   7 - ADC')
    else:
        print ('\033[1;90m   6 - ADC \033[1;32m') 
    print ('')
    print ('')
    print ('   0 - Return to previous menu')
    print ('')
    choice = input('Select menu option (number): \033[0;m')
    print('\033[1;32m')
    if choice.isdigit():        
        if int(choice) == 1:
            hornplot(hc1, hc2,  duration_actual)

        if int(choice) == 2:
            hcplot(hc1, hc2,   duration_actual)
        
        elif int(choice) ==3:
            mTplot(mT1, mT2,  duration_actual)
        
        elif int(choice) ==4:
            rcplot(rc1, rc2,  duration_actual)
                
        elif int(choice) ==5:
            partDeux(rc1, rc2, mT1, mT2, hc1, hc2,  user_inputs, duration_actual)
            parameters = np.load (DATA_PATH+'/temp/parameters.npy')
            if parameters[5] == 'True':
                print ('')
                save = input ('Do you want to save a printer friendly copy of this plot? (Y/N): ')
                if str(save) == 'N' or str(save) == 'n':
                    pass
                elif str(save) == 'Y' or str(save) == 'y':
                    threeINone(rc1, rc2, mT1, mT2, hc1, hc2, user_inputs, duration_actual)
            else:
                pass

    
        if int(choice) == 6:
            if RPG:
                RPG_TEMPS()
            else:
                pass

        if int(choice) == 7:
            if RPG:
                ADC_NORM()
            else:
                pass
  

        elif int(choice) == 8:
            avgsplot(hc1, hc2,mT1, mT2,rc1, rc2, duration_actual)

        elif int(choice) == 0:
            looper = False
            pass

        else:
            pass
           # print('\033[1;31m No such option. Please try again.\033[1;32m')
           # OnewireMenu(duration_actual, user_inputs)
         
    else:
        print('\033[1;31m Invalid selection by user. Please try again. \033[1;32m')
        OnewireMenu(duration_actual, user_inputs)

    return looper

#------------------------------------------------------------------------------

duration_actual = np.load(DATA_PATH+'/temp/duration_actual.npy')
temp_data = np.load(DATA_PATH+'/temp/one_wire.npy', allow_pickle=True)

user_inputs = np.load(DATA_PATH+'/temp/inputs.npy', allow_pickle=True)


    
print ('\033[1;32m ')


looper = True
while looper:
    looper = OnewireMenu(duration_actual, user_inputs)


os.system('chmod -R -f 0777 /mirror/scratch/pblack || true')

