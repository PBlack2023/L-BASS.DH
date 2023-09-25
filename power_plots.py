#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:56:25 2022

@author: pblack
"""

DATA_PATH = '/mirror/scratch/pblack'


import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.dates as mdates
from matplotlib.dates import AutoDateFormatter, AutoDateLocator, DateFormatter
#import matplotlib as mpl
import numpy as np
import math
from astropy.time import Time 
from astropy.time import TimeDelta
import datetime
import os
import gc

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
                    print('\033[0;0m Normalising against bandpass profile:',profiles[i,0],' \033[1;32m')
                    bandpass_norms = [profiles[i,4],profiles[i,5],profiles[i,6],profiles[i,7]]
                else:
                    pass
            except:
                print('')
                print('\033[1;31m Bandpass profile failed to load. \033[1;32m')

    if flatten:
                normload = np.load(bandpass_norms[0]+'.npy')
                a1p1 = a1p1 / np.mean(normload[112:569]) #flatten bandpass
                normload = np.load(bandpass_norms[1]+'.npy')
                a1p2 = a1p2 / np.mean(normload[112:569]) #flatten bandpass
                normload = np.load(bandpass_norms[2]+'.npy')
                a2p1 = a2p1 / np.mean(normload[112:569]) #flatten bandpass
                normload = np.load(bandpass_norms[3]+'.npy')
                a2p2 = a2p2 / np.mean(normload[112:569]) #flatten bandpass
  
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


##################################################################################


def rawData(duration_actual,save_it=False, first_loop=True):

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    passive_totals = np.load(DATA_PATH+'/temp/passives_totals.npy')
# 0 time, 1 p11 trans factor, 2 p12 transfactor, 3 p21 transfactor, 4 p22 transfactor, 5 p11 addtherm, 6 p12 addtherm, 7 p21 addtherm, 8 p22 addtherm     

    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    p12 = np.load (DATA_PATH+'/temp/a1p2_power.npy')
    p21 = np.load (DATA_PATH+'/temp/a2p1_power.npy')
    p22 = np.load (DATA_PATH+'/temp/a2p2_power.npy')

    p11,p12,p21,p22 = corrections(p11,p12,p21,p22)

    if parameters[8] == 'True':
        p11 = p11/np.mean(p11)
        p12 = p12/np.mean(p12)
        p21 = p21/np.mean(p21)
        p22 = p22/np.mean(p22)

  #  p11_boosted = (( p11[:] / (1 - passive_totals[:,1]) )  )# / 0.1436 ) #- passive_totals[:,5]
  #  p11_boosted = (( p11[:] / (1 - passive_totals[:,1])  )/ 0.151055177) - passive_totals[:,5]
  #  p12_boosted = ((p12[:] / (1 - passive_totals[:,2])) / 0.151055177) - passive_totals[:,6]
  #  p21_boosted = ((p21[:] / (1 - passive_totals[:,3])) / 0.151055177) - passive_totals[:,7]
  #  p22_boosted = ((p22[:] / (1 - passive_totals[:,4])) / 0.151055177) - passive_totals[:,8]
  
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = 'Powers Time Series, All Inputs'

    time2 = time_series()

 
    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

    ax.plot(time2,p21[:], color='firebrick', linewidth=2, label='P(R,$\pi$) E') # linestyle=(0,(5,1)),
    ax.plot(time2,p12[:], color='deeppink', linewidth=1, label='P(L,0) E')
    ax.plot(time2,p22[:], color='c', linewidth=2, label='P(R,0) W') #, linestyle=(0,(5,1))
    ax.plot(time2,p11[:], color='b', linewidth=1, label='P(L,$\pi$) W')
#    ax.plot(time2[:],p11_boosted[:], color='b', linestyle='--',linewidth=1, label='P(L,$\pi$) W')
 #   ax.plot(time2[:],p12_boosted[:], color='deeppink', linestyle='--',linewidth=1, label='P(L,0) E')
  #  ax.plot(time2[:],p21_boosted[:], color='firebrick', linestyle='--',linewidth=2, label='P(R,$\pi$) E')
   # ax.plot(time2[:],p22_boosted[:], color='c', linestyle='--',linewidth=2, label='P(R,0) W')

    ax.grid(c='darkgrey', which='major')
    ax.grid(c='gainsboro', which='minor')
    ax.set(xlabel='Time')

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


    if parameters[8]=='True' or parameters[1]=='True':
        if parameters[1]=='True' and parameters[8]=='True':
            ax.set(ylabel='Global Normalised Power (Normalised Axis)')
        elif parameters[1]=='True':
            ax.set(ylabel='Global Normalised Power')
        else:  
            ax.set(ylabel='Power (Normalised Axis)')
       # plt.ylim([0.985,1.035])
    else:
        ax.set(ylabel='Power / a.u.')

    y_limits = np.load(DATA_PATH+'/temp/ylims.npy')
    if y_limits:
        print('')
        man_lims = input('\033[1;32m Would you like to specify the y-axis limits on this plot? (Y/N): \033[0;m')
        if man_lims == str('Y') or man_lims == str('y'):
            print('')
            lower_limit = float(input('\033[1;32m Please enter an integer or decimal Lower Limit: \033[0;m'))
            upper_limit = float(input('\033[1;32m Please enter an integer or decimal Upper Limit: \033[0;m'))
        else:
            lower_limit = None
            upper_limit = None
    else:
        lower_limit = None
        upper_limit = None

    print('\033[1;32m')

    ax.set_ylim([lower_limit,upper_limit])

    if save_it:
        ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        ax.legend(bbox_to_anchor=(1.1,1),loc="upper right", fontsize=9)
    ax.set_title(title_string)
    fig.tight_layout()
    ax.margins(x=0)
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_PowerTS.png', bbox_inches="tight")
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
            rawData(duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 

    del p11,p12,p21,p22

    #-----------------------------------------------------------------

def firstDiff(duration_actual,save_it=False, first_loop=True):
    
    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = 'Powers Time Series, Single Differences'

    time2 = time_series()

    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    p12 = np.load (DATA_PATH+'/temp/a1p2_power.npy')
    p21 = np.load (DATA_PATH+'/temp/a2p1_power.npy')
    p22 = np.load (DATA_PATH+'/temp/a2p2_power.npy')

    p11,p12,p21,p22 = corrections(p11,p12,p21,p22)

    if parameters[8] == 'True':
        p11 = p11/np.mean(p11)
        p12 = p12/np.mean(p12)
        p21 = p21/np.mean(p21)
        p22 = p22/np.mean(p22)

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

    d_a = p12 - p11 # l0 - lpi 
    d_b = p22 - p21 #r0 - rpi 
    d_c = p12 - p22  # l0 - r0 
    d_d = p11 - p21  #   lpi - rpi 

    del p11,p12,p21,p22

    ax.plot(time2,d_a, c='b', label='$\delta$a = P(L,0)E - P(L,$\pi$)W') #WMAPa scheme
    ax.plot(time2,d_b, c='orange', label='$\delta$b = P(R,0)W - P(R,$\pi$)E') #WMAPb scheme
    ax.plot(time2,d_c, c='mediumseagreen', label='$\delta$c = P(L,0)E - P(R,0)W') #OCRAa scheme
    ax.plot(time2,d_d, c='darkgreen', label='$\delta$d = P(L,$\pi$)W - P(R,$\pi$)E') #OCRAb scheme

    ax.grid(c='darkgrey', which='major')
    ax.grid(c='gainsboro', which='minor')
    ax.set(xlabel='Time')

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

    if parameters[8]=='True' or parameters[1]=='True':
        if parameters[1]=='True' and parameters[8]=='True':
            ax.set(ylabel='Global Normalised Power (Normalised Axis)')
        elif parameters[1]=='True':
            ax.set(ylabel='Global Normalised Power')
        else:  
            ax.set(ylabel='Power (Normalised Axis)')
       # plt.ylim([0.985,1.035])
    else:
        ax.set(ylabel='Power / a.u.')
       # ax.set_ylim(30,40)

    y_limits = np.load(DATA_PATH+'/temp/ylims.npy')
    if y_limits:
        print('')
        man_lims = input('\033[1;32m Would you like to specify the y-axis limits on this plot? (Y/N): \033[0;m')
        if man_lims == str('Y') or man_lims == str('y'):
            print('')
            lower_limit = float(input('\033[1;32m Please enter an integer or decimal Lower Limit: \033[0;m'))
            upper_limit = float(input('\033[1;32m Please enter an integer or decimal Upper Limit: \033[0;m'))
        else:
            lower_limit = None
            upper_limit = None
    else:
        lower_limit = None
        upper_limit = None

    print('\033[1;32m')

    ax.set_ylim([lower_limit,upper_limit])

    del d_a, d_b, d_c, d_d

    plt.title(title_string)
    if save_it:
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        plt.legend(bbox_to_anchor=(1.1,1), loc="upper right", fontsize=9)
    plt.margins(x=0)
    fig.tight_layout()
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_single_diffs.png', bbox_inches="tight")
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
            firstDiff(duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 
    
#-------------------------------------------------------------------
 
def doubleDiff(save_it=False, first_loop=True):

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    p12 = np.load (DATA_PATH+'/temp/a1p2_power.npy')
    p21 = np.load (DATA_PATH+'/temp/a2p1_power.npy')
    p22 = np.load (DATA_PATH+'/temp/a2p2_power.npy')

    p11,p12,p21,p22 = corrections(p11,p12,p21,p22)

    if parameters[8] == 'True':
        p11 = p11/np.mean(p11)
        p12 = p12/np.mean(p12)
        p21 = p21/np.mean(p21)
        p22 = p22/np.mean(p22)

    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = 'Powers Time Series, Double Difference'

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
   # DD = (d_c - d_d)/2 # OCRA
    NULL_JPL = (JPL_d1 - JPL_d2)/2
    del d_a, d_b, d_c, d_d
    
    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

   # ax.plot(time2,NULL_JPL, c='tab:grey', label ='JPL Null', linewidth=0.5)
    ax.plot(time2,DD, c='r', linestyle='-', linewidth=1, label ='Signal') #WMAPa - WMAPb

    ax.grid(c='darkgrey', which='major')
    ax.grid(c='gainsboro', which='minor')
    ax.set(xlabel='Time')

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


    if parameters[8]=='True' or parameters[1]=='True':
        if parameters[1]=='True' and parameters[8]=='True':
            ax.set(ylabel='Global Normalised Power (Normalised Axis)')
        elif parameters[1]=='True':
            ax.set(ylabel='Global Normalised Power')
        else:  
            ax.set(ylabel='Power (Normalised Axis)')
       # plt.ylim([0.985,1.035])
    else:
        ax.set(ylabel='Power / a.u.')
       # ax.set_ylim(30,40)

    y_limits = np.load(DATA_PATH+'/temp/ylims.npy')
    if y_limits:
        print('')
        man_lims = input('\033[1;32m Would you like to specify the y-axis limits on this plot? (Y/N): \033[0;m')
        if man_lims == str('Y') or man_lims == str('y'):
            print('')
            lower_limit = float(input('\033[1;32m Please enter an integer or decimal Lower Limit: \033[0;m'))
            upper_limit = float(input('\033[1;32m Please enter an integer or decimal Upper Limit: \033[0;m'))
        else:
            lower_limit = None
            upper_limit = None
    else:
        lower_limit = None
        upper_limit = None

    print('\033[1;32m')

    ax.set_ylim([lower_limit,upper_limit])

    if save_it:
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        plt.legend(bbox_to_anchor=(1.1,1), loc="upper right", fontsize=9)
    plt.margins(x=0)
    fig.tight_layout()

    plt.title(title_string)

    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_double_diffs.png', bbox_inches="tight")
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
            doubleDiff(save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 
    
#---------------------------------------------------------------------

def PowerNulls(save_it=False, first_loop=True):

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    p12 = np.load (DATA_PATH+'/temp/a1p2_power.npy')
    p21 = np.load (DATA_PATH+'/temp/a2p1_power.npy')
    p22 = np.load (DATA_PATH+'/temp/a2p2_power.npy')

    p11,p12,p21,p22 = corrections(p11,p12,p21,p22)

    if parameters[8] == 'True':
        p11 = p11/np.mean(p11)
        p12 = p12/np.mean(p12)
        p21 = p21/np.mean(p21)
        p22 = p22/np.mean(p22)

    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = 'Powers Time Series, Nulls'

    time2 = time_series()

    JPL_d1 = p11 - p12 # lpi - l0 W-E
    JPL_d2 = p22 - p21 # r0 - rpi W-E
    NULL_IB2_a = p12 - p21 # l0 - rpi  E-E
    NULL_IB2_b = p22 - p11 # r0 - lpi   W-W
    del p11,p12,p21,p22

    NULL_JPL = (JPL_d1 - JPL_d2)/2
    NULL_IBX = (NULL_IB2_a + NULL_IB2_b) /2
    
    NULLEST = NULL_JPL + NULL_IBX

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

    ax.plot(time2,NULL_IB2_a, c='r', label ='IB Null E-E', linewidth=1)
    ax.plot(time2,NULL_IB2_b, c='b', label ='IB Null W-W', linewidth=1)
    ax.plot(time2,NULL_JPL, c='k', label ='JPL Null', linewidth=1)
   # ax.plot(time2,NULL_IBX, c='k', label ='IBx Null', linewidth=1)
   # ax.plot(time2,NULLEST, c='purple', label ='Pips Null', linewidth=1)

    ax.grid(c='darkgrey', which='major')
    ax.grid(c='gainsboro', which='minor')
    ax.set(xlabel='Time')

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

    plt.margins(x=0)

    if parameters[8]=='True' or parameters[1]=='True':
        if parameters[1]=='True' and parameters[8]=='True':
            ax.set(ylabel='Global Normalised Power (Normalised Axis)')
        elif parameters[1]=='True':
            ax.set(ylabel='Global Normalised Power')
        else:  
            ax.set(ylabel='Power (Normalised Axis)')
       # plt.ylim([0.985,1.035])
    else:
        ax.set(ylabel='Power / a.u.')
       # ax.set_ylim(30,40)

    y_limits = np.load(DATA_PATH+'/temp/ylims.npy')
    if y_limits:
        print('')
        man_lims = input('\033[1;32m Would you like to specify the y-axis limits on this plot? (Y/N): \033[0;m')
        if man_lims == str('Y') or man_lims == str('y'):
            print('')
            lower_limit = float(input('\033[1;32m Please enter an integer or decimal Lower Limit: \033[0;m'))
            upper_limit = float(input('\033[1;32m Please enter an integer or decimal Upper Limit: \033[0;m'))
        else:
            lower_limit = None
            upper_limit = None
    else:
        lower_limit = None
        upper_limit = None

    print('\033[1;32m')

    ax.set_ylim([lower_limit,upper_limit])

    plt.title(title_string)
    if save_it:
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        plt.legend(bbox_to_anchor=(1.1,1), loc="upper right", fontsize=9)
    plt.margins(x=0)
    fig.tight_layout()

    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_nulls.png', bbox_inches="tight")
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
            PowerNulls(save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 

#-----------------------------------------------------------------------------------------

def powerRatio(duration_actual,save_it=False, first_loop=True):

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    p12 = np.load (DATA_PATH+'/temp/a1p2_power.npy')
    p21 = np.load (DATA_PATH+'/temp/a2p1_power.npy')
    p22 = np.load (DATA_PATH+'/temp/a2p2_power.npy')

    p11,p12,p21,p22 = corrections(p11,p12,p21,p22)
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = 'Powers Time Series, Input Ratios'

    time2 = time_series()

    #load first


    if save_it:
        fig, axarr = plt.subplots(2, 2, figsize=(12,8), dpi=300)
    else:
        fig, axarr = plt.subplots(2, 2)
 
# A1P1 - P(l,pi)
# A1P2 - P(l,0)
# A2P1 - P(r,pi)
# A2P2 - P(r,0)

    p22a = p22 / p11
    p11a = p11 / p11
    p12a = p12 / p11  #0 1
    p21a = p21 / p11

    p22b = p22 / p12
    p11b = p11 / p12  #0 0
    p12b = p12 / p12
    p21b = p21 / p12

    p22c = p22 / p21
    p11c = p11 / p21  #1 0
    p12c = p12 / p21
    p21c = p21 / p21

    p22d = p22 / p22
    p11d = p11 / p22  #1 1
    p12d = p12 / p22
    p21d = p21 / p22

    axarr[0,0].plot(time2, p21b, color='firebrick', label='P(R,$\pi$) E')
    axarr[0,0].plot(time2,p12b, color='deeppink', label='P(L,0) E')
    axarr[0,0].plot(time2,p11b, color='b', label='P(L,$\pi$) W')
    axarr[0,0].plot(time2,p22b, color='c', label='P(R,0) W')
    axarr[0,0].plot(time2,p12b, color='deeppink')
    axarr[0,0].set_title('Inputs divided by P(L,0)')
    axarr[0,0].legend()
    axarr[0,0].margins(x=0)

    axarr[0,1].plot(time2,p21a, color='firebrick', label='P(R,$\pi$) E')
    axarr[0,1].plot(time2,p12a, color='deeppink', label='P(L,0) E')
    axarr[0,1].plot(time2,p11a, color='b', label='P(L,$\pi$) W')
    axarr[0,1].plot(time2,p22a, color='c', label='P(R,0) W')
    axarr[0,1].plot(time2,p11a, color='b')
    axarr[0,1].set_title('Inputs divided by P(L,$\pi$)')
    axarr[0,1].legend()
    axarr[0,1].margins(x=0)

    axarr[1,0].plot(time2,p21c, color='firebrick', label='P(R,$\pi$) E')
    axarr[1,0].plot(time2,p12c, color='deeppink', label='P(L,0) E')
    axarr[1,0].plot(time2,p11c, color='b', label='P(L,$\pi$) W')
    axarr[1,0].plot(time2,p22c, color='c', label='P(R,0) W')
    axarr[1,0].set_title('Inputs divided by P(R,$\pi$)')
    axarr[1,0].plot(time2,p21c, color='firebrick')
    axarr[1,0].legend()
    axarr[1,0].margins(x=0)

    axarr[1,1].plot(time2,p21d, color='firebrick', label='P(R,$\pi$) E')
    axarr[1,1].plot(time2,p12d, color='deeppink', label='P(L,0) E')
    axarr[1,1].plot(time2,p11d, color='b', label='P(L,$\pi$) W')
    axarr[1,1].plot(time2,p22d, color='c', label='P(R,0) W')
    axarr[1,1].set_title('Inputs divided by P(R,0)')
    axarr[1,1].legend()
    axarr[1,1].margins(x=0)

    for ax in axarr.flat:
        ax.set(xlabel='Time', ylabel='Power Ratio')

        ax.grid(c='darkgrey', which='major')
        ax.grid(c='gainsboro', which='minor')
        ax.set(xlabel='Time')

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

    plt.suptitle(title_string)
    plt.tight_layout()

    
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_input_ratios_TS.png', bbox_inches="tight")
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
            powerRatio(duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 

    del p11,p12,p21,p22

#--------------------------------------------------------------------------------

def EASTWESTRATIO(duration_actual,save_it=False, first_loop=True):

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    p12 = np.load (DATA_PATH+'/temp/a1p2_power.npy')
    p21 = np.load (DATA_PATH+'/temp/a2p1_power.npy')
    p22 = np.load (DATA_PATH+'/temp/a2p2_power.npy')

    p11,p12,p21,p22 = corrections(p11,p12,p21,p22)

    if parameters[8] == 'True':
        p11 = p11/np.mean(p11)
        p12 = p12/np.mean(p12)
        p21 = p21/np.mean(p21)
        p22 = p22/np.mean(p22)

#P11 - P(l,pi)
#P22 - P(r,0)
#P12 - P(l,0)
#P21 - P(r,pi)

    PR1 = p22 / p21
    PR2 = p11 / p12
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = 'Powers Time Series, West/East Ratios'

    time2 = time_series()
    #load first

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()


    ax.plot(time2,PR1, color='firebrick', linewidth=1, label='P(R,0) W / P(R,$\pi$) E') # linestyle=(0,(5,1)),
    ax.plot(time2,PR2, color='b', linewidth=1, label='P(L,$\pi$) W / P(L,0) E')

    ax.grid(c='darkgrey', which='major')
    ax.grid(c='gainsboro', which='minor')
    ax.set(xlabel='Time')

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

    y_limits = np.load(DATA_PATH+'/temp/ylims.npy')
    if y_limits:
        print('')
        man_lims = input('\033[1;32m Would you like to specify the y-axis limits on this plot? (Y/N): \033[0;m')
        if man_lims == str('Y') or man_lims == str('y'):
            print('')
            lower_limit = float(input('\033[1;32m Please enter an integer or decimal Lower Limit: \033[0;m'))
            upper_limit = float(input('\033[1;32m Please enter an integer or decimal Upper Limit: \033[0;m'))
        else:
            lower_limit = None
            upper_limit = None
    else:
        lower_limit = None
        upper_limit = None

    print('\033[1;32m')

    ax.set_ylim([lower_limit,upper_limit])

    #plt.ylim([0.965,1.005])
    #plt.ylim([0.9875,1.0275])
    #plt.ylim([0.998,1.002]) #<<<<<<



    if parameters[8]=='True' or parameters[1]=='True':
        if parameters[1]=='True' and parameters[8]=='True':
            plt.ylabel('RATIO (Global Normalised Power (Normalised Axis))')
        elif parameters[1]=='True':
            plt.ylabel('RATIO (Global Normalised Power)')
        else:  
            plt.ylabel('RATIO (Power (Normalised Axis))')
    else:
        plt.ylabel('RATIO')

    if save_it:
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        plt.legend(bbox_to_anchor=(1.1,1), loc="upper right", fontsize=9)
    plt.margins(x=0)
    plt.title(title_string)
    fig.tight_layout()


    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_raw_data_TS_WoverE.png', bbox_inches="tight")
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
            EASTWESTRATIO(duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 

    del p11,p12,p21,p22

    #-----------------------------------------------------------------

####################################################################################


####################################################################

def PowerMenu(duration_actual):

    looper = True

    print ('')
    print ('   -------------------------------------')
    print ('   >>>       POWER PLOTS MENU        <<<')
    print ('   -------------------------------------')
    print ('')
    print ('   1 - All Inputs')
    print ('')
    print ('   2 - Single Differences')
    print ('')
    print ('   3 - Double Difference')
    print ('')
    print ('   4 - Nulls')
    print ('')
    print ('   5 - All Power Ratios')
    print ('')
    print ('   6 - West/East Ratios')
    print ('')
    print ('')
    print ('   0 - Return to Quick-look menu')
    print ('')
    choice = input('Select menu option (number): \033[0;m')
    print('\033[1;32m')
    if choice.isdigit():
        if int(choice) ==1:
            rawData(duration_actual)
        
        elif int(choice) ==2:
            firstDiff(duration_actual)
        
        elif int(choice) ==3:
            doubleDiff()
  
        elif int(choice) ==4:
            PowerNulls()

        elif int(choice) ==5:
            powerRatio(duration_actual)

        elif int(choice)  == 6:
            EASTWESTRATIO(duration_actual)

        elif int(choice) ==7:
            rawData_total(duration_actual)

        elif int(choice) == 8:
            RMSpowers(duration_actual)

        elif int(choice) == 0:
            looper = False
            pass

    else:
        print('\033[1;31m Invalid selection by user. Please try again. \033[1;32m')
        PowerMenu(duration_actual)

    return looper

#---------------------------------------------------------------------


duration_actual = np.load(DATA_PATH+'/temp/duration_actual.npy')

looper = True
while looper:
    looper = PowerMenu(duration_actual)

print ('\033[1;32m ')


os.system('chmod -R -f 0777 /mirror/scratch/pblack || true')

