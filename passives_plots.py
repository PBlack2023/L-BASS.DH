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




passive_totals = np.load(DATA_PATH+'/temp/passives.npy')
passive_table = np.load(DATA_PATH+'/temp/passives.npy')


def trans_ant_plot(save_it=False, first_loop=True):

    passive_table = np.load(DATA_PATH+'/temp/passives.npy')

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = 'Transmission Loss from Passives, Time Series, Antenna, All Inputs'

    time2 = time_series()
# 0 time, 1 p11 antenna,   2 p11 cabscons, 3  p11 receiver,  4  p12 antenna,  5  p12 cabscons,  6 p12 receiver
#         7 p22 antenna,   8 p22 cabscons, 9  p22 receiver,  10 p21 antenna,  11 p21 cabscons,  12 p21 receiver
#         13 p11 antenna, 14 p11 cabscons, 15 p11 receiver,  16 p12 antenna,  17 p12 cabscons,  18 p12 receiver
#         19 p22 antenna, 20 p22 cabscons, 21 p22 receiver,  22 p21 antenna,  23 p21 cabscons,  24 p21 receiver

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

    ax.plot(time2,passive_table[:,10]*-100, color='firebrick', linewidth=2, label='P(R,$\pi$) E') 
    ax.plot(time2,passive_table[:,4]*-100, color='deeppink', linewidth=1, label='P(L,0) E')
    ax.plot(time2,passive_table[:,7]*-100, color='c', linewidth=2, label='P(R,0) W') 
    ax.plot(time2,passive_table[:,1]*-100, color='b', linewidth=1, label='P(L,$\pi$) W')
# A1P1 - P(l,pi)
# A2P2 - P(r,0)
# A1P2 - P(l,0)
# A2P1 - P(r,pi)
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


    ax.set(ylabel='Signal Loss / %')

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
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_transfactors_antenna.png', bbox_inches="tight")
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
            trans_ant_plot(duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 


#----------------------------------------------------------------------------------------------------------------------------------------

def trans_cabcon_plot(save_it=False, first_loop=True):

    passive_table = np.load(DATA_PATH+'/temp/passives.npy')

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = 'Transmission Loss from Passives, Time Series, Cables & Connectors, All Inputs'

    time2 = time_series()
# 0 time, 1 p11 antenna,   2 p11 cabscons, 3  p11 receiver,  4  p12 antenna,  5  p12 cabscons,  6 p12 receiver
#         7 p22 antenna,   8 p22 cabscons, 9  p22 receiver,  10 p21 antenna,  11 p21 cabscons,  12 p21 receiver
#         13 p11 antenna, 14 p11 cabscons, 15 p11 receiver,  16 p12 antenna,  17 p12 cabscons,  18 p12 receiver
#         19 p22 antenna, 20 p22 cabscons, 21 p22 receiver,  22 p21 antenna,  23 p21 cabscons,  24 p21 receiver

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

    ax.plot(time2,passive_table[:,11]*-100, color='firebrick', linewidth=2, label='P(R,$\pi$) E') 
    ax.plot(time2,passive_table[:,5]*-100, color='deeppink', linewidth=1, label='P(L,0) E')
    ax.plot(time2,passive_table[:,8]*-100, color='c', linewidth=2, label='P(R,0) W') 
    ax.plot(time2,passive_table[:,2]*-100, color='b', linewidth=1, label='P(L,$\pi$) W')
# A1P1 - P(l,pi)
# A2P2 - P(r,0)
# A1P2 - P(l,0)
# A2P1 - P(r,pi)
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


    ax.set(ylabel='Signal Loss / %')

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
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_transfactors_cabcon.png', bbox_inches="tight")
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
            trans_ant_plot(duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 

#----------------------------------------------------------------------------------------------------------------------------------------



def trans_receiver_plot(save_it=False, first_loop=True):

    passive_table = np.load(DATA_PATH+'/temp/passives.npy')

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = 'Transmission Loss from Passives, Time Series, Receiver, All Inputs'

    time2 = time_series()
# 0 time, 1 p11 antenna,   2 p11 cabscons, 3  p11 receiver,  4  p12 antenna,  5  p12 cabscons,  6 p12 receiver
#         7 p22 antenna,   8 p22 cabscons, 9  p22 receiver,  10 p21 antenna,  11 p21 cabscons,  12 p21 receiver
#         13 p11 antenna, 14 p11 cabscons, 15 p11 receiver,  16 p12 antenna,  17 p12 cabscons,  18 p12 receiver
#         19 p22 antenna, 20 p22 cabscons, 21 p22 receiver,  22 p21 antenna,  23 p21 cabscons,  24 p21 receiver

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

    ax.plot(time2,passive_table[:,12]*-100, color='firebrick', linewidth=2, label='P(R,$\pi$) E') 
    ax.plot(time2,passive_table[:,6]*-100, color='deeppink', linewidth=1, label='P(L,0) E')
    ax.plot(time2,passive_table[:,9]*-100, color='c', linewidth=2, label='P(R,0) W') 
    ax.plot(time2,passive_table[:,3]*-100, color='b', linewidth=1, label='P(L,$\pi$) W')
# A1P1 - P(l,pi)
# A2P2 - P(r,0)
# A1P2 - P(l,0)
# A2P1 - P(r,pi)
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


    ax.set(ylabel='Signal Loss / %')

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
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_transfactors_receiver.png', bbox_inches="tight")
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
            trans_receiver_plot(duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 


#-------------------------------------------------------------------------------------------------------------------------------------------

def trans_totals_plot(save_it=False, first_loop=True):

    passive_totals = np.load(DATA_PATH+'/temp/passives_totals.npy')

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = 'Transmission Loss from Passives, Time Series, Totals, All Inputs'

    time2 = time_series()
# 0 time, 1 p11 trans factor, 2 p12 transfactor, 3 p21 transfactor, 4 p22 transfactor, 5 p11 addtherm, 6 p12 addtherm, 7 p21 addtherm, 8 p22 addtherm

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

    ax.plot(time2,passive_totals[:,3]*-100, color='firebrick', linewidth=2, label='P(R,$\pi$) E') 
    ax.plot(time2,passive_totals[:,2]*-100, color='deeppink', linewidth=1, label='P(L,0) E')
    ax.plot(time2,passive_totals[:,4]*-100, color='c', linewidth=2, label='P(R,0) W') 
    ax.plot(time2,passive_totals[:,1]*-100, color='b', linewidth=1, label='P(L,$\pi$) W')

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


    ax.set(ylabel='Signal Loss / %')

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
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_transfactors_totals.png', bbox_inches="tight")
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
            trans_totals_plot(duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 

#----------------------------------------------------------------------------------------------------------------------
##############################################################################################

def therm_ant_plot(save_it=False, first_loop=True):

    passive_table = np.load(DATA_PATH+'/temp/passives.npy')

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = 'Thermal Addition from Passives, Time Series, Antenna, All Inputs'

    time2 = time_series()
# 0 time, 1 p11 antenna,   2 p11 cabscons, 3  p11 receiver,  4  p12 antenna,  5  p12 cabscons,  6 p12 receiver
#         7 p22 antenna,   8 p22 cabscons, 9  p22 receiver,  10 p21 antenna,  11 p21 cabscons,  12 p21 receiver
#         13 p11 antenna, 14 p11 cabscons, 15 p11 receiver,  16 p12 antenna,  17 p12 cabscons,  18 p12 receiver
#         19 p22 antenna, 20 p22 cabscons, 21 p22 receiver,  22 p21 antenna,  23 p21 cabscons,  24 p21 receiver

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

    ax.plot(time2,passive_table[:,22], color='firebrick', linewidth=2, label='P(R,$\pi$) E') 
    ax.plot(time2,passive_table[:,16], color='deeppink', linewidth=1, label='P(L,0) E')
    ax.plot(time2,passive_table[:,19], color='c', linewidth=2, label='P(R,0) W') 
    ax.plot(time2,passive_table[:,13], color='b', linewidth=1, label='P(L,$\pi$) W')
# A1P1 - P(l,pi)
# A2P2 - P(r,0)
# A1P2 - P(l,0)
# A2P1 - P(r,pi)
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


    ax.set(ylabel='Additional Thermal Contribution / K')

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
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_thermadd_antenna.png', bbox_inches="tight")
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
            therm_ant_plot(duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 


#----------------------------------------------------------------------------------------------------------------------------------------


def therm_cabcons_plot(save_it=False, first_loop=True):

    passive_table = np.load(DATA_PATH+'/temp/passives.npy')

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = 'Thermal Addition from Passives, Time Series, Cables & Connectors, All Inputs'

    time2 = time_series()
# 0 time, 1 p11 antenna,   2 p11 cabscons, 3  p11 receiver,  4  p12 antenna,  5  p12 cabscons,  6 p12 receiver
#         7 p22 antenna,   8 p22 cabscons, 9  p22 receiver,  10 p21 antenna,  11 p21 cabscons,  12 p21 receiver
#         13 p11 antenna, 14 p11 cabscons, 15 p11 receiver,  16 p12 antenna,  17 p12 cabscons,  18 p12 receiver
#         19 p22 antenna, 20 p22 cabscons, 21 p22 receiver,  22 p21 antenna,  23 p21 cabscons,  24 p21 receiver

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

    ax.plot(time2,passive_table[:,23], color='firebrick', linewidth=2, label='P(R,$\pi$) E') 
    ax.plot(time2,passive_table[:,17], color='deeppink', linewidth=1, label='P(L,0) E')
    ax.plot(time2,passive_table[:,20], color='c', linewidth=2, label='P(R,0) W') 
    ax.plot(time2,passive_table[:,14], color='b', linewidth=1, label='P(L,$\pi$) W')
# A1P1 - P(l,pi)
# A2P2 - P(r,0)
# A1P2 - P(l,0)
# A2P1 - P(r,pi)
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


    ax.set(ylabel='Additional Thermal Contribution / K')

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
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_thermadd_cabcons.png', bbox_inches="tight")
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
            therm_cabcons_plot(duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 


#----------------------------------------------------------------------------------------------------------------------------------------

def therm_receiver_plot(save_it=False, first_loop=True):

    passive_table = np.load(DATA_PATH+'/temp/passives.npy')

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = 'Thermal Addition from Passives, Time Series, Reciever, All Inputs'

    time2 = time_series()
# 0 time, 1 p11 antenna,   2 p11 cabscons, 3  p11 receiver,  4  p12 antenna,  5  p12 cabscons,  6 p12 receiver
#         7 p22 antenna,   8 p22 cabscons, 9  p22 receiver,  10 p21 antenna,  11 p21 cabscons,  12 p21 receiver
#         13 p11 antenna, 14 p11 cabscons, 15 p11 receiver,  16 p12 antenna,  17 p12 cabscons,  18 p12 receiver
#         19 p22 antenna, 20 p22 cabscons, 21 p22 receiver,  22 p21 antenna,  23 p21 cabscons,  24 p21 receiver

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

    ax.plot(time2,passive_table[:,24], color='firebrick', linewidth=2, label='P(R,$\pi$) E') 
    ax.plot(time2,passive_table[:,18], color='deeppink', linewidth=1, label='P(L,0) E')
    ax.plot(time2,passive_table[:,21], color='c', linewidth=2, label='P(R,0) W') 
    ax.plot(time2,passive_table[:,15], color='b', linewidth=1, label='P(L,$\pi$) W')
# A1P1 - P(l,pi)
# A2P2 - P(r,0)
# A1P2 - P(l,0)
# A2P1 - P(r,pi)
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


    ax.set(ylabel='Additional Thermal Contribution / K')

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
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_thermadd_receiver.png', bbox_inches="tight")
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
            therm_receiver_plot(duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 

#----------------------------------------------------------------------------------------------------------------------------------


def therm_totals_plot(save_it=False, first_loop=True):

    passive_totals = np.load(DATA_PATH+'/temp/passives_totals.npy')

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = 'Thermal Addition from Passives, Time Series, Totals, All Inputs'

    time2 = time_series()
# 0 time, 1 p11 trans factor, 2 p12 transfactor, 3 p21 transfactor, 4 p22 transfactor, 5 p11 addtherm, 6 p12 addtherm, 7 p21 addtherm, 8 p22 addtherm

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

    ax.plot(time2,passive_totals[:,7], color='firebrick', linewidth=2, label='P(R,$\pi$) E') 
    ax.plot(time2,passive_totals[:,6], color='deeppink', linewidth=1, label='P(L,0) E')
    ax.plot(time2,passive_totals[:,8], color='c', linewidth=2, label='P(R,0) W') 
    ax.plot(time2,passive_totals[:,5], color='b', linewidth=1, label='P(L,$\pi$) W')



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


    ax.set(ylabel='Additional Thermal Contribution / K')

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
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_thermadd_totals.png', bbox_inches="tight")
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
            therm_totals_plot(duration_actual, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 




#plt.title('Transmission Factor')
#plt.plot(x, trans_table[:,0], color ='b', linewidth=1, label='East Antenna',linestyle='--')
#plt.plot(x, trans_table[:,1], color ='b', linewidth=1, label='East 4m Cabs&Cons',linestyle='-.')
#plt.plot(x, trans_table[:,2], color ='b', linewidth=1, label='East Receiver',linestyle=':')

#plt.plot(x, trans_table[:,4], color ='r', linewidth=1, label='West Antenna',linestyle='--')
#plt.plot(x, trans_table[:,5], color ='r', linewidth=1, label='West 4m Cabs&Cons',linestyle='-.')
#plt.plot(x, trans_table[:,6], color ='r', linewidth=1, label='West Receiver',linestyle=':')

#plt.xlabel('Minutes')
#plt.ylabel('Percentage loss')
#plt.margins(x=0)
#plt.legend()
#plt.show()

#plt.title('Radiometric Temperature Contribution')
#plt.plot(x, therm_table[:,0], color ='b', linewidth=1, label='East Antenna',linestyle='--')
#plt.plot(x, therm_table[:,1], color ='b', linewidth=1, label='East 4m Cabs&Cons',linestyle='-.')
#plt.plot(x, therm_table[:,2], color ='b', linewidth=1, label='East Receiver',linestyle=':')

#plt.plot(x, therm_table[:,4], color ='r', linewidth=1, label='West Antenna',linestyle='--')
#plt.plot(x, therm_table[:,5], color ='r', linewidth=1, label='West 4m Cabs&Cons',linestyle='-.')
#plt.plot(x, therm_table[:,6], color ='r', linewidth=1, label='West Receiver',linestyle=':')

#plt.xlabel('Minutes')
#plt.ylabel('Thermal Addition / K')
#plt.margins(x=0)
#plt.legend()
#plt.show()






#---------------------------------------------------------------------



def PassMenu(duration_actual, user_inputs):

    looper = True

    print ('')
    print ('   -------------------------------------')
    print ('   >>>     PASSIVES PLOTS MENU       <<<')
    print ('   -------------------------------------')
    print ('')
    print ('   Transmission Factors')
    print ('   1 - Antenna')  
    print ('   2 - Cables & Connectors')
    print ('   3 - Reciever')
    print ('   4 - Total')
    print ('')
    print ('   Thermal Contribution')
    print ('   5 - Antenna')  
    print ('   6 - Cables & Connectors')
    print ('   7 - Reciever')
    print ('   8 - Total')
    print ('')
    print ('')
    print ('   0 - Return to previous menu')
    print ('')
    choice = input('Select menu option (number): \033[0;m')
    print('\033[1;32m')
    if choice.isdigit():    

        if int(choice) == 1:
            trans_ant_plot()

        if int(choice) == 2:
            trans_cabcon_plot()

        if int(choice) == 3:
            trans_receiver_plot()
    
        if int(choice) == 4:
            trans_totals_plot()

        if int(choice) == 5:
            therm_ant_plot()

        if int(choice) == 6:
            therm_cabcons_plot()

        if int(choice) == 7:
            therm_receiver_plot()

        if int(choice) == 8:
            therm_totals_plot()

   #     if int(choice) == 2:
    #        hcplot(hc1, hc2,   duration_actual)
        
     #   elif int(choice) ==3:
      #      mTplot(mT1, mT2,  duration_actual)
        
       # elif int(choice) ==4:
        #    rcplot(rc1, rc2,  duration_actual)
                
    #    elif int(choice) ==5:
     #       partDeux(rc1, rc2, mT1, mT2, hc1, hc2,  user_inputs, duration_actual)
      #      parameters = np.load (DATA_PATH+'/temp/parameters.npy')
       #     if parameters[5] == 'True':
        #        print ('')
         #       save = input ('Do you want to save a printer friendly copy of this plot? (Y/N): ')
          #      if str(save) == 'N' or str(save) == 'n':
           #         pass
            #    elif str(save) == 'Y' or str(save) == 'y':
             #       threeINone(rc1, rc2, mT1, mT2, hc1, hc2, user_inputs, duration_actual)
           # else:
            #    pass

    
      #  if int(choice) == 6:
       #     if RPG:
        #        RPG_TEMPS()
         #   else:
          #      pass

  #      if int(choice) == 7:
   #         if RPG:
    #            ADC_NORM()
     #       else:
      #          pass
  

       # elif int(choice) == 8:
        #    avgsplot(hc1, hc2,mT1, mT2,rc1, rc2, duration_actual)

        elif int(choice) == 0:
            looper = False
            pass

      #  else:
       #     pass
           # print('\033[1;31m No such option. Please try again.\033[1;32m')
           # OnewireMenu(duration_actual, user_inputs)
         
    else:
        print('\033[1;31m Invalid selection by user. Please try again. \033[1;32m')
        PassMenu(duration_actual, user_inputs)

    return looper

#------------------------------------------------------------------------------

duration_actual = np.load(DATA_PATH+'/temp/duration_actual.npy')
temp_data = np.load(DATA_PATH+'/temp/one_wire.npy', allow_pickle=True)

user_inputs = np.load(DATA_PATH+'/temp/inputs.npy', allow_pickle=True)


    
print ('\033[1;32m ')


looper = True
while looper:
    looper = PassMenu(duration_actual, user_inputs)


os.system('chmod -R -f 0777 /mirror/scratch/pblack || true')

