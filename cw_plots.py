#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:56:25 2022

@author: pblack
"""

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
from matplotlib import ticker
import matplotlib.dates as mdates
from matplotlib.dates import AutoDateFormatter, AutoDateLocator, DateFormatter
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

    parameters = np.load('/mirror/scratch/pblack/temp/parameters.npy')
    profiles = np.load('/mirror/scratch/pblack/bp_profiles/profiles.npy')
    if parameters[1] == 'True':
        flatten = True
    else:
        flatten = False
    np.save('/mirror/scratch/pblack/temp/flatten.npy', flatten)   

    # A1P1 - P(l,pi)
    # A1P2 - P(l,0)
    # A2P1 - P(r,pi)
    # A2P2 - P(r,0)
    #power_ratios = [1.03,1.03,1,1] #Peter specified corrections - 21/12/2022
    power_ratios = [1,1,1,1] #no correction
    power_ratios = np.load('/mirror/scratch/pblack/bp_profiles/PRC.npy')
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

    quickload = np.load('/mirror/scratch/pblack/temp/quickload.npy')

    if quickload:
        array_time = np.load('/mirror/scratch/pblack/temp/time_array.npy') 
        array_time = Time(array_time,format='mjd',scale='utc',precision=9)
        array_time.format = 'iso'
        time2 = array_time.tt.datetime


    else:
        try:
            a1p1B = np.load ('/mirror/scratch/pblack/temp/a1p1_binned.npy')
            obsheader = np.load('/mirror/scratch/pblack/temp/obshdr.npy')
            MJD = Time(obsheader[0,8],format='mjd', scale='utc', precision=9)
            time_p1 = MJD + TimeDelta(a1p1B[:,0].astype(float), format='sec') 
            time_p1.format = 'iso'
            time2 = time_p1.tt.datetime
        except:   #possibly a nan in the a1p1 array causing a problem, so try a1p2 instead
            a1p1B = np.load ('/mirror/scratch/pblack/temp/a1p2_binned.npy')
            obsheader = np.load('/mirror/scratch/pblack/temp/obshdr.npy')
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

#------------------------------------------------------------------------------
#Power time plotting
def DualCWPowerPlot(duration_actual,save_it=False, first_loop=True):

    CW_Present = np.load('/mirror/scratch/pblack/temp/CW_Present.npy')

    parameters = np.load ('/mirror/scratch/pblack/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    if CW_Present[1]==True and CW_Present[0]==False:
        hornchoice = 'west'
    if CW_Present[0]==True and CW_Present[1]==False:
        hornchoice = 'east'
    if CW_Present[0]==False and CW_Present[1]==False:
        print('')
        print('\033[1;31m No CW Signal detected. Unable to plot. \033[1;32m')
    if CW_Present[0]==True and CW_Present[1]==True:
        hornchoice = 'both'

        
   # hornchoice = 'west' #force look at west horn, temporary 
    try:
        if hornchoice.lower() == 'west':
            a1p1_cw = np.load('/mirror/scratch/pblack/temp/a1p1_cw_binned.npy')
            a2p2_cw = np.load('/mirror/scratch/pblack/temp/a2p2_cw_binned.npy')

            if parameters[8] == 'True':
                a1p1_cw = a1p1_cw / np.mean(a1p1_cw)
                a2p2_cw = a2p2_cw / np.mean(a2p2_cw)
                a1p1_cw,a2p2_cw,a1p1_cw,a2p2_cw = corrections(a1p1_cw,a2p2_cw,a1p1_cw,a2p2_cw)
       #     a1p2_cw = np.load('/mirror/scratch/pblack/temp/a1p2_cw.npy')#111111111
        #    a2p1_cw = np.load('/mirror/scratch/pblack/temp/a2p1_cw.npy')#111111111



        elif hornchoice.lower() =='east':
            a1p2_cw = np.load('/mirror/scratch/pblack/temp/a1p2_cw_binned.npy')
            a2p1_cw = np.load('/mirror/scratch/pblack/temp/a2p1_cw_binned.npy')
            a1p2_cw,a1p2_cw,a2p1_cw,a2p1_cw = corrections(a1p2_cw,a1p2_cw,a2p1_cw,a2p1_cw)

            if parameters[8] == 'True':
                a1p2_cw = a1p2_cw / np.mean(a1p2_cw)
                a2p1_cw = a2p1_cw / np.mean(a2p1_cw)

        elif hornchoice.lower() =='both':
            a1p2_cw = np.load('/mirror/scratch/pblack/temp/a1p2_cw_binned.npy')
            a2p1_cw = np.load('/mirror/scratch/pblack/temp/a2p1_cw_binned.npy')
            a1p1_cw = np.load('/mirror/scratch/pblack/temp/a1p1_cw_binned.npy')
            a2p2_cw = np.load('/mirror/scratch/pblack/temp/a2p2_cw_binned.npy')

            if parameters[8] == 'True':
                a1p2_cw = a1p2_cw / np.mean(a1p2_cw)
                a2p1_cw = a2p1_cw / np.mean(a2p1_cw)
    

        time2 = time_series()

    
        user_inputs = np.load('/mirror/scratch/pblack/temp/inputs.npy')
        

        frequency = np.load('/mirror/scratch/pblack/temp/freq.npy')
        frequency = np.around(frequency, decimals=2)
    
        titlestring = 'Power Time Series, CW Signal'
    
        if save_it:
            fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        else:
            fig, ax = plt.subplots()
    
    
        if hornchoice.lower() == 'west':  
            ax.plot(time2, a1p1_cw, color = 'b', 
                    label = 'P(L,$\pi$) W ')
    
            ax.plot(time2, a2p2_cw, color = 'c', 
                 label = 'P(R,0) W')

#=---------

        elif hornchoice.lower() == 'east': 
            ax.plot(time2, a1p2_cw, color = 'r', 
                 label = 'P(L,0) E')
    
            ax.plot(time2, a2p1_cw, color = 'm', 
                 label = 'P(R,$\pi$) E')

        elif hornchoice.lower() == 'both':
            ax.plot(time2, a1p1_cw, color = 'b', 
                    label = 'P(L,$\pi$) W')
    
            ax.plot(time2, a2p2_cw, color = 'c', 
                 label = 'P(R,0) W')
            ax.plot(time2, a1p2_cw, color = 'r', 
                 label = 'P(L,0) E')
            ax.plot(time2, a2p1_cw, color = 'm', 
                 label = 'P(R,$\pi$) E')

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
            if save_it:
                plt.legend(bbox_to_anchor=(1.04,1), loc="lower left")
            else:
                plt.legend(loc="upper right", fontsize=11)
            ax.margins(x=0)
            plt.title(titlestring)
            fig.tight_layout()
    
        if save_it:
            plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_Dual_CW_Powers.png', bbox_inches="tight")
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
                DualCWPowerPlot(duration_actual, hornchoice, save_it, first_loop)
                print('\033[1;32m')
            else:
                print('')
                print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
                pass 

    except:
    #except Exception as e: print(repr(e))
        pass
    
#DualCWPowerPlot(duration_actual, 'west')
#------------------------------------------------------------------------------
#Power Spectrum Plotting

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

def DualPowerSpectrum(save_it=False, first_loop=True):

    CW_Present = np.load('/mirror/scratch/pblack/temp/CW_Present.npy')
    parameters = np.load ('/mirror/scratch/pblack/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    
    if CW_Present[1]:
        hornchoice = 'west'
    if CW_Present[0]:
        hornchoice = 'east'
    if CW_Present[0]==False and CW_Present[1]==False:
        print('')
        print('\033[1;31m No CW Signal detected. Unable to plot. \033[1;32m')

    hornchoice = 'west'

    try:
        user_inputs = np.load('/mirror/scratch/pblack/temp/inputs.npy')
        if hornchoice.lower() == 'west':

            a1p1b = np.load('/mirror/scratch/pblack/temp/a1p1_binned.npy')
            sampletime1 = getSamplingTime(a1p1b[:,0])

            a2p2b = np.load('/mirror/scratch/pblack/temp/a2p2_binned.npy')
            sampletime2 = getSamplingTime(a2p2b[:,0])
            del a1p1b, a2p2b

            a1p1_cw = np.load('/mirror/scratch/pblack/temp/a1p1_cw_binned.npy')
            numberofsamples1 = np.size(a1p1_cw)
            a2p2_cw = np.load('/mirror/scratch/pblack/temp/a2p2_cw_binned.npy')
            numberofsamples2 = np.size(a2p2_cw)

            sig_fft1 = rfft(a1p1_cw)
            power1 = sig_fft1 * np.conjugate(sig_fft1)
            freq1 = rfftfreq(numberofsamples1, sampletime1)
            sig_fft2 = rfft(a2p2_cw)
            power2 = sig_fft2 * np.conjugate(sig_fft2)
            freq2 = rfftfreq(numberofsamples2, sampletime2)
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

            titlestring1 = 'P(L,$\pi$) W'
            titlestring2 = 'P(R,0) W'
    
        elif hornchoice.lower() == 'east':
            a1p2b = np.load('/mirror/scratch/pblack/temp/a1p2_binned.npy')
            sampletime1 = getSamplingTime(a1p2b[:,0])
    
            a2p1b = np.load('/mirror/scratch/pblack/temp/a2p1_binned.npy')
            sampletime2 = getSamplingTime(a2p1b[:,0])
            del a1p2b, a2p1b
    
            a1p2_cw = np.load('/mirror/scratch/pblack/temp/a1p2_cw_binned.npy')
            numberofsamples1 = np.size(a1p2_cw)
    
            a2p1_cw = np.load('/mirror/scratch/pblack/temp/a2p1_cw_binned.npy')
            numberofsamples2 = np.size(a2p1_cw)

            sig_fft1 = rfft(a1p2_cw)
            power1 = sig_fft1 * np.conjugate(sig_fft1)
            freq1 = rfftfreq(numberofsamples1, sampletime1)
            sig_fft2 = rfft(a2p1_cw)
            power2 = sig_fft2 * np.conjugate(sig_fft2)
            freq2 = rfftfreq(numberofsamples2, sampletime2)
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
    
            titlestring1 = 'P(L,0) E'
            titlestring2 = 'P(R,$\pi$) E'
    
        plt.figure(1)
        if save_it:
            plt.figure(figsize=(12, 8), dpi=300)
    
        if hornchoice == 'west':
            plt.plot(bins, amps, c='b', label=titlestring1)
            plt.plot(bins2, amps2, c='c', label=titlestring2)
        elif hornchoice == 'east':
            plt.plot(bins, amps, c='r', label=titlestring1)
            plt.plot(bins2, amps2, c='m', label=titlestring2)
        plt.title('Power Spectra, CW Signal')
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel('Fourier Amplitude')
        plt.axis('scaled')
        plt.margins(x=0)
        plt.legend()
        plt.xlabel('Frequency [Hz]')

        plt.grid(c='darkgrey', which='major')
        plt.grid(c='gainsboro', which='minor')

        plt.tight_layout()
    
        if save_it:
            plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_Dual_PowerSpectrums.png', bbox_inches="tight")
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
                DualPowerSpectrum(duration_actual, save_it, first_loop)
                print('\033[1;32m')
            else:
                print('')
                print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
                pass 


    
    except:
      
        pass
    
    return



#~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#load background files and data tables
user_inputs = np.load('/mirror/scratch/pblack/temp/inputs.npy')
frequency = np.load('/mirror/scratch/pblack/temp/freq.npy') #its in hertz

#-------------------------------------------------------------------------
#(date, user_hour, user_minute, user_sample, user_duration)
#file_table(fits_name, begs, ends, start_times, end_times, samFIRSTs, samLASTs, multi_day, same_day, same_run, corrupts)
# the frequency channels within the digital bandpass are 112 to 568
#---------------------------------------------------------------------


def xaxis(range_example, duration_actual): #range example being a1p1B etc

    binnable = np.load('/mirror/scratch/pblack/temp/rebinnable.npy')
    if binnable:
        range_example = np.load('/mirror/scratch/pblack/temp/one_wire.npy') 
    freq = np.load('/mirror/scratch/pblack/temp/freq.npy')
    obsheader = np.load('/mirror/scratch/pblack/temp/obshdr.npy')
    MJD = Time(obsheader[0,8],format='mjd', scale='utc', precision=4)

    sample_to_datetime = np.size(range_example[:,0])
    time_p1 = MJD + TimeDelta(range_example[:,0].astype(float), format='sec') 
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
    
####################################################################

def get_minmax(band):

    ind = np.argmin(band[:,112:569])
    indx = np.argmax(band[:,112:569])
    band = band[:,112:569].flatten()
    bandmin = band[ind] 
    bandmax = band[indx]   

    return bandmin, bandmax

#--------------------------------------------------------------------

def CW_waterfall(save_it=False, first_loop=True):

    flatten = np.load('/mirror/scratch/pblack/temp/flatten.npy')
    p11 = np.load ('/mirror/scratch/pblack/temp/a1p1_power.npy')
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    del p11

    cmap = np.load('/mirror/scratch/pblack/temp/cmap.npy')
    if cmap:
        cstr='viridis'
    else:
        #cstr='gist_heat' 
        cstr='turbo' 

    a2p1_Index = np.load('/mirror/scratch/pblack/temp/a2p1_cw_binNO.npy',allow_pickle=True)
    a1p1_Index = np.load('/mirror/scratch/pblack/temp/a1p1_cw_binNO.npy',allow_pickle=True)
    a1p2_Index = np.load('/mirror/scratch/pblack/temp/a1p2_cw_binNO.npy',allow_pickle=True)
    a2p2_Index = np.load('/mirror/scratch/pblack/temp/a2p2_cw_binNO.npy',allow_pickle=True)

    a1p1_Index = int(np.mean(a1p1_Index))
    a1p2_Index = int(np.mean(a1p2_Index))
    a2p1_Index = int(np.mean(a2p1_Index))
    a2p2_Index = int(np.mean(a2p2_Index))


    user_inputs = np.load('/mirror/scratch/pblack/temp/inputs.npy')

    a1p1B = np.load ('/mirror/scratch/pblack/temp/a1p1_binned.npy')
    mins,first_tick, sample_to_datetime, frequency = yaxis(a1p1B, duration_actual)
    del a1p1B

    frequency = np.round(frequency, decimals=2)

    parameters = np.load ('/mirror/scratch/pblack/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False


    if parameters[8]=='True' or parameters[1]=='True':
        if parameters[1]=='True' and parameters[8]=='True':
            title_string = 'Spectrogram, CW Signal, All Inputs\n'+user_inputs[0]+' Global Normalised, Normalised'
        elif parameters[1]=='True':
            title_string =  'Spectrogram, CW Signal, All Inputs\n'+user_inputs[0]+' Global Normalised'
        else:  
            title_string = 'Spectrogram, CW Signal, All Inputs\n'+user_inputs[0]+' Normalised'

    else:
        title_string = 'Spectrogram, CW Signal, All Inputs\n'+user_inputs[0]

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
     

   # plt.setp(axarr, xticks=[0,1,2,3,4,5,6,7,8,9,10], xticklabels=[str(frequency[Ind-5]),'','','','',str(frequency[Ind]),'','','','',str(frequency[Ind+5])])
    plt.setp(axarr, xlabel='Frequency / MHz', ylabel='Time')



    a1p1_cw = np.load('/mirror/scratch/pblack/temp/a1p1_bandpass.npy',allow_pickle=True)
    a1p2_cw = np.load('/mirror/scratch/pblack/temp/a1p2_bandpass.npy',allow_pickle=True)
    a2p1_cw = np.load('/mirror/scratch/pblack/temp/a2p1_bandpass.npy',allow_pickle=True)
    a2p2_cw = np.load('/mirror/scratch/pblack/temp/a2p2_bandpass.npy',allow_pickle=True) 

    a1p1_cw, a1p2_cw,a2p1_cw,a2p2_cw = corrections(a1p1_cw,a1p2_cw,a2p1_cw,a2p2_cw)

    if parameters[8] == 'True':
        a1p1_cw = a1p1_cw / np.mean(a1p1_cw)
        a2p2_cw = a2p2_cw / np.mean(a2p2_cw)
        a1p2_cw = a1p2_cw / np.mean(a1p2_cw)
        a2p1_cw = a2p1_cw / np.mean(a2p1_cw)

    axarr[0,1].imshow(a1p1_cw[:,(a1p1_Index-10):(a1p1_Index+11)], cmap=cstr,aspect='auto', interpolation='none')#112:569  WAS -5 and +6
    d1 = axarr[0,1].imshow(a1p1_cw[:,(a1p1_Index-10):(a1p1_Index+11)], cmap=cstr, aspect='auto', interpolation='none')
    axarr[0,1].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    axarr[0,1].set_xticklabels([str(frequency[a1p1_Index-10]),'','','','','','','','',
       '',str(frequency[a1p1_Index]),'','','','','','','','','',str(frequency[a1p1_Index+11])])
    axarr[0,1].set_title('P(L,$\pi$) W')
    f.colorbar(d1, ax=axarr[0, 1])
       

    axarr[0,0].imshow(a1p2_cw[:,(a1p2_Index-10):(a1p2_Index+11)], cmap=cstr, aspect='auto', interpolation='none')
    d2 = axarr[0,0].imshow(a1p2_cw[:,(a1p2_Index-10):(a1p2_Index+11)], cmap=cstr,aspect='auto', interpolation='none')
    axarr[0,0].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    axarr[0,0].set_xticklabels([str(frequency[a1p2_Index-10]),'','','','','','','','','',str(frequency[a1p2_Index]),
        '','','','','','','','','',str(frequency[a1p2_Index+11])])
    axarr[0,0].set_title('P(L,0) E')
    f.colorbar(d2, ax=axarr[0, 0])
    

    axarr[1,0].imshow(a2p1_cw[:,(a2p1_Index-10):(a2p1_Index+11)], cmap=cstr,aspect='auto', interpolation='none')
    d3 = axarr[1,0].imshow(a2p1_cw[:,(a2p1_Index-10):(a2p1_Index+11)], cmap=cstr,aspect='auto', interpolation='none')
    axarr[1,0].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    axarr[1,0].set_xticklabels([str(frequency[a2p1_Index-10]),'','','','','','','','','',str(frequency[a2p1_Index]),
        '','','','','','','','','',str(frequency[a2p1_Index+11])])
    axarr[1,0].set_title('P(R,$\pi$) E')
    f.colorbar(d3, ax=axarr[1, 0])

    axarr[1,1].imshow(a2p2_cw[:,(a2p2_Index-10):(a2p2_Index+11)], cmap=cstr,aspect='auto', interpolation='none')
    d4 = axarr[1,1].imshow(a2p2_cw[:,(a2p2_Index-10):(a2p2_Index+11)], cmap=cstr,aspect='auto', interpolation='none')
    axarr[1,1].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    axarr[1,1].set_xticklabels([str(frequency[a2p2_Index-10]),'','','','','','','','',''
          ,str(frequency[a2p2_Index]),'','','','','','','','','',str(frequency[a2p2_Index+11])])
    axarr[1,1].set_title('P(R,0) W')
    f.colorbar(d4, ax=axarr[1, 1])

    plt.suptitle(title_string)
    plt.tight_layout()
 
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_4xCW_WF.png', bbox_inches="tight")
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

#-----------------------------------------------------------------------

def yaxis(range_example, duration_actual): #range example being a1p1B etc


    binnable = np.load('/mirror/scratch/pblack/temp/rebinnable.npy')
    if binnable:
        range_example = np.load('/mirror/scratch/pblack/temp/one_wire.npy') 
    freq = np.load('/mirror/scratch/pblack/temp/freq.npy')
    obsheader = np.load('/mirror/scratch/pblack/temp/obshdr.npy')
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

    quickload = np.load('/mirror/scratch/pblack/temp/quickload.npy')

    if quickload:
        time_p1 = np.load('/mirror/scratch/pblack/temp/time_array.npy') 
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




def DualCWFrequencyPlot(duration_actual,save_it=False, first_loop=True):

    CW_Present = np.load('/mirror/scratch/pblack/temp/CW_Present.npy')

    parameters = np.load ('/mirror/scratch/pblack/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    if CW_Present[1]==True and CW_Present[0]==False:
        hornchoice = 'west'
    if CW_Present[0]==True and CW_Present[1]==False:
        hornchoice = 'east'
    if CW_Present[0]==False and CW_Present[1]==False:
        print('')
        print('\033[1;31m No CW Signal detected. Unable to plot. \033[1;32m')
    if CW_Present[0]==True and CW_Present[1]==True:
        hornchoice = 'both'

        
   # hornchoice = 'west' #force look at west horn, temporary 
    try:
        if hornchoice.lower() == 'west':
            a1p1_cw = np.load('/mirror/scratch/pblack/temp/a1p1_cw_frequency.npy', allow_pickle=True)
            a2p2_cw = np.load('/mirror/scratch/pblack/temp/a2p2_cw_frequency.npy', allow_pickle=True)


        elif hornchoice.lower() =='east':
            a1p2_cw = np.load('/mirror/scratch/pblack/temp/a1p2_cw_frequency.npy', allow_pickle=True)
            a2p1_cw = np.load('/mirror/scratch/pblack/temp/a2p1_cw_frequency.npy', allow_pickle=True)


        elif hornchoice.lower() =='both':
            a1p2_cw = np.load('/mirror/scratch/pblack/temp/a1p2_cw_frequency.npy', allow_pickle=True)
            a2p1_cw = np.load('/mirror/scratch/pblack/temp/a2p1_cw_frequency.npy', allow_pickle=True)
            a1p1_cw = np.load('/mirror/scratch/pblack/temp/a1p1_cw_frequency.npy', allow_pickle=True)
            a2p2_cw = np.load('/mirror/scratch/pblack/temp/a2p2_cw_frequency.npy', allow_pickle=True)

    

        time2 = time_series()

    
        user_inputs = np.load('/mirror/scratch/pblack/temp/inputs.npy')
        

        frequency = np.load('/mirror/scratch/pblack/temp/freq.npy')
        frequency = np.around(frequency, decimals=2)
    
        titlestring = 'Frequency Time Series, CW Signal'
    
        if save_it:
            fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        else:
            fig, ax = plt.subplots()
    
    
        if hornchoice.lower() == 'west':  
            ax.plot(time2, a1p1_cw, color = 'b', 
                    label = 'P(L,$\pi$) W ')
    
            ax.plot(time2, a2p2_cw, color = 'c', 
                 label = 'P(R,0) W')

#=---------

        elif hornchoice.lower() == 'east': 
            ax.plot(time2, a1p2_cw, color = 'r', 
                 label = 'P(L,0) E')
    
            ax.plot(time2, a2p1_cw, color = 'm', 
                 label = 'P(R,$\pi$) E')

        elif hornchoice.lower() == 'both':
            ax.plot(time2, a1p1_cw, color = 'b', 
                    label = 'P(L,$\pi$) W')
    
            ax.plot(time2, a2p2_cw, color = 'c', 
                 label = 'P(R,0) W')
            ax.plot(time2, a1p2_cw, color = 'r', 
                 label = 'P(L,0) E')
            ax.plot(time2, a2p1_cw, color = 'm', 
                 label = 'P(R,$\pi$) E')

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

 
        ax.set(ylabel='Frequency / MHz')

       
        if save_it:
            plt.legend(bbox_to_anchor=(1.04,1), loc="lower left")
        else:
            plt.legend(loc="upper right", fontsize=11)
        ax.margins(x=0)
        plt.title(titlestring)
        fig.tight_layout()
    
        if save_it:
            plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_Dual_CW_Powers.png', bbox_inches="tight")
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
                DualCWPowerPlot(duration_actual, hornchoice, save_it, first_loop)
                print('\033[1;32m')
            else:
                print('')
                print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
                pass 

    #except:
    except Exception as e: print(repr(e))
     #   pass

#-----------------------------------------------------------------------




def CW_Menu(duration_actual):
    print('')
    raw_samples = np.load('/mirror/scratch/pblack/temp/raw_samples.npy')
    CW_Present = np.load('/mirror/scratch/pblack/temp/CW_Present.npy')

    looper = True
   
    print ('')
    print ('   -------------------------------------')
    print ('   >>>        CW PLOTS MENU          <<<')
    print ('   -------------------------------------')
    print ('')
    print ('   1 - CW Powers')
    print ('')
    if raw_samples:
        print ('   2 - CW Power Spectrum')
    else:
        print (' \033[1;90m  2 - CW Power Spectrum \033[1;32m')
    print ('')
    if raw_samples or duration_actual > 24:
        print(' \033[1;90m  3 - CW Waterfall \033[1;32m')
    else:
        print ('   3 - CW Waterfall')
    print ('')
    print ('   4 - CW Frequencies')
    print ('')
    print ('')
    print ('   5 - Launch CW & Power Meter Analysis Program')
    print ('')
    print ('')
    print ('   0 - Return to Quick-look menu')
    print ('')
    
    choice = input('Select menu option (number): \033[0;m')
    print('\033[1;32m')
    if choice.isdigit():
        if int(choice) ==1:
            DualCWPowerPlot(duration_actual) #hornchoice

        elif int(choice) ==2:
            DualPowerSpectrum() #hornchoice

        elif int(choice) == 3:
            CW_waterfall()

        elif int(choice) ==5:
            try:
                os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/LBASSPowerMeterAnalyser.py')
            except:
                pass

        elif int(choice) == 4:
            DualCWFrequencyPlot(duration_actual)

        elif int(choice) == 0: 
            looper = False
            pass

        else:
            print('\033[1;31m No such option. Please try again.\033[1;32m')

        return looper


#########################################################################

duration_actual = np.load('/mirror/scratch/pblack/temp/duration_actual.npy')

print ('\033[1;32m ')

CWmenu=True
while CWmenu:
    CWmenu = CW_Menu(duration_actual)


os.system('chmod -R -f 0777 /mirror/scratch/pblack || true')


