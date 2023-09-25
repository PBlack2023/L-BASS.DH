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

def DualPowerSpectrum(save_it=False, first_loop=True):

    CW_Present = np.load(DATA_PATH+'/temp/CW_Present.npy')
    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
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

    try:
        user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
        if hornchoice.lower() == 'west':
            a1p1b = np.load(DATA_PATH+'/temp/a1p1_binned.npy')
            sampletime1 = getSamplingTime(a1p1b[:,0])

            a2p2b = np.load(DATA_PATH+'/temp/a2p2_binned.npy')
            sampletime2 = getSamplingTime(a2p2b[:,0])
            del a1p1b, a2p2b
    
            a1p1_cw = np.load(DATA_PATH+'/temp/a1p1_cw.npy')
            numberofsamples1 = np.size(a1p1_cw)
            a2p2_cw = np.load(DATA_PATH+'/temp/a1p1_cw.npy')
            numberofsamples2 = np.size(a2p2_cw)
    
            sig_fft1 = rfft(a1p1_cw)
            freq1 = rfftfreq(numberofsamples1, sampletime1)
            sig_fft2 = rfft(a2p2_cw)
            freq2 = rfftfreq(numberofsamples2, sampletime2)
    
            titlestring1 = 'P(L,$\pi$) W  CW Signal'
            titlestring2 = 'P(R,0) W  CW Signal'
    
        elif hornchoice.lower() == 'east':
            a1p2b = np.load(DATA_PATH+'/temp/a1p2_binned.npy')
            sampletime1 = getSamplingTime(a1p2b[:,0])
    
            a2p1b = np.load(DATA_PATH+'/temp/a2p1_binned.npy')
            sampletime2 = getSamplingTime(a2p1b[:,0])
            del a1p2b, a2p1b
    
            a1p2_cw = np.load(DATA_PATH+'/temp/a1p2_cw.npy')
            numberofsamples1 = np.size(a1p2_cw)
    
            a2p1_cw = np.load(DATA_PATH+'/temp/a2p1_cw.npy')
            numberofsamples2 = np.size(a2p1_cw)
    
            sig_fft1 = rfft(a1p2_cw)
            freq1 = rfftfreq(numberofsamples1, sampletime1)
            sig_fft2 = rfft(a2p1_cw)
            freq2 = rfftfreq(numberofsamples2, sampletime2)
    
            titlestring1 = 'P(L,0) E  CW Signal'
            titlestring2 = 'P(R,$\pi$) E  CW Signal'
    
        plt.figure(1)
        if save_it:
            plt.figure(figsize=(12, 8), dpi=300)
    
        plt.subplot(211)
        if hornchoice == 'west':
            plt.plot(freq1, np.abs(sig_fft1), c='b')
        elif hornchoice == 'east':
            plt.plot(freq1, np.abs(sig_fft1), c='r')
        plt.title(titlestring1)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel('Fourier Amplitude')
        plt.margins(x=0)
    
        plt.subplot(212)
        if hornchoice == 'west':
            plt.plot(freq2, np.abs(sig_fft2), c='c')
        elif hornchoice == 'east':
            plt.plot(freq2, np.abs(sig_fft2), c='m')
        plt.title(titlestring2)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel('Fourier Amplitude')
        plt.margins(x=0)
        plt.xlabel('Frequency [Hz]')
        plt.tight_layout()
    
        if save_it:
            plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_Dual_PowerSpectrums.png', bbox_inches="tight")
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
                DualPowerSpectrum(duration_actual, save_it, first_loop)
 
            else:
                print('')
                print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
                pass 
    
    except:
        pass
    
    return

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

def AltAzToGalactic(azimuth, elevation, time):
    """
    Converts Input of time of observation and elevation with known location of
    telescope into Galactic Coordinates. Time in UTC

    """
    AltAzCoord = AltAz(location = LOC, obstime=Time(time), 
                  az = azimuth*u.deg, alt = elevation*u.deg)
    
    AltAzCoord = SkyCoord(AltAzCoord)
    
    #ICRSCoord = AltAzCoord.transform_to('icrs')
    
    GalCoord = AltAzCoord.transform_to('galactic')
    
    return GalCoord

#±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±

def GalacticPathPlotter(duration_actual):
    """
    Original code by Jordan Norris, further adapted by Phillip Black

    Code plots and also generates an array of galactic coordinates in 
    an array of skycoord objects and also as a 2D array of longitue and 
    lattitude which corresponds to each time.

    Parameters
    ----------
    duration_actual : float
        length of duration of data set

    Returns
    -------
    GalCoord : array
        Array of SkyCoord type objects corresponding to where LBass is looking
        at each time.
        
    galCoordArray2D: array
        2D array of longitude and lattidue corresponding to each time.

    """
    a1p1b = np.load(DATA_PATH+'/temp/a1p1_binned.npy')
    UTC_array = UTCDateTime(a1p1b, duration_actual)

    obshdr = np.load(DATA_PATH+'/temp/obshdr.npy')

    #obshdr[2] is the observing mode - there are a number of options that eventually need to be accomodated.
    if obshdr[0,2] == 'NCP-SCANNING':
        scan_elevation = np.load (DATA_PATH+'/temp/a1p2_binned.npy')
        horn = 'East'
        althorn = 'West'
    elif obshdr[0,2] == 'SCANNING-NCP':
        scan_elevation = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
        horn = 'West'
        althorn = 'East'
    else:
        scan_elevation = np.load (DATA_PATH+'/temp/a1p2_binned.npy')
        horn = 'East'
        althorn = 'West'
  
      #LBASS records more than 90° elevation. To convert for astopy needs corrected azimuth & elevation.
    if float(scan_elevation[0,2]) > 90.0: 
        rotated_frame = float(obshdr[0,6]) - 180 #'rotate' the frame to north facing
        obaz = float(rotated_frame)
        scan_elevation[:,2] = np.where(scan_elevation[:,2] <90.0, scan_elevation[:,2], 180 - scan_elevation[:,2])
    else:
        alt = (scan_elevation[:,2])
        obaz = float(obshdr[0,6])
    NCP_elevation = np.where(scan_elevation[:,2] <0.1, scan_elevation[:,2], 53.24)
    NCP_azimuth = float(obshdr[0,6]) - 180
   
    AltAzCoord = AltAz(location = LOC, obstime=Time(UTC_array), az = obaz*u.deg, alt = scan_elevation[:,2]*u.deg)
    AltAzCoord = SkyCoord(AltAzCoord)
    GalCoord = AltAzCoord.transform_to('galactic')

    #AltAzCoordUPPER = AltAz(location = LOC, obstime=Time(UTC_array), az = obaz*u.deg, alt = (scan_elevation[:,2]+10)*u.deg)
    #AltAzCoordUPPER = SkyCoord(AltAzCoordUPPER)
    #GalCoordUPPER = AltAzCoordUPPER.transform_to('galactic')
    #AltAzCoordLOWER = AltAz(location = LOC, obstime=Time(UTC_array), az = obaz*u.deg, alt = (scan_elevation[:,2]-20)*u.deg)
    #AltAzCoordLOWER = SkyCoord(AltAzCoordLOWER)
    #GalCoordLOWER = AltAzCoordLOWER.transform_to('galactic')

    AltAzCoord2 = AltAz(location = LOC, obstime=Time(UTC_array),az = NCP_azimuth*u.deg, alt = (NCP_elevation*u.deg))
    AltAzCoord2 = SkyCoord(AltAzCoord2)
    GalCoord2 = AltAzCoord2.transform_to('galactic')
    
    #galCoordArray2D = np.zeros((0,2))
    #for i in range(0, np.size(UTC_array)):
     #   galCoord = AltAzToGalactic(0,(scan_elevation[i,2]), UTC_array[i])
      #  l = galCoord.l.degree #long.
       # b = galCoord.b.degree #lat.
        #line = np.array([l, b])
       # galCoordArray2D = np.vstack((galCoordArray2D,line))

    plt.figure()
    plt.subplot(projection = "aitoff")
    plt.plot(GalCoord.l.wrap_at('180d').radian, GalCoord.b.radian, c='r', label='Scanning: '+horn)
    plt.plot(GalCoord2.l.wrap_at('180d').radian, GalCoord2.b.radian, marker='o', c='blue', label='NCP: '+althorn)
   # plt.plot(GalCoordUPPER.l.wrap_at('180d').radian, GalCoordUPPER.b.radian, c='c', label='Beam')
    #plt.plot(GalCoordLOWER.l.wrap_at('180d').radian, GalCoordLOWER.b.radian, c='c')
    plt.grid(True)
    plt.legend()
    plt.title('Scanning Path of Telescope in Galactic Coordinates')
    plt.tight_layout()
    plt.show()


#~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#load background files and data tables
user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
frequency = np.load(DATA_PATH+'/temp/freq.npy') #its in hertz
#frequency = frequency * (10**-9)

    
#------------------------------------------------------------------------------

def Mapping_Menu(duration_actual):
    print('')

    obshdr = np.load(DATA_PATH+'/temp/obshdr.npy')
    looper = True
  
    print ('')
    print ('   -------------------------------------')
    print ('   >>>      MAPPING PLOTS MENU       <<<')
    if obshdr[0,2] == 'FIELD TEST' or obshdr[0,2] == 'LAB TEST':
        print ('\033[1;31m         **Field or Lab test** \033[1;32m')
    print ('   -------------------------------------')
    print ('')
    print ('   1 - Scanning Path Plot (under development)')
    print ('')
    print ('')
    print ('')
    print ('   0 - Return to Quick-look menu')
    print ('')
    
    choice = input('Select menu option (number): \033[0;m')
    print('\033[1;32m')
    if choice.isdigit():
        if int(choice) ==1:
            GalacticPathPlotter(duration_actual)

  
        elif int(choice) == 0:
            looper = False
            pass

        else:
            print('\033[1;31m No such option. Please try again.\033[1;32m')
            Mapping_Menu(duration_actual)

    return looper

#########################################################################

duration_actual = np.load(DATA_PATH+'/temp/duration_actual.npy')

temp_data = np.load(DATA_PATH+'/temp/file2.npy', allow_pickle=True)
temp_times = temp_data[:,0]
temp_data = []

print ('\033[1;32m ')

looper = True
while looper:
    looper = Mapping_Menu(duration_actual)


os.system('chmod -R -f 0777 /mirror/scratch/pblack || true')

