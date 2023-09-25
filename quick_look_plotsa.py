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


#~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#load background files and data tables
user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
frequency = np.load(DATA_PATH+'/temp/freq.npy') #its in hertz

#-------------------------------------------------------------------------
#(date, user_hour, user_minute, user_sample, user_duration)
#file_table(fits_name, begs, ends, start_times, end_times, samFIRSTs, samLASTs, multi_day, same_day, same_run, corrupts)
# the frequency channels within the digital bandpass are 112 to 568
#---------------------------------------------------------------------


def xaxis(range_example, duration_actual): #range example being a1p1B etc

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
    time_p2 = np.where(time_p2 != '00:00', time_p2, time_p1.strftime('%H:%M %d')) #at midnight add day number
    time_p1 = np.asarray(time_p1.strftime('%M'))
    time_p1 = np.where(time_p1 == '00') #these two lines find round hours
 
    mins = time_p2[time_p1].tolist() #values of hours for the tick labels
    first_tick = time_p1[0][:].tolist() #list of bin numbers for the hours tick labels
    
    del time_p1, time_p2, range_example

    return mins, first_tick, sample_to_datetime, freq
    
####################################################################

def QuickLookMenu(duration_actual):

    CW_Present = np.load(DATA_PATH+'/temp/CW_Present.npy')
    plotmenu=True

    print ('')
    print ('   -------------------------------------')
    print ('   >>        QUICK-LOOK MENU          <<')
    print ('   -------------------------------------')
    print ('')
    print ('   1 - Bandpass Plots')
    print ('   2 - Power Plots')
    if duration_actual > 24:
        print ('\033[1;90m   3 - Waterfall Plots \033[1;32m')
    else:
        print ('   3 - Waterfall Plots')
    print ('   4 - RMS Plots')
    print ('   5 - Temperature Plots')
    print ('\033[1;90m   6 - Power Spectra Plots \033[1;32m')
    print ('   7 - Mapping Plots')
    if CW_Present[0]==False and CW_Present[1]==False:
        print ('\033[1;90m   8 - CW & Power Meter Menu \033[1;32m') 
    else:
        print ('   8 - CW & Power Meter Menu')    
    print ('\033[1;90m   9 - Ad Hoc Menu \033[1;32m')
    print('')
    print ('\033[1;90m  10 - Fits Header & Comments \033[1;32m')
    print ('\033[1;90m  11 - Export Data \033[1;32m')
    print ('  12 - Plot Settings')
    print ('  13 - Passives Plots')
    print ('')
    print('   0 - Return to Main Menu')
    print('')
    choice = input('Select menu option (number): \033[0;m')
    print('\033[1;32m')
    if choice.isdigit():
        if int(choice) == 1: 
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/bandpass_plots.py')
    
        elif int(choice) ==2:
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/power_plots.py')
        
        elif int(choice) ==3:
            if duration_actual > 24:
                pass
            else:
                os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/waterfall_plots.py')
                
        elif int(choice)  == 4:
            os.system("/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/RMS_plots.py")

        elif int(choice) ==5:
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/onewire_plots.py')

        elif int(choice)==6:      
            pass

        elif int(choice)==7: 
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/mapping_plots.py') 

        elif int(choice)==8:

            if CW_Present[0]==False and CW_Present[1]==False:
                pass
            else:
                os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/cw_plots.py') 

        elif int(choice)==9:  
            pass

        elif int(choice)==10:
            pass

        elif int(choice) == 11:
            pass

        elif int(choice)  == 12:
            os.system("/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/PLOT_PARAMETERS_ONLY.py")

        elif int(choice) == 13:
            os.system("/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/passives_plots.py")

        elif int(choice) ==0:
            plotmenu=False

    else:
        print('\033[1;31m Invalid selection by user. Please try again. \033[1;32m')

    return plotmenu


def BinnedMenu(duration_actual):

    CW_Present = np.load(DATA_PATH+'/temp/CW_Present.npy')
    plotmenu=True

    print ('')
    print ('   -------------------------------------')
    print ('   >>           BINNED MENU           <<')
    print ('   -------------------------------------')
    print ('')
    print ('   1 - Bandpass Plots ')
    print ('   2 - Power Plots')
    if duration_actual > 24:
        print ('\033[1;90m   3 - Waterfall Plots \033[1;32m')
    else:
        print ('   3 - Waterfall Plots')
    print ('   4 - RMS Plots')
    print ('   5 - Temperature Plots')
    print ('\033[1;90m   6 - Power Spectra Plots \033[1;32m')
    print ('   7 - Mapping Plots')
    if CW_Present[0]==False and CW_Present[1]==False:
        print ('\033[1;90m   8 - CW & Power Meter Menu \033[1;32m') 
    else:
        print ('   8 - CW & Power Meter Menu')  
    print ('   9 - Ad Hoc Menu')
    print('')
    print ('   10 - Fits Header & Comments')
    print ('   11 - Export Data ')
    print ('   12 - Plot Settings')
    print ('   13 - Passives Plots')
    print ('')
    print('   0 - Return to Main Menu')
    print('')
    choice = input('Select menu option (number): \033[0;m')
    print('\033[1;32m')
    if choice.isdigit():
        if int(choice) == 1: 
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/bandpass_plots.py')
    
        elif int(choice) ==2:
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/power_plots.py')
        
        elif int(choice) ==3:
            if duration_actual > 24:
                pass
            else:
                os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/waterfall_plots.py')
                
        elif int(choice)  == 4:
            os.system("/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/RMS_plots.py")

        elif int(choice) ==5:
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/onewire_plots.py')

        elif int(choice)==6:      
            #os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/powerspectra_plots.py')
            pass

        elif int(choice)==7: 
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/mapping_plots.py') 

        elif int(choice)==8:

            if CW_Present[0]==False and CW_Present[1]==False:
                pass
            else:
                os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/cw_plots.py') 

        elif int(choice)==9:  
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/ad_hoc_plots.py') 

        elif int(choice)==10:
            Obs_Details() 

        elif int(choice) == 11:
            os.system("/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/export_smoothed.py")

        elif int(choice)  == 12:
            os.system("/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/PLOT_PARAMETERS_ONLY.py")

        elif int(choice) == 13:
            os.system("/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/passives_plots.py")

        elif int(choice) ==0:
            plotmenu=False

    else:
        print('\033[1;31m Invalid selection by user. Please try again. \033[1;32m')

    return plotmenu


def RawMenu(duration_actual):

    CW_Present = np.load(DATA_PATH+'/temp/CW_Present.npy')
    plotmenu=True
    print ('')
    print ('   -------------------------------------')
    print ('   >>          UNBINNED MENU          <<')
    print ('   -------------------------------------')
    print ('')
    print ('\033[1;90m   1 - Bandpass Plots \033[1;32m')
    print ('   2 - Power Plots')
    print ('\033[1;90m   3 - Waterfall Plots \033[1;32m')
    print ('   4 - RMS Plots')
    print ('   5 - Temperature Plots')
    print ('   6 - Power Spectra Plots ')
    print ('\033[1;90m   7 - Mapping Plots \033[1;32m')
    if CW_Present[0]==False and CW_Present[1]==False:
        print ('\033[1;90m   8 - CW & Power Meter Menu \033[1;32m') 
    else:
        print ('   8 - CW & Power Meter Menu')   
    print ('\033[1;90m   9 - Ad Hoc Menu \033[1;32m')
    print('')
    print ('   10 - Fits Header & Comments')
    print ('\033[1;90m   11 - Export Data \033[1;32m')
    print ('   12 - Plot Settings')
    print ('   13 - Passives Plots')
    print ('')
    print('   0 - Return to Main Menu')
    print('')
    choice = input('Select menu option (number): \033[0;m')
    print('\033[1;32m')
    if choice.isdigit():
        if int(choice) == 1: 
            pass
    
        elif int(choice) ==2:
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/power_plots.py')
        
        elif int(choice) ==3:
            pass
                
        elif int(choice)  == 4:
            os.system("/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/RMS_plots.py")

        elif int(choice) ==5:
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/onewire_plots.py')

        elif int(choice)==6:      
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/powerspectra_plots.py')

        elif int(choice)==7: 
            pass

        elif int(choice)==8:

            if CW_Present[0]==False and CW_Present[1]==False:
                pass
            else:
                os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/cw_plots.py') 

        elif int(choice)==9:  
            pass

        elif int(choice)==10:
            Obs_Details() 

        elif int(choice) == 11:
            pass

        elif int(choice)  == 12:
            os.system("/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/PLOT_PARAMETERS_ONLY.py")

        elif int(choice) == 13:
            os.system("/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/passives_plots.py")

        elif int(choice) ==0:
            plotmenu=False
           


    else:
        print('\033[1;31m Invalid selection by user. Please try again. \033[1;32m')

    return plotmenu


def PlotMenu(plotmenu, duration_actual):
    
    if os.path.exists(user_inputs[6]):
        pass
    else:
        os.system('mkdir '+user_inputs[6])
    
  #  if float(duration_actual) > 24:
  #      warnlen='\033[1;31m (dataset too large) \033[1;32m'
  #  elif float(duration_actual) < 0.25:
  #      warnlen='\033[1;31m (dataset too small) \033[1;32m'
  #  else:
  #      warnlen=''
    CWmenu=True
    gc.collect()

    quickload = np.load(DATA_PATH+'/temp/quickload.npy')
    raw_samples = np.load(DATA_PATH+'/temp/raw_samples.npy')


    if quickload:
        plotmenu = QuickLookMenu(duration_actual)
    elif raw_samples:
        plotmenu = RawMenu(duration_actual)
    else:
        plotmenu = BinnedMenu(duration_actual)


        
    return plotmenu, CWmenu

#–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––



def Obs_Details():

    obshdr = np.load(DATA_PATH+'/temp/obshdr.npy')
    a1p1B = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
    MJD = Time(obshdr[0,8],format='mjd', scale='utc', precision=9)
    comment = np.load(DATA_PATH+'/temp/comment.npy')
    
    print('')
    print('Operator          ',obshdr[0,1])
    print('Location          ',obshdr[0,0])
    print('Observing Mode    ',obshdr[0,2])
    print('Azimuth           ',obshdr[0,6])
    print('Elevation         ',a1p1B[0,2])
    print('Run began ref MJD ',MJD.iso)
    print('')
    for i in range (0,np.size(comment)):
        if i == 0:
            print(comment[i])
        elif comment[i] != 'end observer comments':
            print(comment[i])
        else:
            print(comment[i])
            print('')
            break

    del a1p1B

    input('Press enter to return to the Quick-look Menu')
    obshdr=[]
    a1p1B =[]

#########################################################################

duration_actual = np.load(DATA_PATH+'/temp/duration_actual.npy')

print ('\033[1;32m ')

plotmenu=True
while plotmenu:
    plotmenu, CWmenu = PlotMenu(plotmenu, duration_actual)


os.system('chmod -R -f 0777 /mirror/scratch/pblack || true')


