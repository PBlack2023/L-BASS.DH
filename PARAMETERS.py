#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:48:39 2022

@author: gibbsphillip
"""

DATA_PATH = '/mirror/scratch/pblack'

import math
import scipy
import numpy as np
import astropy.io
from astropy.io import fits
from astropy.time import Time 
from astropy.time import TimeDelta
import datetime
from datetime import timedelta
import time
from tqdm import tqdm #progress bars
import glob
import os
import gc
from math import nan

#--------------------------------------------------------------------

b = True

while b:

    profiles = np.load(DATA_PATH+'/bp_profiles/profiles.npy', allow_pickle = True)
    PRC = np.load(DATA_PATH+'/bp_profiles/PRC.npy')
    parameters = np.load(DATA_PATH+'/temp/parameters.npy', allow_pickle = True)
   # parameters = ['True','True','True',1420.405,20,'True','True','True','True','True','Peter','True']
#these are [0 temp binning, 1 bandpass flatten, 2 mask HI, 3 HI bin, 4 HI mask width, 
#  5 300DPI plots, 6 HI line markers, 7 Which PC, 8 Normalise Plots axes, 9 apply PS model ]
#  ,

    if parameters[0]=='True':
        binning = '\033[0;m Centered on One-Wire data points, 60s \033[1;32m'
    else:
        binning = '\033[0;m Centred mid-minute, 60s \033[1;32m' 

    if parameters[1]=='True':
        flatten = '\033[0;m Enabled \033[1;32m '
    else:
        flatten = '\033[0;m Disabled \033[1;32m'

    if parameters[2]=='True':
        HI_MASK = '\033[0;m Enabled \033[1;32m'
    else:
        HI_MASK = '\033[0;m Disabled \033[1;32m'

    HI_CENTRE = parameters[3]
    HI_MASK_WIDTH = parameters[4] 

    if parameters[5]=='True':
        HIdpi = '\033[0;m Enabled \033[1;32m'
    else:
        HIdpi = '\033[0;m Disabled \033[1;32m'

    if parameters[6]=='True':
        HI_MARK = '\033[0;m Enabled \033[1;32m'
    else:
        HI_MARK = '\033[0;m Disabled \033[1;32m'

    if parameters[7]=='True':
        WHICHPC = '\033[0;m Moonhut \033[1;32m'
    else:
        WHICHPC = '\033[0;m Buzzard \033[1;32m'

    if parameters[8]=='True':
        NORMALISE = '\033[0;m Enabled \033[1;32m'
    else:
        NORMALISE = '\033[0;m Disabled \033[1;32m'

    if parameters[9]=='True':
        PSmodel = '\033[0;m Enabled \033[1;32m'
    else:
        PSmodel = '\033[0;m Disabled \033[1;32m'

    for i in range (0, np.size(profiles[:,0])):
        try:
            if profiles[i,8] == 'True':
                profile_name = profiles[i,0]
            else:
                pass
        except:
            pass

    cmap = np.load(DATA_PATH+'/temp/cmap.npy')
    if cmap:
        cmapstr = '\033[0;m Viridis \033[1;32m'
    else:
      #  cmapstr = '\033[0;m Gist Heat \033[1;32m'
        cmapstr = '\033[0;m Turbo \033[1;32m'

    y_limits = np.load(DATA_PATH+'/temp/ylims.npy')
    if y_limits:
        y_lims = 'Enabled'
    else:
        y_lims = 'Disabled'
 
 
    print ('')
    print ('   -------------------------------------')
    print ('   >>             SETTINGS            <<')
    print ('   -------------------------------------')
    print ('')
    print ('   1  - Radiometer Data Binning:    ',binning)
    print ('   2  - Global Normalisation:       ', flatten)
    print ('   3  - Global Norms Profile:       ', '\033[0;m',profile_name,'\033[1;32m')
    print ('   4  - Power Correction Factors:    \033[0;m',PRC,'\033[1;32m')
    print ('   5  - Hydrogen Line Masking:      ',HI_MASK)
    print ('   6  - Hydrogen Line Mask Width:    \033[0;m',HI_MASK_WIDTH,'bins \033[1;32m')
    print ('   7  - Hydrogen Line Markers:      ',HI_MARK)
    print ('   8  - Offer 300dpi saved plots:   ', HIdpi)
    print ('   9  - Waterfall Colormap:         ',cmapstr)
    print ('   10 - Offer custom y-axis limits:  \033[0;m',y_lims,'\033[1;32m')
    print ('   11 - Normalise Plots Axes:       ',NORMALISE)
    print ('   12 - Power Spectra Model:        ',PSmodel, ' (under development)')
    print ('   13 - Program running on:         ',WHICHPC)
    print ('')
    print ('   14 - Generate new bandpass flattening profile')
    print ('   15 - Solar Observation Fitting Routines')
    print ('')
    print ('   0 - Return to Main Menu')
    print ('')
    
    choice = input('Select menu option (number) to change setting: \033[0;m')
    print('\033[1;32m')
    if choice.isdigit():
        if int(choice) ==1:
            if parameters[0]=='True':
                parameters[0] = False
            else:
                parameters[0] = True
            np.save(DATA_PATH+'/temp/parameters.npy',parameters)

        elif int(choice) ==2:
            if parameters[1] == 'True':
                parameters[1] = False
            else:
                parameters[1] = True
            np.save(DATA_PATH+'/temp/parameters.npy',parameters)

        elif int(choice) == 3:
            #loop through available choices until user accepts one!
            print('')
            print('Available profiles will be shown until user accepts one.')
            print('')
            proloop = True
            for i in range (0,np.size(profiles[:,0])): #set status of all profiles to inactive
                try:
                    profiles[i,8] = False
                    np.save(DATA_PATH+'/bp_profiles/profiles.npy',profiles)
                except:
                    pass
            while proloop:
                for i in range (0,np.size(profiles[:,0])):
                    print ('')
                    print ('Profile name: ',profiles[i,0])
                    print ('Generated from data on ', profiles[i,1],' from ',profiles[i,2],' over ',profiles[i,3], ' hours.')
                    print('')
                    YorN = input('Do you wish to select this profile? (Y/N): ')
                    if YorN == 'Y' or YorN == 'y':
                       
                        parameters[10] = str(profiles[i,0])
                        profiles[i,8] = True  #mark as the active profile
                        np.save(DATA_PATH+'/bp_profiles/profiles.npy',profiles)
                        np.save(DATA_PATH+'/temp/parameters.npy',parameters)
                        proloop = False
                        break
                    else:
                        pass 

        elif int(choice) == 4:
            print('Please enter only integer or decimal values.\n A value must be entered for each input.')
            print('')
            a = float(input('\033[1;32m Enter the correction factor for P(L,\u03C0) W : \033[0;m'))
            b = float(input('\033[1;32m Enter the correction factor for P(L,0) E : \033[0;m'))
            c = float(input('\033[1;32m Enter the correction factor for P(R,\u03C0) E : \033[0;m'))
            d = float(input('\033[1;32m Enter the correction factor for P(R,0) W : \033[0;m'))
            tup = [a,b,c,d]
            np.save(DATA_PATH+'/bp_profiles/PRC.npy',tup)               

        elif int(choice) ==5:
            if parameters[2] == 'True':
                parameters[2] = False
            else:
                parameters[2] = True
            np.save(DATA_PATH+'/temp/parameters.npy',parameters)

        elif int(choice) ==6:
            print('')
            new_width = input('Please specify a new width in bins: ')
            if new_width.isdigit() and int(new_width) < 100 and int(new_width) > 1:
                parameters[4] = int(new_width)
            else:
                print('\033[1;31m Input not accepted, please try again \033[1;32m')
                print('')
            np.save(DATA_PATH+'/temp/parameters.npy',parameters)

        elif int(choice) ==7:
            if parameters[6] == 'True':
                parameters[6] = False
            else:
                parameters[6] = True
            np.save(DATA_PATH+'/temp/parameters.npy',parameters)

        elif int(choice) ==8:
            if parameters[5] == 'True':
                parameters[5] = False
            else:
                parameters[5] = True
            np.save(DATA_PATH+'/temp/parameters.npy',parameters)

        elif int(choice) ==9:
            if cmap:
                cmap = False
                np.save(DATA_PATH+'/temp/cmap.npy',cmap)
            else:
                cmap = True
                np.save(DATA_PATH+'/temp/cmap.npy',cmap) 


        elif int(choice) == 10:
            if y_limits:
                y_limits=False
                np.save(DATA_PATH+'/temp/ylims.npy',y_limits)
            else:
                y_limits=True
                np.save(DATA_PATH+'/temp/ylims.npy',y_limits)

        elif int(choice) ==11:
            if parameters[8] == 'True':
                parameters[8] = False
            else:
                parameters[8] = True
            np.save(DATA_PATH+'/temp/parameters.npy',parameters)

        elif int(choice) ==12:
            if parameters[9] == 'True':
                parameters[9] = False
            else:
                parameters[9] = True
            np.save(DATA_PATH+'/temp/parameters.npy',parameters)

        elif int(choice) ==13:
            if parameters[7] == 'True':
                parameters[7] = False
            else:
                parameters[7] = True
            np.save(DATA_PATH+'/temp/parameters.npy',parameters)


        elif int(choice)  == 14:
            print ('This feature is still under development & currently unavailable.')
            print ('This process assumes equivalent inputs in both channels. NCP-NCP or LOAD-LOAD.') 
            print ('System stability is also assumed - i.e. no largescale local temp variations.')    
            print ('Please ensure dates and times selected meet this criteria.')
            print('')
            parameters[1] = False
            np.save(DATA_PATH+'/temp/parameters.npy',parameters)
            exec(open("/local5/scratch/pblack/scripts/new_funcs.py").read())
            exec(open("/local5/scratch/pblack/scripts/breaky_tables.py").read())
            exec(open("/local5/scratch/pblack/scripts/global_norms_ratios.py").read())
        
        elif int(choice)  == 15:
            print ('This feature is still under development & currently unavailable.')
            exec(open("/local5/scratch/pblack/scripts/new_funcs.py").read())    
            exec(open("/local5/scratch/pblack/scripts/breaky_tables.py").read())
            exec(open("/local5/scratch/pblack/scripts/bandpass_guass_fit.py").read())

       
        elif int(choice) == 0:
            b = False
            pass
        



np.save(DATA_PATH+'/temp/parameters.npy',parameters)

os.system('chmod -R -f 0777 /local5/scratch/pblack || true')
    
