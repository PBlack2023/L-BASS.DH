#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:48:39 2022

@author: gibbsphillip
"""

DATA_PATH = '/mirror/scratch/pblack'


import numpy as np
import os
import gc


#--------------------------------------------------------------------

b = True

while b:

    parameters = np.load(DATA_PATH+'/temp/parameters.npy')
#parameters = ['True','True','True',1420.405,20,'True','True','True','True','True','True']
#these are [0 temp binning, 1 bandpass flatten, 2 mask HI, 3 HI bin, 4 HI mask width, 
#  5 300DPI plots, 6 HI line markers, 7 Which PC, 8 Normalise Plots axes, 9 apply PS model ]

    if parameters[0]=='True':
        binning = '\033[0;m Centered on One-Wire data points, 60s \033[1;32m'
    else:
        binning = '\033[0;m Arbitrary bins, 60s \033[1;32m' 

    if parameters[1]=='True':
        flatten = '\033[0;m Applying stored profile\033[1;32m '
    else:
        flatten = '\033[0;m None \033[1;32m'

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
    print ('')
    print ('   -------------------------------------')
    print ('   >>          PLOT SETTINGS          <<')
    print ('   -------------------------------------')
    print ('')
    print ('   1  - Hydrogen Line Masking:      ',HI_MASK)
    print ('   2  - Hydrogen Line Mask Width:    \033[0;m',HI_MASK_WIDTH,'bins \033[1;32m')
    print ('   3  - Hydrogen Line Markers:      ',HI_MARK)
    print ('   4  - Normalise Plots Axes:       ',NORMALISE)
    print ('   5  - Power Spectra Model:        ',PSmodel, ' (under development)')
    print ('   6  - Offer 300dpi saved plots:   ', HIdpi)
    print ('   7  - Waterfall Colormap:         ',cmapstr)
    print ('   8  - Offer custom y-axis limits:  \033[0;m',y_lims,'\033[1;32m')
    print ('')
    print ('')
    print ('')
    print ('   0 - Return to Quick-look Menu')
    print ('')
    
    choice = input('Select menu option (number) to change setting: \033[0;m')
    print('\033[1;32m')
    if choice.isdigit():
        if int(choice) ==1:
            if parameters[2]=='True':
                parameters[2] = False
            else:
                parameters[2] = True
            np.save(DATA_PATH+'/temp/parameters.npy',parameters)

        elif int(choice) ==2:
            print('')
            new_width = input('Please specify a new width in bins: ')
            if new_width.isdigit() and int(new_width) < 100 and int(new_width) > 1:
                parameters[4] = int(new_width)
            else:
                print('\033[1;31m Input not accepted, please try again \033[1;32m')
                print('')
            np.save(DATA_PATH+'/temp/parameters.npy',parameters)

        elif int(choice) ==3:
            if parameters[6] == 'True':
                parameters[6] = False
            else:
                parameters[6] = True
            np.save(DATA_PATH+'/temp/parameters.npy',parameters)

        elif int(choice) ==4:
            if parameters[8] == 'True':
                parameters[8] = False
            else:
                parameters[8] = True
            np.save(DATA_PATH+'/temp/parameters.npy',parameters)


        elif int(choice) ==5:
            if parameters[9] == 'True':
                parameters[9] = False
            else:
                parameters[9] = True
            np.save(DATA_PATH+'/temp/parameters.npy',parameters)

        elif int(choice) ==6:
            if parameters[5] == 'True':
                parameters[5] = False
            else:
                parameters[5] = True
            np.save(DATA_PATH+'/temp/parameters.npy',parameters)

        elif int(choice) ==7:
            if cmap:
                cmap = False
                np.save(DATA_PATH+'/temp/cmap.npy',cmap)
            else:
                cmap = True
                np.save(DATA_PATH+'/temp/cmap.npy',cmap) 


        elif int(choice) == 8:
            if y_limits:
                y_limits=False
                np.save(DATA_PATH+'/temp/ylims.npy',y_limits)
            else:
                y_limits=True
                np.save(DATA_PATH+'/temp/ylims.npy',y_limits)


        elif int(choice) == 0:
            b = False
            pass


        else:
            print('\033[1;31m No such option. Please try again.\033[1;32m')

np.save(DATA_PATH+'/temp/parameters.npy',parameters)

os.system('chmod -R -f 0777 /local5/scratch/pblack || true')
    
