#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:16:10 2022

@author: pblack
"""

DATA_PATH = '/mirror/scratch/pblack'


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import math
import time
import scipy
import numpy as np
import astropy.io
from astropy.io import fits
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import SkyCoord
from astropy.time import Time 
from astropy.time import TimeDelta
import datetime
from tqdm import tqdm #progress bars
import glob
import os
from math import nan
import csv
from csv import DictReader
import pandas as pd
from matplotlib.pyplot import figure

os.chdir('/scratch/nas_lbass/raw_data/')

def LIVEboard(looper=True):
    
    temp_files = []
    temp_files = sorted(glob.glob('1W_Temp-*.csv'))

    df_from_each_file = (pd.read_csv(f) for f in temp_files)
    concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)
    #concatenated_df.to_csv(DATA_PATH+'/temp/temp_data.csv', index=False)

    temp_data = concatenated_df.to_numpy()


    jeff = plt.imread(DATA_PATH+'/plot_test/diatest.png')

    ken = np.where(jeff == 1.0, nan, 55) #min temp -60

    i=-1
    #---------------------
    print('')
    print ('Current temperature data:')
    rc1 = np.column_stack((temp_data[-1,1], temp_data[-1,2], temp_data[-1,6], temp_data[-1,7], temp_data[-1,10], temp_data[-1,11])) 
    rc2 = np.column_stack((temp_data[-1,3], temp_data[-1,4], temp_data[-1,5], temp_data[-1,8], temp_data[-1,9], temp_data[-1,12])) 

    avgT = (np.mean(rc1) + np.mean(rc2)) / 2
    avgT = np.round(avgT, decimals=2)
       
    if np.amax(rc1) > np.amax(rc2):
        print ('   Inner Box | Max',np.round(np.amax(rc1), decimals=2), end='')
    else:
        print ('   Inner Box | Max',np.round(np.amax(rc2), decimals=2), end='')
    if np.amin(rc1) < np.amin(rc2):
        print ('°C | Min',np.round(np.amin(rc1), decimals=2), end='')
    else:
        print ('°C | Min',np.round(np.amin(rc2), decimals=2), end='')
    print('°C | Avg', str(avgT)+'°C')

# temperatures of the MAGIC T and first cable
 
    mT1 = np.column_stack((temp_data[-1,14], temp_data[-1,27], temp_data[-1,28]))
    mT2 = np.column_stack((temp_data[-1,13], temp_data[-1,15], temp_data[-1,16]))

    avgT = (np.mean(mT1) + np.mean(mT2)) / 2
    avgT = np.round(avgT, decimals=2)
       
    if np.amax(mT1) > np.amax(mT2):
        print ('   Outer Box | Max',np.round(np.amax(mT1), decimals=2), end='')
    else:
        print ('   Outer Box | Max',np.round(np.amax(mT2), decimals=2), end='')
    if np.amin(mT1) < np.amin(mT2):
        print ('°C | Min',np.round(np.amin(mT1), decimals=2), end='')
    else:
        print ('°C | Min',np.round(np.amin(mT2), decimals=2), end='')
    print('°C | Avg', str(avgT)+'°C')

# Temperatures of horns and cables:

    hc1 = np.column_stack((temp_data[-1,38], temp_data[-1,37], temp_data[-1,36], temp_data[-1,35], temp_data[-1,34], temp_data[-1,33], temp_data[-1,32], temp_data[-1,31], temp_data[-1,30], temp_data[-1,29])) 
    hc2 = np.column_stack((temp_data[-1,26], temp_data[-1,25], temp_data[-1,24], temp_data[-1,23], temp_data[-1,22], temp_data[-1,21], temp_data[-1,20], temp_data[-1,19], temp_data[-1,18], temp_data[-1,17])) 
 
    avgT = (np.mean(hc1) + np.mean(hc2)) / 2
    avgT = np.round(avgT, decimals=2)
    
    if np.amax(hc1) > np.amax(hc2):
        print ('   Externals | Max',np.round(np.amax(hc1), decimals=2), end='')
    else:
        print ('   Externals | Max',np.round(np.amax(hc2), decimals=2), end='')
    if np.amin(hc1) < np.amin(hc2):
        print ('°C | Min',np.round(np.amin(hc1), decimals=2), end='')
    else:
        print ('°C | Min',np.round(np.amin(hc2), decimals=2), end='')
    print('°C | Avg', str(avgT)+'°C')
    
    gc.collect()
#-------

    
    ken[75,75] = 40 #max temp 60 (blues below freezing, the -1 is to flip the color map)
               
    ken[45:48,29:32] = (temp_data[i,28]) #28 E Cab1 (inside box)
    ken[39:42,29:32] = (temp_data[i,27]) #27 MgT Port2/Cab1 Connector
    ken[39:42,37:40] = (temp_data[i,14]) #14 MgT Port4 (CH1 Input Cable)
        
    ken[23:26,34:37] = (temp_data[i,1]) #1 CH1 P162 LNA (1st)
    ken[23:26,30:33] = (temp_data[i,2]) #2 CH1 P162 LNA (2nd)
    ken[19:22,30:33] = (temp_data[i,6]) #6 CH1 P162 LNA (3rd)
    ken[15:18,30:33] = (temp_data[i,11]) #11 CH1 P162 LNA (4th)
    ken[15:18,34:37] = (temp_data[i,7]) #7 CH1 2534 LNA
    ken[19:22,34:37] = (temp_data[i,10]) #10 CH1 Phase Switch
 
    ken[19:22,39:42] = (temp_data[i,9]) #9 CH2 Phase Switch
    ken[15:18,39:42] = (temp_data[i,8]) #8 CH2 2534 LNA
    ken[15:18,43:46] = (temp_data[i,12]) #12 CH2 P162 LNA (4th)
    ken[19:22,43:46] = (temp_data[i,5]) #5 CH2 P162 LNA (3rd)
    ken[23:26,43:46] = (temp_data[i,4]) #4 CH2 P162 LNA (2nd)
    ken[23:26,39:42] = (temp_data[i,3]) #3 CH2 P162 LNA (1st)
        
    ken[33:36,39:42] = (temp_data[i,13]) #13 MgT Port3 (ChH2 Input Cable)
    ken[39:42,44:47] = (temp_data[i,15]) #15 MgT Port1/Cab2 Connector
    ken[45:48,44:47] = (temp_data[i,16]) #16 (inside box)

    
    plt.title('Reciever Box ' +str(temp_data[-1,0]))
    plt.ylim(52,6)
    plt.yticks([])
    plt.xticks([])
    plt.xlim(22,53)
            
    plt.imshow(ken, cmap='YlOrRd')
    plt.colorbar()    
    plt.pause(2)
    #plt.savefig(DATA_PATH+'/plot_test/LIVE_BOARD_TEMPS.png')
    
    print('')
    #refresh = input('Refresh live temperature diagram? (Y or N): ')
    #if refresh == 'N' or refresh == 'n':
     #   looper = False 
    #else:
     #   pass

    del temp_data, ken, df_from_each_file, concatenated_df
    
    input('Press enter to return to Main Menu')
    looper=False
    plt.clf()
    return looper
    
    #-------------------------------------------------------------

looper = True
while looper:
    looper = LIVEboard()
