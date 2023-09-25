#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:49:47 2022

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

def LIVEtemps():
        
    temp_files = []
    temp_files = sorted(glob.glob('1W_Temp-*.csv'))

    df_from_each_file = (pd.read_csv(f) for f in temp_files)
    concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)
    #concatenated_df.to_csv(DATA_PATH+'/temp/temp_data.csv', index=False)

    temp_data = concatenated_df.to_numpy()


    jeff = plt.imread(DATA_PATH+'/plot_test/diatest.png')

    ken = np.where(jeff == 1.0, nan, 50) #min temp -50

    i=-1
    
    ken[75,75] = -60 #max temp 60 (blues below freezing, the -1 is to flip the color map)

    ken[20:23,11:14] = (-1 * temp_data[i,38]) #38 E Horn
    ken[41:44,11:14] = (-1 * temp_data[i,37]) #37 E Throat
    ken[50:53,11:14] = (-1 * temp_data[i,36]) #36 E Polariser
    ken[59:62,11:14] = (-1 * temp_data[i,35]) #35 E Pol/Cab Connector

    ken[65:68,11:14] = (-1 * temp_data[i,34]) #34 E Cab1
    ken[65:68,17:20] = (-1 * temp_data[i,33]) #33 E Cab1
    ken[65:68,23:26] = (-1 * temp_data[i,32]) #32 E Cab1
    ken[65:68,29:32] = (-1 * temp_data[i,31]) #31 E Cab1
    ken[59:62,29:32] = (-1 * temp_data[i,30]) #30 E Cab1
    ken[53:56,29:32] = (-1 * temp_data[i,29]) #29 E Cab1
    
    ken[45:48,29:32] = (-1 * temp_data[i,28]) #28 E Cab1 (inside box)
    ken[39:42,29:32] = (-1 * temp_data[i,27]) #27 MgT Port2/Cab1 Connector
    ken[39:42,37:40] = (-1 * temp_data[i,14]) #14 MgT Port4 (CH1 Input Cable)

    ken[23:26,34:37] = (-1 * temp_data[i,1]) #1 CH1 P162 LNA (1st)
    ken[23:26,30:33] = (-1 * temp_data[i,2]) #2 CH1 P162 LNA (2nd)
    ken[19:22,30:33] = (-1 * temp_data[i,6]) #6 CH1 P162 LNA (3rd)
    ken[15:18,30:33] = (-1 * temp_data[i,11]) #11 CH1 P162 LNA (4th)
    ken[15:18,34:37] = (-1 * temp_data[i,7]) #7 CH1 2534 LNA
    ken[19:22,34:37] = (-1 * temp_data[i,10]) #10 CH1 Phase Switch

    ken[19:22,39:42] = (-1 * temp_data[i,9]) #9 CH2 Phase Switch
    ken[15:18,39:42] = (-1 * temp_data[i,8]) #8 CH2 2534 LNA
    ken[15:18,43:46] = (-1 * temp_data[i,12]) #12 CH2 P162 LNA (4th)
    ken[19:22,43:46] = (-1 * temp_data[i,5]) #5 CH2 P162 LNA (3rd)
    ken[23:26,43:46] = (-1 * temp_data[i,4]) #4 CH2 P162 LNA (2nd)
    ken[23:26,39:42] = (-1 * temp_data[i,3]) #3 CH2 P162 LNA (1st)

    ken[33:36,39:42] = (-1 * temp_data[i,13]) #13 MgT Port3 (ChH2 Input Cable)
    ken[39:42,44:47] = (-1 * temp_data[i,15]) #15 MgT Port1/Cab2 Connector
    ken[45:48,44:47] = (-1 * temp_data[i,16]) #16 (inside box)

    ken[53:56,44:47] = (-1 * temp_data[i,17]) #17 W Cab2
    ken[59:62,44:47] = (-1 * temp_data[i,18]) #18 W Cab2
    ken[65:68,44:47] = (-1 * temp_data[i,19]) #19 W Cab2
    ken[65:68,50:53] = (-1 * temp_data[i,20]) #20 W Cab2
    ken[65:68,56:59] = (-1 * temp_data[i,21]) #21 W Cab2
    ken[65:68,62:65] = (-1 * temp_data[i,22]) #22 W Cab2

    ken[59:62,62:65] = (-1 * temp_data[i,23]) #23 W Pol/Cab Connector
    ken[50:53,62:65] = (-1 * temp_data[i,24]) #24 W Polariser
    ken[41:44,62:65] = (-1 * temp_data[i,25]) #25 W Throat
    ken[20:23,62:65] = (-1 * temp_data[i,26]) #26 W Horn

    plt.title('LBASS instrument ' +str(temp_data[-1,0]))
    plt.ylim(74,0)
    plt.yticks([])
    plt.xticks([])
    
    
    plt.imshow(ken, cmap='RdYlBu')
    
    
    plt.savefig(DATA_PATH+'/plot_test/LIVE_INST_TEMPS.png')
    
    #-------------------------------------------------------------

def LIVEboard():
    
    temp_files = []
    temp_files = sorted(glob.glob('1W_Temp-*.csv'))

    df_from_each_file = (pd.read_csv(f) for f in temp_files)
    concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)
    #concatenated_df.to_csv(DATA_PATH+'/temp/temp_data.csv', index=False)

    temp_data = concatenated_df.to_numpy()


    jeff = plt.imread(DATA_PATH+'/plot_test/diatest.png')

    ken = np.where(jeff == 1.0, nan, 50) #min temp -50

    i=-1
    for i in range (0,np.size(temp_data[:,0]),5):
    
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
        
    
        plt.savefig(DATA_PATH+'/plot_test/LIVE_BOARD_TEMPS.png')
    
    
    
#----------------------------------------------------------------------




exec(open("/local5/scratch/pblack/scripts/INST.py").read())

input('Plot one done, close Macro, Press enter')
exec(open("/local5/scratch/pblack/scripts/BOARD.py").read())
