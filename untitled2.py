#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 17:19:48 2022

@author: pblack
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:33:26 2022

@author: pblack
"""

DATA_PATH = '/mirror/scratch/pblack'


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import math
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

os.system('rm /local5/scratch/pblack/temp/temp_data.npy') #clear contents of previous run
os.system('rm /local5/scratch/pblack/temp/file2.npy') #clear contents of previous run

def temps():
    
    temp_files = []
    temp_files = sorted(glob.glob('1W_Temp-*.csv'))

    df_from_each_file = (pd.read_csv(f) for f in temp_files)
    concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)
    #concatenated_df.to_csv(DATA_PATH+'/temp/temp_data.csv', index=False)

    all_temps = concatenated_df.to_numpy()
    np.save(DATA_PATH+'/temp/temp_data.npy', all_temps)

temps()

#---------------------------------------------------------------------------

def selectobs():
    
    date = input ('Date of observation (YYYY-MM-DD): ') #user enters date
    if isinstance(date, tuple):
        print ('Initial input miread as tuple')
    try:
        year, month, day = date.split('-')
    except:
        print ('Incorrect format or data type. Please try again.')
        selectobs()

    isValidDate = True
    try:
        datetime.datetime(int(year), int(month), int(day))
    except ValueError:
        isValidDate = False
        print ('Invalid date entered.  Please try again.')
        selectobs()
        
    if isValidDate:
        print ('')

    go_date = datecheck(date)

    return date, go_date
#-------------------------------------------------------------------------
    
def selecttime(go_date): # choose time of day as callable function
    
    time_in = input ('Time to inspect data from (HH:MM): ')
    
    try:
        hour, minute = time_in.split(':')
    except:
        print ('Incorrect format or data type. Please try again.')
        selecttime()

    isValidTime = True
    try:
        datetime.time(int(hour), int(minute))
    except ValueError:
        isValidTime = False

    if(isValidTime):
        pass
        
    else:
        print("Invalid time entered. Please try again. Use 24HR clock, starting at 00:00 midnight.")
        selecttime()
    
    user_sample = float((int(hour) * 3600) + (int(minute) * 60))
    
    go_time = timecheck(hour, go_date)
    
    return hour, minute, user_sample, time_in, go_time

#-------------------------------------------------------------------------
    
def selectduration(): # choose duration as callable function
     
    duration_in = input ('Please enter duration of observations in hours (or 0 for all available data): ')

    if duration_in.isnumeric():
        pass
    else:
        print ('Input not recognised, please enter an numeric value, e.g. 6 or 2.5')
        selectduration()
    
    if float(duration_in) == 0:
        a = 5000 #daft number of observing hours that will never be reached
    else:
        a = float(duration_in)
    
    return a

#---------------------------------------------------------------------------

def datecheck(date):
    
        
    all_temps = np.load(DATA_PATH+'/temp/temp_data.npy', allow_pickle=True)
    go_date='banana'
    i=0
    for i in range (0,np.size(all_temps[:,0])):
        x = str(all_temps[i,0])
        y = x.split()
        if str(y[0]) == str(date):
            go_date = i
            break
    
    if go_date == 'banana':
        print ('No data available on that date. Please try again.')
        selectobs()

    return go_date
    
#---------------------------------------------------------------------

def timecheck(hour, go_date):

    all_temps = np.load(DATA_PATH+'/temp/temp_data.npy', allow_pickle=True)
    go_time='banana'

    i = go_date
    for i in range(go_date,(go_date+1440)): #there is only 24 hours in a day!
        x = str(all_temps[i,0])
        y = x.split()
        h,m,s = y[1].split(':')
        if h == str(hour):
            go_time = i - 5
            break
    
    if go_time == 'banana':
        print ('No data found at time specified. Next available data will be shown.')
        go_time = go_date
    
    return (go_time)    

#--------------------------------------------------------------------
    
def temp_selects(user_inputs, go_time):

    
    all_temps = np.load(DATA_PATH+'/temp/temp_data.npy', allow_pickle=True)
    
    run_length = (go_time + (int(float(user_inputs[5])) + 1) * 60)
    
    selected_temps = all_temps[go_time:run_length,:]
    
    
    i = 0
    for i in tqdm(range (0,(np.size(selected_temps[:,0]))-1)):
        a = Time(selected_temps[i,0], format='iso')
        b = Time(selected_temps[(i+1),0], format='iso')
        c = b-a
        gap = int(c.sec + 0.5)
        if gap > 300:
            print ('Temperature data unavailable for full duration specified.')
            print ('Temperature records end at ',selected_temps[i,0])
            selected_temps = selected_temps[:i,:]
            break
        else:
            pass
        
    np.save(DATA_PATH+'/temp/file2.npy', selected_temps)
    
    return selected_temps
    
#--------------------------------------------------------------------------------

def groupSensors():
    
    temp_data = np.load(DATA_PATH+'/temp/file2.npy', allow_pickle=True)

    print ('Grouping data by sensor location')

# Temperatures of receiver box: sequence is Magic T input, 1st, 2nd, 3rd P162 LNAs, 2534 LNA, Phase switch, 4th P162 LNA. 

    rc1 = np.column_stack((temp_data[:,1], temp_data[:,2], temp_data[:,6], temp_data[:,7], temp_data[:,10], temp_data[:,11])) 
    rc2 = np.column_stack((temp_data[:,3], temp_data[:,4], temp_data[:,5], temp_data[:,8], temp_data[:,9], temp_data[:,12])) 

# temperatures of the MAGIC T and first cable
 
    mT1 = np.column_stack((temp_data[:,14], temp_data[:,27], temp_data[:,28]))
    mT2 = np.column_stack((temp_data[:,13], temp_data[:,15], temp_data[:,16]))

# Temperatures of horns and cables:

    hc1 = np.column_stack((temp_data[:,38], temp_data[:,37], temp_data[:,36], temp_data[:,35], temp_data[:,34], temp_data[:,33], temp_data[:,32], temp_data[:,31], temp_data[:,30], temp_data[:,29])) 
    hc2 = np.column_stack((temp_data[:,26], temp_data[:,25], temp_data[:,24], temp_data[:,23], temp_data[:,22], temp_data[:,21], temp_data[:,20], temp_data[:,19], temp_data[:,18], temp_data[:,17])) 
 
    
    return rc1, rc2, hc1, hc2, mT1, mT2, temp_data[:,0]


#---------------------------------------------------------------------

def rcplot(rc1, rc2, mins):
    
    x = np.arange(0,np.size(temp_times),1)
    
    figure(figsize=(12, 8), dpi=300)

    plt.plot(x,rc1[:,0], linestyle='-', c='r', linewidth=0.2)
    plt.plot(x,rc1[:,1], linestyle='-', c='g', linewidth=0.2)
    plt.plot(x,rc1[:,2], linestyle='-', c='blueviolet', linewidth=0.2)
    plt.plot(x,rc1[:,3], linestyle='-', c='b', linewidth=0.2)
    plt.plot(x,rc1[:,4], linestyle='-', c='m', linewidth=0.2)
    plt.plot(x,rc1[:,5], linestyle='-', c='orange', linewidth=0.2)
    
    plt.plot(x,rc2[:,0], linestyle='--', c='r', linewidth=0.2)
    plt.plot(x,rc2[:,1], linestyle='--', c='g', linewidth=0.2)
    plt.plot(x,rc2[:,2], linestyle='--', c='blueviolet', linewidth=0.2)
    plt.plot(x,rc2[:,3], linestyle='--', c='b', linewidth=0.2)
    plt.plot(x,rc2[:,4], linestyle='--', c='m', linewidth=0.2)
    plt.plot(x,rc2[:,5], linestyle='--', c='orange', linewidth=0.2)
    
    plt.savefig(DATA_PATH+'/plot_test/rc.png')
   
    
#-------------------------------------------------------------------------


def mTplot(rc1, rc2, mins):
    
    x = np.arange(0,np.size(temp_times),1)
   
    
    plt.plot(x,mT1[:,0], linestyle='-', c='r', linewidth=0.2)
    plt.plot(x,mT1[:,1], linestyle='-', c='g', linewidth=0.2)
    plt.plot(x,mT1[:,2], linestyle='-', c='b', linewidth=0.2)
    plt.plot(x,mT2[:,0], linestyle='--', c='r', linewidth=0.2)
    plt.plot(x,mT2[:,1], linestyle='--', c='g', linewidth=0.2)
    plt.plot(x,mT2[:,2], linestyle='--', c='b', linewidth=0.2)
    plt.xticks(rotation = 270)
    
    plt.savefig(DATA_PATH+'/plot_test/magicT.png')


#-------------------------------------------------------------------------
    
def hcplot(hc1, hc2, mins):
    
    x = np.arange(0,np.size(temp_times),1)
    
    plt.plot(x,hc1[:,0], linestyle='-', c='r', linewidth=0.5)
    plt.plot(x,hc1[:,1], linestyle='-', c='g', linewidth=0.5)
    plt.plot(x,hc1[:,2], linestyle='-', c='blueviolet', linewidth=0.5)
    plt.plot(x,hc1[:,3], linestyle='-', c='b', linewidth=0.5)
    plt.plot(x,hc1[:,4], linestyle='-', c='m', linewidth=0.5)
    plt.plot(x,hc1[:,5], linestyle='-', c='y', linewidth=0.5)
    plt.plot(x,hc1[:,6], linestyle='-', c='orange', linewidth=0.5)
    plt.plot(x,hc1[:,7], linestyle='-', c='lime', linewidth=0.5)
    plt.plot(x,hc1[:,8], linestyle='-', c='teal', linewidth=0.5)
    plt.plot(x,hc1[:,9], linestyle='-', c='c', linewidth=0.5)
    
    plt.plot(x,hc2[:,0], linestyle='--', c='r', linewidth=0.5)
    plt.plot(x,hc2[:,1], linestyle='--', c='g', linewidth=0.5)
    plt.plot(x,hc2[:,2], linestyle='--', c='blueviolet', linewidth=0.5)
    plt.plot(x,hc2[:,3], linestyle='--', c='b', linewidth=0.5)
    plt.plot(x,hc2[:,4], linestyle='--', c='m', linewidth=0.5)
    plt.plot(x,hc2[:,5], linestyle='--', c='y', linewidth=0.5)
    plt.plot(x,hc2[:,6], linestyle='--', c='orange', linewidth=0.5)
    plt.plot(x,hc2[:,7], linestyle='--', c='lime', linewidth=0.5)
    plt.plot(x,hc2[:,8], linestyle='--', c='teal', linewidth=0.5)
    plt.plot(x,hc2[:,9], linestyle='--', c='c', linewidth=0.5)
    
    plt.savefig(DATA_PATH+'/plot_test/hc.png')
  
    
    
#---------------------------------------------------------------------

def threeINone(rc1, rc2, mT1, mT2, hc1, hc2, temp_times, mins, first_tick, user_inputs):
    
         
    fig = plt.figure(figsize=(12, 8), dpi=300)
    
    gs = fig.add_gridspec(3, hspace=0)
    axs = gs.subplots(sharex=True, sharey=False)
    #fig.suptitle('Sharing both axes')
    
    x = np.arange(0,np.size(temp_times),1)
  
    
    plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60), labels=mins, rotation=270)
    plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),15))
    
    
    #plt.style.use('default')
    #radiometer
    # Hide x labels and tick labels for all but bottom plot.
    for ax in axs:
        ax.label_outer()
        ax.margins(x=0)
    
    for ax in axs.flat:
        ax.set(xlabel='Time')
        
        
    axs[0].plot(x,rc1[:,0], linestyle='-', c='r', linewidth=0.5)
    axs[0].plot(x,rc1[:,1], linestyle='-', c='g', linewidth=0.5)
    axs[0].plot(x,rc1[:,2], linestyle='-', c='blueviolet', linewidth=0.5)
    axs[0].plot(x,rc1[:,3], linestyle='-', c='b', linewidth=0.5)
    axs[0].plot(x,rc1[:,4], linestyle='-', c='m', linewidth=0.5)
    axs[0].plot(x,rc1[:,5], linestyle='-', c='orange', linewidth=0.5)
    
    axs[0].plot(x,rc2[:,0], linestyle='--', dashes=(5, 10), c='r', linewidth=0.5)
    axs[0].plot(x,rc2[:,1], linestyle='--', dashes=(5, 10), c='g', linewidth=0.5)
    axs[0].plot(x,rc2[:,2], linestyle='--', dashes=(5, 10), c='blueviolet', linewidth=0.5)
    axs[0].plot(x,rc2[:,3], linestyle='--', dashes=(5, 10), c='b', linewidth=0.5)
    axs[0].plot(x,rc2[:,4], linestyle='--', dashes=(5, 10), c='m', linewidth=0.5)
    axs[0].plot(x,rc2[:,5], linestyle='--', dashes=(5, 10), c='orange', linewidth=0.5)
    
    # Magic T
    
    axs[1].plot(x,mT1[:,0], linestyle='-', c='r', linewidth=0.5)
    axs[1].plot(x,mT1[:,1], linestyle='-', c='g', linewidth=0.5)
    axs[1].plot(x,mT1[:,2], linestyle='-', c='b', linewidth=0.5)
    axs[1].plot(x,mT2[:,0], linestyle='--', dashes=(5, 10), c='r', linewidth=0.5)
    axs[1].plot(x,mT2[:,1], linestyle='--', dashes=(5, 10), c='g', linewidth=0.5)
    axs[1].plot(x,mT2[:,2], linestyle='--', dashes=(5, 10), c='b', linewidth=0.5)
    
    axs[1].set(ylabel='Temperature / °C')
    
    # Horns
    
    axs[2].plot(x,hc1[:,0], linestyle='-', c='r', linewidth=0.5)
    axs[2].plot(x,hc1[:,1], linestyle='-', c='g', linewidth=0.5)
    axs[2].plot(x,hc1[:,2], linestyle='-', c='blueviolet', linewidth=0.5)
    axs[2].plot(x,hc1[:,3], linestyle='-', c='b', linewidth=0.5)
    axs[2].plot(x,hc1[:,4], linestyle='-', c='m', linewidth=0.5)
    axs[2].plot(x,hc1[:,5], linestyle='-', c='y', linewidth=0.5)
    axs[2].plot(x,hc1[:,6], linestyle='-', c='orange', linewidth=0.5)
    axs[2].plot(x,hc1[:,7], linestyle='-', c='lime', linewidth=0.5)
    axs[2].plot(x,hc1[:,8], linestyle='-', c='teal', linewidth=0.5)
    axs[2].plot(x,hc1[:,9], linestyle='-', c='c', linewidth=0.5)
    
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
    
    plt.suptitle(user_inputs[0]+'_'+user_inputs[3]+' for '+user_inputs[5]+' hours. 1Wire Temperature Data')
    
    plt.savefig(DATA_PATH+'/plot_test/'+user_inputs[0]+'_'+user_inputs[3]+'_'+user_inputs[5]+'_1Wtemps.png')
    
#------------------------------------------------------------------------

def diagram():
    
    temp_data = np.load(DATA_PATH+'/temp/file2.npy', allow_pickle=True)
    
    jeff = plt.imread(DATA_PATH+'/plot_test/diatest.png')

    ken = np.where(jeff == 1.0, nan, 50) #min temp -50

    i=0
    for i in range (0,np.size(temp_data[:,0]),5):
    
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

        plt.title('LBASS instrument ' +str(temp_data[i,0]))
        plt.ylim(75,0)
        plt.yticks([])
        plt.xticks([])
        
        plt.pause(0.01)
        plt.imshow(ken, cmap='RdYlBu')
        #plt.colorbar()
        plt.pause(1)
        
    plt.savefig(DATA_PATH+'/plot_test/temp_diagram.png')

#------------------------------------------------------------------------------
    
def diagramBOX():
    
    temp_data = np.load(DATA_PATH+'/temp/file2.npy', allow_pickle=True)
    
    jeff = plt.imread(DATA_PATH+'/plot_test/diatest.png')

    ken = np.where(jeff == 1.0, nan, 50) #min temp -50

    i=0
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


        plt.title('Reciever Box ' +str(temp_data[i,0]))
        plt.ylim(52,6)
        plt.yticks([])
        plt.xticks([])
        plt.xlim(22,53)
        
        plt.pause(0.001)
        plt.imshow(ken, cmap='YlOrRd')
        plt.colorbar()
        
        plt.pause(1)
        
    plt.savefig(DATA_PATH+'/plot_test/temp_BOX_diagram.png')

#---------------------------------------------------------------------------------



#--------------------------------------------------------------------------------

# load in the user selections and related fits / csv files

temps()
date, go_date = selectobs()
user_hour, user_minute, user_sample, time_in, go_time = selecttime(go_date)
user_duration = selectduration()

user_inputs = np.array((date, user_hour, user_minute, time_in, user_sample, user_duration))

selected_temps = temp_selects(user_inputs, go_time)

rc1, rc2, hc1, hc2, mT1, mT2, temp_times = groupSensors()
#diagram()
#diagramBOX()

i = 0

mins =[]
first_tick=[]
for i in range(0,np.size(temp_times)): #there is only 24 hours in a day!
    x = str(temp_times[i])
    y = x.split()
    h,m,s = y[1].split(':')
    if m == '00': 
        mins.append(h+':'+m)
        first_tick.append(i)
    else:
        pass
        #mins.append(None)




#mTplot(mT1, mT2, mins)
#rcplot(rc1, rc2, mins)
#hcplot(hc1, hc2, mins)

threeINone(rc1, rc2, mT1, mT2, hc1, hc2, temp_times, mins, first_tick, user_inputs)

#----------------------------------------------------------------------

