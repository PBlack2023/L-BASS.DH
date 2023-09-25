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
import matplotlib.dates as mdates
import matplotlib.animation as animation
import math
import scipy
import numpy as np
import astropy.io
from astropy.io import fits
from astropy.time import Time 
from astropy.time import TimeDelta
import datetime
import time
from tqdm import tqdm #progress bars
import glob
import os
from math import nan
import csv
from csv import DictReader
import pandas as pd
from matplotlib.pyplot import figure

os.chdir('/scratch/nas_lbass/raw_data/')


def temps():
    
    if os.path.exists(DATA_PATH+'/temp/temp_data.npy'):
        os.system('rm /local5/scratch/pblack/temp/temp_data.npy')
    
    print('\033[0;0m Loading csv files \033[1;32m')
    print('')
    
    temp_files = []
    temp_files = sorted(glob.glob('1W_Temp-*.csv'))
    

    df_from_each_file = (pd.read_csv(f) for f in temp_files)
    concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)
    #concatenated_df.to_csv(DATA_PATH+'/temp/temp_data.csv', index=False)

    all_temps = concatenated_df.to_numpy()
    
    return all_temps



#---------------------------------------------------------------------------

def selectobs(all_temps, getDate):
    
    path_string='Error'
    go_date='Error'
    date = input ('Date of observation (YYYY-MM-DD): ') #user enters date
    if isinstance(date, tuple):
        print ('\033[1;31m Initial input miread as tuple. \033[1;32m')
    if len(date) < 10:
        date = 'Error'
    
    try:
        year, month, day = date.split('-')
    
        isValidDate = True
        try:
            datetime.datetime(int(year), int(month), int(day))
        except ValueError:
            isValidDate = False
            print ('')
            print ('\033[1;31m Invalid date entered. Please try again. \033[1;32m')
            print ('')
        
        if isValidDate:
            print ('')
            getDate=False #end loop
            path_string = '/scratch/nas_lbass/analysis/'+str(year)+'_'+str(month)+'/' 
            
            go_date, getDate = datecheck(date, all_temps, getDate)
    
    except:
        print ('')
        print ('\033[1;31m Incorrect format or data type. Please try again. \033[1;32m')
        print ('')
        

    

    return date, go_date, path_string, getDate
#-------------------------------------------------------------------------
    
def selecttime(go_date, all_temps, getTime): # choose time of day as callable function
    
    go_time='Error'
    user_sample='Error'
    hour = 'Error'
    minute='Error'
    time_in = input ('\033[1;32m Time to inspect data from (HH:MM): ')
    if len(time_in) < 5:
        time_in='Error'
    try:
        hour, minute = time_in.split(':')
    
        isValidTime = True
        try:
            datetime.time(int(hour), int(minute))
        except ValueError:
            isValidTime = False

        if(isValidTime):
            getTime=False #end loop
            print('')
            
            user_sample = float((int(hour) * 3600) + (int(minute) * 60))
    
            go_time = timecheck(hour, go_date, all_temps, minute)
        
        else:
            print('')
            print("\033[1;31m Invalid time entered, use 24HR clock. Please try again.\033[1;32m")
            print('')
    
    except:
        print('')
        print ('\033[1;31m Incorrect format or data type. Please try again \033[1;32m')
        print('')

    
    return hour, minute, user_sample, time_in, go_time, getTime

#-------------------------------------------------------------------------
    
def selectduration(getDuration): # choose duration as callable function
    
    a='Error'
    duration_in = input ('Duration to inspect in hours: ')
    print('')
    
    try:
        float(duration_in)
        getDuration=False
    
        if float(duration_in) == 0:
            a = 5000 #daft number of observing hours that will never be reached
        else:
            a = float(duration_in)
    except:
    
        print ('\033[1;31m Input not recognised, please enter an numeric value, e.g. 6 or 2.5 \033[1;32m')
        print ('')
    
    return a, getDuration

#---------------------------------------------------------------------------

def datecheck(date, all_temps, getDate):
    
        
    #all_temps = np.load(DATA_PATH+'/temp/temp_data.npy', allow_pickle=True)
    go_date='banana'
    i=0
    for i in range (0,np.size(all_temps[:,0])):
        x = str(all_temps[i,0])
        y = x.split()
        if str(y[0]) == str(date):
            go_date = i
            break
    
    if go_date == 'banana':
        getDate=True #restart loop
        print('')
        print ('\033[1;31m No data available on that date. Please try again. \033[1;32m')
        print('')

    return go_date, getDate
    
#---------------------------------------------------------------------

def timecheck(hour, go_date, all_temps, minute):

    #all_temps = np.load(DATA_PATH+'/temp/temp_data.npy', allow_pickle=True)
    go_time='banana'

    i = go_date
    for i in range(go_date,(go_date+1440)): #there is only 24 hours in a day!
        x = str(all_temps[i,0])
        y = x.split()
        h,m,s = y[1].split(':')
        if h == str(hour) and m == str(minute):
            go_time = i
            break
    
    if go_time == 'banana':
        print ('\033[1;31m No data found at time specified. Next available data will be shown.\033[1;32m')
        go_time = go_date
    
    return (go_time)    

#--------------------------------------------------------------------
    
def temp_selects(user_inputs, go_time, all_temps):

    if os.path.exists(DATA_PATH+'/temp/file2.npy'):
        os.system('rm /local5/scratch/pblack/temp/file2.npy')
    
    #all_temps = np.load(DATA_PATH+'/temp/temp_data.npy', allow_pickle=True)
    
    run_length = int(go_time + (float(user_inputs[5])) * 60) #the +1 is because the integer function rounds down
    
    selected_temps = all_temps[go_time:run_length,:]
    
    print('\033[0;0m Searching 1Wire Temperature Data')
    time.sleep(1)
    
    i = 0
    for i in tqdm(range (0,(np.size(selected_temps[:,0]))-1), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        a = Time(selected_temps[i,0], format='iso')
        b = Time(selected_temps[(i+1),0], format='iso')
        c = b-a
        gap = int(c.sec + 0.5)
        if gap > 300:
            print('')
            print ('\033[1;31m Temperature data unavailable for full duration specified.')
            print('')
            print ('\033[1;32m Temperature data available until ',selected_temps[i,0])
            print('')
            selected_temps = selected_temps[:i,:]
            break
        else:
            pass
    time.sleep(1)   
    np.save(DATA_PATH+'/temp/file2.npy', selected_temps)
    
    return selected_temps
    
#--------------------------------------------------------------------------------

def groupSensors(duration_actual):
    
    temp_data = np.load(DATA_PATH+'/temp/file2.npy', allow_pickle=True)

    print ('\033[0;m Sorting data by sensor location')
    print ('')
    print ('\033[1;32m During this',duration_actual,'hour period:')

# Temperatures of receiver box: sequence is Magic T input, 1st, 2nd, 3rd P162 LNAs, 2534 LNA, Phase switch, 4th P162 LNA. 

    rc1 = np.column_stack((temp_data[:,1], temp_data[:,2], temp_data[:,6], temp_data[:,7], temp_data[:,10], temp_data[:,11])) 
    rc2 = np.column_stack((temp_data[:,3], temp_data[:,4], temp_data[:,5], temp_data[:,8], temp_data[:,9], temp_data[:,12])) 

    avgT = (np.mean(rc1) + np.mean(rc2)) / 2
    avgT = np.round(avgT, decimals=2)
       
    if np.amax(rc1) > np.amax(rc2):
        print ('   Inner Box | Max',np.amax(rc1), end='')
    else:
        print ('   Inner Box | Max',np.amax(rc2), end='')
    if np.amin(rc1) < np.amin(rc2):
        print ('°C | Min',np.amin(rc1), end='')
    else:
        print ('°C | Min',np.amin(rc2), end='')
    print('°C | Avg', str(avgT)+'°C')

# temperatures of the MAGIC T and first cable
 
    mT1 = np.column_stack((temp_data[:,14], temp_data[:,27], temp_data[:,28]))
    mT2 = np.column_stack((temp_data[:,13], temp_data[:,15], temp_data[:,16]))

    avgT = (np.mean(mT1) + np.mean(mT2)) / 2
    avgT = np.round(avgT, decimals=2)
       
    if np.amax(mT1) > np.amax(mT2):
        print ('   Outer Box | Max',np.amax(mT1), end='')
    else:
        print ('   Outer Box | Max',np.amax(mT2), end='')
    if np.amin(mT1) < np.amin(mT2):
        print ('°C | Min',np.amin(mT1), end='')
    else:
        print ('°C | Min',np.amin(mT2), end='')
    print('°C | Avg', str(avgT)+'°C')

# Temperatures of horns and cables:

    hc1 = np.column_stack((temp_data[:,38], temp_data[:,37], temp_data[:,36], temp_data[:,35], temp_data[:,34], temp_data[:,33], temp_data[:,32], temp_data[:,31], temp_data[:,30], temp_data[:,29])) 
    hc2 = np.column_stack((temp_data[:,26], temp_data[:,25], temp_data[:,24], temp_data[:,23], temp_data[:,22], temp_data[:,21], temp_data[:,20], temp_data[:,19], temp_data[:,18], temp_data[:,17])) 
 
    avgT = (np.mean(hc1) + np.mean(hc2)) / 2
    avgT = np.round(avgT, decimals=2)
    
    if np.amax(hc1) > np.amax(hc2):
        print ('   Externals | Max',np.amax(hc1), end='')
    else:
        print ('   Externals | Max',np.amax(hc2), end='')
    if np.amin(hc1) < np.amin(hc2):
        print ('°C | Min',np.amin(hc1), end='')
    else:
        print ('°C | Min',np.amin(hc2), end='')
    print('°C | Avg', str(avgT)+'°C')
    
    return rc1, rc2, hc1, hc2, mT1, mT2, temp_data[:,0]


#---------------------------------------------------------------------

def rcplot(rc1, rc2, mins, first_tick, temp_times, duration_actual, save_it=False, first_loop=True):
    
    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    x = np.arange(0,np.size(temp_times),1)
    
    avgrc1 = rc1.mean(axis=1)
    avgrc2 = rc2.mean(axis=1)
    
    avgrc = (avgrc1+avgrc2)/2
    
    np.save(DATA_PATH+'/temp/4norm/rc-mean.npy', avgrc)
    
    title_string = user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours. Inner Box, Solid East, Dashed West.'

    if save_it:
        plt.figure(figsize=(12, 8), dpi=300)

    try:
        plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60), labels=mins, rotation=270)
        if duration_actual > 48 and duration_actual < 300:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60))
        elif duration_actual > 300:
            pass
        else:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),15))
    
    except:
        print('')
        print(' \033[0;m Applying x-axis tick mark correction. \033[1;32m ')
        print('')
        mins = mins[1:]
        plt.xticks(ticks=np.arange(first_tick[0],(np.size(temp_times)),60), labels=mins, rotation=270)
        if duration_actual > 48 and duration_actual < 300:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60))
        elif duration_actual > 300:
            pass
        else:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),15))

    plt.plot(x, avgrc, linestyle='-', c='k', linewidth=1, label='Mean Temperature')

    plt.plot(x,rc1[:,0], linestyle='-', c='r', linewidth=0.5, label='LNA1')
    plt.plot(x,rc1[:,1], linestyle='-', c='g', linewidth=0.5, label='LNA2')
    plt.plot(x,rc1[:,2], linestyle='-', c='blueviolet', linewidth=0.5, label='LNA3')
    plt.plot(x,rc1[:,3], linestyle='-', c='b', linewidth=0.5,label='2534 LNA')
    plt.plot(x,rc1[:,4], linestyle='-', c='m', linewidth=0.5, label='Phase Switch')
    plt.plot(x,rc1[:,5], linestyle='-', c='orange', linewidth=0.5, label='LNA4')
    
    plt.plot(x,rc2[:,0], linestyle='--', c='r', linewidth=0.5)
    plt.plot(x,rc2[:,1], linestyle='--', c='g', linewidth=0.5)
    plt.plot(x,rc2[:,2], linestyle='--', c='blueviolet', linewidth=0.5)
    plt.plot(x,rc2[:,3], linestyle='--', c='b', linewidth=0.5)
    plt.plot(x,rc2[:,4], linestyle='--', c='m', linewidth=0.5)
    plt.plot(x,rc2[:,5], linestyle='--', c='orange', linewidth=0.5)
    
    plt.xlabel('Time')
    plt.ylabel('Temperature / °C')
    plt.margins(x=0)
    plt.tight_layout(h_pad=0.2, w_pad=0.2, rect=[0.117,0.175,0.977,0.922]) #tuple (left, bottom, right, top),
    plt.title(title_string)
    if save_it:
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_InnerBoxTemps.png', bbox_inches="tight")
    else:   
        plt.legend(loc="upper right", fontsize=8)
        plt.show()
    
    plt.close()

    if first_loop:
        print ('')
        save = input ('Do you want to save a printer friendly copy of this plot? (Y/N): ')
        if str(save) == 'N' or str(save) == 'n':
            pass
        elif str(save) == 'Y' or str(save) == 'y':
            save_it=True
            first_loop=False
            rcplot(rc1, rc2, mins, first_tick, temp_times, duration_actual, save_it, first_loop)
 
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 
    
    
    
#-------------------------------------------------------------------------


def mTplot(mT1, mT2, mins, first_tick, duration_actual, save_it=False, first_loop=True):

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    x = np.arange(0,np.size(temp_times),1)
   
    avgmT1 = mT1.mean(axis=1)
    avgmT2 = mT2.mean(axis=1)
    
    avgmT = (avgmT1+avgmT2)/2
    
    np.save(DATA_PATH+'/temp/4norm/mT-mean.npy', avgmT)
    
    title_string = user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours. Outter Box.'
    if save_it:
        plt.figure(figsize=(12, 8), dpi=300)
    try:
        plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60), labels=mins, rotation=270)
        if duration_actual > 48 and duration_actual < 300:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60))
        elif duration_actual > 300:
            pass
        else:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),15))
    
    except:
        print('')
        print(' \033[0;m Applying x-axis tick mark correction. \033[1;32m ')
        print('')
        mins = mins[1:]
        plt.xticks(ticks=np.arange(first_tick[0],(np.size(temp_times)),60), labels=mins, rotation=270)
        if duration_actual > 48 and duration_actual < 300:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60))
        elif duration_actual > 300:
            pass
        else:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),15))
    
    plt.plot(x,avgmT, linestyle='-', c='k', linewidth=1, label='Mean Temperature')
    plt.plot(x,mT1[:,0], linestyle='-', c='r', linewidth=0.5, label='T Port4')
    plt.plot(x,mT1[:,1], linestyle='-', c='g', linewidth=0.5, label ='T Port2')
    plt.plot(x,mT1[:,2], linestyle='-', c='b', linewidth=0.5, label='Cable 1A S1')
    plt.plot(x,mT2[:,0], linestyle='--', c='r', linewidth=0.5, label='T Port3')
    plt.plot(x,mT2[:,1], linestyle='--', c='g', linewidth=0.5, label='T Port1')
    plt.plot(x,mT2[:,2], linestyle='--', c='b', linewidth=0.5, label='Cable 2A S1')
    
    plt.xticks(rotation = 270)
    
    plt.xlabel('Time')
    plt.ylabel('Temperature / °C')
    plt.margins(x=0)
    plt.tight_layout(h_pad=0.2, w_pad=0.2, rect=[0.117,0.175,0.977,0.922]) #tuple (left, bottom, right, top),
    plt.title(title_string)
    if save_it:
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_OutterBoxTemps.png', bbox_inches="tight")
    else:
        plt.legend(loc="upper right", fontsize=8)
        plt.show()
    plt.close()

    if first_loop:
        print ('')
        save = input ('Do you want to save a printer friendly copy of this plot? (Y/N): ')
        if str(save) == 'N' or str(save) == 'n':
            pass
        elif str(save) == 'Y' or str(save) == 'y':
            save_it=True
            first_loop=False
            mTplot(mT1, mT2, mins, first_tick, duration_actual, save_it, first_loop)
 
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 

#-------------------------------------------------------------------------
    
def hcplot(hc1, hc2, mins, first_tick, duration_actual, save_it=False, first_loop=True):
    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    x = np.arange(0,np.size(temp_times),1)
    
    avgeast = hc1.mean(axis=1)
    avgwest = hc2.mean(axis=1)
    
    hcMean = (avgeast + avgwest) / 2
    
    np.save(DATA_PATH+'/temp/4norm/hc-mean.npy', hcMean)

    title_string = user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours. Horns & Cables, Solid East, Dashed West.'
    if save_it:
        plt.figure(figsize=(12, 8), dpi=300)
    try:
        plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60), labels=mins, rotation=270)
        if duration_actual > 48 and duration_actual < 300:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60))
        elif duration_actual > 300:
            pass
        else:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),15))
    
    except:
        print('')
        print(' \033[0;m Applying x-axis tick mark correction. \033[1;32m ')
        print('')
        mins = mins[1:]
        plt.xticks(ticks=np.arange(first_tick[0],(np.size(temp_times)),60), labels=mins, rotation=270)
        if duration_actual > 48 and duration_actual < 300:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60))
        elif duration_actual > 300:
            pass
        else:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),15))
    
    plt.plot(x,avgeast, linestyle='-', c='k', linewidth=1, label='Mean Temperature')
    plt.plot(x,avgwest, linestyle='--', c='k', linewidth=1)
    
    plt.plot(x,hc1[:,0], linestyle='-', c='r', linewidth=0.5, label='Horn')
    plt.plot(x,hc1[:,1], linestyle='-', c='g', linewidth=0.5, label='Throat')
    plt.plot(x,hc1[:,2], linestyle='-', c='blueviolet', linewidth=0.5, label='Polarizer')
    plt.plot(x,hc1[:,3], linestyle='-', c='b', linewidth=0.5, label='Pol Connect')
    plt.plot(x,hc1[:,4], linestyle='-', c='m', linewidth=0.5, label='Cable S7')
    plt.plot(x,hc1[:,5], linestyle='-', c='y', linewidth=0.5, label='Cable S6')
    plt.plot(x,hc1[:,6], linestyle='-', c='orange', linewidth=0.5, label='Cable S5')
    plt.plot(x,hc1[:,7], linestyle='-', c='lime', linewidth=0.5, label= 'Cable S4')
    plt.plot(x,hc1[:,8], linestyle='-', c='teal', linewidth=0.5, label='Cable S3')
    plt.plot(x,hc1[:,9], linestyle='-', c='c', linewidth=0.5, label='Cable S2')
    
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
    
    plt.xticks(rotation = 270)
    
    plt.xlabel('Time')
    plt.ylabel('Temperature / °C')
    plt.margins(x=0)
    plt.tight_layout(h_pad=0.2, w_pad=0.2, rect=[0.097,0.175,0.977,0.922]) #tuple (left, bottom, right, top),
    plt.title(title_string)
    if save_it:
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_HornCableTemps.png', bbox_inches="tight")
    else:
        plt.legend(loc="upper right", fontsize=8)
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
            hcplot(hc1, hc2, mins, first_tick, duration_actual, save_it, first_loop)
 
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 
        
#---------------------------------------------------------------------

def threeINone(rc1, rc2, mT1, mT2, hc1, hc2, temp_times, mins, first_tick, user_inputs, duration_actual):
    
    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    x = np.arange(0,np.size(temp_times),1)
  
    fig = plt.figure(figsize=(12, 8), dpi=300)
    
    gs = fig.add_gridspec(3, hspace=0) #fig.suptitle('Sharing both axes')
    axs = gs.subplots(sharex=True, sharey=False)
 #   try:
  #      plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60), labels=mins, rotation=270)
   #     if duration_actual > 48 and duration_actual < 300:
    #        plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60))
     #   elif duration_actual > 300:
      #      pass
       # else:
        #    plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),15))
    
  #  except:
   #     print('')
    #    print(' \033[0;m Applying x-axis tick mark correction. \033[1;32m ')
     #   print('')
      #  mins = mins[1:]
       # plt.xticks(ticks=np.arange(first_tick[0],(np.size(temp_times)),60), labels=mins, rotation=270)
        #if duration_actual > 48 and duration_actual < 300:
   #         plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60))
    #    elif duration_actual > 300:
     #       pass
      #  else:
       #     plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),15))
    
    
    #plt.style.use('default')
    #radiometer
    # Hide x labels and tick labels for all but bottom plot.

    time_p1 = Time(temp_times.astype(str), format='iso', scale='utc', precision=4) 

    x = time_p1.tt.datetime

    hrFmt=mdates.DateFormatter('%H:00 (%d)')

   # plt.gca().xaxis.set_major_locator(mdates.HourLocator())
   # plt.gca().xaxis.set_minor_locator(mdates.MinuteLocator(byminute=(15,30,45)))
   # plt.gca().xaxis.set_major_formatter(hrFmt)

    for ax in axs:
        ax.label_outer()
        ax.margins(x=0)
    
    for ax in axs.flat:
        ax.set(xlabel='Time')
        
        #receiver board
        
    axs[0].plot(x,rc1[:,0], linestyle='-', c='r', linewidth=0.5, label='LNA1')
    axs[0].plot(x,rc1[:,1], linestyle='-', c='g', linewidth=0.5, label='LNA2')
    axs[0].plot(x,rc1[:,2], linestyle='-', c='blueviolet', linewidth=0.5, label='LNA3')
    axs[0].plot(x,rc1[:,3], linestyle='-', c='b', linewidth=0.5, label='2534 LNA')
    axs[0].plot(x,rc1[:,4], linestyle='-', c='m', linewidth=0.5, label='Ph/S')
    axs[0].plot(x,rc1[:,5], linestyle='-', c='orange', linewidth=0.5, label='LNA4)')
    
    axs[0].plot(x,rc2[:,0], linestyle='--', dashes=(5, 10), c='r', linewidth=0.5)
    axs[0].plot(x,rc2[:,1], linestyle='--', dashes=(5, 10), c='g', linewidth=0.5)
    axs[0].plot(x,rc2[:,2], linestyle='--', dashes=(5, 10), c='blueviolet', linewidth=0.5)
    axs[0].plot(x,rc2[:,3], linestyle='--', dashes=(5, 10), c='b', linewidth=0.5)
    axs[0].plot(x,rc2[:,4], linestyle='--', dashes=(5, 10), c='m', linewidth=0.5)
    axs[0].plot(x,rc2[:,5], linestyle='--', dashes=(5, 10), c='orange', linewidth=0.5)
    axs[0].legend(bbox_to_anchor=(1.04,1), loc="upper left")
    
    # Magic T
    
    axs[1].plot(x,mT1[:,0], linestyle='-', c='r', linewidth=0.5, label='T Port4')
    axs[1].plot(x,mT1[:,1], linestyle='-', c='g', linewidth=0.5, label ='T Port2')
    axs[1].plot(x,mT1[:,2], linestyle='-', c='b', linewidth=0.5, label ='Cable 1A Sensor1')
    axs[1].plot(x,mT2[:,0], linestyle='--', dashes=(5, 10), c='r', linewidth=0.5, label='T Port3')
    axs[1].plot(x,mT2[:,1], linestyle='--', dashes=(5, 10), c='g', linewidth=0.5, label='T Port1')
    axs[1].plot(x,mT2[:,2], linestyle='--', dashes=(5, 10), c='b', linewidth=0.5, label='Cable 2A Sensor1')
    axs[1].legend(bbox_to_anchor=(1.04,0.5), loc="center left")
    axs[1].set(ylabel='Temperature / °C')
    
    # Horns
    
    axs[2].plot(x,hc1[:,0], linestyle='-', c='r', linewidth=0.5, label='Horn')
    axs[2].plot(x,hc1[:,1], linestyle='-', c='g', linewidth=0.5, label='Throat')
    axs[2].plot(x,hc1[:,2], linestyle='-', c='blueviolet', linewidth=0.5, label='Polarizer')
    axs[2].plot(x,hc1[:,3], linestyle='-', c='b', linewidth=0.5, label = 'Pol Connect')
    axs[2].plot(x,hc1[:,4], linestyle='-', c='m', linewidth=0.5, label='Cable S7')
    axs[2].plot(x,hc1[:,5], linestyle='-', c='y', linewidth=0.5, label='Cable S6')
    axs[2].plot(x,hc1[:,6], linestyle='-', c='orange', linewidth=0.5, label='Cable S5')
    axs[2].plot(x,hc1[:,7], linestyle='-', c='lime', linewidth=0.5, label='Cable S4')
    axs[2].plot(x,hc1[:,8], linestyle='-', c='teal', linewidth=0.5, label='Cable S3')
    axs[2].plot(x,hc1[:,9], linestyle='-', c='c', linewidth=0.5, label='Cable S2')
    
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
    axs[2].legend(bbox_to_anchor=(1.04,0), loc="lower left")
    plt.suptitle(user_inputs[0]+'_'+user_inputs[3]+' for '+str(duration_actual)+' hours. 1Wire Temperature Data')

    plt.tight_layout()
    plt.xticks(rotation=270)
    plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_1Wtemps.png', bbox_inches="tight")
    plt.close()

    print('')
    print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
#------------------------------------------------------------------------
    
def partDeux(rc1, rc2, mT1, mT2, hc1, hc2, temp_times, mins, first_tick, user_inputs, duration_actual):
    
    x = np.arange(0,np.size(temp_times),1)
    
    fig = plt.figure()
    gs = fig.add_gridspec(3, hspace=0) #fig.suptitle('Sharing both axes')
    axs = gs.subplots(sharex=True, sharey=False)
 #   try:
  #      plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60), labels=mins, rotation=270)
   #     if duration_actual > 48 and duration_actual < 300:
    #        plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60))
     #   elif duration_actual > 300:
      #      pass
       # else:
        #    plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),15))
    
 #   except:
  #      print('')
   #     print(' \033[0;m Applying x-axis tick mark correction. \033[1;32m ')
    #    print('')
     #   mins = mins[1:]
     #   plt.xticks(ticks=np.arange(first_tick[0],(np.size(temp_times)),60), labels=mins, rotation=270)
      #  if duration_actual > 48 and duration_actual < 300:
       #     plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60))
 #       elif duration_actual > 300:
  #          pass
   #     else:
    #        plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),15))
    
    
    #plt.style.use('default')

    time_p1 = Time(temp_times.astype(str), format='iso', scale='utc', precision=4) 

    x = time_p1.tt.datetime

    hrFmt=mdates.DateFormatter('%H:00 (%d)')

  #  plt.gca().xaxis.set_major_locator(mdates.HourLocator())
   # plt.gca().xaxis.set_minor_locator(mdates.MinuteLocator(byminute=(15,30,45)))
    #plt.gca().xaxis.set_major_formatter(hrFmt)
    #radiometer
    # Hide x labels and tick labels for all but bottom plot.
    for ax in axs:
        ax.label_outer()
        ax.margins(x=0)
    
    for ax in axs.flat:
        ax.set(xlabel='Time')
        
        #receiver board
        
    axs[0].plot(x,rc1[:,0], linestyle='-', c='r', linewidth=0.5, label='LNA1')
    axs[0].plot(x,rc1[:,1], linestyle='-', c='g', linewidth=0.5, label='LNA2')
    axs[0].plot(x,rc1[:,2], linestyle='-', c='blueviolet', linewidth=0.5, label='LNA3')
    axs[0].plot(x,rc1[:,3], linestyle='-', c='b', linewidth=0.5, label='2534 LNA')
    axs[0].plot(x,rc1[:,4], linestyle='-', c='m', linewidth=0.5, label='Ph/S')
    axs[0].plot(x,rc1[:,5], linestyle='-', c='orange', linewidth=0.5, label='LNA4)')
    
    axs[0].plot(x,rc2[:,0], linestyle='--', dashes=(5, 10), c='r', linewidth=0.5)
    axs[0].plot(x,rc2[:,1], linestyle='--', dashes=(5, 10), c='g', linewidth=0.5)
    axs[0].plot(x,rc2[:,2], linestyle='--', dashes=(5, 10), c='blueviolet', linewidth=0.5)
    axs[0].plot(x,rc2[:,3], linestyle='--', dashes=(5, 10), c='b', linewidth=0.5)
    axs[0].plot(x,rc2[:,4], linestyle='--', dashes=(5, 10), c='m', linewidth=0.5)
    axs[0].plot(x,rc2[:,5], linestyle='--', dashes=(5, 10), c='orange', linewidth=0.5)
    axs[0].legend(loc="upper right", fontsize=8)
    
    # Magic T
    
    axs[1].plot(x,mT1[:,0], linestyle='-', c='r', linewidth=0.5, label='T Port4')
    axs[1].plot(x,mT1[:,1], linestyle='-', c='g', linewidth=0.5, label ='T Port2')
    axs[1].plot(x,mT1[:,2], linestyle='-', c='b', linewidth=0.5, label ='Cable 1A Sensor1')
    axs[1].plot(x,mT2[:,0], linestyle='--', dashes=(5, 10), c='r', linewidth=0.5, label='T Port3')
    axs[1].plot(x,mT2[:,1], linestyle='--', dashes=(5, 10), c='g', linewidth=0.5, label='T Port1')
    axs[1].plot(x,mT2[:,2], linestyle='--', dashes=(5, 10), c='b', linewidth=0.5, label='Cable 2A Sensor1')
    axs[1].legend(loc="center right", fontsize=8)
    axs[1].set(ylabel='Temperature / °C')
    
    # Horns
    
    axs[2].plot(x,hc1[:,0], linestyle='-', c='r', linewidth=0.5, label='Horn')
    axs[2].plot(x,hc1[:,1], linestyle='-', c='g', linewidth=0.5, label='Throat')
    axs[2].plot(x,hc1[:,2], linestyle='-', c='blueviolet', linewidth=0.5, label='Polarizer')
    axs[2].plot(x,hc1[:,3], linestyle='-', c='b', linewidth=0.5, label = 'Pol Connect')
    axs[2].plot(x,hc1[:,4], linestyle='-', c='m', linewidth=0.5, label='Cable S7')
    axs[2].plot(x,hc1[:,5], linestyle='-', c='y', linewidth=0.5, label='Cable S6')
    axs[2].plot(x,hc1[:,6], linestyle='-', c='orange', linewidth=0.5, label='Cable S5')
    axs[2].plot(x,hc1[:,7], linestyle='-', c='lime', linewidth=0.5, label='Cable S4')
    axs[2].plot(x,hc1[:,8], linestyle='-', c='teal', linewidth=0.5, label='Cable S3')
    axs[2].plot(x,hc1[:,9], linestyle='-', c='c', linewidth=0.5, label='Cable S2')
    
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
    axs[2].legend(loc="lower right", fontsize=8)
  
    plt.suptitle(user_inputs[0]+'_'+user_inputs[3]+' for '+str(duration_actual)+' hours. 1Wire Temperature Data')
    plt.tight_layout()
    plt.xticks(rotation=270)
    
    plt.show()
    plt.close

#----------------------------------------------------------------------------

def avgsplot(hc1, hc2,mT1, mT2,rc1, rc2, mins, first_tick, duration_actual, save_it=False, first_loop=True):
    
    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    x = np.arange(0,np.size(temp_times),1)
    
    avgeast = hc1.mean(axis=1)
    avgwest = hc2.mean(axis=1)
    hcMean = (avgeast + avgwest) / 2

    avgrc1 = rc1.mean(axis=1)
    avgrc2 = rc2.mean(axis=1)
    avgrc = (avgrc1+avgrc2)/2

    avgmT1 = mT1.mean(axis=1)
    avgmT2 = mT2.mean(axis=1)
    avgmT = (avgmT1+avgmT2)/2

    title_string = user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours. Mean One-Wire Temperatures.'
    if save_it:
        plt.figure(figsize=(12, 8), dpi=300)
    try:
        plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60), labels=mins, rotation=270)
        if duration_actual > 48 and duration_actual < 300:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60))
        elif duration_actual > 300:
            pass
        else:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),15))
    
    except:
        print('')
        print(' \033[0;m Applying x-axis tick mark correction. \033[1;32m ')
        print('')
        mins = mins[1:]
        plt.xticks(ticks=np.arange(first_tick[0],(np.size(temp_times)),60), labels=mins, rotation=270)
        if duration_actual > 48 and duration_actual < 300:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60))
        elif duration_actual > 300:
            pass
        else:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),15))
    
    plt.plot(x,avgeast, linestyle='-', c='c', linewidth=1, label='East Externals')
    plt.plot(x,avgwest, linestyle='-', c='teal', linewidth=1, label='West Externals')
    plt.plot(x,hcMean, linestyle='-', c='b', linewidth=1, label='All Externals')
    #plt.plot(x,avgmT1, linestyle='-', c='b', linewidth=1, label='Outer Box East Pathway')
    #plt.plot(x,avgmT2, linestyle='--', c='b', linewidth=1, label='West')
    plt.plot(x,avgmT, linestyle='-', c='orange', linewidth=1, label='All Outer Box')
    plt.plot(x,avgrc, linestyle='-', c='r', linewidth=1, label='All Inner Box')
    
    plt.xticks(rotation = 270)
    
    plt.xlabel('Time')
    plt.ylabel('Temperature / °C')
    plt.margins(x=0)
    plt.tight_layout(h_pad=0.2, w_pad=0.2, rect=[0.097,0.175,0.977,0.922]) #tuple (left, bottom, right, top),
    plt.title(title_string)
    if save_it:
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_AverageTemps.png', bbox_inches="tight")
    else:
        plt.legend(loc="upper right", fontsize=8)
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
            avgsplot(hc1, hc2, mins, first_tick, duration_actual, save_it, first_loop)
 
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 
        
#---------------------------------------------------------------------------------------------

def diagram():
    
    print('')
    print('Once the animation has concluded the program will exit.')
    input('Press enter to continue. \033[0;0m')
    print('')
    
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
        

#------------------------------------------------------------------------------
    
def diagramBOX():
    
    print('')
    print('Once the animation has concluded the program will exit.')
    input('Press enter to continue. \033[0;0m')
    print('')
    
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
        
        plt.pause(1)
        

#---------------------------------------------------------------------------------

def TempsMenu(rc1, rc2, hc1, hc2, mT1, mT2, temp_times, user_inputs, duration_actual,temptemp):

    if os.path.exists(user_inputs[6]):
        pass
    else:
        os.system('mkdir '+user_inputs[6])
    
    i = 0
    print('\033[1;32m')
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
    
    if duration_actual > 48:
        i=0
        j=0
        for j in range(0,np.size(mins)):
            for i in range (1,10):
                try:
                    mins[j] = str(mins[j]).replace('0'+str(i)+':00','')
                except:
                    pass
            for i in range (10,12):
                mins[j] = str(mins[j]).replace(str(i)+':00','')
            for i in range (13,24):
                mins[j] = str(mins[j]).replace(str(i)+':00','')

#----------------------------------------------------------------------------------


#----------------------------------------------------------------------------

def avgsplot(hc1, hc2,mT1, mT2,rc1, rc2, mins, first_tick, duration_actual, save_it=False, first_loop=True):
    
    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    x = np.arange(0,np.size(temp_times),1)
    
    avgeast = hc1.mean(axis=1)
    avgwest = hc2.mean(axis=1)
    hcMean = (avgeast + avgwest) / 2

    avgrc1 = rc1.mean(axis=1)
    avgrc2 = rc2.mean(axis=1)
    avgrc = (avgrc1+avgrc2)/2

    avgmT1 = mT1.mean(axis=1)
    avgmT2 = mT2.mean(axis=1)
    avgmT = (avgmT1+avgmT2)/2

    title_string = user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours. Mean One-Wire Temperatures.'
    if save_it:
        plt.figure(figsize=(12, 8), dpi=300)
    try:
        plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60), labels=mins, rotation=270)
        if duration_actual > 48 and duration_actual < 300:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60))
        elif duration_actual > 300:
            pass
        else:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),15))
    
    except:
        print('')
        print(' \033[0;m Applying x-axis tick mark correction. \033[1;32m ')
        print('')
        mins = mins[1:]
        plt.xticks(ticks=np.arange(first_tick[0],(np.size(temp_times)),60), labels=mins, rotation=270)
        if duration_actual > 48 and duration_actual < 300:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),60))
        elif duration_actual > 300:
            pass
        else:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(temp_times),15))
    
    plt.plot(x,avgeast, linestyle='-', c='c', linewidth=1, label='East Externals')
    plt.plot(x,avgwest, linestyle='-', c='teal', linewidth=1, label='West Externals')
    plt.plot(x,hcMean, linestyle='-', c='b', linewidth=1, label='All Externals')
    #plt.plot(x,avgmT1, linestyle='-', c='b', linewidth=1, label='Outer Box East Pathway')
    #plt.plot(x,avgmT2, linestyle='--', c='b', linewidth=1, label='West')
    plt.plot(x,avgmT, linestyle='-', c='orange', linewidth=1, label='All Outer Box')
    plt.plot(x,avgrc, linestyle='-', c='r', linewidth=1, label='All Inner Box')
    
    plt.xticks(rotation = 270)
    
    plt.xlabel('Time')
    plt.ylabel('Temperature / °C')
    plt.margins(x=0)
    plt.tight_layout(h_pad=0.2, w_pad=0.2, rect=[0.097,0.175,0.977,0.922]) #tuple (left, bottom, right, top),
    plt.title(title_string)
    if save_it:
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_HornCableTemps.png', bbox_inches="tight")
    else:
        plt.legend(loc="upper right", fontsize=8)
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
            hcplot(hc1, hc2, mins, first_tick, duration_actual, save_it, first_loop)
 
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 
        
#---------------------------------------------------------------------
    
def TempsMenu(rc1, rc2, hc1, hc2, mT1, mT2, temp_times, user_inputs, duration_actual, temptemp):

    if os.path.exists(user_inputs[6]):
        pass
    else:
        os.system('mkdir '+user_inputs[6])

    i = 0
    print('\033[1;32m')
    mins =[]
    first_tick=[]
    for i in range(0,np.size(temp_times)): #there is only 24 hours in a day!
        x = str(temp_times[i])
        y = x.split()
        yr,mn,day = y[0].split('-')
        h,m,s = y[1].split(':')
        if m == '00' and h != '00': 
            mins.append(h+':'+m)
            first_tick.append(i)
        elif m == '00' and h == '00':
            mins.append(h+':'+m+'  '+day)
            first_tick.append(i)
        else:
            pass
    
    if duration_actual > 48:
        i=0
        j=0
        for j in range(0,np.size(mins)):
            for i in range (1,10):
                try:
                    mins[j] = str(mins[j]).replace('0'+str(i)+':00','')
                except:
                    pass
            for i in range (10,12):
                mins[j] = str(mins[j]).replace(str(i)+':00','')
            for i in range (13,24):
                mins[j] = str(mins[j]).replace(str(i)+':00','')

    print ('-------------------------------------')
    print ('>>      Quicklook 1Wire menu       <<')
    print ('-------------------------------------')
    print ('')
    print ('1 - Horns & Cables')
    print ('')
    print ('2 - Magic T / Outer Box')
    print ('')
    print ('3 - Reciever Board / Inner Box')
    print ('')
    print ('4 - All 1Wire Data')
    print('')
    print ('5 - Average Temperatures Only')
    print('')
    #print('5 - Instrument Diagram Animation')
    #print('')
    #print ('6 - Recevier Board Diagram Animation')
    #print('')
    print('0 - Return to main menu')
    print('')
    choice = input('Select menu option (number): ')
    if choice.isdigit():
        if int(choice) == 1:
            hcplot(hc1, hc2, mins, first_tick, duration_actual)
            
        
        elif int(choice) ==2:
            mTplot(mT1, mT2, mins, first_tick, duration_actual)
            
        
        elif int(choice) ==3:
            rcplot(rc1, rc2, mins, first_tick, temp_times, duration_actual)
           
                
        elif int(choice) ==4:
            
            partDeux(rc1, rc2, mT1, mT2, hc1, hc2, temp_times, mins, first_tick, user_inputs, duration_actual)
            parameters = np.load (DATA_PATH+'/temp/parameters.npy')
            if parameters[5] == 'True':
                print ('')
                save = input ('Do you want to save a printer friendly copy of this plot? (Y/N):')
                if str(save) == 'N' or str(save) == 'n':
                    pass
                elif str(save) == 'Y' or str(save) == 'y':
                    threeINone(rc1, rc2, mT1, mT2, hc1, hc2, temp_times, mins, first_tick, user_inputs, duration_actual)
            else:
                pass
        

        elif int(choice) == 5:
            avgsplot(hc1, hc2,mT1, mT2,rc1, rc2, mins, first_tick, duration_actual)
        
        elif int(choice) ==6:
            diagramBOX()
            temptemp=False
        
        elif int(choice) ==0:
            temptemp=False
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /local5/scratch/pblack/lbass.py')
            pass
        elif int(choice) ==99:
            temptemp=False
            pass
        else:
            print('\033[1;31m No such option. Please try again.\033[1;32m')
    else:
        print('\033[1;31m Invalid selection by user. Please try again. \033[1;32m')
    
    return temptemp
#--------------------------------------------------------------------------------

# load in the user selections and related fits / csv files

all_temps = temps()

getDate=True
while getDate:
    date, go_date, path_string, getDate = selectobs(all_temps, getDate)
    
getTime=True
while getTime:
    user_hour, user_minute, user_sample, time_in, go_time, getTime = selecttime(go_date, all_temps, getTime)

getDuration=True
while getDuration:
    user_duration, getDuration = selectduration(getDuration)

user_inputs = np.array((date, user_hour, user_minute, time_in, user_sample, user_duration, path_string))

selected_temps = temp_selects(user_inputs, go_time, all_temps)

duration_actual = (np.size(selected_temps[:,1]) / 60)
duration_actual = np.round(duration_actual, decimals=2)

if duration_actual < 0.25:
    print ('')
    print ('\033[1;31m Insufficient or missing data. Please try again. \033[1;32m')
    print ('')
    os.system('/usr/local/anaconda-python-3.6/bin/python3 /local5/scratch/pblack/lbass.py')
    pass

else:
    rc1, rc2, hc1, hc2, mT1, mT2, temp_times = groupSensors(duration_actual)

if os.path.exists(path_string):
    pass
else:
    os.system('mkdir '+path_string)

temptemp=True
while temptemp:
    temptemp = TempsMenu(rc1, rc2, hc1, hc2, mT1, mT2, temp_times, user_inputs, duration_actual, temptemp)


#----------------------------------------------------------------------

