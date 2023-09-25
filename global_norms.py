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

os.chdir('/scratch/nas_lbass/raw_data/')

# GOOD CHANNELS ARE 116-606

#--------------------------------------------------------------------------
#Earth rotates at ~ 15 secs of arc per 1 sec of time
#load background files and data tables
user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
frequency = np.load(DATA_PATH+'/temp/freq.npy')
sq_scale = np.load(DATA_PATH+'/temp/sq_scale.npy')

#-------------------------------------------------------------------------
#(date, user_hour, user_minute, user_sample, user_duration)
#file_table(fits_name, begs, ends, start_times, end_times, samFIRSTs, samLASTs, multi_day, same_day, same_run, corrupts)
# the frequency channels within the digital bandpass are 113 to 603
#---------------------------------------------------------------------

def meanChannels():
    
    a1p1B = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
    a1p2B = np.load (DATA_PATH+'/temp/a1p2_binned.npy')
    a2p1B = np.load (DATA_PATH+'/temp/a2p1_binned.npy')
    a2p2B = np.load (DATA_PATH+'/temp/a2p2_binned.npy')
   
    band11 = a1p1B[:,3:].mean(axis=0) #mean each column
    band12 = a1p2B[:,3:].mean(axis=0)
    band21 = a2p1B[:,3:].mean(axis=0)
    band22 = a2p2B[:,3:].mean(axis=0)
    

    return band11, band12, band21, band22
#-----------------------------------------------------------------------
def meanSpectrum():
   
    a1p1B = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
    a1p2B = np.load (DATA_PATH+'/temp/a1p2_binned.npy')
    a2p1B = np.load (DATA_PATH+'/temp/a2p1_binned.npy')
    a2p2B = np.load (DATA_PATH+'/temp/a2p2_binned.npy')
    
    p11 = a1p1B[:,116:606].mean(axis=1) #mean each row
    p12 = a1p2B[:,116:606].mean(axis=1) 
    p21 = a2p1B[:,116:606].mean(axis=1)
    p22 = a2p2B[:,116:606].mean(axis=1)
    
        
    return p11, p12, p21, p22
        
#-----------------------------------------------------------------

def normaliseBandpass(p11,p12,p21,p22,band11,band12,band21,band22):
    
    print('Normalising bandpass')
    
    band11 = band11 / np.mean(p11)
    band12 = band12 / np.mean(p12)
    band21 = band21 / np.mean(p21)
    band22 = band22 / np.mean(p22)

    return band11, band12, band21, band22

#----------------------------------------------------------


def waterfallPERmin(p11):
    
    
    a1p1B = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
    a1p2B = np.load (DATA_PATH+'/temp/a1p2_binned.npy')
    a2p1B = np.load (DATA_PATH+'/temp/a2p1_binned.npy')
    a2p2B = np.load (DATA_PATH+'/temp/a2p2_binned.npy')
    
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    
    z = a1p1B - a1p2B*0.99268377
    q = a2p2B*0.9421594 - a2p1B*0.90373844
 

    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = user_inputs[0]+' from '+user_inputs[3]+ ' for '+str(duration_actual)+' hours. (1min bins)'
    title_string = user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours. (1min bins)'
    
    mins,first_tick, sample_to_datetime, frequency = yaxis(a1p1B, duration_actual)

    f, axarr = plt.subplots(2,2)
       
    plt.setp(axarr, yticks=first_tick, yticklabels=mins)
    
    if duration_actual > 12 and duration_actual < 48:
        plt.setp(axarr, yticks=np.arange(first_tick[0],np.size(sample_to_datetime),60))
    elif duration_actual > 48:
        pass
    else:
        plt.setp(axarr, yticks=np.arange(first_tick[0],np.size(sample_to_datetime),15))
    
    plt.setp(axarr, xticks=np.arange(3,721,102), xticklabels=[frequency[0],'',frequency[204], '', frequency[408], '', frequency[612],''])
    
    axarr[0,0].imshow(a1p1B)
    axarr[0,0].set_title('A1P1')
    axarr[0,0].axvline(x=116,linestyle='--', color='w', linewidth=1.5)
    axarr[0,0].axvline(x=606,linestyle='--', color='w', linewidth=1.5)
   
    
    axarr[0,1].imshow(a1p2B*0.99268377)
    axarr[0,1].set_title('A1P2')
    axarr[0,1].axvline(x=116,linestyle='--', color='w', linewidth=1.5)
    axarr[0,1].axvline(x=606,linestyle='--', color='w', linewidth=1.5)
   

    axarr[1,0].imshow(a2p1B*0.90373844)
    axarr[1,0].set_title('A2P1')
    axarr[1,0].axvline(x=116,linestyle='--', color='w', linewidth=1.5)
    axarr[1,0].axvline(x=606,linestyle='--', color='w', linewidth=1.5)
   

    axarr[1,1].imshow(a2p2B*0.9421594)
    axarr[1,1].set_title('A2P2')
    axarr[1,1].axvline(x=116,linestyle='--', color='w', linewidth=1.5)
    axarr[1,1].axvline(x=606,linestyle='--', color='w', linewidth=1.5)
    
    plt.suptitle(title_string)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    print ('')
    save = input ('Do you want to save a printer friendly copy of this plot? (Y/N):')
    if str(save) == 'N' or str(save) == 'n':
        pass
    elif str(save) == 'Y' or str(save) == 'y':
      
        f, axarr = plt.subplots(2,2, figsize=(12,8), dpi=300)
       
        plt.setp(axarr, yticks=first_tick, yticklabels=mins)
    
        if duration_actual > 12 and duration_actual < 48:
            plt.setp(axarr, yticks=np.arange(first_tick[0],np.size(sample_to_datetime),60))
        elif duration_actual > 48:
            pass
        else:
            plt.setp(axarr, yticks=np.arange(first_tick[0],np.size(sample_to_datetime),15))
    
        plt.setp(axarr, xticks=np.arange(3,721,102), xticklabels=[frequency[0],'',frequency[204], '', frequency[408], '', frequency[612],''])
    
    
        axarr[0,0].imshow(a1p1B)
        axarr[0,0].set_title('A1P1')
        axarr[0,0].axvline(x=116,linestyle='--', color='w', linewidth=1.5)
        axarr[0,0].axvline(x=606,linestyle='--', color='w', linewidth=1.5)
   
    
        axarr[0,1].imshow(a1p2B)
        axarr[0,1].set_title('A1P2')
        axarr[0,1].axvline(x=116,linestyle='--', color='w', linewidth=1.5)
        axarr[0,1].axvline(x=606,linestyle='--', color='w', linewidth=1.5)
   

        axarr[1,0].imshow(a2p1B)
        axarr[1,0].set_title('A2P1')
        axarr[1,0].axvline(x=116,linestyle='--', color='w', linewidth=1.5)
        axarr[1,0].axvline(x=606,linestyle='--', color='w', linewidth=1.5)
   

        axarr[1,1].imshow(a2p2B)
        axarr[1,1].set_title('A2P2')
        axarr[1,1].axvline(x=116,linestyle='--', color='w', linewidth=1.5)
        axarr[1,1].axvline(x=606,linestyle='--', color='w', linewidth=1.5)
    
        plt.suptitle(title_string)
        plt.tight_layout()
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_4xWF-1min-NORMED.png', bbox_inches="tight")
        plt.close()
     
        print('')
        print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
        
    else:
        print('')
        print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
        pass
    
    plotArray2(q,z,mins,first_tick, sample_to_datetime, frequency, duration_actual)
    
    
    
    
def plotArray2(q,z,mins,first_tick, sample_to_datetime,frequency, duration_actual):
    
   user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
   title_string = user_inputs[0]+' from '+user_inputs[3]+ ' for '+str(duration_actual)+' hours. (1min bins)'
   
   fig, (ax1, ax2) = plt.subplots(1, 2)
   
   plt.setp(ax1, yticks=first_tick, yticklabels=mins)
   plt.setp(ax2, yticks=first_tick, yticklabels=mins)

   plt.setp(ax1, xticks=np.arange(3,721,102), xticklabels=[frequency[0],'',frequency[204], '', frequency[408], '', frequency[612],''])
   plt.setp(ax2, xticks=np.arange(3,721,102), xticklabels=[frequency[0],'',frequency[204], '', frequency[408], '', frequency[612],''])

   if duration_actual > 12 and duration_actual < 48:
       plt.setp(ax1, yticks=np.arange(first_tick[0],np.size(sample_to_datetime),60))
       plt.setp(ax2, yticks=np.arange(first_tick[0],np.size(sample_to_datetime),60))
   elif duration_actual > 48:
       pass
   else:
       plt.setp(ax1, yticks=np.arange(first_tick[0],np.size(sample_to_datetime),15))
       plt.setp(ax2, yticks=np.arange(first_tick[0],np.size(sample_to_datetime),15))
   
   ax1.imshow(z)
   ax1.set_title('A1P1 - A1P2')
   ax1.axvline(x=116, linestyle='--', color='w', linewidth=1.5)
   ax1.axvline(x=606,linestyle='--', color='w', linewidth=1.5)
   ax2.imshow(q)
   ax2.set_title('A2P2 - A2P1')
   ax2.axvline(x=116,linestyle='--', color='w', linewidth=1.5)
   ax2.axvline(x=606,linestyle='--', color='w', linewidth=1.5)
   plt.tight_layout()
   plt.suptitle(title_string)
   plt.show()
   plt.close()
   
   print ('')
   save = input ('Do you want to save a printer friendly copy of this plot? (Y/N):')
   if str(save) == 'N' or str(save) == 'n':
       pass
   elif str(save) == 'Y' or str(save) == 'y': 
        
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8), dpi=300)
   
       plt.setp(ax1, yticks=first_tick, yticklabels=mins)
       plt.setp(ax2, yticks=first_tick, yticklabels=mins)

       plt.setp(ax1, xticks=np.arange(3,721,102), xticklabels=[frequency[0],'',frequency[204], '', frequency[408], '', frequency[612],''])
       plt.setp(ax2, xticks=np.arange(3,721,102), xticklabels=[frequency[0],'',frequency[204], '', frequency[408], '', frequency[612],''])
   
       if duration_actual > 12 and duration_actual < 48:
           plt.setp(ax1, yticks=np.arange(first_tick[0],np.size(sample_to_datetime),60))
           plt.setp(ax2, yticks=np.arange(first_tick[0],np.size(sample_to_datetime),60))
       elif duration_actual > 48:
           pass
       else:
           plt.setp(ax1, yticks=np.arange(first_tick[0],np.size(sample_to_datetime),15))
           plt.setp(ax2, yticks=np.arange(first_tick[0],np.size(sample_to_datetime),15))

       ax1.imshow(z)
       ax1.set_title('A1P1 - A1P2')
       ax1.axvline(x=116, linestyle='--', color='w', linewidth=1.5)
       ax1.axvline(x=606,linestyle='--', color='w', linewidth=1.5)
       ax2.imshow(q)
       ax2.set_title('A2P2 - A2P1')
       ax2.axvline(x=116,linestyle='--', color='w', linewidth=1.5)
       ax2.axvline(x=606,linestyle='--', color='w', linewidth=1.5)
   
   
       plt.suptitle(title_string)
       plt.tight_layout()
       plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_2xWF-1min-NORMED.png', bbox_inches="tight")
       plt.close()
        
        
       print('')
       print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
        
   else:
      print('')
      print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
      pass
        

def firstDiff():
    
    a1p1B = np.load (DATA_PATH+'/temp/4norm/a1p1_binned.npy')
    
    p11 = np.load (DATA_PATH+'/temp/4norm/p11.npy')
    p12 = np.load (DATA_PATH+'/temp/4norm/p12.npy')
    p21 = np.load (DATA_PATH+'/temp/4norm/p21.npy')
    p22 = np.load (DATA_PATH+'/temp/4norm/p22.npy')
    
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours. Normalised Power First Differences - with offset'
    
    mins,first_tick, sample_to_datetime, frequency = xaxis(a1p1B, duration_actual)
    
    plt.xticks(ticks=first_tick, labels=mins, rotation=270)
    if duration_actual > 48 and duration_actual < 150:
        plt.xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),60))
    elif duration_actual > 150:
        pass
    else:
        plt.xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),15))
        
    firstDiff1 = p11 - p12
    firstDiff2 = p22 - p21
    firstDiff3 = p11 - p21
    firstDiff4 = p22 - p12   
    
    plt.plot(firstDiff1, c='r', label='A1P1 - A1P2')
    plt.plot(firstDiff2, c='g', label='A2P2 - A2P1')
    plt.plot(firstDiff3, c='b', label='A1P1 - A2P1')
    plt.plot(firstDiff4, c='c', label='A2P2 - A1P2')
    
    plt.tight_layout()
    plt.title(title_string)
    plt.legend(loc="upper right", fontsize=8)
    plt.margins(x=0)
    plt.show()
    plt.close()
    
    print ('')
    save = input ('Do you want to save a printer friendly copy of this plot? (Y/N):')
    if str(save) == 'N' or str(save) == 'n':
        pass
    elif str(save) == 'Y' or str(save) == 'y':
        
        plt.figure(figsize=(12, 8), dpi=300)
        plt.xticks(ticks=first_tick, labels=mins, rotation=270)
        if duration_actual > 48 and duration_actual < 150:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),60))
        elif duration_actual > 150:
            pass
        else:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),15))
        
        firstDiff1 = p11 - p12
        firstDiff2 = p22 - p21
        firstDiff3 = p11 - p21
        firstDiff4 = p22 - p12  
    
        plt.plot(firstDiff1, c='r', label='A1P1 - A1P2')
        plt.plot(firstDiff2, c='g', label='A2P2 - A2P1')
        plt.plot(firstDiff3, c='b', label='A1P1 - A2P1')
        plt.plot(firstDiff4, c='c', label='A2P2 - A1P2')
    
        plt.title(title_string)
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.margins(x=0)
        plt.tight_layout() #tuple (left, bottom, right, top),)

        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_first_diffs-normed-with-offset.png', bbox_inches="tight")
        plt.close()

        print('')
        print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
  
    else:
        print('')
        print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
        pass
    
    
#-----------------------------------------------------------------    
    

def rawData():
    
    a1p1B = np.load (DATA_PATH+'/temp/4norm/a1p1_binned.npy')
    p11 = a1p1B[:,116:606].mean(axis=1)
    
    a1p2B = np.load (DATA_PATH+'/temp/4norm/a1p2_binned.npy')
    p12 = a1p2B[:,116:606].mean(axis=1) 

    
    a2p1B = np.load (DATA_PATH+'/temp/4norm/a2p1_binned.npy')
    p21 = a2p1B[:,116:606].mean(axis=1)

    
    a2p2B = np.load (DATA_PATH+'/temp/4norm/a2p2_binned.npy')
    p22 = a2p2B[:,116:606].mean(axis=1)


    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)

    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours. Normalised Raw Data, with offset (Power Signal)'
    
    mins,first_tick, sample_to_datetime, frequency = xaxis(a1p1B, duration_actual)

    #load first
    plt.xticks(ticks=first_tick, labels=mins, rotation=270)
    if duration_actual > 48 and duration_actual < 150:
        plt.xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),60))
    elif duration_actual > 150:
        pass
    else:
        plt.xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),15))
    
    
  
    
    
    #p11 = p11 / np.mean(p11)
    #p12 = p12 / np.mean(p12)
    #p21 = p21 / np.mean(p21)
    #p22 = p22 / np.mean(p22)
    
    # normalises the scale height 0-1
    
    x1 = (np.mean(p11) + np.mean(p22)) /2
    x2 = (np.mean(p12) + np.mean(p21)) /2
    Diffx = (x2 + x1) / 2
    
    
    p21 = p21/1.42
    p12 = p12/1.3
    p22 = p22/1.05
    
    #fit by eye
    print ('p12 offset',((p11[0] - p12[0])))
    print ('p21 offset',(p11[0] - p21[0]))
    print ('p22 offset',(p11[0] - p22[0]))

    p12 = p12 - (np.mean(p12) - np.mean(p11))
    p21 = p21 - (np.mean(p21) - np.mean(p11))
    p22 = p22 - (np.mean(p22) - np.mean(p11))
    
    #condenses plots (removes gaps in power signal)
    
    #p12 = p12 + (p11[0] - p12[0])
    #p22 = p22 + (p11[0] - p22[0])
    #p21 = p21 + (p11[0] - p21[0])
    
    #bring initial values in line
    
    np.save(DATA_PATH+'/temp/4norm/p11.npy', p11)
    np.save(DATA_PATH+'/temp/4norm/p12.npy', p12)
    np.save(DATA_PATH+'/temp/4norm/p21.npy', p21)
    np.save(DATA_PATH+'/temp/4norm/p22.npy', p22)
    
    print ('p11 open to fit ratio',(np.mean(p11) / np.mean(a1p1B[:,116:606].mean(axis=1))))
    print ('p12 open to fit ratio',(np.mean(p12) / np.mean(a1p2B[:,116:606].mean(axis=1))))
    print ('p21 open to fit ratio',(np.mean(p21) / np.mean(a2p1B[:,116:606].mean(axis=1))))
    print ('p22 open to fit ratio',(np.mean(p22) / np.mean(a2p2B[:,116:606].mean(axis=1))))
    print('')
    print ('p11 open to p11 fit ratio',(np.mean(p11) / np.mean(a1p1B[:,116:606].mean(axis=1))))
    print ('p12 open to p11 fit ratio',(np.mean(p11) / np.mean(a1p2B[:,116:606].mean(axis=1))))
    print ('p21 open to p11 fit ratio',(np.mean(p11) / np.mean(a2p1B[:,116:606].mean(axis=1))))
    print ('p22 open to p11 fit ratio',(np.mean(p11) / np.mean(a2p2B[:,116:606].mean(axis=1))))

    
    plt.plot(p11, color='b', label='A1P1')
    plt.plot(p12, color='c', label='A1P2')
    plt.plot(p21, color='r', label='A2P1')
    plt.plot(p22, color='m', label='A2P2')
    
    plt.legend(loc="upper right", fontsize=8)
    plt.title(title_string)
    plt.tight_layout()
    plt.margins(x=0)
    plt.show()
    plt.close()
    
    #save if want
    
    print ('')
    save = input ('Do you want to save a printer friendly copy of this plot? (Y/N):')
    if str(save) == 'N' or str(save) == 'n':
        pass
    elif str(save) == 'Y' or str(save) == 'y':
        
        plt.figure(figsize=(12, 8), dpi=300)

        plt.xticks(ticks=first_tick, labels=mins, rotation=270)
        if duration_actual > 48 and duration_actual < 150:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),60))
        elif duration_actual > 150:
            pass
        else:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),15))
    
        plt.plot(p11, color='b', label='A1P1')
        plt.plot(p12, color='c', label='A1P2')
        plt.plot(p21, color='r', label='A2P1')
        plt.plot(p22, color='m', label='A2P2')
        
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.title(title_string)
        plt.tight_layout()
        plt.margins(x=0)
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_normed_with-offset_PS.png', bbox_inches="tight")
        plt.close()
    
        print('')
        print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
        
    else:
        print('')
        print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
        pass

    
  
#-------------------------------------------------------------------
 
def doubleDiff():
    
    a1p1B = np.load (DATA_PATH+'/temp/4norm/a1p1_binned.npy')
    
    p11 = np.load (DATA_PATH+'/temp/4norm/p11.npy')
    p12 = np.load (DATA_PATH+'/temp/4norm/p12.npy')
    p21 = np.load (DATA_PATH+'/temp/4norm/p21.npy')
    p22 = np.load (DATA_PATH+'/temp/4norm/p22.npy')
    
    
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours. Normalised Power Double Differences - with offset'
    
    mins,first_tick, sample_to_datetime, frequency = xaxis(a1p1B, duration_actual)

    firstDiff1 = p11 - p12
    firstDiff2 = p22 - p21
    firstDiff3 = p11 - p21
    firstDiff4 = p22 - p12
    
    doubleDiff = ( firstDiff3 + firstDiff4 ) / 2
    null = ( firstDiff3 - firstDiff4 ) / 2
    
    null = null + (doubleDiff[0] - null[0])
    
    plt.xticks(ticks=first_tick, labels=mins, rotation=270)
    if duration_actual > 48 and duration_actual < 150:
        plt.xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),60))
    elif duration_actual > 150:
        pass
    else:
        plt.xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),15))
        
    plt.plot(null, c='grey', linestyle='-', linewidth=0.5, label ='Null')
    plt.plot(doubleDiff, c='r', linestyle='-', label='Sky Signal')

    plt.margins(x=0)
    plt.title(title_string)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    print ('')
    save = input ('Do you want to save a printer friendly copy of this plot? (Y/N):')
    if str(save) == 'N' or str(save) == 'n':
        pass
    elif str(save) == 'Y' or str(save) == 'y':
    
        plt.figure(figsize=(12, 8), dpi=300)
        plt.xticks(ticks=first_tick, labels=mins, rotation=270)
        if duration_actual > 48 and duration_actual < 150:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),60))
        elif duration_actual > 150:
            pass
        else:
            plt.xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),15))

        plt.plot(null, c='grey', linestyle='-', linewidth=0.5, label ='Null')
        plt.plot(doubleDiff, c='r', linestyle='-', label='Sky Signal')
    
        plt.margins(x=0)
        plt.title(title_string)
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.tight_layout() #tuple (left, bottom, right, top),

        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_double_diffs_normed-with-offset.png', bbox_inches="tight")
        plt.close()
    
        print('')
        print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
        
    else:
        print('')
        print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
        pass
    
#---------------------------------------------------------------------
    
def xaxis(range_example, duration_actual): #range example being a1p1B etc
    
     
    p = np.load(DATA_PATH+'/temp/run_start_date.npy', allow_pickle=True)
    run_start_date = Time(str(p), out_subfmt='date_hms')
    re_date = np.load(DATA_PATH+'/temp/re_date.npy')
    obsheader = np.load(DATA_PATH+'/temp/obshdr.npy')
    mid_obs = Time(obsheader[0,7],format='isot', scale='utc', precision=0, out_subfmt='date')
    obs_date = Time(str(mid_obs),format='iso', scale='utc', precision=0, out_subfmt='date_hms')
    freq = np.load(DATA_PATH+'/temp/freq.npy')
    jeff = (freq * 10**-9)
    frequency = np.around(jeff, decimals =3)
    
    #bin length assumed to be one mnute - make that adjustable
    
    
    sample_to_datetime = []
    i=0
    
    if re_date: #this means sample time restarts at 0
        for i in range (0,np.size(range_example[:,0])):
            sb = obs_date + TimeDelta(float(range_example[i,0]), format='sec')
            r = sb.strftime("%H:%M")   
            sample_to_datetime.append(r)
        
    else:
        for i in range (0,np.size(range_example[:,0])): #this means sample time is days long
            sb = run_start_date + TimeDelta(float(range_example[i,0]), format='sec')
            r = sb.strftime("%H:%M")
            sample_to_datetime.append(r)
        
    mins =[]
    first_tick=[]
    for i in range(0,np.size(sample_to_datetime)): #there is only 24 hours in a day!
        x = str(sample_to_datetime[i])
        bb = x.strip("'")
        h,m = bb.split(':')
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

    return mins, first_tick, sample_to_datetime, frequency
    

#-------------------------------------------------------------------------------------

def PlotMenu(plotmenu, duration_actual):

    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    
    if os.path.exists(user_inputs[6]):
        pass
    else:
        os.system('mkdir '+user_inputs[6])
    
    sq_scale = np.load(DATA_PATH+'/temp/sq_scale.npy')
    
    if float(duration_actual) > 24:
        warnlen='\033[1;31m (dataset too large) \033[1;32m'
    elif float(duration_actual) < 0.25:
        warnlen='\033[1;31m (dataset too small) \033[1;32m'
    else:
        warnlen=''

    print ('')
    print ('   -------------------------------------')
    print ('           Quicklook plots menu   ')
    print ('   -------------------------------------')
    print ('')
    print ('   1 - Bandpass averaged over', duration_actual,'hours')
    print ('')
    print ('   2 - Bandpass stacked')
    print ('')
    print ('   3 - Waterfall plot (',np.round(sq_scale,decimals=2),'minutes per bin)', warnlen)
    print ('')
    print ('   4 - Waterfall plot',warnlen)
    print('')
    print('   5 - Raw Power data')
    print('')
    print ('   6 - Single Difference')
    print ('')
    print ('   7 - Double Difference')
    print('')
    print('   0 - Return to main menu')
    print('')
    choice = input('Select menu option (number): ')
    if choice.isdigit():
        if int(choice) == 1:
            FourPlots(p11, frequency, band11, band12, band21, band22)   
    
        elif int(choice) ==2:
            FourPlotsStacked(frequency) 
        
        elif int(choice) ==3:
            binIN718s()
            waterfallrebinned718(p11)
                
        elif int(choice) ==4:
            waterfallPERmin(p11)
        
        elif int(choice) ==5:
            rawData()
            
        elif int(choice) ==6:
            firstDiff()
        
        elif int(choice) ==7:
            doubleDiff()
        
        elif int(choice) ==0:
            plotmenu=False
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /local5/scratch/pblack/lbass.py')
            pass
    
        elif int(choice) ==99:
            plotmenu=False
            print ('\033[0;m' )
            pass
        else:
            print('\033[1;31m No such option. Please try again.\033[1;32m')
    else:
        print('\033[1;31m Invalid selection by user. Please try again. \033[1;32m')
        

    return plotmenu

#-----------------------------------------------------------------


band11, band12, band21, band22 = meanChannels()    
p11, p12, p21, p22 = meanSpectrum()
duration_actual = (np.size(p11) / 60)
duration_actual = np.round(duration_actual, decimals=2)
normaliseBandpass(p11,p12,p21,p22,band11,band12,band21,band22)

np.save(DATA_PATH+'/temp/band11-4norm',band11)
np.save(DATA_PATH+'/temp/band12-4norm',band12)
np.save(DATA_PATH+'/temp/band21-4norm',band21)
np.save(DATA_PATH+'/temp/band22-4norm',band22)


print ('\033[1;32m ')

plotmenu=True
while plotmenu:
    plotmenu = PlotMenu(plotmenu, duration_actual)





