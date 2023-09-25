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


def ratios():
    a1p1B = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
    p11 = a1p1B[:,116:606].mean(axis=1)
    a1p2B = np.load (DATA_PATH+'/temp//a1p2_binned.npy')
    p12 = a1p2B[:,116:606].mean(axis=1) 
    a2p1B = np.load (DATA_PATH+'/temp//a2p1_binned.npy')
    p21 = a2p1B[:,116:606].mean(axis=1) 
    a2p2B = np.load (DATA_PATH+'/temp//a2p2_binned.npy')
    p22 = a2p2B[:,116:606].mean(axis=1)


    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    
    #ratios for each channel
    #pair one is 11 and 22
    p11vp11 = p11 / p11 #baseline = 1
    p22vp11 = p22 / p11
    #pair two is 12 and 21
    p12vp12 = p12 / p12 #baselines = 1
    p21vp12 = p21 / p12
    
    #all rations against a single channel
    #p11vp11 = p11 / p11 #= baseline = 1
    #p22vp11 = p22 / p11
    p12vp11 = p12 / p11
    p21vp11 = p21 / p11
    #---------
    p12vp12 = p12 / p12 #baselines = 1
    p21vp12 = p21 / p12
    
    #all rations against a single channel
    p11vp11 = p11 / p11 #= baseline = 1
    p22vp11 = p22 / p11
    p11vp12 = p11 / p12
    p22vp12 = p22 / p12
    

    #p11 = p11 / np.mean(p11)
    #p12 = p12 / np.mean(p12)
    #p21 = p21 / np.mean(p21)
    #p22 = p22 / np.mean(p22)
    
    # normalises the scale height 0-1
    
    #x1 = (np.mean(p11) + np.mean(p22)) /2
    #x2 = (np.mean(p12) + np.mean(p21)) /2
    #Diffx = (x2 + x1) / 2

    p11n = p11/np.mean(p11vp11)
    p21n = p21/np.mean(p21vp11)
    p12n = p12/np.mean(p12vp11)
    p22n = p22/np.mean(p22vp11)

    print (np.mean(p11vp11))
    print (np.mean(p12vp11))
    print (np.mean(p21vp11))
    print (np.mean(p22vp11))
    
    #fit by means
    #print ('p12 offset',((p11[0] - p12[0])))
    #print ('p21 offset',(p11[0] - p21[0]))
    #print ('p22 offset',(p11[0] - p22[0]))

    #p12 = p12 - (np.mean(p12) - np.mean(p11))
    #p21 = p21 - (np.mean(p21) - np.mean(p11))
    #p22 = p22 - (np.mean(p22) - np.mean(p11))
    
    #condenses plots (removes gaps in power signal)
    
    #p12 = p12 + (p11[0] - p12[0])
    #p22 = p22 + (p11[0] - p22[0])
    #p21 = p21 + (p11[0] - p21[0])
    
    #bring initial values in line
    
    np.save(DATA_PATH+'/temp/4norm/p11n.npy', p11n)
    np.save(DATA_PATH+'/temp/4norm/p12n.npy', p12n)
    np.save(DATA_PATH+'/temp/4norm/p21n.npy', p21n)
    np.save(DATA_PATH+'/temp/4norm/p22n.npy', p22n)
    
    np.save(DATA_PATH+'/temp/4norm/p11.npy', p11)
    np.save(DATA_PATH+'/temp/4norm/p12.npy', p12)
    np.save(DATA_PATH+'/temp/4norm/p21.npy', p21)
    np.save(DATA_PATH+'/temp/4norm/p22.npy', p22)
    
    np.save(DATA_PATH+'/temp/4norm/p11vp11.npy', p11vp11) #
    np.save(DATA_PATH+'/temp/4norm/p22vp11.npy', p22vp11) #
   # np.save(DATA_PATH+'/temp/4norm/p12vp12.npy', p12vp12)
    #np.save(DATA_PATH+'/temp/4norm/p21vp12.npy', p21vp12)
    np.save(DATA_PATH+'/temp/4norm/p12vp11.npy', p12vp11) #
    np.save(DATA_PATH+'/temp/4norm/p21vp11.npy', p21vp11) #

def ratiosUP():
    
    #scale up to A2P1
    
    a1p1B = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
    p11 = a1p1B[:,116:606].mean(axis=1)
    a1p2B = np.load (DATA_PATH+'/temp//a1p2_binned.npy')
    p12 = a1p2B[:,116:606].mean(axis=1) 
    a2p1B = np.load (DATA_PATH+'/temp//a2p1_binned.npy')
    p21 = a2p1B[:,116:606].mean(axis=1) 
    a2p2B = np.load (DATA_PATH+'/temp//a2p2_binned.npy')
    p22 = a2p2B[:,116:606].mean(axis=1)


    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    
    #ratios for each channel
    #pair one is 11 and 22
    p11vp11 = p11 / p11 #baseline = 1
    p22vp11 = p22 / p11
    #pair two is 12 and 21
    p12vp12 = p12 / p12 #baselines = 1
    p21vp12 = p21 / p12
    
    #all rations against a single channel
    #p11vp11 = p11 / p11 #= baseline = 1
    #p22vp11 = p22 / p11
    p12vp21 = p12 / p21
    p21vp21 = p21 / p21 #baseline
    p22vp21 = p22 / p21
    p11vp21 = p11 / p21

    #p11 = p11 / np.mean(p11)
    #p12 = p12 / np.mean(p12)
    #p21 = p21 / np.mean(p21)
    #p22 = p22 / np.mean(p22)
    
    # normalises the scale height 0-1
    
    #x1 = (np.mean(p11) + np.mean(p22)) /2
    #x2 = (np.mean(p12) + np.mean(p21)) /2
    #Diffx = (x2 + x1) / 2
    
    
    p11u = p11/np.mean(p11vp21)
    p12u = p12/np.mean(p12vp21)
    p22u = p22/np.mean(p22vp21)
    
    #fit by means
 
    #p12 = p12 - (np.mean(p12) - np.mean(p11))
    #p21 = p21 - (np.mean(p21) - np.mean(p11))
    #p22 = p22 - (np.mean(p22) - np.mean(p11))
    
    #condenses plots (removes gaps in power signal)
    
    #p12 = p12 + (p11[0] - p12[0])
    #p22 = p22 + (p11[0] - p22[0])
    #p21 = p21 + (p11[0] - p21[0])
    
    #bring initial values in line
    
    np.save(DATA_PATH+'/temp/4norm/p11u.npy', p11u)
    np.save(DATA_PATH+'/temp/4norm/p12u.npy', p12u)
    np.save(DATA_PATH+'/temp/4norm/p21u.npy', p21)
    np.save(DATA_PATH+'/temp/4norm/p22u.npy', p22u)
    
    np.save(DATA_PATH+'/temp/4norm/p11.npy', p11)
    np.save(DATA_PATH+'/temp/4norm/p12.npy', p12)
    np.save(DATA_PATH+'/temp/4norm/p21.npy', p21)
    np.save(DATA_PATH+'/temp/4norm/p22.npy', p22)
    
    np.save(DATA_PATH+'/temp/4norm/p11vp11.npy', p11vp11)
    np.save(DATA_PATH+'/temp/4norm/p22vp11.npy', p22vp11)
    np.save(DATA_PATH+'/temp/4norm/p12vp12.npy', p12vp12)
    np.save(DATA_PATH+'/temp/4norm/p21vp12.npy', p21vp12)
    np.save(DATA_PATH+'/temp/4norm/p12vp21.npy', p12vp21)
    np.save(DATA_PATH+'/temp/4norm/p21vp21.npy', p21vp21)
    np.save(DATA_PATH+'/temp/4norm/p11vp21.npy', p11vp21)
    np.save(DATA_PATH+'/temp/4norm/p22vp21.npy', p22vp21)
    

def showMeans():
    
    p11 = np.load(DATA_PATH+'/temp/4norm/p11.npy')
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    
    #p11vp11 = np.load(DATA_PATH+'/temp/4norm/p11vp11.npy')
    p22vp11 = np.load(DATA_PATH+'/temp/4norm/p22vp11.npy')
    #p12vp12 = np.load(DATA_PATH+'/temp/4norm/p12vp12.npy')
    p21vp12 = np.load(DATA_PATH+'/temp/4norm/p21vp12.npy')
    p12vp11 = np.load(DATA_PATH+'/temp/4norm/p12vp11.npy')
    p21vp11 = np.load(DATA_PATH+'/temp/4norm/p21vp11.npy')
    
    print('Over a period of',duration_actual,'hours:')
    print('')
    print('Mean ratio of phases per channel:')
    print('  West:')
    print ('    A2P2 / A1P1 =',np.mean(p22vp11))
    print('  East:')
    print ('    A2P1 / A1P2 =',np.mean(p21vp12))
    print('')
    print('Mean ratio per input (all against A1P1):')
    print ('    A1P2 / A1P1 =',np.mean(p12vp11))
    print ('    A2P1 / A1P1 =',np.mean(p21vp11))
    print ('    A2P2 / A1P1 =',np.mean(p22vp11))
    print('')
    input('Press enter to return to Normalisation Menu')
    print('')
    
def PlotChans():   
    
    a1p1B = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
    p11 = np.load(DATA_PATH+'/temp/4norm/p11.npy')
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    
    p11vp11 = np.load(DATA_PATH+'/temp/4norm/p11vp11.npy')
    p22vp11 = np.load(DATA_PATH+'/temp/4norm/p22vp11.npy')
    p12vp12 = np.load(DATA_PATH+'/temp/4norm/p12vp12.npy')
    p21vp12 = np.load(DATA_PATH+'/temp/4norm/p21vp12.npy')
    p12vp11 = np.load(DATA_PATH+'/temp/4norm/p12vp11.npy')
    p21vp11 = np.load(DATA_PATH+'/temp/4norm/p21vp11.npy')
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = user_inputs[0]+' from '+user_inputs[3]+ ' for '+str(duration_actual)+' hours. Ratio between phases in each channel.'
    mins,first_tick, sample_to_datetime, frequency = xaxis(a1p1B, duration_actual)
    
    plt.close()
    fig, (ax1, ax2) = plt.subplots(1, 2)
   
    #plt.figure(figsize=(12, 8), dpi=300)
    
    #plt.setp(ax1, xticks=first_tick, xticklabels=mins)
    #plt.setp(ax2, xticks=first_tick, xticklabels=mins)
    ax1.set_xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),60))
    ax1.set_xticklabels(mins, rotation=270)
    ax1.plot(p11vp11, color='b', label='A1P1 (Baseline)')
    ax1.plot(p22vp11, color='m', label='A2P2/A1P1')
    #ax1.text('Mean ratio = '+str(np.mean(p22vp11)))
    ax1.set_title('West A2P2/A1P1. Mean ratio = '+str(np.mean(p22vp11)))
    ax1.legend()
    ax1.margins(x=0)
    
    ax2.plot(p12vp12, color='c', label='A1P2 (Baseline)')
    ax2.plot(p21vp12, color='r', label='A2P1/A1P2')
    #ax1.text('Mean ratio = '+str(np.mean(p21vp12)))
    ax2.set_xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),60))
    ax2.set_xticklabels(mins, rotation=270)
    ax2.set_title('East A2P1/A1P2. Mean ratio = '+str(np.mean(p21vp12)))
    ax2.legend()
    ax2.margins(x=0)
    
    plt.tight_layout()
    plt.suptitle(title_string)
    plt.show()
    
def PlotAll():
    
    a1p1B = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
    p11 = np.load(DATA_PATH+'/temp/4norm/p11.npy')
    
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    
    p11vp11 = np.load(DATA_PATH+'/temp/4norm/p11vp11.npy')
    p22vp11 = np.load(DATA_PATH+'/temp/4norm/p22vp11.npy')
    p12vp12 = np.load(DATA_PATH+'/temp/4norm/p12vp12.npy')
    p21vp12 = np.load(DATA_PATH+'/temp/4norm/p21vp12.npy')
    p12vp11 = np.load(DATA_PATH+'/temp/4norm/p12vp11.npy')
    p21vp11 = np.load(DATA_PATH+'/temp/4norm/p21vp11.npy')
    
    title_string = user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours. Ratio of all inputs against A1P1.'
    
    mins,first_tick, sample_to_datetime, frequency = xaxis(a1p1B, duration_actual)
    
    plt.close()
    #load first
    plt.xticks(ticks=first_tick, labels=mins, rotation=270)
    if duration_actual > 48 and duration_actual < 150:
        plt.xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),60))
    elif duration_actual > 150:
        pass
    else:
        plt.xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),15))
    
    plt.plot(p11vp11, color='b', label='A1P1. Baseline')
    plt.plot(p12vp11, color='c', label='A1P2/A1P1. Mean ratio = '+str(np.mean(p12vp11)))
    plt.plot(p21vp11, color='r', label='A2P1/A1P1. Mean ratio = '+str(np.mean(p21vp11)))
    plt.plot(p22vp11, color='m', label='A2P2/A1P1. Mean ratio = '+str(np.mean(p22vp11)))
    
    plt.legend(loc="upper right", fontsize=8)
    plt.title(title_string)
    plt.tight_layout()
    plt.margins(x=0)
    plt.show()
    
    
    #plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_ratios_PS.png', bbox_inches="tight")

def ApplyNorm():   
    
    a1p1B = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
    p11 = a1p1B[:,116:606].mean(axis=1)
    a1p2B = np.load (DATA_PATH+'/temp//a1p2_binned.npy')
    p12 = a1p2B[:,116:606].mean(axis=1) 
    a2p1B = np.load (DATA_PATH+'/temp//a2p1_binned.npy')
    p21 = a2p1B[:,116:606].mean(axis=1) 
    a2p2B = np.load (DATA_PATH+'/temp//a2p2_binned.npy')
    p22 = a2p2B[:,116:606].mean(axis=1)
    
    p11n = np.load(DATA_PATH+'/temp/4norm/p11n.npy')
    p12n = np.load(DATA_PATH+'/temp/4norm/p12n.npy')
    p21n = np.load(DATA_PATH+'/temp/4norm/p21n.npy')
    p22n = np.load(DATA_PATH+'/temp/4norm/p22n.npy')
    
    print(p12n)
    print(np.mean(p12n))

    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)

    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = user_inputs[0]+' from '+user_inputs[3]+ ' for '+str(duration_actual)+' hours. Raw Power vs Normalised Power.'
    mins,first_tick, sample_to_datetime, frequency = xaxis(a1p1B, duration_actual)
    
    plt.close()
    fig, (ax1, ax2) = plt.subplots(1, 2)
   
    #plt.figure(figsize=(12, 8), dpi=300)
    
    #plt.setp(ax1, xticks=first_tick, xticklabels=mins)
    #plt.setp(ax2, xticks=first_tick, xticklabels=mins)
    ax1.set_xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),60))
    ax1.set_xticklabels(mins, rotation=270)
    ax1.plot(p11, color='b', label='A1P1')
    ax1.plot(p12, color='salmon', label='A1P2')
    ax1.plot(p21, color='r', label='A2P1')
    ax1.plot(p22, color='c', label='A2P2')
    #ax1.text('Mean ratio = '+str(np.mean(p22vp11)))
    ax1.set_title('Raw Power Data')
    ax1.legend()
    ax1.margins(x=0)
    
    ax2.plot(p11n, color='b', label='A1P1n')
    ax2.plot(p12n, color='salmon', label='A1P2n')
    ax2.plot(p21n, color='r', label='A2P1n')
    ax2.plot(p22n, color='c', label='A2P2n')
    #ax1.text('Mean ratio = '+str(np.mean(p21vp12)))
    ax2.set_xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),60))
    ax2.set_xticklabels(mins, rotation=270)
    ax2.set_title('Normalised Power Data')
    ax2.legend()
    ax2.margins(x=0)
    
    plt.tight_layout()
    plt.suptitle(title_string)
    plt.show()

def ApplyDBL():
    
    
    a1p1B = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
    p11 = a1p1B[:,116:606].mean(axis=1)
    a1p2B = np.load (DATA_PATH+'/temp//a1p2_binned.npy')
    p12 = a1p2B[:,116:606].mean(axis=1) 
    a2p1B = np.load (DATA_PATH+'/temp//a2p1_binned.npy')
    p21 = a2p1B[:,116:606].mean(axis=1) 
    a2p2B = np.load (DATA_PATH+'/temp//a2p2_binned.npy')
    p22 = a2p2B[:,116:606].mean(axis=1)
    
    p11n = np.load(DATA_PATH+'/temp/4norm/p11n.npy')
    p12n = np.load(DATA_PATH+'/temp/4norm/p12n.npy')
    p21n = np.load(DATA_PATH+'/temp/4norm/p21n.npy')
    p22n = np.load(DATA_PATH+'/temp/4norm/p22n.npy')
    
    p11u = np.load(DATA_PATH+'/temp/4norm/p11u.npy')
    p12u = np.load(DATA_PATH+'/temp/4norm/p12u.npy')
    p21u = np.load(DATA_PATH+'/temp/4norm/p21u.npy')
    p22u = np.load(DATA_PATH+'/temp/4norm/p22u.npy')
    
    firstDiff1 = p11 - p12
    firstDiff2 = p22 - p21
    firstDiff3 = p11 - p21
    firstDiff4 = p22 - p12
    
    doubleDiff = ( firstDiff3 + firstDiff4 ) / 2
    #doubleDiff2 = (firstDiff1 + firstDiff2) / 2
    null = ( firstDiff3 - firstDiff4) / 2
    null2 = (firstDiff1 - firstDiff2) /2
    null = null + (doubleDiff[0] - null[0])
    null2 = null2 + (doubleDiff[0] - null2[0])
    
    firstDiff1n = p11n - p12n
    firstDiff2n = p22n - p21n
    firstDiff3n = p11n - p21n
    firstDiff4n = p22n - p12n
    
    doubleDiffn = ( firstDiff3n + firstDiff4n ) / 2
    #doubleDiff2 = (firstDiff1 + firstDiff2) / 2
    nulln = (( firstDiff3n - firstDiff4n) / 2)*-1
    null2n = (firstDiff1n - firstDiff2n) /2
    nulln = nulln + (doubleDiff[0] - nulln[0])
    null2n = null2n + (doubleDiff[0] - null2n[0])
    
    #nulln = nulln + (null2n[0] - nulln[0])
    
    firstDiff1u = p11u - p12u
    firstDiff2u = p22u - p21u
    firstDiff3u = p11u - p21u
    firstDiff4u = p22u - p12u
    
    doubleDiffu = ( firstDiff3u + firstDiff4u ) / 2
    #doubleDiff2 = (firstDiff1 + firstDiff2) / 2
    nullu = ( firstDiff3u - firstDiff4u) / 2
    null2u = ((firstDiff1u - firstDiff2u) /2)
    nullu = nullu + (doubleDiff[0] - nullu[0])
    null2u= null2u + (doubleDiff[0] - null2u[0])
    
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)

    #DELETE
    powerAVG = (p11n + p12n + p21n + p22n) / 4
    hcAVG = np.load(DATA_PATH+'/temp/4norm/hc-mean.npy', allow_pickle = True)
    rcAVG = np.load(DATA_PATH+'/temp/4norm/rc-mean.npy', allow_pickle = True)
    mTAVG = np.load(DATA_PATH+'/temp/4norm/mT-mean.npy', allow_pickle = True)
    
    powerAVG = powerAVG / np.mean(powerAVG)
    hcAVG = ((hcAVG / np.mean(hcAVG)) * 100000) - 625000
    rcAVG = ((rcAVG / np.mean(rcAVG))*150000) -700000
    mTAVG = ((mTAVG / np.mean(mTAVG))*450000) -1000000
    #those should now scale 0 to 1
    #DELETE
    
    conv = hcAVG[0:14689] + doubleDiffn / 2
    
    baseline = doubleDiffn/doubleDiffn
    sigNorm = (doubleDiffn / np.mean(doubleDiffn))*0.0005
    ratio = doubleDiffn/nulln
    sigNorm = sigNorm + (ratio[0] - sigNorm[0])

    print('Ratio Adujusted up, mean values')
    print('null',np.mean(nullu))
    print('null2',np.mean(null2u))
    print('Seperation & Ratio: '+str(np.mean(nullu)-np.mean(null2u))+' & '+str(np.mean(nullu)/np.mean(null2u)))
    print('')
    print('Ratio Adujusted down, mean values')
    print('null',np.mean(nulln))
    print('null2',np.mean(null2n))
    print('Seperation & Ratio: '+str(np.mean(nulln)-np.mean(null2n))+' & '+str(np.mean(nulln)/np.mean(null2n)))
  
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = user_inputs[0]+' from '+user_inputs[3]+ ' for '+str(duration_actual)+' hours. Raw Power vs Normalised Power.'
    mins,first_tick, sample_to_datetime, frequency = xaxis(a1p1B, duration_actual)
    
    plt.close()
    fig, (ax1, ax2) = plt.subplots(1, 2)
   
    #plt.figure(figsize=(12, 8), dpi=300)
    
    #plt.setp(ax1, xticks=first_tick, xticklabels=mins)
    #plt.setp(ax2, xticks=first_tick, xticklabels=mins)
    ax1.set_xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),60))
    ax1.set_xticklabels(mins, rotation=270)
    #ax1.plot(doubleDiff, color='k', label='Signal')
    #ax1.plot(null, color='k', label='Null F3-F4/2', linewidth=0.5)
    #ax1.plot(null2, color='grey', label='Null F1-F2/2', linewidth=0.5)
    jeff = nulln / np.mean(nulln)
    ax1.plot(sigNorm-0.13, color='r', label='Signal (x 0.0005)')
    ax1.plot((ratio*-1)+0.04, color='k', label='Signal/Null')
    #ax1.plot((jeff*-1)+1, color='grey', label='Null')
    #ax1.text('Mean ratio = '+str(np.mean(p22vp11)))
    ax1.set_title('F3-F4/2 Null against Signal-to-Null Ratio')
    ax1.legend()
    ax1.margins(x=0)
    
    null2n = (null2n / np.mean(null2n))*-0.15
    rcAVG = (rcAVG/np.mean(rcAVG))*-1
    mTAVG = (mTAVG/np.mean(mTAVG))*-0.1
    boxAVG = rcAVG+mTAVG/2
    boxAVG = boxAVG/np.mean(boxAVG)*-1
    
    #ax2.plot(doubleDiffn, color='k', label='Signal')
    #ax2.plot(((nulln)), color='b', label='Null F3-F4/2', linewidth=0.5)#-10000
    #ax2.plot((ratio*-150000)-550000, color='k', label='sig/null')
    ax2.plot((null2n+0.15100), color='c', label='Null F1-F2/2 (x 0.15)', linewidth=0.5)
    
    #ax2.plot(mTAVG+0.100, color='m', label='MagicTee Temp Behaviour')
    #ax2.plot(boxAVG+1.0039, color='y', label='Average Box Temp Behaviour')
    ax2.plot(rcAVG+1.00275, color='r', label='Receiver Board Temp Behaviour')
    #ax2.plot(hcAVG, color='g', label='Horns Cables Temp Behaviour')
    #ax2.plot(conv, color='y', label = 'Convolved signal & temo behaviour')
    #ax2.plot(doubleDiffu, color='g', label='Signal UP')
    #ax2.plot(nullu, color='r', label='Null F3-F4/2 UP', linewidth=0.5)
    #ax2.plot(null2u, color='m', label='Null F1-F2/2 UP', linewidth=0.5)
   
    #ax1.text('Mean ratio = '+str(np.mean(p21vp12)))
    ax2.set_xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),60))
    ax2.set_xticklabels(mins, rotation=270)
    ax2.set_title('F1-F2/2 Null against Average Receiver Board Temperature Behaviour')
    ax2.legend()
    ax2.margins(x=0)
    
    plt.tight_layout()
    plt.suptitle(title_string)
    plt.show()

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
    axarr[0,0].axvline(x=116,linestyle='--', color='w', linewidth=0.5)
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
    
    
    #divide through by its own mean
    
    a1p1B = np.load (DATA_PATH+'/temp/4norm/a1p1_binned.npy')
    p11 = np.load(DATA_PATH+'/temp/4norm/p11.npy')
    p11n = np.load(DATA_PATH+'/temp/4norm/p11n.npy')
    p12n = np.load(DATA_PATH+'/temp/4norm/p12n.npy')
    p21n = np.load(DATA_PATH+'/temp/4norm/p21n.npy')
    p22n = np.load(DATA_PATH+'/temp/4norm/p22n.npy')
    
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    
    
    powerAVG = (p11n + p12n + p21n + p22n) / 4
    hcAVG = np.load(DATA_PATH+'/temp/4norm/hc-mean.npy', allow_pickle = True)
    rcAVG = np.load(DATA_PATH+'/temp/4norm/rc-mean.npy', allow_pickle = True)
    mTAVG = np.load(DATA_PATH+'/temp/4norm/mT-mean.npy', allow_pickle = True)
    
    powerAVG = powerAVG / np.mean(powerAVG)
    hcAVG = ((hcAVG / np.mean(hcAVG)) * 0.03) + 0.9875
    rcAVG = rcAVG / np.mean(rcAVG) +0.01
    mTAVG = mTAVG / np.mean(mTAVG) +0.01
    #those should now scale 0 to 1
    
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours. '
    
    mins,first_tick, sample_to_datetime, frequency = xaxis(a1p1B, duration_actual)
    
    plt.xticks(ticks=first_tick, labels=mins, rotation=270)
    if duration_actual > 48 and duration_actual < 150:
        plt.xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),60))
    elif duration_actual > 150:
        pass
    else:
        plt.xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),15))
    
    
    #plt.plot(mTAVG, color='grey', label="Magic T's", linewidth=0.5)
    plt.plot(hcAVG, color='g', label='Mean Horns & Cables Temp, x 0.03')
    plt.plot(rcAVG, color='r', label='Mean Receiver Board Temp')
    plt.plot(powerAVG, color='k', label='Mean total POWER (all inputs)')
    
    
    plt.legend(loc="upper right", fontsize=8)
    plt.title(title_string)
    plt.tight_layout()
    plt.margins(x=0)
    plt.show()
    
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
    
        plt.plot(p11vp11, color='b', label='A1P1')
        plt.plot(p12vp11, color='c', label='A1P2')
        plt.plot(p21vp11, color='r', label='A2P1')
        plt.plot(p22vp11, color='m', label='A2P2')
        
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.title(title_string)
        plt.tight_layout()
        plt.margins(x=0)
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_ratios_PS.png', bbox_inches="tight")
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
    
    p11n = np.load(DATA_PATH+'/temp/4norm/p11n.npy')
    p12n = np.load(DATA_PATH+'/temp/4norm/p12n.npy')
    p21n = np.load(DATA_PATH+'/temp/4norm/p21n.npy')
    p22n = np.load(DATA_PATH+'/temp/4norm/p22n.npy')
    
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours. Normalised Power Double Differences - with offset'
    
    mins,first_tick, sample_to_datetime, frequency = xaxis(a1p1B, duration_actual)

    firstDiff1 = p11n - p12n
    firstDiff2 = p22n - p21n
    firstDiff3 = p11n - p21n
    firstDiff4 = p22n - p12n
    
    doubleDiff = ( firstDiff3 + firstDiff4 ) / 2
    #doubleDiff2 = (firstDiff1 + firstDiff2) / 2
    null = ( firstDiff3 - firstDiff4) / 2
    null2 = (firstDiff1 - firstDiff2) /2
    
    null = null + (doubleDiff[0] - null[0])
    null2 = null2 + (doubleDiff[0] - null2[0])
    
    plt.xticks(ticks=first_tick, labels=mins, rotation=270)
    if duration_actual > 48 and duration_actual < 150:
        plt.xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),60))
    elif duration_actual > 150:
        pass
    else:
        plt.xticks(ticks=np.arange(first_tick[0],np.size(sample_to_datetime),15))
        
    plt.plot(null, c='b', linestyle='-', linewidth=0.5, label ='Null FD3-FD4/2')
    plt.plot(null2, c='c', linestyle='-', linewidth=0.5, label ='Null2 FD1-FD2/2')
    plt.plot(doubleDiff, c='r', linestyle='-', label='Sky Signal')
    #plt.plot(doubleDiff2, c='g', linestyle='-', label='Sky Signal2')

    plt.margins(x=0)
    plt.title(title_string)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()
    #plt.close()
    
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

def NormMenu(plotmenu):

    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    
    if os.path.exists(user_inputs[6]):
        pass
    else:
        os.system('mkdir '+user_inputs[6])
    

    print ('')
    print ('   -------------------------------------')
    print ('         Global Normalisation menu   ')
    print ('   -------------------------------------')
    print ('')
    print ('   1 - List Mean Ratios')
    print ('')
    print ('   2 - Plot Ratios of Phases per Channel')
    print ('')
    print ('   3 - Plot Ratios of all inputs.')
    print ('')
    print ('   4 - Plot Raw vs Normalised data')
    print('')
    print ('   6 - Plot DblDiff vs Normalised DblDiff')
    print('')
    print('   0 - Return to main menu')
    print('')
    choice = input('Select menu option (number): ')
    if choice.isdigit():
        if int(choice) == 1:
            showMeans()   
    
        elif int(choice) ==2:
            PlotChans() 
        
        elif int(choice) ==3:
            PlotAll()
                
        elif int(choice) ==4:
            ApplyNorm()
            
        elif int(choice) ==5:
            rawData()
        
        elif int(choice) ==6:
            ApplyDBL()
        
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
        


    return normmenu


#duration_actual = (np.size(p11) / 60)
#duration_actual = np.round(duration_actual, decimals=2)

print ('\033[1;32m ')


print('')
profile_name = input ('Please supply a name for this profile (alphanumeric characters only): ')
print('')
user_inputs = np.load(DATA_PATH+'/temp/inputs.npy', allow_pickle=True)
profile_date = user_inputs[0]
profile_time = user_inputs[3]
profile_duration = user_inputs[5]
active_profile = False
profile_filename = profile_name.replace(' ','_')
active = False

a1p1_profile_filepath = DATA_PATH+'/bp_profiles/band11-4norm-'+str(profile_filename)
a1p2_profile_filepath = DATA_PATH+'/bp_profiles/band12-4norm-'+str(profile_filename)
a2p1_profile_filepath = DATA_PATH+'/bp_profiles/band21-4norm-'+str(profile_filename)
a2p2_profile_filepath = DATA_PATH+'/bp_profiles/band22-4norm-'+str(profile_filename)

profile = np.array((profile_name, profile_date, profile_time, profile_duration, a1p1_profile_filepath, a1p2_profile_filepath, a2p1_profile_filepath, a2p2_profile_filepath, active_profile,active))


#load smoothed (60 second) bandpasses from stable observing run
a1p1B = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
a1p2B = np.load (DATA_PATH+'/temp/a1p2_binned.npy')
a2p1B = np.load (DATA_PATH+'/temp/a2p1_binned.npy')
a2p2B = np.load (DATA_PATH+'/temp/a2p2_binned.npy')
  
#mean each column to get average of each frequency bin across the bandpass
band11 = a1p1B[:,3:].mean(axis=0) 
band12 = a1p2B[:,3:].mean(axis=0)
band21 = a2p1B[:,3:].mean(axis=0)
band22 = a2p2B[:,3:].mean(axis=0)


# save for each input
np.save(DATA_PATH+'/bp_profiles/band11-4norm-'+str(profile_filename),band11)
np.save(DATA_PATH+'/bp_profiles/band12-4norm-'+str(profile_filename),band12)
np.save(DATA_PATH+'/bp_profiles/band21-4norm-'+str(profile_filename),band21)
np.save(DATA_PATH+'/bp_profiles/band22-4norm-'+str(profile_filename),band22)

try:
    existing_profiles = np.load(DATA_PATH+'/bp_profiles/profiles.npy')

    profile = np.row_stack((existing_profiles , profile))
except:
    pass

np.save(DATA_PATH+'/bp_profiles/profiles.npy',profile)


#to apply profile to other observations divide through radiometer array by profile values.


ratiosUP()
ratios()

normmenu=True
while normmenu:
    normmenu = NormMenu(normmenu)





