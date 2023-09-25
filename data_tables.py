#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:33:26 2022

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

os.system('rm /local5/scratch/pblack/temp/concats/*.*')

os.system('rm /local5/scratch/pblack/temp/arm1.npy')
os.system('rm /local5/scratch/pblack/temp/arm2.npy')
os.system('rm /local5/scratch/pblack/temp/a1p1.npy')
os.system('rm /local5/scratch/pblack/temp/a1p2.npy')
os.system('rm /local5/scratch/pblack/temp/a2p1.npy')
os.system('rm /local5/scratch/pblack/temp/a2p2.npy')
os.system('rm /local5/scratch/pblack/temp/obshdr.npy')

#---------------------------------------------------------------------------

# aquire positional data, observer and mode

def getHeader(file_table):

    print('')
    print('Reading data header')
    
    observatory = fits.open(file_table[0,0])

    site = observatory[0].header['ORIGIN'] #name of the observatory
    observer = observatory[0].header['OBSERVER'] #who is using the telescope
    obs_mode = observatory[0].header['INSTRUME'] #observing mode
    longitude = observatory[0].header['OBSGEO-L'] #telescope east longitude in deg
    latitude = observatory[0].header['OBSGEO-B'] #telescope north latitude in deg
    altitude = observatory[0].header['OBSGEO-H'] #telescope height above sealevelin m
    obs_date = observatory[0].header['DATE'] #date file was created

    azimuth = observatory[0].header['AZIMUTH'] #azimuth of LBASS frame in deg (usually 180)

#------------------------------------------------------------------------------
# aquire data references and boundaries

    refPIX = observatory[1].header['1CRPX3'] #reference pixel number
    refFREQ = observatory[1].header['1CRVL3'] #reference frequency
    chnSPACE = observatory[1].header['1CDLT3'] # channel spacing/width = 450 MHz / 8192

    MINchnFREQ = 7394
    MAXchnFREQ = 6677 #corresponds to 1293.84 - 1433.22 MHz //// gives 718 channels

    obsheader = np.column_stack((site, observer, obs_mode, longitude, latitude, altitude, azimuth, obs_date))
    np.save(DATA_PATH+'/temp/obshdr', obsheader)


    #not entirely clear on how this works, but give the frequency of each channel bin
    x = np.arange(0,718)
    frequency = refFREQ + (chnSPACE * (x + 1 - refPIX))
    
    np.save(DATA_PATH+'/temp/freq', frequency)

#--------------------------------------------------------------------------
# build the data table which incorporates information from all relecvant files
# THIS might be more length a process if there are lots of files to handle.

def buildTable1(file_table):

    print('Loading data tables')

    i = 0

    for i in range(0,np.size(file_table[:,0])):
        opened = fits.open(file_table[i,0])
        sample_time = np.array(opened[1].data['SAMPLE_TIME'])
        switch_phase = np.array(opened[1].data['SWITCH_PHASE'])
        elevation = np.array(opened[1].data['ELEVATION']) #greater than 90 doesn't change azimuth
        channel1R = np.array(opened[1].data['CHANNEL1']) #one horn 
        
        #channel1 = (math.e**-10) * np.flip(channel1R, axis=1) #the channel data comes out in reserve order, so correct
        channel1 = (math.e**-10) * channel1R
        
        channel1_data = np.column_stack((sample_time, switch_phase, elevation, channel1))

        np.save(DATA_PATH+'/temp/concats/chan1_'+(str(i)), channel1_data)

#---------------------------------------------------------------------------

def buildTable1a(file_table):

    i = 0

    for i in range(0,np.size(file_table[:,0])):
        opened = fits.open(file_table[i,0])
        sample_time = np.array(opened[1].data['SAMPLE_TIME'])
        switch_phase = np.array(opened[1].data['SWITCH_PHASE'])
        elevation = np.array(opened[1].data['ELEVATION']) #greater than 90 doesn't change azimuth
        channel2R = np.array(opened[1].data['CHANNEL2']) #two horn
        
        #channel2 = (math.e**-10) * np.flip(channel2R, axis=1) #multiplied by e^-10 to scale down to sensible arbitrary number
        channel2 = (math.e**-10) * channel2R

        channel2_data = np.column_stack((sample_time, switch_phase, elevation, channel2))

        np.save(DATA_PATH+'/temp/concats/chan2_'+(str(i)), channel2_data)

#---------------------------------------------------------------------------


def buildTable2(kickoff, duration):
    
    cat1_files = []
    cat1_files = sorted(glob.glob(DATA_PATH+'/temp/concats/chan1_*.npy'))
    arm1x=np.load(DATA_PATH+'/temp/concats/chan1_0.npy')

    i = 1
    for i in range(1,len(cat1_files)):
        x = np.load(DATA_PATH+'/temp/concats/chan1_'+str(i)+'.npy')
        arm1x = np.row_stack((arm1x, x))
        
    try:
    
        indgen=[]
        y = np.size(arm1x[:,0])
        i = 0
        for i in range(0,y):
            if float(arm1x[i,0]) >= kickoff and float(arm1x[i,0]) <= duration:
                indgen.append(i)
            else:
                pass
                
        ind1 = indgen[0]
        ind2 = indgen[-1]
                     
        re_date = False
        np.save(DATA_PATH+'/temp/re_date', re_date)
    
    except:
        
        kickoffB = np.load(DATA_PATH+'/temp/kickoffB.npy')
        durationB = np.load(DATA_PATH+'/temp/durationB.npy')
        indgen=[]
        y = np.size(arm1x[:,0])
        i = 0
        for i in range(0,y):
            if float(arm1x[i,0]) >= kickoffB and float(arm1x[i,0]) <= durationB:
                indgen.append(i)
            else:
                pass
        
        ind1 = indgen[0]
        ind2 = indgen[-1]
        
        re_date = True
        np.save(DATA_PATH+'/temp/re_date', re_date)
  
    
    arm1 = arm1x[ind1:ind2,:]
    
    np.save(DATA_PATH+'/temp/arm1.npy', arm1)
    

#----------------------------------------------------------------------

def buildTable2a(kickoff, duration):
    
    cat2_files = []
    cat2_files = sorted(glob.glob(DATA_PATH+'/temp/concats/chan2_*.npy'))
    arm2y=np.load(DATA_PATH+'/temp/concats/chan2_0.npy')
    
    i = 1
    for i in range(1,len(cat2_files)):
      
        y = np.load(DATA_PATH+'/temp/concats/chan2_'+str(i)+'.npy')
        arm2y = np.row_stack((arm2y, y))

    try:
    
        indgen=[]
        y = np.size(arm2y[:,0])
        i = 0
        for i in range(0,y):
            if float(arm2y[i,0]) >= kickoff and float(arm2y[i,0]) <= duration:
                indgen.append(i)
            else:
                pass
        
        ind1 = indgen[0]
        ind2 = indgen[-1]
        
    except:
        
        kickoffB = np.load(DATA_PATH+'/temp/kickoffB.npy')
        durationB = np.load(DATA_PATH+'/temp/durationB.npy')
        indgen=[]
        y = np.size(arm2y[:,0])
        i = 0
        for i in range(0,y):
            if float(arm2y[i,0]) >= kickoffB and float(arm2y[i,0]) <= durationB:
                indgen.append(i)
            else:
                pass
        
        ind1 = indgen[0]
        ind2 = indgen[-1]
    
    arm2 = arm2y[ind1:ind2,:]
    
    np.save(DATA_PATH+'/temp/arm2.npy', arm2)

#----------------------------------------------------------------------------

def buildTable3():
    
    print('Aquiring phase data')
    
    arm1 = np.load(DATA_PATH+'/temp/arm1.npy')
    
    ind1 = np.where(arm1 == 1.0)
    ind2 = np.where(arm1 == 2.0)

    arm1pha1 = np.array(arm1[ind1[0],:])
    arm1pha2 = np.array(arm1[ind2[0],:])
    
    np.save(DATA_PATH+'/temp/a1p1', arm1pha1)
    np.save(DATA_PATH+'/temp/a1p2', arm1pha2)
  
#----------------------------------------------------------------------------

def buildTable3a():
    
    arm2 = np.load(DATA_PATH+'/temp/arm2.npy')

    ind3 = np.where(arm2 == 1.0)
    ind4 = np.where(arm2 == 2.0)
    
    arm2pha1 = np.array(arm2[ind3[0],:])
    arm2pha2 = np.array(arm2[ind4[0],:])
    
    np.save(DATA_PATH+'/temp/a2p1', arm2pha1)
    np.save(DATA_PATH+'/temp/a2p2', arm2pha2)
    
#--------------------------------------------------------------------

def sampleLimits(file_table, user_inputs):
    
    # a new run that starts part way into the day will restart the counter.
    
    reset_day = False
    
    try:
        reset_day = np.load(DATA_PATH+'/temp/reset_day.npy')
    except:
        pass
    
    firstSAM = float(file_table[0,5])
    i = 0
    while firstSAM > 86400:
        firstSAM = firstSAM - 86400
        i = i + 1
    
    
    
    subin = i * 86400    
    kickoff = (float(user_inputs[4]) + subin)
    
    if int(user_inputs[5]==5000):
        duration = none
    else:
        duration = (float(user_inputs[4]) + subin) + (float(user_inputs[5]) * 3600)
        
    if reset_day:
       
        kickoffB = (float(user_inputs[4]))
        np.save(DATA_PATH+'/temp/kickoffB', kickoffB)
        if int(user_inputs[5]==5000):
            durationB = none
            np.save(DATA_PATH+'/temp/durationB', durationB)
        else:
            durationB = (float(user_inputs[4]) + (float(user_inputs[5]) * 3600))
            np.save(DATA_PATH+'/temp/durationB', durationB)
        
        
        
        
        
        
    return kickoff, duration
     
#-------------------------------------------------------------------------

def temp_selects(user_inputs):
    
    all_temps = np.load(DATA_PATH+'/temp/temp_data.npy', allow_pickle=True)
    
    i=0
    for i in range (0,np.size(all_temps[:,0])):
        x = str(all_temps[i,0])
        y = x.split()
        if str(y[0]) == str(user_inputs[0]):
            go_date = i
            break
    
    i = go_date
    for i in range(go_date,np.size(all_temps[:,0])):
        x = str(all_temps[i,0])
        y = x.split()
        h,m,s = y[1].split(':')
        if h == str(user_inputs[1]):
            go_time = i - 5
            break
    
    run_length = (go_time+ (int(float(user_inputs[5])) + 1) * 60)
    
    selected_temps = all_temps[go_time:run_length,:]
    
    
    i = 0
    for i in range (0,(np.size(selected_temps[:,0]))-1):
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
    
#---------------------------------------------------------------------


def binINmins():
    
    print('Rebinning samples')
    
    i = 0
    # bin width (1 minute = 600 samples, but split over two phases is 300)
    # bin centre (middle of min, but better based on temp record times)
    
    #find the number of bins by dividing up the samples by 300, 
    #then whichever is the larger number use to bin up
    
    
    arm1phase1 = np.load(DATA_PATH+'/temp/a1p1.npy')
    arm1phase2 = np.load(DATA_PATH+'/temp/a1p2.npy')
    arm2phase1 = np.load(DATA_PATH+'/temp/a2p1.npy')
    arm2phase2 = np.load(DATA_PATH+'/temp/a2p2.npy')
    
    os.system('rm /local5/scratch/pblack/temp/bin/*.*')
    
    bin_width = 300 #(for each phase per min)
    ts1 = np.size(arm1phase1[:,0]) / 300
    ts2 = np.size(arm1phase2[:,0]) / 300
    if ts1 > ts2:
        no_bins = int(ts2)
    else:
        no_bins = int(ts1)
    #this will vary depending on the files selected, but should be ~60 an hour
   
    #60 bins, 300 samples in each, 
    #arm1phase1[0:300,:] first bin
    #arm2phase1[301:600,:] second bin and so on
    
    #take mean of each column in given range
    
    i = 0 
    a = 0
    b = bin_width
    for i in range (0,no_bins):
         temp_bin = arm1phase1[a:b,:]
         col_means = temp_bin.mean(axis=0)
         np.save(DATA_PATH+'/temp/bin/a1p1_bin'+str(i), col_means)
         temp_bin = arm1phase2[a:b,:]
         col_means = temp_bin.mean(axis=0)
         np.save(DATA_PATH+'/temp/bin/a1p2_bin'+str(i), col_means)
         temp_bin = arm2phase1[a:b,:]
         col_means = temp_bin.mean(axis=0)
         np.save(DATA_PATH+'/temp/bin/a2p1_bin'+str(i), col_means)
         temp_bin = arm2phase2[a:b,:]
         col_means = temp_bin.mean(axis=0)
         np.save(DATA_PATH+'/temp/bin/a2p2_bin'+str(i), col_means)
         a = b +1
         b = b + bin_width

    cat1_files = []
    cat1_files = (glob.glob(DATA_PATH+'/temp/bin/a1p1_*.npy'))
    
    a1p1_binned=np.load(DATA_PATH+'/temp/bin/a1p1_bin0.npy')

    i = 1
    for i in range(1,len(cat1_files)):
        x = np.load(DATA_PATH+'/temp/bin/a1p1_bin'+str(i)+'.npy')
        a1p1_binned = np.row_stack((a1p1_binned, x))
       
    np.save(DATA_PATH+'/temp/a1p1_binned.npy', a1p1_binned)
    
    cat1_files = []
    cat1_files = (glob.glob(DATA_PATH+'/temp/bin/a1p2_*.npy'))
    
    a1p2_binned=np.load(DATA_PATH+'/temp/bin/a1p2_bin0.npy')

    i = 1
    for i in range(1,len(cat1_files)):
        x = np.load(DATA_PATH+'/temp/bin/a1p2_bin'+str(i)+'.npy')
        a1p2_binned = np.row_stack((a1p2_binned, x))
       
    np.save(DATA_PATH+'/temp/a1p2_binned.npy', a1p2_binned)
    
    cat1_files = []
    cat1_files = (glob.glob(DATA_PATH+'/temp/bin/a2p1_*.npy'))
    
    a2p1_binned=np.load(DATA_PATH+'/temp/bin/a2p1_bin0.npy')

    i = 1
    for i in range(1,len(cat1_files)):
        x = np.load(DATA_PATH+'/temp/bin/a2p1_bin'+str(i)+'.npy')
        a2p1_binned = np.row_stack((a2p1_binned, x))
       
    np.save(DATA_PATH+'/temp/a2p1_binned.npy', a2p1_binned)
    
    cat1_files = []
    cat1_files = (glob.glob(DATA_PATH+'/temp/bin/a2p2_*.npy'))
    
    a2p2_binned=np.load(DATA_PATH+'/temp/bin/a2p2_bin0.npy')

    i = 1
    for i in range(1,len(cat1_files)):
        x = np.load(DATA_PATH+'/temp/bin/a2p2_bin'+str(i)+'.npy')
        a2p2_binned = np.row_stack((a2p2_binned, x))
       
    np.save(DATA_PATH+'/temp/a2p2_binned.npy', a2p2_binned)
    
#-----------------------------------------------------------------------------
    
def binIN718s():
    
    #for waterfall plots, to always make square, needs 718 bins
    
    arm1phase1 = np.load(DATA_PATH+'/temp/a1p1.npy')
    arm1phase2 = np.load(DATA_PATH+'/temp/a1p2.npy')
    arm2phase1 = np.load(DATA_PATH+'/temp/a2p1.npy')
    arm2phase2 = np.load(DATA_PATH+'/temp/a2p2.npy')
    
    os.system('rm /local5/scratch/pblack/temp/bin/*.*')
    
    ts1 = np.size(arm1phase1[:,0]) / 718
    ts2 = np.size(arm1phase2[:,0]) / 718
    if ts1 > ts2:
        bin_width = int(ts2)
    else:
        bin_width = int(ts1)
    
    sq_scale = bin_width / 300 # how many minutes per bin
    
    np.save(DATA_PATH+'/temp/sq_scale', sq_scale)
    
    no_bins = 718
    
    i = 0 
    a = 0
    b = bin_width
    for i in range (0,no_bins):
         temp_bin = arm1phase1[a:b,:]
         col_means = temp_bin.mean(axis=0)
         np.save(DATA_PATH+'/temp/bin/sq_a1p1_bin'+str(i), col_means)
         temp_bin = arm1phase2[a:b,:]
         col_means = temp_bin.mean(axis=0)
         np.save(DATA_PATH+'/temp/bin/sq_a1p2_bin'+str(i), col_means)
         temp_bin = arm2phase1[a:b,:]
         col_means = temp_bin.mean(axis=0)
         np.save(DATA_PATH+'/temp/bin/sq_a2p1_bin'+str(i), col_means)
         temp_bin = arm2phase2[a:b,:]
         col_means = temp_bin.mean(axis=0)
         np.save(DATA_PATH+'/temp/bin/sq_a2p2_bin'+str(i), col_means)
         a = b +1
         b = b + bin_width
         
         
    
    cat1_files = []
    cat1_files = sorted(glob.glob(DATA_PATH+'/temp/bin/sq_a1p1_*.npy'))
    
    a1p1_binned=np.load(DATA_PATH+'/temp/bin/sq_a1p1_bin0.npy')

    i = 1
    for i in range(1,len(cat1_files)):
        x = np.load(DATA_PATH+'/temp/bin/sq_a1p1_bin'+str(i)+'.npy')
        a1p1_binned = np.row_stack((a1p1_binned, x))
       
    np.save(DATA_PATH+'/temp/sq_a1p1_binned.npy', a1p1_binned)
    
    cat1_files = []
    cat1_files = sorted(glob.glob(DATA_PATH+'/temp/bin/sq_a1p2_*.npy'))
    
    a1p2_binned=np.load(DATA_PATH+'/temp/bin/sq_a1p2_bin0.npy')

    i = 1
    for i in range(1,len(cat1_files)):
        x = np.load(DATA_PATH+'/temp/bin/sq_a1p2_bin'+str(i)+'.npy')
        a1p2_binned = np.row_stack((a1p2_binned, x))
       
    np.save(DATA_PATH+'/temp/sq_a1p2_binned.npy', a1p2_binned)
    
    cat1_files = []
    cat1_files = sorted(glob.glob(DATA_PATH+'/temp/bin/sq_a2p1_*.npy'))
    
    a2p1_binned=np.load(DATA_PATH+'/temp/bin/sq_a2p1_bin0.npy')

    i = 1
    for i in range(1,len(cat1_files)):
        x = np.load(DATA_PATH+'/temp/bin/sq_a2p1_bin'+str(i)+'.npy')
        a2p1_binned = np.row_stack((a2p1_binned, x))
       
    np.save(DATA_PATH+'/temp/sq_a2p1_binned.npy', a2p1_binned)
    
    cat1_files = []
    cat1_files = sorted(glob.glob(DATA_PATH+'/temp/bin/sq_a2p2_*.npy'))
    
    a2p2_binned=np.load(DATA_PATH+'/temp/bin/sq_a2p2_bin0.npy')

    i = 1
    for i in range(1,len(cat1_files)):
        x = np.load(DATA_PATH+'/temp/bin/sq_a2p2_bin'+str(i)+'.npy')
        a2p2_binned = np.row_stack((a2p2_binned, x))
       
    np.save(DATA_PATH+'/temp/sq_a2p2_binned.npy', a2p2_binned)
         
#-----------------------------------------------------------------------------
    

# load in the user selections and related fits / csv files

file_table = np.load(DATA_PATH+'/temp/file1.npy')
user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')

#selected_temps = temp_selects(user_inputs)

for i in tqdm(range(0,9)):
    
    if i == 0:
        kickoff, duration = sampleLimits(file_table, user_inputs)
    elif i ==1:
        getHeader(file_table)
    elif i ==2:
        buildTable1(file_table)
    elif i ==3:
        buildTable1a(file_table)
    elif i ==4:
        buildTable2(kickoff,duration)
    elif i ==5:
        buildTable2a(kickoff, duration)
    elif i ==6:
        buildTable3()
    elif i ==7:
        buildTable3a()
    elif i ==8:
        binINmins()
    elif i ==9:
        binIN718s()

    

#----------------------------------------------------------------------

#for waterfall diagram????
#start date, subtract out seconds to get sample time for day in question
#how many lines of samples in the selected range
#how many minutes worth of data
# how many samples per minute
#time bin size = ( number of mins / 700+1 ) * samples per minute
# binning is half that, number of bins is samples in range/ 2* binning
