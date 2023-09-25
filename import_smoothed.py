#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:33:26 2022

@author: pblack
"""

DATA_PATH = '/mirror/scratch/pblack'


import numpy as np
from astropy.time import Time 
from astropy.time import TimeDelta
import datetime
import time
from tqdm import tqdm #progress bars
import os
from math import nan
import csv
from csv import DictReader
import pandas as pd
import gc
import tkinter
from scipy import stats
import warnings

os.chdir('/scratch/nas_lbass/raw_data/')

#---------------------------------------------------------------------------

user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')

#search numpy arrays for data in that time range
#length check?
#allow gap?
#save binned, bandpass and powers and temperatures for the relevent range
#report if not possible
#PARTIAL?
#  smooth_string = '/scratch/nas_lbass/binned_data/'+str(year)+'_'+str(month)+'/'


roaring_success = True
smooth_string = user_inputs[7]
iso_string = str(user_inputs[0])+' '+str(user_inputs[1])+':'+str(user_inputs[2])+':00.0'  
start_time = Time(iso_string, format='iso', scale='utc', precision=4)
end_time = start_time + TimeDelta((float(user_inputs[5]) * 3600), format='sec')
start_month = start_time.strftime('%Y %M/')
end_month = end_time.strftime('%Y %M/')

if os.path.exists(smooth_string+'a1p1.npy'):

    a1p1_array = np.load(smooth_string+'a1p1.npy') 
    a1p2_array = np.load(smooth_string+'a1p2.npy') 
    a2p1_array = np.load(smooth_string+'a2p1.npy') 
    a2p2_array = np.load(smooth_string+'a2p2.npy') 
    time_array = np.load(smooth_string+'time_array.npy')
    temp_array = np.load(smooth_string+'one_wire.npy')

else:
    print('Smoothed data for this date has not yet been created.')
    roaring_success = False

if start_month == end_month:
    pass
else:
    if os.path.exists('/scratch/nas_lbass/binned_data/'+end_month+'a1p1.npy'):

        q = np.load(smooth_string+'a1p1.npy') 
        r = np.load(smooth_string+'a1p2.npy') 
        s = np.load(smooth_string+'a2p1.npy') 
        t = np.load(smooth_string+'a2p2.npy') 
        u = np.load(smooth_string+'time_array.npy')
        v = np.load(smooth_string+'one_wire.npy')
        
        a1p1_array = np.row_stack((a1p1_array,q))
        a1p2_array = np.row_stack((a1p2_array,r))
        a2p1_array = np.row_stack((a2p1_array,s))
        a2p2_array = np.row_stack((a2p2_array,t))
        time_array = np.row_stack((time_array,u))
        temp_array = np.row_stack((temp_array,v))
        
        del q, r, s, t, u, v

    else:
        print('Available smoothed data does not extend to the full duration requested.')
        skip = input('Attempt to load and process unsmoothed data instead? (Y/N): ')
        if str(skip) == 'y' or str(skip) == 'Y':
            roaring_success = False
        else:
            pass


index_up = np.where(time_array => start_time.mjd)
index_down = np.where(time_array =< end_time.mjd)
index_actual = np.intersect1d(index_up, index_down) #only times common to both lists

data_limit = Time(time_array[index_actual[-1]],format='mjd',scale='utc',precision=4)
time_diff = end_time - data_limit
time_diff = float(time_diff.sec)
if time_diff > 300:
    print('Available smoothed data does not extend to the full duration requested.')
    print('Short by approximately',np.round((time_diff/3600),decimals=2),'hours of data.')
    skip = input('Attempt to load and process unsmoothed data instead? (Y/N): ')
    if str(skip) == 'y' or str(skip) == 'Y':
        roaring_success = False
 

a1p1_array = a1p1_array[index_actual,:]
a1p2_array = a1p2_array[index_actual,:]
a2p1_array = a2p1_array[index_actual,:]
a2p2_array = a2p2_array[index_actual,:]
temp_array = temp_array[index_actual,:]

np.save(DATA_PATH+'/temp/a1p1_binned.npy',a1p1_array)
np.save(DATA_PATH+'/temp/a1p2_binned.npy',a1p2_array)
np.save(DATA_PATH+'/temp/a2p1_binned.npy',a2p1_array)
np.save(DATA_PATH+'/temp/a2p2_binned.npy',a2p2_array)

a1p1_array = a1p1_array[:,3:]
a1p2_array = a1p2_array[:,3:]
a2p1_array = a2p1_array[:,3:]
a2p2_array = a2p2_array[:,3:]

np.save(DATA_PATH+'/temp/a1p1_power.npy', (a1p1_array[:,112:569].mean(axis=1)))   
np.save(DATA_PATH+'/temp/a1p2_power.npy', (a1p2_array[:,112:569].mean(axis=1)))  
np.save(DATA_PATH+'/temp/a2p1_power.npy', (a2p1_array[:,112:569].mean(axis=1)))   
np.save(DATA_PATH+'/temp/a2p2_power.npy', (a2p2_array[:,112:569].mean(axis=1)))  

np.save(DATA_PATH+'/temp/a1p1_bandpass.npy', (a1p1_array)) 
np.save(DATA_PATH+'/temp/a1p2_bandpass.npy', (a1p2_array)) 
np.save(DATA_PATH+'/temp/a2p1_bandpass.npy', (a2p1_array)) 
np.save(DATA_PATH+'/temp/a2p2_bandpass.npy', (a2p2_array)) 

np.save(DATA_PATH+'/temp/one_wire.npy',temp_array)


def buildTable2(kickoff, duration, cycles, nom, batch, samples_per_min):

    # this opens each numpy array of radiometer data and combines into a single larger array.  Then it uses the user specified start and length of observations to trim the array down to only relevant datapoints. 

    lisst= ['P(L,\u03C0) W','P(L,0) E','P(R,\u03C0) E','P(R,0) W']
    ts = []
    j=0
    for j in range (0,4):
        
        print ('Aquiring',lisst[j])

        arm1x=np.load(DATA_PATH+'/temp/concats/breaky_'+str(nom[j])+'_'+str(batch)+'.npy')  #the first file
      
        try: #attempt to trim data array to user specified start and end points
            kick_ind = np.where(arm1x[:,0] >= kickoff) #indices of data after start time
            arm1x = arm1x[kick_ind[0],:]
            dur_ind = np.where(arm1x[:,0] < duration)
            arm1x = arm1x[dur_ind[0],:]
                   
            if j==0:         
                re_date = False
                np.save(DATA_PATH+'/temp/re_date', re_date)

        except:  #load alternative values to and try these instead
            kickoffB = np.load(DATA_PATH+'/temp/kickoffB.npy')
            durationB = np.load(DATA_PATH+'/temp/durationB.npy')

            kick_ind = np.where(arm1x[:,0] >= kickoffB) #indices of data after start time
            arm1x = arm1x[kick_ind[0],:]
            dur_ind = np.where(arm1x[:,0] < durationB)
            arm1x = arm1x[dur_ind[0],:]

            if j==0:
                re_date = True
                np.save(DATA_PATH+'/temp/re_date', re_date)
        
        np.save(DATA_PATH+'/temp/'+str(nom[j])+'.npy', arm1x)  #save the trimmed array
        ts.append(np.size(arm1x[:,0])/samples_per_min) #how many minutes of data
        gc.collect()
    
    
    return ts
#----------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------

def sampleLimits(file_table, user_inputs):
    
    # establishes the sample times associate with the beginning and end of the run specified by the user
    # the radiometer data start/end is a total of 62 seconds longer then the temp data start/end to allow for the temp data to set the centre point for the time bins at a later stage.

    # a new run that starts part way into the day will restart the counter.
    
    reset_day = False
    
    try:
        reset_day = np.load(DATA_PATH+'/temp/reset_day.npy')
    except:
        pass
    #can be the right day, but the previous file can be the wrong MJD.
    user_date = np.load(DATA_PATH+'/temp/user_date.npy', allow_pickle=True)
    if user_date == file_table[0,1]:  #if first file was create on day chosen by the user
        if file_table[0,12] == file_table[1,12]: #same MJD
            firstSAM = float(file_table[0,5]) #first sample number of first file
            run_MJD = Time(file_table[0,12], format='mjd', scale='utc', precision=4)
        else:
            #np.delete(file_table, 0 ,axis=0)
            firstSAM = float(file_table[1,5]) #first sample number of first file is actually second fits file in the list
            run_MJD = Time(file_table[1,12], format='mjd', scale='utc', precision=4)
    else:
        firstSAM = float(file_table[1,5]) 
        run_MJD = Time(file_table[1,12], format='mjd', scale='utc', precision=4)
        
    input_str = (str(user_date)+' '+str(user_inputs[3])+':00')
    
    user_datetime = Time(str(input_str), format='iso', scale='utc', out_subfmt='date_hm') #creates user specified datetime object
  
    
    kick_datetime = (user_datetime - run_MJD) - TimeDelta(31, format='sec') #sample time with 30 seconds ahead to allow for binning centred on the first minute of the first sample
    if kick_datetime.sec < 0:
        kick_datetime = kick_datetime + TimeDelta(86400, format='sec')

    duration_datetime = kick_datetime + TimeDelta(((float(user_inputs[5]) * 3600)+62), format='sec')
    #user duration convert to seconds, plus enough time to allow binnning past final sample.
    #31 seconds before and after the user selected duration.
    kickoff = float(kick_datetime.sec)  
    duration = float(duration_datetime.sec)
    
    kick4temps = (user_datetime - run_MJD)
    dur4temps = kick4temps + TimeDelta((float(user_inputs[5]) * 3600), format='sec')
    kick4temps = float(kick4temps.sec)  
    dur4temps = float(dur4temps.sec)  

    if reset_day:
       
        kickoffB = (float(user_inputs[4]))
        np.save(DATA_PATH+'/temp/kickoffB', kickoffB)
        durationB = (float(user_inputs[4]) + (float(user_inputs[5]) * 3600))
        np.save(DATA_PATH+'/temp/durationB', durationB)

    gc.collect()
    return kickoff, duration, kick4temps, dur4temps
     
#---------------------------------------------------------------------


# load in the user selections and related fits / csv files

file_table = np.load(DATA_PATH+'/temp/file1.npy', allow_pickle=True)  #all relevant fits files
user_inputs = np.load(DATA_PATH+'/temp/inputs.npy', allow_pickle=True) 

if os.path.exists(user_inputs[7]):
    pass
else:
    os.system('mkdir '+user_inputs[7])

getHeader(file_table)
kickoff, duration, kick4temps, dur4temps = sampleLimits(file_table, user_inputs)
parameters = np.load(DATA_PATH+'/temp/parameters.npy')

if parameters[7] == 'True': #lighter processing load on Moonhut.
    batches = int(np.size(file_table[:,0])/8)+1
    start = 0
    if np.size(file_table[:,0]) == 10:
        end = 13
    else:
        end = 10
else:
    batches = int(np.size(file_table[:,0])/26)+1
    start = 0
    if np.size(file_table[:,0]) == 28:
        end = 30
    else:
        end = 28


for i in range (0,batches):
    #np.save(DATA_PATH+'/temp/list_concats/file_table_'+str(i)+'.npy',file_table[a:b,:])
    print('Processing Batch',(i+1),'of',batches,':')
    try:
        file_tableX = file_table[start:end,:]

        cycles, nom = buildTable1(file_tableX, i)
        if i == 0:
            samples_per_min = sample_rate_check() 
        ts = buildTable2(kickoff,duration,cycles, nom, i,samples_per_min)

        no_bins, bin_width, one_wire_sts, rebinnable, adjust = buildTable3(cycles, kick4temps, dur4temps, i, samples_per_min)

        if rebinnable and parameters[0]=='True':
            rebinnable = binONEWIRE(nom, no_bins, bin_width, one_wire_sts, rebinnable, i,samples_per_min)

        elif rebinnable and parameters[0]=='False':
            binINminsX(nom,ts,samples_per_min)
        if rebinnable:
            pass
        else:
            binINminsX(nom,ts,samples_per_min)
        np.save(DATA_PATH+'/temp/rebinnable.npy', rebinnable)

        if parameters[7] == 'True': #lighter processing load on Moonhut.
            start = start+8
            end = end+8
        else:
            start = start + 26
            end = end + 26

        os.system('rm -r /local5/scratch/pblack/temp/concats/')
        os.system('mkdir /local5/scratch/pblack/temp/concats/')

        os.system('cp /local5/scratch/pblack/temp/a1p1.npy /local5/scratch/pblack/temp/list_concats/a1p1_binned_'+str(i)+'.npy')
        os.system('rm /local5/scratch/pblack/temp/a1p1.npy')

        os.system('cp /local5/scratch/pblack/temp/a1p2.npy /local5/scratch/pblack/temp/list_concats/a1p2_binned_'+str(i)+'.npy')
        os.system('rm /local5/scratch/pblack/temp/a1p2.npy')

        os.system('cp /local5/scratch/pblack/temp/a2p1.npy /local5/scratch/pblack/temp/list_concats/a2p1_binned_'+str(i)+'.npy')
        os.system('rm /local5/scratch/pblack/temp/a2p1.npy')

        os.system('cp /local5/scratch/pblack/temp/a2p2.npy /local5/scratch/pblack/temp/list_concats/a2p2_binned_'+str(i)+'.npy')
        os.system('rm /local5/scratch/pblack/temp/a2p2.npy')

    except:
        pass

#if batches > 8:
#    print('Unsmoothed data unavailable due to insufficient memory. ~24 hours unsmoothed is max available.')
#for i in range (0,5): #more than this will likely crash the program
 #   if i == 0:
  #      a1p1 = np.load(DATA_PATH+'/temp/list_concats/a1p1_'+str(i)+'.npy')
   # else:
    #    try:
     #       x = np.load(DATA_PATH+'/temp/list_concats/a1p1_'+str(i)+'.npy')
      #      a1p1 = np.row_stack((a1p1,x))
       # except:
        #    pass

#np.save(DATA_PATH+'/temp/a1p1.npy',a1p1)
#del a1p1

#for i in range (0,5): #more than this will likely crash the program
 #   if i == 0:
  #      a1p2 = np.load(DATA_PATH+'/temp/list_concats/a1p2_'+str(i)+'.npy')
   # else:
    #    try:
     #       x = np.load(DATA_PATH+'/temp/list_concats/a1p2_'+str(i)+'.npy')
      #      a1p2 = np.row_stack((a1p2,x))
       # except:
        #    pass

#np.save(DATA_PATH+'/temp/a1p2.npy',a1p2)
#del a1p2

#for i in range (0,5): #more than this will likely crash the program
 #   if i == 0:
  #      a2p1 = np.load(DATA_PATH+'/temp/list_concats/a2p1_'+str(i)+'.npy')
   # else:
    #    try:
     #       x = np.load(DATA_PATH+'/temp/list_concats/a2p1_'+str(i)+'.npy')
      #      a2p1 = np.row_stack((a2p1,x))
       # except:
        #    pass

#np.save(DATA_PATH+'/temp/a2p1.npy',a2p1)
#del a2p1

#for i in range (0,5): #more than this will likely crash the program
 #   if i == 0:
  #      a2p2 = np.load(DATA_PATH+'/temp/list_concats/a2p2_'+str(i)+'.npy')
   # else:
    #    try:
     #       x = np.load(DATA_PATH+'/temp/list_concats/a2p2_'+str(i)+'.npy')
      #      a2p2 = np.row_stack((a2p2,x))
       # except:
        #    pass

#np.save(DATA_PATH+'/temp/a2p2.npy',a2p2)
#del a2p2



def remove_dups(data_array):
    try:
        only_times = data_array[:,0]
        uniques, indices = np.unique(only_times,return_index=True)
        data_array = data_array[indices,:]
    except:
        pass
    return data_array

for i in range (0,batches):
    if i == 0:
        a1p1b = np.load(DATA_PATH+'/temp/list_concats/a1p1_binned_'+str(i)+'.npy')
        a1p2b = np.load(DATA_PATH+'/temp/list_concats/a1p2_binned_'+str(i)+'.npy')
        a2p1b = np.load(DATA_PATH+'/temp/list_concats/a2p1_binned_'+str(i)+'.npy')
        a2p2b = np.load(DATA_PATH+'/temp/list_concats/a2p2_binned_'+str(i)+'.npy')
    else:
        try:
            q = np.load(DATA_PATH+'/temp/list_concats/a1p1_binned_'+str(i)+'.npy')
            r = np.load(DATA_PATH+'/temp/list_concats/a1p2_binned_'+str(i)+'.npy')
            s = np.load(DATA_PATH+'/temp/list_concats/a2p1_binned_'+str(i)+'.npy')
            t = np.load(DATA_PATH+'/temp/list_concats/a2p2_binned_'+str(i)+'.npy')
            a1p1b = np.row_stack((a1p1b,q))
            a1p2b = np.row_stack((a1p2b,r))
            a2p1b = np.row_stack((a2p1b,s))
            a2p2b = np.row_stack((a2p2b,t))
        except:
            pass

try:
    del q, r, s, t
except:
    pass

a1p1b = remove_dups(a1p1b)
a1p2b = remove_dups(a1p2b)
a2p1b = remove_dups(a2p1b)
a2p2b = remove_dups(a2p2b)

a = np.where(np.isnan(a1p1b[:,0]))
a1p1b = np.delete(a1p1b,a,0)
a = np.where(np.isnan(a1p2b[:,0]))
a1p2b = np.delete(a1p2b,a,0)
a = np.where(np.isnan(a2p1b[:,0]))
a2p1b = np.delete(a2p1b,a,0)
a = np.where(np.isnan(a2p2b[:,0]))
a2p2b = np.delete(a2p2b,a,0)

np.save(DATA_PATH+'/temp/a1p1_binned.npy',a1p1b)
np.save(DATA_PATH+'/temp/a1p2_binned.npy',a1p2b)
np.save(DATA_PATH+'/temp/a2p1_binned.npy',a2p1b)
np.save(DATA_PATH+'/temp/a2p2_binned.npy',a2p2b)

time_diff = (float(a1p1b[-1,0]) - float(a1p1b[0,0])) / 60
DA = np.round((np.size(a1p1b[:,0])/60), decimals=2)
if time_diff < DA + 2 or time_diff > DA -2:
    np.save(DATA_PATH+'/temp/duration_actual', DA)
else:
    np.save(DATA_PATH+'/temp/duration_actual', np.round(time_diff,decimals=2))

a1p1b = a1p1b[:,3:]
a1p2b = a1p2b[:,3:]
a2p1b = a2p1b[:,3:]
a2p2b = a2p2b[:,3:]

np.save(DATA_PATH+'/temp/a1p1_power.npy', (a1p1b[:,112:569].mean(axis=1))) #power ratios applied  
np.save(DATA_PATH+'/temp/a1p2_power.npy', (a1p2b[:,112:569].mean(axis=1))) #power ratios applied  
np.save(DATA_PATH+'/temp/a2p1_power.npy', (a2p1b[:,112:569].mean(axis=1))) #power ratios applied  
np.save(DATA_PATH+'/temp/a2p2_power.npy', (a2p2b[:,112:569].mean(axis=1))) #power ratios applied  

np.save(DATA_PATH+'/temp/a1p1_bandpass.npy', (a1p1b)) 
np.save(DATA_PATH+'/temp/a1p2_bandpass.npy', (a1p2b)) 
np.save(DATA_PATH+'/temp/a2p1_bandpass.npy', (a2p1b)) 
np.save(DATA_PATH+'/temp/a2p2_bandpass.npy', (a2p2b)) 

print(np.shape(a1p1b))
print(np.shape(a1p2b))
print(np.shape(a2p1b))
print(np.shape(a2p2b))

del a1p1b,a2p1b,a1p2b,a2p2b

for i in range (0,batches):
    if i == 0:
        one_wire = np.load(DATA_PATH+'/temp/list_concats/one_wire_'+str(i)+'.npy')
    else:
        try:
            q = np.load(DATA_PATH+'/temp/list_concats/one_wire_'+str(i)+'.npy')
            one_wire = np.row_stack((one_wire,q))
            one_wire = remove_dups(one_wire)
        except:
            pass

np.save(DATA_PATH+'/temp/one_wire.npy',one_wire)
del one_wire

#CW_Power_Array_Generator()

gc.collect()

os.system('chmod -R -f 0777 /local5/scratch/pblack || true')

