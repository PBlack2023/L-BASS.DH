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
from astropy.io import fits
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
import math
from datetime import timedelta
import glob


os.chdir('/scratch/nas_lbass/raw_data/')

os.system('rm -r /mirror/scratch/pblack/temp/concats/')
os.system('mkdir /mirror/scratch/pblack/temp/concats/')

os.system('rm -r /mirror/scratch/pblack/temp/list_concats/')
os.system('mkdir /mirror/scratch/pblack/temp/list_concats/')

if os.path.exists(DATA_PATH+'/temp/a1p1.npy'):
    os.system('rm /mirror/scratch/pblack/temp/a1p1.npy')
if os.path.exists(DATA_PATH+'/temp/a1p2.npy'):
    os.system('rm /mirror/scratch/pblack/temp/a1p2.npy')
if os.path.exists(DATA_PATH+'/temp/a2p1.npy'):
    os.system('rm /mirror/scratch/pblack/temp/a2p1.npy')
if os.path.exists(DATA_PATH+'/temp/a2p2.npy'):
    os.system('rm /mirror/scratch/pblack/temp/a2p2.npy')
if os.path.exists(DATA_PATH+'/temp/obshdr.npy'):
    os.system('rm /mirror/scratch/pblack/temp/obshdr.npy')

#---------------------------------------------------------------------------

# aquire positional data, observer and mode

def getHeader(file_table):

   #reads the header of the fits file and extracts information about the observation, observatory and frequency

    print('')
    print('\033[0;0m Reading header')

    observatory = fits.open(file_table[0,0])
    try:
        get = fits.open(file_table[1,0])
    except:
        get = fits.open(file_table[0,0])
    HDR_MJD = float(get[2].header['MJDREF']) #change this to filetable[0,12]

    

    try:

        site = observatory[0].header['ORIGIN'] #name of the observatory
        observer = observatory[0].header['OBSERVER'] #who is using the telescope
        obs_mode = observatory[0].header['INSTRUME'] #observing mode
        #longitude = observatory[0].header['OBSGEO-L'] #telescope east longitude in deg
        longitude = 53.234338
        #latitude = observatory[0].header['OBSGEO-B'] #telescope north latitude in deg
        latitude = -2.305018
        altitude = observatory[0].header['OBSGEO-H'] #telescope height above sealevelin m
        obs_date = observatory[0].header['DATE'] #date file was created
        comment = observatory[0].header['COMMENT']

        try:
            azimuth = observatory[0].header['AZIMUTH'] #azimuth of LBASS frame in deg (usually 180)
        except:
            azimuth = 180
          #  print('\033[1;31m No Azimuth value recorded by user. Azimuth default of 180° applied. \033[0;0m')
    
# aquire data references and boundaries

        refPIX = observatory[1].header['1CRPX3'] #reference pixel number
        refFREQ = observatory[1].header['1CRVL3'] #reference frequency
        chnSPACE = observatory[1].header['1CDLT3'] # channel spacing/width = 450 MHz / 8192

        MINchnFREQ = 7394
        MAXchnFREQ = 6677 #corresponds to 1293.84 - 1433.22 MHz //// gives 718 channels

        obsheader = np.column_stack((site, observer, obs_mode, longitude, latitude, altitude, azimuth, obs_date, HDR_MJD))

        np.save(DATA_PATH+'/temp/obshdr', obsheader)
        np.save(DATA_PATH+'/temp/comment', comment)

    
        x = np.arange(0,718) #606 gives 1.427, 568 gives 1.425
        frequency = refFREQ + (chnSPACE * (x + 1 - refPIX))
        frequency = frequency / 1000000
    
        np.save(DATA_PATH+'/temp/freq', frequency)
        observatory.close()
        get.close()

    except:
        print('\033[1;31m Required data header field missing or corrupt. \033[0;0m')
     #####   bad_list = np.load(DATA_PATH+'/temp/bad_file.npy')
                
      #####  bad_list = np.append(bad_list, file_table[0,0])
     #####   np.save(DATA_PATH+'/temp/bad_file.npy', bad_list)
        time.sleep(1)

#--------------------------------------------------------------------------
# build the data table which incorporates information from all relecvant files
# THIS might be more length a process if there are lots of files to handle.
# the system records 2 values every 10th of a second

def routine1(opened, sample_time, switch_phase, elevation, channel, phase, nom,i):

    #opens the data tables of each fits file and extracts each rpg channel and phase and saves it in a numpy array

	too_many_bad_sampletimes = False
	channelR = np.array(opened[1].data[channel]) #radiometer data from chosen RPG channel
	channelR = channelR / 10**11  #reduce arbitrary units down to order 1
	channel_data = np.column_stack((sample_time, switch_phase, elevation, channelR))
	del channelR
	ind = np.where(channel_data[:,1] == phase) #select only indices of radiometer data in the chosen phase state
	channel_data = channel_data[ind[0],:]
#	print(nom,'number of samples:',np.size(channel_data[:,0]))
	del ind
	bad_samples=0
	#np.save(DATA_PATH+'/temp/concats/breaky_'+str(nom)+'_'+(str(i)), channel_data)  #save as numpy array
	Isdup = stats.mode(channel_data[:,0]) #checks for duplicate sample timecodes
	if int(Isdup[1]) > 1: 
#		print(nom,'Repeat second count value:',Isdup[0][0])
#		baddies = np.where(channel_data[:,0] == Isdup[0][0])
#		print(nom,'index of repeat value:',baddies)
		too_many_bad_sampletimes = True
	
	return too_many_bad_sampletimes, channel_data
#-------------------------------------------------------------------------------------             
def routine2(opened, i):

    #attempts to recover the times when one-wire temperature data was recorded.

	temp_time = np.array(opened[2].data['TIME'])

	temps = np.array(opened[2].data['One-wire temperatures'])

	temp_data = np.column_stack((temp_time[1:61],temps[1:61,:]))
	#np.save(DATA_PATH+'/temp/concats/tempdata_'+(str(i)), temp_data)

	return temp_data

#-----------------------------------------------------------------------
def buildTable1(file_table, batch):

    # a for loop which initiates the process of selecting each RPG channel and phase of the data.  This is repeated for each fits file and the data saved in seperate numpy arrays.

	print('Unpacking fits file data tables')
	time.sleep(1)
	i = 0
	for i in tqdm(range(0,np.size(file_table[:,0]))):  #, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
#	for i in range(0,np.size(file_table[:,0])):
		opened = fits.open(file_table[i,0])
		sample_time = np.array(opened[1].data['SAMPLE_TIME'])
		switch_phase = np.array(opened[1].data['SWITCH_PHASE'])
		elevation = np.array(opened[1].data['ELEVATION']) #greater than 90 doesn't change azimuth


		Isdup = stats.mode(sample_time[:]) #checks for duplicate sample timecodes
		if int(Isdup[1]) > 1: 
			print('Repeat count of seconds value:',Isdup[0][0])
			baddies = np.where(sample_time[:] == Isdup[0][0])
			print('Index of repeat value(s):',baddies,'of',np.size(sample_time[:]))  
		
        #radiometer data
		channels = ['CHANNEL1','CHANNEL2']  #channel1 is Left RPG, channel2 is Right RPG
		nom = ['a1p1','a1p2','a2p1','a2p2']
		for a in range(0,4):
			if a == 0 or a==1:
				channel = channels[0]
				if a == 0:
					phase = 1.0   #phase1 is pi state
				else:
					phase = 2.0    #phase2 is 0 state
			elif a == 2 or a ==3:
				channel = channels[1]
				if a == 2:
					phase = 1.0
				else:
					phase = 2.0
			
			busted, sorted_data = routine1(opened, sample_time, switch_phase, elevation, channel, phase, nom[a],i)

			if i == 0:
				if a == 0:
					a1p1 = sorted_data
				if a == 1:
					a1p2 = sorted_data
				if a == 2:
					a2p1 = sorted_data
				if a == 3:
					a2p2 = sorted_data

			else:
				if a == 0:
					a1p1 = np.row_stack((a1p1, sorted_data))
				if a == 1:
					a1p2 = np.row_stack((a1p2, sorted_data))
				if a == 2:
					a2p1 = np.row_stack((a2p1, sorted_data))
				if a == 3:
					a2p2 = np.row_stack((a2p2, sorted_data))


		#temperature data
		temperatures = routine2(opened,i)

		if i == 0:
			temp_data = temperatures
		else:
			temp_data = np.row_stack((temp_data, temperatures))


		if busted: #more than 10% of samples in data recorded with irregular timings between datapoints
			print('\033[1;31m Radiometer data recorded at irregular intervals in: \033[0;0m')
			print(file_table[i,0])
			#bad_list = np.load(DATA_PATH+'/temp/bad_file.npy')
			#bad_list = np.append(bad_list, file_table[i,0])
			#np.save(DATA_PATH+'/temp/bad_file.npy', bad_list)	#put that fits file on the naughty list
		opened.close()
		gc.collect()

	np.save(DATA_PATH+'/temp/concats/tempdata_'+(str(batch))+'.npy', temp_data)

	np.save(DATA_PATH+'/temp/concats/breaky_a1p1_'+(str(batch))+'.npy', a1p1)
	np.save(DATA_PATH+'/temp/concats/breaky_a1p2_'+(str(batch))+'.npy', a1p2)
	np.save(DATA_PATH+'/temp/concats/breaky_a2p1_'+(str(batch))+'.npy', a2p1)
	np.save(DATA_PATH+'/temp/concats/breaky_a2p2_'+(str(batch))+'.npy', a2p2)


    #nom = ['a1p1','a1p2','a2p1','a2p2']
	
	return i+1, nom 

#---------------------------------------------------------------------------

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

def buildTable3(cycles, kickoff, duration, batch, samples_per_min,start_at, end_at, rebinnable):


    # opens each numpy array of onewire temperature data and combines into a single array - which is then trimmed to match the user specified start time and duration.  


    print ('Aquiring RPG temperature data')
    ST = Time(start_at,format='mjd',scale='utc',precision=9)
    ET = Time(end_at,format='mjd',scale='utc',precision=9)
    start_path = ST.strftime('%Y_%m/')
    end_path = ET.strftime('%Y_%m/')

    if os.path.exists('/scratch/nas_lbass/binned_data/'+start_path+'RPG_temps.npy'):
        rpg_array = np.load('/scratch/nas_lbass/binned_data/'+start_path+'RPG_temps.npy')
    else:
        print('Failed to aquire RPG temperature data')

    if start_path == end_path:
        pass
    else:
        if os.path.exists('/scratch/nas_lbass/binned_data/'+end_path+'RPG_temps.npy'):
            q = np.load('/scratch/nas_lbass/binned_data/'+end_path+'RPG_temps.npy')
            try:
                rpg_array = np.row_stack((rpg_array,q))
            except:
                pass
    try:
        index_up = np.where(rpg_array[:,0] >= start_at)
        index_down = np.where(rpg_array[:,0] <= end_at)
        index_actual = np.intersect1d(index_up, index_down) #only times common to both lists
        rpg_array = rpg_array[index_actual,:]
        if np.size(rpg_array) <= 1:
            RPG=False
        else:
            np.save(DATA_PATH+'/temp/RPG_temps.npy',rpg_array)
            RPG = True
    except:
        RPG = False
    np.save(DATA_PATH+'/temp/RPG_exist.npy',RPG)

#    np.save(DATA_PATH+'/temp/concats/tempdata_'+(str(i)), temp_data)
    print ('Aquiring One-wire temperature data')
    i = 0
    one_wire=np.load(DATA_PATH+'/temp/concats/tempdata_'+str(batch)+'.npy')


    try: #trim the data array to match user specified start and end times
        kick_ind = np.where(one_wire[:,0] >= kickoff) #indices of data after start time
        one_wire = one_wire[kick_ind[0],:]
        dur_ind = np.where(one_wire[:,0] < duration)
        one_wire = one_wire[dur_ind[0],:]



    except: #load alternate values and trim to those instead
        kickoffB = np.load(DATA_PATH+'/temp/kickoffB.npy')
        durationB = np.load(DATA_PATH+'/temp/durationB.npy')
        
        kick_ind = np.where(one_wire[:,0] >= kickoffB) #indices of data after start time
        one_wire = one_wire[kick_ind[0],:]
        dur_ind = np.where(one_wire[:,0] < durationB)
        one_wire = one_wire[dur_ind[0],:]

    np.save(DATA_PATH+'/temp/one_wire', one_wire)
    
    #if np.mean(one_wire) == 0.0: #temp data not stored correctly
    #    one_wire[np.where(one_wire == 0.0)] = 'nan'

    i =0
    est=0
    gaps=[]
    adjust=[]
    nan_list=[]
 #   rebinnable=True
    nan_err = False

    for i in range (0, np.size(one_wire[:,0])):  #looks for missing sample times and inserts nans. necessary for correct array sizes.
        try:
            a = one_wire[i+1,0] - one_wire[i,0]
          #  if a == 120: # a single missing bin
            if a > 117 and a < 123: # a single missing bin plus small margn of error
                b=one_wire[i,0] + 60 
                nan_fill=([b,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan])
                nan_fill=np.array(nan_fill)
                one_wire = np.insert(one_wire, i+1, nan_fill, axis=0)
                print('\033[1;31m Missing sample time detected. Inserting NaNs.\033[0;0m')
                a=60
        except:
            try:
                a = one_wire[i,0] - one_wire[i-1,0]
            except:
                a=0
        
        if a > 63 or a < 57:  #checks to see if temperature data is recorded at regular intervals, allows for 5% drift
            drift = (a/60)*100
            print('\033[1;31m One-wire data recorded at irregular intervals. \033[0;0m')
            rebinnable=False
        else:
            pass
           # gaps.append(a)
            #adjust.append(a-60) #no longer needed really

        if np.sum(one_wire[i,1:]) == 0.0:
            if i == 0:
                print('\033[1;31m Fits file temperature records all show 0°C. \033[0;0m')
            nan_fill=([one_wire[i,0],nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan])
            one_wire[i,:] = nan_fill




    for i in range (0,np.size(one_wire[:,0])):  #f
        if np.isnan(one_wire[i,1]):
            try:
                if np.isnan(one_wire[i-1,1]) or np.isnan(one_wire[i+1,1]): #is it a run of nans?
                    nan_list.append(one_wire[i,0])
                    nan_err = True
                elif i == np.size(one_wire[:,0]):  #is the last data point a nan?
                    nan_list.append(one_wire[i,0])
                    nan_err = True      
                else: #if its a lone nan then replace with estimated value halfway between the last and next data points 
                    one_wire[i,1:] = (one_wire[i-1,1:] + one_wire[i+1,1:])/2 
                    est=est+1    
            except:
                pass
    
    if est > 0:  #inform user of the state of things
    	print(est,'sample times missing One-wire data. NaNs replaced with estimates.')
    if est/np.size(one_wire[:,0]) > 0.05:
        print('\033[1;31m One-wire data from more than 5% of sample times have been replaced by estimates. \033[0;0m')
    
    no_bins = np.size(one_wire[:,0])
    bin_width = samples_per_min #int(np.mean(gaps))*5 #5 samples per second  #300 samples per minute

    #if there is missing temperature data that wasnt replaced with estimates then search csv files for the data instead
    if nan_err:  
        print('\033[1;31m Fits files are missing some temperature data \033[0;0m')
        corrections = temp_csv(nan_list)
        for i in range(0,np.size(nan_list)):
            try:
                ind = np.where(one_wire == nan_list[i])
                one_wire[ind[0],1:] = corrections[i,:]

                success=True
            except:
                success=False
        for i in range (0,np.size(one_wire[:,0])):  #look for any remaining single nans that might be replaced by estimated values
        
            if np.isnan(one_wire[i,1]):
                try:
                    if np.isnan(one_wire[i-1,1]) or np.isnan(one_wire[i+1,1]):
                        success=False
                    elif i == np.size(one_wire[:,0]):
                        success=False   
                    else:
                        one_wire[i,1:] = (one_wire[i-1,1:] + one_wire[i+1,1:])/2 #halfway between the last and next data points
                        est=est+1
                except:
                    success=False

        if success:
            print('Temperature data successfully recovered.',est,'estimates used.')
        else:
            print('\033[1;31m Unrecoverable gaps remain in temperature data. \033[0;0m')


    np.save(DATA_PATH+'/temp/list_concats/one_wire_'+str(batch)+'.npy', one_wire)  #save the onewire data into a numpy array.
    gc.collect()

    return no_bins, bin_width, one_wire[:,0], rebinnable, adjust, one_wire

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
        try:
            if file_table[0,12] == file_table[1,12]: #same MJD
                firstSAM = float(file_table[0,5]) #first sample number of first file
                run_MJD = Time(file_table[0,12], format='mjd', scale='utc', precision=9)
            else:
                #np.delete(file_table, 0 ,axis=0)
                firstSAM = float(file_table[1,5]) #first sample number of first file is actually second fits file in the list
                run_MJD = Time(file_table[1,12], format='mjd', scale='utc', precision=9)

        except:
            firstSAM = float(file_table[0,5]) #first sample number of first file
            run_MJD = Time(file_table[0,12], format='mjd', scale='utc', precision=9)
    else:
        firstSAM = float(file_table[1,5]) 
        run_MJD = Time(file_table[1,12], format='mjd', scale='utc', precision=9)
        
    input_str = (str(user_date)+' '+str(user_inputs[3])+':00')
    
    user_datetime = Time(str(input_str), format='iso', scale='utc', out_subfmt='date_hm', precision=9) #creates user specified datetime object
    start_at = user_datetime.mjd
    dur = user_datetime + TimeDelta( (float(user_inputs[5]) * 3600), format='sec')
    end_at = dur.mjd
  
    
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
    return kickoff, duration, kick4temps, dur4temps, start_at, end_at
     
#---------------------------------------------------------------------

def binONEWIRE(nom, no_bins, bin_width, one_wire_sts, rebinnable, batch, samples_per_min):
           
    #averages the data over 1 minute periods, matched to the onewire data sample times.
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in true_divide')
  
    bin_width = int(samples_per_min)

    i = 0
    #sts means starts
    os.system('rm -r /mirror/scratch/pblack/temp/bina/')
    os.system('mkdir /mirror/scratch/pblack/temp/bina/')
    one_wire_sts = np.round(one_wire_sts, 1)
    parameters = np.load(DATA_PATH+'/temp/parameters.npy')

    j=0
    bad_bin=[]
    #nom j order:
    # A1P1 - P(l,pi)
    # A1P2 - P(l,0)
    # A2P1 - P(r,pi)
    # A2P2 - P(r,0)
    
    for j in range(0,4):
        
        i = 0 
        armPX = np.load(DATA_PATH+'/temp/'+str(nom[j])+'.npy') #open the radiometer data array
        #print(nom[j],'   ',np.shape(armPX)) #checks how many samples in each phase
        bad_samples=0
        bad_flag=[]
        just_sample_times = np.round(armPX[:,0], 1) #will be one for each minute, plus one minute - radiometer time codes
                  
        if int(samples_per_min) == 300:
            Isdup = stats.mode(just_sample_times) #checks for duplicate sample timecodes
            if int(Isdup[1]) > 1: #if there are irregularities then...
                for i in range (0,np.size(just_sample_times)):  #deals with the midnight problem - incorrect sample times recorded
                    try:
                        f = just_sample_times[i+1] - just_sample_times[i]   
                        if f > 0.3 or f < 0.1:  # gap between samples should be 0.2 if data taken at 600 Hz
                            bad_flag.append(True)
                            bad_samples = bad_samples +1 
                        else:
                            bad_flag.append(False) 
                    except:
                        f = just_sample_times[i] - just_sample_times[i-1]  # if its the last sample time then look back rather than forward
                        if f > 0.3 or f < 0.1:
                            bad_flag.append(True)
                            bad_samples = bad_samples +1
                        else:
                            bad_flag.append(False)
        
                for i in range (0,np.size(bad_flag)):
                    if bad_flag[i] and i == 0:  #no correction can be applied to the first sample time
                        pass
                    elif bad_flag[i] and i>0: 
                        just_sample_times[i] = just_sample_times[i-1] + 0.2  #overright bad sample time with correct step in time
                        armPX[i,0] = armPX[i-1,0] + 0.2

        missing_samples=[]

        for i in tqdm(range(0,no_bins)):#, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'): #rebin 1 min averages
      #  for i in range(0,no_bins): #rebin 1 min averages
        #for i in range (0,no_bins):
            best_match = find_nearest(just_sample_times, one_wire_sts[i]) #closest radiometer sample time to one_wire sample time
            bin_centre = np.where(just_sample_times == best_match) 
           # if i == 0:
            #    print('bin centre ',bin_centre[0])
            #print(best_match - original)
            try:
                if np.abs(one_wire_sts[i] - best_match) < 15: #less than 15 second gap allow binning -- should this be 6??
                    pass
                else:
                    f=1/0 #to cause an exception deliberately if the gap is too large
                a = int(bin_centre[0])-int(bin_width/2)
                b = int(bin_centre[0])+int(bin_width/2)
                if i == 0:
                    binned = armPX[a:b,:].mean(axis=0)
                    if np.size(binned) < (bin_width *0.75):
                        missing_samples.append(np.size(binned))
                    #print (np.size(armPX[a:b,0]))
                    d = b
                else:
                    if d > a: #stops the same sample being averaged into two different one-minute bins
                        a = d
                    #print (np.size(armPX[a:b,0]))
                    if np.size(armPX[a:b,0]) < (bin_width *0.75):
                        missing_samples.append(np.size(binned))
                    binned = np.row_stack((binned, armPX[a:b,:].mean(axis=0)))
                    d = b

            except:
                rebinnable=False
     #           try:
      #              tup_ex = bin_centre[0]
       #             tup_ex = int(np.mean(tup_ex))
        #            a = int(tup_ex)-int(bin_width/2)
         #           b = int(tup_ex)+int(bin_width/2)
          #          np.save(DATA_PATH+'/temp/bina/'+str(nom[j])+'_bin'+str(i), armPX[a:b,:].mean(axis=0))
           #         d = b
            #    except:
             #       bad_bin.append(i)
              #      empty = np.empty((1,721))
               #     np.save(DATA_PATH+'/temp/bina/'+str(nom[j])+'_bin'+str(i), empty)
                #rebinnable=False 
            gc.collect()

        if np.size(missing_samples) > 0:
            print ('\033[1;31m',np.size(missing_samples),'bins in this batch were missing 25% or more of the expected samples.\033[0;0m')
        
        try:
            #with rebinning complete, save as numpy arrays 

            np.save(DATA_PATH+'/temp/'+str(nom[j])+'.npy', binned) #includes time phase and elevation.

            del binned, armPX
        except:
            del armPX
            rebinnable=False
            pass


        gc.collect()

    if np.size(bad_bin) / (no_bins*4) > 0.05:
        rebinnable = False

    if rebinnable:
        pass
    else:
        print ('\033[1;31m Binning process failed. \033[0;0m')
    gc.collect()
   # rebinnable = True
    return rebinnable

#---------------------------------------------------------


def binINminsX(nom,ts,samples_per_min):
    
    #bins the data into 1 minute bins, but does not attempt to match up to the onewire datapoints

    bin_width = int(samples_per_min)
    parameters = np.load(DATA_PATH+'/temp/parameters.npy')
   
    print('\033[0;m Rebinning radiometer data, 1 minute bins.')
    
    i = 0
    

    os.system('rm -r /mirror/scratch/pblack/temp/bina/')
    os.system('mkdir /mirror/scratch/pblack/temp/bina/')
    
    
    #take mean of each column in given range to average for that minute of data
    
    obsheader = np.load(DATA_PATH+'/temp/obshdr.npy')
    MJD = Time(obsheader[0,8],format='mjd', scale='utc', precision=9)
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    USRDT = str(user_inputs[0])+' '+str(user_inputs[1])+':'+str(user_inputs[2]+':00.0')
    USRTM = Time(USRDT,format='iso', scale='utc', precision=9)
    TD = USRTM - MJD
    TD_secs = int(TD.sec) #time delta of seconds passed between midnight on mjd and the user specified time
    UserDur = int((float(user_inputs[5]) * 60)) #duration in mins 
    j=0
    for j in range(0,4):

        armPX = np.load(DATA_PATH+'/temp/'+str(nom[j])+'.npy')
        bad_samples=0
        bad_flag=[]
       

        missing_samples=[]
        just_sample_times = np.round(armPX[:,0], 1)

        i = 0 
   #     a = range((TD_secs - 30),((TD_secs - 30) + (60 * UserDur)),60)  #centre on the turn of the minute (0:0:00)
    #    b = range((TD_secs + 30),((TD_secs + 30) + (60 * UserDur)),60)
        a = range((TD_secs),((TD_secs) + (60 * UserDur)),60)   #centre on the middle of the minute (0:0:30 secs in)
        b = range((TD_secs + 60),((TD_secs + 60) + (60 * UserDur)),60)

        for i in tqdm(range(0,no_bins)):#, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
      #  for i in range(0,no_bins):
            if i ==0:
                this_bin = np.where(armPX[:,0] >= a[i])
                also_this_bin = np.where(armPX[:,0] < b[i])
                binned = armPX[this_bin[0][0]:also_this_bin[0][-1],:].mean(axis=0)
                if np.size(binned) < (bin_width *0.75):
                    missing_samples.append(np.size(binned))
            else:
                try:
                    this_bin = np.where(armPX[:,0] >= a[i])
                    also_this_bin = np.where(armPX[:,0] < b[i])
                    if np.isnan(armPX[this_bin[0][0]:also_this_bin[0][-1],0].mean(axis=0)):
                        pass
                    else:
                        kim = armPX[this_bin[0][0]:also_this_bin[0][-1],0]
                        if np.size(kim) < (bin_width *0.75):
                            missing_samples.append(np.size(kim))
                        binned = np.row_stack((binned, armPX[this_bin[0][0]:also_this_bin[0][-1],:].mean(axis=0)))
                except:
                    pass
        del armPX

      #  binned = binned[:UserDur,:] #restrict list to required duration.  !!!!!!!!!! is this a problem???????

        if np.size(missing_samples) > 0:
            print ('\033[1;31m',np.size(missing_samples),'bins in this batch were missing 25% or more of the expected samples.\033[0;0m')
        
        np.save(DATA_PATH+'/temp/'+str(nom[j])+'.npy', binned) #includes time phase and elevation.

    #del binned
    gc.collect()

#---------------------------------------------------------

def temp_csv(nan_list):

    #looks in the onewire csv files to try and find missing temperature data if it isn't in the fits files

    if os.path.exists(DATA_PATH+'/temp/temp_data.npy'):
        os.system('rm /mirror/scratch/pblack/temp/temp_data.npy')
        
    temp_files = []
    temp_files = sorted(glob.glob('1W_Temp-*.csv'))
 
    df_from_each_file = (pd.read_csv(f) for f in temp_files)
    concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)
    #concatenated_df.to_csv(DATA_PATH+'/temp/temp_data.csv', index=False)
    del df_from_each_file
    all_temps = concatenated_df.to_numpy()
    del concatenated_df
    
    file_table = np.load(DATA_PATH+'/temp/file1.npy', allow_pickle=True)
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy', allow_pickle=True)
    nan_list = np.array(nan_list)
    datetime_list=[]
   # mjd + sampletime ... into iso ... search for matches ... overwrite nans
    try:
        MJD = Time(file_table[1,12], format='mjd', scale='utc', precision=9)
    except:
        MJD = Time(file_table[0,12], format='mjd', scale='utc', precision=9)

  
    i=0
    for i in range (0,np.size(nan_list)):
        sample_to_time = MJD + TimeDelta(nan_list[i], format='sec')
        time_iso = Time(sample_to_time, format='iso', scale='utc', precision=9, out_subfmt='date_hm')
        datetime_list.append(str(time_iso))

    for i in range (0,np.size(all_temps[:,0])): #what is this for????? ... but don't delete, it works!
        try:
            a = all_temps[i,0]
            b= a[:-3]
            all_temps[i,0] = b
        except:
            pass
    alltemps_list=[]
    for i in range (0,np.size(all_temps[:,0])):
        alltemps_list.append(all_temps[i,0])    


    nan_fill=([nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan])
    nan_fill=np.array(nan_fill)
    print('\033[0;0m Searching One-wire csv files for missing data')
    time.sleep(1)
    corrections=[]
    problem=0
    
    index_match = find_matching_index(alltemps_list, datetime_list)
    
    ind1=[]
    ind2=[]
    try:
        for i in range (len(index_match)):
            ind1.append(int(index_match[i][0]))
            ind2.append(int(index_match[i][1]))  

        if len(index_match) == np.size(nan_list):
            pass
        else:
            for i in range (np.size(nan_list)):
                if i == ind1[i]:
                    pass
                else:
                    ind1.insert(i,i)
                    ind2.insert(i,None)
    except:
        pass

    for i in tqdm(range(0,np.size(nan_list))):#, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
 #   for i in range(0,np.size(nan_list)):
        try:
            if i == 0 and ind2[0] != None :            
                corrections = all_temps[0,ind2[0],1:]
     
            elif i == 0 and ind2[0] == None:
                corrections.append(nan_fill)

            elif ind2[i] != None:
                corrections = np.row_stack((corrections, all_temps[ind2[i],1:]))

            elif ind2[i] == None:
                corrections = np.row_stack((corrections, nan_fill))
                problem = problem + 1
  
        except:
            if i == 0:
                corrections.append(nan_fill)
            else:
                corrections = np.row_stack((corrections, nan_fill))
    
    del all_temps           
    gc.collect()

    return corrections

#--------------------------------------------------------------------------------------------

def find_matching_index(list1, list2):

    #rapid searching for matching values between two lists (eg onewire times in csv vs fits files)

    inverse_index = { element: index for index, element in enumerate(list1) }

    return [(index, inverse_index[element])
        for index, element in enumerate(list2) if element in inverse_index]

#-----------------------------------------------------------------------------

def find_nearest(array, value):  #used to find nearest matching values between onewire and radiometer sample times for binning
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#-------------------------------------------------------

def CW_Power_Array_Generator(hornchoice='west'):

    #based on code by Jordan Norris
    """
    Reads in and performs a background subtraction on the CW signal region. Has
    range over a few indices incase the CW signal is split between channels in
    the future. Saves the .npy files of the CW data in a similar method to
    a1p1_powers.npy

    Returns
    -------
    signal1 : 
        DESCRIPTION.
    signal2 : TYPE
        DESCRIPTION.

    """
    if os.path.exists(DATA_PATH+'/temp/a1p1_cw.npy'):
        os.system('rm /mirror/scratch/pblack/temp/a1p1_cw.npy')
    if os.path.exists(DATA_PATH+'/temp/a1p2_cw.npy'):
        os.system('rm /mirror/scratch/pblack/temp/a1p2_cw.npy')
    if os.path.exists(DATA_PATH+'/temp/a2p1_cw.npy'):
        os.system('rm /mirror/scratch/pblack/temp/a2p1_cw.npy')
    if os.path.exists(DATA_PATH+'/temp/a2p2_cw.npy'):
        os.system('rm /mirror/scratch/pblack/temp/a2p2_cw.npy')

    frequency = np.load('/mirror/scratch/pblack/temp/freq.npy')


    
    signal1, signal2 = [], []
    CW_present = [False, False]
    hornchoice = 'west'
    if hornchoice.lower() == 'west':
        a1p1b = np.load('/mirror/scratch/pblack/temp/a1p1_binned.npy')
        a1p1b = a1p1b[:,3:]
        a2p2b = np.load('/mirror/scratch/pblack/temp/a2p2_binned.npy')
        a2p2b = a2p2b[:,3:]

        CW_bin_W = []
        CW_frequency_W = []
        CW_power_W = []
        for i in range (0,np.size(a1p1b[:,0])):
            this_it_bin = np.argmax(a1p1b[i,:])
            this_it_signal = np.sum(a1p1b[i,int(this_it_bin)-2:int(this_it_bin)+3])
            if this_it_signal / (np.mean(a1p1b[:,112:569])) > 5: #5 times larger than anything else on the bandpass
            
                this_it_baseline1 = np.average(a1p1b[i,int(this_it_bin)-5:int(this_it_bin)-2])
                this_it_baseline2 = np.average(a1p1b[i,int(this_it_bin)+3:int(this_it_bin)+6])
                this_it_baseline = (this_it_baseline1 + this_it_baseline2) / 2
                this_it_power = this_it_signal - (this_it_baseline * 5) #CW integrated over 5 channels, so subtract 5 lots of baseline)
                CW_power_W.append(this_it_power)
                CW_bin_W.append(int(this_it_bin))
                CW_frequency_W.append(float(frequency[this_it_bin]))

            else:
                CW_power_W.append(0)
                CW_bin_W.append(0)
                CW_frequency_W.append(np.nan)
 
        np.save('/mirror/scratch/pblack/temp/a1p1_cw_frequency.npy',np.asarray(CW_frequency_W))
        np.save('/mirror/scratch/pblack/temp/a1p1_cw_binned.npy', CW_power_W)
        np.save('/mirror/scratch/pblack/temp/a1p1_cw_binNO.npy', CW_bin_W)

        if np.sum(CW_power_W) > 0:
            print('CW signal detected in P(L,\u03C0) W')
            CW_present = [False, True]

        CW_power_W = []
        CW_bin_W = []
        CW_frequency_W = []

        for i in range (0,np.size(a2p2b[:,0])):
            this_it_bin = np.argmax(a2p2b[i,:])
            this_it_signal = np.sum(a2p2b[i,int(this_it_bin)-2:int(this_it_bin)+3])
            if this_it_signal / (np.mean(a2p2b[:,112:569])) > 5: #5 times larger than anything else on the bandpass
            
                this_it_baseline1 = np.average(a2p2b[i,int(this_it_bin)-5:int(this_it_bin)-2])
                this_it_baseline2 = np.average(a2p2b[i,int(this_it_bin)+3:int(this_it_bin)+6])
                this_it_baseline = (this_it_baseline1 + this_it_baseline2) / 2
                this_it_power = this_it_signal - (this_it_baseline * 5) #CW integrated over 5 channels, so subtract 5 lots of baseline)
                CW_power_W.append(this_it_power)
                CW_bin_W.append(int(this_it_bin))
                CW_frequency_W.append(float(frequency[this_it_bin]))

            else:
                CW_power_W.append(0)
                CW_bin_W.append(0)
                CW_frequency_W.append(np.nan)

        np.save('/mirror/scratch/pblack/temp/a2p2_cw_frequency.npy',np.asarray(CW_frequency_W))
        np.save('/mirror/scratch/pblack/temp/a2p2_cw_binned.npy', CW_power_W)
        np.save('/mirror/scratch/pblack/temp/a2p2_cw_binNO.npy', CW_bin_W)

        if np.sum(CW_power_W) > 0:
            print('CW signal detected in P(R,0) W')
            CW_present = [False, True]
 


       

              
    hornchoice = 'east'     
    if hornchoice.lower() == 'east':
        a1p2b = np.load('/mirror/scratch/pblack/temp/a1p2_binned.npy')#----
        a1p2b = a1p2b[:,3:]
        a2p1b = np.load('/mirror/scratch/pblack/temp/a2p1_binned.npy')#----
        a2p1b = a2p1b[:,3:]


        CW_bin_E = []
        CW_frequency_E = []
        CW_power_E = []
        for i in range (0,np.size(a1p2b[:,0])):
            this_it_bin = np.argmax(a1p2b[i,:])
            this_it_signal = np.sum(a1p2b[i,int(this_it_bin)-2:int(this_it_bin)+3])
            if this_it_signal / (np.mean(a1p2b[:,112:569])) > 5: #5 times larger than anything else on the bandpass
            
                this_it_baseline1 = np.average(a1p2b[i,int(this_it_bin)-5:int(this_it_bin)-2])
                this_it_baseline2 = np.average(a1p2b[i,int(this_it_bin)+3:int(this_it_bin)+6])
                this_it_baseline = (this_it_baseline1 + this_it_baseline2) / 2
                this_it_power = this_it_signal - (this_it_baseline * 5) #CW integrated over 5 channels, so subtract 5 lots of baseline)
                CW_power_E.append(this_it_power)
                CW_bin_E.append(int(this_it_bin))
                CW_frequency_E.append(float(frequency[this_it_bin]))

            else:
                CW_power_E.append(0)
                CW_bin_E.append(0)
                CW_frequency_E.append(np.nan)

        np.save('/mirror/scratch/pblack/temp/a1p2_cw_frequency.npy',np.asarray(CW_frequency_E))
        np.save('/mirror/scratch/pblack/temp/a1p2_cw_binned.npy', CW_power_E)
        np.save('/mirror/scratch/pblack/temp/a1p2_cw_binNO.npy', CW_bin_E)

        if np.sum(CW_power_E) > 0:
            print('CW signal detected in P(L,0) E')
            if CW_present[1] == True:
                CW_present = [True,True]
            else:
                CW_present = [True,False]

        CW_power_E = []
        CW_bin_E = []
        CW_frequency_E = []


        for i in range (0,np.size(a2p1b[:,0])):
            this_it_bin = np.argmax(a2p1b[i,:])
            this_it_signal = np.sum(a2p1b[i,int(this_it_bin)-2:int(this_it_bin)+3])
            if this_it_signal / (np.mean(a2p1b[:,112:569])) > 5: #5 times larger than anything else on the bandpass
          
                this_it_baseline1 = np.average(a2p1b[i,int(this_it_bin)-5:int(this_it_bin)-2])
                this_it_baseline2 = np.average(a2p1b[i,int(this_it_bin)+3:int(this_it_bin)+6])
                this_it_baseline = (this_it_baseline1 + this_it_baseline2) / 2
                this_it_power = this_it_signal - (this_it_baseline * 5) #CW integrated over 5 channels, so subtract 5 lots of baseline)
                CW_power_E.append(this_it_power)
                CW_bin_E.append(int(this_it_bin))
                CW_frequency_E.append(float(frequency[this_it_bin]))

            else:
                CW_power_E.append(0)
                CW_bin_E.append(0)
                CW_frequency_E.append(np.nan)

        np.save('/mirror/scratch/pblack/temp/a2p1_cw_frequency.npy',np.asarray(CW_frequency_E))
        np.save('/mirror/scratch/pblack/temp/a2p1_cw_binned.npy', CW_power_E)
        np.save('/mirror/scratch/pblack/temp/a2p1_cw_binNO.npy', CW_bin_E)

        if np.sum(CW_power_E) > 0:
            print('CW signal detected in P(R,\u03C0) E')
            if CW_present[1] == True:
                CW_present = [True,True]
            else:
                CW_present = [True,False]


    np.save('/mirror/scratch/pblack/temp/CW_Present.npy', CW_present)
    
    if CW_present[0] == False and CW_present[1] == False:
        print ('CW Signal not detected')
    
    return signal1, signal2

#==========================================================================

def sample_rate_check():

    try:
        try:
            data_sample = np.load(DATA_PATH+'/temp/concats/breaky_a1p1_1.npy')
        except:
            data_sample = np.load(DATA_PATH+'/temp/concats/breaky_a1p1_0.npy')
        data_sample = data_sample[:,0] #seconds
        sample_diffs = np.diff(data_sample) 

        sample_rate = int(60/((np.mean(sample_diffs))/2))+1

        if ((sample_rate)/60) < 0 or ((sample_rate)/60) > 100:
            print('\033[1;31m Implausible mean sampling rate detected. \033[0;0m')
            mode_rate = stats.mode(sample_diffs)
            mode_rate = float(mode_rate[0])
            mode_rate = (1/mode_rate)*2
            if mode_rate > 100 or mode_rate < 0:
                print('\033[1;31m Implausible modal sampling rate detected. \033[0;0m')
                rebinnable = False
            else:
                print('Modal sample rate detected:',((mode_rate)),'Hz')
                samples_per_min = (mode_rate*60)/2
                rebinnable = True
        else:
            print('Mean sample rate detected:',((sample_rate)/60),'Hz')
            samples_per_min = (sample_rate / 2)
            rebinnable = True

    except:
        print('\033[1;31m Unable to detect sample rate. Default of 10 Hz applied. \033[0;0m')
        samples_per_min = 300
        rebinnable = True

    return samples_per_min, rebinnable

#±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±

#§§§§§§§

def remove_dups(data_array):
    try:
        only_times = data_array[:,0]
        uniques, indices = np.unique(only_times,return_index=True)
        data_array = data_array[indices,:]
    except:
        only_times = data_array[:] #1D list
        uniques, indices = np.unique(only_times,return_index=True)
        data_array = data_array[indices]
    return data_array

#§§§§§§§§§§§

def smooth_loader():

    roaring_success = True
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    smooth_string = user_inputs[7]
    iso_string = str(user_inputs[0])+' '+str(user_inputs[1])+':'+str(user_inputs[2])+':00.0'  
    start_time = Time(iso_string, format='iso', scale='utc', precision=9)
    end_time = start_time + TimeDelta((float(user_inputs[5]) * 3600), format='sec')
    start_month = start_time.strftime('%Y_%m/')
    end_month = end_time.strftime('%Y_%m/')
    expected_duration = float(user_inputs[5]) * 60

    if os.path.exists(smooth_string+'a1p1.npy'):

        a1p1_array = np.load(smooth_string+'a1p1.npy') 
        a1p2_array = np.load(smooth_string+'a1p2.npy') 
        a2p1_array = np.load(smooth_string+'a2p1.npy') 
        a2p2_array = np.load(smooth_string+'a2p2.npy') 
        time_array = np.load(smooth_string+'time_array.npy')
        temp_array = np.load(smooth_string+'one_wire.npy')
        temp_time_array = np.load(smooth_string+'temp_time_array.npy')
        try:
            rpg_array = np.load(smooth_string+'RPG_temps.npy')
        except:
            pass

    else:
        print('Smoothed data for this date has not yet been created.')
        roaring_success = False

    if start_month == end_month:
        pass
    else:
        if os.path.exists('/scratch/nas_lbass/binned_data/'+end_month+'a1p1.npy'):
            q = np.load('/scratch/nas_lbass/binned_data/'+end_month+'a1p1.npy') 
            r = np.load('/scratch/nas_lbass/binned_data/'+end_month+'a1p2.npy') 
            s = np.load('/scratch/nas_lbass/binned_data/'+end_month+'a2p1.npy') 
            t = np.load('/scratch/nas_lbass/binned_data/'+end_month+'a2p2.npy') 
            u = np.load('/scratch/nas_lbass/binned_data/'+end_month+'time_array.npy')    
            v = np.load('/scratch/nas_lbass/binned_data/'+end_month+'one_wire.npy')
            w = np.load('/scratch/nas_lbass/binned_data/'+end_month+'temp_time_array.npy')
            try:
                ww = np.load('/scratch/nas_lbass/binned_data/'+end_month+'RPG_temps.npy')
            except:
                pass
        
            a1p1_array = np.row_stack((a1p1_array,q))
            a1p2_array = np.row_stack((a1p2_array,r))
            a2p1_array = np.row_stack((a2p1_array,s))
            a2p2_array = np.row_stack((a2p2_array,t))
            time_array = np.append(time_array,u)
            temp_array = np.row_stack((temp_array,v))
            temp_time_array = np.append(temp_time_array,w)
            try:
                rpg_array = np.row_stack((rpg_array,ww))
            except:
                pass

        
            del q, r, s, t, u, v, w
            
        else:
            pass
       #     print('Available exported data does not extend to the full duration requested.')
         #   skip = input('Attempt to load and process unsmoothed data instead? (Y/N): ')
          #  if str(skip) == 'y' or str(skip) == 'Y':
           #     roaring_success = False
            #else:
             #   pass
   # print(temp_time_array)


    try:

        index_up = np.where(time_array >= start_time.mjd)
        index_down = np.where(time_array <= end_time.mjd)
        index_actual = np.intersect1d(index_up, index_down) #only times common to both lists
        try:
            time_array = time_array[index_actual]
        except:
            pass

        data_limit = Time(time_array[-1],format='mjd',scale='utc',precision=9)
        time_diff = end_time - data_limit
        time_diff = float(time_diff.sec)

        if time_diff > 300:
            print('Available exported data does not extend to the full duration requested.')
            print('  > Short by approximately',np.round((time_diff/3600),decimals=2),'hours of data.')
          #  skip = input('Attempt to load and process unsmoothed data instead? (Y/N): ')
          #  if str(skip) == 'y' or str(skip) == 'Y':
           #     roaring_success = False
 
        a1p1_array = a1p1_array[index_actual,:]
        a1p2_array = a1p2_array[index_actual,:]
        a2p1_array = a2p1_array[index_actual,:]
        a2p2_array = a2p2_array[index_actual,:]


        if (float(expected_duration) - np.size(a1p1_array[:,0])) >= 5:
            print('There are',np.round(((expected_duration - np.size(a1p1_array[:,0])) - (time_diff/60)),decimals=0),'minutes with no data available.')
            print('  > It may be no data was taken or no processed data has been exported yet.')
          #  skip = input('Attempt to load and process unsmoothed data instead? (Y/N): ')
           # if str(skip) == 'y' or str(skip) == 'Y':
            #    roaring_success = False

        # 0.00069444444 1minute MJD
        # 0.00081018519 70 seconds MJD
        jeff = np.diff(time_array)
        insert_index = np.where(jeff > 0.00081018519) #more than 60 seconds gap (plus 16% margin of error)
        for i in range (np.size(insert_index[0][:])):
            empty = np.empty((1,721))
            empty[0,1:] = np.nan
            if i == 0:

                time_forbump = Time(time_array[insert_index[0][i]],format='mjd',scale='utc',precision=9)
                time_bump = time_forbump + TimeDelta(60, format='sec')
                time_array = np.insert(time_array,(insert_index[0][i]),time_bump.mjd,axis=0)

                empty[0,0] = (a1p1_array[(insert_index[0][i]),0] + 60) #create a timecode for the nan 
                a1p1_array = np.insert(a1p1_array,(insert_index[0][i]),empty,axis=0)
                a1p2_array = np.insert(a1p2_array,(insert_index[0][i]),empty,axis=0)
                a2p1_array = np.insert(a2p1_array,(insert_index[0][i]),empty,axis=0)
                a2p2_array = np.insert(a2p2_array,(insert_index[0][i]),empty,axis=0)

            else: #each insertion changes the length of array and therefore index needs adjusting
                time_forbump = Time(time_array[insert_index[0][i]+i],format='mjd',scale='utc',precision=9)
                time_bump = time_forbump + TimeDelta(60, format='sec')
                time_array = np.insert(time_array,(insert_index[0][i])+i,time_bump.mjd,axis=0) 

                empty[0,0] = (a1p1_array[(insert_index[0][i])+i,0] + 60) #create a timecode for the nan 
                a1p1_array = np.insert(a1p1_array,(insert_index[0][i])+i,empty,axis=0)
                a1p2_array = np.insert(a1p2_array,(insert_index[0][i])+i,empty,axis=0)
                a2p1_array = np.insert(a2p1_array,(insert_index[0][i])+i,empty,axis=0)
                a2p2_array = np.insert(a2p2_array,(insert_index[0][i])+i,empty,axis=0)

        a1p1_array = a1p1_array[time_array[:].argsort()] #sort the array by time in case above routine shuffles things about
        a1p2_array = a1p2_array[time_array[:].argsort()] 
        a2p1_array = a2p1_array[time_array[:].argsort()] 
        a2p2_array = a2p2_array[time_array[:].argsort()] 
        time_array = time_array[time_array[:].argsort()]

        if os.path.exists(DATA_PATH+'/temp/time_array.npy'):
            os.system('rm /mirror/scratch/pblack/temp/time_array.npy')
        try:
            np.save(DATA_PATH+'/temp/time_array.npy',np.asarray(time_array))
        except:
            pass

        index_up = np.where(temp_time_array >= start_time.mjd)
        index_down = np.where(temp_time_array <= end_time.mjd)
        index_actual = np.intersect1d(index_up, index_down) #only times common to both lists

        temp_time_array = temp_time_array[index_actual]
        temp_array = temp_array[index_actual,:]

        diff = np.diff(temp_array[:,0])
        insert_index = np.where(diff > 70) #more than 60 seconds gap (plus 16% margin of error)

 
        for i in range (np.size(insert_index[0][:])):
            tempty = np.empty((1,39))
            tempty[0,:] = np.nan
            if i == 0:
                time_forbump = Time(temp_time_array[insert_index[0][i]+1],format='mjd',scale='utc',precision=9)
                time_bump = time_forbump + TimeDelta(60, format='sec')
                temp_time_array = np.insert(temp_time_array,(insert_index[0][i]+1),time_bump.mjd,axis=0)
                tempty[0,0] = (temp_array[(insert_index[0][i]+1),0] + 60) #create a timecode for the nan 
                temp_array = np.insert(temp_array,(insert_index[0][i]+1),tempty,axis=0)

            else: #each insertion changes the length of array and therefore index needs adjusting

                time_forbump = Time(temp_time_array[insert_index[0][i]+i+1],format='mjd',scale='utc',precision=9)
                time_bump = time_forbump + TimeDelta(60, format='sec')
                temp_time_array = np.insert(temp_time_array,(insert_index[0][i])+i+1,time_bump.mjd,axis=0)

                tempty[0,0] = (temp_array[(insert_index[0][i])+i+1,0] + 60) #create a timecode for the nan 
                temp_array = np.insert(temp_array,(insert_index[0][i])+i+1,tempty,axis=0)

        temp_array = temp_array[temp_time_array[:].argsort()] #sort the array by time
        temp_time_array = temp_time_array[temp_time_array[:].argsort()]

        if os.path.exists(DATA_PATH+'/temp/temp_time_array.npy'):
            os.system('rm /mirror/scratch/pblack/temp/temp_time_array.npy')
        try:
            np.save(DATA_PATH+'/temp/temp_time_array.npy',np.asarray(temp_time_array))
        except:
            pass

        np.save(DATA_PATH+'/temp/one_wire.npy',temp_array)

        try:
            index_up = np.where(rpg_array[:,0] >= start_time.mjd)
            index_down = np.where(rpg_array[:,0] <= end_time.mjd)
            index_actual = np.intersect1d(index_up, index_down) #only times common to both lists

            rpg_array = rpg_array[index_actual,:]
            if np.size(rpg_array) <= 1:
                RPG=False
            else:
                np.save(DATA_PATH+'/temp/RPG_temps.npy',rpg_array)
                RPG = True
        except:
            RPG = False
        np.save(DATA_PATH+'/temp/RPG_exist.npy',RPG)

        DA_t1 = Time(time_array[0],format='mjd',scale='utc',precision=9)
        DA_t2 = Time(time_array[-1],format='mjd',scale='utc',precision=9)
        DA_diff = DA_t2 - DA_t1
        time_diff = DA_diff.sec / 3600
        np.save(DATA_PATH+'/temp/duration_actual', np.round(time_diff,decimals=2))

        no_power = np.where(a1p1_array[:,480] < 0.000001)
        a1p1_array[no_power,3:] = np.nan
        no_power = np.where(a1p2_array[:,480] < 0.000001)
        a1p2_array[no_power,3:] = np.nan
        no_power = np.where(a2p1_array[:,480] < 0.000001)
        a2p1_array[no_power,3:] = np.nan
        no_power = np.where(a2p2_array[:,480] < 0.000001)
        a2p2_array[no_power,3:] = np.nan

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

        del a1p1_array, a1p2_array, a2p1_array,a2p2_array,temp_array,time_array

    except:
        roaring_success = False

    return roaring_success


#±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±

#--------------------------------------------------------------------

# load in the user selections and related fits / csv files

file_table = np.load(DATA_PATH+'/temp/file1.npy', allow_pickle=True)  #all relevant fits files
user_inputs = np.load(DATA_PATH+'/temp/inputs.npy', allow_pickle=True) 

if os.path.exists(user_inputs[7]):
    pass
else:
    os.system('mkdir '+user_inputs[7])

getHeader(file_table)

quickload = np.load(DATA_PATH+'/temp/quickload.npy')
raw_samples = np.load(DATA_PATH+'/temp/raw_samples.npy')
parameters = np.load(DATA_PATH+'/temp/parameters.npy')

if quickload:

    roaring_success = smooth_loader()
else:
    roaring_success = False

if roaring_success:
    pass
else:
    
    kickoff, duration, kick4temps, dur4temps,start_at, end_at = sampleLimits(file_table, user_inputs)


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
        print('PROCESSING BATCH',(i+1),'OF',batches,':')
        try:
            file_tableX = file_table[start:end,:]

            cycles, nom = buildTable1(file_tableX, i)
            if i == 0:
                samples_per_min,rebinnable = sample_rate_check()     
            ts = buildTable2(kickoff,duration,cycles, nom, i,samples_per_min)

            no_bins, bin_width, one_wire_sts, rebinnable, adjust, one_wire = buildTable3(cycles, kick4temps, dur4temps, i, samples_per_min,start_at, end_at,rebinnable)


            if raw_samples:
    
                raw_a1p1 = np.load(DATA_PATH+'/temp/a1p1.npy')
                raw_a1p2 = np.load(DATA_PATH+'/temp/a1p2.npy')
                raw_a2p1 = np.load(DATA_PATH+'/temp/a2p1.npy')
                raw_a2p2 = np.load(DATA_PATH+'/temp/a2p2.npy')
                np.save(DATA_PATH+'/temp/list_concats/a1p1_'+str(i)+'.npy',raw_a1p1)  
                np.save(DATA_PATH+'/temp/list_concats/a1p2_'+str(i)+'.npy',raw_a1p2)  
                np.save(DATA_PATH+'/temp/list_concats/a2p1_'+str(i)+'.npy',raw_a2p1)  
                np.save(DATA_PATH+'/temp/list_concats/a2p2_'+str(i)+'.npy',raw_a2p2)   

                if parameters[7] == 'True': #lighter processing load on Moonhut.
                    start = start+8
                    end = end+8
                else:
                    start = start + 26
                    end = end + 26             
        
            
            else:

                if rebinnable and parameters[0]=='True':
                    print('Generating',np.size(one_wire[:,0]),'one-minute bins per output, centred on One-Wire sample times.')
                    rebinnable = binONEWIRE(nom, no_bins, bin_width, one_wire_sts, rebinnable, i,samples_per_min)

                elif rebinnable and parameters[0]=='False':
                    print('Generating',np.size(one_wire[:,0]),'one-minute bins per output.')
                    binINminsX(nom,ts,samples_per_min)
                if rebinnable:
                    pass
                else:
                    print('\033[1;31m Radiometer data cannot be binned in line with One-wire data. \033[0;0m')
                    print('Generating',np.size(one_wire[:,0]),'one-minute bins per output.')
                    binINminsX(nom,ts,samples_per_min)
                np.save(DATA_PATH+'/temp/rebinnable.npy', rebinnable)

                if parameters[7] == 'True': #lighter processing load on Moonhut.
                    start = start+8
                    end = end+8
                else:
                    start = start + 26
                    end = end + 26

                os.system('rm -r /mirror/scratch/pblack/temp/concats/')
                os.system('mkdir /mirror/scratch/pblack/temp/concats/')

                os.system('cp /mirror/scratch/pblack/temp/a1p1.npy /mirror/scratch/pblack/temp/list_concats/a1p1_binned_'+str(i)+'.npy')
                os.system('rm /mirror/scratch/pblack/temp/a1p1.npy')

                os.system('cp /mirror/scratch/pblack/temp/a1p2.npy /mirror/scratch/pblack/temp/list_concats/a1p2_binned_'+str(i)+'.npy')
                os.system('rm /mirror/scratch/pblack/temp/a1p2.npy')

                os.system('cp /mirror/scratch/pblack/temp/a2p1.npy /mirror/scratch/pblack/temp/list_concats/a2p1_binned_'+str(i)+'.npy')
                os.system('rm /mirror/scratch/pblack/temp/a2p1.npy')

                os.system('cp /mirror/scratch/pblack/temp/a2p2.npy /mirror/scratch/pblack/temp/list_concats/a2p2_binned_'+str(i)+'.npy')
                os.system('rm /mirror/scratch/pblack/temp/a2p2.npy')

      #  except:
        except Exception as e: print(repr(e))
          #  pass

    if raw_samples:
        pass
    else:

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


        print(a1p1b[15,250:260])
        print(a2p2b[15,250:260])
        print(a2p1b[15,250:260])
        print(a1p2b[15,250:260])

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

        no_power = np.where(a1p1b[:,480] < 0.000001)
        a1p1b[no_power,3:] = np.nan
        no_power = np.where(a1p2b[:,480] < 0.000001)
        a1p2b[no_power,3:] = np.nan
        no_power = np.where(a2p1b[:,480] < 0.000001)
        a2p1b[no_power,3:] = np.nan
        no_power = np.where(a2p2b[:,480] < 0.000001)
        a2p2b[no_power,3:] = np.nan

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

if raw_samples:

    if batches > 4:
        print('Loading of raw data is restricted to a maximum of 24 hours.')
    for i in range (0,4): #more than this will likely crash the program
        if i == 0:
            a1p1 = np.load(DATA_PATH+'/temp/list_concats/a1p1_'+str(i)+'.npy')
    else:
        try:
            x = np.load(DATA_PATH+'/temp/list_concats/a1p1_'+str(i)+'.npy')
            a1p1 = np.row_stack((a1p1,x))
        except:
            pass

    for i in range (0,4): #more than this will likely crash the program
        if i == 0:
            a1p2 = np.load(DATA_PATH+'/temp/list_concats/a1p2_'+str(i)+'.npy')
        else:
            try:
                x = np.load(DATA_PATH+'/temp/list_concats/a1p2_'+str(i)+'.npy')
                a1p2 = np.row_stack((a1p2,x))
            except:
                pass

    for i in range (0,4): #more than this will likely crash the program
        if i == 0:
            a2p1 = np.load(DATA_PATH+'/temp/list_concats/a2p1_'+str(i)+'.npy')
        else:
            try:
                x = np.load(DATA_PATH+'/temp/list_concats/a2p1_'+str(i)+'.npy')
                a2p1 = np.row_stack((a2p1,x))
            except:
                pass

    for i in range (0,4): #more than this will likely crash the program
        if i == 0:
            a2p2 = np.load(DATA_PATH+'/temp/list_concats/a2p2_'+str(i)+'.npy')
        else:
            try:
                x = np.load(DATA_PATH+'/temp/list_concats/a2p2_'+str(i)+'.npy')
                a2p2 = np.row_stack((a2p2,x))
            except:
                pass

    a1p1 = remove_dups(a1p1)
    a1p2 = remove_dups(a1p2)
    a2p1 = remove_dups(a2p1)
    a2p2 = remove_dups(a2p2)

    a = np.where(np.isnan(a1p1[:,0]))
    a1p1 = np.delete(a1p1,a,0)
    a = np.where(np.isnan(a1p2[:,0]))
    a1p2 = np.delete(a1p2,a,0)
    a = np.where(np.isnan(a2p1[:,0]))
    a2p1 = np.delete(a2p1,a,0)
    a = np.where(np.isnan(a2p2[:,0]))
    a2p2 = np.delete(a2p2,a,0)

  #  np.save(DATA_PATH+'/temp/a1p1.npy',a1p1)
  #  np.save(DATA_PATH+'/temp/a1p2.npy',a1p2)
   # np.save(DATA_PATH+'/temp/a2p1.npy',a2p1)
   # np.save(DATA_PATH+'/temp/a2p2.npy',a2p2)

    obsheader = np.load(DATA_PATH+'/temp/obshdr.npy')
    MJD = Time(obsheader[0,8],format='mjd', scale='utc', precision=9)
    time_p1 = MJD + TimeDelta(a1p1[:,0].astype(float), format='sec') 
    time_p1.format = 'iso'

    DA_t1 = time_p1 + TimeDelta(a1p1[0,0].astype(float),format='sec')
    DA_t2 = time_p1 + TimeDelta(a1p1[-1,0].astype(float),format='sec')
    DA_diff = DA_t2 - DA_t1
    time_diff = DA_diff.sec / 3600
    dennis = np.round(time_diff,decimals=2)
 
    np.save(DATA_PATH+'/temp/duration_actual', dennis[0])


    empty = np.empty((1,721))
    empty[0,:] = np.nan
    if np.size(a1p1[:,0]) > np.size (a1p2[:,0]): #common for unbinned samples to mismatch by a single phase switch count. Correct to plot.
        a1p2 = np.insert(a1p2, (np.size(a1p2[:,0])-1) , empty, axis=0)
        a2p2 = np.insert(a2p2, (np.size(a2p2[:,0])-1), empty, axis=0)
    if np.size(a1p1[:,0]) < np.size (a1p2[:,0]):
        a1p1 = np.insert(a1p1, (np.size(a1p1[:,0])-1), empty, axis=0)  
        a2p1 = np.insert(a2p1, (np.size(a2p1[:,0])-1), empty, axis=0) 

    no_power = np.where(a1p1[:,480] < 0.000001)
    a1p1[no_power,3:] = np.nan
    no_power = np.where(a1p2[:,480] < 0.000001)
    a1p2[no_power,3:] = np.nan
    no_power = np.where(a2p1[:,480] < 0.000001)
    a2p1[no_power,3:] = np.nan
    no_power = np.where(a2p2[:,480] < 0.000001)
    a2p2[no_power,3:] = np.nan

    np.save(DATA_PATH+'/temp/a1p1_binned.npy',a1p1)
    np.save(DATA_PATH+'/temp/a1p2_binned.npy',a1p2)
    np.save(DATA_PATH+'/temp/a2p1_binned.npy',a2p1)
    np.save(DATA_PATH+'/temp/a2p2_binned.npy',a2p2)

    np.save(DATA_PATH+'/temp/a1p1.npy',a1p1)
    np.save(DATA_PATH+'/temp/a1p2.npy',a1p2)
    np.save(DATA_PATH+'/temp/a2p1.npy',a2p1)
    np.save(DATA_PATH+'/temp/a2p2.npy',a2p2)

    a1p1 = a1p1[:,3:]
    a1p2 = a1p2[:,3:]
    a2p1 = a2p1[:,3:]
    a2p2 = a2p2[:,3:]

    np.save(DATA_PATH+'/temp/a1p1_power.npy', (a1p1[:,112:569].mean(axis=1))) 
    np.save(DATA_PATH+'/temp/a1p2_power.npy', (a1p2[:,112:569].mean(axis=1)))  
    np.save(DATA_PATH+'/temp/a2p1_power.npy', (a2p1[:,112:569].mean(axis=1)))  
    np.save(DATA_PATH+'/temp/a2p2_power.npy', (a2p2[:,112:569].mean(axis=1))) 

   # np.save(DATA_PATH+'/temp/a1p1_bandpass.npy', (a1p1)) 
   # np.save(DATA_PATH+'/temp/a1p2_bandpass.npy', (a1p2)) 
   # np.save(DATA_PATH+'/temp/a2p1_bandpass.npy', (a2p1)) 
   # np.save(DATA_PATH+'/temp/a2p2_bandpass.npy', (a2p2)) 

    del a2p2, a2p1, a1p2, a1p1

CW_Power_Array_Generator()

gc.collect()

os.system('chmod -R -f 0777 /mirror/scratch/pblack || true')

