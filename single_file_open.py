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
from scipy import stats
import astropy.io
from astropy.io import fits

os.chdir('/scratch/nas_lbass/raw_data/')

os.system('rm -r /local5/scratch/pblack/temp/concats/')
os.system('mkdir /local5/scratch/pblack/temp/concats/')

if os.path.exists(DATA_PATH+'/temp/a1p1.npy'):
    os.system('rm /local5/scratch/pblack/temp/a1p1.npy')
if os.path.exists(DATA_PATH+'/temp/a1p2.npy'):
    os.system('rm /local5/scratch/pblack/temp/a1p2.npy')
if os.path.exists(DATA_PATH+'/temp/a2p1.npy'):
    os.system('rm /local5/scratch/pblack/temp/a2p1.npy')
if os.path.exists(DATA_PATH+'/temp/a2p2.npy'):
    os.system('rm /local5/scratch/pblack/temp/a2p2.npy')
if os.path.exists(DATA_PATH+'/temp/obshdr.npy'):
    os.system('rm /local5/scratch/pblack/temp/obshdr.npy')

#---------------------------------------------------------------------------

# aquire positional data, observer and mode

def getHeader(file_table):

   #reads the header of the fits file and extracts information about the observation, observatory and frequency

    print('')
    print('\033[0;0m Reading header')
    
    observatory = fits.open(file_table)
    get = fits.open(file_table)
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
            print('\033[1;31m No Azimuth value recorded by user. Azimuth default of 180° applied. \033[0;0m')
    
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
	del ind
	bad_samples=0
	np.save(DATA_PATH+'/temp/concats/breaky_'+str(nom)+'_'+(str(i)), channel_data)  #save as numpy array
	Isdup = stats.mode(channel_data[:,0]) #checks for duplicate sample timecodes
	#if int(Isdup[1]) > 1: 
	#	too_many_bad_sampletimes = True
	
	return too_many_bad_sampletimes
#-------------------------------------------------------------------------------------             
def routine2(opened, i):

    #attempts to recover the times when one-wire temperature data was recorded.

	temp_time = np.array(opened[2].data['TIME'])
	temps = np.array(opened[2].data['One-wire temperatures'])
	temp_data = np.column_stack((temp_time[1:61],temps[1:61,:]))
	np.save(DATA_PATH+'/temp/concats/tempdata_'+(str(i)), temp_data)

#-----------------------------------------------------------------------
def buildTable1(file_table):

    # a for loop which initiates the process of selecting each RPG channel and phase of the data.  This is repeated for each fits file and the data saved in seperate numpy arrays.

	print('Unpacking fits file data tables')
	time.sleep(1)
    
	i = 0
	for i in tqdm(range(0,1), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
		opened = fits.open(file_table)
		sample_time = np.array(opened[1].data['SAMPLE_TIME'])
		switch_phase = np.array(opened[1].data['SWITCH_PHASE'])
		elevation = np.array(opened[1].data['ELEVATION']) #greater than 90 doesn't change azimuth
		
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
			
			busted = routine1(opened, sample_time, switch_phase, elevation, channel, phase, nom[a],i)
		if busted: #more than 10% of samples in data recorded with irregular timings between datapoints
			print(file_table[i,0],'\033[1;31m Radiometer data recorded at irregular intervals. \033[0;0m')
			#bad_list = np.load(DATA_PATH+'/temp/bad_file.npy')
			#bad_list = np.append(bad_list, file_table[i,0])
			#np.save(DATA_PATH+'/temp/bad_file.npy', bad_list)	#put that fits file on the naughty list

		#temperature data
		routine2(opened,i)
		opened.close()
		gc.collect()
	
	return i+1, nom 

#---------------------------------------------------------------------------

def buildTable2(kickoff, duration, cycles, nom):

    # this opens each numpy array of radiometer data and combines into a single larger array.  Then it uses the user specified start and length of observations to trim the array down to only relevant datapoints. 

    lisst= ['P(L,\u03C0) W','P(L,0) E','P(R,\u03C0) E','P(R,0) W']
    ts = []
    j=0
    for j in range (0,4):
        
        print ('Aquiring',lisst[j])

        i = 0
        for i in tqdm(range(0,cycles), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            
            if i ==0:
                arm1x=np.load(DATA_PATH+'/temp/concats/breaky_'+str(nom[j])+'_0.npy')  #the first file
            else:
                x = np.load(DATA_PATH+'/temp/concats/breaky_'+str(nom[j])+'_'+str(i)+'.npy') #each of the next files in sequence
                arm1x = np.row_stack((arm1x, x)) #add data from each file into arm1x array
            gc.collect()
      
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
        ts.append(np.size(arm1x[:,0])/300) #how many minutes of data
        gc.collect()
    
    
    return ts
#----------------------------------------------------------------------------

def buildTable3(cycles, kickoff, duration):


    # opens each numpy array of onewire temperature data and combines into a single array - which is then trimmed to match the user specified start time and duration.  


#    np.save(DATA_PATH+'/temp/concats/tempdata_'+(str(i)), temp_data)
    print ('Aquiring One-wire temperature data')
    i = 0
    for i in tqdm(range(0,cycles), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            
        if i ==0:
            one_wire=np.load(DATA_PATH+'/temp/concats/tempdata_0.npy')
        else:
            x = np.load(DATA_PATH+'/temp/concats/tempdata_'+str(i)+'.npy')
            one_wire = np.row_stack((one_wire, x))
        gc.collect()

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
    
    i =0
    est=0
    gaps=[]
    adjust=[]
    nan_list=[]
    rebinnable=True
    nan_err = False

    for i in range (0, np.size(one_wire[:,0])):  #looks for missing sample times and inserts nans. necessary for correct array sizes.
        try:
            a = one_wire[i+1,0] - one_wire[i,0]
            if a == 120: # a single missing bin
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
        
        if a > 60.5 or a < 59.5:  #checks to see if tempmerature data is recorded at regular intervals
            print('\033[1;31m One-wire data recorded at irregular intervals. \033[0;0m')
            rebinnable=False
        else:
            gaps.append(a)
            adjust.append(a-60) #no longer needed really

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
    bin_width = 300 #int(np.mean(gaps))*5 #5 samples per second  #300 samples per minute

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

    if rebinnable:
        print('Generating',np.size(one_wire[:,0]),'one-minute bins per output.')
        
    else:
        print('\033[1;31m Radiometer data cannot be binned in line with One-wire data. \033[0;0m')

    np.save(DATA_PATH+'/temp/file2.npy', one_wire)  #save the onewire data into a numpy array.
    gc.collect()

    return no_bins, bin_width, one_wire[:,0], rebinnable, adjust

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

def binONEWIRE(nom, no_bins, bin_width, one_wire_sts, rebinnable):
           
    #averages the data over 1 minute periods, matched to the onewire data sample times.

    i = 0
    #sts means starts
    os.system('rm -r /local5/scratch/pblack/temp/bina/')
    os.system('mkdir /local5/scratch/pblack/temp/bina/')
    one_wire_sts = np.round(one_wire_sts, 1)
    parameters = np.load(DATA_PATH+'/temp/parameters.npy')
    if parameters[1] == 'True':
        flatten = True
    else:
        flatten = False
    np.save(DATA_PATH+'/temp/flatten.npy', flatten)

    j=0
    bad_bin=[]
    #nom j order:
    # A1P1 - P(l,pi)
    # A1P2 - P(l,0)
    # A2P1 - P(r,pi)
    # A2P2 - P(r,0)
    #power_ratios = [1.08,1.08,1,1] #LEFT CHANNEL UP  17th from 4pm 6 hrs = [1.0972,1.0972,1,1] #LEFT CHANNEL UP
    #power_ratios = [1,1,0.9,0.9] #RIGHT CHANNEL DOWN
    power_ratios = [1,1,1,1] #don't apply any correction for now

    bandpass_norms = [DATA_PATH+'/temp/band11-4norm-preOct22.npy',DATA_PATH+'/temp/band12-4norm-preOct22.npy',DATA_PATH+'/temp/band21-4norm-preOct22.npy',DATA_PATH+'/temp/band22-4norm-preOct22.npy']
    # bandpass_norms = [DATA_PATH+'/temp/band11-4norm.npy',DATA_PATH+'/temp/band12-4norm.npy',DATA_PATH+'/temp/band21-4norm.npy',DATA_PATH+'/temp/band22-4norm.npy']
    
    for j in range(0,4):
        
        i = 0 
        armPX = np.load(DATA_PATH+'/temp/'+str(nom[j])+'.npy') #open the radiometer data array
        #print(nom[j],'   ',np.shape(armPX)) #checks how many samples in each phase
        bad_samples=0
        bad_flag=[]
        just_sample_times = np.round(armPX[:,0], 1) #will be one for each minute, plus one minute - radiometer time codes
                     #could check average seperation, and if good forgoe more exacting binning......
                     #next_check = (just_sample_times[-1] - just_sample_times[0]) / 60
                     #bbbbb = np.size(just_sample_times) / np.round(next_check, 1)
                     #print((bbbbb))  #an if statement about bbbbb>300.3 bad, bbbbb<299.7 bad etc, otherwise just assume 300.
        Isdup = stats.mode(just_sample_times) #checks for duplicate sample timecodes
        if int(Isdup[1]) > 1: #if there are irregularities then...
            for i in range (0,np.size(just_sample_times)):  #deals with the midnight problem - incorrect sample times recorded
                try:
                    f = just_sample_times[i+1] - just_sample_times[i]   
                    if f > 0.3 or f < 0.1:  # gap between samples should be 0.2
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


        for i in tqdm(range(0,no_bins), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'): #rebin 1 min averages
        #for i in range (0,no_bins):
            best_match = find_nearest(just_sample_times, one_wire_sts[i]) #closest radiometer sample time to one_wire sample time
            bin_centre = np.where(just_sample_times == best_match) 
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
                else:
                    binned = np.row_stack((binned, armPX[a:b,:].mean(axis=0)))

            except:
                try:
                    tup_ex = bin_centre[0]
                    tup_ex = int(np.mean(tup_ex))
                    a = int(tup_ex)-int(bin_width/2)
                    b = int(tup_ex)+int(bin_width/2)
                    np.save(DATA_PATH+'/temp/bina/'+str(nom[j])+'_bin'+str(i), armPX[a:b,:].mean(axis=0))
                except:
                    bad_bin.append(i)
                    empty = np.empty((1,721))
                    np.save(DATA_PATH+'/temp/bina/'+str(nom[j])+'_bin'+str(i), empty)
                #rebinnable=False 
            gc.collect()
        try:
            #del armPX
           # binned=np.load(DATA_PATH+'/temp/bina/'+str(nom[j])+'_bin1.npy') 
           # i = 0
            #for i in range (0,no_bins):
            #for i in tqdm(range(0,no_bins), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
             #   x = np.load(DATA_PATH+'/temp/bina/'+str(nom[j])+'_bin'+str(i)+'.npy')
              #  binned = np.row_stack((binned, x))
            if j==0:
                DA = np.round((no_bins/60), decimals=2)
                np.save(DATA_PATH+'/temp/duration_actual', DA)

            #with rebinning complete, save as numpy arrays 

            normload = np.load(bandpass_norms[j])
            if flatten:
                pass
            else:
                normload = 1

            binned[:,3:] = binned[:,3:] / normload #flatten bandpass

            np.save(DATA_PATH+'/temp/'+str(nom[j])+'_binned.npy', binned) #includes time phase and elevation.

            binned = binned[:,3:]

            np.save(DATA_PATH+'/temp/'+str(nom[j])+'_power.npy', (binned[:,112:569].mean(axis=1))*power_ratios[j]) #power ratios applied  
            np.save(DATA_PATH+'/temp/'+str(nom[j])+'_bandpass.npy', (binned)) #no normalisation // radiometer data only
            
            del binned
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
        print ('\033[1;31m Binning process failed due to',np.size(bad_bin),'bad bins. \033[0;0m')
    gc.collect()
    
    return rebinnable

#---------------------------------------------------------


def binINminsX(nom,ts):
    
    #bins the data into 1 minute bins, but does not attempt to match up to the onewire datapoints

    parameters = np.load(DATA_PATH+'/temp/parameters.npy')
    if parameters[1] == 'True':
        flatten = True
    else:
        flatten = False
    np.save(DATA_PATH+'/temp/flatten.npy', flatten)

    print('\033[0;m Rebinning radiometer data, arbitrary 1 minute bins.')
    
    i = 0
    # bin width (1 minute = 600 samples, but split over two phases is 300)
    # bin centre (middle of min, but better based on temp record times)
    
    #find the number of bins by dividing up the samples by 300, 
    #then whichever is the larger number use to bin up
    
    if ts[0] > ts[1]:
        no_bins = int(ts[1])
    else:
        no_bins = int(ts[0])
    #this will vary depending on the files selected, but should be ~60 an hour

    os.system('rm -r /local5/scratch/pblack/temp/bina/')
    os.system('mkdir /local5/scratch/pblack/temp/bina/')
    
    #bin_width = 300 #(for each phase per min)
    #60 bins, 300 samples in each, 
    #arm1phase1[0:300,:] first bin
    #arm2phase1[301:600,:] second bin and so on
    
    #take mean of each column in given range to average for that minute of data
    
    #power_ratios = [1.08,1.08,1,1] #LEFT CHANNEL UP
    #power_ratios = [1,1,0.92,0.92] #RIGHT CHANNEL DOWN
    power_ratios = [1,1,1,1] #don't apply any correction for now

    #corrections for dates pre 5th Oct 2022:
    bandpass_norms = [DATA_PATH+'/temp/band11-4norm-preOct22.npy',DATA_PATH+'/temp/band12-4norm-preOct22.npy',DATA_PATH+'/temp/band21-4norm-preOct22.npy',DATA_PATH+'/temp/band22-4norm-preOct22.npy']
  #  bandpass_norms = [DATA_PATH+'/temp/band11-4norm.npy',DATA_PATH+'/temp/band12-4norm.npy',DATA_PATH+'/temp/band21-4norm.npy',DATA_PATH+'/temp/band22-4norm.npy']

    j=0
    for j in range(0,4):

        i = 0 
        a = 150
        b = bin_width + 150
        armPX = np.load(DATA_PATH+'/temp/'+str(nom[j])+'.npy')

        for i in tqdm(range(0,no_bins), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            if i ==0:
                binned = armPX[a:b,:].mean(axis=0)
            else:
                binned = np.row_stack((binned, armPX[a:b,:].mean(axis=0)))
            a = b +1
            b = b + bin_width
        del armPX

        
        if j==0:
            #DA = np.round((np.size(binned[:,112:569].mean(axis=1))/60), decimals=2)
            DA = np.round((no_bins/60), decimals=2)
            np.save(DATA_PATH+'/temp/duration_actual', DA)
        normload = np.load(bandpass_norms[j])
        if flatten:
            pass
        else:
            normload = 1
        binned[:,3:] = binned[:,3:] / normload #flatten bandpass
        np.save(DATA_PATH+'/temp/'+str(nom[j])+'_binned.npy', binned)
        binned = binned[:,3:]
        np.save(DATA_PATH+'/temp/'+str(nom[j])+'_power.npy', (binned[:,112:569].mean(axis=1))/power_ratios[j]) #mean each row)
        np.save(DATA_PATH+'/temp/'+str(nom[j])+'_bandpass.npy', (binned))
          
    del binned
    gc.collect()

#---------------------------------------------------------

def temp_csv(nan_list):

    #looks in the onewire csv files to try and find missing temperature data if it isn't in the fits files

    if os.path.exists(DATA_PATH+'/temp/temp_data.npy'):
        os.system('rm /local5/scratch/pblack/temp/temp_data.npy')
        
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
    #mjd + sampletime ... into iso ... search for matches ... overwrite nans
    MJD = Time(file_table[1,12], format='mjd', scale='utc', precision=0)
    
    i=0
    for i in range (0,np.size(nan_list)):
        sample_to_time = MJD + TimeDelta(nan_list[i], format='sec')
        time_iso = Time(sample_to_time, format='iso', scale='utc', out_subfmt='date_hm')
        datetime_list.append(str(time_iso))
    for i in range (0,np.size(all_temps[:,0])):
        a = all_temps[i,0]
        b= a[:-3]
        all_temps[i,0] = b
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

    for i in tqdm(range(0,np.size(nan_list)), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
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
        os.system('rm /local5/scratch/pblack/temp/a1p1_cw.npy')
    if os.path.exists(DATA_PATH+'/temp/a1p2_cw.npy'):
        os.system('rm /local5/scratch/pblack/temp/a1p2_cw.npy')
    if os.path.exists(DATA_PATH+'/temp/a2p1_cw.npy'):
        os.system('rm /local5/scratch/pblack/temp/a2p1_cw.npy')
    if os.path.exists(DATA_PATH+'/temp/a2p2_cw.npy'):
        os.system('rm /local5/scratch/pblack/temp/a2p2_cw.npy')

    signal1, signal2 = [], []
    CW_Present = [False, False]
    hornchoice = 'west'
    if hornchoice.lower() == 'west':
        a1p1b = np.load(DATA_PATH+'/temp/a1p1.npy')
        a1p1b = a1p1b[:,3:]
        a2p2b = np.load(DATA_PATH+'/temp/a2p2.npy')
        a2p2b = a2p2b[:,3:]
        a1p2b = np.load(DATA_PATH+'/temp/a1p2.npy')#----
        a1p2b = a1p2b[:,3:]
        a2p1b = np.load(DATA_PATH+'/temp/a2p1.npy')#----
        a2p1b = a2p1b[:,3:]

        W1_bin_find = stats.mode(np.argmax(a1p1b, axis=1))
        peak_value1 = np.max(a1p1b[:,int(W1_bin_find[0])])    
        W2_bin_find = stats.mode(np.argmax(a2p2b, axis=1))    

        if int(W1_bin_find[0]) > 569 and peak_value1/(np.mean(a1p1b[:,112:569])) > 10: #outside digital bandpass, and 10x stronger signal
            max_index1 = int(W1_bin_find[0])
            max_index2 = int(W2_bin_find[0])
            print('CW signal detected in West horn')
            CW_Present = [False, True]
            np.save(DATA_PATH+'/temp/CW_Index.npy', max_index1)
                    
            for line in a1p1b:
                signal = np.sum(line[(max_index1-2):(max_index1+2)])
                sideband1_average = np.average(line[max_index1-6:max_index1-3])
                sideband2_average = np.average(line[max_index1+3:max_index1+6])
                sideband_bin_average = (sideband1_average + sideband2_average) / 2
                signal_subtracted = signal - (sideband_bin_average) #*5
                signal1.append(signal_subtracted)
            signal1 = np.array(signal1)   

            a1p1binned = np.load(DATA_PATH+'/temp/a1p1_binned.npy')
            a1p1binned = a1p1binned[:,3:]
            signalx = []
            for line in a1p1binned:
                signal = np.sum(line[(max_index1-2):(max_index1+2)])
                sideband1_average = np.average(line[max_index1-6:max_index1-3])
                sideband2_average = np.average(line[max_index1+3:max_index1+6])
                sideband_bin_average = (sideband1_average + sideband2_average) / 2
                signal_subtracted = signal - (sideband_bin_average)#*5
                signalx.append(signal_subtracted)
            signalx = np.array(signalx)  
            np.save(DATA_PATH+'/temp/a1p1_cw_binned.npy', signalx)
            del a1p1binned
        
            for line in a2p2b:
                signal = np.sum(line[(max_index2-2):(max_index2+2)])
                sideband1_average = np.average(line[max_index2-6:max_index2-3])
                sideband2_average = np.average(line[max_index2+3:max_index2+6])
                sideband_bin_average = (sideband1_average + sideband2_average) / 2
                signal_subtracted = signal - (sideband_bin_average) #*5
                signal2.append(signal_subtracted)
            signal2 = np.array(signal2)

            a2p2binned = np.load(DATA_PATH+'/temp/a2p2_binned.npy')
            a2p2binned = a2p2binned[:,3:]
            signalx = []
            for line in a2p2binned:
                signal = np.sum(line[(max_index2-2):(max_index2+2)])
                sideband1_average = np.average(line[max_index2-6:max_index2-3])
                sideband2_average = np.average(line[max_index2+3:max_index2+6])
                sideband_bin_average = (sideband1_average + sideband2_average) / 2
                signal_subtracted = signal - (sideband_bin_average)#*5
                signalx.append(signal_subtracted)
            signalx = np.array(signalx)  
            np.save(DATA_PATH+'/temp/a2p2_cw_binned.npy', signalx)
            del a2p2binned
    
            np.save(DATA_PATH+'/temp/a1p1_cw.npy', signal1)
            np.save(DATA_PATH+'/temp/a2p2_cw.npy', signal2)
#¡¡¡¡¡¡¡¡¡

   #         W3_bin_find = stats.mode(np.argmax(a1p2b, axis=1))
    #        peak_value2 = np.max(a1p2b[:,int(W3_bin_find[0])])    
     #       W4_bin_find = stats.mode(np.argmax(a2p1b, axis=1))   
      #      max_index1 = int(W3_bin_find[0])
       #     max_index2 = int(W4_bin_find[0])
        #    signal3=[]
         #   for line in a1p2b:
          #      signal = np.sum(line[(max_index1-2):(max_index1+2)])
           #     sideband1_average = np.average(line[max_index1-6:max_index1-3])
            #    sideband2_average = np.average(line[max_index1+3:max_index1+6])
             #   sideband_bin_average = (sideband1_average + sideband2_average) / 2
              #  signal_subtracted = signal - (sideband_bin_average*5)
               # signal3.append(signal_subtracted)
           # signal3 = np.array(signal3)
            
           # signal4=[]
            #for line in a2p1b:
           #     signal = np.sum(line[(max_index2-2):(max_index2+2)])
            #    sideband1_average = np.average(line[max_index2-6:max_index2-3])
             #   sideband2_average = np.average(line[max_index2+3:max_index2+6])
              #  sideband_bin_average = (sideband1_average + sideband2_average) / 2
              # signal_subtracted = signal - (sideband_bin_average*5)
              #  signal4.append(signal_subtracted)
            #signal4 = np.array(signal4)
    
            #np.save(DATA_PATH+'/temp/a1p2_cw.npy', signal3)
            #np.save(DATA_PATH+'/temp/a2p1_cw.npy', signal4)

#333211112
              
    hornchoice = 'east'     
    if hornchoice.lower() == 'east':
        a1p2b = np.load(DATA_PATH+'/temp/a1p2.npy')#----
        a1p2b = a1p2b[:,3:]
        a2p1b = np.load(DATA_PATH+'/temp/a2p1.npy')#----
        a2p1b = a2p1b[:,3:]
        W1_bin_find = stats.mode(np.argmax(a1p2b, axis=1))
        peak_value1 = np.max(a1p2b[:,int(W1_bin_find[0])])    
        W2_bin_find = stats.mode(np.argmax(a2p1b, axis=1))   

        if int(W1_bin_find[0]) > 569 and peak_value1/(np.mean(a1p2b[:,112:569])) > 10: #outside digital bandpass, and 10x stronger signal
            max_index1 = int(W1_bin_find[0])
            max_index2 = int(W2_bin_find[0])
            print('CW signal detected in East horn')
            CW_Present = [True, False]
            np.save(DATA_PATH+'/temp/CW_Index.npy', max_index1)
    
            for line in a1p2b:
                signal = np.sum(line[(max_index1-2):(max_index1+2)])
                sideband1_average = np.average(line[max_index1-6:max_index1-3])
                sideband2_average = np.average(line[max_index1+3:max_index1+6])
                sideband_bin_average = (sideband1_average + sideband2_average) / 2
                signal_subtracted = signal - (sideband_bin_average)#*5
                signal1.append(signal_subtracted)
            signal1 = np.array(signal1)

            a1p2binned = np.load(DATA_PATH+'/temp/a1p2_binned.npy')
            a1p2binned = a1p2binned[:,3:]
            signalx = []
            for line in a1p2binned:
                signal = np.sum(line[(max_index1-2):(max_index1+2)])
                sideband1_average = np.average(line[max_index1-6:max_index1-3])
                sideband2_average = np.average(line[max_index1+3:max_index1+6])
                sideband_bin_average = (sideband1_average + sideband2_average) / 2
                signal_subtracted = signal - (sideband_bin_average)#*5
                signalx.append(signal_subtracted)
            signalx = np.array(signalx)  
            np.save(DATA_PATH+'/temp/a1p2_cw_binned.npy', signalx)
            del a1p2binned
        
            for line in a2p1b:
                signal = np.sum(line[(max_index2-2):(max_index2+2)])
                sideband1_average = np.average(line[max_index2-6:max_index2-3])
                sideband2_average = np.average(line[max_index2+3:max_index2+6])
                sideband_bin_average = (sideband1_average + sideband2_average) / 2
                signal_subtracted = signal - (sideband_bin_average)#*5
                signal2.append(signal_subtracted)
            signal2 = np.array(signal2)

            a2p1binned = np.load(DATA_PATH+'/temp/a2p1_binned.npy')
            a2p1binned = a2p1binned[:,3:]
            signalx = []
            for line in a2p1binned:
                signal = np.sum(line[(max_index1-2):(max_index1+2)])
                sideband1_average = np.average(line[max_index1-6:max_index1-3])
                sideband2_average = np.average(line[max_index1+3:max_index1+6])
                sideband_bin_average = (sideband1_average + sideband2_average) / 2
                signal_subtracted = signal - (sideband_bin_average)#*5
                signalx.append(signal_subtracted)
            signalx = np.array(signalx)  
            np.save(DATA_PATH+'/temp/a2p1_cw_binned.npy', signalx)
            del a2p1binned
    
            np.save(DATA_PATH+'/temp/a1p2_cw.npy', signal1)
            np.save(DATA_PATH+'/temp/a2p1_cw.npy', signal2)

    np.save(DATA_PATH+'/temp/CW_Present.npy', CW_Present)
    
    if CW_Present[0] == False and CW_Present[1] == False:
        print ('CW Signal not detected')
    
    return signal1, signal2

#==========================================================================



# load in the user selections and related fits / csv files

file_table = np.load(DATA_PATH+'/temp/file1.npy', allow_pickle=True)  #all relevant fits files

file_table = input('name of LBASS fits file to inspect:  ')
file_table = '/scratch/nas_lbass/raw_data/'+file_table

parameters = np.load(DATA_PATH+'/temp/parameters.npy')

getHeader(file_table)

cycles, nom = buildTable1(file_table)

#ts = buildTable2(kickoff,duration,cycles, nom)

#no_bins, bin_width, one_wire_sts, rebinnable, adjust = buildTable3(cycles, kick4temps, dur4temps)



gc.collect()

os.system('chmod -R -f 0777 /local5/scratch/pblack || true')

