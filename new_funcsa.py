#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:48:39 2022

@author: gibbsphillip
"""

DATA_PATH = '/mirror/scratch/pblack'


import math
import scipy
import numpy as np
import astropy.io
from astropy.io import fits
from astropy.time import Time 
from astropy.time import TimeDelta
import datetime
from datetime import timedelta
import time
from tqdm import tqdm #progress bars
import glob
import os
import gc
from math import nan
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning) #prevents non-ascii characters in fits headers crashing the program

os.chdir('/scratch/nas_lbass/raw_data/')

if os.path.exists(DATA_PATH+'/temp/reset_day.npy'):
    os.system('rm /mirror/scratch/pblack/temp/reset_day.npy')

#----------------------------------------------------------------------------

def selectobs(getdate):
    
    fail=True
    path_string='Error'
    date = input ('Date of observation (YYYY-MM-DD): \033[0;m') #user enters date
    print('\033[1;32m')

    if isinstance(date, tuple):
        print ('\033[1;31m Initial input miread as tuple. Please try again. \033[1;32m')
        print('')
        pass
    if len(date) < 10:
        date='Error'
        pass
    
    try:
        year, month, day = date.split('-')
        isValidDate = True

        try:
            datetime.datetime(int(year), int(month), int(day))
        except ValueError:
            isValidDate = False
            print ('\033[1;31m Invalid date entered. Please try again. \033[1;32m')
            print('')
            pass

        if isValidDate:
            getdate=False #to end loop
            path_string = '/scratch/nas_lbass/analysis/'+str(year)+'_'+str(month)+'/'
            smooth_string = '/scratch/nas_lbass/binned_data/'+str(year)+'_'+str(month)+'/'
            np.save(DATA_PATH+'/temp/user_date.npy', date)
    
    except:
        print ('\033[1;31m Incorrect format or type. Must be YYYY-MM-DD. Please try again. \033[1;32m')
        print('')
        fail=False
        smooth_string = None
        pass

    return date, path_string, getdate, fail, smooth_string

#----------------------------------------------------------------------------

def load_files(date):

    all_files = []
    all_files = sorted(glob.glob('/scratch/nas_lbass/raw_data/LBASS-'+date+'*.fits')) #find files created on that date
    parameters = np.load(DATA_PATH+'/temp/parameters.npy')
    if parameters[7] == 'True': #is this running on Moonhut?
        moonhut_files = sorted(glob.glob('/data/LBASS/LBASS-'+date+'*.fits'))  #files not copied to nas drive yet
        for i in range(0,np.size(moonhut_files)):
            all_files.append(moonhut_files[i])
       
    bad_chk = np.load(DATA_PATH+'/temp/bad_file.npy')
    user_date = np.load(DATA_PATH+'/temp/user_date.npy')

    if date == user_date: #if this is the user specified date
        
        try: #check for midnight partfile from previous day, if so adds last file of previous day to file list
            check = fits.open(all_files[0], ignore_missing_end=True)
            sample = check[1].data['SAMPLE_TIME']
     
            if sample[0] > 86400: #it started on a previous day, ergo runs past midnight
                year, month, day = date.split('-')
                d = (datetime.date(int(year), int(month), int(day))) - timedelta(days=1)
                past_files = sorted(glob.glob('/scratch/nas_lbass/raw_data/LBASS-'+str(d)+'*.fits'))
                forward_files = sorted(glob.glob('/scratch/nas_lbass/raw_data/LBASS-'+date+'*.fits'))
                if parameters[7] == 'True': #is this running on Moonhut?
                    moonhut_past = sorted(glob.glob('/data/LBASS/LBASS-'+str(d)+'*.fits'))
                    moonhut_forward = sorted(glob.glob('/data/LBASS/LBASS-'+date+'*.fits'))
                    for i in range (0,np.size(moonhut_forward)):
                        forward_files.append(moonhut_forward[i])
                    for i in range (0,np.size(moonhut_past)):
                        past_files.append(moonhut_past[i])
                    
                all_files=[]
                all_files.append(past_files[-1])
                for i in range (0, np.size(forward_files)):
                    all_files.append(forward_files[i])

            

            check.close() 
        except:
            pass

    for i in range (0, np.size(bad_chk)):  #delete files flagged as bad from the available list
        try:
            all_files.remove(str(bad_chk[i]))
        except:
            pass

    parameters = np.load(DATA_PATH+'/temp/parameters.npy')
    if parameters[7] == 'True': #is this running on Moonhut?
        now_date = Time.now()  
        now_now = Time(now_date, format='iso', scale='utc', precision=4, out_subfmt='date')
        if str(date) == str(now_now):  #if your looking at files from today
            del all_files[-1]  #don't open the final file as it is likely only partly written
    else:
        pass

    index_to = len(all_files) 

    get = []
    beg = []
    end = []
    start_time = []
    end_time = []
    sample = []
    samFIRST = []
    samLAST = []
    multi_day = []
    same_day = []
    same_run = []
    STH = []
    corrupt = []
    MJD=[]
    j = 0

    for j in range(0,index_to): #open all the files and list key date time info 

        try:
            get = fits.open(all_files[j], ignore_missing_end=True)
            ok=True 

            HDR_MJD = get[2].header['MJDREF']           
            StartDate = Time(HDR_MJD, format='mjd', scale='utc', precision=4)
            MJD.append(StartDate)
            sample = get[1].data['SAMPLE_TIME']
            BEG_date = StartDate + TimeDelta(sample[0], format='sec')
            END_date = StartDate + TimeDelta(sample[-1], format='sec')

            chklen = (END_date-BEG_date)*24  #should usually give ~1 (as in 1 hour file length)
            sample_lenchk = (int(sample[-1] - sample[0])/10)/(int(np.size(sample)/100)) #check number of samples in the file is appropriate based on the sample times recorded.
            
            if chklen > 1.1 or int(sample_lenchk) != 1:
                print('\033[1;31m Timecode error, file longer than expected: \033[0;0m')
                print(all_files[j])
               # bad_list = np.load(DATA_PATH+'/temp/bad_file.npy')
               # bad_list = np.append(bad_list, all_files[j])
               # np.save(DATA_PATH+'/temp/bad_file.npy', bad_list)
              #  time.sleep(1)
               # ok=False

            samFIRST.append(sample[0])
            samLAST.append(sample[-1])
                   
            if ok:
                corrupt.append(False)
            else:
                corrupt.append(True)

            beg.append(BEG_date.strftime('%Y-%m-%d'))
            end.append(END_date.strftime('%Y-%m-%d'))
            start_time.append(BEG_date.strftime("%H:%M"))
            STH.append(BEG_date.strftime("%H")) #start time hour only 
            end_time.append(END_date.strftime("%H:%M"))
            
            if sample[0] > 86400:
                multi_day.append(True)
            else:
                multi_day.append(False)
            get.close()
                        
        except:
            print('')
            print(all_files[j],'\033[1;31m No data or corrupt. \033[0;0m')
            bad_list = np.load(DATA_PATH+'/temp/bad_file.npy')
            
            bad_list = np.append(bad_list, all_files[j])
                
            np.save(DATA_PATH+'/temp/bad_file.npy', bad_list)
            corrupt.append(True)
            beg.append(nan)
            end.append(nan)
            start_time.append(nan)
            end_time.append(nan)
            samFIRST.append(nan)
            samLAST.append(nan)
            multi_day.append(False)
            STH.append(nan)
            MJD.append(nan)
            file_table(date)

    i = 0
    for i in range(0, index_to): #upto but not including 
        if str(beg[i]) == str(end[i]): #does the file start and end on the same date?
            same_day.append(True)
        else:
            same_day.append(False)
        same_run.append(True)
            
    file_table = np.column_stack((all_files, beg, end, start_time, end_time, samFIRST, samLAST, multi_day, same_day, same_run, corrupt, STH, MJD))

    if np.size(file_table) < 1: #if the array is empty ask for another date

        print('')
        print ('\033[1;31m Unable to locate files for the date entered. Please try another date. \033[1;32m')
        
    else:
        pass

    gc.collect()

    return file_table, index_to
#----------------------------------------------------------------------------

def add_day_check(file_table):  #finds and adds more days of files where observations are continuous
    
    final_hour, final_minute = str(file_table[-1,4]).split(':')

    if str(file_table[-1,1]) == str(file_table[-1,2]): #same day do nothing
        pass
  
    #if str(file_table[-1,1]) == str(file_table[-1,2]) and int(final_hour) <= 22: #same day do nothing
     #   pass

   # if str(file_table[-1,1]) == str(file_table[-1,2]) and int(final_hour) == 23: #might have midnight problem, look forward
    #    year, month, day = str(file_table[-1,2]).split('-')
     #   date = (datetime.date(int(year), int(month), int(day))) + timedelta(days=1)
      #  date = str(date)
       # file_table_out, index_to = load_files(date)
        #file_table_in = file_table
    #    if index_to < 24: #do not extend file table (no point, cont runs need 24 hours of data each day)
            
      #      file_table = np.append(file_table_in, file_table_out, axis=0)
     #   else: #do extend the file table
       #     file_table = np.append(file_table_in, file_table_out, axis=0)
        #    file_table = add_day_check(file_table)

    else:
        date = str(file_table[-1,2]) #last file, end date
        file_table_out, index_to = load_files(date)
        file_table_in = file_table
        if index_to < 24: #do not extend file table (no point, cont runs need 24 hours of data each day)
            
            file_table = np.append(file_table_in, file_table_out, axis=0)
        else: #do extend the file table
            
            file_table = np.append(file_table_in, file_table_out, axis=0)

            file_table = add_day_check(file_table)   #loop the add date routine
        

    return file_table

#----------------------------------------------------------------------------

def prior_start(file_table): #checks to see if first file is part of obs from previous day
    
    print ('\033[1;32m ')
    if file_table[0,7]:
        t = Time(file_table[0,1], format='isot', scale='utc', precision=0, out_subfmt='date')
        count = float(file_table[0,5]) + 30
        diff_check = t - TimeDelta(count, format='sec')
        run_start_date = Time(diff_check, out_subfmt='date')
         
        #print ('')
        #print ('NB. This data is part of a continuous run which commenced on an earlier date.')
        #print ('')
        np.save(DATA_PATH+'/temp/run_start_date', run_start_date)
        
    return (file_table)
#------------------------------------------------------------------------------

def list_runs(file_table): #show user the available start times and lengths of runs
    


    i = 0
    for i in range(0, np.size(file_table[:,0])):  #upto but not including 
        if int(i) == int(np.size(file_table[:,0])-1):#are these files part of the same run of files?
            file_table[i,9] = False
        else:
            samDiff = file_table[i+1,5] - file_table[i,6]
            if samDiff < 0:  #negative difference means sample count reset - a new run started on current date
                file_table[i,9] = False
                reset_day = True
                np.save(DATA_PATH+'/temp/reset_day', reset_day)  #is this really clumsy?
            elif samDiff >= 300: #if 5 minute gap between current file last sample and next file first sample
                file_table[i,9] = False
            else:
                file_table[i,9] = True
    

    time.sleep(1)
    print('Data is available in the following time ranges:')
    print('')
    user_date = np.load(DATA_PATH+'/temp/user_date.npy')
    i =0
    for i in range(0,np.size(file_table[:,0])): #for each file in file table
        j=i-1
        if i == 0 and str(user_date) == str(file_table[0,1]): #this is the first file of the day, run started on that day
            print ('   ',file_table[0,1],' -',file_table[0,3], ' - run begins')

        elif i == 0 and str(user_date) != str(file_table[0,1]): #if the first file is actually last file of previous day
            print('   ',str(user_date),' - 00:00  - continued from previous day')

        elif i >= 1 and i != (np.size(file_table[:,0])-1): # i is current file, j is previous file
            if file_table[j,9]: #data continues into this file from previous, do nothing
                pass

            else: #data does not continue into this file from previous!

                if str(user_date) == str(file_table[j,1]): #if its still the user specified date
                    print('   ',file_table[j,1],' -',file_table[j,4], ' - run ends') #previous file end


                if str(user_date) == str(file_table[i,1]): #if its still the user specified date
                    print('                -')
                    print('   ',file_table[i,1],' -',file_table[i,3], ' - run begins') #new run of files on same day

                elif str(user_date) != str(file_table[i,1]): #if its a different date
                    print ('\033[1;33m   ',file_table[i,2],'\033[1;32m -',file_table[i,4],' - run ends')
                    break #end the loop here

        elif i == (np.size(file_table[:,0])-1): # last file on the list
            print ('\033[1;33m   ',file_table[i,2],'\033[1;32m -',file_table[i,4],' - run ends')
  
                            
    gc.collect()

#------------------------------------------------------------------------------

def selecttime(gettime): # choose time of day as callable function
    
    hour, minute, user_sample, time_in = 'Error', 'Error','Error','Error'
    
    print('')
    time_in = input ('\033[1;32m Time to inspect data from (HH:MM): \033[0;m')
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
            gettime=False #to end while loop
            pass
        
        else:
            print("\033[1;31m Invalid time entered. Must be HH:MM. Please try again. \033[1;32m")
            print('')
    
        user_sample = float((int(hour) * 3600) + (int(minute) * 60))    
        
        
    except:
        print ('\033[1;31m Incorrect format or type. Please try again. \033[1;32m')
        print('')

    
    
    return hour, minute, user_sample, time_in, gettime

#-----------------------------------------------------------------------------

def selectduration(getduration): # choose duration as callable function
    
    a='Error'
    print('\033[1;32m')
    duration_in = input ('Duration to inspect in hours: \033[0;m')
    
    try:
        float(duration_in)
        getduration=False
    
        if float(duration_in) == 0:
            a = 5000 #daft number of observing hours that will never be reached
        else:
            a = float(duration_in)
    except:
    
        print ('\033[1;31m Input not recognised, please enter an numeric value, e.g. 6 or 2.5 \033[1;32m')
        print ('')
    
    return a, getduration

#------------------------------------------------------------------------------------------

def trim_list(user_hour, file_table, user_duration): #take user inputs and trim the file table to match
    
    i = 0 #convert user time to integer
    for i in range(0,9):
        if user_hour == ('0'+str(i)):
            int_user_hour = int(i)
        else:
            int_user_hour = int(user_hour)

    i = 0 #convert start time from file to integer
    j = 0
    
    
    for i in range(0,np.size(file_table[:,0])):
        for j in range(0,24):
            if file_table[i,11] == ('0'+str(j)):
                file_table[i,11] = int(j)
            elif int(file_table[i,11]) == int(j):
                file_table[i,11] = int(j)
            else:
                pass
    orig_zeros = False
    if int_user_hour == 00:
        orig_zeros = True
        int_user_hour = 23
        add = 1
    elif int_user_hour == 1: #select from hour before
        int_user_hour = 0
        add = 1
    elif int_user_hour >= 2: #select from 2 hours before
        int_user_hour = int_user_hour - 1 #presupposes the hour before exists.
        add = 1
    
    i=0
    user_date = np.load(DATA_PATH+'/temp/user_date.npy')
    
    for i in range(0, np.size(file_table[:,0])):
        if str(user_date) != str(file_table[i,0]) and int(file_table[i,11]) == int(int_user_hour) and orig_zeros:
                ind1 = i
                break
        if str(user_date) == str(file_table[i,1]) and int(file_table[i,11]) == int(int_user_hour):
                ind1 = i
                break
        elif str(user_date) == str(file_table[i,1]) and int(file_table[i,11]) == (int(int_user_hour)+1):
                ind1 = i
                break
        else:
            pass

    if np.size(file_table[:,0]) < (ind1 + user_duration + 1 + add):
        #print(np.size(file_table[:,0]), ind1+user_duration+1+add)
        ind2 = np.size(file_table[:,0])
    else: 
        ind2 = (ind1 + user_duration + 1 + add)
        #print(ind1+user_duration+1+add)
#this should mean we have a couple of files in the list from before and after the user specified times
    
    selected_file_table = file_table[int(ind1):int(ind2),:]
    
    gc.collect()

    return selected_file_table, user_hour

#-------------------------------------------------------------------

if os.path.exists(DATA_PATH+'/temp/file1.npy'):
    os.system('rm /mirror/scratch/pblack/temp/file1.npy')
if os.path.exists(DATA_PATH+'/temp/inputs.npy'):
    os.system('rm /mirror/scratch/pblack/temp/inputs.npy')

getdate = True
while getdate:
    date, path_string, getdate, fail, smooth_string = selectobs(getdate)
    if fail:    
        print ('\033[0;0m Searching fits files')
        time.sleep(1)
        try:
            file_table, index_to = load_files(date)
            if np.size(index_to) < 24 and np.size(file_table[:,0]) > 1:
                file_table = add_day_check(file_table)
                prior_start(file_table)
                list_runs(file_table)

            else:
                prior_start(file_table)
                list_runs(file_table)
        except:
            print('\033[1;32m')
            getdate = True

gettime=True
while gettime:
    user_hour, user_minute, user_sample, time_in, gettime = selecttime(gettime)

getduration=True
while getduration:
    user_duration, getduration = selectduration(getduration)

selected_file_table, user_hour = trim_list(user_hour, file_table, user_duration)

raw_samples = np.load(DATA_PATH+'/temp/raw_samples.npy')
parameters = np.load(DATA_PATH+'/temp/parameters.npy')
if raw_samples:

    if parameters[7] == 'True': #lighter processing load on Moonhut.
        if np.size(selected_file_table[:,0]) > 14:
            print ('')
            print ('\033[1;31m More than 12 hours data selected. This would risk crashing the program. \033[1;32m')
            print ('')
            print ('Please specify a duration of 12 hours or less.')
            print ('')
    
            getduration=True
            while getduration:
                user_duration, getduration = selectduration(getduration)

            selected_file_table, user_hour = trim_list(user_hour, file_table, user_duration)

    else:
        if np.size(selected_file_table[:,0]) > 26:
            print ('')
            print ('\033[1;31m More than 24 hours data selected. This would risk crashing the program. \033[1;32m')
            print ('')
            print ('Please specify a duration of 24 hours or less.')
            print ('')
    
            getduration=True
            while getduration:
                user_duration, getduration = selectduration(getduration)

            selected_file_table, user_hour = trim_list(user_hour, file_table, user_duration)


    
np.save(DATA_PATH+'/temp/file1', selected_file_table)

user_inputs = np.array((date, user_hour, user_minute, time_in, user_sample, user_duration, path_string, smooth_string))
np.save(DATA_PATH+'/temp/inputs', user_inputs)
gc.collect()
os.system('chmod -R -f 0777 /mirror/scratch/pblack || true')







