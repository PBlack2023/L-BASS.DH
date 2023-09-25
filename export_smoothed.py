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
import os
from math import nan
import gc
import warnings

os.chdir('/scratch/nas_lbass/raw_data/')

#---------------------------------------------------------------------------


def remove_dups(data_array, ref_array):
    try:
        only_times = ref_array[:]
        uniques, indices = np.unique(only_times,return_index=True)
        data_array = data_array[indices,:]
    except:
        only_times = ref_array[:] #1D list
        uniques, indices = np.unique(only_times,return_index=True)
        data_array = data_array[indices]
    return data_array

#§§§§§§§§§§§


print('\033[0;m Preparing data for export. \033[1;32m')



one_wire = np.load(DATA_PATH+'/temp/one_wire.npy', allow_pickle=True)
a1p1b = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
a1p2b = np.load (DATA_PATH+'/temp/a1p2_binned.npy')
a2p1b = np.load (DATA_PATH+'/temp/a2p1_binned.npy')
a2p2b = np.load (DATA_PATH+'/temp/a2p2_binned.npy')

#print(np.shape(a1p1b))
#print(np.shape(a1p2b))
#print(np.shape(a2p1b))
#print(np.shape(a2p2b))
#print(np.shape(one_wire))

a1p1b = remove_dups(a1p1b,a1p1b[:,0])
a1p2b = remove_dups(a1p2b,a1p2b[:,0])
a2p1b = remove_dups(a2p1b,a2p1b[:,0])
a2p2b = remove_dups(a2p2b,a2p2b[:,0])

diff = np.diff(a1p1b[:,0])
delete_index = np.where(diff < 50) #more than 60 seconds gap (plus 16% margin of error)
a1p1b = np.delete(a1p1b,delete_index,0)
diff = np.diff(a1p2b[:,0])
delete_index = np.where(diff < 50) #more than 60 seconds gap (plus 16% margin of error)
a1p2b = np.delete(a1p2b,delete_index,0)
diff = np.diff(a2p1b[:,0])
delete_index = np.where(diff < 50) #more than 60 seconds gap (plus 16% margin of error)
a2p1b = np.delete(a2p1b,delete_index,0)
diff = np.diff(a2p2b[:,0])
delete_index = np.where(diff < 50) #more than 60 seconds gap (plus 16% margin of error)
a2p2b = np.delete(a2p2b,delete_index,0)
diff = np.diff(one_wire[:,0])
delete_index = np.where(diff < 50) #more than 60 seconds gap (plus 16% margin of error)
one_wire = np.delete(one_wire,delete_index,0)

empty = np.empty((1,721))
empty[0,:] = np.nan
if np.size(a1p1b[:,0]) > np.size (a1p2b[:,0]): #common for unbinned samples to mismatch by a single phase switch count. Correct to plot.
    while np.size(a1p1b[:,0]) > np.size (a1p2b[:,0]):
        a1p2b = np.insert(a1p2b, (np.size(a1p2b[:,0])-1) , empty, axis=0)
        a2p2b = np.insert(a2p2b, (np.size(a2p2b[:,0])-1), empty, axis=0)
if np.size(a1p1b[:,0]) < np.size (a1p2b[:,0]):
    while np.size(a1p1b[:,0]) < np.size (a1p2b[:,0]):
        a1p1b = np.insert(a1p1b, (np.size(a1p1b[:,0])-1), empty, axis=0)  
        a2p1b = np.insert(a2p1b, (np.size(a2p1b[:,0])-1), empty, axis=0) 

user_inputs = np.load(DATA_PATH+'/temp/inputs.npy', allow_pickle=True) 
obsheader = np.load(DATA_PATH+'/temp/obshdr.npy')
MJD = Time(obsheader[0,8],format='mjd', scale='utc', precision=9)
time_p1 = MJD + TimeDelta(a1p1b[:,0].astype(float), format='sec') 
time_p1 = np.asarray(time_p1.mjd)

time_p2 = MJD + TimeDelta(one_wire[:,0].astype(float), format='sec')
time_p2 = np.asarray(time_p2.mjd)

smooth_string = user_inputs[7]
if os.path.exists(smooth_string+'a1p1.npy'): #if the smooth files already exist
    q = np.load(smooth_string+'a1p1.npy')
    r = np.load(smooth_string+'a1p2.npy')
    s = np.load(smooth_string+'a2p1.npy')
    t = np.load(smooth_string+'a2p2.npy')
    u = np.load(smooth_string+'time_array.npy',allow_pickle=True)
    v = np.load(smooth_string+'one_wire.npy')
    w = np.load(smooth_string+'temp_time_array.npy')

 
    a1p1s = np.row_stack((a1p1b,q))
    a1p2s = np.row_stack((a1p2b,r))
    a2p1s = np.row_stack((a2p1b,s))
    a2p2s = np.row_stack((a2p2b,t)) #combine with other smoothed data
    one_wireS = np.row_stack((one_wire,v))
    time_array = np.append(time_p1,u)
    temp_time_array = np.append(time_p2,w)

    save_it = True
    if np.size(a1p1s[:,0]) == np.size(time_array):
        a1p1s = remove_dups(a1p1s,time_array)
        a1p2s = remove_dups(a1p2s,time_array) # remove duplicates
        a2p1s = remove_dups(a2p1s,time_array)
        a2p2s = remove_dups(a2p2s,time_array)
        time_array = remove_dups(time_array,time_array)
    else:
        print('')
        print('\033[1;31m Number of bins in time array and data arrays do not match. \033[1;32m')
        save_it = False

    if np.size(one_wireS[:,0]) == np.size(temp_time_array):
        one_wireS = remove_dups(one_wireS,temp_time_array)
        temp_time_array = remove_dups(temp_time_array,temp_time_array)
    else:
        print('')
        print('\033[1;31m Number of bins in time array and one-wire array do not match. \033[1;32m')
        save_it = False

    try:
        a1p1s = a1p1s[time_array[:].argsort()] #sort the array by time
        a1p2s = a1p2s[time_array[:].argsort()] 
        a2p1s = a2p1s[time_array[:].argsort()] 
        a2p2s = a2p2s[time_array[:].argsort()] 
        time_array = time_array[time_array[:].argsort()]
    except:
        print('')
        print('\033[1;31m Unable to sort data array by timecode. \033[1;32m')
        save_it = False

    try:
        one_wireS = one_wireS[temp_time_array[:].argsort()] #sort the array by time
        temp_time_array = temp_time_array[temp_time_array[:].argsort()]
    except:
        print('')
        print('\033[1;31m Unable to sort one-wire array by timecode. \033[1;32m')
        save_it = False

    if save_it:
        print('')
        print('\033[1;32m Please ensure this data is of desired quality before exporting.')
        proceed = input('Are you certain you wish to export this data? (Y/N): ')
        if proceed == 'Y' or proceed == 'y':
            pass
        else:
            save_it = False

    if save_it:
        np.save(smooth_string+'a1p1.npy',a1p1s)
        np.save(smooth_string+'a1p2.npy',a1p2s) 
        np.save(smooth_string+'a2p1.npy',a2p1s)
        np.save(smooth_string+'a2p2.npy',a2p2s)
        np.save(smooth_string+'one_wire.npy',one_wireS)
        np.save(smooth_string+'time_array.npy',time_array)
        np.save(smooth_string+'temp_time_array.npy',temp_time_array)
        print('')
        print('\033[0;m Data successfully exported. \033[1;32m')

else:
    np.save(smooth_string+'a1p1.npy',a1p1b)
    np.save(smooth_string+'a1p2.npy',a1p2b)
    np.save(smooth_string+'a2p1.npy',a2p1b)
    np.save(smooth_string+'a2p2.npy',a2p2b)
    np.save(smooth_string+'time_array.npy',time_p1)
    np.save(smooth_string+'one_wire.npy',one_wire)
    np.save(smooth_string+'temp_time_array.npy',time_p2)

    print('')
    print('\033[0;m Data successfully exported. \033[1;32m')





os.system('chmod -R -f 0777 /local5/scratch/pblack || true')
os.system('chmod -R -f 0777 /scratch/nas_lbass/binned_data || true')
os.system('chmod -R -f 0777 /scratch/nas_lbass/analysis || true')

