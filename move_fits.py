#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:48:39 2022

@author: gibbsphillip
"""

import numpy as np
import astropy.io
from astropy.io import fits
from astropy.time import Time 
from astropy.time import TimeDelta
import datetime
from datetime import timedelta
import time
import glob
import os

always = True

while always:

    try:
        print('')
        print('Back-up routine:')
        all_files = []
        all_files = sorted(glob.glob('/data/LBASS/*.fits')) #find all fits files

        now_date = Time.now()  
        now_now = Time(now_date, format='iso', scale='utc', precision=4, out_subfmt='date')
        if str(now_now) in str(all_files[-1]):  #if your looking at files from today
            del all_files[-1]  #don't include last file on the list as it is probably part-written.
        else:
            pass

        for i in range (0, np.size(all_files)):
            os.system('mv '+str(all_files[i])+' /scratch/nas_lbass/raw_data')
            print(all_files[i],'moved to nas drive.')
        print('Back-up routine complete.')
        time.sleep(86400)

    except:
        print('')
        print('Back-up routine failed.')
        time.sleep(86400)






