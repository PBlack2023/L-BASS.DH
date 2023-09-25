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


print ('Beginning RPG Temperature Data Aquisition Routine.')
try:
    os.chdir('/USBstick/ffts_uni-manchester_2016-12-01_001/tools/afftstools/afftsclient/')
    always = True
except:
    print('Unable to locate afftstools directory. RPG Temperature Aquisition Failed.')
    always = False


while always:

    try:

        os.system("./afftsclient --Temperature -I 192.168.10.1 -P 'RPG:XFFTS:' | cat > /scratch/nas_lbass/raw_data/RPG_temps.txt")
        now_date = Time.now()
        now_date = Time(now_date,format='iso',scale='utc',precision=9)
        now_date = now_date.mjd

        get_data = np.loadtxt('/scratch/nas_lbass/raw_data/RPG_temps.txt',dtype='str',delimiter=' ')

        print(np.shape(get_data))
        print(get_data)

        if os.path.exists('/scratch/nas_lbass/raw_data/RPG_temps.npy'):
            current_data = np.empty((1,4))
            current_data[0,0] = now_date
            
            RPG_temps = np.load('/scratch/nas_lbass/raw_data/RPG_temps.npy')
            RPG_temps = np.row_stack((RPG_temps, current_data))
            np.save('/scratch/nas_lbass/raw_data/RPG_temps.npy',RPG_temps)

        else:
            RPG_temps = np.empty((1,4))
            RPG_temps[0,0] = now_date
            
            np.save('/scratch/nas_lbass/raw_data/RPG_temps.npy',RPG_temps)

    except:
        pass

    try:

        os.system('chmod -R -f 0777 /scratch/nas_lbass/binned_data || true')
        os.system('chmod -R -f 0777 /scratch/nas_lbass/analysis || true')
    except:
        pass


    time.sleep(60)









