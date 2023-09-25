#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:33:26 2022

@author: pblack
"""

DATA_PATH = '/mirror/scratch/pblack'

import math
import numpy as np
import astropy.io
from astropy.io import fits
from astropy.time import Time 
from astropy.time import TimeMJD
from astropy.time import TimeDelta
import datetime
import time
from tqdm import tqdm #progress bars
import glob
import os
from math import nan

get = fits.open('/scratch/nas_lbass/raw_data/LBASS-2022-06-03T17:11:38.fits')
look = fits.open('/scratch/nas_lbass/raw_data/LBASS-2022-05-18T10:11:05.fits')

#mjd = TimeMJD(mjd, 0, scale='utc', precision=4, in_subfmt='float', out_subfmt='date')


#bad_list = []
#np.save(DATA_PATH+'/temp/bad_file.npy',bad_list)

#print (t1)
#print(get[2].header)
#print(look[2].header)
#a = get[0].header['DATE-END']
#b = get[0].header['DATE-BEG']

comment = get[0].header['COMMENT']
print(np.shape(comment))
print(comment)


#print(b)
#print(a)

