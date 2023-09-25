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

#badadd = input('Full filename of bad/corrupt file: ')

bad_list = np.load(DATA_PATH+'/temp/bad_file.npy')
#print(bad_list)
bad_list = []
                
#bad_list = np.append(bad_list, badadd)
                
np.save(DATA_PATH+'/temp/bad_file.npy', bad_list)
