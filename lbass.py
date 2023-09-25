#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:21:35 2022

@author: pblack
"""

DATA_PATH = '/mirror/scratch/pblack'


########################################################################

#Packages needed to support the functionality of the script

import numpy as np
from astropy.time import Time 
from astropy.time import TimeDelta
import datetime
from datetime import datetime
import glob
import gc
import os
import time
#import itur

#print(dir(itur))

#25x70
##############################################################################
#setenv PATH /usr/local/anaconda-python-3.6/bin:$PATH


print('')
print ('\033[0;m V2.2 - Feb 2023 - Phillip Black')
print('\033[1;32m ')
print ('         -------------------------------------------------------')
print ('               Data handling for the L-Band All Sky Survey              ')
print ('         -------------------------------------------------------')
print ('')
#print (' \033[5;93m *    *    \033[0;29m ')
#print ('    \033[5;29m   **  ')
print ('                                                      \033[1;m ______   ______ \033[1;32m')
print (' \033[0;31m  ##        ######      ####      ####      ####     \033[1;m |    |   |    | \033[1;32m')
print (' \033[0;33m  ##        ##    ##  ##    ##  ##    ##  ##    ##   \033[1;m \    /   \    / \033[1;32m')
print (' \033[0;93m  ##        ##    ##  ##    ##  ##        ##         \033[1;m  |  |     |  |  \033[1;32m')
print (' \033[0;32m  ##        ######    ########    ####      ####     \033[1;m  |  |==@==|  |  \033[1;32m')
print (' \033[0;36m  ##        ##    ##  ##    ##        ##        ##   \033[1;m  |  |  ^  |  |  \033[1;32m')
print (' \033[0;34m  ##        ##    ##  ##    ##  ##    ##  ##    ##   \033[1;m  \  /     \  /  \033[1;32m')
print (' \033[0;95m  ########  ######    ##    ##    ####      ####     \033[1;m   ||       ||   \033[1;32m')
print ('                                                      \033[1;m   --       --   \033[1;32m')
print ('                                                                      ')
#print ('\033[0;m Loading background packages')
os.system('chmod -R -f 0777 /scratch/nas_lbass/binned_data || true')
os.system('chmod -R -f 0777 /scratch/nas_lbass/analysis || true')

os.chdir('/scratch/nas_lbass/raw_data/')

def feedback():
    

    print('')
    user = input('Your name: ')
    print ('')
    comment = input('Your feedback/request: ')
    today = datetime.now()
    dt_string = today.strftime("%Y_%m_%d___%H_%M")
    print('')
    remarks = np.array((dt_string, user, comment))
    np.save(DATA_PATH+'/feedback/'+dt_string+'.npy', remarks)  
    print('Your comments have been logged.')
    print('')
    
def review():
    
    comment_files = []
    comment_files = sorted(glob.glob(DATA_PATH+'/feedback/*.npy'))
    
    i=0
    for i in range (i,np.size(comment_files)):
       readit = np.load(comment_files[i], allow_pickle=True)
       print(readit)
       input('Press enter to read next comment')
       

########################################################################

def mainMenu():

    print ('\033[1;32m ')
    print ('    -----------------------')
    print ('    >      MAIN MENU      <')
    print ('    -----------------------')
    print (' ')
    print ('   1 - Quicklook')
    print ('   2 - Load & Bin Data')
    print ('   3 - Load Raw Data')
    print ('   4 - Resume')
    print ('')
    print ('   5 - Settings')
    print ('')
    print ('   6 - Passives')
    print ('   7 - ITU-R')
    print ('')
    print ('   0 - Exit Program')
    print('')

    choice = input('Select menu option (number): \033[0;m')
    print('\033[1;32m')
    if choice.isdigit():

        if int(choice) == 1: #quickload 
            quickload=True
            raw_samples=False
            if os.path.exists(DATA_PATH+'/temp/quickload.npy'):
                os.system('rm /mirror/scratch/pblack/temp/quickload.npy')    
            if os.path.exists(DATA_PATH+'/temp/raw_samples.npy'):
                os.system('rm /mirror/scratch/pblack/temp/raw_samples.npy')
            np.save(DATA_PATH+'/temp/quickload.npy',quickload)
            np.save(DATA_PATH+'/temp/raw_samples.npy',raw_samples)
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/new_funcsa.py')
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/breaky_tablesb.py')
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/passives.py')
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/quick_look_plotsa.py')
            loopmenu = True

        elif int(choice) == 2: #load fits and process
            raw_samples=False 
            quickload=False
            if os.path.exists(DATA_PATH+'/temp/quickload.npy'):
                os.system('rm /mirror/scratch/pblack/temp/quickload.npy')    
            if os.path.exists(DATA_PATH+'/temp/raw_samples.npy'):
                os.system('rm /mirror/scratch/pblack/temp/raw_samples.npy')
            np.save(DATA_PATH+'/temp/quickload.npy',quickload)
            np.save(DATA_PATH+'/temp/raw_samples.npy',raw_samples)
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/new_funcsa.py')
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/breaky_tablesb.py')
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/passives.py')
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/quick_look_plotsa.py')
            loopmenu = True
        
        elif int(choice)  == 3: #load fits but don't time bin
            raw_samples=True
            quickload=False
            if os.path.exists(DATA_PATH+'/temp/quickload.npy'):
                os.system('rm /mirror/scratch/pblack/temp/quickload.npy')
            if os.path.exists(DATA_PATH+'/temp/raw_samples.npy'):
                os.system('rm /mirror/scratch/pblack/temp/raw_samples.npy')
            np.save(DATA_PATH+'/temp/raw_samples.npy',raw_samples)
            np.save(DATA_PATH+'/temp/quickload.npy',quickload)
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/new_funcsa.py')
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/breaky_tablesb.py')
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/passives.py')
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/quick_look_plotsa.py')
            loopmenu = True

        elif int(choice) == 4:
            try:
                os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/quick_look_plotsa.py')
            except:
                print('\033[1;31m Unable to resume previous session.\033[1;32m')
            loopmenu = True


 #   elif int(choice)  == 5:
  #      exec(open("/mirror/scratch/pblack/scripts/INST.py").read())

        elif int(choice)  == 6:
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/passives.py')
            loopmenu = True

        elif int(choice)  == 7:
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/itur2.py')
            loopmenu = True

        elif int(choice)  == 8:
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/itur4a.py')
            loopmenu = True

        elif int(choice)  == 9:
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/itur8.py')
            loopmenu = True

#        exec(open("/mirror/scratch/pblack/scripts/BOARD.py").read())

        elif int(choice)  == 5:
            os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/scripts/PARAMETERS.py')
            loopmenu = True


        
   # elif int(choice) == 9:
    #    feedback()
     #   os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/lbass.py')
  #  elif int(choice) == 99:
   #     review()
    #    os.system('/usr/local/anaconda-python-3.6/bin/python3 /mirror/scratch/pblack/lbass.py')
        elif int(choice) == 0:
            print ('\033[0;m' )
            loopmenu = False

    else:
        print('\033[1;31m Invalid selection by user. Please try again.')
    
    return loopmenu


loopmenu = True
while loopmenu:
    gc.collect()
    loopmenu = mainMenu()

print('')
print ('\033[0;m Goodbye' )
print('')
print('')

#----------------------------------------------------------------------
