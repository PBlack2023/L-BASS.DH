#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:56:25 2022

@author: pblack
"""
DATA_PATH = '/mirror/scratch/pblack'


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import math
from math import nan
import numpy as np
from astropy.time import Time 
from astropy.time import TimeDelta
import datetime
import time
import os
import gc

os.chdir('/scratch/nas_lbass/raw_data/')

# GOOD FREQ ARE 112-568  BUT remember channels start at 3 in a1p1 etc
# A1P1 - P(l,pi)
# A2P2 - P(r,0)
# A1P2 - P(l,0)
# A2P1 - P(r,pi)

####################################################################################

def corrections (a1p1,a1p2,a2p1,a2p2): 

    parameters = np.load(DATA_PATH+'/temp/parameters.npy')
    profiles = np.load(DATA_PATH+'/bp_profiles/profiles.npy')
    if parameters[1] == 'True':
        flatten = True
    else:
        flatten = False
    np.save(DATA_PATH+'/temp/flatten.npy', flatten)   

    # A1P1 - P(l,pi)
    # A1P2 - P(l,0)
    # A2P1 - P(r,pi)
    # A2P2 - P(r,0)
    #power_ratios = [1.03,1.03,1,1] #Peter specified corrections - 21/12/2022
    power_ratios = [1,1,1,1] #no correction
    power_ratios = np.load(DATA_PATH+'/bp_profiles/PRC.npy')
  #  power_ratios = [1.03,1.0309,1,1.03] #Pips attempt at corrections - 21/12/2022

    if flatten:
    
        for i in range (0, np.size(profiles[:,0])):
            try:
                if profiles[i,8] == 'True':
                    print('')
                    print('\033[0;0m Normalising against bandpass profile:',profiles[i,0],' \033[1;32m')
                    bandpass_norms = [profiles[i,4],profiles[i,5],profiles[i,6],profiles[i,7]]
                else:
                    pass
            except:
                print('')
                print('\033[1;31m Bandpass profile failed to load. \033[1;32m')

    if flatten:
                normload = np.load(bandpass_norms[0]+'.npy')
                a1p1 = a1p1 / normload #flatten bandpass
                normload = np.load(bandpass_norms[1]+'.npy')
                a1p2 = a1p2 / normload #flatten bandpass
                normload = np.load(bandpass_norms[2]+'.npy')
                a2p1 = a2p1 / normload #flatten bandpass
                normload = np.load(bandpass_norms[3]+'.npy')
                a2p2 = a2p2 / normload #flatten bandpass
  
    if power_ratios[0] == 1 and power_ratios[1] == 1 and power_ratios[2] == 1 and power_ratios[3] == 1:
        pass
    else:
        print('\033[0;0m Applying input power ratio corrections. \033[1;32m')
        a1p1 = a1p1 * power_ratios[0] #adjust power ratios
        a1p2 = a1p2 * power_ratios[1] #adjust power ratios
        a2p1 = a2p1 * power_ratios[2] #adjust power ratios
        a2p2 = a2p2 * power_ratios[3] #adjust power ratios

    return a1p1, a1p2, a2p1, a2p2



################################################################################


def get_minmax(band):

    ind = np.argmin(band[:,112:569])
    indx = np.argmax(band[:,112:569])
    band = band[:,112:569].flatten()
    bandmin = band[ind] 
    bandmax = band[indx]   

    return bandmin, bandmax

#---------------------------------------------------------------------

def onePlotoneBin(frequency, save_it=False, first_loop=True, which_input=None, which_time=None):

    #prints up a single time bin

    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    del p11
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    date_time = Time(user_inputs[0]+'T'+user_inputs[3], format='isot', scale='utc', precision=0)    
    
    if first_loop:
        print('')
        print('  1 - P(L,\u03C0) W')
        print('  2 - P(L,0) E')
        print('  3 - P(R,\u03C0) E')
        print('  4 - P(R,0) W')
        print('')
        which_input = input('Please select an input: ')
        try:
            which_input = int(which_input)
        except:
            pass
        print('')
        print('Start time is '+user_inputs[3]+' & duration is '+str(duration_actual)+' hours.')
        print('')
        which_time = input('Please specify a bin time (HH:MM) ')
        try:
            which_timetime = Time(user_inputs[0]+'T'+which_time, format='isot', scale='utc', precision=4)
        except:
            print('Specified time not recognised, use 24hr clock. Please try again.')
            onePlotoneBin(frequency)
    
    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    if save_it:
        which_timetime = Time(user_inputs[0]+'T'+which_time, format='isot', scale='utc', precision=4)
    time_diff = which_timetime - date_time
    bin_number = int(time_diff.sec / 60)

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

    mask_width = parameters[4].astype(np.float)
    mask_width = 1+int(mask_width/2)
    if parameters[6] == 'True':
        ax.axvline(x=frequency[483-mask_width],linestyle='--', color='salmon', linewidth=1)
        ax.axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
        ax.axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
        ax.axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')

    ax.set(xlabel='Frequency /MHz')
    #plt.ylim([0,25])
    if parameters[8]=='True' or parameters[1]=='True':
        if parameters[1]=='True' and parameters[8]=='True':
            ax.set(ylabel='Global Normalised Power (Normalised Axis)')
        elif parameters[1]=='True':
            ax.set(ylabel='Global Normalised Power')
        else:  
            ax.set(ylabel='Power (Normalised Axis)')
    else:
        ax.set(ylabel='Power / a.u.')

    band11 = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
    band12 = np.load (DATA_PATH+'/temp/a1p2_bandpass.npy')
    band21 = np.load (DATA_PATH+'/temp/a2p1_bandpass.npy')
    band22 = np.load (DATA_PATH+'/temp/a2p2_bandpass.npy')

    band11,band12,band21,band22 = corrections(band11,band12,band21,band22)

    if parameters[8]=='True':
        band11 = band11/np.mean(band11[:,112:569])
        band12 = band12/np.mean(band12[:,112:569])
        band21 = band21/np.mean(band21[:,112:569])
        band22 = band22/np.mean(band22[:,112:569])

    if which_input == 1:
        if parameters[2]=='True':
            band11[:,(484-mask_width):(483+mask_width)+1] = 'nan'
        ax.hist(frequency[112:569], bins=456, weights=band11[bin_number,112:569], rwidth=0.7, align='mid', color='k')
        ax.set_title('Bandpass Spectrum, P(L,$\pi$) W\n'+user_inputs[0]+'_'+which_time)
        del band11
    
    elif which_input ==2:
        if parameters[2]=='True':
            band12[:,(484-mask_width):(483+mask_width)+1] = 'nan'
        ax.hist(frequency[112:569], bins=456, weights=band12[bin_number,112:569], rwidth=0.7, align='mid', color='k')
        ax.set_title('Bandpass Spectrum, P(L,0) E\n'+user_inputs[0]+'_'+which_time)
        del band12

    elif which_input ==3:
        if parameters[2]=='True':
            band21[:,(484-mask_width):(483+mask_width)+1] = 'nan'
        ax.hist(frequency[112:569], bins=456, weights=band21[bin_number,112:569], rwidth=0.7, align='mid', color='k')
        ax.set_title('Bandpass Spectrum, P(R,$\pi$) E\n'+user_inputs[0]+'_'+which_time)
        del band21

    elif which_input ==4:
        if parameters[2]=='True':
            band22[:,(484-mask_width):(483+mask_width)+1] = 'nan'
        ax.hist(frequency[112:569], bins=456, weights=band22[bin_number,112:569], rwidth=0.7, align='mid', color='k')
        ax.set_title('Bandpass Spectrum, P(R,0) W\n'+user_inputs[0]+'_'+which_time)
        del band22
    
    else:
        print('\033[1;31m Input not recognised, please try again. \033[1;32m')
        onePlotoneBin(frequency)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(c='darkgrey', which='major')
    ax.grid(c='gainsboro', which='minor')
    ax.margins(x=0)

    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+which_time+'_BP-.png', bbox_inches="tight")
        print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
    else:
        plt.show()
    plt.close()
    
    if first_loop:
        print ('')
        save = input ('Do you want to save a printer friendly copy of this plot? (Y/N): \033[0;m')
        if str(save) == 'N' or str(save) == 'n':
            print('\033[1;32m')
            pass
        elif str(save) == 'Y' or str(save) == 'y':
            save_it=True
            first_loop = False
            onePlotoneBin(frequency, save_it, first_loop, which_input, which_time)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 


#------------------------------------------------------------------------------------
    
def onePlotsStacked(frequency, save_it=False, first_loop=True, which_input=None):

    #prints up one smin bins ontop of one another for single input

    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    del p11
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy') 
    date_time = Time(user_inputs[0]+'T'+user_inputs[3], format='isot', scale='utc', precision=4)    

    band11 = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
    colour_n = np.size(band11[:,0])
    del band11
    R = []
    G = []
    B = []
    for i in range(1,colour_n+1):
        if i <= (colour_n/4):
            red = 255
            green = int(255/((colour_n/4)*i))
            blue = 0
        elif i <= (colour_n/2) and i > (colour_n/4):
            red = int((-255)/(colour_n/4)*i + 255 * 2)
            green = 255
            blue = 0
        elif i <= (colour_n*0.75) and i > (colour_n/2):
            red = 0
            green = 255
            blue = int((255 / (colour_n / 4) * i + (-255 * 2)))
        elif i > (colour_n*0.75):
            red = 0
            green = int(-255 * i / (colour_n / 4) + (255 * 4))
            blue = 255
        R.append(red/255)
        G.append(green/255)
        B.append(blue/255)
    
    if first_loop:
        print('')
        print('  1 - P(L,\u03C0) W')
        print('  2 - P(L,0) E')
        print('  3 - P(R,\u03C0) E')
        print('  4 - P(R,0) W')
        print('')
        which_input = input('Please select an input: ')
        try:
            which_input = int(which_input)
        except:
            pass

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

    mask_width = parameters[4].astype(np.float)
    mask_width = 1+int(mask_width/2)
    if parameters[6] == 'True':
        ax.axvline(x=frequency[483-mask_width+1],linestyle='--', color='salmon', linewidth=1)
        ax.axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
        ax.axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
        ax.axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1)

        ax.set(xlabel='Frequency /MHz')
        if parameters[8]=='True' or parameters[1]=='True':
            if parameters[1]=='True' and parameters[8]=='True':
                ax.set(ylabel='Global Normalised Power (Normalised Axis)')
            elif parameters[1]=='True':
                ax.set(ylabel='Global Normalised Power')
            else:  
                ax.set(ylabel='Power (Normalised Axis)')
        else:
            ax.set(ylabel='Power / a.u.')

    band11 = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
    band12 = np.load (DATA_PATH+'/temp/a1p2_bandpass.npy')
    band21 = np.load (DATA_PATH+'/temp/a2p1_bandpass.npy')
    band22 = np.load (DATA_PATH+'/temp/a2p2_bandpass.npy')

    band11,band12,band21,band22 = corrections(band11,band12,band21,band22)

    if parameters[8]=='True':
        band11 = band11/np.mean(band11[:,112:569])
        band12 = band12/np.mean(band12[:,112:569])
        band21 = band21/np.mean(band21[:,112:569])
        band22 = band22/np.mean(band22[:,112:569])

    if which_input == 1:
        if parameters[2]=='True':
            band11[:,(484-mask_width):(483+mask_width)+1] = 'nan'
        for i in range (0,np.size(band11[:,0])):
            ax.plot(frequency[112:569], band11[i,112:569], color =(R[i],G[i],B[i]), linewidth=0.5)
            ax.set_title('Bandpass Spectrum, P(L,$\pi$) W\n'+user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours.')
        del band11
    
    elif which_input ==2:
        if parameters[2]=='True':
            band12[:,(484-mask_width):(483+mask_width)+1] = 'nan'
        for i in range (0,np.size(band12[:,0])):
            ax.plot(frequency[112:569], band12[i,112:569], color =(R[i],G[i],B[i]), linewidth=0.5)
            ax.set_title('Bandpass Spectrum, P(L,0) E \n'+user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours.')
        del band12
        
    elif which_input ==3:
        if parameters[2]=='True':
            band21[:,(484-mask_width):(483+mask_width)+1] = 'nan'
        for i in range (0,np.size(band21[:,0])):
            ax.plot(frequency[112:569], band21[i,112:569], color =(R[i],G[i],B[i]), linewidth=0.5)
            ax.set_title('Bandpass Spectrum, P(R,$\pi$) E \n'+user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours.')
        del band21

    elif which_input ==4:
        if parameters[2]=='True':
            band22[:,(484-mask_width):(483+mask_width)+1] = 'nan'
        for i in range (0,np.size(band22[:,0])):
            ax.plot(frequency[112:569], band22[i,112:569], color =(R[i],G[i],B[i]), linewidth=0.5)
            ax.set_title('Bandpass Spectrum, P(R,0) W \n'+user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours.')
        del band22

            
    else:
        print('\033[1;31m Input not recognised, please try again. \033[1;32m')
        onePlotsStacked(frequency)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(c='darkgrey', which='major')
    ax.grid(c='gainsboro', which='minor')
    ax.margins(x=0)
    
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_'+'_'+str(which_input)+'_BP-stacked.png', bbox_inches="tight")
        print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
    else:
        plt.show()
    plt.close()
    
    if first_loop:
        print ('')
        save = input ('Do you want to save a printer friendly copy of this plot? (Y/N): \033[0;m')
        if str(save) == 'N' or str(save) == 'n':
            print('\033[1;32m')
            pass
        elif str(save) == 'Y' or str(save) == 'y':
            save_it=True
            first_loop = False
            onePlotsStacked(frequency, save_it, first_loop, which_input)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass 

#------------------------------------------------------------------------------------

def bandpassANI(frequency):

    #animation of bandpass by bin

    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    del p11
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    date_time = Time(user_inputs[0]+'T'+user_inputs[3], format='isot', scale='utc', precision=4)

    print('')
    print('  1 - P(L,\u03C0) W')
    print('  2 - P(L,0) E')
    print('  3 - P(R,\u03C0) E')
    print('  4 - P(R,0) W')
    print('')
    which_input = input('Please select an input: ')
    try:
        which_input = int(which_input)
    except:
        pass
    print('')
    print('Please specify run-speed of animation, recommend < 1 second per bin')
    print('')
    which_speed = input('Seconds per time-bin (e.g. 0.5): ')
    try:
       which_speed = float(which_speed)
    except:
       which_input = 'eggs' #nonsense value to deliberately trigger an error

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    mask_width = parameters[4].astype(np.float)
    mask_width = 1+int(mask_width/2)  #don't forget rounding

    fig, ax = plt.subplots()

    band11 = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
    band12 = np.load (DATA_PATH+'/temp/a1p2_bandpass.npy')
    band21 = np.load (DATA_PATH+'/temp/a2p1_bandpass.npy')
    band22 = np.load (DATA_PATH+'/temp/a2p2_bandpass.npy')

    band11,band12,band21,band22 = corrections(band11,band12,band21,band22)

    if parameters[8]=='True':
        band11 = band11/np.mean(band11[:,112:569])
        band12 = band12/np.mean(band12[:,112:569])
        band21 = band21/np.mean(band21[:,112:569])
        band22 = band22/np.mean(band22[:,112:569])
    
    if which_input == 1:
        if parameters[2]=='True':
            band11[:,(484-mask_width):(483+mask_width)+1] = 'nan'
        a = np.argmin(band11[:,112:569])
        b = np.argmax(band11[:,112:569])
        bandFd = band11[:,112:569].flatten()
        for i in range (0,(np.size(band11[:,0]))):

            try:
                ax.plot(frequency[112:569], (band11[i-3,112:569]), color ='grey' , linewidth=1)
                ax.plot(frequency[112:569], (band11[i-5,112:569]), color ='silver' , linewidth=1)
            except:
                pass
            ax.plot(frequency[112:569], (band11[i,112:569]), color ='k' , linewidth=1)
            ax.set(xlabel='Frequency /MHz')
            if parameters[8]=='True' or parameters[1]=='True':
                if parameters[1]=='True' and parameters[8]=='True':
                    ax.set(ylabel='Global Normalised Power (Normalised Axis)')
                elif parameters[1]=='True':
                    ax.set(ylabel='Global Normalised Power')
                else:  
                    ax.set(ylabel='Power (Normalised Axis)')
            else:
                ax.set(ylabel='Power / a.u.')
            ax.set_ylim(((bandFd[a])*0.95),((bandFd[b])*1.05))
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.grid(c='darkgrey', which='major')
            ax.grid(c='gainsboro', which='minor')
            if parameters[6] == 'True':
                ax.axvline(x=frequency[483-mask_width],linestyle='--', color='salmon', linewidth=1)
                ax.axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
                ax.axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
                ax.axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
            ax.margins(x=0)
            roll_time = date_time + TimeDelta(i*60, format='sec')
            time_str = roll_time.strftime("%H:%M") 
            ax.set_title('Bandpass Spectrum, P(l,$\pi$) W \n'+user_inputs[0]+' '+time_str)
            plt.pause(which_speed)
            plt.cla()
        del band11, bandFd
    
    elif which_input ==2:
        if parameters[2]=='True':
            band12[:,(484-mask_width):(483+mask_width)+1] = 'nan'
        a = np.argmin(band12[:,112:569])
        b = np.argmax(band12[:,112:569])
        bandFd = band12[:,112:569].flatten()
        for i in range (0,np.size(band12[:,0])):
            ax.plot(frequency[112:569], band12[i,112:569], color ='k' , linewidth=1)
            plt.set(xlabel='Frequency /MHz')
            if parameters[8]=='True' or parameters[1]=='True':
                if parameters[1]=='True' and parameters[8]=='True':
                    ax.set(ylabel='Global Normalised Power (Normalised Axis)')
                elif parameters[1]=='True':
                    ax.set(ylabel='Global Normalised Power')
                else:  
                    ax.set(ylabel='Power (Normalised Axis)')
            else:
                ax.set(ylabel='Power / a.u.')
            ax.set_ylim(((bandFd[a])*0.95),((bandFd[b])*1.05))
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.grid(c='darkgrey', which='major')
            ax.grid(c='gainsboro', which='minor')
            if parameters[6] == 'True':
                ax.axvline(x=frequency[483-mask_width],linestyle='--', color='salmon', linewidth=1)
                ax.axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
                ax.axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
                ax.axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
            ax.margins(x=0)
            roll_time = date_time + TimeDelta(i*60, format='sec')
            time_str = roll_time.strftime("%H:%M") 
            ax.set_title('Bandpass Spectrum, P(l,0) E \n'+user_inputs[0]+' '+time_str)
            plt.pause(which_speed)
            plt.cla()
        del band12, bandFd
    
    elif which_input ==3:
        if parameters[2]=='True':
            band21[:,(484-mask_width):(483+mask_width)+1] = 'nan'
        a = np.argmin(band21[:,112:569])
        b = np.argmax(band21[:,112:569])
        bandFd = band21[:,112:569].flatten()
        for i in range (0,np.size(band21[:,0])):
            ax.plot(frequency[112:569], band21[i,112:569], color ='k' , linewidth=1)
            ax.set(xlabel='Frequency /MHz')
            if parameters[8]=='True' or parameters[1]=='True':
                if parameters[1]=='True' and parameters[8]=='True':
                    ax.set(ylabel='Global Normalised Power (Normalised Axis)')
                elif parameters[1]=='True':
                    ax.set(ylabel='Global Normalised Power')
                else:  
                    ax.set(ylabel='Power (Normalised Axis)')
            else:
                ax.set(ylabel='Power / a.u.')
            ax.set_ylim(((bandFd[a])*0.95),((bandFd[b])*1.05))
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.grid(c='darkgrey', which='major')
            ax.grid(c='gainsboro', which='minor')
            if parameters[6] == 'True':
                ax.axvline(x=frequency[483-mask_width],linestyle='--', color='salmon', linewidth=1)
                ax.axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
                ax.axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
                ax.axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
            ax.margins(x=0)
            roll_time = date_time + TimeDelta(i*60, format='sec')
            time_str = roll_time.strftime("%H:%M") 
            ax.set_title('Bandpass Spectrum, P(r,$\pi$) E \n'+user_inputs[0]+' '+time_str)
            plt.pause(which_speed)
            plt.cla()
        del band21, bandFd
    
    elif which_input ==4:
        if parameters[2]=='True':
            band22[:,(484-mask_width):(483+mask_width)+1] = 'nan'
        a = np.argmin(band22[:,112:569])
        b = np.argmax(band22[:,112:569])
        bandFd = band22[:,112:569].flatten()
        for i in range (0,np.size(band22[:,0])):
            try:

                ax.plot(frequency[112:569], (band11[i-3,112:569]), color ='grey' , linewidth=1)
                ax.plot(frequency[112:569], (band11[i-5,112:569]), color ='silver' , linewidth=1)

            except:
                pass
            ax.plot(frequency[112:569], band22[i,112:569], color ='k' , linewidth=1)

            ax.set(xlabel='Frequency /MHz')
            if parameters[8]=='True' or parameters[1]=='True':
                if parameters[1]=='True' and parameters[8]=='True':
                    ax.set(ylabel='Global Normalised Power (Normalised Axis)')
                elif parameters[1]=='True':
                    ax.set(ylabel='Global Normalised Power')
                else:  
                    ax.set(ylabel='Power (Normalised Axis)')
            else:
                ax.set(ylabel='Power / a.u.')
            plt.ylim(((bandFd[a])*0.95),((bandFd[b])*1.05))
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.grid(c='darkgrey', which='major')
            ax.grid(c='gainsboro', which='minor')
            if parameters[6] == 'True':
                plt.axvline(x=frequency[483-mask_width],linestyle='--', color='salmon', linewidth=1)
                plt.axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
                plt.axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
                plt.axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
            ax.margins(x=0)
            roll_time = date_time + TimeDelta(i*60, format='sec')
            time_str = roll_time.strftime("%H:%M") 
            ax.set_title('Bandpass Spectrum, P(r,0) W \n'+user_inputs[0]+' '+time_str)
            plt.pause(which_speed)
            plt.cla()
        del band22, bandFd

    else:
        print('\033[1;31m Input not recognised, please try again. \033[1;32m')
        bandpassANI(frequency)

    plt.close()

   
#==============================================================================

def bandpassANIFOUR(frequency):

    #animation of bandpass by bin

    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    del p11
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    date_time = Time(user_inputs[0]+'T'+user_inputs[3], format='isot', scale='utc', precision=4)

    print('')
    print('Please specify run-speed of animation, recommend displaying < 1 second per bin')
    print('')
    which_speed = input('Seconds per time-bin (e.g. 0.5): ')
    try:
       which_speed = float(which_speed)
    except:
       print('\033[1;31m Input not recognised, please try again. \033[1;32m')
       bandpassANIFOUR(frequency)

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    mask_width = parameters[4].astype(np.float)
    mask_width = 1+int(mask_width/2)

    band11 = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
    band12 = np.load (DATA_PATH+'/temp/a1p2_bandpass.npy')
    band21 = np.load (DATA_PATH+'/temp/a2p1_bandpass.npy')
    band22 = np.load (DATA_PATH+'/temp/a2p2_bandpass.npy')

    band11,band12,band21,band22 = corrections(band11,band12,band21,band22)

    if parameters[8]=='True':
        band11 = band11/np.mean(band11[:,112:569])
        band12 = band12/np.mean(band12[:,112:569])
        band21 = band21/np.mean(band21[:,112:569])
        band22 = band22/np.mean(band22[:,112:569])

    if parameters[2]=='True':
        band11[:,(484-mask_width):(483+mask_width)+1] = 'nan'
        band12[:,(484-mask_width):(483+mask_width)+1] = 'nan'
        band21[:,(484-mask_width):(483+mask_width)+1] = 'nan'
        band22[:,(484-mask_width):(483+mask_width)+1] = 'nan'


    Pmax = []  #find the minimum and maximum values to use across all 4 plots
    Pmin = []
    bandmin, bandmax = get_minmax(band11)
    Pmax.append(bandmax)
    Pmin.append(bandmin)
    bandmin, bandmax = get_minmax(band12)
    Pmax.append(bandmax)
    Pmin.append(bandmin)
    bandmin, bandmax = get_minmax(band21)
    Pmax.append(bandmax)
    Pmin.append(bandmin)
    bandmin, bandmax = get_minmax(band22)
    Pmax.append(bandmax)
    Pmin.append(bandmin)
    Pmax = np.asarray(Pmax)
    Pmin = np.asarray(Pmin)
    Lmin = np.argmin(Pmin)
    Pmin = Pmin[Lmin] * 0.95  #add small 5% margin to plots
    Lmax = np.argmax(Pmax)
    Pmax = Pmax[Lmax] * 1.05

    fig, axs = plt.subplots(2, 2)
    for i in range (0,np.size(band11[:,0])):
        axs[0,1].plot(frequency[112:569], band11[i,112:569], color ='k' , linewidth=1)
        axs[0,1].set_title('P(L,$\pi$) W')
        axs[0,1].margins(x=0)
        axs[0,0].plot(frequency[112:569], band12[i,112:569], color ='k' , linewidth=1)
        axs[0,0].set_title('P(L,0) E')
        axs[0,0].margins(x=0)
        axs[1,0].plot(frequency[112:569], band21[i,112:569], color ='k' , linewidth=1)
        axs[1, 0].set_title('P(R,$\pi$) E')
        axs[1,0].margins(x=0)
        axs[1,1].plot(frequency[112:569], band22[i,112:569], color ='k' , linewidth=1)
        axs[1, 1].set_title('P(R,0) W')
        axs[1,1].margins(x=0)
        if parameters[6] == 'True':
            axs[0,1].axvline(x=frequency[483-mask_width],linestyle='--', color='salmon', linewidth=1)
            axs[0,1].axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
            axs[0,1].axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
            axs[0,1].axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1)
            axs[0,0].axvline(x=frequency[483-mask_width],linestyle='--', color='salmon', linewidth=1)
            axs[0,0].axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
            axs[0,0].axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
            axs[0,0].axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1)
            axs[1,1].axvline(x=frequency[483-mask_width],linestyle='--', color='salmon', linewidth=1)
            axs[1,1].axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
            axs[1,1].axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
            axs[1,1].axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1)
            axs[1,0].axvline(x=frequency[483-mask_width],linestyle='--', color='salmon', linewidth=1)
            axs[1,0].axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
            axs[1,0].axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
            axs[1,0].axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1)
        axs[0,0].set_ylim(Pmin,Pmax)
        axs[0,1].set_ylim(Pmin,Pmax)
        axs[1,0].set_ylim(Pmin,Pmax)
        axs[1,1].set_ylim(Pmin,Pmax)
        axs[0,0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[0,0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[1,0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[1,0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[0,1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[0,1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[1,1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[1,1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[0,0].grid(c='darkgrey', which='major')
        axs[0,0].grid(c='gainsboro', which='minor')
        axs[0,1].grid(c='darkgrey', which='major')
        axs[0,1].grid(c='gainsboro', which='minor')
        axs[1,0].grid(c='darkgrey', which='major')
        axs[1,0].grid(c='gainsboro', which='minor')
        axs[1,1].grid(c='darkgrey', which='major')
        axs[1,1].grid(c='gainsboro', which='minor')

        
        roll_time = date_time + TimeDelta(i*60, format='sec')
        time_str = roll_time.strftime("%H:%M") 
        plt.suptitle('Bandpass Spectrum, All Inputs.\n'+user_inputs[0]+' '+time_str)
       # plt.tight_layout()
        for ax in axs.flat:
            ax.set(xlabel='Frequency /MHz')
            if parameters[8]=='True' or parameters[1]=='True':
                if parameters[1]=='True' and parameters[8]=='True':
                    ax.set(ylabel='Global Normalised Power (Normalised Axis)')
                elif parameters[1]=='True':
                    ax.set(ylabel='Global Normalised Power')
                else:  
                    ax.set(ylabel='Power (Normalised Axis)')
            else:
                ax.set(ylabel='Power / a.u.')

        plt.pause(which_speed)
        
        axs[0,0].cla()
        axs[0,1].cla()
        axs[1,0].cla()
        axs[1,1].cla()

    del band11, band12, band21, band22

    plt.close()
    
#==============================================================================


def onePlotFOURBin(frequency, save_it=False, first_loop=True, which_input=None, which_time=None):

    #prints up a single time bin

    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    del p11
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    date_time = Time(user_inputs[0]+'T'+user_inputs[3], format='isot', scale='utc', precision=0)    
    
    if first_loop:
        print('')
        print('Start time is '+user_inputs[3]+' & duration is '+str(duration_actual)+' hours.')
        print('')
        which_time = input('Please specify a bin time (HH:MM) ')
        try:
            which_timetime = Time(user_inputs[0]+'T'+which_time, format='isot', scale='utc', precision=4)
        except:
            print('Specified time not recognised, use 24hr clock. Please try again.')
            onePlotFOURBin(frequency)
    
    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    if save_it:
        which_timetime = Time(user_inputs[0]+'T'+which_time, format='isot', scale='utc', precision=4)
    time_diff = which_timetime - date_time
    bin_number = int(time_diff.sec / 60)

    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = 'Bandpass Spectrum, All Inputs \n'+user_inputs[0]+'_'+which_time

    if save_it:
        fig, axs = plt.subplots(2, 2, figsize=(12,8), dpi=300)
    else:
        fig, axs = plt.subplots(2, 2)

    mask_width = parameters[4].astype(np.float)
    mask_width = 1+int(mask_width/2)
    if parameters[6] == 'True':
        axs[0,1].axvline(x=frequency[483-mask_width],linestyle='--', color='salmon', linewidth=1)
        axs[0,1].axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
        axs[0,1].axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
        axs[0,1].axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1)
        axs[0,0].axvline(x=frequency[483-mask_width],linestyle='--', color='salmon', linewidth=1)
        axs[0,0].axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
        axs[0,0].axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
        axs[0,0].axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1)
        axs[1,1].axvline(x=frequency[483-mask_width],linestyle='--', color='salmon', linewidth=1)
        axs[1,1].axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
        axs[1,1].axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
        axs[1,1].axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1)
        axs[1,0].axvline(x=frequency[483-mask_width],linestyle='--', color='salmon', linewidth=1)
        axs[1,0].axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
        axs[1,0].axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
        axs[1,0].axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1)

    band11 = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
    band12 = np.load (DATA_PATH+'/temp/a1p2_bandpass.npy')
    band21 = np.load (DATA_PATH+'/temp/a2p1_bandpass.npy')
    band22 = np.load (DATA_PATH+'/temp/a2p2_bandpass.npy')

    band11,band12,band21,band22 = corrections(band11,band12,band21,band22)

    if parameters[8]=='True':
        band11 = band11/np.mean(band11[:,112:569])
        band12 = band12/np.mean(band12[:,112:569])
        band21 = band21/np.mean(band21[:,112:569])
        band22 = band22/np.mean(band22[:,112:569])

    Pmax = []  #find the minimum and maximum values to use across all 4 plots
    Pmin = []
    bandmin, bandmax = get_minmax(band11)
    Pmax.append(bandmax)
    Pmin.append(bandmin)
    bandmin, bandmax = get_minmax(band12)
    Pmax.append(bandmax)
    Pmin.append(bandmin)
    bandmin, bandmax = get_minmax(band21)
    Pmax.append(bandmax)
    Pmin.append(bandmin)
    bandmin, bandmax = get_minmax(band22)
    Pmax.append(bandmax)
    Pmin.append(bandmin)
    Pmax = np.asarray(Pmax)
    Pmin = np.asarray(Pmin)
    Lmin = np.argmin(Pmin)
    Pmin = Pmin[Lmin] * 0.95  #add small 5% margin to plots
    Lmax = np.argmax(Pmax)
    Pmax = Pmax[Lmax] * 1.05


    if parameters[2]=='True':
        band11[:,(484-mask_width):(483+mask_width)+1] = 'nan'
    axs[0,1].plot(frequency[112:569], band11[bin_number,112:569], color ='k' , linewidth=1)
    axs[0,1].set_title('P(L,$\pi$) W')
    axs[0,1].margins(x=0)
    del band11


    if parameters[2]=='True':
        band12[:,(484-mask_width):(483+mask_width)+1] = 'nan'
    axs[0,0].plot(frequency[112:569], band12[bin_number,112:569], color ='k' , linewidth=1)
    axs[0,0].set_title('P(L,0) E')
    axs[0,0].margins(x=0)
    del band12
  

    if parameters[2]=='True':
        band21[:,(484-mask_width):(483+mask_width)+1] = 'nan'
    axs[1,0].plot(frequency[112:569], band21[bin_number,112:569], color ='k' , linewidth=1)
    axs[1, 0].set_title('P(R,$\pi$) E')
    axs[1,0].margins(x=0)
    del band21
 

    if parameters[2]=='True':
        band22[:,(484-mask_width):(483+mask_width)+1] = 'nan'
    axs[1,1].plot(frequency[112:569], band22[bin_number,112:569], color ='k' , linewidth=1)
    axs[1, 1].set_title('P(R,0) W')
    axs[1,1].margins(x=0)
    del band22
    #Axis [1, 1]
    for ax in axs.flat:
        ax.set(xlabel='Frequency /MHz')
        if parameters[8]=='True' or parameters[1]=='True':
            if parameters[1]=='True' and parameters[8]=='True':
                ax.set(ylabel='Global Normalised Power (Normalised Axis)')
            elif parameters[1]=='True':
                ax.set(ylabel='Global Normalised Power')
            else:  
                ax.set(ylabel='Power (Normalised Axis)')
        else:
            ax.set(ylabel='Power / a.u.')

    plt.suptitle(title_string)
    fig.tight_layout()
    axs[0,0].set_ylim(Pmin,Pmax)
    axs[0,1].set_ylim(Pmin,Pmax)
    axs[1,0].set_ylim(Pmin,Pmax)
    axs[1,1].set_ylim(Pmin,Pmax)
    axs[0,0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[0,0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[1,0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[1,0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[0,1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[0,1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[1,1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[1,1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[0,0].grid(c='darkgrey', which='major')
    axs[0,0].grid(c='gainsboro', which='minor')
    axs[0,1].grid(c='darkgrey', which='major')
    axs[0,1].grid(c='gainsboro', which='minor')
    axs[1,0].grid(c='darkgrey', which='major')
    axs[1,0].grid(c='gainsboro', which='minor')
    axs[1,1].grid(c='darkgrey', which='major')
    axs[1,1].grid(c='gainsboro', which='minor')


    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(which_time)+'_BP.png', bbox_inches="tight")
        print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
    else:
        plt.show()
    plt.close()
    
    if first_loop:
        print ('')
        save = input ('Do you want to save a printer friendly copy of this plot? (Y/N): \033[0;m')
        if str(save) == 'N' or str(save) == 'n':
            print('\033[1;32m')
            pass
        elif str(save) == 'Y' or str(save) == 'y':
            save_it=True
            first_loop = False
            onePlotFOURBin(frequency, save_it, first_loop, which_input, which_time)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass


#------------------------------------------------------------------------------------
    
def FourPlotsStacked(frequency, save_it=False, first_loop=True):

    #prints up one smin bins ontop of one another
    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    
    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    del p11

    band11 = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
    colour_n = np.size(band11[:,0])
    del band11
    R = []
    G = []
    B = []
    for i in range(1,colour_n+1):
        if i <= (colour_n/4):
            red = 255
            green = int(255/((colour_n/4)*i))
            blue = 0
        elif i <= (colour_n/2) and i > (colour_n/4):
            red = int((-255)/(colour_n/4)*i + 255 * 2)
            green = 255
            blue = 0
        elif i <= (colour_n*0.75) and i > (colour_n/2):
            red = 0
            green = 255
            blue = int((255 / (colour_n / 4) * i + (-255 * 2)))
        elif i > (colour_n*0.75):
            red = 0
            green = int(-255 * i / (colour_n / 4) + (255 * 4))
            blue = 255
        R.append(red/255)
        G.append(green/255)
        B.append(blue/255)

    band11 = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
    band12 = np.load (DATA_PATH+'/temp/a1p2_bandpass.npy')
    band21 = np.load (DATA_PATH+'/temp/a2p1_bandpass.npy')
    band22 = np.load (DATA_PATH+'/temp/a2p2_bandpass.npy')

    print(np.shape(band11))
    print(np.shape(band12))
    print(np.shape(band21))
    print(np.shape(band22))

    print(band11[10,250:260])
    print(band22[10,250:260])
    print(band21[10,250:260])
    print(band12[10,250:260])

    band11,band12,band21,band22 = corrections(band11,band12,band21,band22)

    if parameters[8]=='True':
        band11 = band11/np.mean(band11[:,112:569])
        band12 = band12/np.mean(band12[:,112:569])
        band21 = band21/np.mean(band21[:,112:569])
        band22 = band22/np.mean(band22[:,112:569])

    Pmax = []  #find the minimum and maximum values to use across all 4 plots
    Pmin = []
    bandmin, bandmax = get_minmax(band11)
    Pmax.append(bandmax)
    Pmin.append(bandmin)
    bandmin, bandmax = get_minmax(band12)
    Pmax.append(bandmax)
    Pmin.append(bandmin)
    bandmin, bandmax = get_minmax(band21)
    Pmax.append(bandmax)
    Pmin.append(bandmin)
    bandmin, bandmax = get_minmax(band22)
    Pmax.append(bandmax)
    Pmin.append(bandmin)
    Pmax = np.asarray(Pmax)
    Pmin = np.asarray(Pmin)
    Lmin = np.argmin(Pmin)
    Pmin = Pmin[Lmin] * 0.95  #add small 5% margin to plots
    Lmax = np.argmax(Pmax)
    Pmax = Pmax[Lmax] * 1.05

    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = 'Bandpass Spectrum, All Inputs.\n'+user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours.'
    
    if save_it:
        fig, axs = plt.subplots(2, 2, figsize=(12,8), dpi=300)
    else:
        fig, axs = plt.subplots(2, 2)

    mask_width = parameters[4].astype(np.float)
    mask_width = 1+int(mask_width/2)
    if parameters[6] == 'True':
        axs[0,1].axvline(x=frequency[483-mask_width],linestyle='--', color='salmon', linewidth=1)
        axs[0,1].axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
        axs[0,1].axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
        axs[0,1].axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1)
        axs[0,0].axvline(x=frequency[483-mask_width],linestyle='--', color='salmon', linewidth=1)
        axs[0,0].axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
        axs[0,0].axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
        axs[0,0].axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1)
        axs[1,1].axvline(x=frequency[483-mask_width],linestyle='--', color='salmon', linewidth=1)
        axs[1,1].axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
        axs[1,1].axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
        axs[1,1].axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1)
        axs[1,0].axvline(x=frequency[483-mask_width],linestyle='--', color='salmon', linewidth=1)
        axs[1,0].axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
        axs[1,0].axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
        axs[1,0].axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1)


    if parameters[2]=='True':
        band11[:,(484-mask_width):(483+mask_width)+1] = 'nan'
    for i in range (0,np.size(band11[:,0])):
        axs[0,1].plot(frequency[:], band11[i,:], color =(R[i],G[i],B[i]), linewidth=0.5) #112:569
    
    #axs[0,1].plot(frequency[112:569], band11[0,112:569], color = (R[0],G[0],B[0]), linewidth =1, label='First Bin')
    #axs[0,1].plot(frequency[112:569], band11[-1,112:569], color = (R[-1],G[-1],B[-1]), linewidth =1, label='Final Bin')
    axs[0,1].set_title('P(L,$\pi$) W')
    axs[0,1].margins(x=0)
    del band11


    if parameters[2]=='True':
        band12[:,(484-mask_width):(483+mask_width)+1] = 'nan'
    for i in range (0,np.size(band12[:,0])):
        axs[0,0].plot(frequency[:], band12[i,:], color =(R[i],G[i],B[i]), linewidth=0.5)

    #axs[0,0].plot(frequency[112:569], band12[0,112:569], color = (R[0],G[0],B[0]), linewidth =1, label='First Bin')
    #axs[0,0].plot(frequency[112:569], band12[-1,112:569], color = (R[-1],G[-1],B[-1]), linewidth =1, label='Final Bin')
    axs[0,0].set_title('P(L,0) E')
    axs[0,0].margins(x=0)
    del band12
  

    if parameters[2]=='True':
        band21[:,(484-mask_width):(483+mask_width)+1] = 'nan'
    for i in range (0,np.size(band21[:,0])):
        axs[1,0].plot(frequency[:], band21[i,:], color =(R[i],G[i],B[i]), linewidth=0.5)
    
    #axs[1,0].plot(frequency[112:569], band21[0,112:569], color = (R[0],G[0],B[0]), linewidth =1)
    #axs[1,0].plot(frequency[112:569], band21[-1,112:569], color = (R[-1],G[-1],B[-1]), linewidth =1)
    axs[1, 0].set_title('P(R,$\pi$) E')
    axs[1,0].margins(x=0)
    del band21
 

    if parameters[2]=='True':
        band22[:,(484-mask_width):(483+mask_width)+1] = 'nan'
    for i in range (0,np.size(band22[:,0])):
        axs[1,1].plot(frequency[:], band22[i,:], color =(R[i],G[i],B[i]), linewidth=0.5)
    
    #axs[1,1].plot(frequency[112:569], band22[0,112:569], color = (R[0],G[0],B[0]), linewidth =1)
    #axs[1,1].plot(frequency[112:569], band22[-1,112:569], color = (R[-1],G[-1],B[-1]), linewidth =1)
    axs[1, 1].set_title('P(R,0) W')
    axs[1,1].margins(x=0)
    del band22

    try:
        axs[0,0].set_ylim(Pmin,Pmax)
        axs[0,1].set_ylim(Pmin,Pmax)
        axs[1,0].set_ylim(Pmin,Pmax)
        axs[1,1].set_ylim(Pmin,Pmax)
    except:
        pass
    axs[0,0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[0,0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[1,0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[1,0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[0,1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[0,1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[1,1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[1,1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[0,0].grid(c='darkgrey', which='major')
    axs[0,0].grid(c='gainsboro', which='minor')
    axs[0,1].grid(c='darkgrey', which='major')
    axs[0,1].grid(c='gainsboro', which='minor')
    axs[1,0].grid(c='darkgrey', which='major')
    axs[1,0].grid(c='gainsboro', which='minor')
    axs[1,1].grid(c='darkgrey', which='major')
    axs[1,1].grid(c='gainsboro', which='minor')

    #Axis [1, 1]
    for ax in axs.flat:
        ax.set(xlabel='Frequency /MHz')
        if parameters[8]=='True' or parameters[1]=='True':
            if parameters[1]=='True' and parameters[8]=='True':
                ax.set(ylabel='Global Normalised Power (Normalised Axis)')
            elif parameters[1]=='True':
                ax.set(ylabel='Global Normalised Power')
            else:  
                ax.set(ylabel='Power (Normalised Axis)')
        else:
            ax.set(ylabel='Power / a.u.')

    plt.suptitle(title_string)
    plt.tight_layout()
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_BP-stacked.png', bbox_inches="tight")
        print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
    else:
        plt.show()
    plt.close()
    
    if first_loop:
        print ('')
        save = input ('Do you want to save a printer friendly copy of this plot? (Y/N): \033[0;m')
        if str(save) == 'N' or str(save) == 'n':
            print('\033[1;32m')
            pass
        elif str(save) == 'Y' or str(save) == 'y':
            save_it=True
            first_loop = False
            FourPlotsStacked(frequency, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass


#-----------------------------------------------------------------------


def bandpassNULLS(save_it=False, first_loop=True):

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    freq = np.load(DATA_PATH+'/temp/freq.npy') #its in hertz
    flatten = np.load(DATA_PATH+'/temp/flatten.npy')
    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    del p11

    band11 = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
    colour_n = np.size(band11[:,0])
    del band11
    R = []
    G = []
    B = []
    for i in range(1,colour_n+1):
        if i <= (colour_n/4):
            red = 255
            green = int(255/((colour_n/4)*i))
            blue = 0
        elif i <= (colour_n/2) and i > (colour_n/4):
            red = int((-255)/(colour_n/4)*i + 255 * 2)
            green = 255
            blue = 0
        elif i <= (colour_n*0.75) and i > (colour_n/2):
            red = 0
            green = 255
            blue = int((255 / (colour_n / 4) * i + (-255 * 2)))
        elif i > (colour_n*0.75):
            red = 0
            green = int(-255 * i / (colour_n / 4) + (255 * 4))
            blue = 255
        R.append(red/255)
        G.append(green/255)
        B.append(blue/255)


    a1p1B = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
    a1p2B = np.load (DATA_PATH+'/temp/a1p2_bandpass.npy')
    a2p1B = np.load (DATA_PATH+'/temp/a2p1_bandpass.npy')
    a2p2B = np.load (DATA_PATH+'/temp/a2p2_bandpass.npy')

    a1p1B,a1p2B,a2p1B,a2p2B = corrections(a1p1B,a1p2B,a2p1B,a2p2B)

    a1p1B = a1p1B[:,112:569]
    a1p2B = a1p2B[:,112:569]
    a2p1B = a2p1B[:,112:569]
    a2p2B = a2p2B[:,112:569]

    if parameters[8]=='True':
        a1p1B  = a1p1B /np.mean(a1p1B)
        a1p2B  = a1p2B /np.mean(a1p2B)
        a2p1B  = a2p1B /np.mean(a2p1B)
        a2p2B  = a2p2B /np.mean(a2p2B)

    JPL_d1 = a1p1B - a1p2B # lpi - l0 W-E
    JPL_d2 = a2p2B - a2p1B # r0 - rpi W-E
    NULL_IB2_a = a1p2B - a2p1B # l0 - rpi  E-E
    NULL_IB2_b = a2p2B - a1p1B # r0 - lpi   W-W

    NULL_JPL = (JPL_d1 - JPL_d2)/2
   
    del a1p2B, a2p1B, a2p2B, a1p1B
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = 'Bandpass Spectrum, Nulls\n'+user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours.'

    if save_it:
        f, axarr = plt.subplots(1,3, figsize=(12,8), dpi=300)
    else:
        f, axarr = plt.subplots(1,3)


    for i in range (0,np.size(NULL_JPL[:,0])):
        axarr[0].plot(freq[112:569], NULL_JPL[i], color =(R[i],G[i],B[i]), linewidth=0.2 )
        axarr[0].set_title('JPL Null [(L,$\pi$-L,0)-(R,0-R,$\pi$)]/2')
        axarr[0].margins(x=0)
       
        axarr[1].plot(freq[112:569], NULL_IB2_a[i], color =(R[i],G[i],B[i]), linewidth=0.2 )
        axarr[1].set_title('IB Null [(L,0)E - (R,$\pi$)E]')
        axarr[1].margins(x=0)
  
        axarr[2].plot(freq[112:569], NULL_IB2_b[i], color =(R[i],G[i],B[i]), linewidth=0.2 )
        axarr[2].set_title('IB Null [(R,0)W - (L,$\pi$)W]')
        axarr[2].margins(x=0)

    for ax in axarr.flat:
        ax.set(xlabel='Frequency /MHz')
        if parameters[8]=='True' or parameters[1]=='True':
            if parameters[1]=='True' and parameters[8]=='True':
                ax.set(ylabel='Global Normalised Power (Normalised Axis)')
            elif parameters[1]=='True':
                ax.set(ylabel='Global Normalised Power')
            else:  
                ax.set(ylabel='Power (Normalised Axis)')
        else:
            ax.set(ylabel='Power / a.u.')
    
    axarr[0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    axarr[0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axarr[0].grid(c='darkgrey', which='major')
    axarr[0].grid(c='gainsboro', which='minor')
    axarr[1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    axarr[1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axarr[1].grid(c='darkgrey', which='major')
    axarr[1].grid(c='gainsboro', which='minor')
    axarr[2].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    axarr[2].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axarr[2].grid(c='darkgrey', which='major')
    axarr[2].grid(c='gainsboro', which='minor')

    plt.suptitle(title_string)
    plt.tight_layout()

    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_WFNULLS.png', bbox_inches="tight")
        print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
    else:
        plt.show()
    plt.close()
    
    if first_loop:
        print ('')
        save = input ('Do you want to save a printer friendly copy of this plot? (Y/N): \033[0;m')
        if str(save) == 'N' or str(save) == 'n':
            print('\033[1;32m')
            pass
        elif str(save) == 'Y' or str(save) == 'y':
            save_it=True
            first_loop = False
            bandpassNULLS(save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass



#-----------------------------------------------------------------------------------------

def bandpassNULLSani(save_it=False, first_loop=True):

    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False

    print('')
    print('Please specify run-speed of animation, recommend displaying < 1 second per bin')
    print('')
    which_speed = input('Seconds per time-bin (e.g. 0.5): ')
    try:
       which_speed = float(which_speed)
    except:
       print('\033[1;31m Input not recognised, please try again. \033[1;32m')
       bandpassANIFOUR(frequency)

    freq = np.load(DATA_PATH+'/temp/freq.npy') #its in hertz
    flatten = np.load(DATA_PATH+'/temp/flatten.npy')
    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    date_time = Time(user_inputs[0]+'T'+user_inputs[3], format='isot', scale='utc', precision=4)
    del p11

    a1p1B = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
    a1p2B = np.load (DATA_PATH+'/temp/a1p2_bandpass.npy')
    a2p1B = np.load (DATA_PATH+'/temp/a2p1_bandpass.npy')
    a2p2B = np.load (DATA_PATH+'/temp/a2p2_bandpass.npy')

    a1p1B,a1p2B,a2p1B,a2p2B = corrections(a1p1B,a1p2B,a2p1B,a2p2B)

    a1p1B = a1p1B[:,112:569]
    a1p2B = a1p2B[:,112:569]
    a2p1B = a2p1B[:,112:569]
    a2p2B = a2p2B[:,112:569]

    if parameters[8]=='True':
        a1p1B  = a1p1B /np.mean(a1p1B)
        a1p2B  = a1p2B /np.mean(a1p2B)
        a2p1B  = a2p1B /np.mean(a2p1B)
        a2p2B  = a2p2B /np.mean(a2p2B)

    JPL_d1 = a1p1B - a1p2B # lpi - l0 W-E
    JPL_d2 = a2p2B - a2p1B # r0 - rpi W-E
    NULL_IB2_a = a1p2B - a2p1B # l0 - rpi  E-E
    NULL_IB2_b = a2p2B - a1p1B # r0 - lpi   W-W

    NULL_JPL = (JPL_d1 - JPL_d2)/2  
   
    del a1p2B, a2p1B, a2p2B, a1p1B
    
    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours. Nulls'



    f, axarr = plt.subplots(1,3)


    for i in range (0,np.size(NULL_JPL[:,0])):
        axarr[0].plot(freq[112:569], NULL_JPL[i], color ='k', linewidth=1 )
        axarr[0].set_title('JPL Null [(L,$\pi$-L,0)-(R,0-R,$\pi$)]/2')
        axarr[0].margins(x=0)
        #axarr[0].set_ylim(top=3,bottom=-1)
       
        axarr[1].plot(freq[112:569], NULL_IB2_a[i], color ='k', linewidth=1 )
        axarr[1].set_title('IB Null [(L,0)E - (R,$\pi$)E]')
        axarr[1].margins(x=0)
        #axarr[1].set_ylim(top=3,bottom=-1)
  
        axarr[2].plot(freq[112:569], NULL_IB2_b[i], color ='k', linewidth=1 )
        axarr[2].set_title('IB Null [(R,0)W - (L,$\pi$)W]')
        axarr[2].margins(x=0)
        #axarr[2].set_ylim(top=3,bottom=-1)
        roll_time = date_time + TimeDelta(i*60, format='sec')
        time_str = roll_time.strftime("%H:%M") 
        plt.suptitle(user_inputs[0]+' '+time_str+' Nulls '+str(i))
 

        plt.pause(which_speed)
        
        axarr[0].cla()
        axarr[1].cla()
        axarr[2].cla()

#----------------------------------------------------------------------------------------

def FourPlotsStacked_firstdiff(frequency, save_it=False, first_loop=True):

    #prints up one smin bins ontop of one another
    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    
    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    del p11

    band11 = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
    colour_n = np.size(band11[:,0])
    del band11
    R = []
    G = []
    B = []
    for i in range(1,colour_n+1):
        if i <= (colour_n/4):
            red = 255
            green = int(255/((colour_n/4)*i))
            blue = 0
        elif i <= (colour_n/2) and i > (colour_n/4):
            red = int((-255)/(colour_n/4)*i + 255 * 2)
            green = 255
            blue = 0
        elif i <= (colour_n*0.75) and i > (colour_n/2):
            red = 0
            green = 255
            blue = int((255 / (colour_n / 4) * i + (-255 * 2)))
        elif i > (colour_n*0.75):
            red = 0
            green = int(-255 * i / (colour_n / 4) + (255 * 4))
            blue = 255
        R.append(red/255)
        G.append(green/255)
        B.append(blue/255)

    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')
    title_string = 'Bandpass Spectrum, Single Differences.\n'+user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours.'
    
    if save_it:
        fig, axs = plt.subplots(2, 2, figsize=(12,8), dpi=300)
    else:
        fig, axs = plt.subplots(2, 2)

    mask_width = parameters[4].astype(np.float)
    mask_width = 1+int(mask_width/2)
    if parameters[6] == 'True':
        axs[0,1].axvline(x=frequency[483-mask_width],linestyle='--', color='salmon', linewidth=1)
        axs[0,1].axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
        axs[0,1].axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
        axs[0,1].axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1)
        axs[0,0].axvline(x=frequency[483-mask_width],linestyle='--', color='salmon', linewidth=1)
        axs[0,0].axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
        axs[0,0].axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
        axs[0,0].axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1)
        axs[1,1].axvline(x=frequency[483-mask_width],linestyle='--', color='salmon', linewidth=1)
        axs[1,1].axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
        axs[1,1].axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
        axs[1,1].axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1)
        axs[1,0].axvline(x=frequency[483-mask_width],linestyle='--', color='salmon', linewidth=1)
        axs[1,0].axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
        axs[1,0].axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
        axs[1,0].axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1)

    band11 = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
    band12 = np.load (DATA_PATH+'/temp/a1p2_bandpass.npy')
    band21 = np.load (DATA_PATH+'/temp/a2p1_bandpass.npy')
    band22 = np.load (DATA_PATH+'/temp/a2p2_bandpass.npy')

    band11,band12,band21,band22 = corrections(band11,band12,band21,band22)

    if parameters[8]=='True':
        band11 = band11/np.mean(band11[:,112:569])
        band12 = band12/np.mean(band12[:,112:569])
        band21 = band21/np.mean(band21[:,112:569])
        band22 = band22/np.mean(band22[:,112:569])


    d_a = band12 - band11# l0 - lpi 
    d_b = band22 - band21#r0 - rpi 
    d_c = band12 - band22# l0 - r0 
    d_d = band11 - band21#   lpi - rpi 
    del band11, band12, band21, band22

    if parameters[2]=='True':
        d_b[:,(484-mask_width):(483+mask_width)+1] = 'nan'
    for i in range (0,np.size(d_b[:,0])):
        axs[0,1].plot(frequency[112:569], d_b[i,112:569], color =(R[i],G[i],B[i]), linewidth=0.5)
    
    #axs[0,1].plot(frequency[112:569], d_b[0,112:569], color = (R[0],G[0],B[0]), linewidth =1, label='First Bin')
    #axs[0,1].plot(frequency[112:569], d_b[-1,112:569], color = (R[-1],G[-1],B[-1]), linewidth =1, label='Final Bin')
    axs[0,1].set_title('$\delta$b = P(R,0)W - P(R,$\pi$)E')
    axs[0,1].margins(x=0)
    del d_b

    if parameters[2]=='True':
        d_a[:,(484-mask_width):(483+mask_width)+1] = 'nan'
    #axs[0,0].plot(frequency[112:569], d_a[0,112:569], color = (R[0],G[0],B[0]), linewidth =1, label='First Bin')
    for i in range (0,np.size(d_a[:,0])):
        axs[0,0].plot(frequency[112:569], d_a[i,112:569], color =(R[i],G[i],B[i]), linewidth=0.5)
    #axs[0,0].plot(frequency[112:569], d_a[-1,112:569], color = (R[-1],G[-1],B[-1]), linewidth =1, label='Final Bin')
    axs[0,0].set_title('$\delta$a = P(L,0)E - P(L,$\pi$)W')
    axs[0,0].margins(x=0)
    del d_a
  

    if parameters[2]=='True':
        d_c[:,(484-mask_width):(483+mask_width)+1] = 'nan'
    for i in range (0,np.size(d_c[:,0])):
        axs[1,0].plot(frequency[112:569], d_c[i,112:569], color =(R[i],G[i],B[i]), linewidth=0.5)
    
    #axs[1,0].plot(frequency[112:569], d_c[0,112:569], color = (R[0],G[0],B[0]), linewidth =1)
    #axs[1,0].plot(frequency[112:569], d_c[-1,112:569], color = (R[-1],G[-1],B[-1]), linewidth =1)
    axs[1, 0].set_title('$\delta$c = P(L,0)E - P(R,0)W')
    axs[1,0].margins(x=0)
    del d_c
 

    if parameters[2]=='True':
        d_d[:,(484-mask_width):(483+mask_width)+1] = 'nan'
    for i in range (0,np.size(d_d[:,0])):
        axs[1,1].plot(frequency[112:569], d_d[i,112:569], color =(R[i],G[i],B[i]), linewidth=0.5)
    
    #axs[1,1].plot(frequency[112:569], d_d[0,112:569], color = (R[0],G[0],B[0]), linewidth =1)
    #axs[1,1].plot(frequency[112:569], d_d[-1,112:569], color = (R[-1],G[-1],B[-1]), linewidth =1)
    axs[1, 1].set_title('$\delta$d = P(L,$\pi$)W - P(R,$\pi$)E')
    axs[1,1].margins(x=0)
    del d_d
    #Axis [1, 1]


    for ax in axs.flat:
        ax.set(xlabel='Frequency /MHz')
        if parameters[8]=='True' or parameters[1]=='True':
            if parameters[1]=='True' and parameters[8]=='True':
                ax.set(ylabel='Global Normalised Power (Normalised Axis)')
            elif parameters[1]=='True':
                ax.set(ylabel='Global Normalised Power')
            else:  
                ax.set(ylabel='Power (Normalised Axis)')
        else:
            ax.set(ylabel='Power / a.u.')

    axs[0,0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[0,0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[1,0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[1,0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[0,1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[0,1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[1,1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[1,1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[0,0].grid(c='darkgrey', which='major')
    axs[0,0].grid(c='gainsboro', which='minor')
    axs[0,1].grid(c='darkgrey', which='major')
    axs[0,1].grid(c='gainsboro', which='minor')
    axs[1,0].grid(c='darkgrey', which='major')
    axs[1,0].grid(c='gainsboro', which='minor')
    axs[1,1].grid(c='darkgrey', which='major')
    axs[1,1].grid(c='gainsboro', which='minor')
    plt.suptitle(title_string)
    fig.tight_layout()
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_BP-stacked_SD.png', bbox_inches="tight")
        print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
    else:
        plt.show()
    plt.close()
    
    if first_loop:
        print ('')
        save = input ('Do you want to save a printer friendly copy of this plot? (Y/N): \033[0;m')
        if str(save) == 'N' or str(save) == 'n':
            print('\033[1;32m')
            pass
        elif str(save) == 'Y' or str(save) == 'y':
            save_it=True
            first_loop = False
            FourPlotsStacked_firstdiff(frequency, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass


#-----------------------------------------------------------------------

def OnePlotsStacked_dbldiff(frequency, save_it=False, first_loop=True):

    #prints up one smin bins ontop of one another
    parameters = np.load (DATA_PATH+'/temp/parameters.npy')
    if parameters[5] == 'True':
        pass
    else:
        first_loop=False
    
    p11 = np.load (DATA_PATH+'/temp/a1p1_power.npy')
    duration_actual = (np.size(p11) / 60)
    duration_actual = np.round(duration_actual, decimals=2)
    del p11

    band11 = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
    colour_n = np.size(band11[:,0])
    del band11
    R = []
    G = []
    B = []
    for i in range(1,colour_n+1):
        if i <= (colour_n/4):
            red = 255
            green = int(255/((colour_n/4)*i))
            blue = 0
        elif i <= (colour_n/2) and i > (colour_n/4):
            red = int((-255)/(colour_n/4)*i + 255 * 2)
            green = 255
            blue = 0
        elif i <= (colour_n*0.75) and i > (colour_n/2):
            red = 0
            green = 255
            blue = int((255 / (colour_n / 4) * i + (-255 * 2)))
        elif i > (colour_n*0.75):
            red = 0
            green = int(-255 * i / (colour_n / 4) + (255 * 4))
            blue = 255
        R.append(red/255)
        G.append(green/255)
        B.append(blue/255)

    user_inputs = np.load(DATA_PATH+'/temp/inputs.npy')

    title_string = 'Bandpass Spectrum, Double Difference.\n'+user_inputs[0]+'_'+user_inputs[3]+ ' for '+str(duration_actual)+' hours.'

   
    
    if save_it:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    else:
        fig, ax = plt.subplots()

    mask_width = parameters[4].astype(np.float)
    mask_width = 1+int(mask_width/2)
    if parameters[6] == 'True':
        plt.axvline(x=frequency[483-mask_width],linestyle='--', color='salmon', linewidth=1)
        plt.axvline(x=frequency[483+mask_width+1],linestyle='--', color='c', linewidth=1)
        plt.axvline(x=frequency[483],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')
        plt.axvline(x=frequency[484],linestyle='-', color='navy', linewidth=1, label='1420.405 MHz')

    band11 = np.load (DATA_PATH+'/temp/a1p1_bandpass.npy')
    band12 = np.load (DATA_PATH+'/temp/a1p2_bandpass.npy')
    band21 = np.load (DATA_PATH+'/temp/a2p1_bandpass.npy')
    band22 = np.load (DATA_PATH+'/temp/a2p2_bandpass.npy')

    band11,band12,band21,band22 = corrections(band11,band12,band21,band22)

    if parameters[8]=='True':
        band11 = band11/np.mean(band11[:,112:569])
        band12 = band12/np.mean(band12[:,112:569])
        band21 = band21/np.mean(band21[:,112:569])
        band22 = band22/np.mean(band22[:,112:569])


    d_a = band12 - band11# l0 - lpi 
    d_b = band22 - band21#r0 - rpi 
    DD = (d_a - d_b)/2 # WMAP 
    del band11, band12, band21, band22, d_a, d_b
    
    if parameters[2]=='True':
        DD[:,(484-mask_width):(483+mask_width+1)] = 'nan'
    for i in range (0,np.size(DD[:,0])):
        ax.plot(frequency[112:569], DD[i,112:569], color =(R[i],G[i],B[i]), linewidth=0.5)
    ax.set(xlabel='Frequency / MHz')

    if parameters[8]=='True' or parameters[1]=='True':
        if parameters[1]=='True' and parameters[8]=='True':
            ax.set(ylabel='Global Normalised Power (Normalised Axis)')
        elif parameters[1]=='True':
            ax.set(ylabel='Global Normalised Power')
        else:  
            ax.set(ylabel='Power (Normalised Axis)')
    else:
        ax.set(ylabel='Power / a.u.')
    ax.set_title(title_string)
    ax.margins(x=0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(c='darkgrey', which='major')
    ax.grid(c='gainsboro', which='minor')

    del DD

    fig.tight_layout()
    if save_it:
        plt.savefig(user_inputs[6]+user_inputs[0]+'_'+user_inputs[3]+'_'+str(duration_actual)+'_BP-stacked_DD.png', bbox_inches="tight")
        print('\033[0;m Plot file saved in',user_inputs[6],'\033[1;32m')
    else:
        plt.show()
    plt.close()
    
    if first_loop:
        print ('')
        save = input ('Do you want to save a printer friendly copy of this plot? (Y/N): \033[0;m')
        if str(save) == 'N' or str(save) == 'n':
            print('\033[1;32m')
            pass
        elif str(save) == 'Y' or str(save) == 'y':
            save_it=True
            first_loop = False
            OnePlotsStacked_dbldiff(frequency, save_it, first_loop)
            print('\033[1;32m')
        else:
            print('')
            print('\033[1;31m User input not recognised, plot not saved. \033[1;32m')
            pass

    
####################################################################

def BandPassMenu(duration_actual, frequency):

    looper = True

    print ('')
    print ('   -------------------------------------')
    print ('   >>>      BANDPASS PLOTS MENU      <<<')
    print ('   -------------------------------------')
    print ('')
    print ('   1 - All Inputs')
    print ('   2 - Single Differences') 
    print ('   3 - Double Difference') 
    print ('   4 - Nulls') 
    print ('   5 - One Input') 
    print ('   6 - All Inputs @ Specific Time') 
    print ('   7 - One Input @ Specific Time') 
    print ('   8 - All Inputs Animation') 
    print ('   9 - One Input Animation') 
   # print ('  10 - Nulls Animation')  
    print ('')
    print ('')
    print ('   0 - Return to Quick-look menu')
    print ('')
    choice = input('Select menu option (number): \033[0;m')
    print('\033[1;32m')
    if choice.isdigit():

        if int(choice)==1:
            FourPlotsStacked(frequency) 

        elif int(choice) == 2:
            FourPlotsStacked_firstdiff(frequency)

        elif int(choice) == 3:
            OnePlotsStacked_dbldiff(frequency)

        elif int(choice) == 4:
            bandpassNULLS()
            
        elif int(choice) ==5:
            onePlotsStacked(frequency)
  
        elif int(choice) ==6:
            onePlotFOURBin(frequency)

        elif int(choice) == 7:
            onePlotoneBin(frequency)

        elif int(choice) == 8:
            bandpassANIFOUR(frequency)

        elif int(choice) == 9:
            bandpassANI(frequency)

     #   elif int(choice) == 10:
      #      bandpassNULLSani()

        elif int(choice) == 0:
            looper = False
            pass

    else:
        print('\033[1;31m Invalid selection by user. Please try again. \033[1;32m')
        BandPassMenu(duration_actual, frequency)
   
    return looper


duration_actual = np.load(DATA_PATH+'/temp/duration_actual.npy')
frequency = np.load(DATA_PATH+'/temp/freq.npy') #its in MHz

looper = True
while looper:
    looper = BandPassMenu(duration_actual, frequency)

print ('\033[1;32m ')


os.system('chmod -R -f 0777 /mirror/scratch/pblack || true')

