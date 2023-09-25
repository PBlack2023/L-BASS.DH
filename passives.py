#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 17:19:48 2022

@author: pblack
"""
DATA_PATH = '/mirror/scratch/pblack'


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:33:26 2022

@author: pblack
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import math
import scipy
import numpy as np
import astropy.io
from astropy.io import fits
from astropy.time import Time 
from astropy.time import TimeDelta
import datetime
import time
from tqdm import tqdm #progress bars
import glob
import os
from math import nan
import csv
from csv import DictReader
import pandas as pd
from matplotlib.pyplot import figure

os.chdir('/scratch/nas_lbass/raw_data/')

# Passive system lossy behaviour vs temperature 

# 273.15 K = 0 celius


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
                a1p1 = a1p1 / np.mean(normload[112:569]) #flatten bandpass
                normload = np.load(bandpass_norms[1]+'.npy')
                a1p2 = a1p2 / np.mean(normload[112:569]) #flatten bandpass
                normload = np.load(bandpass_norms[2]+'.npy')
                a2p1 = a2p1 / np.mean(normload[112:569]) #flatten bandpass
                normload = np.load(bandpass_norms[3]+'.npy')
                a2p2 = a2p2 / np.mean(normload[112:569]) #flatten bandpass
  
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


def passive_factors(west_data, east_data):  

    temp_data = np.load(DATA_PATH+'/temp/one_wire.npy', allow_pickle=True) 

    series_E_a1 = []
    series_W_a1 = []
    series_E_a2 = []
    series_W_a2 = []
    series_E_a3 = []
    series_W_a3 = []
    series_E_a_total = []
    series_W_a_total = []

    series_E_AN = []
    series_W_AN = []
    series_E_4nM = []
    series_W_4nM = []
    series_E_RxN = []
    series_W_RxN = []
    series_E_TN_total = []
    series_W_TN_total = []

    i=0

    for i in tqdm(range (0,np.size(temp_data[:,0]),1)):

        #--------------------------------------------------------------------------------
        ################################################################################
	
        #INPUT PARAMARTERS
	
        #-------------------------------------------------------------------------------
	
        # INITIAL RADIOMETRIC INPUT (K)

     #   E_T_ant, W_T_ant = 6.2,6.2

        E_T_ant = east_data[i]
        W_T_ant = west_data[i]

     #   E_T_ant = 277.116
     #   W_T_ant = 277.116

        # this needs modifying to sequentially load bins of radio data	


        # ANTENNA
	
        E_a_ant, W_a_ant = 0.982447, 0.984211 # zone V transmission factor is dominated by polariser hence effectively = aant
	
        E_r_ant, W_r_ant = 0.000664, 0.001928 # reflection ratio (also as measured for the antenna with metal sheet over mouth)
	
        E_lambda_ant, W_lambda_ant = 1.000000, 1.000000 # decrease (increase) in antenna transmission factor /K above (below) 	297K  
	
        # 4m CABLE & CONNECTORS
	
        #East is cable 1A, West is cable 2A
	
        E_a_4m, W_a_4m = 0.85133, 0.85133 #cable transmission factor
	
        E_a_con1, W_a_con1 = 0.98572, 0.98572 #connector1 transmission factor
	
        E_a_con2, W_a_con2 = 0.99529, 0.99529 #connector2 transmission factor
	
        E_lambda_4m, W_lambda_4m = 1.000242, 1.000242 #decrease (increase) in cable transmission factor /K above (below) 297K    
	
        E_lambda_con, W_lambda_con = 1.000092, 1.000092 # decrease (increase) in connector transmission factor /K above (below) 	297K      
	
        E_r_con1, W_r_con1 = 0.006607, 0.003162 # fraction of input power reflected at connector 1 (assumed all at connector 1 	for simplicity)
	
        E_r_con2, W_r_con2 = 0.00000, 0.00000 # fraction of input power reflected at connector 2 
	
	
        # INSIDE RECEIVER BOX
	 
        E_a_T, W_a_T = 1.00000, 1.00000 # 1st magic-T transmission factor (assumed 1 currently)    
	
        E_r_T, W_r_T = 0.00000, 0.00000 # 1st magic-T reflection ratio (assumed 0 currently)   
	
        E_lambda_T, W_lambda_T = 1.00000, 1.00000 # decrease (increase) in magic-T transmission factor /K above (below) 297K     
	
        E_a_NSMA, W_a_NSMA = 0.98628, 0.98628 # NSMA transmission factor
	
        E_a_SMA, W_a_SMA = 0.90136, 0.90116 # 1m SMA cable transmission factor
	
        E_r_NSMA, W_r_NSMA = 0.00000, 0.00000 # NSMA reflection ratio (assumed 0 currently)
	
        E_r_SMA, W_r_SMA = 0.00000, 0.00000 # reflection ratio at input to 1-m SMA cable (currently set to 0)
	
        E_r_LNA, W_r_LNA = 0.00000, 0.00000 # reflection ratio at input to the 1st LNA (currently set to 0)
	
	
        # ANTENNA SENSORS & PHYSICAL TEMPERATURES (in K)
	
        E_T_W, W_T_W = (temp_data[i,38]+273.15), (temp_data[i,26]+273.15) # W sensor on horn
	
        E_T_X, W_T_X = (temp_data[i,37]+273.15), (temp_data[i,25]+273.15) # X sensor on throat
	
        E_T_Y, W_T_Y = (temp_data[i,36]+273.15), (temp_data[i,24]+273.15) # Y sensor on polariser
	
        E_T_con1, W_T_con1 = (temp_data[i,35]+273.15), (temp_data[i,23]+273.15) # connector 1 on polariser
	
	
        # CABLE SENSORS, DISTANCES (from connector 1) & PHYSICAL TEMPERATURES (in K)
	
        E_d1, W_d1 = 0,0 #sensor position distances in cm
        E_d2, W_d2 = 32,40
        E_d3, W_d3 = 90,96
        E_d4, W_d4 = 148,153
        E_d5, W_d5 = 191,218
        E_d6, W_d6 = 249,261
        E_d7, W_d7 = 319,313
        E_d8, W_d8 = 368,369
        E_d9, W_d9 = 404,404
	
        E_T_alpha, W_T_alpha = (temp_data[i,34]+273.15), (temp_data[i,22]+273.15) 
	
        E_T_beta, W_T_beta = (temp_data[i,33]+273.15), (temp_data[i,21]+273.15) 
	
        E_T_gamma, W_T_gamma = (temp_data[i,32]+273.15), (temp_data[i,20]+273.15) 
	
        E_T_delta, W_T_delta = (temp_data[i,31]+273.15), (temp_data[i,19]+273.15) 
	
        E_T_epsilon, W_T_epsilon = (temp_data[i,30]+273.15), (temp_data[i,18]+273.15) 
	
        E_T_zeta, W_T_zeta = (temp_data[i,29]+273.15), (temp_data[i,17]+273.15) 
	
        E_T_eta, W_T_eta = (temp_data[i,28]+273.15), (temp_data[i,16]+273.15) 
	
        E_T_con2, W_T_con2 = (temp_data[i,27]+273.15), (temp_data[i,15]+273.15) 
	
	
        # SENSORS & PHYSICAL TEMPERATURES INSIDE RECEIVER BOX (in K)
	
        E_T_NSMA, W_T_NSMA = (temp_data[i,14]+273.15), (temp_data[i,13]+273.15) # sensor on output port of 1st magic-T 
        E_T_T, W_T_T = E_T_NSMA, W_T_NSMA
	
        E_T_RXin, W_T_RXin = (temp_data[i,1]+273.15), (temp_data[i,3]+273.15) # sensor on 1st LNA
	
	
       ############################################################################################
        #–----------------------------------------------------------------------------------------
	
        # CALCULATIONS
	
        #–----------------------------------------------------------------------------------------
        ############################################################################################
	
        # ANTENNA ZONE: TEMPERATURES
	
        E_T_zT, W_T_zT = (E_T_W + E_T_X)/2, (W_T_W + W_T_X)/2 # Zone T (can be ignored with current assumption of no loss in 	horn)
	
        E_T_U, W_T_U = (E_T_X + E_T_Y)/2, (W_T_X + W_T_Y)/2 # Zone U (can be ignored with current assumption of no loss in horn)
	
        E_T_V, W_T_V = (E_T_Y + E_T_con1)/2, (W_T_Y + W_T_con1)/2  # the mean temperature of the polariser= effectively Tphys-ant
	
	
        # CABLE ZONES: FRACTION OF TOTAL CABLE LENGTH
	
        E_f_A, W_f_A = (E_d2-E_d1)/E_d9, (W_d2-W_d1)/W_d9 # zone A
        E_f_B, W_f_B = (E_d3-E_d2)/E_d9, (W_d3-W_d2)/W_d9 # zone B
        E_f_C, W_f_C = (E_d4-E_d3)/E_d9, (W_d4-W_d3)/W_d9 # zone C
        E_f_D, W_f_D = (E_d5-E_d4)/E_d9, (W_d5-W_d4)/W_d9 # zone D
        E_f_E, W_f_E = (E_d6-E_d5)/E_d9, (W_d6-W_d5)/W_d9 # zone E
        E_f_F, W_f_F = (E_d7-E_d6)/E_d9, (W_d7-W_d6)/W_d9 # zone F
        E_f_G, W_f_G = (E_d8-E_d7)/E_d9, (W_d8-W_d7)/W_d9 # zone G
        E_f_H, W_f_H = (E_d9-E_d8)/E_d9, (W_d9-W_d8)/W_d9 # zone H
        #these should sum to 1.000000 maybe write a checksum?
	
	
        # CABLE ZONES: TEMPERATURES (in K)
	
        E_T_A, W_T_A = (E_T_con1 + E_T_alpha)/2, (W_T_con1 + W_T_alpha)/2 # ZONE A
	
        E_T_B, W_T_B = (E_T_alpha + E_T_beta)/2, (W_T_alpha + W_T_beta)/2 # ZONE B
	
        E_T_C, W_T_C = (E_T_beta + E_T_gamma)/2, (W_T_beta + W_T_gamma)/2 # ZONE C
	
        E_T_D, W_T_D = (E_T_gamma + E_T_delta)/2, (W_T_gamma + W_T_delta)/2 # ZONE D
	
        E_T_E, W_T_E = (E_T_delta + E_T_epsilon)/2, (W_T_delta + W_T_epsilon)/2 # ZONE E
	
        E_T_F, W_T_F = (E_T_epsilon + E_T_zeta)/2, (W_T_epsilon + W_T_zeta)/2 # ZONE F
	
        E_T_G, W_T_G = (E_T_zeta + E_T_eta)/2, (W_T_zeta + W_T_eta)/2 # ZONE G
	
        E_T_H, W_T_H = (E_T_eta + E_T_con2)/2, (W_T_eta + W_T_con2)/2 # ZONE H
	
	
	
        # SMA CABLE: MEAN TEMPERATURE
	
        E_T_SMA, W_T_SMA = (E_T_NSMA + E_T_RXin)/2, (W_T_NSMA + W_T_RXin)/2
	
	
        # ANTENNA: TRANSMISSION FACTOR CORRECTED FOR TEMPERATURE DEPENDENCE
	
        E_aSTAR_ANT, W_aSTAR_ANT = E_a_ant * (E_lambda_ant**(297 - E_T_V)), W_a_ant * (W_lambda_ant**(297 - W_T_V))
	
        # CONNECTORS: TRANSMISSION FACTORS CORRECTED FOR TEMPERATURE DEPENDENCE
	
        E_aSTAR_con1, W_aSTAR_con1 = E_a_con1 * (E_lambda_con **(297 - E_T_con1)) , W_a_con1 * (W_lambda_con **(297 - W_T_con1))
	
        E_aSTAR_con2, W_aSTAR_con2 = E_a_con2 * (E_lambda_con **(297 - E_T_con2)) , W_a_con2 * (W_lambda_con **(297 - W_T_con2))
	
        # 4m CABLE ZONES: TRANSMISSION FACTORS CORRECTED FOR TEMPERATURE DEPENDENCE
	
        E_aSTAR_cabA, W_aSTAR_cabA = (E_a_4m * (E_lambda_4m ** (297 - E_T_A)))**E_f_A, (W_a_4m * (W_lambda_4m ** (297 - 	W_T_A)))**W_f_A  # ZONE A
	
        E_aSTAR_cabB, W_aSTAR_cabB = (E_a_4m * (E_lambda_4m ** (297 - E_T_B)))**E_f_B, (W_a_4m * (W_lambda_4m ** (297 - 	W_T_B)))**W_f_B  # ZONE B
	
        E_aSTAR_cabC, W_aSTAR_cabC = (E_a_4m * (E_lambda_4m ** (297 - E_T_C)))**E_f_C, (W_a_4m * (W_lambda_4m ** (297 - 	W_T_C)))**W_f_C  # ZONE C
	
        E_aSTAR_cabD, W_aSTAR_cabD = (E_a_4m * (E_lambda_4m ** (297 - E_T_D)))**E_f_D, (W_a_4m * (W_lambda_4m ** (297 - 	W_T_D)))**W_f_D  # ZONE D
	
        E_aSTAR_cabE, W_aSTAR_cabE = (E_a_4m * (E_lambda_4m ** (297 - E_T_E)))**E_f_E, (W_a_4m * (W_lambda_4m ** (297 - 	W_T_E)))**W_f_E  # ZONE E
	
        E_aSTAR_cabF, W_aSTAR_cabF = (E_a_4m * (E_lambda_4m ** (297 - E_T_F)))**E_f_F, (W_a_4m * (W_lambda_4m ** (297 - 	W_T_F)))**W_f_F  # ZONE F
	
        E_aSTAR_cabG, W_aSTAR_cabG = (E_a_4m * (E_lambda_4m ** (297 - E_T_G)))**E_f_G, (W_a_4m * (W_lambda_4m ** (297 - 	W_T_G)))**W_f_G  # ZONE G
	
        E_aSTAR_cabH, W_aSTAR_cabH = (E_a_4m * (E_lambda_4m ** (297 - E_T_H)))**E_f_H, (W_a_4m * (W_lambda_4m ** (297 - 	W_T_H)))**W_f_H  # ZONE H
	
        # potential checksum, totals of these should equal a_4m (for uniform 297 K)
	
	
        # RECIEVER BOX: TRANSMISSION FACTORS CORRECTED FOR TEMPERATURE DEPENDENCE
	
        E_aSTAR_T, W_aSTAR_T = E_a_T * (E_lambda_T ** (297 - E_T_T)), W_a_T * (W_lambda_T ** (297 - W_T_T)) # 1st magic-T
	
        E_aSTAR_NSMA, W_aSTAR_NSMA = E_a_NSMA * (E_lambda_con ** (297 - E_T_T)), W_a_NSMA * (W_lambda_con ** (297 - W_T_T)) # 	NSMA connector on output port of 1st magic-T
	
        E_aSTAR_SMA, W_aSTAR_SMA = E_a_SMA * (E_lambda_4m ** (297 - E_T_SMA)), W_a_SMA * (W_lambda_4m ** (297 - W_T_SMA)) # 1m 	SMA cable input into 1st LNA
	
        # RADIOMETRIC TEMPERATURE (in K) AT INPUT TO 4m CABLE (T_ant_out = T_in)
	
        E_T_ant_out, W_T_ant_out = (1 - E_r_ant) * ((E_aSTAR_ANT * E_T_ant) + (E_T_V * (1 - E_aSTAR_ANT))), (1 - W_r_ant) * 	((W_aSTAR_ANT * W_T_ant) + (W_T_V * (1 - W_aSTAR_ANT))) # where E_T_V is taken to be E_T_phys_ant etc
	 
        E_T_in, W_T_in = E_T_ant_out, W_T_ant_out
	
        # RADIOMETRIC TEMPERATURES ALONG THE CABLE AND INTO THE 1st MAGIC-T
	
        E_T1, W_T1 = (E_T_in * (1 - E_r_con1) * E_aSTAR_con1) + ((1 - E_aSTAR_con1) * E_T_con1), (W_T_in * (1 - W_r_con1) * 	W_aSTAR_con1) + ((1 - W_aSTAR_con1) * W_T_con1)  # transmitted signal through connector1 + thermal emission from connector1 
        E_T2, W_T2 = (E_aSTAR_cabA * E_T1) + ((1 - E_aSTAR_cabA) * E_T_A), (W_aSTAR_cabA * W_T1) + ((1 - W_aSTAR_cabA) * W_T_A) 	# radiometric temperature at start of zone B
        E_T3, W_T3 = (E_aSTAR_cabB * E_T2) + ((1 - E_aSTAR_cabB) * E_T_B), (W_aSTAR_cabB * W_T2) + ((1 - W_aSTAR_cabB) * W_T_B) 	# radiometric temperature at start of zone C
        E_T4, W_T4 = (E_aSTAR_cabC * E_T3) + ((1 - E_aSTAR_cabC) * E_T_C), (W_aSTAR_cabC * W_T3) + ((1 - W_aSTAR_cabC) * W_T_C) 	# radiometric temperature at start of zone D
        E_T5, W_T5 = (E_aSTAR_cabD * E_T4) + ((1 - E_aSTAR_cabD) * E_T_D), (W_aSTAR_cabD * W_T4) + ((1 - W_aSTAR_cabD) * W_T_D) 	# radiometric temperature at start of zone E
        E_T6, W_T6 = (E_aSTAR_cabE * E_T5) + ((1 - E_aSTAR_cabE) * E_T_E), (W_aSTAR_cabE * W_T5) + ((1 - W_aSTAR_cabE) * W_T_E) 	# radiometric temperature at start of zone F
        E_T7, W_T7 = (E_aSTAR_cabF * E_T6) + ((1 - E_aSTAR_cabF) * E_T_F), (W_aSTAR_cabF * W_T6) + ((1 - W_aSTAR_cabF) * W_T_F) 	# radiometric temperature at start of zone G
        E_T8, W_T8 = (E_aSTAR_cabG * E_T7) + ((1 - E_aSTAR_cabG) * E_T_G), (W_aSTAR_cabG * W_T7) + ((1 - W_aSTAR_cabG) * W_T_G) 	# radiometric temperature at start of zone H
        E_T9, W_T9 = (E_aSTAR_cabH * E_T8) + ((1 - E_aSTAR_cabH) * E_T_H), (W_aSTAR_cabH * W_T8) + ((1 - W_aSTAR_cabH) * W_T_H) 	# radiometric temperature at input to connector 2
        E_T_Tin, W_T_Tin = (E_T9 * (1 - E_r_con2) * E_aSTAR_con2) + ((1 - E_aSTAR_con2) * E_T_con2), (W_T9 * (1 - W_r_con2) * 	W_aSTAR_con2) + ((1 - W_aSTAR_con2) * W_T_con2) # radiometric input temperature to the 1st magic-T
	
	
        # RADIOMETRIC TEMPERATURES INTO THE 1st LNA
	
        E_T_Tout, W_T_Tout = (E_T_Tin * (1 - E_r_T) * E_aSTAR_T) + ((1 - E_aSTAR_T) * E_T_T), (W_T_Tin * (1 - W_r_T) * 	W_aSTAR_T) + ((1 - W_aSTAR_T) * W_T_T) # radiometric temperature at output of 1st magic-T
	
        E_T_TNSMA, W_T_TNSMA = (E_T_Tout * (1 - E_r_NSMA) * E_aSTAR_NSMA) + ((1 - E_aSTAR_NSMA) *  E_T_NSMA), (W_T_Tout * (1 - 	W_r_NSMA) * W_aSTAR_NSMA) + ((1 - W_aSTAR_NSMA) * W_T_NSMA) # through NSMA  connector on 1st magic-T
	
        E_T_SMA_out, W_T_SMA_out = (E_T_TNSMA * (1 - E_r_SMA) * E_aSTAR_SMA) + ((1 - E_aSTAR_SMA) * E_T_SMA), (W_T_TNSMA * (1 - 	W_r_SMA) * W_aSTAR_SMA) + ((1 - W_aSTAR_SMA) * W_T_SMA)  # radiometric temperature at end of 1-m SMA cable
	
        E_T_RX_in, W_T_RX_in = (1-E_r_LNA) * E_T_SMA_out , (1-W_r_LNA) * W_T_SMA_out  # radiometric temperature input to 1st LNA
	
	
        # SIGNAL TRANSMISSION FACTOR THROUGH PASSIVES 
	
        E_a1, W_a1 = (1 -E_r_ant) * E_aSTAR_ANT, (1 -W_r_ant) * W_aSTAR_ANT # antenna
	
        E_a2, W_a2 = ((1 - E_r_con1) * E_aSTAR_con1) * E_aSTAR_cabA * E_aSTAR_cabB * E_aSTAR_cabC * E_aSTAR_cabD * E_aSTAR_cabE 	* E_aSTAR_cabF * E_aSTAR_cabG * E_aSTAR_cabH * ((1 - E_r_con2) * E_aSTAR_con2) , ((1 - W_r_con1) * W_aSTAR_con1) * 	W_aSTAR_cabA * W_aSTAR_cabB * W_aSTAR_cabC * W_aSTAR_cabD * W_aSTAR_cabE * W_aSTAR_cabF * W_aSTAR_cabG * W_aSTAR_cabH * ((1 	- W_r_con2) * W_aSTAR_con2)  # 4m cable + cons
	
        E_a3, W_a3 = (1 - E_r_T) * E_aSTAR_T * (1 - E_r_NSMA) * E_aSTAR_NSMA * (1 - E_r_SMA) * E_aSTAR_SMA * (1 - E_r_LNA), (1 - 	W_r_T) * W_aSTAR_T * (1 - W_r_NSMA) * W_aSTAR_NSMA * (1 - W_r_SMA) * W_aSTAR_SMA * (1 - W_r_LNA) # receiver box
	
        E_a_total, W_a_total = E_a1 * E_a2 * E_a3, W_a1 * W_a2 * W_a3 #totals
	
	
        # THERMAL NOISE (in K) ADDED BY PASSIVES (on top of Tant)
	  
        E_AN, W_AN = E_T_in - E_T_ant, W_T_in - W_T_ant  # antenna, T_ant_out - T_ant
	
        E_4mN, W_4mN = E_T_Tin - E_T_in, W_T_Tin - W_T_in # 4m cable + cons
	
        E_RxN, W_RxN = E_T_RX_in - E_T_Tin, W_T_RX_in - W_T_Tin # receiver box
	
        E_TN_total, W_TN_total = E_AN + E_4mN + E_RxN, W_AN + W_4mN + W_RxN # totals
	

        #SAVE INTO TIME SERIES

        series_E_a1.append((1-E_a1))
        series_W_a1.append((1-W_a1)) # * - 100 to get negative percentage
        series_E_a2.append((1-E_a2))
        series_W_a2.append((1-W_a2))
        series_E_a3.append((1-E_a3))
        series_W_a3.append((1-W_a3))
        series_E_a_total.append((1-E_a_total))
        series_W_a_total.append((1-W_a_total))

        series_E_AN.append(E_AN)
        series_W_AN.append(W_AN)
        series_E_4nM.append(E_4mN)
        series_W_4nM.append(W_4mN)
        series_E_RxN.append(E_RxN)
        series_W_RxN.append(W_RxN)
        series_E_TN_total.append(E_TN_total)
        series_W_TN_total.append(W_TN_total)

    #END OF LOOP
    #/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\

    series_E_a1s=np.array(series_E_a1)
    series_W_a1s=np.array(series_W_a1)
    series_E_a2s=np.array(series_E_a2)
    series_W_a2s=np.array(series_W_a2)
    series_E_a3s=np.array(series_E_a3)
    series_W_a3s=np.array(series_W_a3)
    series_E_a_totals=np.array(series_E_a_total)
    series_W_a_totals=np.array(series_W_a_total)

    series_E_ANs=np.array(series_E_AN)
    series_W_ANs=np.array(series_W_AN)
    series_E_4nMs=np.array(series_E_4nM)
    series_W_4nMs=np.array(series_W_4nM)
    series_E_RxNs=np.array(series_E_RxN)
    series_W_RxNs=np.array(series_W_RxN)
    series_E_TN_totals=np.array(series_E_TN_total)
    series_W_TN_totals=np.array(series_W_TN_total)

    trans_table = np.column_stack((series_E_a1s, series_E_a2, series_E_a3, series_W_a1s, series_W_a2, series_W_a3))
	
    therm_table = np.column_stack((series_E_ANs, series_E_4nMs, series_E_RxNs, series_W_ANs, series_W_4nMs, series_W_RxNs))

    trans_tots = np.column_stack((series_W_a_totals, series_E_a_totals))
    therm_tots = np.column_stack((series_W_TN_totals, series_E_TN_totals))

    return trans_tots, therm_tots, trans_table, therm_table


#----------------------------------------------------------------------------------------------------------------------------


def run_passives():

    a1p1 = np.load (DATA_PATH+'/temp/a1p1_power.npy') #W  L
    a1p2 = np.load (DATA_PATH+'/temp/a1p2_power.npy') #E  L
    a2p1 = np.load (DATA_PATH+'/temp/a2p1_power.npy') #E  R
    a2p2 = np.load (DATA_PATH+'/temp/a2p2_power.npy') #W  R

    a1p1,a1p2,a2p1,a2p2 = corrections(a1p1,a1p2,a2p1,a2p2)

    print('Running Passives model on Left RPG inputs')
    L_trans, L_therm, L_transT, L_thermT = passive_factors(a1p1, a1p2)
    print('Running Passives model on Right RPG inputs')
    R_trans, R_therm, R_transT, R_thermT = passive_factors(a2p2, a2p1)

    temp_data = np.load(DATA_PATH+'/temp/one_wire.npy', allow_pickle=True) 


    passives_totals = np.column_stack((temp_data[:,0],L_trans[:,0],L_trans[:,1],R_trans[:,1],R_trans[:,0],L_therm[:,0],L_therm[:,1],R_therm[:,1],R_therm[:,0]))
# 0 time, 1 p11 trans factor, 2 p12 transfactor, 3 p21 transfactor, 4 p22 transfactor, 5 p11 addtherm, 6 p12 addtherm, 7 p21 addtherm, 8 p22 addtherm   #WE

    passives_table = np.column_stack((temp_data[:,0],  L_transT[:,3],L_transT[:,4],L_transT[:,5],  L_transT[:,0],L_transT[:,1],L_transT[:,2],   
                                                   R_transT[:,3],R_transT[:,4],R_transT[:,5],  R_transT[:,0],R_transT[:,1],R_transT[:,2], 
                                                   L_thermT[:,3],L_thermT[:,4],L_thermT[:,5],  L_thermT[:,0],L_thermT[:,1],L_thermT[:,2],   
                                                   R_thermT[:,3],R_thermT[:,4],R_thermT[:,5],  R_thermT[:,0],R_thermT[:,1],R_thermT[:,2]   ))

# 0 time, 1 p11 antenna,   2 p11 cabscons, 3  p11 receiver,  4  p12 antenna,  5  p12 cabscons,  6 p12 receiver    ----trans
#         7 p22 antenna,   8 p22 cabscons, 9  p22 receiver,  10 p21 antenna,  11 p21 cabscons,  12 p21 receiver   -----trans
#         13 p11 antenna, 14 p11 cabscons, 15 p11 receiver,  16 p12 antenna,  17 p12 cabscons,  18 p12 receiver   -----therm
#         19 p22 antenna, 20 p22 cabscons, 21 p22 receiver,  22 p21 antenna,  23 p21 cabscons,  24 p21 receiver   -----therm

    np.save(DATA_PATH+'/temp/passives_totals.npy',passives_totals)
    np.save(DATA_PATH+'/temp/passives.npy',passives_table)

#----------------------------------------------------------------------------------------------------------

try:
    run_passives()

except:
    print('Unable to apply passives model.')



os.system('chmod -R -f 0777 /mirror/scratch/pblack || true')









