# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:29:10 2022
Program that will open and observe data from LBASS, RPG and the Anritsu Power
Meters

DATA_PATH = '/mirror/scratch/pblack'



@author: Jordan McKenzie Norris
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mpl_dates
from astropy.time import Time
from astropy.time import TimeDelta
from scipy.fft import fft, fftfreq, rfft, rfftfreq
import csv
from datetime import datetime, timedelta
from tkinter import *
from tkinter import filedialog

print('\033[1;33m')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Read in data
def UTCDateTime(range_example, duration_actual, rpgdatafilepath):
    """
    Code developed by Phillip Black. This code generates arrays of sample times
    by using the astropy time backage for use in plotting data.

    Parameters
    ----------
    range_example : Array
        Array that may end up being used to test the length of the array
        corresponding to the time
    duration_actual : Float
        Float of the observation period that is being looked at in Hours

    Returns
    -------
    sample_to_datetime : Array
        Array of strings specifing the time in UTC that each line in the plotted
        arrays corresponds to.
    """ 
    inputs = np.load(rpgdatafilepath+'inputs.npy')
    
    mid_obs = Time(inputs[0],format='isot', scale='utc', precision=0, out_subfmt='date')
    
    obs_date = Time(str(mid_obs),format='iso', scale='utc', precision=0, out_subfmt='date_hms')
    
    #bin length assumed to be one mnute - make that adjustable, PIPs
    sample_to_datetime = []
    i=0
    
    for i in range (0,np.size(range_example[:,0])): #this means sample time is days long
            sb = obs_date + TimeDelta(float(range_example[i,0]), format='sec') #changed run_start_date to obs_date
            sb = sb.tt.datetime
            #r = sb.strftime("%Y-%m-%d %H:%M:%S")
            sample_to_datetime.append(sb)
    
           
    
    return sample_to_datetime

def RPGFileReader(rpgdatafilepath, inputhorn, duration_actual):
    """
    Reads in the RPG data and outputs two arrays

    Returns
    -------
    rpg_power : array
        Array of Powers which corresponds to the array of times
    rpg_datetime : array
        Array of times which corresponds to the array of powers

    """
    try:
        a1p1b = np.load(rpgdatafilepath+inputhorn+'_binned.npy')
        #max_index = 580
        max_index = np.where(a1p1b[0,:] == np.max(a1p1b[0,3:]))
        rpg_power = a1p1b[:,max_index]
        
        rpg_power = np.resize(rpg_power,(np.size(rpg_power),))
        rpg_datetime = np.array(UTCDateTime(a1p1b, duration_actual, rpgdatafilepath))

    except:
        pass
    
    
    return rpg_power, rpg_datetime




def dBmTOmW(x):
    return 10**(x/10)

def convertHHMMSStoFullDate(calendardate, timestring):
    t = datetime.strptime(timestring, "%H:%M:%S")
    dt = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
    timeformat = calendardate + dt
    return timeformat

def PowerMeterFileReader(anritsufilepath):
    """
    Reads in the Antritsu power meter data and then converts the powers into
    linear mW as well as adding proper time formats to each measurement so that
    they may be used in conjunction with the RPG data. The date is taken from
    RPG data assuming both measurements were initialised on the same day
    Returns
    -------
    mw_powers : array
        A numpy array of power data in mW
    pm_time_array : array
        A numpy array of datetime data corresponding to the powers.

    """
    dbm_powers = np.genfromtxt(anritsufilepath ,delimiter=',',
                          skip_header=7, usecols=1)
    mw_powers = dBmTOmW(dbm_powers) #power array in milliWatts
    del dbm_powers
    
    times = []
    with open(anritsufilepath) as file:
        data = csv.reader(file)
        for row in data:
            times.append(str(row[0]))
    times = np.array(times[7:]) #gathers times as aray of strings
    
    #split anritsufilepath string with /
    filename = anritsufilepath.split('/')[-1]
    filename = filename.split('_')
    
    # calendardate = obsheader[0,7]
    # calendardate = datetime.strptime(calendardate, "isot")
    # print(calendardate)
    obs_date = datetime(int(filename[1]), int(filename[2]), int(filename[3]))
    
    pm_time_array = []
    firstline = True
    
    for i in range(0, np.size(times)):
        if firstline == True:
            timeformat = convertHHMMSStoFullDate(obs_date, times[i])
            pm_time_array.append(timeformat)
            firstline = False
        else:
            if convertHHMMSStoFullDate(obs_date, times[i]).hour < convertHHMMSStoFullDate(obs_date, times[i-1]).hour:
                obs_date = obs_date + timedelta(days=1)
                timeformat = convertHHMMSStoFullDate(obs_date, times[i])
                pm_time_array.append(timeformat)
            else:
                timeformat = convertHHMMSStoFullDate(obs_date, times[i])
                pm_time_array.append(timeformat)
    pm_time_array = np.array(pm_time_array)
    #pm_time_arrray = pm_time_array
            
    return mw_powers, pm_time_array


# RPG_POWERS, RPG_DATETIME = RPGFileReader()
# POWERMETER_POWERS_mW, POWERMETER_DATETIME = PowerMeterFileReader()

def PowerNormaliser(power_array):
    return power_array / np.average(power_array)

# NORMAL_RPG_POWERS = PowerNormaliser(RPG_POWERS)
# NORMAL_PM_POWERS = PowerNormaliser(POWERMETER_POWERS_mW)

#------------------------------------------------------------------------------
#Average the arrays by different time intervals

def arrayAverager(power_array, times_array, averagingpoints):
    """
    Averages values over a number of averaging points that would correspond to
    a period of time. For RPG this is a point per minute. For PowerMeter is 2
    points per minute
    
    Parameters
    ----------
    power_array : array
        Linear array of powers to be averaged
    times_array : array
        Corresponding times for each power
    averagingpoints : integer
        Number of points of data that will be averaged over.

    Returns
    -------
    avg_power_array : array
        Array of averaged data
    avg_time_array : TYPE
        Array of times corresponding to the start of each averaging period.

    """
    upperIndex = int(np.size(power_array) / averagingpoints) 
    
    avg_power_array = []
    avg_time_array = []
    
    for i in range(0, upperIndex):
        
        average = np.mean(power_array[(i*averagingpoints):(i*averagingpoints)+averagingpoints])
        
        time = times_array[i*averagingpoints + int(averagingpoints/2)]
        
        avg_power_array.append(average)
        avg_time_array.append(time)
    avg_power_array = np.array(avg_power_array)
    avg_time_array = np.array(avg_time_array)
    
    return avg_power_array, avg_time_array

#UNNORMALISED HOUR AVERAGE DATA

# HRAVG_RPGPOWERS, HRAVG_RPGTIMES = arrayAverager(RPG_POWERS, RPG_DATETIME, 60)

# HRAVG_POWERMETERPOWERS, HRAVG_POWERMETERTIMES = arrayAverager(POWERMETER_POWERS_mW,
#                                                                       POWERMETER_DATETIME, 120)
# #NORMALISED HOUR AVERAGE DATA

# NORMAL_HR_RPG, NORMAL_HR_RPG_TIMES = arrayAverager(NORMAL_RPG_POWERS, RPG_DATETIME, 60)
# NORMAL_HR_PM, NORMAL_HR_PM_TIMES  = arrayAverager(NORMAL_PM_POWERS, POWERMETER_DATETIME, 120)

#------------------------------------------------------------------------------
#PLOTS OF DATA

def PowerTimePlotter(rpg_power_array, rpg_times_array, rpg_avgpower, rpg_avgtime,
                     pm_power_array, pm_times_array, pm_avgpower, pm_avgtime,
                     isNormalised, serialnumber,toptitle):
    
    plt.figure(1)
    date_format = mpl_dates.DateFormatter('%d  %H:%M')
    plt.subplot(211)
    plt.plot_date(rpg_times_array, rpg_power_array, c='b')
    plt.plot_date(rpg_avgtime, rpg_avgpower, linestyle='solid', c='r')
    if isNormalised == True:
        plt.ylabel('Normalised Power')
    else:
        plt.ylabel('Power [arb]')
    plt.xticks(rotation=45)
    #plt.yscale('log')
    plt.title(toptitle)
    plt.gca().xaxis.set_major_formatter(date_format)
    
    
        
    
    plt.subplot(212)
    plt.plot_date(pm_times_array, pm_power_array, c='b')
    plt.plot_date(pm_avgtime, pm_avgpower, linestyle='solid', c='r')
    if isNormalised == True:
        plt.ylabel('Normalised Power')
    else:
        plt.ylabel('Power [mW]')
    plt.xticks(rotation=45)
    plt.title('Power Meter '+serialnumber+' Data')
    
    plt.tight_layout()
    plt.xticks(rotation=45)
    #plt.yscale('log')
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.show()
    return

def PowerTimePlotterLonePM(pm_power_array, pm_times_array, pm_avgpower, pm_avgtime,
                           isNormalised, serialnumber):
    
    date_format = mpl_dates.DateFormatter('%d  %H:%M')
    
    plt.plot_date(pm_times_array, pm_power_array, c='b')
    plt.plot_date(pm_avgtime, pm_avgpower, linestyle='solid', c='r')
    if isNormalised == True:
        plt.ylabel('Normalised Power')
    else:
        plt.ylabel('Power [mW]')
    plt.xticks(rotation=45)
    plt.title('Power Meter Data' +'  #'+serialnumber)
    plt.gca().xaxis.set_major_formatter(date_format)
    
    
    plt.show()
    return

def PowerTimePlotterLoneLBASS(pm_power_array, pm_times_array, pm_avgpower, pm_avgtime,
                           isNormalised):
    
    date_format = mpl_dates.DateFormatter('%d  %H:%M')
    
    plt.plot_date(pm_times_array, pm_power_array, c='b')
    plt.plot_date(pm_avgtime, pm_avgpower, linestyle='solid', c='r')
    if isNormalised == True:
        plt.ylabel('Normalised Power')
    else:
        plt.ylabel('Power [arb]')
    plt.xticks(rotation=45)
    plt.title('LBASS CW Data')
    plt.gca().xaxis.set_major_formatter(date_format)
    
    
    plt.show()
    return


# PowerTimePlotter(RPG_POWERS, RPG_DATETIME, HRAVG_RPGPOWERS, HRAVG_RPGTIMES,
#                   POWERMETER_POWERS_mW, POWERMETER_DATETIME, HRAVG_POWERMETERPOWERS, HRAVG_POWERMETERTIMES
#                   ,False)

# PowerTimePlotter(NORMAL_RPG_POWERS , RPG_DATETIME, NORMAL_HR_RPG, NORMAL_HR_RPG_TIMES,
#                   NORMAL_PM_POWERS, POWERMETER_DATETIME, NORMAL_HR_PM , NORMAL_HR_PM_TIMES
#                   ,True)


def getSamplingTime(time_array):
    """
    Takes differences in times between starts and ends of arrays to measure the
    average time between measurements for use in Fourier Transforming.
    Parameters
    ----------
    time_array : array
        Numpy array of timedate objects specifying time at each observation.

    Returns
    -------
    timestep : float
        Average time between measurements

    """
    starttime = time_array[0]
    latertime = time_array[-1]
    
    diff = latertime - starttime
    diff = diff.total_seconds()
    
    timestep = diff / np.size(time_array)
    
    return timestep

def PowerSpectrum(power_array, times_array, titlestring):
    numbersamples = np.size(power_array)
    sampletime = getSamplingTime(times_array)
    sig_fft = rfft(power_array)
    freq = rfftfreq(numbersamples, sampletime)
    
    plt.plot(freq, np.abs(sig_fft))
    plt.title(titlestring)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Fourier Amplitude')
    plt.xlabel('Frequency [Hz]')
    plt.tight_layout()
    plt.show()
    return
    
    
# PowerSpectrum(RPG_POWERS, RPG_DATETIME, 'RPG Power Spectrum')
# PowerSpectrum(POWERMETER_POWERS_mW, POWERMETER_DATETIME, 'PowerMeter Power Spectrum')

def DualPowerSpectrum(rpg_power_array, rpg_times_array, pm_power_array, pm_times_array):
    plt.figure(1)
    #RPG subplot
    plt.subplot(211)
    numbersamples1 = np.size(rpg_power_array)
    sampletime1 = getSamplingTime(rpg_times_array)
    sig_fft1 = rfft(rpg_power_array)
    freq1 = rfftfreq(numbersamples1, sampletime1)
    
    plt.plot(freq1, np.abs(sig_fft1))
    plt.title('RPG Power Spectrum')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Fourier Amplitude')
    
    # Power Meter Subplot
    plt.subplot(212)
    numbersamples2 = np.size(pm_power_array)
    sampletime2 = getSamplingTime(pm_times_array)
    sig_fft2 = rfft(pm_power_array)
    freq2 = rfftfreq(numbersamples2, sampletime2)
    
    plt.plot(freq2, np.abs(sig_fft2))
    plt.title('PowerMeter - Power Spectrum')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Fourier Amplitude')
    
    plt.xlabel('Frequency [Hz]')
    
    plt.tight_layout()
    plt.show()
    return

#DualPowerSpectrum(RPG_POWERS, RPG_DATETIME, POWERMETER_POWERS_mW, POWERMETER_DATETIME)

#------------------------------------------------------------------------------
# RMS Analyser and Plotter

def RMSDAnalyser(power_array, time_data_array):
    arraysize = np.size(power_array)
    i = 1
    fullindexarray = []
    
    while arraysize/i >=2:
        j=i
        i = i*2
        arraydivision = int(arraysize/j)
        indexarray = []
        for k in range(0, j):
            inf = k*arraydivision
            ins = inf + arraydivision -1
            line = [inf, ins]
            indexarray.append(line)
        fullindexarray.append(indexarray)
        #creates array of start and stop indices of mean locations
    
    
    rms_values = []
    interval_times = []
    
    for i in range(0, np.size(fullindexarray)):
        diff_square_array = []
        time_array = []
        
        for line in fullindexarray[i]:
            
            mean_value = np.average(power_array[line[0]:line[1]])
            
            laterindex = int(line[1]) #list index
            
            earlyindex = int(line[0])
            
            intervaltime = time_data_array[laterindex] - time_data_array[earlyindex]
            
            time_of_average = (intervaltime).total_seconds()
            
            time_array.append(time_of_average)
            
            for l in range(line[0], line[1]):
                diff_square = (power_array[l] - mean_value)**2
                diff_square_array.append(diff_square)
        
        average_time = np.average(np.array(time_array))
        interval_times.append(average_time)
        
        sum_diff_square = np.sum(np.array(diff_square_array))
        number_of_points = fullindexarray[i][-1][-1] + 1
        rmsd = np.sqrt(sum_diff_square / number_of_points)
        rms_values.append(rmsd)
        
    rms_values = np.array(rms_values)
    interval_times = np.array(interval_times)
    
    return rms_values, interval_times

# rms_values, interval_times = RMSDAnalyser(POWERMETER_POWERS_mW, POWERMETER_DATETIME)
#------------------------------------------------------------------------------
def RMSDAverageAnalyser(power_array, time_data_array):
    arraysize = np.size(power_array)
    i = 1
    fullindexarray = []
    
    while arraysize/i >=2:
        j=i
        i = i*2
        arraydivision = int(arraysize/j)
        indexarray = []
        for k in range(0, j):
            inf = k*arraydivision
            ins = inf + arraydivision -1
            line = [inf, ins]
            indexarray.append(line)
        fullindexarray.append(indexarray)
        #creates array of start and stop indices of mean locations
    
    
    rms_values = []
    interval_times = []
    fullaverage = np.average(power_array)
    
    for i in range(0, np.size(fullindexarray)):
        diff_square_array = []
        time_array = []
        
        for line in fullindexarray[i]:
            
            mean_value = np.average(power_array[line[0]:line[1]])
            
            diff_square_value = (mean_value - fullaverage)**2
            diff_square_array.append(diff_square_value)
            laterindex = int(line[1]) #list index
            
            earlyindex = int(line[0])
            
            intervaltime = time_data_array[laterindex] - time_data_array[earlyindex]
            
            time_of_average = (intervaltime).total_seconds()
            
            time_array.append(time_of_average)
            
        
        average_time = np.average(np.array(time_array))
        interval_times.append(average_time)
        
        sum_diff_square = np.sum(np.array(diff_square_array))
        number_of_points = np.size(diff_square_array)
        rmsd = np.sqrt(sum_diff_square / number_of_points)
        rms_values.append(rmsd)
        
    rms_values = np.array(rms_values)
    interval_times = np.array(interval_times)
    
    return rms_values, interval_times

def RMSDPlotter(power_array, time_array, title, units):
    rms_values, interval_times = RMSDAnalyser(power_array, time_array)
    plt.scatter(interval_times, rms_values)
    plt.title(title)
    plt.xlabel('Averaging Time [s]')
    plt.ylabel('RMSD ['+units+']')
    plt.xscale('log')
    plt.grid()
    plt.tight_layout()
    plt.show()
    return

def RMSDAveragePlotter(power_array, time_array, title, units):
    rms_values, interval_times = RMSDAverageAnalyser(power_array, time_array)
    plt.scatter(interval_times, rms_values)
    plt.title(title)
    plt.xlabel('Averaging Time [s]')
    plt.ylabel('RMSD of Averaged Intervals from full mean ['+units+']')
    plt.xscale('log')
    plt.tight_layout()
    plt.show()
    return

# RMSDPlotter(POWERMETER_POWERS_mW, POWERMETER_DATETIME, 'Power Meter', 'mW')
# RMSDPlotter(NORMAL_PM_POWERS , POWERMETER_DATETIME, 'Power Meter', 'fractional')

# RMSDPlotter(RPG_POWERS, RPG_DATETIME, 'RPG', 'arb.')
# RMSDPlotter(NORMAL_RPG_POWERS ,RPG_DATETIME, 'RPG', 'fractional')

def ObservationPlots(rpgdatafilepath, anritsufilepath, duration_actual,
                     inputhorn, serialnumber):
    
    rpg_powers_raw, rpg_datetime = RPGFileReader(rpgdatafilepath, inputhorn, duration_actual)
    pm_powers_mw, pm_datetime = PowerMeterFileReader(anritsufilepath)
    pm_sampling_time = getSamplingTime(pm_datetime)
    normal_rpg_powers = PowerNormaliser(rpg_powers_raw)
    normal_pm_powers = PowerNormaliser(pm_powers_mw)
    
    hr_rpg_average, hr_rpg_times = arrayAverager(rpg_powers_raw, rpg_datetime, 60)
    hr_pm_average, hr_pm_times = arrayAverager(pm_powers_mw, pm_datetime, round(3600/pm_sampling_time))
    nm_hr_rpg_average, hr_rpg_times = arrayAverager(normal_rpg_powers, rpg_datetime, 60)
    nm_hr_pm_average, hr_pm_times = arrayAverager(normal_pm_powers, pm_datetime, round(3600/pm_sampling_time))
    
    print('')
    print('Display Plots')
    print('-------------')
    print('1 :Power Time Series Comparison')
    print('')
    print('2 :Power Spectrums')
    print('')
    print('3 :Root Mean Square Deviations with Averaging Time')
    print('')
    print('')
    print('0 :Return')
    choice = int(input('Enter number corresponding to plot choice: '))
    
    if choice == 1:
        choice = input('Display in Normalised Units? Y/N: ')
        if choice.lower() == 'y' or choice.lower() == 'yes':
            PowerTimePlotter(normal_rpg_powers , rpg_datetime, nm_hr_rpg_average, hr_rpg_times,
                              normal_pm_powers, pm_datetime, nm_hr_pm_average , hr_pm_times
                              ,True, serialnumber, 'RPG Data')
            ObservationPlots(rpgdatafilepath, anritsufilepath, duration_actual,
                                 inputhorn, serialnumber)
        else:
            PowerTimePlotter(rpg_powers_raw, rpg_datetime, hr_rpg_average, hr_rpg_times,
                              pm_powers_mw, pm_datetime, hr_pm_average, hr_pm_times
                              ,False, serialnumber, 'RPG Data')
            ObservationPlots(rpgdatafilepath, anritsufilepath, duration_actual,
                                 inputhorn, serialnumber)
    elif choice == 2:
        print('')
        print('1 :View RPG CW Power Spectrum')
        print('')
        print('2 :View Power Meter CW Power Spectrum')
        print('')
        print('3 :View Both Power Spectrums')
        print('')
        print('0 :Return')
        choice = int(input('Enter number corresponding to plot choice: '))
        
        if choice == 1:
            PowerSpectrum(rpg_powers_raw, rpg_datetime, 'RPG Power Spectrum')
            ObservationPlots(rpgdatafilepath, anritsufilepath, duration_actual,
                                 inputhorn, serialnumber)
        elif choice == 2:
            PowerSpectrum(pm_powers_mw, pm_datetime, 'PowerMeter '+ serialnumber +' Power Spectrum')
            ObservationPlots(rpgdatafilepath, anritsufilepath, duration_actual,
                                 inputhorn, serialnumber)
        else:
            DualPowerSpectrum(rpg_powers_raw, rpg_datetime, 
                              pm_powers_mw, pm_datetime)
            ObservationPlots(rpgdatafilepath, anritsufilepath, duration_actual,
                                 inputhorn, serialnumber)
    
    elif choice == 3:
        print('View variations of Root Mean Square Deviations from Mean with')
        print('averaging time')
        print('')
        print('1 :Power Meter ')
        print('')
        print('2 :CW input into RPG')
        print('')
        print('')
        print('0 :Return')
        choice = int(input('Enter number corresponding to plot choice: '))
        if choice == 1:
            choice = input('Display in Normalised Units? Y/N: ')
            if choice.lower() == 'y' or choice.lower() == 'yes':
                RMSDPlotter(normal_pm_powers , pm_datetime, 'Power Meter  '+serialnumber, 'fractional')
                ObservationPlots(rpgdatafilepath, anritsufilepath, duration_actual,
                                     inputhorn, serialnumber)
            else:
                RMSDPlotter(pm_powers_mw, pm_datetime, 'Power Meter  '+serialnumber, 'mW')
                ObservationPlots(rpgdatafilepath, anritsufilepath, duration_actual,
                                     inputhorn, serialnumber)
        elif choice == 2:
            choice = input('Display in Normalised Units? Y/N: ')
            if choice.lower() == 'y' or choice.lower() == 'yes':
                RMSDPlotter(normal_rpg_powers ,rpg_datetime, 'RPG', 'fractional')
                ObservationPlots(rpgdatafilepath, anritsufilepath, duration_actual,
                                     inputhorn, serialnumber)
            else:
                RMSDPlotter(rpg_powers_raw, rpg_datetime, 'RPG', 'arb.')
                ObservationPlots(rpgdatafilepath, anritsufilepath, duration_actual,
                                     inputhorn, serialnumber)
    else:
        mainmenu()
      
    
    return

def LBASSPowerMeterComparison(lbassfilepath, anritsufilepath, duration_actual,
                     inputhorn, serialnumber):
    
    lbass_cw_power_raw, lbass_datetime_raw =  RPGFileReader(lbassfilepath, inputhorn, duration_actual)
    pm_powers_mw, pm_datetime = PowerMeterFileReader(anritsufilepath) # second sampling
    
    pm_sampling_time = getSamplingTime(pm_datetime)
    
    normal_pm_powers = PowerNormaliser(pm_powers_mw)
    normal_lbass_powers = PowerNormaliser(lbass_cw_power_raw)
    
    
    pm_powers_mw_minute, pm_datetime_minute = arrayAverager(pm_powers_mw, pm_datetime, round(60/pm_sampling_time))
    nm_pm_powers_mw_minute, nm_pw_datetime_minute = arrayAverager(normal_pm_powers, pm_datetime, round(60/pm_sampling_time))
    
    
    hr_lbass_average, hr_lbass_times = arrayAverager(lbass_cw_power_raw, lbass_datetime_raw, 60)
    hr_pm_average, hr_pm_times = arrayAverager(pm_powers_mw_minute, pm_datetime_minute, 60)
    
    
    nm_hr_lbass_average, hr_lbass_times = arrayAverager(normal_lbass_powers, lbass_datetime_raw, 60)
    nm_hr_pm_average, hr_pm_times = arrayAverager(nm_pm_powers_mw_minute, nm_pw_datetime_minute, 60)
    
    print('')
    print('Display Plots')
    print('-------------')
    print('1 :Power Time Series Comparison')
    print('')
    print('2 :Power Spectrums')
    print('')
    print('3 :Root Mean Square Deviations with Averaging Time')
    print('')
    print('')
    print('0 :Return')
    choice = int(input('Enter number corresponding to plot choice: '))
    
    if choice == 1:
        choice = input('Display in Normalised Units? Y/N: ')
        
        if choice.lower() == 'y' or choice.lower() == 'yes':
            PowerTimePlotter(normal_lbass_powers , lbass_datetime_raw, nm_hr_lbass_average, hr_lbass_times,
                              nm_pm_powers_mw_minute, nm_pw_datetime_minute, nm_hr_pm_average , hr_pm_times
                              ,True, serialnumber, 'LBASS CW')
            LBASSPowerMeterComparison(lbassfilepath, anritsufilepath, duration_actual,
                                 inputhorn, serialnumber)
        
        elif choice.lower() == 'n' or choice.lower() == 'no':
            PowerTimePlotter(lbass_cw_power_raw, lbass_datetime_raw, hr_lbass_average, hr_lbass_times,
                              pm_powers_mw_minute, pm_datetime_minute, hr_pm_average, hr_pm_times
                              ,False, serialnumber, 'LBASS CW')
            LBASSPowerMeterComparison(lbassfilepath, anritsufilepath, duration_actual,
                                 inputhorn, serialnumber)
        else:
            print('Enter Valid input')
            LBASSPowerMeterComparison(lbassfilepath, anritsufilepath, duration_actual,
                                 inputhorn, serialnumber)
    
    
    elif choice == 2:
        print('')
        print('1 :View LBASS CW Power Spectrum')
        print('')
        print('2 :View Power Meter CW Power Spectrum')
        print('')
        print('3 :View Both Power Spectrums')
        print('')
        print('0 :Return')
        
        choice = int(input('Enter number corresponding to plot choice: '))
        
        if choice == 1:
            PowerSpectrum(lbass_cw_power_raw, lbass_datetime_raw, 'LBASS CW Power Spectrum')
            LBASSPowerMeterComparison(lbassfilepath, anritsufilepath, duration_actual,
                                 inputhorn, serialnumber)
        elif choice == 2:
            PowerSpectrum(pm_powers_mw, pm_datetime, 'PowerMeter '+ serialnumber +' Power Spectrum')
            LBASSPowerMeterComparison(lbassfilepath, anritsufilepath, duration_actual,
                                 inputhorn, serialnumber)
        elif choice == 3:
            DualPowerSpectrum(lbass_cw_power_raw, lbass_datetime_raw, 
                              pm_powers_mw, pm_datetime)
            LBASSPowerMeterComparison(lbassfilepath, anritsufilepath, duration_actual,
                                 inputhorn, serialnumber)
        else:
            LBASSPowerMeterComparison(lbassfilepath, anritsufilepath, duration_actual,
                                 inputhorn, serialnumber)
            
    elif choice == 3:
        print('View variations of Root Mean Square Deviations from Mean with')
        print('averaging time')
        print('')
        print('1 :Power Meter ')
        print('')
        print('2 :L-BASS CW input in horn')
        print('')
        print('')
        print('0 :Return')
        
        choice = int(input('Enter number corresponding to plot choice: '))
        
        if choice == 1:
            choice = input('Display in Normalised Units? Y/N: ')
            
            if choice.lower() == 'y' or choice.lower() == 'yes':
                RMSDPlotter(nm_pm_powers_mw_minute, nm_pw_datetime_minute, 'Power Meter  '+serialnumber, 'fractional')
                LBASSPowerMeterComparison(lbassfilepath, anritsufilepath, duration_actual,
                                     inputhorn, serialnumber)
            
            elif choice.lower() == 'n' or choice.lower() == 'no':
                RMSDPlotter(pm_powers_mw_minute, pm_datetime_minute, 'Power Meter  '+serialnumber, 'mW')
                LBASSPowerMeterComparison(lbassfilepath, anritsufilepath, duration_actual,
                                     inputhorn, serialnumber)
            else:
                RMSDPlotter(pm_powers_mw_minute, pm_datetime_minute, 'Power Meter  '+serialnumber, 'mW')
                LBASSPowerMeterComparison(lbassfilepath, anritsufilepath, duration_actual,
                                     inputhorn, serialnumber)
        elif choice == 2:
            choice = input('Display in Normalised Units? Y/N: ')
            
            if choice.lower() == 'y' or choice.lower() == 'yes':
                RMSDPlotter(normal_lbass_powers ,lbass_datetime_raw, 'LBASS CW 1.426GHz', 'fractional')
                LBASSPowerMeterComparison(lbassfilepath, anritsufilepath, duration_actual,
                                     inputhorn, serialnumber)
            elif choice.lower() == 'n' or choice.lower() == 'no':
                RMSDPlotter(lbass_cw_power_raw, lbass_datetime_raw, 'LBASS CW 1.426GHz', 'arb.')
                LBASSPowerMeterComparison(lbassfilepath, anritsufilepath, duration_actual,
                                     inputhorn, serialnumber)
            else:
                RMSDPlotter(lbass_cw_power_raw, lbass_datetime_raw, 'LBASS CW 1.426GHz', 'arb.')
                LBASSPowerMeterComparison(lbassfilepath, anritsufilepath, duration_actual,
                                     inputhorn, serialnumber)
            
    else:
        mainmenu()
    
    return

def LonePowerMeterFullSpeed(anritsufilepath, serialnumber):
    pm_powers_mw, pm_datetime  = PowerMeterFileReader(anritsufilepath)
    normal_pm_powers = PowerNormaliser(pm_powers_mw)
    
    pm_sampling_time = getSamplingTime(pm_datetime)
    minute_pm_average, minute_pm_times = arrayAverager(pm_powers_mw, pm_datetime, round(60/pm_sampling_time))
    nm_minute_pm_average, nm_minute_pm_times = arrayAverager(normal_pm_powers, pm_datetime, round(60/pm_sampling_time))
    
    hr_pm_average, hr_pm_times = arrayAverager(minute_pm_average, minute_pm_times, 60)
    nm_hr_pm_average, nm_hr_pm_times = arrayAverager(nm_minute_pm_average, nm_minute_pm_times, 60)
    #Data with minute and hour averageds
    
    print('')
    print('Display Plots')
    print('-------------')
    print('1 :Power Time Plot: Minute Averages with Hour Averages')
    print('')
    print('2 :Power Time Plot: All Samples with Hour Averages')
    print('')
    print('3 :Power Spectrum')
    print('')
    print('4 :Root Mean Square Deviations with Averaging Time: Full Data Set')
    print('')
    print('5 :Root Mean Square Deviations with Averaging Time: Minute Averaged Data Set')
    print('')
    print('')
    print('0 :Return')
    choice = int(input('Enter number corresponding to plot choice: '))
    
    if choice == 1:
        choice = input('Display in Normalised Units? Y/N: ')
        
        if choice.lower() == 'y' or choice.lower() == 'yes':
            PowerTimePlotterLonePM(nm_minute_pm_average, nm_minute_pm_times, nm_hr_pm_average, nm_hr_pm_times,
                                       True, serialnumber)
            LonePowerMeterFullSpeed(anritsufilepath, serialnumber)
        else:
            PowerTimePlotterLonePM(minute_pm_average, minute_pm_times, hr_pm_average, hr_pm_times,
                                       False, serialnumber)
            LonePowerMeterFullSpeed(anritsufilepath, serialnumber)
    
    elif choice == 2:
        choice = input('Display in Normalised Units? Y/N: ')
        
        if choice.lower() == 'y' or choice.lower() == 'yes':
            PowerTimePlotterLonePM(normal_pm_powers, pm_datetime, nm_hr_pm_average, nm_hr_pm_times,
                                       True, serialnumber)
            LonePowerMeterFullSpeed(anritsufilepath, serialnumber)
        else:
            PowerTimePlotterLonePM(pm_powers_mw, pm_datetime, hr_pm_average, hr_pm_times,
                                       False, serialnumber)
            LonePowerMeterFullSpeed(anritsufilepath, serialnumber)
    
    elif choice == 3:
        PowerSpectrum(pm_powers_mw, pm_datetime, 'PowerMeter '+ serialnumber +' Power Spectrum')
        LonePowerMeterFullSpeed(anritsufilepath, serialnumber)
        
    elif choice == 4:
        choice = input('Display in Normalised Units? Y/N: ')
        if choice.lower() == 'y' or choice.lower() == 'yes':
            RMSDPlotter(normal_pm_powers , pm_datetime, 'Power Meter  '+serialnumber+' Full Set', 'fractional')
            LonePowerMeterFullSpeed(anritsufilepath, serialnumber)
        else:
            RMSDPlotter(pm_powers_mw, pm_datetime, 'Power Meter  '+serialnumber+' Full Set', 'mW')
            LonePowerMeterFullSpeed(anritsufilepath, serialnumber)
    
    elif choice == 5:
        choice = input('Display in Normalised Units? Y/N: ')
        if choice.lower() == 'y' or choice.lower() == 'yes':
            RMSDPlotter(nm_minute_pm_average , nm_minute_pm_times, 'Power Meter  '+serialnumber+' MinuteAveraged', 'fractional')
            LonePowerMeterFullSpeed(anritsufilepath, serialnumber)
        else:
            RMSDPlotter(minute_pm_average, minute_pm_times, 'Power Meter  '+serialnumber+' MinuteAveraged', 'mW')
            LonePowerMeterFullSpeed(anritsufilepath, serialnumber)
    
    else:
        mainmenu()
    
def LoneRPGMeasurementAnalysis(lbassfilepath, duration_actual,
                     inputhorn):
    lbass_cw_power_raw, lbass_datetime_raw =  RPGFileReader(lbassfilepath, inputhorn, duration_actual)
    normal_lbass_powers = PowerNormaliser(lbass_cw_power_raw)
    
    hr_lbass_average, hr_lbass_times = arrayAverager(lbass_cw_power_raw, lbass_datetime_raw, 60)
    nm_hr_lbass_average, nm_hr_lbass_times = arrayAverager(normal_lbass_powers, lbass_datetime_raw, 60)
    
    print('')
    print('Display Plots')
    print('-------------')
    print('1 :Power Time Plot: Minute Averages with Hour Averages')
    print('')
    print('2 :Power Spectrum: CW Channel')
    print('')
    print('3 :Root Mean Square Deviations with Averaging Time: Minute Averaged Data Set')
    print('')
    print('')
    print('0 :Return')
    choice = int(input('Enter number corresponding to plot choice: '))
    
    if choice == 1:
        choice = input('Display in Normalised Units? Y/N: ')
        
        if choice.lower() == 'y' or choice.lower() == 'yes':
            PowerTimePlotterLoneLBASS(normal_lbass_powers, lbass_datetime_raw, 
                                      nm_hr_lbass_average, nm_hr_lbass_times,
                                       True)
            LoneRPGMeasurementAnalysis(lbassfilepath, duration_actual,
                                 inputhorn)
        else:
            PowerTimePlotterLoneLBASS(lbass_cw_power_raw, lbass_datetime_raw, 
                                      hr_lbass_average, hr_lbass_times,
                                       False)
            LoneRPGMeasurementAnalysis(lbassfilepath, duration_actual,
                                 inputhorn)
    elif choice == 2:
        PowerSpectrum(lbass_cw_power_raw, lbass_datetime_raw,
                      'L-BASS CW Channel Power Spectrum')
        LoneRPGMeasurementAnalysis(lbassfilepath, duration_actual,
                             inputhorn)
    elif choice == 3:
        choice = input('Display in Normalised Units? Y/N: ')
        if choice.lower() == 'y' or choice.lower() == 'yes':
            RMSDPlotter(normal_lbass_powers ,lbass_datetime_raw,
                        'LBASS CW Channel', 'fractional')
            LoneRPGMeasurementAnalysis(lbassfilepath, duration_actual,
                                 inputhorn)
        else:
            RMSDPlotter(lbass_cw_power_raw, lbass_datetime_raw, 
                        'LBASS CW Channel', 'arb.')
            LoneRPGMeasurementAnalysis(lbassfilepath, duration_actual,
                                 inputhorn)
    else:
        mainmenu()
    
    
    
    

#------------------------------------------------------------------------------
def RetrievePowerMeterFileName():
    print('Select Anritsu Power Meter .csv File for Analysis')
    #n = input()
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    filepath = filedialog.askopenfilename()
    
    print('Selected Filepath is:'+filepath)
    return filepath

def RetrieveLBASSObsDataFileName():
    print('Select Folder containing LBASS data for Analysis')
    #n = input()
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    filepath = filedialog.askdirectory()
    filepath = filepath+'/'
    print('Selected Filepath is:'+filepath)
    return filepath

def inputhornretriever():
    print('')
    print('1 : West Horn P(l,$\pi$)')
    print('2 : West Horn P(r,0)')
    print('3 : East Horn P(l,0)')
    print('4 : East Horn P(r,$\pi$)')
    print('')
    print('Enter number corresponding to where the CW signal was injected.')
    print('If injected into horn select either 1 or 3.')
    choice = int(input(': '))
    if choice == 1:
        return 'a1p1'
    elif choice == 2:
        return 'a2p2'
    elif choice == 3:
        return 'a1p2'
    elif choice == 4:
        return 'a2p1'
    else:
        mainmenu()
    

#Menus
#anritsufilepath = RetrievePowerMeterFileName() #------------------------------
def mainmenu():
    print('--------------- L-BASS POWERMETER AND CW ANALYSER -----------------')
    print('Verson 1.00 - Developed By Jordan Norris')
    print('')
    print('')
    print('--------------------------------------------------------------------')
    print('Select Data Analysis Option:')
    print('1: CW injected into Horn of L-BASS and Power Meter')
    print('')
    print('2: Lab Test of CW Signal Split into RPG and Power Meter')
    print('')
    print('3: Analysis of Power Meter at full speed sampling')
    print('')
    print('4: Analysis of CW injected into L-BASS or RPG')
    print('')
    print('0: Close Program')
    choice = int(input('Enter Number Corresponding To Analysis option: '))
    if choice == 1:
        CWinLBASSMenu()
    elif choice == 2:
        CWLabTestMenu()
    elif choice == 3:
        OnlyPowerMeterMenu()
    elif choice == 4:
        OnlyCWinLBASSMenu()
    
    elif choice == 0:
        pass
    else:
        mainmenu()
  
def OnlyCWinLBASSMenu():
    print('--------------- CW on Horn and PowerMeter ------------------------')
    print('')
    print('------------------------------------------------------------------')
    print('')
    print('')
    lbassfilepath = RetrieveLBASSObsDataFileName()
    duration_actual = np.load(lbassfilepath+'/duration_actual.npy')
    inputhorn = inputhornretriever()
    
    LoneRPGMeasurementAnalysis(lbassfilepath, duration_actual,
                         inputhorn)
    
    #print('Choose Plots to View:')
    return    
  
def CWinLBASSMenu():
    print('--------------- CW on Horn and PowerMeter ------------------------')
    print('')
    print('------------------------------------------------------------------')
    print('')
    print('')
    anritsufilepath = RetrievePowerMeterFileName()
    print('')
    print('')
    lbassfilepath = RetrieveLBASSObsDataFileName()
    duration_actual = np.load(lbassfilepath+'/duration_actual.npy')
    inputhorn = inputhornretriever()
    serialnumber = input('Enter PowerMeter Serial Number or identifier: ')
    LBASSPowerMeterComparison(lbassfilepath, anritsufilepath, duration_actual,
                         inputhorn, serialnumber)
    
    #print('Choose Plots to View:')
    return

def CWLabTestMenu():
    print('--------------- CW into RPG and PowerMeter -----------------------')
    print('')
    print('------------------------------------------------------------------')
    print('')
    print('')
    anritsufilepath = RetrievePowerMeterFileName()
    print('')
    print('')
    rpgdatafilepath = RetrieveLBASSObsDataFileName()
    duration_actual = np.load(rpgdatafilepath+'/duration_actual.npy')
    inputhorn = inputhornretriever()
    serialnumber = input('Enter PowerMeter Serial Number or identifier: ')
    ObservationPlots(rpgdatafilepath, anritsufilepath, duration_actual,
                         inputhorn, serialnumber)
    return
    
def OnlyPowerMeterMenu():
    print('--------------- PowerMeter Analysis ------------------------')
    print('')
    print('------------------------------------------------------------')
    print('')
    print('')
    anritsufilepath = RetrievePowerMeterFileName()
    print('')
    print('')
    serialnumber = input('Enter PowerMeter Serial Number or identifier: ')
    LonePowerMeterFullSpeed(anritsufilepath, serialnumber)
    return

#----------------------MAINCODE------------------------------------------------

try:
    mainmenu()
except:
    print()
    print('\033[1;31m CW Program Error \033[1;33m')
    print()
    mainmenu()

os.system('chmod -R -f 0777 /local5/scratch/pblack || true')

print ('\033[1;32m')
