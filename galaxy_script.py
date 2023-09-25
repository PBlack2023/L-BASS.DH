DATA_PATH = '/mirror/scratch/pblack'


def UTCDateTime(range_example, duration_actual):
    """
    Based on code developed by Phillip Black. This code generates arrays of 
    sample times by using the astropy time backage for use in plotting data.

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
    obsheader = np.load(DATA_PATH+'/temp/obshdr.npy')
    
    mid_obs = Time(obsheader[0,7],format='isot', scale='utc', precision=0, 
                   out_subfmt='date')
    
    obs_date = Time(str(mid_obs),format='iso', scale='utc', precision=0, 
                    out_subfmt='date_hms')
    
    #bin length assumed to be one mnute - make that adjustable, PIPs
    sample_to_datetime = []
    i=0
    
    for i in range (0,np.size(range_example[:,0])): #this means sample time is days long
            sb = obs_date + TimeDelta(float(range_example[i,0]), format='sec') #changed run_start_date to obs_date
            r = sb.strftime("%Y-%m-%d %H:%M:%S")
            sample_to_datetime.append(r)
            
    return sample_to_datetime



obshdr = np.load(DATA_PATH+'/temp/obshdr.npy')
file_table = np.load(DATA_PATH+'/temp/file1.npy', allow_pickle=True)

duration_actual = np.load(DATA_PATH+'/temp/duration_actual.npy')



#obshdr[2] is the observing mode - there are a number of options that eventually need to be accomodated.
if obshdr[0,2] == 'NCP-SCANNING':
    scan_elevation = np.load (DATA_PATH+'/temp/a1p2_binned.npy')
    horn = 'East'
elif obshdr[0,2] == 'SCANNING-NCP':
    scan_elevation = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
    horn = 'West'
else:
    scan_elevation = np.load (DATA_PATH+'/temp/a1p2_binned.npy')
    horn = 'East'
NCP_elevation = float(126.76)*u.deg


MJD = Time(file_table[1,12], format='mjd', scale='utc', precision=4)
temp_times = UTCDateTime(scan_elevation, duration_actual)

obloc = EarthLocation(lat=float(obshdr[0,4])*u.deg, lon=float(obshdr[0,3])*u.deg, height=float(obshdr[0,5])*u.m)

TIME=[]
HORN=[]
G_COORDS=[]
I_COORDS=[]
l =[]
b=[]
ra=[]
dec=[]
aa =[]
ab = []

for i in range (0,np.size(temp_times)):
    sample_to_time = UTCDateTime(scan_elevation, duration_actual)
    obtime = sample_to_time
    #LBASS records more than 90° elevation. To convert for astopy needs corrected azimuth & elevation.
    if float(scan_elevation[i,2]) > 90.0: 
        rotated_frame = float(obshdr[0,6]) #- 180
        obaz = float(rotated_frame)*u.deg
        adjusted_angle = 180 - float(scan_elevation[i,2])
        obalt = float(adjusted_angle)*u.deg
    else:
        obalt = float(horn_elevation[i,2])*u.deg
        obaz = float(obshdr[0,6])*u.deg
    pointing = SkyCoord(az=obaz , alt= obalt, obstime=obtime , location= obloc, frame='altaz')
    G_COORDS.append(pointing.galactic)
    I_COORDS.append(pointing.icrs)
    ICRSCoords = pointing.transform_to('icrs')
    ra.append(ICRSCoords.ra.wrap_at('180d').radian)
    dec.append(ICRSCoords.dec.radian)
    GalCoords = pointing.transform_to('galactic')
    l.append(GalCoords.l.wrap_at('180d').radian)
    b.append(GalCoords.b.radian)
    TIME.append(obtime.iso)
    HORN.append(horn)
    fixed = SkyCoord(az=(0*u.deg), alt=((180 - 126.76)*u.deg), obstime=obtime, location= obloc, frame='altaz')
    fixeded = fixed.transform_to('icrs')
    aa.append(fixeded.ra.wrap_at('180d').radian)
    ab.append(fixeded.dec.radian)

coords_table = np.column_stack((np.array(TIME), np.array(HORN), np.array(G_COORDS), np.array(l), np.array(b), np.array(I_COORDS), np.array(ra), np.array(dec), np.array(aa),np.array(ab)))



colour_n = np.size(temp_times)
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

plt.figure()
plt.subplot(projection = "aitoff")
plt.grid(True)
plt.title('Scanning Path of Telescope in Galactic Coordinates')
plt.tight_layout()
for i in range (0,np.size(temp_times)):
    plt.plot(coords_table[i,8], coords_table[i,9], marker='o', color ='k')
    plt.plot(coords_table[i,6], coords_table[i,7], marker='o', color =(R[i],G[i],B[i]))
plt.show()


#==================================================================#


# Galactic and RA Dec plotter - This code has no effect on the CW analysis code
# above
def UTCDateTime(range_example, duration_actual):
    """
    Based on code developed by Phillip Black. This code generates arrays of 
    sample times by using the astropy time backage for use in plotting data.

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
    obsheader = np.load(DATA_PATH+'/temp/obshdr.npy')
    
    mid_obs = Time(obsheader[0,7],format='isot', scale='utc', precision=0, 
                   out_subfmt='date')
    
    obs_date = Time(str(mid_obs),format='iso', scale='utc', precision=0, 
                    out_subfmt='date_hms')
    
    #bin length assumed to be one mnute - make that adjustable, PIPs
    sample_to_datetime = []
    i=0
    
    for i in range (0,np.size(range_example[:,0])): #this means sample time is days long
            sb = obs_date + TimeDelta(float(range_example[i,0]), format='sec') #changed run_start_date to obs_date
            r = sb.strftime("%Y-%m-%d %H:%M:%S")
            sample_to_datetime.append(r)
            
    return sample_to_datetime

def AltAzToGalactic(azimuth, elevation, time):
    """
    Converts Input of time of observation and elevation with known location of
    telescope into Galactic Coordinates. Time in UTC

    """
    AltAzCoord = AltAz(location = LOC, obstime=Time(time), 
                  az = azimuth*u.deg, alt = elevation*u.deg)
    
    AltAzCoord = SkyCoord(AltAzCoord)
    
    #ICRSCoord = AltAzCoord.transform_to('icrs')
    
    GalCoord = AltAzCoord.transform_to('galactic')
    
    return GalCoord

def GalacticPathPlotter(duration_actual):
    """
    Code plots and also generates an array of galactic coordinates in 
    an array of skycoord objects and also as a 2D array of longitue and 
    lattitude which corresponds to each time.

    Parameters
    ----------
    duration_actual : float
        length of duration of data set

    Returns
    -------
    GalCoord : array
        Array of SkyCoord type objects corresponding to where LBass is looking
        at each time.
        
    galCoordArray2D: array
        2D array of longitude and lattidue corresponding to each time.

    """
    a1p1b = np.load(DATA_PATH+'/temp/a1p1_binned.npy')
    UTC_array = UTCDateTime(a1p1b, duration_actual)
    
    AltAzCoord = AltAz(location = LOC, obstime=Time(UTC_array),
                        az = 0*u.deg, alt = (180-a1p1b[:,2])*u.deg)
    AltAzCoord = SkyCoord(AltAzCoord)
    GalCoord = AltAzCoord.transform_to('galactic')

    AltAzCoord2 = AltAz(location = LOC, obstime=Time(UTC_array),
                        az = 0*u.deg, alt = ((176.24-a1p1b[:,2])*u.deg))
    AltAzCoord2 = SkyCoord(AltAzCoord2)
    GalCoord2 = AltAzCoord2.transform_to('galactic')

    galCoordArray2D = np.zeros((0,2))
    for i in range(0, np.size(UTC_array)):
        galCoord = AltAzToGalactic(0,(180-a1p1b[i,2]), UTC_array[i])
        l = galCoord.l.degree #long.
        b = galCoord.b.degree #lat.
        line = np.array([l, b])
        galCoordArray2D = np.vstack((galCoordArray2D,line))
        
    
    plt.figure()
    plt.subplot(projection = "aitoff")
    plt.plot(GalCoord.l.wrap_at('180d').radian, GalCoord.b.radian, marker='o', c='r')
    plt.plot(GalCoord2.l.wrap_at('180d').radian, GalCoord2.b.radian, marker='o', c='grey')
    plt.grid(True)
    plt.title('Scanning Path of Telescope in Galactic Coordinates')
    plt.tight_layout()
    plt.show()
    return GalCoord, galCoordArray2D

GalCoord, galCoordArray2D = GalacticPathPlotter(duration_actual)


#±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±§#



obshdr = np.load(DATA_PATH+'/temp/obshdr.npy')
file_table = np.load(DATA_PATH+'/temp/file1.npy', allow_pickle=True)

duration_actual = np.load(DATA_PATH+'/temp/duration_actual.npy')



#obshdr[2] is the observing mode - there are a number of options that eventually need to be accomodated.
if obshdr[0,2] == 'NCP-SCANNING':
    scan_elevation = np.load (DATA_PATH+'/temp/a1p2_binned.npy')
    horn = 'East'
elif obshdr[0,2] == 'SCANNING-NCP':
    scan_elevation = np.load (DATA_PATH+'/temp/a1p1_binned.npy')
    horn = 'West'
else:
    scan_elevation = np.load (DATA_PATH+'/temp/a1p2_binned.npy')
    horn = 'East'
NCP_elevation = float(126.76)*u.deg


MJD = Time(file_table[1,12], format='mjd', scale='utc', precision=4)
temp_times = UTCDateTime(scan_elevation, duration_actual)

obloc = EarthLocation(lat=float(obshdr[0,4])*u.deg, lon=float(obshdr[0,3])*u.deg, height=float(obshdr[0,5])*u.m)

TIME=[]
HORN=[]
G_COORDS=[]
I_COORDS=[]
l =[]
b=[]
ra=[]
dec=[]
aa =[]
ab = []

for i in range (0,np.size(temp_times)):
    sample_to_time = UTCDateTime(scan_elevation, duration_actual)
    obtime = sample_to_time
    #LBASS records more than 90° elevation. To convert for astopy needs corrected azimuth & elevation.
    if float(scan_elevation[i,2]) > 90.0: 
        rotated_frame = float(obshdr[0,6]) #- 180
        obaz = float(rotated_frame)*u.deg
        adjusted_angle = 180 - float(scan_elevation[i,2])
        obalt = float(adjusted_angle)*u.deg
    else:
        obalt = float(horn_elevation[i,2])*u.deg
        obaz = float(obshdr[0,6])*u.deg
    pointing = SkyCoord(az=obaz , alt= obalt, obstime=Time(sample_to_time[i]), location= obloc, frame='altaz')
    G_COORDS.append(pointing.galactic)
    I_COORDS.append(pointing.icrs)
    ICRSCoords = pointing.transform_to('icrs')
    ra.append(ICRSCoords.ra.wrap_at('180d').radian)
    dec.append(ICRSCoords.dec.radian)
    GalCoords = pointing.transform_to('galactic')
    l.append(GalCoords.l.wrap_at('180d').radian)
    b.append(GalCoords.b.radian)
    TIME.append(obtime)
    HORN.append(horn)
   # fixed = SkyCoord(az=(0*u.deg), alt=((180 - 126.76)*u.deg), obstime=Time(sample_to_time[i]), location= obloc, frame='altaz')
    #fixeded = fixed.transform_to('icrs')
    #aa.append(fixeded.ra.wrap_at('180d').radian)
    #ab.append(fixeded.dec.radian)

coords_table = np.column_stack((np.array(TIME), np.array(HORN), np.array(G_COORDS), np.array(l), np.array(b), np.array(I_COORDS), np.array(ra), np.array(dec))) #, np.array(aa),np.array(ab)



colour_n = np.size(temp_times)
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

plt.figure()
plt.subplot(projection = "aitoff")
plt.grid(True)
plt.title('Scanning Path of Telescope in Galactic Coordinates')
plt.tight_layout()
for i in range (0,np.size(temp_times)):
   # plt.plot(coords_table[i,8], coords_table[i,9], marker='o', color ='k')
    plt.plot(coords_table[i,6], coords_table[i,7], marker='o', color =(R[i],G[i],B[i]))
plt.show()




    


