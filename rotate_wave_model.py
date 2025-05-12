#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Function to rotate the Storm and find wave parameters in a desired position within the
#Tropical Cyclone
#_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_
# Reference:
# Grossmann-Matheson et al., 2025
# The spatial distribution of ocean wave parameters in tropical cyclones 
# Ocean Eng, 317, 120091
# DOI: https://doi.org/10.1016/j.oceaneng.2024.120091
#_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_
#Input parameters:
#HS = Hs spatial distribution 'meters'[n,m]
#TP = Tp spatial distribution 'seconds'[n,m]
#U and V components spatial distribution [n,m]
#XX = grid position axis x 'km' [n]
#YY = grid position axis y 'km' [m]
#Lat = Latitude of Storm center (degrees)[-90 +90]
#Lon = Longitude of Storm center (degrees)[-180 +180]
#lat_point = latitude of desired point (degrees)[-90 +90]
#lon_point = longitude of desired point (degrees)[-180 +180]
#rotationAngle = angle to rotate the storm, negative value is anticlockwise
#from North [-180 +180], it doesn't matter the hemisphere. (get this value
#prior from two consecutive track positions)

# Output parameters:
# Hs_rotated = Hs for required position in the TC wavefield (ex: buoy
# position)
# Tp_rotated = Tp for required position in the TC wavefield
# U_rotated = U component for required position in the TC wavefield
# V_rotated = V component for required position in the TC wavefield

import numpy as np
from scipy.ndimage import rotate
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt

def rotate_get_HsTpdir(HS, TP, U, V, XX, YY, Lat, Lon, lat_point, lon_point, rotationAngle):
    # Constants
    km_per_degree = 111.137  # km per degree

    # Distance between desired point and Storm eye
    dt_lat = (lat_point - Lat) * km_per_degree
    dt_lon = (lon_point - Lon) * km_per_degree
    dist_eye = np.sqrt(dt_lat**2 + dt_lon**2)  # distance in km

    # Rotate the Storm (using bilinear interpolation)
    hs_rotated = rotate(HS, rotationAngle, reshape=False, order=1, mode='nearest')  # bilinear interpolation
    tp_rotated = rotate(TP, rotationAngle, reshape=False, order=1, mode='nearest')

    # Convert u,v to degrees, rotate the image, then convert back to u,v
    dir_temp = np.degrees(np.arctan2(-U, -V))  # Convert to direction
    dir_rotated = rotate(dir_temp, rotationAngle, reshape=False, order=1, mode='nearest')
    dir_rotated[dir_rotated == 0] = np.nan  # clean up borders

    # Convert rotated direction back to u, v components (meteorological convention)
    u_rotated = -(np.sin(np.radians(dir_rotated + rotationAngle)))
    v_rotated = -(np.cos(np.radians(dir_rotated + rotationAngle)))

    # Define the lat, long grid to plot (for interpolation)
    DLONG = (XX / km_per_degree) / np.cos(np.radians(Lat))
    DLAT = YY / km_per_degree
    ALONG = DLONG + Lon
    ALAT = DLAT + Lat

    # Interpolate values at the required point (lon_point, lat_point)
    interp_Hs = interp2d(ALONG, ALAT, hs_rotated, kind='linear')
    interp_Tp = interp2d(ALONG, ALAT, tp_rotated, kind='linear')
    interp_U = interp2d(ALONG, ALAT, u_rotated, kind='linear')
    interp_V = interp2d(ALONG, ALAT, v_rotated, kind='linear')

    Hs_rotated = interp_Hs(lon_point, lat_point)
    Tp_rotated = interp_Tp(lon_point, lat_point)
    U_rotated = interp_U(lon_point, lat_point)
    V_rotated = interp_V(lon_point, lat_point)

    # Calculate wave direction
    Dir_rotated = np.degrees(np.arctan2(-U_rotated, -V_rotated))
    Dir_rotated[Dir_rotated < 0] += 360  # Ensure it's within 0-360 degrees

    # Clean up zero values
    Hs_rotated = np.nan if Hs_rotated == 0 else Hs_rotated
    Tp_rotated = np.nan if Tp_rotated == 0 else Tp_rotated

    # Return the results
    return Hs_rotated, Tp_rotated, U_rotated, V_rotated

