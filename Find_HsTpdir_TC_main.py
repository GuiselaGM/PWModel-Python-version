#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Code with an example to get the significant wave height,dir and period in 
# a buoy location crossed by a Tropical Cyclone using the PWModel 
#_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_
# Reference:
# Grossmann-Matheson et al., 2025
# The spatial distribution of ocean wave parameters in tropical cyclones 
# Ocean Eng, 317, 120091
# DOI: https://doi.org/10.1016/j.oceaneng.2024.120091
#_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_
#
# Parameters required to run this code: 
# Vmax = 'Maximum sustained wind speed' 'm/s'
# Vfm = 'Storm translation speed' 'm/s'
# R34 = 'Radius of 34 knot winds (mean of all quadrants)' 'meters'
# Rmax = 'Radius of maximum winds' 'meters'
# Lat = Latitude of Storm center 'degrees'
# Lon = Longitude of Storm center 'degrees' 
# lat_point = latitude of desired point/buoy (degrees)[-90 +90]
# lon_point = longitude of desired point/buoy (degrees)[-180 +180]
# rotationAngle = angle to rotate the storm, negative value is anticlockwise
# from North [-180 +180]. Default is zero (storm moving Northward). It doesn't matter the hemisphere,
# get this value from the storm track positions.
#
# OUTPUT: Hs, dir (including components) and Tp for the required point (ex: buoy position)
# Hs_point = Significant wave Height
# Tp_point = Peak wave Period
# U_point = zonal wave dir component
# V_point = meriodional wave dir component
# Dir_point = Peak wave direction coming from (meteorological convention)
# Dir_point_oc = Peak wave direction going to (oceanographic convention)

import numpy as np
import math
import warnings
import pickle

# Define your storm input parameters (example values)
Vmax = [50]        # m/s
Vfm = [5]          # m/s
Rmax = [30e3]      # m
R34 = [300e3]      # m
Lat = [29.4]       # degrees
Lon = [-77.3]      # degrees

# Define point of interest (e.g., buoy location)
lon_point = -78.5
lat_point = 28.9

# Check parameter limits
def check_parameters(Vmax, Vfm, Rmax, R34):
    if (Vmax > 78 or Vfm > 15 or Rmax > 60e3 or R34 > 400e3 or
        Vmax < 17 or Rmax < 15e3 or R34 < 200e3):
        warnings.warn("Parameter(s) over the limits. Default(s) will be used instead.")

# Placeholder functions for PWModel and rotate_get_HsTpdir 
# (You need to replace these with your actual implementations)

def PWModel(Vmax, Vfm, Rmax, R34, Lat):
    # Placeholder implementation
    HS, TP, U, V, XX, YY = np.random.rand(6, 10)  # Replace with actual model implementation
    return HS, TP, U, V, XX, YY

def rotate_get_HsTpdir(HS, TP, U, V, XX, YY, Lat, Lon, lat_point, lon_point, rotationAngle):
    # Placeholder for rotating and extracting Hs, Tp, U, V at buoy location
    Hs_rotated = HS.mean()
    Tp_rotated = TP.mean()
    U_rotated = U.mean()
    V_rotated = V.mean()
    return Hs_rotated, Tp_rotated, U_rotated, V_rotated

def main():
    # Check parameters for storm input
    for k in range(len(Vmax)):
        check_parameters(Vmax[k], Vfm[k], Rmax[k], R34[k])

        # Call your storm wave model (replace with actual model)
        HS, TP, U, V, XX, YY = PWModel(Vmax[k], Vfm[k], Rmax[k], R34[k], Lat[k])

        # Simulate user input for rotation angle (can be hardcoded, adjusted, or passed as a list)
        rotationAngle = 45  # Replace with your desired rotation angle

        # Rotate and extract Hs, Tp, U, V at buoy location
        Hs_rotated, Tp_rotated, U_rotated, V_rotated = rotate_get_HsTpdir(
            HS, TP, U, V, XX, YY, Lat[k], Lon[k], lat_point, lon_point, rotationAngle
        )

        # Collect results
        Hs_rotated_all = []
        Tp_rotated_all = []
        U_rotated_all = []
        V_rotated_all = []
        
        Hs_rotated_all.append(Hs_rotated)
        Tp_rotated_all.append(Tp_rotated)
        U_rotated_all.append(U_rotated)
        V_rotated_all.append(V_rotated)

        # Convert to numpy arrays for further processing
        Hs_point = np.array(Hs_rotated_all)
        Tp_point = np.array(Tp_rotated_all)
        U_point = np.array(U_rotated_all)
        V_point = np.array(V_rotated_all)

        # Calculate wave direction (meteorological)
        Dir_point = np.degrees(np.arctan2(-U_point, -V_point))
        Dir_point[Dir_point < 0] += 360
        Dir_point = np.round(Dir_point)

        # Calculate wave direction (oceanographic)
        Dir_point_oc = np.degrees(np.arctan2(U_point, V_point))
        Dir_point_oc[Dir_point_oc < 0] += 360
        Dir_point_oc = np.round(Dir_point_oc)

        # Save results using pickle
        fname = f"HsTpDir_TC_{Vmax[0]}_{Vfm[0]}_{int(Rmax[0]/1e3)}_{int(R34[0]/1e3)}.pkl"
        with open(fname, 'wb') as f:
            pickle.dump({
                'Hs_point': Hs_point,
                'Tp_point': Tp_point,
                'U_point': U_point,
                'V_point': V_point,
                'Dir_point': Dir_point,
                'Dir_point_oc': Dir_point_oc
            }, f)

if __name__ == "__main__":
    main()

