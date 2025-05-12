#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_
#Function to calculate the fetch using parameterised polynomial equation
#and parameters Hs, peak dir and Tp spatial distribution within a Tropical
#Cyclone
#by Guisela October, 2025
#_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_/(`_
#Input parameters:
#Vmax = 'Maximum sustained wind speed' 'm/s'
#Vfm = 'Storm translation speed' 'm/s'
#R34 = 'Radius of 34 knot winds (mean of all quadrants)' 'meters'
#Rmax = 'Radius of maximum winds' 'meters'
#Lat = Latitude of Storm center 'degrees'
#Output parameters:
#Hs = Hs spatial distribution 'meters'
#Tp = Tp spatial distribution 'seconds'
#U = u component peak direction 
#V = v component peak direction 
#XX = grid position axis x 'km'
#YY = grid position axis y 'km'
#==========================================================================
# Reference:
# Grossmann-Matheson et al, 2025
# The spatial distribution of ocean wave parameters in tropical cyclones 
# Ocean Engineering (submitted)
# DOI: (to update)
# ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> ><>


import numpy as np
import warnings
from scipy.interpolate import interpn
from scipy.io import loadmat
import itertools
import matplotlib.pyplot as plt
import os

# Example inputs (replace with your actual values)
Vmax = np.array([80])  # Can also be scalar
Vfm = np.array([16])
Rmax = np.array([70000])
R34 = np.array([100000])

# Apply limits
Vmax = np.clip(Vmax, 17, 78)
Vfm = np.minimum(Vfm, 15)
Rmax = np.clip(Rmax, 15000, 60000)
R34 = np.clip(R34, 200000, 400000)

# Check if any parameter was out of bounds
if (
    np.any(Vmax > 78) or np.any(Vfm > 15) or np.any(Rmax > 60000) or np.any(R34 > 400000) or
    np.any(Vmax < 17) or np.any(Rmax < 15000) or np.any(R34 < 200000)
):
    warnings.warn("Parameter(s) over the limits. Default(s) will be used instead.")

# Coefficients
a = 0.54
b = -169
c = -1442
d = 0.3
e = 14.3
f = -43
g = 9600
h = 4470
i = 100000
C = 0.1

# Parameterised Fetch Equation
F_P32 = (
    a * Vmax**3 +
    b * Vmax**2 +
    c * Vfm**2 +
    d * Vmax**2 * Vfm +
    e * Vmax * Vfm**2 +
    f * Vmax * Vfm +
    g * Vmax +
    h * Vfm +
    i
) * np.exp(C * Vfm)

# Correction factors
lambda_corr = 0.85 * np.log10(Rmax / 30000) + 1
gamma_corr = 0.65 * np.log10(R34 / 300000) + 1

# Final Fetch
Fetch = F_P32 * lambda_corr * gamma_corr
F = Fetch / 1000  # Fetch in km

# Calculate Hs_max
gr = 9.81
alpha = 0.89
Hs_max = alpha * ((0.0016 * ((gr * F * 1000)**0.5) * Vmax) / gr)

print("Hs_max:", Hs_max)

# ===================================================================================
# Find Hs/Hsmax wavefield diagram(s) to get Hs correspondent to the calculated Hs(max)
# ===================================================================================
# Run combinations
Vmax_runs = np.array([17, 30, 40, 50, 65, 78])
Dp_runs = np.array([10, 30, 50, 70, 110, 150])
Vfm_runs = np.array([0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0])
R34_runs = np.array([200, 300, 400])  # in km
Rmax_runs = np.array([15, 30, 60])  # in km

def get_two_closest(run_values, input_val):
    deltas = np.abs(run_values - input_val)
    exact_matches = np.where(deltas == 0)[0]

    if len(exact_matches) > 0:
        return run_values[exact_matches[0]], run_values[exact_matches[0]]

    idx_sorted = np.argsort(deltas)
    val1 = run_values[idx_sorted[0]]
    val2 = run_values[idx_sorted[1]]

    # adjust if both values are on the same side
    if val1 < input_val and val2 < input_val:
        higher = run_values[run_values > input_val]
        val2 = higher[0] if len(higher) > 0 else val1
    elif val1 > input_val and val2 > input_val:
        lower = run_values[run_values < input_val]
        val2 = lower[-1] if len(lower) > 0 else val1

    return val1, val2

# 1. Determine Vmax/Dp run names
vmax1, vmax2 = get_two_closest(Vmax_runs, Vmax)
dp1 = Dp_runs[np.where(Vmax_runs == vmax1)[0][0]]
dp2 = Dp_runs[np.where(Vmax_runs == vmax2)[0][0]]

# 2. Determine Vfm run names
vfm1, vfm2 = get_two_closest(Vfm_runs, Vfm)

# 3. Determine R34 run names
r34_1, r34_2 = get_two_closest(R34_runs, R34 / 1000)

# 4. Determine Rmax run names
rmax1, rmax2 = get_two_closest(Rmax_runs, Rmax / 1000)

# Outputs
results = {
    "vmax": [vmax1, vmax2],
    "dp": [dp1, dp2],
    "vfm": [vfm1, vfm2],
    "r34": [r34_1, r34_2],
    "rmax": [rmax1, rmax2],
}

print(results)

#import itertools
#from scipy.io import loadmat

# Generate all possible name combinations (16 in total)
for i, (d, v, r3, rm) in enumerate(itertools.product(dp, vfm, r34, rmax), start=1):
    fname = f"wave_{d}_{v}_{r3}_{rm}.mat"
    print(f"Loading file: {fname}")

    try:
        data = loadmat(fname)
        
        # Extract variables from the .mat file
        Z = data['Z']
        Tp = data['Tp']
        upeak = data['upeak']
        vpeak = data['vpeak']
        xrm = data['xrm']
        yrm = data['yrm']

        # Assign to specific variable names if needed (like Z1, T1, etc.)
        globals()[f"Z{i}"] = Z
        globals()[f"T{i}"] = Tp
        globals()[f"U{i}"] = upeak
        globals()[f"V{i}"] = vpeak
        globals()[f"xrm{i}"] = xrm
        globals()[f"yrm{i}"] = yrm

        # Store in results dictionary if you want a structured way to access them
        results.append({
            "name": fname,
            "Z": Z,
            "Tp": Tp,
            "u": upeak,
            "v": vpeak,
            "xrm": xrm,
            "yrm": yrm
        })

    except FileNotFoundError:
        print(f"File not found: {fname}")
    except KeyError as e:
        print(f"Missing variable in {fname}: {e}")

# Preallocate arrays with the same shape as the reference arrays
ZZ = np.zeros_like(Z1)
TT = np.zeros_like(T1)
UU = np.zeros_like(U1)
VV = np.zeros_like(V1)

#-----------------------------------------------------------------------
# Calculate final Z,T,U,V (Hs,Tp,u and v interpolated from combinations)
#-----------------------------------------------------------------------

# Interpolations 
if vmax[0] != vmax[1] and vfm[0] != vfm[1] and rmax[0] != rmax[1] and r34[0] != r34[1]:
    for xx in range(len(xrm1)):
        for yy in range(len(yrm1)):
            z = np.zeros((2, 2, 2, 2))
            t = np.zeros((2, 2, 2, 2))
            u = np.zeros((2, 2, 2, 2))
            v = np.zeros((2, 2, 2, 2))

            # Assign values to z, t, u, v (reshaping according to your 4D matrix structure)
            z[0, 0, 0, 0] = Z1[xx, yy]
            z[1, 0, 0, 0] = Z2[xx, yy]
            z[0, 1, 0, 0] = Z3[xx, yy]
            z[1, 1, 0, 0] = Z4[xx, yy]
            z[0, 0, 1, 0] = Z5[xx, yy]
            z[1, 0, 1, 0] = Z6[xx, yy]
            z[0, 1, 1, 0] = Z7[xx, yy]
            z[1, 1, 1, 0] = Z8[xx, yy]
            z[0, 0, 0, 1] = Z9[xx, yy]
            z[1, 0, 0, 1] = Z10[xx, yy]
            z[0, 1, 0, 1] = Z11[xx, yy]
            z[1, 1, 0, 1] = Z12[xx, yy]
            z[0, 0, 1, 1] = Z13[xx, yy]
            z[1, 0, 1, 1] = Z14[xx, yy]
            z[0, 1, 1, 1] = Z15[xx, yy]
            z[1, 1, 1, 1] = Z16[xx, yy]

            # Interpolate Z using scipy's interpn
            ZZ[xx, yy] = interpn(
                (vmax, vfm, r34, rmax), z,
                (Vmax, Vfm, R34 / 10**3, Rmax / 10**3)
            )

            # Repeat for t, u, v arrays
            t[0, 0, 0, 0] = T1[xx, yy]
            t[1, 0, 0, 0] = T2[xx, yy]
            t[0, 1, 0, 0] = T3[xx, yy]
            t[1, 1, 0, 0] = T4[xx, yy]
            t[0, 0, 1, 0] = T5[xx, yy]
            t[1, 0, 1, 0] = T6[xx, yy]
            t[0, 1, 1, 0] = T7[xx, yy]
            t[1, 1, 1, 0] = T8[xx, yy]
            t[0, 0, 0, 1] = T9[xx, yy]
            t[1, 0, 0, 1] = T10[xx, yy]
            t[0, 1, 0, 1] = T11[xx, yy]
            t[1, 1, 0, 1] = T12[xx, yy]
            t[0, 0, 1, 1] = T13[xx, yy]
            t[1, 0, 1, 1] = T14[xx, yy]
            t[0, 1, 1, 1] = T15[xx, yy]
            t[1, 1, 1, 1] = T16[xx, yy]
            TT[xx, yy] = interpn(
                (vmax, vfm, r34, rmax), t,
                (Vmax, Vfm, R34 / 10**3, Rmax / 10**3)
            )

            # Repeat for u and v arrays
            u[0, 0, 0, 0] = U1[xx, yy]
            u[1, 0, 0, 0] = U2[xx, yy]
            u[0, 1, 0, 0] = U3[xx, yy]
            u[1, 1, 0, 0] = U4[xx, yy]
            u[0, 0, 1, 0] = U5[xx, yy]
            u[1, 0, 1, 0] = U6[xx, yy]
            u[0, 1, 1, 0] = U7[xx, yy]
            u[1, 1, 1, 0] = U8[xx, yy]
            u[0, 0, 0, 1] = U9[xx, yy]
            u[1, 0, 0, 1] = U10[xx, yy]
            u[0, 1, 0, 1] = U11[xx, yy]
            u[1, 1, 0, 1] = U12[xx, yy]
            u[0, 0, 1, 1] = U13[xx, yy]
            u[1, 0, 1, 1] = U14[xx, yy]
            u[0, 1, 1, 1] = U15[xx, yy]
            u[1, 1, 1, 1] = U16[xx, yy]
            UU[xx, yy] = interpn(
                (vmax, vfm, r34, rmax), u,
                (Vmax, Vfm, R34 / 10**3, Rmax / 10**3)
            )

            # Repeat for v array
            v[0, 0, 0, 0] = V1[xx, yy]
            v[1, 0, 0, 0] = V2[xx, yy]
            v[0, 1, 0, 0] = V3[xx, yy]
            v[1, 1, 0, 0] = V4[xx, yy]
            v[0, 0, 1, 0] = V5[xx, yy]
            v[1, 0, 1, 0] = V6[xx, yy]
            v[0, 1, 1, 0] = V7[xx, yy]
            v[1, 1, 1, 0] = V8[xx, yy]
            v[0, 0, 0, 1] = V9[xx, yy]
            v[1, 0, 0, 1] = V10[xx, yy]
            v[0, 1, 0, 1] = V11[xx, yy]
            v[1, 1, 0, 1] = V12[xx, yy]
            v[0, 0, 1, 1] = V13[xx, yy]
            v[1, 0, 1, 1] = V14[xx, yy]
            v[0, 1, 1, 1] = V15[xx, yy]
            v[1, 1, 1, 1] = V16[xx, yy]
            VV[xx, yy]

# Final interpolated values:
Znew = ZZ
Tnew = TT
Unew = UU
Vnew = VV

#-------------------------------------------------
# Determine wave heights (resultant Hs wave field)
#-------------------------------------------------
 xcontent=xrm;
 ycontent=yrm;
# Define the hemisphere check
if Lat < 0:  # Southern Hemisphere case
    zcontent = np.fliplr(Znew.T)  # Transpose and flip left-right
    tcontent = np.fliplr(Tnew.T)
    ucontent = np.fliplr(-Unew.T)  # Flip and negate Unew
    vcontent = np.fliplr(Vnew.T)
else:  # Northern Hemisphere case
    zcontent = Znew.T  # Transpose without flipping
    tcontent = Tnew.T
    ucontent = Unew.T
    vcontent = Vnew.T

# Calculate significant wave height (Hs) and other parameters
HS = zcontent * Hs_max  # Significant Wave Height (m)
TP = tcontent  # Peak Period (s)
U = ucontent
V = content

# Set the spatial distribution factor
Req = 30  # Spatial distribution Hs/Hsmax was built using R = 30 km

# Prepare the spatial distribution for the wave field
XX = xcontent * Req
YY = ycontent * Req
TP = tcontent  
HS = zcontent * Hs_max

# ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> 
# Plot Interpolated Spatial Distribution 
# ><> ><> ><> ><> ><> ><> ><> ><> ><> ><> 

sname = "Hemisphere North" if Lat >= 0 else "Hemisphere South"  # Hemisphere based on latitude

# Prepare filenames (similar to your MATLAB code)
filename_tp = f"Tp_field_TC_{int(Vmax)}_{int(Vfm)}_{int(Rmax / 10**3)}_{int(R34 / 10**3)}.png"
filename_hs = f"Hsdir_field_TC_{int(Vmax)}_{int(Vfm)}_{int(Rmax / 10**3)}_{int(R34 / 10**3)}.png"

# Plot the TP field (Peak Period)
plt.figure(figsize=(8, 6))
cp = plt.contourf(XX, YY, TP, cmap='viridis')  # Filled contour plot
cb = plt.colorbar(cp)
cb.set_label('Tp (s)')

# Contour lines
v = np.arange(2, 21, 2)  # Contour levels
CS = plt.contour(XX, YY, TP, v, colors='k')
plt.clabel(CS, inline=True, fontsize=8)

# Labels and title
plt.xlabel('Distance (km)')
plt.ylabel('Distance (km)')
plt.xticks(np.arange(-300, 301, 100))  # X axis ticks
plt.yticks(np.arange(-300, 301, 100))  # Y axis ticks
plt.title('Wave field interpolated diagram (T_{p})')

# Mark the center of rotation
x_center = xcontent[140]  # Example index
y_center = ycontent[140]  # Example index
plt.plot(x_center, y_center, 'k+', markersize=10, linewidth=2)

# Add subtitle and text boxes
tpm = np.max(TP)
subname1 = f"T_{p}^{max} = {tpm:.1f} s"
plt.figtext(0.12, 0.79, subname1, ha='left', fontsize=8, backgroundcolor='white')

# Additional information
str2 = [f'V_max = {Vmax} m/s', f'V_fm = {Vfm:.1f} m/s', f'R_max = {Rmax / 1000:.1f} km', f'R_34 = {R34 / 1000:.1f} km']
plt.figtext(0.12, 0.12, '\n'.join(str2), ha='left', fontsize=7, backgroundcolor='white')

# Set color axis and colormap
plt.clim(0, np.round(np.max(TP)))
plt.colormap('viridis')

# Save the plot
plt.savefig(os.path.join("plots", filename_tp))  # Ensure "plots" folder exists
plt.close()

# Plot the HS field (Significant Wave Height)
plt.figure(figsize=(8, 6))
v1 = np.arange(0, 21, 2)
CS = plt.contourf(XX, YY, HS, v1, cmap='viridis')
plt.clabel(CS, inline=True, fontsize=8)

# Labels and title
plt.xlabel('Distance (km)')
plt.ylabel('Distance (km)')
plt.xticks(np.arange(-300, 301, 100))
plt.yticks(np.arange(-300, 301, 100))
plt.title('Wave field interpolated diagram (H_{s})')

# Plot Peak Direction arrows (U, V components)
q = plt.quiver(XX, YY, U, V, color='black', scale=20)
cb = plt.colorbar(CS)
cb.set_label('H_{s} (m)')

# Mark the center of rotation
plt.plot(x_center, y_center, 'k+')

# Add subtitle and text boxes
hsm = np.max(HS)
subname2 = f"H_{s}^{max} = {hsm:.1f} m"
plt.figtext(0.12, 0.79, subname2, ha='left', fontsize=8, backgroundcolor='white')

# Additional information
plt.figtext(0.12, 0.12, '\n'.join(str2), ha='left', fontsize=7, backgroundcolor='white')

# Set color axis and colormap
plt.clim(0, np.round(np.max(HS)))
plt.colormap('viridis')

# Save the plot
plt.savefig(os.path.join("plots", filename_hs))  # Ensure "plots" folder exists
plt.close()

