'''
Name: Sophie Turner.
Date: 24/11/2025.
Contact: st838@cam.ac.uk.
Check specific field values in a domain to compare against values in UKCA code. 
'''

import cf
import numpy as np
import file_paths as paths

# A sample UM output .pp file.
um_file = f'{paths.pp}/dt341a.pn19820101.pp'

print(f'\n{um_file}')
  
# Open it in CF Python.
print('Loading data.')
day = cf.read(um_file)

times = day[0].coord('time').hour.array
lats = day[0].coord('latitude').array
lons = day[0].coord('longitude').array
humidity = day[0].array
cloud = day[1].array
pressure = day[2].array
sza = day[3].array 
sw_flux_up = day[4].array
sw_flux_down = day[5].array
temp = day[6].array
o3 = day[7].array

lats = lats[0:6]
lons = lons[0:16]
print(f"Latitudes: {np.min(lats)} to {np.max(lats)}")
print(f"Longitudes: {np.min(lons)} to {np.max(lons)}")
print(f"Size of domain: {sw_flux_up[0,:,0:6,0:16].size}")
print(f"Shape of domain: {sw_flux_up[0,:,0:6,0:16].shape}")

print("\nTimestep: 3")
print(f"Hour: {times[0]}:00")
print(f"Humidity: {humidity[0,:,0:6,0:16].min()} to {humidity[0,:,0:6,0:16].max()}")
print(f"Cloud: {cloud[0,:,0:6,0:16].min()} to {cloud[0,:,0:6,0:16].max()}")
print(f"Pressure: {pressure[0,:,0:6,0:16].min()} to {pressure[0,:,0:6,0:16].max()}")
print(f"SZA: {sza[0,0:6,0:16].min()} to {sza[0,0:6,0:16].max()}")
print(f"Upward shortwave flux: {sw_flux_up[0,:,0:6,0:16].min()} to {sw_flux_up[0,:,0:6,0:16].max()}")
print(f"Downward shortwave flux: {sw_flux_down[0,:,0:6,0:16].min()} to {sw_flux_down[0,:,0:6,0:16].max()}")
print(f"Temperature: {temp[0,:,0:6,0:16].min()} to {temp[0,:,0:6,0:16].max()}")
print(f"Ozone column: {o3[0,:,0:6,0:16].min()} to {o3[0,:,0:6,0:16].max()}")
