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

times = day[4].coord('time').hour.array
lats = day[4].coord('latitude').array
lons = day[4].coord('longitude').array
sw_flux_up = day[4].array
sw_flux_down = day[5].array
print(type(lons))

lats = lats[0:6]
lons = lons[0:16]
print(f"Latitudes: {np.min(lats)} to {np.max(lats)}")
print(f"Longitudes: {np.min(lons)} to {np.max(lons)}")
print(f"Size of domain: {sw_flux_up[0,:,0:6,0:16].size}")
print(f"Shape of domain: {sw_flux_up[0,:,0:6,0:16].shape}")

for i in range(6):
  print(f"\nTimestep: {(i+1)*3}")
  print(f"Hour: {times[i]}:00")
  print(f"Upward shortwave flux: {sw_flux_up[i,:,0:6,0:16].min()} to {sw_flux_up[i,:,0:6,0:16].max()}")
  print(f"Downward shortwave flux: {sw_flux_down[i,:,0:6,0:16].min()} to {sw_flux_down[i,:,0:6,0:16].max()}")
