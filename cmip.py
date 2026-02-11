'''
Get the CMIP data required for comparison with my data.
'''

import cf
import time
import numpy as np
import file_paths as paths

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def pad_dim(dim, rep, stride, table):
  start = time.time()
  padded = np.empty(0, dtype=np.float32)  
  for i in range(rep):
    new_dim = np.repeat(dim, stride).astype(np.float32)
    padded = np.append(padded, new_dim)
  table = np.r_[table, [padded]] 
  end = time.time()
  minutes = (end - start) / 60
  print(f'That took {round(minutes, 1)} minutes.')
  return(table)


name = 'o3_CNRM'
print()
print(name)
cmip_file = f'{paths.data}/cmip/{name}*.nc'
npy_file = f'{paths.npy}/{name}.npy'
data = cf.read(cmip_file)[0]

# Select the section from 1982 to 2012.
data = data.subspace(T=cf.wi(cf.dt('1982-01-01'), cf.dt('2012-01-01')))
print()
print(data)
print()
  
# Columns for dimensions.
print('Getting the dimension values as np arrays.')
years = data.coord('time').year.array.astype(np.float32)
months = data.coord('time').month.array.astype(np.float32)
press = data.coord('air_pressure').array.astype(np.float32)
lats = data.coord('latitude').array.astype(np.float32)
lons = data.coord('longitude').array.astype(np.float32)

ntimes = len(years)
npress = len(press) 
nlats = len(lats)
nlons = len(lons)

stride_time = npress * nlats * nlons 
stride_pres = nlats * nlons 
stride_lat = nlons
stride_lon = 1

rep_time = 1
rep_pres = ntimes
rep_lat = ntimes * npress
rep_lon = ntimes * npress * nlats
  
full_size = ntimes * npress * nlats * nlons  
   
print('Padding flat time.')  
table = np.empty((0, full_size), dtype=np.float32)   
table = pad_dim(years, rep_time, stride_time, table)
table = pad_dim(months, rep_time, stride_time, table)
print('Padding flat pressure.')
table = pad_dim(press, rep_pres, stride_pres, table)
print('Padding flat latitude.')
table = pad_dim(lats, rep_lat, stride_lat, table)
print('Padding flat longitude.')
table = pad_dim(lons, rep_lon, stride_lon, table) 

# Flatten data.
print('Flattening field data.')
data = data.flatten()

# Turn missing data into NaNs.
print('Fixing missing data.')
data = data.array
data = data.filled(np.nan)
print('Max value in dataset:', np.nanmax(data))

# Add as a new column.
table = np.vstack((table, data), dtype=np.float32)
print('Shape of table:', table.shape)  
  
# Save the 2D numpy file.
print('Saving 2D numpy dataset.')
np.save(npy_file, table)
