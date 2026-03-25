'''
Name: Sophie Turner.
Date: 3/12/2025.
Contact: st838@cam.ac.uk
Compile data from daily UM .pp files and make a big 2D .npy file 
of shape (samples, features) for ML, containing all UKCA dimensions 
and fields flattened. Time and space are added as columns. 
This is different to the 2015 version because it is designed for the
data within the UM, to pass to a random forest in Fortran, not Python.
'''

import os
import cf
import glob
import time
import numpy as np
import file_paths as paths


def count_days(day, month, year):
  print('day, month, year:', day, month, year)
  # Turn day of month into day of year. 
  # Don't need December because nothing comes after it.
  days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]
  # Is it a leap year?
  leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
  if leap:
    days_in_months[2] = 29
  # Cumulative sum of days in year by month.  
  days_so_far = np.cumsum(days_in_months)
  # Not needed for January.
  if month == 1:
    day_of_yr = day
  else:
    day_of_yr = day + days_so_far[month - 2] 
  print('day of year:', day_of_yr)
  return day_of_yr


def pad_dim(dim, rep, stride, dims_table):
  start = time.time()
  padded = np.empty(0, dtype=np.float32)  
  for i in range(rep):
    new_dim = np.repeat(dim, stride).astype(np.float32)
    padded = np.append(padded, new_dim)
  dims_table = np.r_[dims_table, [padded]] 
  del(new_dim, padded) 
  end = time.time()
  minutes = (end - start) / 60
  print(f'That took {round(minutes, 1)} minutes.')
  return(dims_table)


def pad_dims(field, out_path):
  # Get the dimension values.
  print('Getting the dimension values as np arrays.')
  hours = field.coord('time').hour.array.astype(np.float32) # Hour of day.
  heights = field.coord('atmosphere_hybrid_height_coordinate').array.astype(np.float32)
  lats = field.coord('latitude').array.astype(np.float32)
  lons = field.coord('longitude').array.astype(np.float32)

  nhours = len(hours)
  nlvls = len(heights) 
  nlats = len(lats)
  nlons = len(lons)
  
  # We want level number as well as hybrid height. Usually 1 to 85.
  lvls = np.arange(1, nlvls + 1, dtype=np.float32)
  # Relative lvl to 85 in case it's not an 85-lvl, 85km grid, for transferrability.
  sea_alts = field.domain_ancillary('domainancillary0').array.astype(np.float32)
  top = sea_alts[-1] / 1000 # m to km.
  rel_lvls = (top / len(lvls)) * lvls

  stride_hour = nlvls * nlats * nlons 
  stride_lvl = nlats * nlons 
  stride_lat = nlons
  stride_lon = 1

  rep_hour = 1
  rep_lvl = nhours
  rep_lat = nhours * nlvls
  rep_lon = nhours * nlvls * nlats
  
  full_size = nhours * nlvls * nlats * nlons  
  
  dims_table = np.empty((0, full_size), dtype=np.float32)   
  print('Padding flat hours.')
  dims_table = pad_dim(hours, rep_hour, stride_hour, dims_table)
  print('Padding flat levels.')
  dims_table = pad_dim(rel_lvls, rep_lvl, stride_lvl, dims_table)
  print('Padding flat heights.')
  dims_table = pad_dim(heights, rep_lvl, stride_lvl, dims_table)
  print('Padding flat latitude.')
  dims_table = pad_dim(lats, rep_lat, stride_lat, dims_table)
  print('Padding flat longitude.')
  dims_table = pad_dim(lons, rep_lon, stride_lon, dims_table)

  print(f'Saving entire {dims_table.shape} dims array as {out_path}.')
  np.save(out_path, dims_table)    


data_dir = 'ml1day_masked_noH2O' 

# Input files. 
pp_files = sorted(glob.glob(f'{paths.data}/{data_dir}/dv???a.px????????.pp'))

# File path for padded dims to match flattened fields.
dims_path = f'{paths.data}/{data_dir}/dims.npy' 

# Make the dims if not already available.
if not os.path.exists(dims_path):
  # Read 1 field to quickly get dims.
  field = cf.read(pp_files[0], select='stash_code=50501')[0]
  pad_dims(field, dims_path)  
  
# Get the flattened dimensions.  
print('\nLoading dims data.')
dims = np.load(dims_path)
n_samples = dims.shape[1]

# Keep track of which files have missing data.
missing = []

# Pick the day file to read.
for file_i in range(len(pp_files)):
  # Name of the new file to be saved.
  pp_file = pp_files[file_i]
  npy_file = f'{paths.data}/{data_dir}/{pp_file[-11:-3]}.npy'
  is_complete = True
  
  # Don't waste time overwriting existing files.
  if not os.path.exists(npy_file):
    start = time.time()
    print(f'\nReading .pp file {file_i + 1} of {len(pp_files)}:')
    print(pp_file)
    try:
      day = cf.read(pp_file)
    except:
      print('Could not read the file. Skipping for now. Try downloading it again.')
      missing.append(pp_file)
      continue
      
    # Get the number of vertical levels.
    field = day[0]
    nlvls = field.coord('atmosphere_hybrid_height_coordinate').size

    # Re-use the dims. 
    table = dims.copy()

    # Get the day of the year.
    day_of_month = field.coord('time').day.array[0]
    month = field.coord('time').month.array[0]
    year = field.coord('time').year.array[0]
    day_of_yr = count_days(day_of_month, month, year)
    
    # Put day of year before dims and pad for the whole day.
    print('Padding day of year.')
    day_of_yr = np.full(table.shape[1], day_of_yr, dtype=np.float32)
    table = np.vstack((day_of_yr, table), dtype=np.float32)

    # For each field, save all the field data in another row.
    for i in range(len(day)):
      print(f'Converting field {i+1} of {len(day)}.')
      field = day[i]
      print(field.long_name)   
      # If there are only 3 dimensions add a 4th and pad it by lvl 
      # so that the data match the flattened dims.
      if field.ndim == 3:
        print('Missing dimension. Padding.')    
        field = field.array
        times = nlvls
        stride = field.shape[1] * field.shape[2]
        field = field.flatten()
        # Make a new 1d field array.
        padded = np.empty(0, dtype=np.float32)
        # Loop through old 1d field indices with strides of nlat x nlon until the end.
        for j in range(0, len(field), stride):
          # Pick sub arrays in sections of len nlat x nlon
          rep = field[j : j + stride]
          # Repeat that sub array ntime x nalt.
          rep = np.tile(rep, times)
          # Append it to the new field array.
          padded = np.append(padded, rep)
        field = padded
      else:
        field = field.flatten()
      # Check for missing data.
      if field.size == n_samples:   
        table = np.vstack((table, field), dtype=np.float32)
        print('Shape of table:', table.shape)
      else:
        print(f'Missing data in {pp_file}.\nSkipping this file for now. Try downloading it again.')
        missing.append(pp_file)
        is_complete = False
        break

    if is_complete: 
      # Save the np array as a npy file containing this day of data.
      print('Saving .npy file:', npy_file)
      np.save(npy_file, table)
      # Update us on how long it's taking, if applicable.
      end = time.time()
      elapsed = end - start
      remaining = elapsed * (len(pp_files) - (file_i + 1))
      print(f'That file took {round(elapsed / 60)} minutes.')
      print(f'Approximately {round(remaining / 60)} minutes remaining.') 

if len(missing) != 0:
  print('\nThe following files had missing data and were not converted:')
  for name in missing:
    print(name)
