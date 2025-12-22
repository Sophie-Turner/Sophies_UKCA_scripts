'''
Name: Sophie Turner.
Date: 3/10/2024.
Contact: st838@cam.ac.uk
Reduce the size of a whole year of UKCA output data into a dataset small enough to fit in < 400 GB memory.
'''

import os
import time
import glob
import math
import numpy as np
import constants as con
import functions as fns
import file_paths as paths


def shift_radiation(data_today, data_yesterday):
  # Shift all radiation fields by 1 hour to match state at photolysis timesteps in the UM.
  # data_yesterday is None if it's the first day of the dataset.
  # Indices of radiation fields (shortwave fluxes and SZA).
  rad_idx = [9, 10, 11]
  hours = np.unique(data[1])
  # Move radiation fields to 1 hour earlier in time.
  # Loop backwards through time from the end to the start.
  for i in range((len(hours) - 1), -1, -1):
    hour_ts = hours[i]
    # Do this for every timestep except the first.
    # May need to be changed to hour_ts != 1 if dataset includes 1st model run day.
    if (data_yesterday is not None) or (hour_ts != hours[0]):
      # If it's midnight we need to get the data from 11pm of the previous day.
      if hour_ts == 0:
        data_rad = data_yesterday
        hour_rad = 23
      # Otherwise, get the data from the previous hour of this day.
      else:
        data_rad = data_today
        hour_rad = hours[i-1]
      # Get the current timestep.
      data_ts = data_today[:, (data_today[1] == hour_ts)]
      # Get the required raditation timestep.
      data_rad = data_rad[:, (data_rad[1] == hour_rad)]
      # Get the required radiation data.
      rads = data_rad[rad_idx]
      # Move it to the current timestep.
      data_ts[rad_idx] = rads
      data_today[:, (data_today[1] == hour_ts)] = data_ts
  return(data_today)
  

def sample_day(data, points, data_new):
  # Takes ~ 1 minute per day file for 2 million points.
  data = np.load(day_file)
  # Remove night and upper portion where Fast-J might not work. 
  data = fns.day_trop(data)
  # Reduce data randomly. Seems naiive but is suitable due to large amount of data.
  full_size = len(data[0]) 
  ids = con.rng.integers(0, full_size, points)
  ids = np.sort(ids)
  data = data[:, ids]
  # Add the data to the array of the new dataset.
  data_new = np.hstack((data_new, data))    
  return(data_new)
  
  
def sample_day_night(data, points, data_new):
  # Split into day and night portions and remove any negative values. 
  # Index 11 is downward shortwave flux.
  day_data = data[:, data[11] > 0] 
  night_data = data[:, data[11] == 0]
  # Sample 99.5% of the data from daytime points.
  day_size = math.floor(points * 0.995) 
  day_ids = con.rng.integers(0, len(day_data[0]), day_size)
  day_ids = np.sort(day_ids)
  day_data = day_data[:, day_ids]
  # Sample 0.5% of the data from night points.
  night_size = math.floor(points * 0.005)
  night_ids = con.rng.integers(0, len(night_data[0]), night_size)
  night_idx = np.sort(night_ids)
  night_data = night_data[:, night_ids]
  # Stick the samples back together.
  # The night and day prtions will not be in order. Probably doesn't matter?
  data_new = np.hstack((data_new, day_data, night_data))
  return(data_new)
  

# How many data points we want per chosen day of data.
points = 504000

# Prepare the new file and data array of 32 bit floats.
name_new = '1982_182m_rad_shift'
path_data_new = f'{paths.npy}/{name_new}.npy'
path_meta_new = f'{paths.npy}/{name_new}_metadata.txt'
#data_new = np.empty((con.n_fields, 0), dtype=np.float32)
data_new = np.empty((49, 0), dtype=np.float32)

# Check if we are accidentally overwriting something.
if os.path.exists(path_data_new):
  print(f'Warning: The file {path_data_new} already exists.')
  overwrite = input('Do you want to overwrite the data in this file? (y/n): ').strip().lower()
  if overwrite != 'y':
    exit(1)

# Get the daily npy files for the year.
year_files = sorted(glob.glob(f'{paths.npy}/1982????.npy'))
# Out dataset starts with no yesterday.
data_yesterday = None
# Every day file...
#for i in range(len(year_files)):
for i in range(2):
  start = time.time()
  day_file = year_files[i]
  print(f'\nProcessing file {i+1} of {len(year_files)}:', day_file)
  data = np.load(day_file)    
  print('Shifting radiation fields by 1 hour.')
  data = shift_radiation(data, data_yesterday)
  print('Sampling data.')
  data_new = sample_day_night(data, points, data_new)
  print('New dataset so far:', data_new.shape)
  # Keep today's dataset open while looking at tomorrow. 
  data_yesterday = data 
  # Remove the first timestep of the dataset as we have no radiation to put in that.
  if i == 0:
    print('Removing first timestep of dataset.', data_new.shape)
    mask = ~(data_new[1] == 0)
    data_new = data_new[:, mask]
  end = time.time()
  elapsed = end - start
  remaining = elapsed * (len(year_files) - (i + 1))
  minutes = round(remaining / 60)
  print(f'Approximately {minutes} minutes remaining.') 
    
# Set SZA to 100 degrees instead of 90 when there's no light to avoid confusing the RF.
# 100 degrees in cos radians = -0.173648178.
# 9 is SZA and 11 is downward SW flux. 
print('Setting night SZA to 100 deg.')
data_new[9, (data_new[9] < 7e-17) & (data_new[11] == 0)] = -0.173648178  
  
# Write & save metadata about the dataset.
text = f'The dataset, {name_new}.npy, is a year of UM output data at a resolution low enough to fit in < 400 GB program memory.\n\
Each day of data contains {points} hourly grid points (reduced from 56.4 million), sampled from the full, unchanged range of data, \
with 99.5% of the sample at daytime and 0.5% at night.'
print(f'Writing metadata to {path_meta_new}')
meta = open(path_meta_new, 'w')
meta.write(text)
meta.close()  

# Save the new dataset.
print(f'Saving new dataset at {path_data_new}')
np.save(path_data_new, data_new)
