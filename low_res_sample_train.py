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
  
  
def sample_day_night_random(i, year_files, points, data_new):
  print(f'Processing file {i+1} of {len(year_files)}.')
  # Get this file.
  day_file = year_files[i]
  data = np.load(day_file)
  print(data.shape)  
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
  print('New dataset so far:', data_new.shape)
  print("Random number sum:", sum(day_ids))
  return(data_new) 
  
  
# How many data points we want per chosen day of data.
points = 504000

# Prepare the new file and data array of 32 bit floats.
name_new = '1982_train'
path_data_new = f'{paths.npy}/{name_new}.npy'
path_meta_new = f'{paths.npy}/{name_new}_metadata.txt'
#data_new = np.empty((con.n_fields, 0), dtype=np.float32)
data_new = np.empty((44, 0), dtype=np.float32)

# Check if we are accidentally overwriting something.
if os.path.exists(path_data_new):
  print(f'Warning: The file {path_data_new} already exists.')
  overwrite = input('Do you want to overwrite the data in this file? (y/n): ').strip().lower()
  if overwrite != 'y':
    exit(1)

# Get the npy files in the dataset.
files = sorted(glob.glob(f'{paths.npy}/1982????.npy'))
# Every file...
for i in range(len(files)):
  start = time.time()
  data_new = sample_test_set(i, files, points, data_new)
  end = time.time()
  elapsed = end - start
  remaining = elapsed * (len(files) - (i + 1))
  minutes = round(remaining / 60)
  print(f'Approximately {minutes} minutes remaining.') 
  
# Save the new dataset.
print(f'Saving new dataset at {path_data_new}')
np.save(path_data_new, data_new)  
  
# Write & save metadata about the dataset.
text = f'The dataset, {name_new}.npy, is a year of UM output data at a resolution low enough to fit in < 400 GB program memory.\n\
Each day of data contains {points} hourly grid points (reduced from 56.4 million), sampled from the full, unchanged range of data.'
print(f'Writing metadata to {path_meta_new}')
meta = open(path_meta_new, 'w')
meta.write(text)
meta.close()  
