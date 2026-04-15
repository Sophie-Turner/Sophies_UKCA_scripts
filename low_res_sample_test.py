'''
Name: Sophie Turner.
Date: 3/10/2024.
Contact: st838@cam.ac.uk
Reduce the size of a whole UKCA output dataset into a dataset small enough to fit into memory.
Make a matching pair of datasets which can be directly compared; a control and a test dataset.
'''

import os
import time
import glob
import math
import numpy as np
import constants as con
import functions as fns
import file_paths as paths
  
  
def sample_test_set(i, files, points, data_new, ids):
  # Sample online outputs only for testing, not training.
  # Get this file.
  this_file = files[i]
  this_data = np.load(this_file)
  print('This files data:', this_data.shape)
  if this_data.shape[0] < 44:
    print('Missing data. Skipping this file.')
  else:
    this_data = this_data[:, ids]
    data_new = np.hstack((data_new, this_data))
    print('New dataset so far:', data_new.shape)
  return(data_new)  
  
  
def get_date(file_path):
  date = file_path[-8:-4]
  print(date)
  #date = file_path[-12:-4]
  return date
  
  
def write_meta(name, path):
  # Write & save metadata about the dataset.
  text = f'The dataset, {name}.npy, is a UM output dataset at a resolution\
  low enough to fit into program memory, sampled from the full, unchanged range of data.'
  print(f'Writing metadata to {path}')
  meta = open(path, 'w')
  meta.write(text)
  meta.close() 
  

# How many data points we want per chosen file of data.
points = 51667

# Prepare the new file and data array of 32 bit floats.
name_ctl = 'fj1yr'
name_test = 'ml1yr'
path_data_ctl = f'{paths.npy}/{name_ctl}.npy'
path_meta_ctl = f'{paths.npy}/{name_ctl}_metadata.txt'
path_data_test = f'{paths.npy}/{name_test}.npy'
path_meta_test = f'{paths.npy}/{name_test}_metadata.txt'
#data_ctl = np.empty((44, 0), dtype=np.float32)
data_test = np.empty((44, 0), dtype=np.float32)

# Check if we are accidentally overwriting something.
if os.path.exists(path_data_ctl) or os.path.exists(path_data_test):
  print(f'Warning: The file already exists.')
  overwrite = input('Do you want to overwrite the data in this file? (y/n): ').strip().lower()
  if overwrite != 'y':
    exit(1)

# Get the npy files in the dataset.
files_ctl = sorted(glob.glob(f'{paths.data}/{name_ctl}/???????.npy'))
files_test = sorted(glob.glob(f'{paths.data}/{name_test}/????????.npy'))

# Make sure both datasets have the same date files and no extras.
dates_ctl = {get_date(f) for f in files_ctl}
dates_test = {get_date(f) for f in files_test}
matched = dates_ctl & dates_test
files_ctl = [f for f in files_ctl if get_date(f) in matched]
files_test = [f for f in files_test if get_date(f) in matched]

# Open the 1st file to get length for random numbers.
print('Getting data sizes for random indices.')
data = np.load(files_test[0])
fullsize = len(data[0]) 
del data

# Every file...
for i in range(len(files_test)):
  start = time.time()
  # Make sure both datasets get the same indices sampled.
  ids = np.linspace(0, fullsize, points, dtype=np.int32)
  ids = np.sort(ids)  
  #print(f'Processing file {i+1} of {len(matched)} in control dataset.')
  #data_ctl = sample_test_set(i, files_ctl, points, data_ctl, ids)
  print(f'Processing file {i+1} of {len(files_test)} in test dataset.')
  data_test = sample_test_set(i, files_test, points, data_test, ids)
  end = time.time()
  elapsed = end - start
  remaining = elapsed * (len(files_test) - (i + 1))
  minutes = round(remaining / 60)
  print(f'Approximately {minutes} minutes remaining.') 
  
# Save the new datasets.
print(f'Saving new datasets at {path_data_ctl} and {path_data_test}')
#np.save(path_data_ctl, data_ctl)  
np.save(path_data_test, data_test) 
  
# Write & save metadata about the dataset.
#write_meta(name_ctl, path_meta_ctl)
#write_meta(name_test, path_meta_test)
