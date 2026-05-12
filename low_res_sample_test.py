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
  

# How many data points we want per chosen file of data.
points = 51667

# Prepare the new file and data array of 32 bit floats.
name = 'ml30yr_masked'
path_data = f'{paths.npy}/{name}.npy'

# Check for accidentally overwriting something.
if os.path.exists(path_data):
  print(f'Warning: The file {path_data} already exists.')
  overwrite = input('Do you want to overwrite the data in this file? (y/n): ').strip().lower()
  if overwrite != 'y':
    exit(1)

# Get the npy files in the dataset.
files = sorted(glob.glob(f'{paths.data}/{name}/???????.npy'))

# Open the 1st file to get length.
print('Getting data sizes for indices.')
data = np.load(files[0])
fullsize = len(data[0]) 
ids = np.linspace(0, fullsize-1, points, dtype=np.int32) 

data = np.empty((44, 0), dtype=np.float32)

# Every file...
for i in range(len(files)):
  start = time.time()
  print(f'Processing file {i+1} of {len(files)} in dataset.')
  data = sample_test_set(i, files, points, data, ids)
  end = time.time()
  elapsed = end - start
  remaining = elapsed * (len(files) - (i + 1))
  minutes = round(remaining / 60)
  print(f'Approximately {minutes} minutes remaining.') 
  
# Save the new datasets.
print(f'Saving new dataset at {path_data}')
np.save(path_data, data) 

'''  
data = np.load(files[0])
shape = data.shape
print(f'All the files should be of shape {shape}')
for f in files:
  data = np.load(f)
  if data.shape != shape:
    #print(f'{f} is a different shape! {data.shape}')
    print(f)
'''
