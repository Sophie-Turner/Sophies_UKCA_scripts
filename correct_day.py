'''
Correct error in day of year field.
'''

import os
import glob
import time
import numpy as np
import file_paths as paths

np_files = sorted(glob.glob(f'{paths.npy}/1982????.npy'))

days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]
days_so_far = np.cumsum(days_in_months)

for file_i in range(327, 335):
  start = time.time()
  np_file = np_files[file_i]
  print(f'\nProcessing file {file_i + 1} of {len(np_files)}.')
  print('Loading data from', np_file)
  data = np.load(np_file)
  day = int(np_file[-6:-4])
  month = int(np_file[-8:-6])
  print(day, month)
  if month == 1:
    day_of_yr = day
  else:
    day_of_yr = day + days_so_far[month - 2] 
  print('day of year:', day_of_yr)  

  data[0] = day_of_yr
  
  print('Saving updated file.')
  np.save(np_file, data)
  
  end = time.time()
  elapsed = end - start
  remaining = elapsed * (len(np_files) - (file_i + 1))
  print(f'That file took {round(elapsed / 60)} minutes.')
  print(f'Approximately {round(remaining / 60)} minutes remaining.') 
