'''
Name: Sophie Turner.
Date: 16/4/2026.
Make sure there are no missing or extra files in 2 datasets which need to be matched 1:1.
'''

import os
import glob
import numpy as np
import file_paths as paths

name1 = 'ml30yr_masked'
name2 = 'fj30yr'
names = [name1, name2]
dates = [[], []]

for i in range(2):
  name = names[i]
  files = sorted(glob.glob(f'{paths.data}/{name}/???????.npy'))
  files = np.array(files)
  dates[i] = np.array([f[-11:-4] for f in files])
  
for i in range(len(dates[0])):
  date = dates[0][i]
  if date not in dates[1]:
    print(f'{date} is missing from the {names[1]} dataset.')
    
for i in range(len(dates[1])):
  date = dates[1][i]
  if date not in dates[0]:
    print(f'{date} is missing from the {names[0]} dataset.')
