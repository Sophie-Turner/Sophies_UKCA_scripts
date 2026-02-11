'''
Name: Sophie Turner.
Date: 4/4/2025.
Contact: st838@cam.ac.uk.
Check the order and number of fields in .pp file and their values' ranges.
'''

import cf
import numpy as np
import file_paths as paths

# A sample UM output .pp file or list of files.
um_files = [f'{paths.data}/fj30yr/dv850a.px1982feb.pp']

for um_file in um_files:
  print(f'\n{um_file}')
  
  # Open it in CF Python.
  print('Loading data.')
  day = cf.read(um_file)
  
  for i in range(len(day)):
    field = day[i]
    # Long name is more informative than identity.
    try:
      name = field.long_name
    # Sometimes a field has no long name.
    except:
      name = field.identity()
    print(i, name)
    print(f'{name} ranges from {field.min()} to {field.max()}')
    print('Shape of field:', field.shape)
  
  print(f'There are {len(day)} fields in the dataset.\n')  
