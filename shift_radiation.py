'''
Shift all radiation fields by 1 hour to match state at photolysis timesteps in the UM.
'''

import os
import numpy as np
import file_paths as paths

# Input and output file paths.
file_name = '1982_183m'
data_path = f'{paths.npy}/{file_name}.npy'
out_path = f'{paths.npy}/{file_name}_rad_shift.npy'

# Check if we are accidentally overwriting something.
if os.path.exists(out_path):
  print(f'Warning: The file {out_path} already exists.')
  overwrite = input('Do you want to overwrite the data in this file? (y/n): ').strip().lower()
  if overwrite != 'y':
    exit(1)

# Load year sample dataset.
print('Loading data from', data_path)
data = np.load(data_path)
print(data.shape)

# Indices of radiation fields (shortwave fluxes and SZA).
rad_idx = [9, 10, 11]

# Get time values in data.
days = np.unique(data[0])
hours = np.unique(data[1])

# Move radiation fields to 1 hour earlier in time.
# Loop backwards through time from the end to the start.
for i in range((len(days) - 1), -1, -1):
  day_ts = days[i]
  for j in range((len(hours) - 1), -1, -1):
    hour_ts = hours[j]
    # Do this for every timestep except the first.
    # May need to be changed to hour_ts != 1 if dataset includes 1st model run day.
    if day_ts != days[0] or hour_ts != hours[0]: 
      # If it's midnight we need to get the data from 11pm of the previous day.
      if hour_ts == 0:
        day_rad = days[i-1]
        hour_rad = 23
      # Otherwise, get the data from the previous hour of this day.
      else:
        day_rad = day_ts
        hour_rad = hours[j-1]
      # Get the current timestep.
      data_ts = data[:, (data[0] == day_ts) & (data[1] == hour_ts)]
      # Get the required raditation timestep.
      data_rad = data[:, (data[0] == day_rad) & (data[1] == hour_rad)]
      # Get the required radiation data.
      rads = data_rad[rad_idx]
      # Move it to the current timestep.
      data_ts[rad_idx] = rads
      data[:, (data[0] == day_ts) & (data[1] == hour_ts)] = data_ts
    
# Remove the first timestep of the dataset as we have no radiation to put in that.
mask = ~((data[0] == 0) & (data[1] == 0))
data = data[:, mask]

# Save altered dataset.
print('Saving new dataset at', out_path)
np.save(out_path, data)
