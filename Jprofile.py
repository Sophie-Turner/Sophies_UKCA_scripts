'''
April 2026.
Make a vertical profile of a photolysis rate coefficient
averaged over a specified region of time and space.
See dataset metadata.txt for more info.
'''

import numpy as np
import matplotlib.pyplot as plt


def avg_data(x, y):
  # Average y data by unique x values.
  xs = np.unique(x)
  y_avg = [y[x == val].mean() for val in xs] # Mean.
  return xs, y_avg


# UKCA dataset: randomly sampled points from a UM free-run of the whole year of 1982.
data_path = '/scratch/st838/netscratch/data/ukca_npys/1982_182m.npy'
print('Loading data.')
data = np.load(data_path)
print(data.shape)

# Select the equator.
lat_min, lat_max = -10, 10
data = data[:, ((data[4] > lat_min) & (data[4] < lat_max))] # 4 is latitude.

# Select day-time points only.
data = data[:, (data[11] > 0)] # 11 is downward shortwave flux.

# J rates to look at.
NO = data[47]
N2O = data[43]

# Vertical height (hybrid height coordinate).
alt = data[3] 

# Make a new dataset containing only J rates, alt and day.
day_means = []

# For every day of data, get daily mean.
for day in np.unique(data[0]):
  NO_day = NO[(data[0] == day)]
  N2O_day = N2O[(data[0] == day)]
  alt_day = alt[(data[0] == day)]
  
  # Average data over time, latitude and longitude, resulting in a mean vertical column.
  _, NO_day = avg_data(alt_day, NO_day)
  alt_day, N2O_day = avg_data(alt_day, N2O_day) 

  # Add these data to daily means dataset.
  alt_day = np.asarray(alt_day)
  NO_day = np.asarray(NO_day)
  N2O_day = np.asarray(N2O_day)

  # Build rows for this day.
  day_col = np.full_like(alt_day, day, dtype=np.float32)
  day_block = np.vstack((day_col, alt_day, NO_day, N2O_day))
  day_means.append(day_block)

# Combine everything at the end.
data_means = np.hstack(day_means).astype(np.float32)
print(data_means.shape)

# Save dataset of daily averages.
np.save('/scratch/st838/netscratch/J_profiles/JNxO_alt_daily_mean.npy', data_means)
