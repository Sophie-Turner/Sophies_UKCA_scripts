'''
Compare the effect of ML photolysis on ozone
with ozone from CMIP runs of other models.
'''

import numpy as np
import datetime as dt
import constants as con
import functions as fns
import file_paths as paths
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score


def get_date(data):
  years = data[0].astype(int)
  months = data[1].astype(int)
  time = np.array([dt.datetime(y,m,1) for y,m in zip(years, months)])
  return time
  
  
def avg_data(x, y):
  # Average y data by unique x values.
  xs = np.unique(x)
  y_avg = [y[x == val].mean() for val in xs]
  return xs, y_avg  


# Fast-JX dataset.
path_fj = f'{paths.npy}/fj30yr.npy'

# ML-predicted photolysis dataset.
path_ml = f'{paths.npy}/ml30yr.npy'

# CMIP datasets.
path_cnrm = f'{paths.npy}/o3_CNRM.npy'
path_ncar = f'{paths.npy}/o3_NCAR.npy'
path_bcc = f'{paths.npy}/o3_BCC.npy'

# Names of models.
names = ['UM with Fast-JX', 'UM with random forest', 'CNRM', 'CESM2', 'BCC'] 

# Molar masses used for conversion.
mr_air = 28.97
mr_o3 = 48

# Load all datasets.
data_paths = [path_fj, path_ml, path_cnrm, path_ncar, path_bcc]
for i in range(5):
  data_path = data_paths[i]
  print('\nLoading dataset', data_path)
  data = np.load(data_path)
  print(data.shape)

  # Select ozone MMR or mol frac.
  # UM data.
  if i < 2:
    o3 = data[7]
    # Convert MMR to mole fraction for consistency.
    print('Converting MMR to mole fraction.')
    o3 = o3 * (mr_air / mr_o3)
  # CMIP data.
  else:
    o3 = data[5]   
  print(np.min(o3), np.max(o3)) 

  print('Averaging ozone by month.')
  # Get years and months as dates.
  time = get_date(data)
  times, o3_avg = avg_data(time, o3)
  print(np.min(o3_avg), np.max(o3_avg)) 
'''  
  # Save dataset.
  data = np.vstack((times, o3_avg))
  out_path = data_path[:-4] + '_avg.npy'
  print('Saving dataset', out_path)
  np.save(out_path, data)
  
  # Make the plot. 
  print('Plotting line.')
  name = names[i]
  plt.plot(times, o3_avg, label=name)
  plt.yscale('log')
 
# Format the plot.
plt.title('Ozone from CMIP6 models using different photolysis codes in a 30-year simulation')
plt.xlabel('Simulated year')
plt.ylabel(f'Average ozone mole fraction in atmosphere / mol mol{con.supminus}{con.sup1}')
plt.legend()
plt.show()
plt.close()
'''
