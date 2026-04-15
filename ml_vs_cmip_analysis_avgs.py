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
  y_avg = np.array([np.nanmean(y[x == val]) for val in xs])
  return xs, y_avg  


# Fast-JX dataset.
path_fj = f'{paths.npy}/fj30yr_avg.npy'

# ML-predicted photolysis datasets.
path_ml = f'{paths.npy}/ml30yr_masked_avg.npy'

# CMIP datasets.
path_cnrm = f'{paths.npy}/o3_CNRM_avg.npy'
path_ncar = f'{paths.npy}/o3_NCAR_avg.npy'
path_bcc = f'{paths.npy}/o3_BCC_avg.npy'

# Names of models.
names = ['BCC', 'UKCA with Fast-JX', 'UKCA with random forest', 'CESM2', 'CNRM'] 

# Molar masses used for conversion.
mr_air = 28.97
mr_o3 = 48

# Load all datasets.
data_paths = [path_bcc, path_fj, path_ml, path_ncar, path_cnrm]
for i in range(5):
  data_path = data_paths[i]
  print('\nLoading dataset', data_path)
  data = np.load(data_path, allow_pickle=True)
  print(data.shape)  
  
  times = data[0]
  o3_avg = data[1]
  '''
  if i == 2:
    o3_avg[:10] = o3_avg[:10] * 0.99
    o3_avg[10:] = o3_avg[10:] * 1.01
  '''
  # Make the plot. 
  print('Plotting line.')
  name = names[i]
  plt.plot(times, o3_avg, label=name)
  #plt.yscale('log')
 
# Format the plot.
plt.title('Ozone from CMIP6 models using different photolysis codes in a 30-year simulation')
plt.xlabel('Simulated year')
plt.ylabel(f'Average ozone mole fraction in atmosphere / mol mol{con.supminus}{con.sup1}')
plt.legend()
plt.show()
plt.close()
