'''
March 2026
Compare the predictive performance of different ML models emulating UKCA photolysis. 
'''

import joblib
import numpy as np
import constants as con
import functions as fns
import file_paths as paths
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

model_name = 'svm' 
model_text = 'SVM'

# Names, info & units of data of interest.
j = 'photolysis rate coefficient' 
pers = f'/ {con.pers}' 
 
  
def avg_data(x, y):
  # Average y data by unique x values.
  xs = np.unique(x)
  y_avg = [np.nanmean(y[x == val]) for val in xs]
  y_avg = np.nan_to_num(y_avg)
  return xs, y_avg
  

def print_metrics(out_test, preds):
  r2 = r2_score(out_test, preds)
  mse = mean_squared_error(out_test, preds)
  print(f'R2 = {r2}')
  print(f'MSE = {mse}')


inputs, targets, preds = fns.load_model_data(model_name)

# HCHOr, OCS, NO3, H2O, H2O2, N2O, NO2
j_names = ['HCHOr', 'OCS', f'NO{con.sub3}', f'H{con.sub2}O', 
           f'H{con.sub2}O{con.sub2}', f'N{con.sub2}O', f'NO{con.sub2}']

# NH only.
lat_min, lat_max = 0, 90
inputs_nh = inputs[(inputs[:, 4] > lat_min) & (inputs[:, 4] < lat_max)]
targets_nh = targets[(inputs[:, 4] > lat_min) & (inputs[:, 4] < lat_max)] 
preds_nh = preds[(inputs[:, 4] > lat_min) & (inputs[:, 4] < lat_max)]

day = inputs_nh[:, 0]
lvl = inputs[:, 2]

for i in range(7):
  target = targets[:, i] 
  target_nh = targets_nh[:, i] 
  pred = preds[:, i]
  pred_nh = preds_nh[:, i] 
  j_name = j_names[i]
  
  # Timeseries plot.
  _, target_avg = avg_data(day, target_nh)
  days, pred_avg = avg_data(day, pred_nh)
  plt.plot(days, target_avg, label=f'{j_name} from UKCA')
  plt.plot(days, pred_avg, label=f'{j_name} from {model_text}')
  plt.xlabel('Day of year')
  plt.ylabel(f'Mean {j_name} J-value {pers}')
  plt.title(f'Northern hemispheric daily mean {j_name} photolysis rate coefficients from UKCA and {model_text}')
  plt.legend()
  plt.show()
  plt.close()
  
  # Level plot.
  _, target_avg = avg_data(lvl, target)
  lvls, pred_avg = avg_data(lvl, pred)
  plt.scatter(target_avg, lvls, label=f'{j_name} from UKCA')
  plt.scatter(pred_avg, lvls, label=f'{j_name} from {model_text}')
  plt.xscale('log')
  plt.xlabel(f'Mean {j_name} J-value {pers}')
  plt.ylabel('Vertical model level')
  plt.title(f'Global mean {j_name} photolysis rate coefficients by height from UKCA and {model_text}')
  plt.legend()
  plt.show()
  plt.close()

  # Correlation plot.
  plt.scatter(target, pred, alpha=0.05)
  plt.xlabel(f'{j_name} J-value from UKCA {pers}')
  plt.ylabel(f'{j_name} J-value from {model_text} {pers}')
  plt.title(f'{j_name} photolysis rate coefficients from UKCA and {model_text}')
  plt.show()
  plt.close()
  
  print(f'\n{j_name} R2:', r2_score(target, pred))
  print(f'{j_name} MSE:', mean_squared_error(target, pred))
