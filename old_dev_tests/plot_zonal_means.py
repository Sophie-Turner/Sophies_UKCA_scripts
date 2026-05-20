'''
Plot zonal means of predictions vs targets.
'''

import numpy as np
import constants as con
import file_paths as paths
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def get_area(inputs, outputs, alt_min, alt_max, lat_min, lat_max):
  # Get an area to average over.
  outputs = outputs[(inputs[:, 2] > alt_min) & (inputs[:, 2] < alt_max)] # 2 is model level.
  inputs = inputs[(inputs[:, 2] > alt_min) & (inputs[:, 2] < alt_max)]
  outputs = outputs[(inputs[:, 3] > lat_min) & (inputs[:, 3] < lat_max)] # 3 is latitude.    
  inputs = inputs[(inputs[:, 3] > lat_min) & (inputs[:, 3] < lat_max)]
  return inputs, outputs
  

def avg_data(x, y):
  # Average y data by unique x values.
  xs = np.unique(x)
  y_avg = [y[x == val].mean() for val in xs]
  return xs, y_avg


# Random forest model.
model = 'rf_fortran_poc_fixed_days'
path = f'{paths.mod}/{model}'

# Where to save the plots.
exp = 'offline_poc_trop_trop'
fig_path = f'{paths.analysis}/{exp}'

print('\nLoading test data from', path)
print(exp)

# Inputs dataset.
input_path = f'{path}/{model}_inputs.npy'
input_data = np.load(input_path)

# Test targets dataset.
target_path = f'{path}/{model}_targets.npy'
target_data = np.load(target_path)

# Predictions dataset.
pred_path = f'{path}/{model}_pred.npy'
pred_data = np.load(pred_path)

print(input_data.shape, target_data.shape, pred_data.shape)

# Names of J rates.
names = [
         'HCHO (radical reaction)', 
         'HCHO (molecular reaction)',
	 'MeCOCHO', 
	 f'Cl{con.sub2}O{con.sub2}',
         'OCS', 
	 f'SO{con.sub3}',  
	 f'MeONO{con.sub2}',  
	 'isoprene nitrate',  
	 f'MeCHO {con.to} MeOO',  
         'propanal',  
	 f'NO{con.sub3}', 
	 f'H{con.sub2}O', 
	 'HOBr',  
	 'HOCl', 
	 f'HONO{con.sub2}', 
	 f'HO{con.sub2}NO{con.sub2}', 
	 f'H{con.sub2}O{con.sub2}', 
	 'MeOOH',  
	 f'O{con.sub2}', 
	 f'O{con.sub3}',  
	 f'N{con.sub2}O', 
	 'methacrolein', 
	 'MACROOH', 
	 f'MeCHO {con.to} CH{con.sub4}', 
	 'NO',  
	 f'NO{con.sub2}',  
	]

# Pick a region to average over.
# Altitude. Free troposphere. ~ 0 to 10 km above sea-level.
alt_min, alt_max = 0, 39
# Altitude. Stratosphere. ~ 25 to 40 km above sea-level.
#alt_min, alt_max = 60, 73
# Latitude. The equator. ~ 20N to 20S.
lat_min, lat_max = -20, 20
# Latitude. The South pole.
#lat_min, lat_max = -90, -60 
inputs, target_data = get_area(input_data, target_data, alt_min, alt_max, lat_min, lat_max)
inputs, pred_data = get_area(input_data, pred_data, alt_min, alt_max, lat_min, lat_max)
#inputs = input_data
print(inputs.shape, target_data.shape, pred_data.shape)

# For every J rate...
for i in range(len(target_data[0])):

  # Get targets and preds.
  targets = target_data[:, i] 
  preds = pred_data[:, i]
  
  # Get its name.
  name = names[i]
  #print(name) 
  title = f'{name} photolysis rate coefficient in a 1-year simulation'
  
  # Put R2 score on plot.
  r2 = round(r2_score(targets, preds), 3)
  print(r2)
  if r2 >= 0:
    title += f'\nCorrelation coefficient = {r2}' 
  else:
    title += f'\nCorrelation coefficient < 0'

  # Plot by day.
  time = inputs[:, 0]

  # Average the data in this region.
  targets_x, targets_y = avg_data(time, targets)
  preds_x, preds_y = avg_data(time, preds)

  # Plot.
  plt.figure(figsize=(15,7))
  plt.plot(targets_x, targets_y, label='Fast-JX')
  plt.plot(preds_x, preds_y, label='Offline random forest emulation')
  plt.legend()
  plt.xlabel('Day of year')
  plt.ylabel(f'Mean tropospheric, equatorial {name} photolysis rate coefficient / {con.pers}')
  plt.title(title)
  #plt.show()
  plt.savefig(f'{fig_path}/{name}.png')
  plt.close()    
