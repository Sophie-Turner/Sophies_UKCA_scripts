'''
Compare the effect on chemistry of random forest photolysis 
vs the original Fast-JX photolysis scheme online in UKCA.
'''

import numpy as np
import datetime as dt
import constants as con
import functions as fns
import file_paths as paths
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error


def round_digits(num, sig_figs=3):
  '''Round by significant figures instead of decimal places.
     better for different magnitudes of values.'''
  if num != 0:
    places = int(sig_figs - np.floor(np.log10(abs(num))))
    num = round(num, places)
  return num
  
  
def get_area(data, alt_min, alt_max, lat_min, lat_max):
  # Get an area to average over.
  data = data[:, (data[3] > alt_min) & (data[3] < alt_max)] # 3 is hybrid height.
  data = data[:, (data[4] > lat_min) & (data[4] < lat_max)] # 4 is latitude.    
  return data
  
  
def avg_data(x, y):
  # Average y data by unique x values.
  xs = np.unique(x)
  y_avg = [y[x == val].mean() for val in xs]
  return xs, y_avg

# Experiment.
run = '1'

# Fast-JX dataset (control).
path_fj = f'{paths.npy}/fj{run}yr.npy'

# ML-predicted photolysis dataset.
path_ml = f'{paths.npy}/ml{run}yr_masked.npy'

# Where to save the plots.
exp = ''
fig_path = f'{paths.analysis}/{exp}'

# Names, info & units of outputs in dataset order.
j = 'photolysis rate coefficient' 
mmr = 'mass mixing ratio'
pers = f'/ {con.pers}' 
kg = f'/ kg kg{con.supminus}{con.sup1}'
mols = f'/ mol {con.pers}' 
du = '/ DU'
ks = f'/ K {con.pers}'

names = [
         ['Shortwave heating rates', '', ks], 
	 [f'O{con.sub3}', mmr, kg], 
	 ['NO', mmr, kg], 
	 ['Peroxyacetyl nitrate', mmr, kg], 
	 ['Cl', mmr, kg], 
	 [f'N{con.sub2}O', mmr, kg], 
	 ['OH', mmr, kg], 
	 [f'HO{con.sub2}', mmr, kg], 
	 [f'H{con.sub2}O', mmr, kg], 
	 [f'O{con.subx} production', '', mols], 
	 [f'CH{con.sub4} + OH reaction flux', '', mols], 
	 [f'O{con.sub3} column', '', du],
	 ['HCHO (radical reaction)', j, pers], 
         ['HCHO (molecular reaction)', j, pers],
	 ['MeCOCHO', j, pers], 
	 [f'Cl{con.sub2}O{con.sub2}', j, pers],
         ['OCS', j, pers], 
	 [f'SO{con.sub3}', j, pers], 
	 [f'MeONO{con.sub2}', j, pers], 
	 ['Isoprene nitrate', j, pers], 
	 [f'MeCHO {con.to} MeOO', j, pers], 
         ['Propanal', j, pers], 
	 [f'NO{con.sub3}', j, pers], 
	 [f'H{con.sub2}O', j, pers], 
	 ['HOBr', j, pers], 
	 ['HOCl', j, pers], 
	 [f'HONO{con.sub2}', j, pers], 
	 [f'HO{con.sub2}NO{con.sub2}', j, pers], 
	 [f'H{con.sub2}O{con.sub2}', j, pers], 
	 ['MeOOH', j, pers], 
	 [f'O{con.sub2}', j, pers],
	 [f'O{con.sub3}', j, pers], 
	 [f'N{con.sub2}O', j, pers], 
	 ['Methacrolein', j, pers], 
	 ['MACROOH', j, pers], 
	 [f'MeCHO {con.to} CH{con.sub4}', j, pers],
	 ['NO', j, pers], 
	 [f'NO{con.sub2}', j, pers] 
	]

# Load both datasets.
print('\nLoading data.')
#print(exp)
data_fj = np.load(path_fj)
data_ml = np.load(path_ml)
print(data_fj.shape, data_ml.shape)

# All levels.
#alt_min, alt_max = 0, 1
# Troposphere. ~ 0 to 10 km above sea-level.
alt_min, alt_max = 0, 0.121
# Mid stratosphere. 25 to 40 km.
#alt_min, alt_max = 0.29, 0.47
# Surface. 0-1 km.
#alt_min, alt_max = 0, 0.0018 

# Whole globe.
#lat_min, lat_max = -90, 90
# The equator. ~ 20N to 20S.
#lat_min, lat_max = -20, 20
# South pole. 
lat_min, lat_max = -90, -60

data_fj = get_area(data_fj, alt_min, alt_max, lat_min, lat_max)
data_ml = get_area(data_ml, alt_min, alt_max, lat_min, lat_max)
print(data_fj.shape, data_ml.shape)

# Print accuracy metrics.
print('\nR2 scores:')
for i in range(18, len(data_fj)):
  item_fj = data_fj[i]
  item_ml = data_ml[i]
  r2 = r2_score(item_fj, item_ml)
  print(r2)
 
print('\nMSE:')
for i in range(18, len(data_fj)):
  item_fj = data_fj[i]
  item_ml = data_ml[i]
  mse = mean_squared_error(item_fj, item_ml)
  print(mse)  
'''  
print('\nMean relative error:')
for i in range(18, len(data_fj)):
  item_fj = data_fj[i]
  item_ml = data_ml[i]
  mask = item_fj != 0
  rel_err = (np.abs(item_ml[mask] - item_fj[mask]) / np.abs(item_fj)[mask]) * 100
  #rel_err = ((item_ml - item_fj) / item_fj) * 100
  rel_err_mean = np.nanmean(rel_err)
  print(rel_err_mean)   
  
print('\nMaximum relative error:')
for i in range(18, len(data_fj)):
  item_fj = data_fj[i]
  item_ml = data_ml[i]
  mask = item_fj != 0
  rel_err = (np.abs(item_ml[mask] - item_fj[mask]) / np.abs(item_fj[mask])) * 100
  rel_err_max = np.max(rel_err)
  print(rel_err_max)  
  
print('\nMAPE:')
for i in range(18, len(data_fj)):
  item_fj = data_fj[i]
  item_ml = data_ml[i]
  mape = mean_absolute_percentage_error(item_fj, item_ml)
  print(mape)    

print('\nSMAPE:')
for i in range(18, len(data_fj)):
  item_fj = data_fj[i]
  item_ml = data_ml[i]
  smape = fns.sMAPE(item_fj, item_ml)
  print(smape)  

# For every output, make a correlation plot of ML and Fast-JX.
for i in range(6, len(data_fj)):
  item_fj = data_fj[i].squeeze()
  item_ml = data_ml[i].squeeze()
  
  # Get its name.
  fullname = names[i-6]
  print(fullname[0])
  
  # Text for plot.
  title = f'{fullname[0]} {fullname[1]} correlation from UM using original and emulated photolysis'
  xlab = f'{fullname[0]} {fullname[1]} {fullname[2]} from UM using Fast-JX photolysis'
  ylab = f'{fullname[0]} {fullname[1]} {fullname[2]} from UM using random forest photolysis'

  # Put R2 score on plot if it's a J rate.
  if fullname[1] == j:
    r2 = r2_score(item_fj, item_ml)
    if r2 > 0:
      r2 = round_digits(r2)
      title += f'\nCorrelation coefficient = {r2}' 

  # Don't plot >10000 points.
  item_fj, item_ml, a = fns.shrink(item_fj, item_ml)
  fig, ax = plt.subplots(figsize=(10,7))
  ax.scatter(item_fj, item_ml, alpha=a)
  # Make a 1:1 reference line.
  ax.plot(item_fj, item_fj, linestyle=':', color='grey', alpha=0.5)
  
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.set_xlim(left=0)
  ax.set_ylim(bottom=0)
  
  ax.set_title(title)
  ax.set_xlabel(xlab)
  ax.set_ylabel(ylab)
  plt.show()
  #plt.savefig(f'{fig_path}/{fullname[0]}_correlation.png')
  plt.close()

# For every output, 
# make a plot of Fast-JX, difference between ML & Fast-JX, 
# and CMIP6 standard deviation, by time and latitude.

# For every output,  
# make a plot of Fast-J and ML values over time. 
for i in range(17, len(data_fj)):
  item_fj = data_area_fj[i]
  item_ml = data_area_ml[i]
  
  # Get its name.
  fullname = names[i-6]
  print(fullname[0]) 
  title = f'Monthly mean {fullname[0]} {fullname[1]} in a 30-year simulation'
  
  # Put R2 score on plot if it's a J rate.
  label_ml = '' 
  if fullname[1] == j:
    r2_ml = round(r2_score(item_fj, item_ml), 3)
    if r2_ml >= 0:
      label_ml = f' Correlation coefficient = {r2_ml}'
  
  # Get day (hourly or daily datasets).
  time_fj = data_area_fj[0]
  time_ml = data_area_ml[0]
  
  # Get years and months as dates (monthly datasets).
  years_fj, years_ml = data_area_fj[0].astype(int), data_area_ml[0].astype(int)
  months_fj, months_ml = data_area_fj[1].astype(int), data_area_ml[1].astype(int)
  time_fj = np.array([dt.datetime(y,m,1) for y,m in zip(years_fj, months_fj)])
  time_ml = np.array([dt.datetime(y,m,1) for y,m in zip(years_ml, months_ml)])
  
  # Get average FJ and ML by time.
  times_fj, item_avg_fj = avg_data(time_fj, item_fj)
  times_ml, item_avg_ml = avg_data(time_ml, item_ml)
  
  # Make plot.
  fig, ax = plt.subplots(figsize=(15,7))
  ax.plot(times_fj, item_avg_fj, label=f'UM with Fast-JX photolysis.')
  ax.plot(times_ml, item_avg_ml, label=f'UM with random forest photolysis.')
  
  # Time axis for monthly data.
  ax.set_xlabel('Simulated date')
  
  # Time axis for daily data.
  #ax.set_xlabel('Simulated day of year')
  
  # Format plot.
  ax.set_title(title)
  ax.set_ylabel(f'{fullname[0]} {fullname[1]} {fullname[2]}')
  ax.legend()
  plt.savefig(f'{fig_path}/{fullname[0]} {fullname[1]}.png')
  plt.close()
  
  # Put margins around the CMIP6 line.      
'''
