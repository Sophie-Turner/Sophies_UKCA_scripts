'''
Compare the effect on chemistry of random forest photolysis 
vs the original Fast-JX photolysis scheme in UKCA.
'''

import numpy as np
import constants as con
import functions as fns
import file_paths as paths
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Experiment.
run = '1'

# Fast-JX dataset (control).
path_fj = f'{paths.npy}/fj{run}yr.npy'

# ML-predicted photolysis dataset.
path_ml = f'{paths.npy}/ml{run}yr.npy'

# Names, info & units of outputs in dataset order.
j = 'photolysis rate coefficient' 
mmr = 'mass mixing ratio'
pers = f'/ {con.pers}' 
kg = f'/ kg kg{con.supminus}{con.sup1}'
mols = f'/ mol {con.pers}' 
du = '/ DU'

names = [['HCHO (radical reaction)', j, pers], 
         ['HCHO (molecular reaction)', j, pers],
	 ['MeCOCHO', j, pers], 
	 [f'Cl{con.sub2}O{con.sub2}', j, pers],
         ['OCS', f'SO{con.sub3}', j, pers], 
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
	 [f'NO{con.sub2}', j, pers], 
	 ['Shortwave heating rates', '', ''], 
	 [f'O{con.sub3}', mmr, kg], 
	 ['NO', mmr, kg], 
	 [f'NO{con.sub2}', mmr, kg],
	 ['Peroxyacetyl nitrate', mmr, kg], 
	 ['Cl', mmr, kg], 
	 [f'N{con.sub2}O', mmr, kg], 
	 ['OH', mmr, kg], 
	 [f'HO{con.sub2}', mmr, kg], 
	 [f'H{con.sub2}O', mmr, kg], 
	 [f'O{con.subx} production', '', mols], 
	 [f'CH{con.sub4} lifetime', '', mols], 
	 [f'O{con.sub3} column', '', du]]

# Load both datasets.
print('Loading data.')
data_fj = np.load(path_fj)
data_ml = np.load(path_ml)
print(data_fj.shape, data_ml.shape)

# For every output...
for i in range(len(data_fj)):
  item_fj = data_fj[i]
  item_ml = data_ml[i]
  print(item_fj.shape, item_ml.shape)
  
  # Get its name.
  fullname = names[i]
  print(fullname[0])

  title = f'{fullname[0]} {fullname[1]} correlation from UM using original and emulated photolysis'
  xlab = f'{fullname[0]} {fullname[1]} {fullname[2]} from UM using Fast-JX photolysis'
  ylab = f'{fullname[0]} {fullname[1]} {fullname[2]} from UM using random forest photolysis'

  # Get metrics.
  # Put R2 score on plot if it's a J rate.
  if fullname[1] == j:
    r2 = round(r2_score(item_fj, item_ml), 3)
    title += f'\nCorrelation coefficient = {r2}' 
  # Put quartiles on plot instead if it's not a J rate.
  else:
    low_fj, med_fj, high_fj = round(np.percentile(item_fj, [25, 50, 75]), 5)
    low_ml, med_ml, high_ml = round(np.percentile(item_ml, [25, 50, 75]), 5)
    xlab += f'\nLower quartile = {low_fj}, Median = {med_fj}, Upper quartile = {high_fj}'
    ylab += f'\nLower quartile = {low_ml}, Median = {med_ml}, Upper quartile = {high_ml}'

  # Make a correlation plot of ML and Fast-JX.
  # Don't plot >10000 points.
  item_fj, item_ml, a = fns.shrink(item_fj, item_ml)
  print(item_fj.shape, item_ml.shape)
  plt.scatter(item_fj, item_ml, alpha=a)
  # Make a 1:1 reference line.
  plt.plot(item_fj, item_fj, linestyle=':', color='grey', alpha=0.5)
  # Force axes to be identical.
  fns.force_axes()
  plt.title(title)
  plt.xlabel(xlab)
  plt.ylabel(ylab)
  plt.show() 
  plt.close()
  
  # Make a plot of Fast-JX, difference between ML & Fast-JX, 
  # and CMIP6 standard deviation, by time and latitude.
  
  # Make a plot of CMIP6 standard deviation, Fast-J and ML values over time.
  # Put margins around the CMIP6 line.    
