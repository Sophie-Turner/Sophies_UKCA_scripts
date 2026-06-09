'''
Investigate which features correlate to tropospheric
overestimation of J rates and their related chemistry
diagnostics from online UM tests.
'''

import numpy as np
import constants as con
import functions as fns
import file_paths as paths
import matplotlib.pyplot as plt


def avg_data(x, y):
  # Average y data by unique x values.
  xs = np.unique(x)
  y_avg = [y[x == val].mean() for val in xs] # Mean.
  #y_avg = np.array([np.median(y[x == val]) for val in xs]) # Median.
  return xs, y_avg


# Fast-JX dataset (control).
#path_fj = f'{paths.data}/fj1yr/19820102.npy'
path_fj = f'{paths.npy}/fj1mon.npy'
# ML dataset.
#path_ml = f'{paths.data}/ml1yr/19820102.npy'
path_ml = f'{paths.npy}/ml1mon.npy'
# ML dataset, masked.
#path_mask = f'{paths.data}/ml1yr_masked/19820102.npy'
path_mask = f'{paths.npy}/ml1mon_masked_tuned.npy'

# Names, info & units of outputs in dataset order.
j = 'photolysis rate coefficient' 
mmr = 'mass mixing ratio'
pers = f'/ {con.pers}' 
kg = f'/ kg kg{con.supminus}{con.sup1}'
mols = f'/ mol {con.pers}' 
du = '/ DU'

names = [
         ['day of year', '', ''],
	 ['hour of day', '', ''],
	 ['model level', '', ''],
	 ['hybrid height', '', '/ proportion of top'],
	 ['latitude', '', '/ deg N'],
	 ['longitude', '', '/ deg E'],            # 5
         ['Shortwave heating rates', '', ''], 
	 [f'O{con.sub3}', mmr, kg], 
	 ['NO', mmr, kg], 
	 ['Peroxyacetyl nitrate', mmr, kg], 
	 ['Cl', mmr, kg],                          # 10
	 [f'N{con.sub2}O', mmr, kg], 
	 ['OH', mmr, kg], 
	 [f'HO{con.sub2}', mmr, kg], 
	 [f'H{con.sub2}O', mmr, kg], 
	 [f'O{con.subx} production', '', mols],   # 15
	 [f'CH{con.sub4} + OH reaction flux', '', mols], 
	 [f'O{con.sub3} column', '', du],
	 ['HCHO (radical reaction)', j, pers], 
         ['HCHO (molecular reaction)', j, pers],
	 ['MeCOCHO', j, pers],                     # 20 
	 [f'Cl{con.sub2}O{con.sub2}', j, pers],
         ['OCS', j, pers], 
	 [f'SO{con.sub3}', j, pers], 
	 [f'MeONO{con.sub2}', j, pers], 
	 ['Isoprene nitrate', j, pers],           # 25
	 [f'MeCHO {con.to} MeOO', j, pers], 
         ['Propanal', j, pers], 
	 [f'NO{con.sub3}', j, pers], 
	 [f'H{con.sub2}O', j, pers], 
	 ['HOBr', j, pers],                        # 30
	 ['HOCl', j, pers], 
	 [f'HONO{con.sub2}', j, pers], 
	 [f'HO{con.sub2}NO{con.sub2}', j, pers], 
	 [f'H{con.sub2}O{con.sub2}', j, pers], 
	 ['MeOOH', j, pers],                      # 35
	 [f'O{con.sub2}', j, pers],
	 [f'O{con.sub3}', j, pers], 
	 [f'N{con.sub2}O', j, pers], 
	 ['Methacrolein', j, pers], 
	 ['MACROOH', j, pers],                     # 40
	 [f'MeCHO {con.to} CH{con.sub4}', j, pers],
	 ['NO', j, pers], 
	 [f'NO{con.sub2}', j, pers]                # 43
	]

# Load both datasets.
print('\nLoading data from', path_mask)
data_fj = np.load(path_fj)
data_ml = np.load(path_ml)
data_mask = np.load(path_mask)
print(data_fj.shape, data_ml.shape, data_mask.shape)

# Lower model levels.
lvl_max = 70
data_fj = data_fj[:, (data_fj[2] < lvl_max)] # 2 is model level.
data_ml = data_ml[:, (data_ml[2] < lvl_max)]
data_mask = data_mask[:, (data_mask[2] < lvl_max)]

# Sample the data down to a plottable size.
idx_fj = con.rng.choice(data_fj.shape[1], size=10000, replace=False)
idx_ml = con.rng.choice(data_ml.shape[1], size=10000, replace=False)
idx_fj = np.sort(idx_fj)
idx_ml = np.sort(idx_ml)
data_fj_sample = data_fj[:, idx_fj]
data_ml_sample = data_ml[:, idx_ml]

# Problematic items to test.
problems = [7, 12, 13, 15, 16, 17, 22, 23, 25, 29, 36, 37, 38, 41, 42]
#problems = [12, 13, 29]
for i in problems:
  item_name = names[i] 
  
  # Spatial and temporal features.
  for j in range(4):
    feature_name = names[j]
    
    # Line plots of averages.
    item_fj, item_ml, item_mask = data_fj[i], data_ml[i], data_mask[i]
    feature_fj, feature_ml, feature_mask = data_fj[j], data_ml[j], data_mask[j]
    # Average y by x.
    features_fj, item_avg_fj = avg_data(feature_fj, item_fj)
    features_ml, item_avg_ml = avg_data(feature_ml, item_ml)
    features_mask, item_avg_mask = avg_data(feature_mask, item_mask)
    # Make plot.
    fig, ax = plt.subplots()
    ax.plot(features_ml, item_avg_ml, color='tab:orange', label=f'UM with random forest photolysis.')
    ax.plot(features_mask, item_avg_mask, color='tab:pink', label=f'UM with random forest photolysis, masked.')
    ax.plot(features_fj, item_avg_fj, color='tab:blue', label=f'UM with Fast-JX photolysis.')
    ax.set_title(f'Mean {item_name[0]} {item_name[1]} by {feature_name[0]} in a 1-month simulation')
    ax.set_xlabel(f'{feature_name[0]} {feature_name[1]} {feature_name[2]}')  
    ax.set_ylabel(f'{item_name[0]} {item_name[1]} {item_name[2]}')
    ax.legend()
    plt.show()
    #plt.savefig(f'/scratch/st838/netscratch/analysis/online_masked_tuned_global/{item_name[0]} {feature_name[0]} 3.png')
    plt.close()
    
    # Scatter plots of instantaneous points.
    item_fj, item_ml = data_fj_sample[i], data_ml_sample[i] 
    feature_fj, feature_ml = data_fj_sample[j], data_ml_sample[j] 
    fig, ax = plt.subplots()
    ax.scatter(feature_ml, item_ml, color='tab:orange', label=f'UM with random forest photolysis.', alpha=0.1)
    ax.scatter(feature_fj, item_fj, color='tab:blue', label=f'UM with Fast-JX photolysis.', alpha=0.1)
    ax.set_title(f'{item_name[0]} {item_name[1]} by {feature_name[0]} in a 1-month simulation')
    ax.set_xlabel(f'{feature_name[0]} {feature_name[1]} {feature_name[2]}')  
    ax.set_ylabel(f'{item_name[0]} {item_name[1]} {item_name[2]}')
    ax.legend()
    plt.show()
    plt.close()
    
  '''
  # Correlation plot.
  plt.scatter(item_fj, item_ml, alpha=0.1)
  plt.title(f'Targets and predictions of {item_name[0]} {item_name[1]}')
  plt.xlabel(f'{item_name[0]} {item_name[1]} {item_name[2]} from UKCA')
  plt.ylabel(f'{item_name[0]} {item_name[1]} {item_name[2]} from random forest')
  plt.show()
  plt.close()
  '''
