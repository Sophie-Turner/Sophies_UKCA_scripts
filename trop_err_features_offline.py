'''
Investigate which features correlate to tropospheric
overestimation of J rates.
'''

import numpy as np
import constants as con
import functions as fns
import file_paths as paths
import matplotlib.pyplot as plt


def plot_data(x, y):
  # Average y data by unique x values.
  xs = np.unique(x)
  # Unless there are too many values of x, i.e. continuous data.
  if len(xs) < 1000:
    plot_type = plt.plot
    alpha = 1
    #ys = [y[x == val].mean() for val in xs]
    ys = np.array([np.median(y[x == val]) for val in xs])
  else:
    plot_type = plt.scatter
    alpha = 0.1
    # Cut the data down to a plottable size.
    if len(x) > 10000:
      idx = con.rng.choice(len(x), size=10000, replace=False)
      xs, ys = x[idx], y[idx]
    else:
      xs, ys = x, y
  return xs, ys, plot_type, alpha


model_name = 'rf_L70_1982'
print()
print(model_name)
  
# Load the model data.
inputs, targets, preds = fns.load_model_data(model_name)

# Names, info & units of outputs in dataset order.
j = 'photolysis rate coefficient' 
mmr = 'mass mixing ratio'
pers = f'/ {con.pers}' 
kg = f'/ kg kg{con.supminus}{con.sup1}'
mols = f'/ mol {con.pers}' 
du = '/ DU'

names_in = [
         ['day of year', '', ''],
	 ['hour of day', '', ''],
	 ['model level', '', ''],
	 ['hybrid height', '', '/ proportion of top'],
	 ['latitude', '', '/ deg N'],
	 ['longitude', '', '/ deg E'],
         ['humidity', '', ''],
	 ['cloud fraction', '', ''],
	 ['solar zenith angle', '', '/ cos radians'],
	 ['upward shortwave flux', '', f'/ {con.Wperm2}'],
	 ['downward shortwave flux', '', f'/ {con.Wperm2}'],
	 ['pressure', '', '/ Pa'],
	 ['temperature', '', '/ K']
	 ]
names_out = [
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

# Troposphere. ~ 0 to 10 km above sea-level.
#lvl_max = 70
#targets = targets[inputs[:,2] < lvl_max] 
#preds = preds[inputs[:,2] < lvl_max] 
#inputs = inputs[inputs[:,2] < lvl_max]

# Problematic J rates to test.
problems = [0, 4, 7, 11, 18, 20, 23, 24]
for i in problems:
#for i in range(len(targets[0])):
  target, pred = targets[:, i], preds[:, i] 
  J_name = names_out[i] 
  
  # Spatial and temporal features.
  for j in range(3):
    feature = inputs[:, j] 
    feature_name = names_in[j]
    # Average y by x.
    features, target_plot, plot_type, a = plot_data(feature, target)
    features, pred_plot, plot_type, a = plot_data(feature, pred)
    # Make plot.
    plt.figure()
    plot_type(features, pred_plot, label=f'L70 random forest', alpha=a)
    plot_type(features, target_plot, label=f'UKCA', alpha=a)
    plt.title(f'Median {J_name[0]} {J_name[1]} by {feature_name[0]}')
    plt.xlabel(f'{feature_name[0]} {feature_name[1]} {feature_name[2]}')  
    plt.ylabel(f'{J_name[0]} {J_name[1]} {J_name[2]}')
    plt.legend()
    plt.show()
    plt.close()
  
  # Cut the data down to a plottable size.
  idx = con.rng.choice(targets.shape[0], size=10000, replace=False)
  target = target[idx]
  pred = pred[idx]
  
  # Correlation plot.
  plt.scatter(target, pred, alpha=0.1)
  plt.title(f'Targets and predictions of {J_name[0]} {J_name[1]}')
  plt.xlabel(f'{J_name[0]} {J_name[1]} {J_name[2]} from UKCA')
  plt.ylabel(f'{J_name[0]} {J_name[1]} {J_name[2]} from L70 random forest')
  plt.show()
  plt.close()  
  
