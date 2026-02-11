'''
Experiment with random forest design specifically 
to improve problematic, small tropospheric J rates.
'''

import time
import numpy as np
import constants as con
import functions as fns
import file_paths as paths
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor


def avg_data(x, y):
  # Average y data by unique x values.
  xs = np.unique(x)
  y_avg = [y[x == val].mean() for val in xs]
  return xs, y_avg


exp = 'Lvl cutoffs'
print(f'\n{exp}')

data_path = f'{paths.npy}/1982_45m.npy'
print('\nLoading data from', data_path)
data = np.load(data_path)
print(data.shape)

# Problematic rxns.
names = ['OCS', 'ISON', 'H2O', 'O2', 'N2O', 'MeCHO -> CH4', 'NO']
js = [26, 30, 34, 41, 43, 46, 47]

'''
# See where the problematic rxns start.
lvls = list(range(1, 86))
for i in range(len(js)):
  name = names[i]
  j = js[i]
  field = data[j]
  print(f'\n{name} levels:')
  for lvl in lvls:
    field_lvl = field[data[2] == lvl]
    low = min(field_lvl)
    med = np.median(field_lvl) 
    high = max(field_lvl)
    print(f'Level {lvl}: min = {low}, median = {med}, max = {high}')

  # See the range in which most of the data lie.
  print(f'\n{name} percentiles:')
  for i in range(0, 100, 5):
    top = i+5
    portion = np.percentile(field, top)
    print(f'{top}th percentile = {portion}')
  
exit()
'''
# Indices of 1982 training data in full npy datatset.
inputs_idx = [0,1,2,4,5,9,10,11,8,12]
targets_idx = [17,18,19,23,26,27,28] + list(range(30,49))

# Get inputs and targets.
inputs, targets = fns.in_out_swap(data, inputs_idx, targets_idx)  
  
# 95/5 train test split.  
in_train, in_test, out_train, out_test, i_test = fns.tts(inputs, targets, 0.05)

# Problematic J rates.
js = [4, 7, 11, 18, 20, 23, 24] 

# Set stratospheric J rates to 0 below their important heights.
#lvls = [61, 56, 56, 51, 50, 62, 55] # Lenient.
#lvls = [60, 60, 60, 60, 60, 60, 60] # Simple.
#lvls = [67, 64, 67, 65, 63, 68, 66] # Strict.
lvls = [64, 61, 70, 61, 59, 61, 64] # Balanced.

# Set problem J rates to 0 when the values are negligible.
mags = [1e-5, 1e-4, 1e-6, 1e-9, 1e-7, 1e-6, 1e-6]

'''
print('Setting small values to 0.')
for i in range(len(js)):
  j = js[i]
  lvl = lvls[i]
  mag = mags[i]
  # Lower altitudes.
  out_train[in_train[:, 2] < lvl, j] = 0.0
  # Negligible values.
  out_train[out_train[:, j] < mag, j] = 0.0
'''

# Make the regression model.
loss = 'squared_error'
trees = 20
samples = 0.1
leaves = 100000
features = 0.2
print(f'Loss fn = {loss}, trees = {trees}, samples = {samples},\n\
        leaves = {leaves}, features = {features}')
model = RandomForestRegressor(n_estimators=trees, n_jobs=-1, max_features=features, 
                              max_samples=samples, max_leaf_nodes=leaves, random_state=con.seed,
			      criterion=loss)

start = time.time()
# Train the model.
print('Training the model.')
model.fit(in_train, out_train)
end = time.time()
elapsed = round(end-start)
print(elapsed, 'seconds.')

# Test the model.
print('Testing the model.')
out_pred, _, _, _, _, r2 = fns.test(model, in_test, out_test)
print(f'Overall average R2 score for all J rates: {r2}') 

# Select problematic J rates: OCS, ISON, H2O, O2, N2O, MeCHO -> CH4, NO.
js = [4, 7, 11, 18, 20, 23, 24]
names = ['OCS', 'ISON', 'H2O', 'O2', 'N2O', 'MeCHO -> CH4', 'NO']
'''
# Set dodgy tropospheric rates to 0.
for i in range(len(js)):
  j = js[i]
  lvl = lvls[i]
  # Lower altitudes.
  out_pred[in_test[:, 2] < lvl, j] = 0.0
'''
'''
# Select test area.
print('Selecting test area model levels.')
alt_max = 75
out_test = out_test[in_test[:, 2] < alt_max]
out_pred = out_pred[in_test[:, 2] < alt_max]
in_test = in_test[in_test[:, 2] < alt_max]
'''
print('Making plots.')
'''
# Get R2 score of these problem targets
# and plot vertical column averages.
times = in_test[:, 0]
for i in range(len(js)):
  j = js[i]
  name = names[i]
  for lvl in range(50, 71):
    preds = out_pred.copy()
    preds[in_test[:, 2] < lvl, j] = 0.0
    targets = out_test[:, j]
    preds = preds[:, j]
    r2 = round(r2_score(targets, preds), 3)
    time, targets = avg_data(times, targets)
    time, preds = avg_data(times, preds)
    plt.plot(time, targets, label='Targets')
    plt.plot(time, preds, label='Preds')
    plt.title(f'{name} below model level 75 with L{lvl} mask. {con.r2} = {r2}')
    plt.legend()
    plt.xlabel(f'Day of year')
    plt.ylabel(f'J rate')
    plt.show()
    #plt.savefig(f'{i}.png')
    plt.close()
'''    

# Get R2 score of these problem targets
# and plot vertical column averages.
for i in range(len(js)):
  j = js[i]
  name = names[i]
  print(f'\n{name}')
  for lvl in range(68, 71):
    preds = out_pred.copy()
    targets = out_test.copy()
    inputs = in_test.copy()
    # Select test area 5 lvls above cutoff.
    alt_max = lvl + 5
    targets = targets[inputs[:, 2] < alt_max]
    preds = preds[inputs[:, 2] < alt_max]
    inputs = inputs[inputs[:, 2] < alt_max]
    times = inputs[:, 0]
    # Set low alt preds to 0.
    preds[inputs[:, 2] < lvl, j] = 0.0
    targets = targets[:, j]
    preds = preds[:, j]
    r2 = round(r2_score(targets, preds), 3)
    print(r2)
    '''
    time, targets = avg_data(times, targets)
    time, preds = avg_data(times, preds)
    plt.plot(time, targets, label='Targets')
    plt.plot(time, preds, label='Preds')
    plt.title(f'{name} below model level {alt_max} with L{lvl} mask. {con.r2} = {r2}')
    plt.legend()
    plt.xlabel(f'Day of year')
    plt.ylabel(f'J rate')
    plt.show()
    #plt.savefig(f'{i}.png')
    plt.close()    
    '''
print(f'\n{exp}\n')
