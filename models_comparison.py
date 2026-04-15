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
from sklearn import linear_model
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler as scaler
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import HistGradientBoostingRegressor as hgbr

print('\nWith input scaling.')

# Names, info & units of data of interest.
j = 'photolysis rate coefficient' 
pers = f'/ {con.pers}' 

names = [[f'NO{con.sub3}', j, pers], 
         [f'H{con.sub2}O', j, pers]]

idxs = [10, 11]

  
def train_test(model, in_train, out_train, in_test):
  # Train.
  print('Training model.')
  model.fit(in_train, out_train)
  # Make preds.
  print('Testing model.')
  preds = model.predict(in_test)
  return model, preds
  
  
def avg_data(x, y):
  # Average y data by unique x values.
  xs = np.unique(x)
  y_avg = [y[x == val].mean() for val in xs]
  return xs, y_avg
  

def print_metrics(out_test, preds):
  r2 = r2_score(out_test, preds)
  mse = mean_squared_error(out_test, preds)
  print(f'R2 = {r2}')
  print(f'MSE = {mse}')
  
  
def get_metrics(in_test, out_test, preds):
  # Metrics for all.
  print('\nAll:')
  print_metrics(out_test, preds)
  # Troposphere. ~ 0 to 18 km above sea-level.
  alt_min, alt_max = 0, 0.213
  out_test_portion = out_test[(in_test[:, 3] > alt_min) & (in_test[:, 3] < alt_max)] 
  preds_portion = preds[(in_test[:, 3] > alt_min) & (in_test[:, 3] < alt_max)] 
  print('\nTroposphere:')
  print_metrics(out_test_portion, preds_portion)
  # Stratosphere. ~> 10 km above sea-level.
  alt_min = 0.121
  out_test_portion = out_test[(in_test[:, 3] > alt_min)] 
  preds_portion = preds[(in_test[:, 3] > alt_min)] 
  print('\nStratosphere:')
  print_metrics(out_test_portion, preds_portion)
  # South pole. 
  lat_min, lat_max = -90, -60
  out_test_portion = out_test[(in_test[:, 4] > lat_min) & (in_test[:, 4] < lat_max)] 
  preds_portion = preds[(in_test[:, 4] > lat_min) & (in_test[:, 4] < lat_max)]
  print('\nSouth pole:')
  print_metrics(out_test_portion, preds_portion) 
  # Equator. 
  lat_min, lat_max = -1, 1
  out_test_portion = out_test[(in_test[:, 4] > lat_min) & (in_test[:, 4] < lat_max)] 
  preds_portion = preds[(in_test[:, 4] > lat_min) & (in_test[:, 4] < lat_max)]
  print('\nEquator:')
  print_metrics(out_test_portion, preds_portion) 
  # NO2 and NO3.
  for i in range(2):
    name = names[i]
    item_fj = out_test[:, idxs[i]]
    item_ml = preds[:, idxs[i]]
    print(f'\n{name[0]}:')
    print_metrics(item_fj, item_ml)


def plot_timeseries(in_test, out_test, preds, model_name):
  for i in range(2):
    name = names[i]
    item_fj = out_test[:, idxs[i]].squeeze()
    item_ml = preds[:, idxs[i]].squeeze()
    lvl = in_test[:, 2].squeeze()
    
    # Get average FJ and ML by lvl.
    lvls, item_avg_fj = avg_data(lvl, item_fj)
    lvls, item_avg_ml = avg_data(lvl, item_ml)
    
    # Make plot.
    fig, ax = plt.subplots()
    ax.plot(item_avg_fj, lvls, label='UKCA')
    ax.plot(item_avg_ml, lvls, label=model_name)
    
    # Format plot.
    ax.set_title(f'Mean {name[0]} {name[1]} by model level in a 1-year {model_name} test')
    ax.set_ylabel('Vertical model level')
    ax.set_xlabel(f'Mean {name[0]} {name[1]} {name[2]}')
    ax.legend()
    plt.show()
    plt.close()
    
    
def save_model_data(model, in_test, out_test, preds, name):
  print('Saving the model and its data.')
  joblib.dump(model, f'{paths.mod}/{name}/{name}.pkl')
  np.save(f'{paths.mod}/{name}/{name}_inputs.npy', in_test)
  np.save(f'{paths.mod}/{name}/{name}_targets.npy', out_test)
  np.save(f'{paths.mod}/{name}/{name}_pred.npy', preds)

	
# Dataset.	
data_path = f'{paths.npy}/1982_45m.npy'
print('\nLoading data from', data_path)
data = np.load(data_path)
print(data.shape)	
	
# Reduce data size.
data = fns.sample(data, 15_000_000)	
	
# Inputs and targets.
inputs_idx = list(range(13))
#targets_idx = [17,18,19,23,26,27,28] + list(range(30,49))
targets_idx = [17,26,33,34,39,43,48]
inputs, targets = fns.in_out_swap(data, inputs_idx, targets_idx)

# Free up some memory.
del data

# 98/2 train test split.
print('Making train-test split.') 
in_train, in_test, out_train, out_test = tts(inputs, targets, test_size=0.02, train_size=0.98)

# Input standardisation.
print('Fitting and saving scaling functions.')
in_scaler = scaler()
in_train_scaled = in_scaler.fit_transform(in_train) 
in_test_scaled = in_scaler.transform(in_test)

# Save scaler for every model that uses it.
#joblib.dump(in_scaler, f'{paths.mod}/ols/ols_in_scaler.pkl')
#joblib.dump(in_scaler, f'{paths.mod}/lasso/lasso_in_scaler.pkl')
#joblib.dump(in_scaler, f'{paths.mod}/svm/svm_in_scaler.pkl')
#joblib.dump(in_scaler, f'{paths.mod}/nn/nn_in_scaler.pkl')
# Free up some memory.
del in_scaler

# Target standardisation. 
eps = 1e-12
out_train_scaled = np.log(out_train + eps)

# Build and use the ML models.
'''
name = 'ordinary least squares'
print(f'\n\n{name}:')
model = linear_model.LinearRegression()
model, preds = train_test(model, in_train, out_train, in_test)
get_metrics(in_test, out_test, preds)
plot_timeseries(in_test, out_test, preds, name)
save_model_data(model, in_test, out_test, preds, 'ols')

name = 'decision tree'
print(f'\n\n{name}:\n')
model = DecisionTreeRegressor(max_leaf_nodes=100000, random_state=con.seed)
model, preds = train_test(model, in_train, out_train, in_test)
get_metrics(in_test, out_test, preds)
plot_timeseries(in_test, out_test, preds, name)
save_model_data(model, in_test, out_test, preds, 'tree')

name = 'random forest'
print(f'\n\n{name}:\n')
model = RandomForestRegressor(criterion='squared_error', n_estimators=20, n_jobs=20, max_features=0.2, 
                              max_samples=0.1, max_leaf_nodes=100000, random_state=con.seed)
model, preds = train_test(model, in_train, out_train, in_test)
get_metrics(in_test, out_test, preds)
plot_timeseries(in_test, out_test, preds, name)
save_model_data(model, in_test, out_test, preds, 'forest')

name = 'LASSO'
print(f'\n\n{name}:\n')
# Find suitable size of Lasso regularisation const.
avg = np.mean(out_train)
e = np.floor(np.log10(avg)) - 1
reg = 10**e
model = linear_model.Lasso(alpha=reg)
#model = linear_model.Lasso(alpha=1e-3, tol=1e-4)
model, preds = train_test(model, in_train, out_train, in_test)
get_metrics(in_test, out_test, preds)
plot_timeseries(in_test, out_test, preds, name)
save_model_data(model, in_test, out_test, preds, 'lasso')

name = 'neural network'
print(f'\n\n{name}:\n')
model = MLPRegressor(hidden_layer_sizes=(100,100,50,50))
model, preds = train_test(model, in_train_scaled, out_train_scaled, in_test_scaled)
# Reverse output scaling.
preds = np.exp(preds) - eps
get_metrics(in_test, out_test, preds)
plot_timeseries(in_test, out_test, preds, name)
save_model_data(model, in_test, out_test, preds, 'nn')

name = 'gradient boosting machine'
print(f'\n\n{name}:\n')
model = MultiOutputRegressor(hgbr(learning_rate=0.05, max_leaf_nodes=100, random_state=con.seed, max_iter=100))
model, preds = train_test(model, in_train, out_train, in_test)
get_metrics(in_test, out_test, preds)
plot_timeseries(in_test, out_test, preds, name)	
save_model_data(model, in_test, out_test, preds, 'gbm')
'''
name = 'support vector machine'
print(f'\n\n{name}:\n')
model = MultiOutputRegressor(LinearSVR(max_iter=500, verbose=1), n_jobs=1)
model, preds = train_test(model, in_train_scaled, out_train, in_test_scaled)
get_metrics(in_test, out_test, preds)
plot_timeseries(in_test, out_test, preds, name)
save_model_data(model, in_test, out_test, preds, 'svm')
'''
# Load all the preds.
models = ['ols', 'lasso', 'tree', 'gbm', 'forest', 'nn'] 
names = ['OLS', 'LASSO', 'decision tree', 'GBM', 'random forest', 'neural network']
colours = ['tab:purple', 'tab:green', 'tab:orange', 'tab:red', 'tab:pink', 'tab:cyan']

# Independent data.
target = out_test[:, 19].squeeze()
lvl = in_test[:, 2].squeeze()
lvls, target_avg = avg_data(lvl, target)
plt.plot(target_avg, lvls, color='black', linewidth=4, label='Ozone J-values from UKCA')

# Plot all models' preds on one graph.
for i in range(len(models)):
  model = models[i]
  name = names[i]
  colour = colours[i]
  inputs, _, preds = fns.load_model_data(model)
  lvl = inputs[:, 2].squeeze()
  pred = preds[:, 19].squeeze()
  _, pred_avg = avg_data(lvl, pred)
  plt.plot(pred_avg, lvls, color=colour, label=f'Ozone J-values from {name}') 
  
plt.xlabel(f'Ozone photolysis rate coefficient {con.pers}')
plt.ylabel('Vertical model level')
plt.title('Mean ozone photolysis rate coefficient profiles from different models')
plt.legend()
plt.show()
plt.close()

# Independent data.
target = out_test[:, 16].squeeze()
day = in_test[:, 0].squeeze()
days, target_avg = avg_data(day, target)
plt.plot(days, target_avg, color='black', linewidth=4, label=f'H{con.sub2}O{con.sub2} J-values from UKCA')

# Plot all models' preds on one graph.
for i in range(len(models)):
  model = models[i]
  name = names[i]
  colour = colours[i]
  inputs, _, preds = fns.load_model_data(model)
  day = inputs[:, 0].squeeze()
  pred = preds[:, 16].squeeze()
  _, pred_avg = avg_data(day, pred)
  plt.plot(days, pred_avg, color=colour, label=f'H{con.sub2}O{con.sub2} J-values from {name}') 
  
plt.xlabel('Day of year')  
plt.ylabel(f'H{con.sub2}O{con.sub2} photolysis rate coefficient {con.pers}')
plt.title(f'Daily mean H{con.sub2}O{con.sub2} photolysis rate coefficients from different models')
plt.legend()
plt.show()
plt.close()
'''
