'''
Name: Sophie Turner.
Date: 20/11/2024.
Contact: st838@cam.ac.uk.
Train a standard random forest and save it and its test data.
'''

import os
import re
import time
import joblib
import datetime
import numpy as np
import constants as con
import functions as fns
import file_paths as paths
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler as scaler

# Whether or not to standardise data.
scale_inputs = False
scale_targets = False

# File paths.
data_path = f'{paths.npy}/1982_182m_fixed_days.npy'
out_name = 'rf_problems_J_inputs_1982'
out_dir = f'{paths.mod}/{out_name}'
out_path = f'{out_dir}/{out_name}'
model_path = f'{out_path}.pkl' 
in_scale_path = f'{out_path}_in_scaler.pkl'
out_scale_path = f'{out_path}_out_scaler.pkl'
out_test_path = f'{out_path}_targets.npy'
pred_path = f'{out_path}_pred.npy'
in_test_path = f'{out_path}_inputs.npy'
meta_path = f'{out_path}_metadata.txt'

print()
print(out_name)

# Prepare directory.
if os.path.exists(out_dir):
  print(f'Warning: The directory {out_dir} already exists.')
  overwrite = input('Do you want to overwrite the model & data in this directory? (y/n): ').strip().lower()
  if overwrite != 'y':
    exit(1)
else:    
  os.mkdir(out_dir)

start = time.time()
print('\nLoading data from', data_path)
data = np.load(data_path)
print(data.shape)
end = time.time()
print(f'Loading the data took {round(end-start)} seconds.')

# Indices of 1982 training data in full npy datatset.
#inputs_idx = list(range(13))
#targets_idx = [17,18,19,23,26,27,28] + list(range(30,49))
inputs_idx = list(range(13)) + [17, 18, 19, 23, 28, 31, 32, 33, 35, 36, 38, 39, 40, 44, 45, 48] # All-alt Js.
# OCS, SO3, ISON, H2O, HNO3, O2, O3, N2O, MeCHO, NO.
targets_idx = [26, 27, 30, 34, 37, 41, 42, 43, 46, 47] # High-alt Js.

# Set Strat-only targets to 0 below model lvl 60.
# Problem J-values are OCS, ISON, H2O, O2, N2O, MeCHO -> CH4 and NO.
#problems_idx = [26, 30, 34, 41, 43, 46, 47]
#lows_idx = np.where(data[2] < 60)[0]
#data[np.ix_(problems_idx, lows_idx)] = 0.0 

# Select levels.
#lvl_max = 71
#data = data[:, data[2] < lvl_max] 
#print('Selected levels:', data.shape)
  
# Get inputs and targets.
inputs, targets = fns.in_out_swap(data, inputs_idx, targets_idx)

# Scale inputs.
if scale_inputs:
  in_scale = scaler()
  inputs = in_scale.fit_transform(inputs)
  
# 99/1 train test split.  
in_train, in_test, out_train, out_test, i_test = fns.tts(inputs, targets, 0.01)

# Scale training targets.
if scale_targets:
  out_scale = scaler()
  out_scale.fit(targets)
  #NO3 = out_train[:, 11].copy() # Leave NO3 unscaled.
  out_train = out_scale.transform(out_train)
  #out_train[:, 11] = NO3

# Make the regression model.
model = RandomForestRegressor(criterion='squared_error', n_estimators=20, n_jobs=20, max_features=0.2, max_samples=0.1, max_leaf_nodes=100000, random_state=con.seed)

# Train the model.
start = time.time()
print('Training model')
model.fit(in_train, out_train)
end = time.time()
print(f'Training the model took {round(end-start)} seconds.')

# Test the model.
print('Testing the model.')
out_pred, _, _, _, _, r2 = fns.test(model, in_test, out_test)
print(f'Overall average R2 score for all J rates: {round(r2, 3)}') 

# Reverse scaling on test inputs and predictions.
if scale_inputs:
  in_test = in_scale.inverse_transform(in_test)
if scale_targets:  
  #NO3 = out_pred[:, 11].copy() # except for NO3, which was not scaled.
  out_pred = out_scale.inverse_transform(out_pred)
  #out_pred[:, 11] = NO3

# Save the trained model, data and scalers.
start = time.time()
print('Saving random forest model, and scalers if chosen, to')
print(model_path)
joblib.dump(model, model_path) 
if scale_inputs:
  joblib.dump(in_scale, in_scale_path)
if scale_targets:  
  joblib.dump(out_scale, out_scale_path)
end = time.time()
print(f'Saving the random forest model, and scalers if chosen, took {round(end-start)} seconds.')

# Save the test dataset.
start = time.time()
print('Saving test dataset.')
np.save(out_test_path, out_test)
np.save(pred_path, out_pred)
np.save(in_test_path, in_test)
end = time.time()
print(f'Saving the test datasets took {round(end-start)} seconds.')

# Metadata text.
text = ''
if scale_inputs:
  text = f'{in_scale_path}: standardising scaler to use on inputs from new datasets. Scaling of provided inputs has been reversed already. Read into Python using joblib.\n'
if scale_targets:
  text = f'{text}\
  {out_scale_path}: standardising scaler to reverse scaling of predictions from new datasets (not the predictions provided here). Scaling of priovided predictions has been reversed already. Read into Python using joblib.\n'

# Write metadata.
meta = f'Date: {datetime.date.today()}\n\
{model_path}: random forest model, made using scikit-learn. Read into Python using joblib.\n\
{text}\
{out_test_path}: 2d numpy array of test targets dataset for the random forest, from a 99% train, 1% test split of the training data, of shape(samples, features).\n\
{pred_path}: 2d numpy array of predictions from the above test set, of shape(samples, features).\n\
{in_test_path}: 2d numpy array of inputs used to make the above datasets, of shape(samples, features).\n\
Training data: {data_path}\n\
Data alterations: 99.5% of samples from day-time data, 0.5% from night-time data.\n\
Inputs: Day of year, hour of day, model level, hybrid height, latitude, longitude, humidity, cloud fraction, solar zenith angle, upward shortwave flux, downward shortwave flux, pressure, temperature,\
  all-altitude J-rates.\
  Corresponds to same indices as preds and targets.\n\
Targets: J-values which are hard to predict due to only happening at high alts.\n\
Trees: {len(model.estimators_)}.\n\
Max leaves per tree: 100000.\n\
Nodes per tree: {model.estimators_[0].tree_.node_count}.\n\
Tree depth: {model.estimators_[0].tree_.max_depth}.\n\
Max features per tree: 20% of data.\n\
Max samples per tree: 10% of data.\n\
Random number generator seed for sampling, test split and forest creation: {con.seed}.\n\
Loss fn: MSE.'

meta_file = open(meta_path, 'w')
meta_file.write(meta)
meta_file.close() 
