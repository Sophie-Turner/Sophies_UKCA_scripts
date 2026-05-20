'''
5/3/2026
There's a problem with some J rates being overestimated below ~L70.
See if a RF with pysics inputs can be used to predict easy J rates
and then those preds used as inputs to predict hard J rates.
A 2-stage model.
'''

import time
import datetime
import numpy as np
import constants as con
import functions as fns
import file_paths as paths
from sklearn.ensemble import RandomForestRegressor

# File paths.
data_path = f'{paths.npy}/1982_182m_fixed_days.npy'

start = time.time()
print('\nLoading data from', data_path)
data = np.load(data_path)
print(data.shape)
end = time.time()
print(f'Loading the data took {round(end-start)} seconds.')

print('\nModel A: all-alt Js from physics.\n')

# Indices of model A training data in full npy datatset.
inputs_idx = list(range(13)) # Physical inputs.
# All the chemicals which do photolysis at low alts.
targets_idx = [17, 18, 19, 23, 31, 32, 33, 35, 36, 39, 40, 44, 45, 48] 
 
# Get inputs and targets.
inputs, targets = fns.in_out_swap(data, inputs_idx, targets_idx)
  
# 50/50 train test split (half for training each model).  
in_train, in_test, out_train, out_test, i_test = fns.tts(inputs, targets, 0.5)

# Make the regression model.
model = RandomForestRegressor(criterion='squared_error', n_estimators=20, n_jobs=20, 
                               max_features=0.2, max_samples=0.1, max_leaf_nodes=100000, 
			       random_state=con.seed)

# Train the model.
start = time.time()
print('Training model A on physical inputs')
model.fit(in_train, out_train)
end = time.time()
print(f'Training model A took {round(end-start)} seconds.')

# Test the model.
print('Testing model A.')
out_pred, _, _, _, _, r2 = fns.test(model, in_test, out_test)
print(f'Overall average R2 score for model A = {round(r2, 3)}') 

# Test model A at low alts.
lvl_max = 40
out_test_low = out_test[in_test[:,2] < lvl_max] 
out_pred_low = out_pred[in_test[:,2] < lvl_max] 
print('Selected levels:', out_pred_low.shape, in_test_low.shape)
# Overall average tropospheric R2 score for all J rates.
r2 = round(r2_score(out_test_low, out_pred_low), 3)
print(f'Overall average tropospheric {con.r2} score for model A = {r2}')


print('\nModel B: high-alt Js from model A preds.\n')

# All the predictions from model A.
print('out pred:', out_pred.shape)
print('in test:', in_test.shape)
inputs = np.concatenate((in_test, out_pred), axis=1)
print('Inputs:', inputs.shape)

# All the chemicals which only do photolysis at high alts.
targets_idx = [26, 27, 28, 30, 34, 37, 38, 41, 42, 43, 46, 47]
targets = np.swapaxes(targets, 0, 1) 
print('Targets:', targets.shape) 

# 99/1 train test split.  
in_train, in_test, out_train, out_test, i_test = fns.tts(inputs, targets, 0.01)

# Make the regression model.
model = RandomForestRegressor(criterion='squared_error', n_estimators=20, n_jobs=20, 
                               max_features=0.2, max_samples=0.1, max_leaf_nodes=100000, 
			       random_state=con.seed)
			    
# Train the model.
start = time.time()
print('Training model B on preds from model A.')
model.fit(in_train, out_train)
end = time.time()
print(f'Training model B took {round(end-start)} seconds.')

# Test the model.
print('Testing model B.')
out_pred, _, _, _, _, r2 = fns.test(model, in_test, out_test)
print(f'Overall average R2 score for model B = {round(r2, 3)}') 

# Test model B at low alts.
lvl_max = 40
out_test_low = out_test[in_test[:,2] < lvl_max] 
out_pred_low = out_pred[in_test[:,2] < lvl_max] 
print('Selected levels:', out_pred_low.shape, in_test_low.shape)
# Overall average tropospheric R2 score for all J rates.
r2 = round(r2_score(out_test_low, out_pred_low), 3)
print(f'Overall average tropospheric {con.r2} score for model B = {r2}')
