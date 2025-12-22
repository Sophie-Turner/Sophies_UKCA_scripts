'''
18/12/2025
This is a test of model integrity when implemented in the UM.
A quick, basic comparison of targets from original data,
predictions from random forest trained in Python,
and predictions from the random forest implemented in the UM's Fortran.
Training data leaking into the test set wouldn't matter
because this is a test of model similarity, not predictive performance.
There may be some differences due to some J rates being copied from FastJX
in the Fortran version, and different timesteps used for inputs.
'''

import os
import joblib
import numpy as np
import constants as con
import functions as fns
import file_paths as paths
from sklearn.metrics import r2_score
  
# The test data. 
test_day = '19820115'

# Original target data from the UM.
target_file = f'{paths.npy}/{test_day}.npy' 
print(f'\nLoading target data from {target_file}.')
target_data = np.load(target_file)
print('Target data from original UM:', target_data.shape)
# Pick out the relevant data from full npy datatset.
inputs_idx = [0,1,2,4,5,9,10,11,8,12]
targets_idx = [17,18,19,23,26,27,28] + list(range(30,49))
# Test inputs and target J rates.
inputs, targets = fns.in_out_swap(target_data, inputs_idx, targets_idx)
del(target_data)

# Predictions from model in Fortran.
preds_file = f'{paths.npy}/{test_day}_preds.npy'
# J rate predictions from Fortran model.
print(f'\nLoading prediction data from {preds_file}.')
preds_f = np.load(preds_file)
# Pick out the relevant J rates from dataset.
preds_idx = [9,10,11,15,18,19,20] + list(range(22,41))
preds_f = preds_f[preds_idx]
if preds_f.ndim > 1:
  preds_f = np.swapaxes(preds_f, 0, 1) 
print('Predictions data from online random forest in UM:', preds_f.shape)

# The random forest trained in Python.
model_name = 'rf_fortran_poc' 
model_file = f'{paths.mod}/{model_name}/{model_name}.pkl'
print(f'\nLoading random forest model from {model_file}.')
model = joblib.load(model_file)
# Use the python implementation of the model on the test inputs.
preds_p = model.predict(inputs)
print('Predictions from offline random forest in Python:', preds_p.shape)

# Get an average R2 score of both models.
r2_p = round(r2_score(targets, preds_p), 3)
r2_f = round(r2_score(targets, preds_f), 3)
print(f'\nOverall average {con.r2} score from offline Python model = {r2_p}') 
print(f'Overall average {con.r2} score from online Fortran model = {r2_f}\n') 

for rxn in range(len(targets[0])):
  target = targets[:, rxn]
  pred_p = preds_p[:, rxn]
  pred_f = preds_f[:, rxn]
  r2_p = round(r2_score(target, pred_p), 3)
  r2_f = round(r2_score(target, pred_f), 3)
  print(f'Target reaction {rxn}: Python model {con.r2} = {r2_p}. Fortran model {con.r2} = {r2_f}.')

# Compare differences between the two models.
diff = preds_f - preds_p
n_diff = np.count_nonzero(diff)
n_total = preds_p.size
percent = 100 * n_diff / n_total
print(f'{n_diff} out of {n_total} predictions are different ({round(percent,1)}%).')
for i in range(20):
  tol = 10**(-i)
  if not np.allclose(preds_f, preds_p, atol=tol):
    print(f'The predictions are within {10**(-(i-1))} of each other.')
    break
else:
  print('The predictions are within 1e-20 of each other so are effectively the same.')
