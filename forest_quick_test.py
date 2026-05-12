'''
21/7/2025
A quick, basic overview of a saved random forest's performance.
'''

import os
import idx_names
import numpy as np
import constants as con
import functions as fns
import file_paths as paths
from sklearn.metrics import r2_score, mean_squared_error

model_name = 'rf_fortran_poc_fixed_days'
print()
print(model_name)
  
# Load the model data.
inputs, targets, preds = fns.load_model_data(model_name)

# Overall average R2 score for all J rates.
r2 = round(r2_score(targets, preds), 3)
print(f'\nOverall average {con.r2} score = {r2}\n')

# Troposphere. ~ 0 to 10 km above sea-level.
#alt_max = 0.121
#targets = targets[inputs[:,1] < alt_max] 
#preds = preds[inputs[:,1] < alt_max] 

# Print out the R2 scores for every J rate.  
lows = []
for rxn in range(len(targets[0])):
  target = targets[:, rxn]
  pred = preds[:, rxn]
  r2 = round(r2_score(target, pred), 3)
  #print(f'Target J rate {rxn} {con.r2} = {r2}')
  print(r2)
  #if r2 < 0.8:
  #  lows.append(name)
    
# See where the problems are.    
#print(f'\nLow-scoring J rates: {lows}')

for rxn in range(len(targets[0])):
  target = targets[:, rxn]
  pred = preds[:, rxn]
  mse = mean_squared_error(target, pred)
  print(mse)
