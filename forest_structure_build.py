'''
10/12/2025.
Build a trained random forest from arrays defining structure
saved in NetCDF format by decomposition of a sklearn model.
'''

import time
import joblib
import numpy as np
import file_paths as paths
from netCDF4 import Dataset


def predict_tree(tree_arrays, inputs, n_samples, n_rxns):
  """
  Predict using a single decision tree.
    
  Parameters:
  - tree_arrays: dict with keys 'features', 'thresholds', 'children_left', 'children_right', 'values'.
  - inputs: numpy array shape (n_samples, n_features).
  - n_samples: int number of time-space data points in inputs.
  - n_rxns: int number of J rates to predict. 
    
  Returns:
  - predictions: numpy array shape (n_samples, n_rxns).
  """
  predictions = np.zeros((n_samples, n_rxns), dtype=np.float32)
 
  for i in range(n_samples):
    node = 0
    while True:
      if tree_arrays['children_left'][node] == -1:
        # Leaf node reached.
        # Value shape should be (n_nodes, n_rxns).
        leaf = tree_arrays['values'][node]        
        predictions[i, :] = leaf
        break
      else:
        feature = tree_arrays['features'][node]
        threshold = tree_arrays['thresholds'][node]
        if inputs[i, feature] <= threshold:
          node = tree_arrays['children_left'][node]
        else:
          node = tree_arrays['children_right'][node]
  return predictions


model_name = 'rf_mini_test'
netcdf_path = f'{paths.mod}/{model_name}/{model_name}_structure.nc'
inputs_path = f'{paths.mod}/{model_name}/{model_name}_inputs.npy'
preds_path = f'{paths.mod}/{model_name}/{model_name}_pred.npy'

print('\nReading the NetCDF and data.')

inputs = np.load(inputs_path)
preds_orig = np.load(preds_path).astype(np.float32)
ds = Dataset(netcdf_path, 'r')
print(ds)

print('\nRetrieving the structural arrays for each tree.')

forest_arrays = {}
for tree in ds.groups:
  group = ds.groups[tree]
  tree_arrays = {
    'features': group.variables['features'][:],
    'thresholds': group.variables['thresholds'][:],
    'children_left': group.variables['children_left'][:],
    'children_right': group.variables['children_right'][:],
    'values': group.variables['values'][:],  
  }
  forest_arrays[tree] = tree_arrays
    
ds.close()

print(forest_arrays)

print('\nSampling inputs & datasets to same size as a UM timestep domain')
print('(1 time x 16 lon x 6 lat x 85 lvls) samples.')
inputs = inputs[:8160]
preds_orig = preds_orig[:8160]
print('Inputs:', inputs.shape)

print('\nUsing the trees data for regression in a loop for every sample, no optimisation.')

# Get dims from 1 tree.
tree = forest_arrays['tree_0']
n_rxns = tree['values'].shape[1]
n_samples = inputs.shape[0]
n_trees = len(forest_arrays) 

# Do the inference!
preds_sum = np.zeros((n_samples, n_rxns), dtype=np.float32)
start = time.time()
for tree_name, tree_arrays in forest_arrays.items(): 
  print('Traversing', tree_name) 
  preds_sum += predict_tree(tree_arrays, inputs, n_samples, n_rxns)
end = time.time()
elapsed = round((end - start) / 60, 2)
print(f'The forest traversal took {elapsed} seconds.')

# Average predictions from trees.
preds_new = preds_sum / n_trees

# Now compare the preds and see if the decomposition & reconstruction actually worked.
print('\nPredictions from original and reconstructed forests:', preds_new.shape, preds_orig.shape)  

diff = preds_new - preds_orig
if np.all(diff == 0):
  print('The predictions from both forests are the same!')
else:
  print('The predictions from the two versions of the forest are different!')

  # See what % of the preds have any difference and by how much they differ.
  n_diff = np.count_nonzero(diff)
  n_total = preds_orig.size
  percent = 100 * n_diff / n_total
  print(f'{n_diff} out of {n_total} predictions are different ({round(percent,1)}%).')
  for i in range(20):
    tol = 10**(-i)
    if not np.allclose(preds_new, preds_orig, atol=tol):
      print(f'The predictions are within {10**(-(i-1))} of each other.')
      break
  else:
    print('The predictions are within 1e-20 of each other so are effectively the same.')


