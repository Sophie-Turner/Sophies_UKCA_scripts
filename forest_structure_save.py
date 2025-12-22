'''
5/12/2025.
Save a random forest's structure as arrays in NetCDF format
For importation into the UM's Fortran code.
'''

import joblib
import numpy as np
import file_paths as paths
from netCDF4 import Dataset

model_name = 'rf_fortran_poc'
model_path = f'{paths.mod}/{model_name}/{model_name}.pkl'
netcdf_path = f'{paths.mod}/{model_name}/{model_name}_structure.nc'

model = joblib.load(model_path)
netcdf = Dataset(netcdf_path, 'w', format='NETCDF4')

# Get model sizes.
n_trees = len(model.estimators_)

netcdf.createDimension('trees', n_trees)

# Store each tree's arrays ('group') in the same netCDF.
for i, estimator in enumerate(model.estimators_):
  print(f'Processing tree {i+1} of {n_trees}.')
  group = netcdf.createGroup(f'tree_{i+1}')
  tree = estimator.tree_
  
  # Get tree arrays as 32-bit types.
  children_left = tree.children_left.astype('i4')
  children_right = tree.children_right.astype('i4')
  features = tree.feature.astype('i4')
  thresholds = tree.threshold.astype('f4')
  # Don't actually need to store all of these values but doing it for now 
  # for simplicity and ease of storage. 
  # Can change later to leaves only, and probably halve memory usage.
  values = tree.value.squeeze().astype('f4') # Shape (n_nodes, n_outputs).
  n_nodes, n_outputs = values.shape
  
  group.createDimension('nodes', n_nodes)
  group.createDimension('outputs', n_outputs)
  
  left = group.createVariable('children_left', 'i4', ('nodes',))
  right = group.createVariable('children_right', 'i4', ('nodes',))
  feat = group.createVariable('features', 'i4', ('nodes',))
  thresh = group.createVariable('thresholds', 'f4', ('nodes',))
  val = group.createVariable('values', 'f4', ('nodes', 'outputs'))
  
  # Put the data in the netCDF dataset.
  left[:] = children_left
  right[:] = children_right
  feat[:] = features
  thresh[:] = thresholds
  val[:,:] = values 
  
  # Metadata.
  netcdf.model_type = 'random forest regressor'
  netcdf.n_features = model.n_features_in_
  netcdf.n_outputs = model.n_outputs_
  netcdf.feature_subsample = model.max_features
  
netcdf.close()
print(f'Saved netCDF version of forest structure to {netcdf_path}.')

print('Opening and reading the new netCDF file.')
netcdf = Dataset(netcdf_path, 'r')
print(netcdf)
