import numpy as np
import constants as con
import functions as fns
import file_paths as paths
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

'''
Plot columns of targets and preds for different sized data samples.
'''


def train_test(model, in_train, out_train, in_test):
  # Train.
  print('Training model.')
  model.fit(in_train, out_train)
  # Make preds.
  print('Testing model.')
  preds = model.predict(in_test)
  return model, preds
  

# Get different sized data samples.
file_small = f'{paths.npy}/low_res_yr_50.npy'
file_large = f'{paths.npy}/low_res_yr_500k.npy'
# Get the full-resolution col.
file_col = f'{paths.npy}/col20151222_cam12.npy'

sample_small = np.load(file_small)
sample_large = np.load(file_large)
full_col = np.load(file_col) 
print(sample_small.shape)
print(sample_large.shape)
print(full_col.shape)

# Train a random forest on both samples.
model = RandomForestRegressor(criterion='squared_error', n_estimators=5, n_jobs=5, max_features=0.2, 
                              max_samples=0.2, max_leaf_nodes=100000, random_state=con.seed)

# Input and output indices.
inputs_idx = [0,1,2,3,4,5,6,7,8,9,10,13]
targets_idx = [16]

# Get test inputs and outputs for the column.
in_test, out_test = fns.in_out_swap(full_col, inputs_idx, targets_idx)  

# Small dataset.
# Set up training data.
in_train_small, out_train_small = fns.in_out_swap(sample_small, inputs_idx, targets_idx)  
# Train the model.
model_small = model.fit(in_train_small, out_train_small)
# Test it on the column.
preds_small = model_small.predict(in_test)

# Large dataset.
# Set up training data.
in_train_large, out_train_large = fns.in_out_swap(sample_large, inputs_idx, targets_idx)  
# Train the model.
model_large = model.fit(in_train_large, out_train_large)
# Test it on the column.
preds_large = model_large.predict(in_test)
  
# View vertical column profile for NO2.
alt = in_test[:,1] * 85
plt.plot(out_test, alt, color='tab:blue', label='UKCA')
plt.plot(preds_small, alt, color='tab:orange', label='Random forest trained on 50 data points per day', )
plt.plot(preds_large, alt, color='tab:pink', label='Random forest trained on 500,000 data points per day')
#plt.xscale('log')
plt.title(f'NO{con.sub2} J-values in a vertical column over Cambridge at midday on 15th July 2015.')
plt.xlabel(f'J-value / {con.pers}')
plt.ylabel('Height / km')
plt.legend()
plt.show()
plt.close()
  
# View error difference plot in vertical column.
diff_small = ((preds_small.squeeze() - out_test.squeeze()) / out_test.squeeze()) * 100
diff_large = ((preds_large.squeeze() - out_test.squeeze()) / out_test.squeeze()) * 100
plt.plot(diff_small, alt, color='tab:orange', label='Random forest trained on 50 data points per day', )
plt.plot(diff_large, alt, color='tab:pink', label='Random forest trained on 500,000 data points per day')
plt.axvline(x=0, color='grey', linestyle='--', linewidth=1)
plt.title(f'NO{con.sub2} J-values in a vertical column over Cambridge at midday on 15th July 2015.')
plt.xlabel(f'Percentage difference of ML predictions to UKCA targets')
plt.ylabel('Height / km')
plt.legend()
plt.show()
plt.close()
