'''
March 2026
Repeat NN tests on 1982 data using Scikit.
'''

import numpy as np
import functions as fns
import file_paths as paths
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error

data_path = f'{paths.npy}/1982_182m.npy'
print('\nLoading data from', data_path)
data = np.load(data_path)
print(data.shape)

# Indices of 1982 training data in full npy datatset.
inputs_idx = [0,1,2,4,5,9,10,11,8,12]
targets_idx = [17,18,19,23,26,27,28] + list(range(30,49))

# Get inputs and targets.
inputs, targets = fns.in_out_swap(data, inputs_idx, targets_idx)  
  
# 95/5 train test split.  
in_train, in_test, out_train, out_test, _ = fns.tts(inputs, targets, 0.05)

model = MLPRegressor(hidden_layer_sizes=(20,20,20,20,20,20,10,10,10,10,10,10))

# Standardisation (optional).
scaler = StandardScaler()
in_train = scaler.fit_transform(in_train)
in_test = scaler.fit_transform(in_test)

# Train the NN.
model.fit(in_train, out_train)

# Test the NN.
pred = model.predict(in_test)

print('R2:', round(r2_score(out_test, pred), 2))
    
# Plotting this many datapoints is excessive and costly. Reduce it to 1%.
length = len(pred)
idxs = np.arange(0, length, 100)
pred = pred[idxs]
out_test = out_test[idxs]
del(idxs)

# Show a plot of results.
plt.scatter(out_test, pred, alpha=0.1)
plt.xlabel('targets from UKCA')
plt.ylabel('predictions by NN')
# Log scale.
plt.xscale('log')
plt.yscale('log')
plt.show()
