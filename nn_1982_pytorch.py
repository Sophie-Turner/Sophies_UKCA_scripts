'''
20/3/2026
Repeat NN tests on 1982 data using Pytorch.
'''

import torch
import functions as fns
import file_paths as paths
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler


# Create a class that inherits nn.Module.
class SmallModel(nn.Module):

  # Set up NN structure.
  def __init__(self, inputs=10, h1=8, h2=8, outputs=26):
    super().__init__() # Instantiate nn.module.
    self.fc1 = nn.Linear(inputs, h1) 
    self.fc2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, outputs) 

  # Set up movement of data through net.
  def forward(self, x):
    x = F.relu(self.fc1(x)) 
    x = F.relu(self.fc2(x)) 
    x = self.out(x) 
    return(x)
    

# Create a class that inherits nn.Module.
class DeepModel(nn.Module):

  # Set up NN structure.
  def __init__(self, inputs=10, h1=20, h2=20, h3=20, h4=20, h5=20, h6=20, h7=10, h8=10, h9=10, h10=10, h11=10, h12=10, outputs=26):
    super().__init__() # Instantiate nn.module.
    self.fc1 = nn.Linear(inputs, h1) 
    self.fc2 = nn.Linear(h1, h2)
    self.fc3 = nn.Linear(h2, h3)
    self.fc4 = nn.Linear(h3, h4)
    self.fc5 = nn.Linear(h4, h5)
    self.fc6 = nn.Linear(h5, h6)
    self.fc7 = nn.Linear(h6, h7)
    self.fc8 = nn.Linear(h7, h8)
    self.fc9 = nn.Linear(h8, h9)
    self.fc10 = nn.Linear(h9, h10)
    self.fc11 = nn.Linear(h10, h11)
    self.fc12 = nn.Linear(h11, h12)
    self.out = nn.Linear(h12, outputs) 

  # Set up movement of data through net.
  def forward(self, x):
    x = F.relu(self.fc1(x)) 
    x = F.relu(self.fc2(x)) 
    x = F.relu(self.fc3(x)) 
    x = F.relu(self.fc4(x)) 
    x = F.relu(self.fc5(x)) 
    x = F.relu(self.fc6(x)) 
    x = F.relu(self.fc7(x)) 
    x = F.relu(self.fc8(x)) 
    x = F.relu(self.fc9(x)) 
    x = F.relu(self.fc10(x)) 
    x = F.relu(self.fc11(x)) 
    x = F.relu(self.fc12(x)) 
    x = self.out(x) 
    return(x)
    
    
# Create a class that inherits nn.Module.
class WideModel(nn.Module):

  # Set up NN structure.
  def __init__(self, inputs=10, h1=128, h2=256, h3=256, outputs=26):
    super().__init__() # Instantiate nn.module.
    self.fc1 = nn.Linear(inputs, h1) 
    self.fc2 = nn.Linear(h1, h2)
    self.fc3 = nn.Linear(h2, h3)
    self.out = nn.Linear(h3, outputs) 

  # Set up movement of data through net.
  def forward(self, x):
    x = F.relu(self.fc1(x)) 
    x = F.relu(self.fc2(x)) 
    x = F.relu(self.fc3(x)) 
    x = self.out(x) 
    return(x)

    
data_path = f'{paths.npy}/1982_182m.npy'
print('\nLoading data from', data_path)
data = np.load(data_path)
print(data.shape)

# Indices of 1982 training data in full npy datatset.
inputs_idx = [0,1,2,4,5,9,10,11,8,12]
targets_idx = [17,18,19,23,26,27,28] + list(range(30,49))

# Get inputs and targets.
inputs, targets = fns.in_out_swap(data, inputs_idx, targets_idx)  
  
# Standardisation (optional).
scaler = StandardScaler()
inputs = scaler.fit_transform(inputs)  
  
# 95/5 train test split.  
in_train, in_test, out_train, out_test, _ = fns.tts(inputs, targets, 0.05)  

# Log transform targets.
eps = 1e-12
out_train = np.log(out_train + eps)
out_test = np.log(out_test + eps)

# Turn them into torch tensors.
in_train = torch.from_numpy(in_train.copy()).float()
in_test = torch.from_numpy(in_test.copy()).float()
out_train = torch.from_numpy(out_train).float()
out_test = torch.from_numpy(out_test).float()

print('\nin_train:', in_train.shape)
print('in_test:', in_test.shape)
print('out_train:', out_train.shape)
print('out_test:', out_test.shape)

# Batch of training data to fit in memory at once.
batch = 65536

train_data = TensorDataset(in_train, out_train)
train_load = DataLoader(train_data, batch_size=batch, shuffle=True)

# Create instance of model.
model = WideModel()

# Tell the model to measure the error as fitness function to compare pred with label.
criterion = nn.MSELoss()
# Choose optimiser and learning rate. Parameters are fc1, fc2, out, defined above.
opt = torch.optim.Adam(model.parameters(), lr=0.001)

# Train model.
epochs = 200 # Choose num epochs.
print()
for i in range(epochs):
  epoch_loss = 0.0
  
  for in_batch, out_batch in train_load:
    # Get predicted results.
    pred = model(in_batch) 
    # Measure error. Compare predicted values to training targets.
    loss = criterion(pred, out_batch) 

    # Backpropagation. Tune weights using loss.
    opt.zero_grad() 
    loss.backward() # Send loss back through the net.
    opt.step() # Step optimiser forward through the net.
  
    epoch_loss += loss.item()
  
  epoch_loss /= len(train_load)
  # Print every 10 epochs.
  if (i+1) % 10 == 0:
    print(f'Epoch {i+1} \tMSE: {loss.detach().numpy()}')

# Evaluate model on test set.
with torch.no_grad(): # Turn off backpropagation.
  pred = model(in_test) # Send the test inputs through the net.
  loss = criterion(pred, out_test) # Compare to test labels.
print('\nMSE on test data:', loss)

# Turn them into np arrays for analysis and reverse transformations.
out_test = np.exp(out_test.detach().numpy()) - eps, 
pred = np.exp(pred.detach().numpy()) - eps

# Make them the right shape.
pred = pred.squeeze()
out_test = out_test.squeeze()

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
