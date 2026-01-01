# Set SZA to 100 degrees instead of 90 when there's no light to avoid confusing the RF.

import numpy as np
import file_paths as paths

name = '1982_182m_fixed_days'
data_path = f'{paths.npy}/{name}.npy'
out_path = f'{paths.npy}/{name}_adjusted_sza.npy'

print('Loading data from', name)
data = np.load(data_path)

# 100 degrees in cos radians = -0.173648178.
# 9 is SZA and 11 is downward SW flux. 
print('Setting night SZA to 100 deg.')
data[9, (data[9] < 7e-17) & (data[11] == 0)] = -0.173648178 

print('Saving new dataset at', out_path)
np.save(out_path, data)
