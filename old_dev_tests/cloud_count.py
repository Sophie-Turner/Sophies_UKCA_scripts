'''
Sum the total grid boxes with cloud in them above each grid box.
'''

import numpy as np
import file_paths as paths
import matplotlib.pyplot as plt

data_path = f'{paths.npy}/20150115cc.npy'
data = np.load(data_path)
print(data.shape)

cloud_sum = data[7] 
alt = data[1] * 85

plt.scatter(cloud_sum, alt)
plt.xlabel('Cloud above')
plt.ylabel('Altitude / km')
plt.show()
