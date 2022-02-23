
# ## Read data from a file

import numpy as np

file_to_read = 'alumina_data_raw.txt'
data = np.genfromtxt(file_to_read, skip_header=1, delimiter=';')

# ## Exercise: plot data
# - From the previous dataset read column 0 (temperature) and column 3 (mass)
# - Add appropriate labels

# ### Solution


import matplotlib.pyplot as plt

# read column 0 for temperature
temperature = # your code here
# read column 3 for mass
mass = # your code here

fig, ax = plt.subplots()

ax.plot(# your code here)
# add the x label
# add the y label