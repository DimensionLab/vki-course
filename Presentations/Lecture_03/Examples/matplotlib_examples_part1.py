# ## Matplotlib: publication quality plots
# * Provides capabilities for plotting similar to Matlab
# * Interactive and animated figures
# * Posibility for LaTeX rendering for improved quality
# * Most other plotting libraries rely on Matplotlib or use a similar syntax
#

# ### The pyplot sub-module is the most useful for us

import matplotlib.pyplot as plt  # import as an alias for easier typing.

import numpy as np
# ## Let's make a simple parabola plot

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
y = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225]

fig, ax = plt.subplots()
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Parabola')
ax.plot(x, y)

# ## Customize your plots

# Use LaTeX fonts:
plt.style.use({'font.family': 'STIXGeneral',
               'font.serif': 'Computer Modern',
               'font.sans-serif': 'Computer Modern Sans serif', })

y_noisy = np.random.normal(y, 5)  # create some noisy data to simulate the experimental points

# create a new plot
fig2, ax2 = plt.subplots()
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y axis')
ax2.plot(x, y, label='2nd order polynomial')
ax2.plot(x, y_noisy, 'rd--', label='Experimental points', alpha=0.5)
ax2.grid()
ax2.set_xlim([0, 10])
ax2.set_ylim([0, 125])
ax2.legend()
fig2.savefig('figure.png')
