# ## Exercise: gaussian histogram (part 1)
# - Generate random data normally distributed
# - Plot it in a histogram
# - Include the theoretical normal distribution
# - Add a legend
# - Save the figure

import numpy as np
import matplotlib.pyplot as plt


mu = 2.5
sigma = 1

# generate random data with N=1000 points, with the given distribution
random_data = np.random.normal(loc=mu, scale=sigma, size=10_000)

fig_hist, ax_hist = plt.subplots()
ax_hist.hist(random_data, density=True, bins=60, label='Data');

# compute the theoretical gaussian
xmin = mu - 3*sigma
xmax = mu + 3*sigma

x = np.linspace(xmin, xmax, num=300)
y = 1/(sigma * (2*np.pi)**0.5)*np.e**(-0.5*((x-mu)/sigma)**2)

# plot the theoretical gaussian
ax_hist.plot(x, y, label='Theoretical')

# add the legend to the graph
ax_hist.legend()

# save it as a pdf or a png
fig_hist.savefig('distribution.png', dpi=300)

plt.show()