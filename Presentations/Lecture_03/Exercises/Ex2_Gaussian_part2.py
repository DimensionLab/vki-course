import numpy as np
import matplotlib.pyplot as plt
import os


def compute_normal(x, mu, sigma):
    y = 1/(sigma * (2*np.pi)**0.5)*np.e**(-0.5*((x-mu)/sigma)**2) # YOUR CODE HERE
    return y


# Set temp folder to save figures
temp_folder = 'temp_histogram'

if not os.path.exists(temp_folder):
    os.mkdir(temp_folder)

## theoretical part
mu = 4.5
sigma = 7

xmin = mu - 3 * sigma
xmax = mu + 3 * sigma

x = np.linspace(xmin, xmax, num=300)
y_theoretical = compute_normal(x, mu, sigma)

fig_hist, ax_hist = plt.subplots()

# plot the theoretical curve. Label it accordingly
ax_hist.plot(x, y_theoretical)

# fix the limits for the figure
ax_hist.set_ylim(0, max(y_theoretical) * 1.1)

##

# create a list of 100 numbers between 5 and 20000, distributed geometrically
# HINT: check for functions similar to linspace
# HINT2: set dtype=int to avoid decimal numbers in the list.
N_samples = np.geomspace(start=5, stop=20_000, num=100, dtype=int) #your code here

# containers for the results
means = []
stds = []

# loop to create the distributions
for n in N_samples:
    random_data = np.random.normal(mu, sigma, n)

    mean = np.mean(random_data)
    std = np.std(random_data)

    # append the current mean and std to the lists created above
    means.append(mean) # YOUR CODE HERE
    stds.append(std) # YOUR CODE HERE

    # plot the random data and set it's color
    count, bins, bars = ax_hist.hist(random_data, color='green', density=True, bins=60, label='Data', alpha=0.7)

    # compute the normal from the current samples
    y_computed = compute_normal(x, mu=mean, sigma=std) # YOUR CODE HERE
    current_curve = ax_hist.plot(x, y_computed, color='black', label='Normal distr. Computed')

    # put the title
    ax_hist.set_title(f'Samples: {n}, mean={mean:.2f}, std={std:.2f}')
    ax_hist.legend(loc='upper left')

    # save the figure
    fig_hist.savefig(f'{temp_folder}/Im_{str(n).zfill(4)}.png')

    # remove the current points for a clean plot
    t = [b.remove() for b in bars]
    line = current_curve.pop(0)
    line.remove()

# We can animate the previous plot

import imageio  # This is used to read the dumped images and create the animation
import shutil  # package to manage files

GIF_name = 'histogram.gif'
cleanup = False  # delete the temporary folder

images = []
# collect the list of images from the temporary folder
for filename in os.listdir(temp_folder):
    images.append(imageio.imread(f'{temp_folder}/{filename}')) # YOUR CODE HERE

imageio.mimsave(GIF_name, images, duration=0.2)

if cleanup:
    shutil.rmtree(temp_folder)


# ### Convergence of mean and std

# create the plot
# plot the means and the standard deviations and label them accordingly.
# set the scale of the x-axis to log. HINT: look for set_xscale
# decorate the graph.
# save the file.

fig, ax = plt.subplots()

ax.plot(N_samples, means, label='mean')
ax.plot(N_samples, stds, label='std')

ax.set_xlabel('Num. Samples')

ax.legend()
ax.set_xscale('log')

fig.savefig('convergence.pdf')

