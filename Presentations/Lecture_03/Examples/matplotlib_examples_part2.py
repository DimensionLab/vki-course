## Creating an animation

import os

import matplotlib.pyplot as plt  # import as an alias for easier typing.
import numpy as np

# ## Let's make a simple parabola plot

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
y = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225]
y_noisy = np.random.normal(y, 5)  # create some noisy data to simulate the experimental points

# Generating the parabola step by step

temp_folder = 'gif_temp'
if not os.path.exists(temp_folder):
    os.mkdir(temp_folder)

figAnim, axanim = plt.subplots()

# set plot properties that do not change.
axanim.set_xlabel('X axis')
axanim.set_ylabel('Y axis')
axanim.set_xlim([0, 10])
axanim.set_ylim([0, 125])
axanim.plot(x, y, label='2nd order polynomial')
axanim.grid()

for i, _ in enumerate(x):
    current_points = axanim.plot(x[:i], y_noisy[:i], 'rd--', label='Experimental points', alpha=0.5)
    figAnim.savefig(f'{temp_folder}/Im_{str(i).zfill(2)}.png')  # the use of zfill allows to recover them ordered
    print(f'Image {i} saved')
    # remove the current points for a clean plot
    points = current_points.pop(0)
    points.remove()

# ### Building the GIF

import imageio  # This is used to read the dumped images and create the animation
import shutil  # package to manage files

GIF_name = 'animated_parabola.gif'
cleanup = True  # delete the temporary folder

images = []
for filename in os.listdir(temp_folder):
    images.append(imageio.imread(f'{temp_folder}/{filename}'))

imageio.mimsave(GIF_name, images, duration=0.2)

if cleanup:
    shutil.rmtree(temp_folder)
