# %%
import sys

sys.path.append('/home/ppxjd3/Code')

from Generic import filedialogs
from ParticleTracking import dataframes

import numpy as np
import matplotlib.pyplot as plt

# %%
file = filedialogs.load_filename()
data = dataframes.DataStore(file)

boundary = data.get_metadata('boundary')
center_of_tray = np.mean(boundary, axis=0)
frames = np.max(data.df.index.values)

mean_position = [list(np.mean(data.get_info(f, ['x', 'y']), axis=0))
                 for f in range(frames)]

mean_position = np.array(mean_position)
fig, ax = plt.subplots()
x = data.get_column('x')
y = data.get_column('y')
# ax.hexbin(mean_position[:, 0], mean_position[:, 1])
ax.hexbin(x, y, mincnt=0)
for n in range(6):
    ax.plot([boundary[n - 1, 0], boundary[n, 0]],
            [boundary[n - 1, 1], boundary[n, 1]], 'g-')
ax.set_aspect('equal')
plt.show()
