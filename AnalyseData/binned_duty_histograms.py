import matplotlib.pyplot as plt
import numpy as np

from Generic import filedialogs
from ParticleTracking import dataframes

# %%
file = filedialogs.load_filename(file_filter='*.hdf5')

data = dataframes.DataStore(file)

data.df['order_mag'] = np.abs(data.df.order_i + data.df.order_r)

bins = [400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
orders = []
for i in range(len(bins) - 1):
    orders.append(
        data.df.loc[(data.df.Duty >= bins[i]) * (data.df.Duty <= bins[i + 1]),
                    'order_mag'].values)

fig, ax = plt.subplots()
for i, order in enumerate(orders):
    freq, hist_bins = np.histogram(order, bins=np.arange(0, 1.01, 0.01),
                                   density=True)
    ax.plot(hist_bins[:-1], freq,
            label=str(bins[i]) + ' to ' + str(bins[i + 1]))
    ax.legend()
    ax.set_title(file)
    ax.set_xlabel('Order')
