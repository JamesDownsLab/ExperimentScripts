import matplotlib.pyplot as plt
import numpy as np

from ParticleTracking import dataframes

direc = "/media/data/Data/July2019/RampsN29/"
extension = ".hdf5"

files_down = ['15790002', '15790005', '15790007', '15790009', '15800001',
              '15800003', '15800005', '15800007', '15810001', '15810003']

files_up = ['15790003', '15790006', '15790008', '15790010', '15800002',
            '15800004', '15800006', '15800008', '15810002', '15810004']

for files in zip(files_up, files_down):
    xdata = []
    ydata = []
    for file in files:
        file = direc + file + extension
        data = dataframes.DataStore(file)
        data.df['order_mag'] = np.abs(data.df.order_r + 1j * data.df.order_i)
        group = data.df.groupby('Duty')['order_mag'].mean()
        xdata.append(group.index.values)
        ydata.append(group.values)
    plt.figure()
    plt.plot(xdata[0], ydata[0], label='up')
    plt.plot(xdata[1], ydata[1], label='down')
    plt.legend()
    plt.title(data.metadata['number_of_particles'])
plt.show()
