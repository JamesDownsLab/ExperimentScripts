from ParticleTracking import dataframes
from Generic import plotting
import matplotlib.pyplot as plt
import numpy as np

def histogram(file, duty):
    data = dataframes.DataStore(file)
    print(data.frame_data.head())
    df = data.frame_data.merge(data.df.drop('frame', 1).reset_index()).set_index('Duty')

    p = plotting.Plotter()
    for d in duty:
        orders = df.loc[d, 'real order']
        n, bins = np.histogram(orders, bins=100, density=True)
        p.add_plot(bins[:-1]+(bins[1]-bins[0])/2, n*(bins[1]-bins[0]), label=d)
    p.configure_legend()
    p.configure_xaxis(xlabel='Order Parameter')
    p.configure_yaxis(ylabel='Relative Frequency')



if __name__ == "__main__":
    from Generic import filedialogs
    file = filedialogs.load_filename()
    histogram(file, [600, 620, 640, 660, 680, 700])
    # histogram(file, [500, 600, 700, 800, 900])