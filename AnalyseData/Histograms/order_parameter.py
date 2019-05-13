from Generic import plotting
from ParticleTracking import dataframes
import numpy as np
import matplotlib.pyplot as plt

def order_histogram(file, frames):
    data = dataframes.DataStore(file)
    plotter = plotting.Plotter()
    for f in frames:
        order = data.get_info(f, ['real order'])
        n, bins = np.histogram(order, np.linspace(0, 1, 100))
        bins = bins[:-1] + (bins[1]-bins[0])/2
        plotter.add_plot(bins, n, label=str(f))
    plotter.configure_xaxis(0, 'Order parameter')
    plotter.configure_yaxis(0, 'Frequency')
    plotter.configure_legend(0, title='Frame')
    plotter.show_figure()


def frame_order(file):
    data = dataframes.DataStore(file)
    print(data.num_frames)
    plotter = plotting.Plotter()
    # duty_cycle = data.frame_data.Duty
    # frames = data.frame_data.Duty
    # order = data.frame_data['mean order']
    # plotter.add_plot(frames, order, fmt='x')
    # plotter.show_figure()
    fdata = data.frame_data
    fdata = fdata.groupby('Duty').mean()
    plotter.add_plot(fdata.index.values, fdata['mean order'].values)
    plotter.show_figure()

def up_and_down(file1, file2):
    data1 = dataframes.DataStore(file1)
    data2 = dataframes.DataStore(file2)
    duty1, order1 = duty_order(data1)
    duty2, order2 = duty_order(data2)
    plotter = plotting.Plotter()
    plotter.add_plot(duty1, order1, label='up')
    plotter.add_plot(duty2, order2, label='down')
    plotter.configure_legend()
    plotter.show_figure()

def duty_order(data):
    fdata = data.frame_data
    fdata = fdata.groupby('Duty').mean()
    return fdata.index.values, fdata['mean order'].values



if __name__ == "__main__":
    from Generic import filedialogs
    file1 = filedialogs.load_filename(
        'Select a dataframe',
        directory="/media/data/Data",
        file_filter='*.hdf5')
    file2 = filedialogs.load_filename(
        'Select a dataframe',
        directory="/media/data/Data",
        file_filter='*.hdf5')
    up_and_down(file1, file2)
    # frame_order(filename)
    # order_histogram(filename, np.arange(0, 20000, 1000))