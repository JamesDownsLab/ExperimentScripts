from Generic import plotting
from ParticleTracking import dataframes
import numpy as np

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


if __name__ == "__main__":
    from Generic import filedialogs
    filename = filedialogs.load_filename(
        'Select a dataframe',
        directory="/home/ppxjd3/Videos",
        file_filter='*.hdf5')
    order_histogram(filename, [0, 20, 50, 100, 300])