from Generic import filedialogs, plotting
from ParticleTracking import dataframes

import numpy as np
import seaborn as sns

sns.set_context('notebook')

def closest(arr, value):
    arr = np.asarray(arr)
    idx = (np.abs(arr - value)).argmin()
    return arr[idx]

def order_histograms(df, duties, plotter, subplot):
    df = df.reset_index().set_index('Duty')
    for di in duties:
        d = closest(df.index, di)
        orders = df.loc[d, 'order_mag'].values
        freq, bins = np.histogram(orders, bins=100, density=True)
        plotter.add_plot(bins[:-1], freq, label=d)
        return plotter

def split_df(df, up_then_down=True):
    df_mean = df.groupby(df.index).mean()
    if up_then_down:
        duty_test = df_mean.Duty.max()
    else:
        duty_test = df_mean.Duty.min()
    frame_switch = round(df_mean.reset_index().loc[df_mean.Duty==duty_test, 'frame'].values.mean())
    df1 = df.loc[df.index < frame_switch]
    df2 = df.loc[df.index >= frame_switch]
    return df1, df2

def run(files):
    plotter = plotting.Plotter(subplot=(3, 2))
    duties = list(range(400, 1050, 50))
    for i, file in enumerate(files):
        print(i)
        with dataframes.DataStore(file) as data:
            df1, df2 = split_df(data.df)
            plotter = order_histograms(df1, duties, plotter, i)
            plotter.configure_legend(i)
            plotter.configure_xaxis(i, 'Order')
            plotter.configure_yaxis(i, 'Density')
            plotter.configure_subplot_title(i, file)
    plotter.show_figure()

if __name__ == "__main__":
    direc = filedialogs.open_directory()
    files = filedialogs.get_files_directory(direc + '/*.hdf5')
    run(files)