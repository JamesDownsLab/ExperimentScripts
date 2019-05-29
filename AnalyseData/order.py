import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Generic import filedialogs
from ParticleTracking import dataframes


def duty_curve(df):
    duty = df.groupby(df.index).first().Duty.values
    fig, ax = plt.subplots()
    ax.plot(duty)
    ax.set_xlabel('frame')
    ax.set_ylabel('Duty / 1000')
    ax.set_title(file)
    plt.show()


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


def order_duty(df1, df2=None, labels=['up', 'down']):
    order_1 = df1.groupby('Duty')['order_mag'].mean()
    if df2 is not None:
        order_2 = df2.groupby('Duty')['order_mag'].mean()
    fig, ax = plt.subplots()
    ax.plot(order_1.index, order_1.values, label=labels[0])
    if df2 is not None:
        ax.plot(order_2.index, order_2.values, label=labels[1])
    ax.legend()
    ax.set_xlabel('Duty')
    ax.set_ylabel('Order')
    ax.set_title(file)
    plt.show()


def closest(arr, value):
    arr = np.asarray(arr)
    idx = (np.abs(arr - value)).argmin()
    return arr[idx]


def order_histograms(df, duties):
    df = df.reset_index().set_index('Duty')
    fig, ax = plt.subplots()
    for di in duties:
        d = closest(df.index, di)
        orders = df.loc[d, 'order_mag'].values
        freq, bins = np.histogram(orders, bins=100, density=True)
        ax.plot(bins[:-1], freq, label=d)
    ax.legend(title='Duty Cycle')
    ax.set_xlabel('Order')
    ax.set_ylabel('Frequency')
    ax.set_title(file)
    plt.show()


def order_histograms_no_edges(df, duties):
    df = df.reset_index().set_index('Duty')
    fig, ax = plt.subplots()
    for di in duties:
        d = closest(df.index, di)
        order_edge = df.loc[d, ['order_abs', 'on_edge']]
        orders = order_edge.reset_index().set_index('on_edge').loc[False, 'order_abs'].values
        print(orders)
        freq, bins = np.histogram(orders, bins=100, density=True)
        ax.plot(bins[:-1], freq, label=d)
    ax.legend(title='Duty Cycle')
    ax.set_xlabel('Order')
    ax.set_ylabel('Frequency')
    ax.set_title(file)
    plt.show()




if __name__ == "__main__":
    import matplotlib as mpl
    import os

    sns.set_context('notebook')
    file = filedialogs.load_filename('Load a dataframe',
                                     '/media/data/data',
                                     '*.hdf5')
    mpl.rcParams["savefig.directory"] = os.path.split(file)[0]
    data = dataframes.DataStore(file)
    # duty_curve(data.df)

    df_up, df_down = split_df(data.df)

    # order_duty(df_up, df_down)

    duties = list(range(400, 1050, 50))
    order_histograms(df_down, duties)
