from Generic import plotting
from ParticleTracking import dataframes
from scipy import signal, optimize, interpolate
import numpy as np
import matplotlib.pyplot as plt

def corr(file, frame):
    corr_data = dataframes.CorrData(file)

    r = corr_data.get_row(frame, 'r')
    g = corr_data.get_row(frame, 'g')
    g6 = corr_data.get_row(frame, 'g6')
    y = g6/g

    r_peaks, g_peaks = find_peaks(r, g)
    r6_peaks, g6_peaks = find_peaks(r, y)

    plot_envelope(r, g, r_peaks, g_peaks)
    plot_envelope(r, y, r6_peaks, g6_peaks)

def plot_envelope(x, y, xpeaks, ypeaks):
    f = interpolate.interp1d(xpeaks, ypeaks, kind='cubic', bounds_error=False)
    fit = f(x)
    plt.figure()
    plt.plot(x, y, '.')
    plt.plot(x, fit)
    plt.show()

def find_peaks(xdata, ydata, plot=False):
    peaks, _ = signal.find_peaks(ydata)
    xpeaks = xdata[peaks]
    ypeaks = ydata[peaks]
    f = interpolate.interp1d(xpeaks, ypeaks, kind='cubic', bounds_error=False,
                             fill_value=(ydata[0], ydata[-1]))
    yinter = f(xdata)
    peaks, _ = signal.find_peaks(yinter)
    proms = signal.peak_prominences(yinter, peaks)[0]
    prom_threshold = 2*np.median(proms)
    peaks, _ = signal.find_peaks(yinter, prominence=prom_threshold)

    xpeaks = xdata[peaks]
    ypeaks = yinter[peaks]
    if plot:
        plt.figure()
        plt.plot(xdata, ydata)
        plt.plot(xdata, yinter)
        plt.plot(xpeaks, ypeaks, 'ro')
        plt.show()
    return xpeaks, ypeaks


