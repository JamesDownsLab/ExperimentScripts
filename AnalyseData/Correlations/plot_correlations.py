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

    g_env = envelope(r, g, r_peaks, g_peaks)
    g6_env = envelope(r, y, r6_peaks, g6_peaks)

    g_exp = fit_exponential(r_peaks, g_peaks, r)
    g6_exp = fit_exponential(r6_peaks, g6_peaks, r)

    plt.figure()
    plt.plot(r, g)
    plt.plot(r, g_exp)
    plt.show()

    plt.figure()
    plt.plot(r, y)
    plt.plot(r, g6_exp)
    plt.show()

def fit_exponential(x, y, x_data):
    popt, pcov = optimize.curve_fit(exponential, x, y)
    yfit = exponential(x_data, popt)
    return yfit

def exponential(x, b):
    return (max(x)/np.exp(1))*np.exp(b*x)


def envelope(x, y, xpeaks, ypeaks):
    f = interpolate.interp1d(xpeaks, ypeaks, kind='cubic', bounds_error=False)
    fit = f(x)
    return fit


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


