import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, optimize

from Generic import filedialogs
from ParticleTracking import dataframes, statistics

# %%
file = filedialogs.load_filename()
data = dataframes.DataStore(file)

# %%
calc = statistics.PropertyCalculator(data)

# %%
duty = calc.duty()
frame_max = duty.idxmax()

# %%
duty = duty.loc[:frame_max]
# %%
r_all = []
g_all = []
g6_all = []
d_all = [800, 851, 900, 950, 980]
for d in d_all:
    print(d)
    frames = duty.index[duty.values == d]
    r, g, g6 = zip(*[calc.correlations(f) for f in frames])
    r_all.append(np.mean(r, axis=0))
    g_all.append(np.mean(g, axis=0))
    g6_all.append(np.mean(g6, axis=0))

# %%
r_all = np.array(r_all)
g_all = np.array(g_all)
g6_all = np.array(g6_all)
G = g_all - 1
G6 = g6_all / g_all

# %%
plt.figure()
for r, g in zip(r_all, G6):
    plt.plot(r, g)


# %%
plt.loglog(r, g)

# %%
i = 3
r = r_all[i]
g6 = G6[i]
plt.plot(r, r ** (-1 / 4) + 0.2)
plt.plot(r, np.exp(-r / 30))
plt.plot(r, g6)


# %%
def power_law(x, a, b):
    return a * x ** (-1 / b)


def exponential(x, a, b):
    return a * np.exp(-x / b)


# %%
fig, ax = plt.subplots(3, 2)
ax = ax.reshape(6, )
for i, (r, g6, d) in enumerate(zip(r_all, G6, d_all)):
    peaks, props = signal.find_peaks(g6, width=10)
    r_peaks = r[peaks].real
    g6_peaks = g6[peaks].real
    popt_power, pcov_power = optimize.curve_fit(power_law, r_peaks, g6_peaks,
                                                p0=[g6_peaks.max(), 4])
    popt_exp, pcov_exp = optimize.curve_fit(exponential, r_peaks, g6_peaks,
                                            p0=[g6_peaks.max(), 20])

    power_fit = power_law(r, *popt_power)
    exp_fit = exponential(r, *popt_exp)
    ax[i].plot(r, g6, label='data')
    ax[i].plot(r, power_fit, label='power fit')
    ax[i].plot(r, exp_fit, label='exponential fit')
    ax[i].legend()
    ax[i].set_title(d)
