import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import signal, optimize
from tqdm import tqdm

from Generic import filedialogs

sns.set()

#%%
direc = filedialogs.open_directory('select a directory')

#%%
times = np.loadtxt(direc + '/times_452.txt')
voltage = np.loadtxt(direc + '/voltage_452.txt')
plt.figure()
plt.plot(times, voltage)
plt.show()

# %%
n0 = 850
n1 = 950
times0, times1 = (np.loadtxt(direc + '/times_{}.txt'.format(n0)),
                  np.loadtxt(direc + '/times_{}.txt'.format(n1)))
voltage0, voltage1 = (np.loadtxt(direc + '/voltage_{}.txt'.format(n0)),
                      np.loadtxt(direc + '/voltage_{}.txt'.format(n1)))
plt.figure()
plt.plot(times0, voltage0, label=n0)
plt.plot(times1, voltage1, label=n1)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.show()
# %%

sos = signal.bessel(1, 100, 'low', fs=1 / (times[1] - times[0]), output='sos')
filtered = signal.sosfilt(sos, voltage)

plt.figure()
plt.plot(filtered)
plt.plot(voltage)
plt.show()

# %%

voltage_fft = np.fft.fft(voltage)
fft_freq = np.fft.fftfreq(len(times), times[1] - times[0])
plt.figure()
plt.plot(fft_freq, voltage_fft, 'o-')
plt.show()

# %%
voltage_fft[50:-50] = 0
voltage_filtered = np.fft.ifft(voltage_fft)
plt.figure()
plt.plot(voltage_filtered)
plt.plot(voltage)
plt.show()

# %%
rms = []
rms_filtered = []
duties = range(0, 1001, 1)
for d in tqdm(duties):
    times = np.loadtxt(direc+'/times_{}.txt'.format(d))
    voltage = np.loadtxt(direc +'/voltage_{}.txt'.format(d))
    voltage_fft = np.fft.fft(voltage)
    voltage_fft[50:-50] = 0
    voltage_filtered = np.fft.ifft(voltage_fft)
    rms.append(np.sqrt(np.mean(voltage ** 2)))
    rms_filtered.append(np.sqrt(np.mean(voltage_filtered ** 2)))
# %%
rms = np.array(rms).real
rms_filtered = np.array(rms_filtered).real
duties = np.array(duties)
# %%
np.savetxt(direc + '/rms.txt', rms)
np.savetxt(direc + '/rms_filtered.txt', rms_filtered)
np.savetxt(direc + '/duties.txt', duties)

# %%
rms = np.loadtxt(direc + '/rms.txt')
rms_filtered = np.loadtxt(direc + '/rms_filtered.txt')
duties = np.loadtxt(direc + '/duties.txt')
#%%
# plt.figure()
# plt.plot(duties, rms)
# # plt.plot(duties, rms_filtered)
# plt.show()

plt.figure()
plt.plot(duties, rms, label='unfiltered')
plt.plot(duties, rms_filtered, label='filtered')
plt.legend()
plt.xlabel('Duty Cycle / 1000')
plt.ylabel('Accelerometer Voltage (V)')
plt.show()

#%%
def sigmoid(x, a, b, c, d):
    return b / (1 + np.exp(-a*(x-c))) + d

# %%

popt, pcov = optimize.curve_fit(sigmoid, duties, rms_filtered)
# a, b, c, d = popt
# ae, be, ce, de = np.diag(pcov)
sigmoid_fit = sigmoid(duties, *popt)
plt.figure()
plt.plot(duties, rms_filtered, label='data')
plt.plot(duties, sigmoid_fit, label='fit')
plt.xlabel('Duty Cycle / 1000')
plt.ylabel('Accelerometer Voltage (V)')
plt.legend()
plt.show()
plt.savefig(direc + '/sigmoid_fit.png')

# %%
print('a =', a, '+/-', ae)
print('b =', b, '+/-', be)
print('c =', c, '+/-', ce)
print('d =', d, '+/-', de)

# %%
sigmoid_fit -= np.min(sigmoid_fit)
sigmoid_fit_percentage = (sigmoid_fit / np.max(sigmoid_fit)) * 100
plt.figure()
plt.plot(duties, sigmoid_fit_percentage)
plt.xlabel('Duty Cycle / 1000')
plt.ylabel('Percentage of total acceleration / %')
plt.savefig(direc + '/percetage_of_total.png')

# %%
np.savetxt(direc + '/percentage_max.txt', sigmoid_fit_percentage)

#%%
