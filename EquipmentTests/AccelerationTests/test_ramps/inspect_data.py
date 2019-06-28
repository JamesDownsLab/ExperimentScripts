import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, optimize
from tqdm import tqdm

from Generic import filedialogs

#%%
direc = filedialogs.open_directory('select a directory')

#%%
times = np.loadtxt(direc + '/times_500.txt')
voltage = np.loadtxt(direc + '/voltage_500.txt')
plt.figure()
plt.plot(times, voltage)
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
duties = range(0, 1001, 1)
for d in tqdm(duties):
    times = np.loadtxt(direc+'/times_{}.txt'.format(d))
    voltage = np.loadtxt(direc +'/voltage_{}.txt'.format(d))
    voltage_fft = np.fft.fft(voltage)
    voltage_fft[50:-50] = 0
    voltage_filtered = np.fft.ifft(voltage_fft)
    rms.append(np.sqrt(np.mean(voltage_filtered ** 2)))

# %%
rms = np.array(rms).real
duties = np.array(duties)
# %%
np.savetxt(direc + '/rms.txt', rms)
np.savetxt(direc + '/duties.txt', duties)

# %%
rms = np.loadtxt(direc + '/rms.txt')
duties = np.loadtxt(direc + '/duties.txt')
#%%
# plt.figure()
# plt.plot(duties, rms)
# # plt.plot(duties, rms_filtered)
# plt.show()

plt.figure()
plt.plot(duties, rms)
plt.xlabel('Duty Cycle / 1000')
plt.ylabel('Accelerometer Voltage (V)')
plt.show()

np.savetxt(direc+'/duties.txt', duties)
np.savetxt(direc+'/rms.txt', rms)

#%%
def sigmoid(x, a, b, c, d):
    return b / (1 + np.exp(-a*(x-c))) + d


# %%

popt, pcov = optimize.curve_fit(sigmoid, duties, rms.real)
a, b, c, d = popt
ae, be, ce, de = np.diag(pcov)
sigmoid_fit = sigmoid(duties, *popt)
plt.figure()
plt.plot(duties, rms, label='data')
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
