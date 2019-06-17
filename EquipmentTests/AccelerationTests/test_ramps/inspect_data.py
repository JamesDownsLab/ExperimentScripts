import numpy as np
import matplotlib.pyplot as plt
from Generic import filedialogs
import scipy.signal as ss
from tqdm import tqdm

#%%
direc = filedialogs.open_directory('select a directory')

#%%
duties = range(0, 1000, 1)
rms = []
rms_filtered = []
for d in tqdm(duties):
    times = np.loadtxt(direc+'/times_{}.txt'.format(d))
    freq = 1/(times[1]-times[0])
    but = ss.butter(1, 100, fs=freq, output='sos')
    voltage = np.loadtxt(direc+'/voltage_{}.txt'.format(d))
    voltage_filtered = ss.sosfilt(but, voltage)
    rms.append(np.sqrt(np.mean(voltage**2)))
    rms_filtered.append(np.sqrt(np.mean(voltage_filtered**2)))
    # plt.figure()
    # plt.plot(times, voltage)
    # plt.plot(times, voltage_filtered)
    # plt.title(str(d))
    # plt.show()

#%%
# plt.figure()
# plt.plot(duties, rms)
# plt.plot(duties, rms_filtered)
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

#%%
import scipy.optimize as op
fit, fit_p = op.curve_fit(sigmoid, duties, rms)
a, b, c, d = fit
sigmoid_fit = sigmoid(duties, a, b, c, d)
plt.figure()
plt.plot(duties, rms, label='data')
plt.plot(duties, sigmoid_fit, label='fit')
plt.xlabel('Duty Cycle / 1000')
plt.ylabel('Accelerometer Voltage (V)')
plt.legend()
plt.show()
plt.savefig(direc+'/plot.png')

print('a = ', a)
print('b = ', b)
print('c = ', c)
print('d = ', d)
