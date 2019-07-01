import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import optimize
from tqdm import tqdm

from Generic import filedialogs

sns.set()

# %%
direc = filedialogs.open_directory('select a directory')

# %%
rms_mean = []
rms_std = []
rms_filtered_mean = []
rms_filtered_std = []

duties = range(0, 1001, 1)
for d in tqdm(duties):
    rms = []
    rms_filtered = []
    for r in range(1, 6):
        times = np.loadtxt(direc + '/times_{}_r{}.txt'.format(d, r))
        voltage = np.loadtxt(direc + '/voltage_{}_r{}.txt'.format(d, r))
        voltage_fft = np.fft.fft(voltage)
        voltage_fft[50:-50] = 0
        voltage_filtered = np.fft.ifft(voltage_fft)
        rms.append(np.sqrt(np.mean(voltage ** 2)))
        rms_filtered.append(np.sqrt(np.mean(voltage_filtered ** 2)))
    rms_mean.append(np.mean(rms))
    rms_std.append(np.std(rms))
    rms_filtered_mean.append(np.mean(rms_filtered))
    rms_filtered_std.append(np.std(rms_filtered))

rms_mean = np.array(rms_mean).real
rms_std = np.array(rms_std).real
rms_filtered_mean = np.array(rms_filtered_mean).real
rms_filtered_std = np.array(rms_filtered_std).real

# %%
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(duties, rms_mean, label='unfiltered')
plt.subplot(1, 2, 2)
plt.errorbar(duties, rms_filtered_mean, rms_filtered_std, label='filtered')
plt.show()

# %%
np.savetxt(direc + '/rms_mean.txt', rms_mean)
np.savetxt(direc + 'rms_std.txt', rms_std)
np.savetxt(direc + 'rms_filtered_mean.txt', rms_filtered_mean)
np.savetxt(direc + 'rms_filtered_std.txt', rms_filtered_std)

# %%
duties = np.arange(0, 1001)
rms_mean = np.loadtxt(direc + '/rms_mean.txt')
rms_std = np.loadtxt(direc + 'rms_std.txt')
rms_filtered_mean = np.loadtxt(direc + 'rms_filtered_mean.txt')
rms_filtered_std = np.loadtxt(direc + 'rms_filtered_std.txt')

# %%
plt.figure()
plt.plot(duties, rms_filtered_mean)
plt.show()


# %%
def sigmoid(x, a, b, c, d):
    return b / (1 + np.exp(-a * (x - c))) + d


# %%
popt, pcov = optimize.curve_fit(sigmoid, duties, rms_filtered_mean)
fit = sigmoid(duties, *popt)

diff = np.sum((fit - rms_filtered_mean) ** 2)
print(diff)

plt.figure()
plt.plot(duties, rms_filtered_mean, label='data')
plt.plot(duties, fit, label='fit')
plt.xlabel('Duty Cycle / 1000')
plt.ylabel('Accelerometer Voltage (V)')
plt.legend()
plt.show()
# plt.savefig(direc + '/sigmoid_fit.png')

# %%
a, b, c, d = popt
ae, be, ce, de = np.sqrt(np.diag(pcov))

print(a, '+/-', ae)
print(b, '+/-', be)
print(c, '+/-', ce)
print(d, '+/-', de)

# %%
fit = fit - np.min(fit)
fit = fit / np.max(fit)

# %%
plt.figure()
plt.plot(duties / 10, fit)
plt.xlabel('Duty Cycle (%)')
plt.ylabel('Percentage of max acceleration')
plt.savefig(direc + '/duty_accel_percent.png')

# %%
np.savetxt(direc + '/fit.txt', fit)
