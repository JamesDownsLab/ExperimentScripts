import numpy as np
import matplotlib.pyplot as plt
from Generic import filedialogs
import scipy.signal as ss

# for corner in [2, 3, 4, 5, 6]:
#
#     duties = range(0, 1001, 1)
#     rms = []
#     for d in duties:
#         times = np.loadtxt('corner{}/times_{}.txt'.format(corner, d))
#         freq = 1/(times[1]-times[0])
#         voltage = np.loadtxt('corner{}/voltage_{}.txt'.format(corner, d))
#         rms.append(np.sqrt(np.mean(voltage**2)))
#     np.savetxt('corner{}/duties.txt'.format(corner), duties)
#     np.savetxt('corner{}/rms.txt'.format(corner), rms)
#
#
direc = filedialogs.open_directory('select a directory')
duties = range(0, 1000, 1)
rms = []
rms_filtered = []
for d in duties:
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

plt.figure()
plt.plot(duties, rms)
plt.plot(duties, rms_filtered)
plt.show()





rms = []
for L in range(100, 1000, 50):
    rms.append(np.sqrt(np.mean(voltage[:-L]**2)))
plt.figure()
plt.plot(rms)
plt.show()

