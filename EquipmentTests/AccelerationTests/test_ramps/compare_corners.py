import numpy as np
import matplotlib.pyplot as plt

duties = np.arange(0, 1001, 1)/10

corners = [1, 2, 3, 4, 5, 6]
rms = []
for corner in corners:
    data = np.loadtxt('corner{}/rms.txt'.format(corner))
    plt.figure()
    plt.plot(duties, data)
    rms.append(data)
plt.show()
plt.figure()
for i, rmss in enumerate(rms):
    plt.plot(duties, rmss)
plt.legend(corners)
plt.xlabel('Duty Cycle %')
plt.ylabel('Voltage / V')
plt.show()