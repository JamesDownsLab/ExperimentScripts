from Shaker import power
from Generic import filedialogs
from Generic.equipment import pico_scope as scope
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# power_supply = power.PowerSupply()
PicoScope = scope.Scope()

direc = filedialogs.open_directory('Create a directory')

ts = []
timestamp = []
rms = []
fig, ax = plt.subplots()
fig.autofmt_xdate()
myFmt = mdates.DateFormatter('%H:%M:%S')
ax.xaxis.set_major_formatter(myFmt)
plot = ax.plot(ts, rms)[0]
ax.set_xlabel('Time HH:MM:SS')
ax.set_ylabel('Voltage (V)')

for t in range(28800):
    time.sleep(1)
    times, data, _ = PicoScope.get_V(refine_range=True)
    rms.append(np.sqrt(np.mean(data ** 2)))
    dt = datetime.datetime.now()
    # ts.append(dt.hour + dt.minute/60 + dt.second/3600)
    ts.append(dt)
    timestamp.append(dt.timestamp())
    plot.set_xdata(ts)
    plot.set_ydata(rms)
    ax.relim()
    ax.autoscale_view()
    # ax.set_xlim(np.min(ts), np.max(ts))
    # ax.set_ylim(np.min(rms), np.max(rms))
    plt.draw()
    plt.pause(0.01)

plt.savefig(direc+'/figure.pdf')
np.savetxt(direc+'/times.txt', timestamp)
np.savetxt(direc+'/rms.txt', rms)
