import matplotlib.pyplot as plt
import numpy as np

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
d_all = [600, 700, 800, 900]
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
r = []
g = []
g6 = []
for f in frames:
    ri, gi, g6i = calc.correlations(f)
    r.append(ri)
    g.append(gi)
    g6.append(g6i)

# %%
r = np.array(r)
g = np.array(g)
g6 = np.array(g6)

# %%
g_mean = np.mean(g, axis=0)
g6_mean = np.mean(g6, axis=0)
r_mean = r[0, :]

# %%
diameter = r_mean[np.argmax(g_mean)]
r_mean /= diameter

# %%
plt.subplot(1, 2, 1)
plt.plot(r_mean, g_mean - 1)
plt.xlabel('r/D')
plt.ylabel('G(r)')
plt.subplot(1, 2, 2)
plt.plot(r_mean, g6_mean / g_mean)
plt.xlabel('r/D')
plt.ylabel('G6(r)')

# %%
