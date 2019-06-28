import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Generic import filedialogs
from ParticleTracking import dataframes, statistics

sns.set_context('notebook')

# %%
file = filedialogs.load_filename(file_filter='*.hdf5')
data = dataframes.DataStore(file)


# %%
def split_df(df, up_then_down=True):
    df_mean = df.groupby(df.index).mean()
    if up_then_down:
        duty_test = df_mean.Duty.max()
    else:
        duty_test = df_mean.Duty.min()
    frame_switch = round(df_mean.reset_index().loc[
                             df_mean.Duty == duty_test, 'frame'].values.mean())
    df1 = df.loc[df.index < frame_switch]
    df2 = df.loc[df.index >= frame_switch]
    return df1, df2


# %%
df_up = split_df(data.df)[0]

# %%

import cv2
import math

boundary = data.metadata['boundary']
tray_area = cv2.contourArea(boundary.astype(np.float32))  # pixels
mm2_per_pixel = 41916 / tray_area
mm_per_pixel = math.sqrt(mm2_per_pixel)
diameter = 4 / mm_per_pixel


# %%
def closest(arr, value):
    arr = np.asarray(arr)
    idx = (np.abs(arr - value)).argmin()
    return idx


# %%
calculator = statistics.PropertyCalculator(data)
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
for d in [400, ]:
    frames = df_up.loc[df_up.Duty == d].index.tolist()
    frames = np.unique(frames)
    r, g, g6 = calculator.correlations(frames[0])
    ax1.loglog(r / diameter, g - 1, label=str(d))
    ax2.loglog(r / diameter, g6 / g, label=str(d))
ax1.loglog(r / diameter, max(g) * r ** (-1 / 3))
ax2.loglog(r / diameter, 2 * r ** (-1 / 4))
ax1.set_xlabel('r / D')
ax1.set_ylabel('G(r)')
ax2.set_xlabel('r / D')
ax2.set_ylabel('$G_6(r)$')
plt.show()

# %%
