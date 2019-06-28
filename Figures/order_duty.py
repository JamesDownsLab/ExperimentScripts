import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Generic import filedialogs
from ParticleTracking import dataframes

sns.set_context('notebook')
# %%
file = filedialogs.load_filename(file_filter='*.hdf5')
data = dataframes.DataStore(file)

# %%
# from ParticleTracking import statistics
# calculator = statistics.PropertyCalculator(data)
# calculator.order()
# %%
duty = data.df.groupby('frame')['Duty'].first().plot()

# %%
data.df['order_mag'] = np.abs(data.df.order_r + 1j * data.df.order_i)


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
df_up, df_down = split_df(data.df)

# %%
group_up = df_up.groupby('Duty')['order_mag'].mean()
group_down = df_down.groupby('Duty')['order_mag'].mean()

# %%
duty_up = group_up.index.values
order_up = group_up.values

duty_down = group_down.index.values
order_down = group_down.values
# %%
fig, ax = plt.subplots()
ax.plot(duty_up / 10, order_up, label='up')
ax.plot(duty_down / 10, order_down, label='down')
ax.set_xlabel('Duty / %')
ax.set_ylabel(r'$\langle\psi_6 \rangle$')
ax.legend()
