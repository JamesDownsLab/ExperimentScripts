import time

from Shaker import power

ps = power.PowerSupply()

# %%
ps.change_duty(850)
time.sleep(5)
ps.ramp(850, 700, 0.1, stop_at_end=False, record=0)

# %%

val = 700
for i in range(50):
    ps.ramp(850, 700, 0.1, stop_at_end=False, record=0)
    ps.init_duty(val)
    time.sleep(5)
    ps.init_duty(val)
    time.sleep(60)
