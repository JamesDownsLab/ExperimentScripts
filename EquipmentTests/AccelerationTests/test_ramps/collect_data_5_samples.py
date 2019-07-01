import os

import numpy as np

from Generic import filedialogs
from Generic.equipment import pico_scope as scope
from Shaker import power

power_supply = power.PowerSupply()
PicoScope = scope.Scope()

direc = filedialogs.open_directory('Create a directory')

if os.path.exists(direc) is False:
    os.mkdir(direc)

for d in range(0, 1001, 1):
    print(d)
    power_supply.change_duty(d)
    for repeat in range(1, 6):
        times, data, _ = PicoScope.get_V(refine_range=True)
        np.savetxt(direc + '/times_{}_r{}.txt'.format(d, repeat), times)
        np.savetxt(direc + '/voltage_{}_r{}.txt'.format(d, repeat), data)
power_supply.quit()
