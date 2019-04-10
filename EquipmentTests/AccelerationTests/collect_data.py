from Shaker import power, scope
import numpy as np
import os

power_supply = power.PowerSupply()
PicoScope = scope.Scope()
direc = 'corner6'
if os.path.exists(direc) is False:
    os.mkdir(direc)

for d in range(0, 1001, 1):
    print(d)
    power_supply.change_duty(d)
    times, data, _ = PicoScope.get_V(refine_range=True)
    np.savetxt(direc+'/times_{}.txt'.format(d), times)
    np.savetxt(direc+'/voltage_{}.txt'.format(d), data)
power_supply.quit()