from Shaker import power

rates = [1]

for rate in rates:

    PS = power.PowerSupply()
    PS.ramp_up_and_down(400, 1000, 1, 1)
    PS.quit()
