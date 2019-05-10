from Shaker import power

rates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for rate in rates:

    PS = power.PowerSupply()
    PS.ramp(400, 1000, rate, 1)
    PS.quit()

    PS = power.PowerSupply()
    PS.ramp(1000, 400, rate, 1)
    PS.quit()

