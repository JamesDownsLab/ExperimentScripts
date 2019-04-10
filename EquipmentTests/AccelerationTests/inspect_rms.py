import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import scipy.stats as stats

# duties_100 = np.loadtxt('duties.txt')
duties_1000 = np.loadtxt('corner1/duties_1000.txt')

# rms_100 = np.loadtxt('rms.txt')
rms_1000 = np.loadtxt('corner1/rms_1000.txt')

def sigmoid(x, a, b, c, d):
    return b / (1 + np.exp(-a*(x-c))) + d

fit, fit_p = op.curve_fit(sigmoid, duties_1000, rms_1000)
a, b, c, d = fit
aerr, berr, cerr, derr = np.sqrt(np.diag(fit_p))
sigmoid_fit = sigmoid(duties_1000, a, b, c, d)
print(a, '+/-', aerr)
print(b, '+/-', berr)
print(c, '+/-', cerr)
print(d, '+/-', derr)




chisq, p = stats.chisquare(rms_1000, sigmoid_fit)
print(chisq, p)


plt.figure()
plt.plot(duties_1000, rms_1000, 'x')
plt.plot(duties_1000, sigmoid_fit)
plt.xlabel('Duty Cycle / 1000')
plt.ylabel('RMS voltage / V')
plt.show()