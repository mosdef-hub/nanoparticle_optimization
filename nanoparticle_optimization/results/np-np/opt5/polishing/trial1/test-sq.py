from __future__ import division

import numpy as np
from scipy.optimize import curve_fit

import nanoparticle_optimization as np_opt


sigma = np.hstack((np.arange(0.9, 1.7, 0.1), np.arange(2.0, 5.0)))
epsilon = [2.41813, 2.87120, 3.40631, 4.11224, 4.43503, 5.29938, 5.68701,
                6.71327, 9.67500, 21.88478, 40.86000]
m = [5.13034, 5.23675, 5.33795, 5.48890, 5.53154, 5.65830, 5.69174, 5.81852,
          6.18750, 7.26483, 8.29855]

order1_fit = lambda x, a, b: a * x + b
order2_fit = lambda x, a, b: a * x**2 + b * x

e_popt, e_pcov = curve_fit(order2_fit, sigma, epsilon)
m_popt, m_pcov = curve_fit(order1_fit, sigma, m)

sigma_test = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0]

forcefields = []
for sigma in sigma_test:
    epsilon = order2_fit(sigma, *e_popt)
    m = order1_fit(sigma, *m_popt)
    n = 35
    forcefield = np_opt.Mie(sigma=np_opt.Parameter(value=sigma, fixed=True),
                            epsilon=np_opt.Parameter(value=epsilon, fixed=True),
                            n=np_opt.Parameter(value=n, fixed=True),
                            m=np_opt.Parameter(value=m, fixed=True))
    forcefields.append(forcefield)

np_opt.test_all(forcefields, tag='sqfit')
