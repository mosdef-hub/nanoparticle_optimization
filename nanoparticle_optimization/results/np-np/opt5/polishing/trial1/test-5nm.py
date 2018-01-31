from __future__ import division

import numpy as np
from scipy.optimize import curve_fit

import nanoparticle_optimization as np_opt
from nanoparticle_optimization.utils.testing import test_25nm


sigma = [0.4, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
epsilon = [0.36192, 1.39136, 2.41813,
           2.87120, 3.40631, 4.11224, 4.43503, 5.29938, 5.68701, 6.71327, 6.96366,
           7.05099, 7.62263, 7.65790]
m = [4.34007, 4.83336, 5.13034,
     5.23675, 5.33795, 5.48890, 5.53134, 5.65830, 5.69174, 5.81852, 5.83916,
     5.84729, 5.88147, 5.88346]

linear_fit = lambda x, a, b: a * x + b
e_popt, e_pcov = curve_fit(linear_fit, sigma, epsilon)

log_fit = lambda x, a, b: a * np.log(x) + b
m_popt, m_pcov = curve_fit(log_fit, sigma, m)

n_test = [35, 45, 55, 65, 75, 85]

forcefields = []
for n in n_test:
    sigma = 5.0
    epsilon = linear_fit(sigma, *e_popt)
    m = log_fit(sigma, *m_popt)
    forcefield = np_opt.Mie(sigma=np_opt.Parameter(value=sigma, fixed=True),
                            epsilon=np_opt.Parameter(value=epsilon, fixed=True),
                            n=np_opt.Parameter(value=n, fixed=True),
                            m=np_opt.Parameter(value=m, fixed=True))
    forcefields.append(forcefield)

test_25nm(forcefields, tag='cgff5')
