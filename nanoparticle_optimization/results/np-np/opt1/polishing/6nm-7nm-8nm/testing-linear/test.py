from collections import namedtuple

import numpy as np

import nanoparticle_optimization as np_opt


epsilon_slope = 5.2095
epsilon_intercept = -2.14467
n_slope = -22.43975
n_intercept = 58.70408
m_slope = 1.39050
m_intercept = 3.89493

sigmas = [0.5, 0.75, 1.25, 1.5, 2.0]

forcefields = []
for sigma in sigmas:
    epsilon = epsilon_slope * sigma + epsilon_intercept
    n = n_slope * sigma + n_intercept
    m = m_slope * sigma + m_intercept
    forcefield = np_opt.Mie(sigma=np_opt.Parameter(value=sigma, fixed=True),
                            epsilon=np_opt.Parameter(value=epsilon, fixed=True),
                            n=np_opt.Parameter(value=n, fixed=True),
                            m=np_opt.Parameter(value=m, fixed=True))
    forcefields.append(forcefield)

np_opt.test_all(forcefields)
