from collections import namedtuple

import numpy as np

import nanoparticle_optimization as np_opt


ParamSet = namedtuple('ParamSet', 'sigma epsilon n m')

parameters = [ParamSet(sigma=0.6, epsilon=1.1388, n=32.4822, m=4.7332),
              ParamSet(sigma=0.8, epsilon=2.0299, n=40.8240, m=5.0024),
              ParamSet(sigma=1.0, epsilon=3.0509, n=36.1209, m=5.2953),
              ParamSet(sigma=1.2, epsilon=4.1137, n=31.8481, m=5.5586)]

forcefields = []
for ps in parameters:
    forcefield = np_opt.Mie(sigma=np_opt.Parameter(value=ps.sigma, fixed=True),
                            epsilon=np_opt.Parameter(value=ps.epsilon, fixed=True),
                            n=np_opt.Parameter(value=ps.n, fixed=True),
                            m=np_opt.Parameter(value=ps.m, fixed=True))
    forcefields.append(forcefield)

np_opt.test_all(forcefields)
