from collections import namedtuple

import numpy as np

import nanoparticle_optimization as np_opt


ParamSet = namedtuple('ParamSet', 'sigma epsilon n m')

parameters = [ParamSet(sigma=0.6, epsilon=1.0880, n=30.0024, m=4.7342),
              ParamSet(sigma=0.8, epsilon=2.0881, n=42.5332, m=5.0203),
              ParamSet(sigma=1.0, epsilon=3.0911, n=35.5500, m=5.3311),
              ParamSet(sigma=1.2, epsilon=3.9924, n=31.3125, m=5.5569)]

for ps in parameters:
    forcefield = np_opt.Mie(sigma=np_opt.Parameter(value=ps.sigma, fixed=True),
                            epsilon=np_opt.Parameter(value=ps.epsilon, fixed=True),
                            n=np_opt.Parameter(value=ps.n, fixed=True),
                            m=np_opt.Parameter(value=ps.m, fixed=True))

    np_opt.test_all(forcefield, 'test-sigma{}'.format(ps.sigma))
