from __future__ import division

import dill
import pkg_resources
import sys

import numpy as np

import nanoparticle_optimization as np_opt


'''
----------
Statepoint
----------
'''
sigma_bead = 0.8
target_radius = 8.0

'''
--------------------
Optimization Details
--------------------
'''
gridpoints = 8
configurations = 10
r_dependent_sampling = True

'''
----------------
Load Target Data
----------------
'''
resource_package = np_opt.__name__
resource_path = '/'.join(('utils', 'U_{}nm_truncated.txt'.format(int(radius))))
target = np_opt.load_target(pkg_resources.resource_filename(resource_package,
                                                            resource_path))
target.separations /= 10 # Convert distances from angstroms to nanometers

'''
------------------------------
Create Two-Nanoparticle System
------------------------------
'''
nano = np_opt.CG_nano(radius, sigma=sigma_bead)
system = np_opt.System(nano)

'''
----------------------------------------
Define Force field Parameters and Bounds
----------------------------------------
'''
sigma = np_opt.Parameter(value=sigma_bead, fixed=True)
m = np_opt.Parameter(value=6.0, upper=7.0, lower=5.0)
epsilon = np_opt.Parameter(value=2.0, upper=4.0, lower=0.5)
n = np_opt.Parameter(value=20.0, upper=45.0, lower=12.0)
forcefield = np_opt.Mie(sigma=sigma, epsilon=epsilon, n=n, m=m)

'''
-----------------------------------
Define and Execute the Optimization
-----------------------------------
'''
optimization = np_opt.Optimization(forcefield=forcefield, systems=system,
                                   targets=target, configurations=configurations)
optimization.optimize(gridpoints=gridpoints, verbose=True, polishing_function=None,
                      r_dependent_sampling=r_dependent_sampling)

'''
--------------------------
Serialize the Optimization
--------------------------
'''
dill.dump(optimization, open('opt-sigma{}.p'.format(sigma_bead), 'wb'))
