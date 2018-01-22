from __future__ import division

import dill
import pkg_resources

import mbuild as mb
import numpy as np

import nanoparticle_optimization as np_opt


'''
----------
Statepoint
----------
'''
sigma_bead = 0.8
sigma_ch2 = 0.395
target_radius = 7.0

'''
--------------------
Optimization Details
--------------------
'''
gridpoints = 8
configurations = 100
r_dependent_sampling = False

'''
----------------
Load Target Data
----------------
'''
resource_package = np_opt.__name__
resource_path = '/'.join(('utils', 'U_{}nm_CH2_truncated.txt'.format(int(radius))))
target = np_opt.load_target(pkg_resources.resource_filename(resource_package,
                                                            resource_path))
target.separations /= 10 # Convert distances from angstroms to nanometers

'''
------------------------------
Create Nanoparticle-CH2 System
------------------------------
'''
nano = np_opt.CG_nano(radius, sigma=sigma_bead)
system = np_opt.System(mb.Compound(pos=np.zeros(3)), nano)

'''
----------------------------------------
Define Force field Parameters and Bounds
----------------------------------------
'''
sigma = np_opt.Parameter(value=(sigma_bead + sigma_ch2) / 2, fixed=True)
epsilon = np_opt.Parameter(value=0.5, upper=1.0, lower=0.1)
n = np_opt.Parameter(value=20.0, upper=45.0, lower=12.0)
m = np_opt.Parameter(value=6.0, upper=7.0, lower=5.0)
forcefield = np_opt.Mie(sigma=sigma, epsilon=epsilon, n=n, m=m)

'''
-----------------------------------
Define and Execute the Optimization
-----------------------------------
'''
optimization = np_opt.Optimization(forcefield=forcefield, systems=systems,
                                   targets=targets, configurations=configurations)
optimization.optimize(gridpoints=gridpoints, verbose=True, polishing_function=None,
                      r_dependent_sampling=r_dependent_sampling)

'''
--------------------------
Serialize the Optimization
--------------------------
'''
dill.dump(optimization, open('opt-sigma{}.p'.format(sigma_bead), 'wb'))
