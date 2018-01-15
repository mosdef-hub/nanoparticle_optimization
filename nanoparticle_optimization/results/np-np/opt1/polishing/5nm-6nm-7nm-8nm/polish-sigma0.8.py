from __future__ import division

import dill
import os
from pkg_resources import resource_filename

import numpy as np

import nanoparticle_optimization as np_opt

'''
----------
Statepoint
----------
'''
sigma_bead = 0.8
radii = [5.0, 6.0, 7.0, 8.0]

'''
------------------
Load Optimizations
------------------
'''
targets = []
systems = []
grid_residuals = []
resource_package = np_opt.__name__
for radius in radii:
    opt_path = os.path.join('results', 'np-np', 'opt1', 'brute',
        '{}nm'.format(int(radius)), 'opt-sigma{}.p'.format(sigma_bead))
    local_optimization = dill.load(open(resource_filename(resource_package,
        opt_path), 'rb'))
    grid_residuals.append(local_optimization.grid_residuals.flatten())

    target_path = os.path.join('utils', 'target_data', 'np_np', 'truncated',
        'U_{}nm_truncated.txt'.format(int(radius)))
    target = np_opt.load_target(resource_filename(resource_package, target_path))
    target.separations /= 10
    targets.append(target)

    nano = np_opt.CG_nano(radius, sigma=sigma_bead)
    system = np_opt.System(nano)
    systems.append(system)

grid_residuals = np.sum(np.asarray(grid_residuals), axis=0)
epsilon_vals = local_optimization.grid[0].flatten()
n_vals = local_optimization.grid[2].flatten()
m_vals = local_optimization.grid[1].flatten()

epsilon_val = epsilon_vals[np.argmin(grid_residuals)]
n_val = n_vals[np.argmin(grid_residuals)]
m_val = m_vals[np.argmin(grid_residuals)]

sigma = np_opt.Parameter(value=sigma_bead, fixed=True)
epsilon = np_opt.Parameter(value=epsilon_val, upper=4.0, lower=0.5)
n = np_opt.Parameter(value=n_val, upper=35.0, lower=12.0)
m = np_opt.Parameter(value=m_val, upper=7.0, lower=5.0)
forcefield = Mie(sigma=sigma, epsilon=epsilon, n=n, m=m)

optimization = np_opt.Optimization(forcefield=forcefield, systems=systems,
                                   targets=targets, configurations=10)
optimization.optimize(brute_force=False, verbose=True, r_dependent_sampling=True,
                      maxiter=50)

for name, param in optimization.forcefield:
    print('{}: {}\n'.format(name, param.value))
print('Residual: {}'.format(optimization.residual()))

dill.dump(optimization, open('opt-sigma{}.p'.format(sigma_bead), 'wb'))
