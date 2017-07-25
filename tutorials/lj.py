'''
This tutorial provides a basic example of how to perform an optimization.  Here,
we use as target data a standard 12-6 LJ potential.  We attempt to return back to
this potential by optimizing the interaction between two single-point particles.
'''

import mbuild as mb
import numpy as np
from scipy.optimize import fmin

import nanoparticle_optimization as np_opt

grid_spacing = 20

# Set up the target potential
sigma_target = np_opt.Parameter(value=5.0, fixed=True)
epsilon_target = np_opt.Parameter(value=5.0, fixed=True)
lj_target = np_opt.LJ(sigma=sigma_target, epsilon=epsilon_target)
r = np.linspace(4.7, 10.0, 50)
U = lj_target.calc_potential(r)
target = np_opt.Target(separations=r, potential=U)

# Set up optimization
sigma = np_opt.Parameter(value=9.5, upper=10.0, lower=1.0)
epsilon = np_opt.Parameter(value=1.5, upper=10.0, lower=1.0)
lj = np_opt.LJ(sigma=sigma, epsilon=epsilon)
point_particle = mb.Compound(pos=np.zeros(3))
system = np_opt.System(point_particle)
optimization = np_opt.Optimization(forcefield=lj, systems=system, targets=target,
                                   configurations=1)
optimization.optimize(grid_spacing=grid_spacing, verbose=False,
                      polishing_function=fmin)

for name, param in optimization.forcefield:
    print('{}: {}\n'.format(name, param.value))
print('Residual: {}'.format(optimization.residual()))

optimization.plot_heatmap('lj-heatmap.pdf')
