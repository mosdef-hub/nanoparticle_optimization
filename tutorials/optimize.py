import dill
import pkg_resources
import sys

import numpy as np

import nanoparticle_optimization
from nanoparticle_optimization.forcefield import Mie, Parameter
from nanoparticle_optimization.lib.CG_nano import CG_nano
from nanoparticle_optimization.optimization import Optimization
from nanoparticle_optimization.system import System
from nanoparticle_optimization.target import load

grid_spacing = 6
sigma_bead = 1.0
target_radii = [5.0]

add_point = False
if len(sys.argv) > 1 and 'add_point' in sys.argv:
    add_point = True
norm_error = False
if len(sys.argv) > 1 and 'norm' in sys.argv:
    norm_error = True

targets = []
systems = []
for radius in target_radii:
    resource_package = nanoparticle_optimization.__name__
    resource_path = '/'.join(('utils', 'U_{}nm.txt'.format(int(radius))))
    target = load(pkg_resources.resource_filename(resource_package, resource_path))

    target.separations /= 10.0
    if add_point:
        bin_sep = target.separations[1] - target.separations[0]
        target.separations = np.insert(target.separations, 0,
            target.separations[0] - bin_sep)
        target.potential = np.insert(target.potential, 0, 0.0)
    targets.append(target)

    nano = CG_nano(radius, sigma=sigma_bead)
    system = System(nano)
    systems.append(system)

sigma = Parameter(value=sigma_bead, fixed=True)
m = Parameter(value=6.0, upper=10.0, lower=2.0)
epsilon = Parameter(value=5.0, upper=15.0, lower=1.0)
n = Parameter(value=15.0, upper=40.0, lower=10.0)
ff = Mie(sigma=sigma, epsilon=epsilon, n=n, m=m)

optimization = Optimization(forcefield=ff, systems=systems, targets=targets,
                            configurations=25, normalize_error=norm_error)
optimization.optimize(grid_spacing=grid_spacing, verbose=True, maxiter=50)

for name, param in optimization.forcefield:
    print('{}: {}\n'.format(name, param.value))
print('Residual: {}'.format(optimization.residual()))

if add_point and norm_error:
    dill.dump(optimization, open('opt-{}nm-point-norm.p'.format(radius),'wb'))
elif add_point:
    dill.dump(optimization, open('opt-{}nm-point-nonorm.p'.format(radius),'wb'))
elif norm_error:
    dill.dump(optimization, open('opt-{}nm-norm.p'.format(radius),'wb'))
else:
    dill.dump(optimization, open('opt-{}nm-nonorm.p'.format(radius),'wb'))
