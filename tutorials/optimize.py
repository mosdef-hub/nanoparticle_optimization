import dill
import numpy as np

from nanoparticle_optimization.forcefield import Mie, Parameter
from nanoparticle_optimization.lib.CG_nano import CG_nano
from nanoparticle_optimization.optimization import Optimization
from nanoparticle_optimization.system import System
from nanoparticle_optimization.target import load

grid_spacing = 6
sigma_bead = 1.0
target_radii = [5.0]

targets = []
systems = []
for radius in target_radii:
    target = load('/Users/asummers/Documents/Coarse-grained-nps/All-atom'
                  '/U-np-np/U_{}nm.txt'.format(int(radius)))
    target.separations /= 10.0
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
                            configurations=25)
optimization.optimize(grid_spacing=grid_spacing, verbose=True)

for name, param in optimization.forcefield:
    print('{}: {}\n'.format(name, param.value))
print('Residual: {}'.format(optimization.residual()))

dill.dump(optimization, open('optimization.p'.format(grid_spacing, sigma_bead),'wb'))
