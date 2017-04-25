from __future__ import division

import dill
import pkg_resources

import matplotlib.pyplot as plt
import numpy as np

import nanoparticle_optimization
from nanoparticle_optimization.forcefield import Mie, Parameter
from nanoparticle_optimization.lib.CG_nano import CG_nano
from nanoparticle_optimization.system import System
from nanoparticle_optimization.target import load

nanoparticle_radius = 5.0

resource_package = nanoparticle_optimization.__name__
resource_path = '/'.join(('utils', 'U_{}nm.txt'.format(int(nanoparticle_radius))))

target = load(pkg_resources.resource_filename(resource_package, resource_path))
target.separations /= 10.0

nano = CG_nano(radius, sigma=1.0)
system = System(nano)

fig, ax = plt.subplots()

ax.plot(target.separations, target.potential, linestyle='None', marker='o',
        color='black')

optimization = dill.load(open('opt-spacing{}-sigma1.0.p'.format(spacing),'rb'))
system = optimization.systems[0]
ff = optimization.forcefield

U = np.array([pot[0] for pot in system.calc_potential(ff, target.separations, 
                                                      configurations=25)])

ax.plot(target.separations, U, marker='None', linestyle='-')

#ax.set_xlim(9.5, 16.0)
#ax.set_ylim(-120, 0)
ax.set_xlabel('r, nm')
ax.set_ylabel('U, kcal/mol')
#ax.set_aspect(0.03)
plt.tight_layout()
fig.savefig('potential-comparison.pdf')
