from __future__ import division
from __future__ import print_function

from copy import deepcopy

import numpy as np
from openmdao.api import Component, IndepVarComp


class Optimization(Component):
    def __init__(self, forcefield):
        super(Optimization, self).__init__()

        self.add_output('residual', val=0.0)

    def driver(self):
        root = Group()
        root.add('p1', IndepVarComp(''))

    def error(self, target_data):

    def solve_nonlinear(self, params, unknowns, residual):

if __name__ == "__main__":
    from nanoparticle_optimization.forcefield import Mie

    ff = Mie()
