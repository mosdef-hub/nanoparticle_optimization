from __future__ import division
from __future__ import print_function

from copy import deepcopy

import numpy as np
from scipy.optimize import brute, least_squares


class Optimization(object):
    def __init__(self, forcefield, systems, targets, configurations=50,
                 normalize_error=True):
        super(Optimization, self).__init__()

        if not hasattr(systems, "__iter__"):
            systems = [systems]
        if not hasattr(targets, "__iter__"):
            targets = [targets]
        if len(systems) != len(targets):
            raise ValueError("The number of systems and target must be equal!")

        self.configurations = configurations
        self.forcefield = forcefield
        self.normalize_error = normalize_error
        self.systems = systems
        self.targets = targets

        self.verbose = False

        self.grid = None
        self.grid_residuals = None

    def optimize(self, algorithm='brute', verbose=False, grid_spacing=10):
        """
        Parameters
        ----------
        algorithm : str, optional, default='brute'
        verbose : bool, optional, default=False

        Other Parameters
        ----------------
        grid_spacing : int, optional, default=10
            Valid for 'brute' algorithm only
        """
        if verbose:
            self.verbose = True
        params = tuple(param for param in self.forcefield.__dict__ if not
            self.forcefield.__dict__[param].fixed)
        if algorithm == 'brute':
            limits = tuple((self.forcefield.__dict__[param].lower,
                self.forcefield.__dict__[param].upper) for param in params)
            x0, fval, grid, Jout = brute(self._brute_residual, ranges=limits,
                args=params, Ns=grid_spacing, full_output=True)
            self.grid = grid
            self.grid_residuals = Jout

        if algorithm == 'leastsq':
            values = np.array([self.forcefield.__dict__[param].value for 
                param in params])
            lower_limits = [self.forcefield.__dict__[param].lower for param in params]
            upper_limits = [self.forcefield.__dict__[param].upper for param in params]
            limits = tuple([lower_limits, upper_limits])
            if verbose:
                verbosity = 2
            else:
                verbosity = 0
            res_log = least_squares(self._leastsq_residual, x0=values,
                bounds=limits, args=(params), method='trf', verbose=verbosity)

    def _brute_residual(self, values, *params):
        for param, value in zip(params, values):
            if self.verbose:
                print('{}: {}\n'.format(param, value))
            self.forcefield.__dict__[param].value = value
        return self._residual()

    def _leastsq_residual(self, values, *params):
        for param, value in zip(params, values):
            self.forcefield.__dict__[param].value = value
        return self._residual_leastsq()

    def _residual(self):
        residual = 0
        for system, target in zip(self.systems, self.targets):
            # Need to introduce a penalty if n < m...
            residual += system.calc_error(self.forcefield, target,
                configurations=self.configurations, norm=self.normalize_error)
        return residual

    def _residual_leastsq(self):
        residual = []
        for system, target in zip(self.systems, self.targets):
            U = [potential[0] for potential in system.calc_potential(self.forcefield,
                 separations=target.separations, configurations=self.configurations)]
            for i, val in enumerate(U):
                error = abs(target.potential[i] - val) / (abs(target.potential[i]) + abs(val))
                residual.append(error)
        return np.asarray(residual)

if __name__ == "__main__":
    import pkg_resources

    import nanoparticle_optimization
    from nanoparticle_optimization.forcefield import Mie, Parameter
    from nanoparticle_optimization.lib.CG_nano import CG_nano
    from nanoparticle_optimization.system import System
    from nanoparticle_optimization.target import load

    sigma = Parameter(value=0.8, fixed=True)
    epsilon = Parameter(value=4.0, upper=15.0, lower=1.0)
    n = Parameter(value=18.0, upper=25.0, lower=10.0)
    m = Parameter(value=6.0, fixed=True)
    ff = Mie(sigma=sigma, epsilon=epsilon, n=n, m=m)

    nano = CG_nano(3.0, sigma=0.8)
    system = System(nano)

    resource_package = nanoparticle_optimization.__name__
    resource_path = '/'.join(('utils', '3nm_target-short.txt'))
    target = load(pkg_resources.resource_filename(resource_package, resource_path))

    target.separations /= 10.0
    optimization = Optimization(ff, system, target, configurations=2)
    optimization.optimize(algorithm='brute', verbose=True)

    '''
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    '''

    #optimization.driver()

    '''
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    '''
