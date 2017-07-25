from __future__ import division
from __future__ import print_function

from copy import deepcopy
from functools import partial

import numpy as np
from scipy.optimize import brute, fmin


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
        self.cut = None
        self.forcefield = forcefield
        self.normalize_error = normalize_error
        self.r_dependent_sampling = False
        self.systems = systems
        self.targets = targets

        self.verbose = False

        self.grid = None
        self.grid_residuals = None

    def optimize(self, brute_force=True, verbose=False, grid_spacing=10,
                 polishing_function=fmin, threads=1, cut=None, 
                 r_dependent_sampling=False, **kwargs):
        """ Optimize force field parameters via potential matching

        Force field parameters are optimized by matching the interaction potential
        of the nanoparticle system defined in the System class with a target defined
        in the Target class. A brute force optimization is performed along a grid,
        ensuring that the optimization does not get stuck in local minima. A
        "polishing" function is then used with the result of the brute force
        optimization as an initial state. All optimizations are performed utilizing
        Scipy's optimize module (https://docs.scipy.org/doc/scipy/reference/
        optimize.html#module-scipy.optimize).

        Parameters
        ----------
        brute_force : bool, optional, default=True
            Perform a brute force optimization, evaluating the error in interaction
            potential along a grid of force field values. See the documentation for
            Scipy's optimize.brute function (https://docs.scipy.org/doc/scipy-0.15.1/
            reference/generated/scipy.optimize.brute.html) for more details.
        verbose : bool, optional, default=False
            Output the value of the parameters to be optimized at each
            point in the optimization.
        grid_spacing : int, optional, default=10
            The number of grid points along each axis.  Only necessary when
            brute_force is True.
        polishing_function : callable, optional, default=scipy.optimize.fmin
            "Polishing" function that uses the result of the brute force
            minimization as an initial guess
        """
        self.cut = cut
        self.r_dependent_sampling = r_dependent_sampling
        if verbose:
            self.verbose = True
        param_names = tuple(name for name, param in self.forcefield if not 
            param.fixed)
        if polishing_function:
            polishing_function = partial(polishing_function, **kwargs)
        if brute_force:
            limits = tuple((param.lower, param.upper) for name, param in 
                self.forcefield if not param.fixed)
            if threads == 1:
                x0, fval, grid, Jout = brute(self._residual, ranges=limits,
                    args=param_names, full_output=True, Ns=grid_spacing,
                    finish=polishing_function)
            else:
                from .utils.parallel import parbrute
                x0, fval, grid, Jout = parbrute(self._residual, ranges=limits,
                    args=param_names, full_output=True, Ns=grid_spacing,
                    finish=polishing_function, threads=threads)
            self.grid = grid
            self.grid_residuals = Jout
        else:
            values = np.array([param.value for name, param in self.forcefield if not 
                param.fixed])
            opt_result = polishing_function(self._residual, x0=values,
                args=param_names)
        self.verbose = False

    def _residual(self, values, *param_names):
        if not param_names:
            param_names = tuple(name for name, param in self.forcefield if not 
            param.fixed)
        for param_name, value in zip(param_names, values):
            self.forcefield[param_name] = value

        if not all(constr() for constr in self.forcefield.constraints):
            if self.verbose:
                print('Forcefield constraint failed, penalizing residual\n\n')
            if self.normalize_error:
                return 1.0 * len(self.systems)
            else:
                return 1e4 * len(self.systems)

        residual = 0
        for system, target in zip(self.systems, self.targets):
            residual += system.calc_error(self.forcefield, target,
                configurations=self.configurations, norm=self.normalize_error,
                cut=self.cut, r_dependent_sampling=self.r_dependent_sampling)
        if self.verbose:
            for param_name, value in zip(param_names, values):
                print('{}: {}\n'.format(param_name, value))
            print('Residual: {}\n\n'.format(residual))
        return residual

    def residual(self):
        param_names = tuple(name for name, param in self.forcefield if not 
            param.fixed)
        values = np.array([param.value for name, param in self.forcefield if not 
            param.fixed])
        return self._residual(values, *param_names)

    def plot_heatmap(self, filename):
        # TODO: Warn if forcefield contains more than 2 parameters
        # TODO: Gather names of x and y variables from forcefield
        import matplotlib.pyplot as plt

        x = self.grid[0]
        x_spacing = x[1,0] - x[0,0]
        x -= x_spacing/2
        y = self.grid[1]
        y_spacing = y[0,0] - y[0,1]
        y -= y_spacing/2

        plt.pcolormesh(x, y, self.grid_residuals, cmap='viridis_r')
        plt.colorbar(label='Residual')
        plt.tight_layout()
        plt.savefig(filename)

if __name__ == "__main__":
    import pkg_resources

    import nanoparticle_optimization
    from nanoparticle_optimization.forcefield import Mie, Parameter
    from nanoparticle_optimization.lib.CG_nano import CG_nano
    from nanoparticle_optimization.system import System
    from nanoparticle_optimization.target import load

    sigma = Parameter(value=0.6, fixed=True)
    epsilon = Parameter(value=4.0, upper=15.0, lower=1.0)
    n = Parameter(value=18.0, upper=25.0, lower=5.0)
    m = Parameter(value=6.0, upper=10.0, lower=2.0)
    ff = Mie(sigma=sigma, epsilon=epsilon, n=n, m=m)

    nano = CG_nano(3.0, sigma=0.6)
    system = System(nano)

    resource_package = nanoparticle_optimization.__name__
    resource_path = '/'.join(('utils', 'U_3nm.txt'))
    target = load(pkg_resources.resource_filename(resource_package, resource_path))

    target.separations /= 10.0
    optimization = Optimization(ff, system, target, configurations=2)
    optimization.optimize(verbose=True, maxiter=10, grid_spacing=3)

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
