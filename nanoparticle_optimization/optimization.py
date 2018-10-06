from __future__ import division
from __future__ import print_function

from copy import deepcopy
from functools import partial
import itertools
from math import ceil, floor
from warnings import warn

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
        self.sample_until = 0.1
        self.systems = systems
        self.targets = targets

        self.verbose = False

        self.grid = None
        self.grid_residuals = None

    def optimize(self, brute_force=True, verbose=False, gridpoints=10,
                 polishing_function=fmin, threads=1, cut=None, 
                 r_dependent_sampling=False, sample_until=0.1, **kwargs):
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
        gridpoints : int, optional, default=10
            The number of gridpoints along each axis.  Only necessary when
            brute_force is True.
        polishing_function : callable, optional, default=scipy.optimize.fmin
            "Polishing" function that uses the result of the brute force
            minimization as an initial guess
        """
        self.cut = cut
        self.r_dependent_sampling = r_dependent_sampling
        self.sample_until = sample_until
        if verbose:
            self.verbose = True
        params = sorted([param for param in self.forcefield if not param[1].fixed],
                        key=lambda param: param[0])
        param_names = tuple(param[0] for param in params)
        if polishing_function:
            polishing_function = partial(polishing_function, **kwargs)
        if brute_force:
            limits = tuple((param[1].lower, param[1].upper) for param in params)
            if threads == 1:
                x0, fval, grid, Jout = brute(self._residual, ranges=limits,
                    args=param_names, full_output=True, Ns=gridpoints,
                    finish=polishing_function)
            else:
                from .utils.parallel import parbrute
                x0, fval, grid, Jout = parbrute(self._residual, ranges=limits,
                    args=param_names, full_output=True, Ns=gridpoints,
                    finish=polishing_function, threads=threads)
            self.grid = grid
            self.grid_residuals = Jout
        else:
            values = np.array([param[1].value for param in params])
            opt_result = polishing_function(self._residual, x0=values,
                args=param_names)
        self.verbose = False

    def _residual(self, values, *param_names):
        if not param_names:
            params = sorted([param for param in self.forcefield
                             if not param[1].fixed], key=lambda param: param[0])
            param_names = tuple(param[0] for param in params)
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
                cut=self.cut, r_dependent_sampling=self.r_dependent_sampling, sample_until=sample_until)
        if self.verbose:
            for param_name, value in zip(param_names, values):
                print('{}: {}\n'.format(param_name, value))
            print('Residual: {}\n\n'.format(residual))
        return residual

    def residual(self):
        params = sorted([param for param in self.forcefield if not param[1].fixed],
                        key=lambda param: param[0])
        param_names = tuple(param[0] for param in params)
        values = tuple(param[1].value for param in params)
        return self._residual(values, *param_names)

    def plot_heatmap(self, filename, draw_box=False):
        # TODO: Warn if forcefield contains more than 2 parameters
        # TODO: Gather names of x and y variables from forcefield
        # TODO: Add `show` argument to show heatmaps in Jupyter notebooks
        # For now assuming first parameter is epsilon, second m, third n
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        cmap = 'Blues_r'

        if len(self.grid[0].shape) == 3:
            warn('Three varying parameters detected. Plotting multiple '
                 'heatmaps.')
            unique_n = np.sort(np.unique(self.grid[2].flatten()))
            fig, ax = plt.subplots(ceil(len(unique_n)**0.5),
                                   floor(len(unique_n)**0.5),
                                   figsize=(12, 12))
            for i, (ax, n) in enumerate(zip(ax.reshape(-1), unique_n)):
                x = self.grid[0, :, 0, 0]
                y = self.grid[1, 0, :, 0]
                ax.set_title('n={}'.format(n))
                ax.set_xlabel(r'$m$')
                ax.set_ylabel(r'$\epsilon$')
                ax.pcolormesh(y, x, self.grid_residuals[:, :, i],
                              cmap=cmap)
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array([])
            cb_ax = fig.add_axes([0.925, 0.1, 0.02, 0.8])
            cbar = fig.colorbar(sm, cax=cb_ax, label='Residual')
            plt.subplots_adjust(hspace=0.65, wspace=0.5)
            fig.savefig(filename)
        elif len(self.grid[0].shape) == 2:
            fig, ax = plt.subplots(figsize=(7, 6))
            x = self.grid[0]
            x_spacing = np.abs(x[1,0] - x[0,0])
            y = self.grid[1]
            y_spacing = np.abs(y[0,0] - y[0,1])

            ax.set_xlabel(r'$\epsilon, \mathrm{\frac{kcal}{mol}}$')
            ax.set_ylabel(r'$m$')
            heatmap = ax.pcolormesh(x, y, self.grid_residuals, cmap=cmap,
                                    zorder=1, shading='gouraud')
            minimum = np.array(np.unravel_index(self.grid_residuals.argmin(),
                                                self.grid_residuals.shape))
            if min(minimum) > 1 and max(minimum) < len(self.grid_residuals) - 1 and draw_box:
                rectx = x[minimum[0] - 2][0]
                recty = y[0][minimum[1] - 2]
                rect = patches.Rectangle((rectx, recty), 4 * x_spacing,
                                         4 * y_spacing, linewidth=2,
                                         edgecolor='w', facecolor='none',
                                         linestyle='-', zorder=4)
                ax.add_patch(rect)
                rect2 = patches.Rectangle((rectx, recty), 4 * x_spacing,
                                         4 * y_spacing, linewidth=1,
                                         edgecolor='k', facecolor='none',
                                         linestyle='-', zorder=5)
                ax.add_patch(rect2)
            #viridis_r = matplotlib.cm.get_cmap(cmap)
            '''
            ax.plot(x[minimum[0]][0],
                    y[0][minimum[1]],
                    color=viridis_r(self.grid_residuals.min()),
                    markersize=10 * 17/len(self.grid_residuals),
                    mew=1, mec='k', marker='o', zorder=6)
            '''
            '''
            ax.set_xticks(np.unique(x), minor=True)
            ax.set_yticks(np.unique(y), minor=True)
            '''
            ax.set_xlim(np.min(x), np.max(x))
            ax.set_ylim(np.min(y), np.max(y))
            '''
            ax.grid(color='w', linestyle='-', linewidth=0.5, which='minor',
                    zorder=2)
            '''
            points = np.array(list(itertools.product(np.unique(x), np.unique(y))))
            ax.scatter(x, y, c='k', s=75, zorder=2)
            ax.scatter(x, y, c=self.grid_residuals, s=40, cmap=cmap,
                       zorder=3)
            fig.colorbar(heatmap, label='Residual')
            fig.tight_layout()
            fig.savefig(filename)
        else:
            warn('Heatmap plotting is only supported for optimizations '
                 'with 2 or 3 varying parameters.')
