from copy import deepcopy
from random import random

from numba import jit
import numpy as np
from scipy.spatial import distance

from mbuild.coordinate_transform import Rotation


class System(object):
    """A system of two nanoparticles.

    Defines a system of two nanoparticles to calculate the interaction between.

    Parameters
    ----------
    nanoparticle : mb.Compound
        The nanoparticle prototype.
    nanoparticle2 : mb.Compound, optional, default=None
        A second nanoparticle prototype.  Should be defined if desiring to calculate
        the interaction potential between two dissimilar nanoparticles.
    """
    def __init__(self, nanoparticle, nanoparticle2=None):
        super(System, self).__init__()

        self.xyz = nanoparticle.xyz
        if nanoparticle2:
            self.xyz2 = nanoparticle2.xyz
        else:
            self.xyz2 = deepcopy(self.xyz)

    def generate_configuration(self, separation=None):
        if separation:
            self._translate_to(np.array([separation, 0, 0]))
        self._spin(np.random.random() * np.pi * 2, np.random.random(3))

    def calc_potential_single_state(self, forcefield, trajectory=False):
        xyz1 = self.xyz
        xyz2 = self.xyz2

        dists = np.ravel(distance.cdist(xyz1, xyz2, 'euclidean'))

        if trajectory:
            traj = open(trajectory, 'a')
            traj.write('{}\n\n'.format(len(xyz1) + len(xyz2)))
            for coord in xyz1:
                traj.write('CG1\t{}\n'.format('\t'.join(map(str,coord))))
            for coord in xyz2:
                traj.write('CG2\t{}\n'.format('\t'.join(map(str,coord))))

        return np.sum(forcefield.calc_potential(dists))

    @jit
    def calc_potential_single_state_fast(self, forcefield):
        xyz1 = self.xyz
        xyz2 = self.xyz2

        dists = np.ravel(distance.cdist(xyz1, xyz2, 'euclidean'))

        return np.sum(forcefield.calc_potential(dists))

    def calc_potential(self, forcefield, separations, configurations=50,
                       trajectory=False, frequency=5):
        """
        Returns
        -------
        U : list of tuples of floats, size = (n,2)
            The mean and standard deviation (provided as a tuple) of the interaction
            potential calculated over a specified number of configuration along a
            series of separations.
        """

        U = []

        if trajectory:
            open(trajectory, 'w')

        for sep in separations:
            U_sep = []
            for i, config in enumerate(range(configurations)):
                if i == 0:
                    self.generate_configuration(separation=sep)
                else:
                    self.generate_configuration()
                if trajectory and i % frequency == 0:
                    U_local = self.calc_potential_single_state(forcefield, trajectory)
                else:
                    U_local = self.calc_potential_single_state_fast(forcefield)
                U_sep.append(U_local)
            U.append((np.mean(U_sep), np.std(U_sep)))

        return U

    def calc_error(self, forcefield, target, configurations=50, norm=True):
        U = [potential[0] for potential in self.calc_potential(forcefield, 
                                                separations=target.separations,
                                                configurations=configurations)]
        error = sum(abs(target.potential - U))
        if norm:
            error /= sum(abs(target.potential) + abs(np.asarray(U)))

        return error

    def _spin(self, theta, around):
        center = np.mean(self.xyz2, axis=0)
        self.xyz2 -= center
        self.xyz2 = Rotation(theta, around).apply_to(self.xyz2)
        self.xyz2 += center

    def _translate_to(self, pos):
        center = np.mean(self.xyz2, axis=0)
        self.xyz2 += pos - center

if __name__ == "__main__":
    from nanoparticle_optimization.forcefield import Mie, Parameter
    from nanoparticle_optimization.lib.CG_nano import CG_nano
    from nanoparticle_optimization.target import load

    nano = CG_nano(3.0, sigma=0.8)
    system = System(nano)

    sigma = Parameter(value=0.8, upper=1.2, lower=0.6)
    epsilon = Parameter(value=0.4, upper=0.6, lower=0.3)
    n = Parameter(value=12, upper=25, lower=10)
    m = Parameter(value=6, fixed=True)
    ff = Mie(sigma=sigma, epsilon=epsilon, n=n, m=m)

    U = system.calc_potential(ff, np.linspace(6, 10, 10), configurations=10, trajectory='traj.xyz', frequency=1)

    '''
    target = load('target.txt')
    error = system.calc_error(ff, target)
    '''
