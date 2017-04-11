from random import random

from numba import jit
import numpy as np
from scipy.spatial import distance

import mbuild as mb

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

        self.nanoparticle = nanoparticle
        if nanoparticle2:
            self.nanoparticle2 = nanoparticle2
        else:
            self.nanoparticle2 = mb.clone(nanoparticle)

    @jit
    def generate_configuration(self, separation=None):
        if separation:
            self.nanoparticle2.translate_to([separation, 0, 0])
        self.nanoparticle2.spin(np.random.random() * np.pi * 2, np.random.random(3))

    def calc_potential_single_state(self, forcefield, trajectory=False):
        np1 = self.nanoparticle
        np2 = self.nanoparticle2

        dists = np.ravel(distance.cdist(np1.xyz, np2.xyz, 'euclidean'))
        U = 0
        for dist in dists:
            U += forcefield.calc_potential(dist)

        if trajectory:
            traj = open(trajectory, 'a')
            traj.write('{}\n\n'.format(self.nanoparticle.n_particles +
                                       self.nanoparticle2.n_particles))
            for coord in self.nanoparticle.xyz:
                traj.write('CG1\t{}\n'.format('\t'.join(map(str,coord))))
            for coord in self.nanoparticle2.xyz:
                traj.write('CG2\t{}\n'.format('\t'.join(map(str,coord))))

        return U

    @jit
    def calc_potential_single_state_fast(self, forcefield):
        np1 = self.nanoparticle
        np2 = self.nanoparticle2

        dists = np.ravel(distance.cdist(np1.xyz, np2.xyz, 'euclidean'))
        U = 0
        for dist in dists:
            U += forcefield.calc_potential(dist)

        return U

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
                if trajectory and i % frequency:
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

    target = load('target.txt')
    error = system.calc_error(ff, target)
