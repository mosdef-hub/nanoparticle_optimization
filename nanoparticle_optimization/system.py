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

    @jit
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
                traj.write('CG1\t{}\n'.format('\t'.join(['{}'*3]).format(*coord)))
            for coord in self.nanoparticle2.xyz:
                traj.write('CG2\t{}\n'.format('\t'.join(['{}'*3]).format(*coord)))

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
                    U_local = self.calc_potential_single_state(forcefield, False)
                U_sep.append(U_local)
            U.append((np.mean(U_sep), np.std(U_sep)))

        return U

if __name__ == "__main__":
    from nanoparticle_optimization.lib.CG_nano import CG_nano
    from forcefield import Mie

    nano = CG_nano(3.0, sigma=0.9)
    system = System(nano)
    ff = Mie(sigma=0.9, epsilon=0.4, n=12, m=6)
    system.calc_potential(forcefield=ff, separations=np.linspace(6,10,10))
