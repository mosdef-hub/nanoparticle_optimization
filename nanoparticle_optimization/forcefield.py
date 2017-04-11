from __future__ import division

from abc import ABCMeta, abstractmethod

from numba import jit
import numpy as np

class Forcefield(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def calc_potential(self, r):
        pass

class Mie(Forcefield):
    def __init__(self, sigma, epsilon, n, m):
        self.sigma = sigma
        self.epsilon = epsilon
        self.n = n
        self.m = m

    def calc_potential(self, r):
        sigma = self.sigma
        epsilon = self.epsilon
        n = self.n
        m = self.m

        C = (n / (n - m)) * ((n / m) ** (m / (n - m)))
        w = sigma / r

        return C * epsilon * ((w ** n) - (w ** m))
