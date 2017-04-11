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
        sigma = self.sigma.value
        epsilon = self.epsilon.value
        n = self.n.value
        m = self.m.value

        C = (n / (n - m)) * ((n / m) ** (m / (n - m)))
        w = sigma / r

        return C * epsilon * ((w ** n) - (w ** m))

class Parameter(object):
    def __init__(self, value, upper=None, lower=None, fixed=False):
        super(Parameter, self).__init__()

        self.value = value
        self.fixed = fixed
        assert fixed or (upper is not None and lower is not None)
        if upper is not None:
            self.upper = upper
            self.lower = lower

if __name__ == "__main__":
    sigma = Parameter(value=0.8, upper=1.2, lower=0.6)
    epsilon = Parameter(value=0.4, upper=0.6, lower=0.3)
    n = Parameter(value=12, upper=25, lower=10)
    m = Parameter(value=6, fixed=True)

    ff = Mie(sigma=sigma, epsilon=epsilon, n=n, m=m)
