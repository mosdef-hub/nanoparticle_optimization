from __future__ import division

from abc import ABCMeta, abstractmethod
from six import string_types

import numpy as np


class Forcefield(object):
    """A metaclass for a Forcefield object

    `Forcefield` is a metaclass that provides the framework for defining
    various force fields. Two abstract methods `calc_potential` and
    `add_constraint` are defined, which should be overridden when inheriting
    from this class.

    Attributes
    ----------
    constraints : list-like of functions
        List of constraints on force field parameters. Each constraint should
        should be a function that returns a boolean

    """
    __metaclass__ = ABCMeta
    def __init__(self):
        self.constraints = []

    @abstractmethod
    def calc_potential(self, r):
        """Potential energy as a function of separation

        Parameters
        ----------
        r : list-like of floats
            List of inter-particle distance values
        """
        pass

    def add_constraint(self, constraint):
        """Add a constraint to the force field

        Parameters
        ----------
        constraint : function returning a boolean
            A function defining a constraint on force field parameters.
        """
        self.constraints.append(constraint)

    def __getitem__(self, selection):
        if isinstance(selection, string_types):
            return self.__dict__[selection].value

    def __iter__(self):
        for name, properties in self.__dict__.items():
            if isinstance(properties, Parameter):
                yield name, properties

    def __setattr__(self, name, value):
        if isinstance(value, (int, float)):
            self[name] = value
        else:
            self.__dict__[name] = value

    def __setitem__(self, key, value):
        self.__dict__[key].value = value

class Mie(Forcefield):
    """Mie potential function

    Parameters
    ----------
    sigma : float
        Determines the location of the potential well
    epsilon : float
        Depth of the potential well
    n : float
        Repulsive exponent
    m : float
        Attractive exponent
    """
    def __init__(self, sigma, epsilon, n, m):
        self.sigma = sigma
        self.epsilon = epsilon
        self.n = n
        self.m = m

        super(Mie, self).__init__()

        self.add_constraint(self.constr1)

    def calc_potential(self, r):
        if not hasattr(r, "__iter__"):
            r = [r]

        sigma = self.sigma.value
        epsilon = self.epsilon.value
        n = self.n.value
        m = self.m.value

        C = (n / (n - m)) * ((n / m) ** (m / (n - m)))

        return C * epsilon * (((sigma/r) ** n) - ((sigma/r) ** m))

    def constr1(self):
        """Constrain `n` to be larger than `m`
        """
        return self.n > self.m


class LJ(Forcefield):
    """12-6 Lennard-Jones potential function

    Parameters
    ----------
    sigma : float
        Determines the location of the potential well
    epsilon : float
        Depth of the potential well
    """
    def __init__(self, sigma, epsilon):
        self.sigma = sigma
        self.epsilon = epsilon

        super(LJ, self).__init__()

    def calc_potential(self, r):
        if not hasattr(r, "__iter__"):
            r = [r]

        sigma = self.sigma.value
        epsilon = self.epsilon.value

        return 4.0 * epsilon * (((sigma/r) ** 12) - ((sigma/r) ** 6))


class LJ_general(Forcefield):
    """Generalized Lennard-Jones potential function

    Parameters
    ----------
    sigma : float
        Determines the location of the potential well
    epsilon : float
        Related to the depth of the potential well
    n : float
        Repulsive exponent
    m : float
        Attractive exponent
    """
    def __init__(self, sigma, epsilon, n, m):
        self.sigma = sigma
        self.epsilon = epsilon
        self.n = n
        self.m = m

        super(LJ_general, self).__init__()

        self.add_constraint(self.constr1)

    def calc_potential(self, r):
        if not hasattr(r, "__iter__"):
            r = [r]

        sigma = self.sigma.value
        epsilon = self.epsilon.value
        n = self.n.value
        m = self.m.value

        return 4.0 * epsilon * (((sigma/r) ** n) - ((sigma/r) ** m))

    def constr1(self):
        """Constrain `n` to be larger than `m`
        """
        return self.n > self.m

class VDW(Forcefield):
    """van der Waals potential function

    Parameters
    ----------
    C : float
    m : float
    """
    def __init__(self, C, m):
        self.C = C
        self.m = m

        super(VDW, self).__init__()

    def calc_potential(self, r):
        if not hasattr(r, "__iter__"):
            r = [r]

        C = self.C.value
        m = self.m.value

        return C / (r ** m)

class Yukawa(Forcefield):
    """Yukawa potential function

    Parameters
    ----------
    C : float
    kappa : float
    """
    def __init__(self, C, kappa):
        self.C = C
        self.kappa = kappa

        super(Yukawa, self).__init__()

    def calc_potential(self, r):
        if not hasattr(r, "__iter__"):
            r = [r]

        C = self.C.value
        kappa = self.kappa.value

        return C * np.exp(-kappa * r) / r

class Parameter(object):
    """Defines a force field Parameter object

    Parameters
    ----------
    value : float
        The value of this parameter.
    upper : float, optional, default=None
        Defines an upper-bound for the Parameter value
    lower : float, optional, default=None
        Defines a lower-bound for the Parameter value
    fixed : bool, optional, default=False
        Whether or not a Parameter value should be fixed during optimization
    """
    def __init__(self, value, upper=None, lower=None, fixed=False):
        super(Parameter, self).__init__()

        self.value = value
        self.fixed = fixed
        if not (fixed or (upper is not None and lower is not None)):
            raise ValueError("Must set fixed=True or define upper and lower bounds")
        if upper is not None:
            self.upper = upper
            self.lower = lower

    # Overwrite comparisons to compare Parameter.value
    def __eq__(self, other):
        return self.value == other.value

    def __ne__(self, other):
        return self.value != other.value

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value
