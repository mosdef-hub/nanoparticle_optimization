from __future__ import division

import math

import mbuild as mb
import numpy as np
from scipy.spatial import distance


def _fast_sphere_pattern(n, radius):
    """Faster version of mBuild's SpherePattern. """
    phi = (1 + np.sqrt(5)) / 2
    long_incr = 2*np.pi / phi
    dz = 2.0 / float(n)
    bands = np.arange(n)
    z = bands * dz - 1.0 + (dz/2.0)
    r = np.sqrt(1.0 - z*z)
    az = bands * long_incr
    x = r * np.cos(az)
    y = r * np.sin(az)
    points = np.column_stack((x, y, z)) * np.asarray([radius])

    return points


class CG_nano(mb.Compound):
    """Nanoparticle constructed as a shell of evenly-spaced beads.

    Parameters
    ----------
    radius : float
        The van der Waals radius of the nanoparticle. This takes into
        consideration the diameter of the beads. For example, if a radius
        of 5nm is specified with a bead diameter of 0.5nm, then the bead
        centers will be located on a sphere with a radius of 4.75nm.
    bead_diameter : float
        The diameter of the beads used to compose the nanoparticle. Depending
        on the force field used, this may be equivalent to the sigma of the
        bead-bead interactions.
    bvf : float, optional, default=None
        If not `None`, specifies the volume fraction of beads within the
        spherical nanoparticle shell. This is controlled by altering the
        bead spacing. If not specified (which is the default) beads will
        be placed in an arrangement of maximum surface density without
        overlaps (where overlaps are determined based on the bead diameter).
        If `bvf` is specified overlaps or extra spacing between beads will
        occur to match the desired value.
    force_build : boolean, optional, default=False
        By default if a nanoparticle is expected to contain more than 5e5 beads
        the script will exit with an error (as this can lead to memory issues).
        Set this parameter to `True` to override this behavior.
    """
    def __init__(self, radius, bead_diameter, bvf=None, force_build=False):
        super(CG_nano, self).__init__()

        r_CG = sigma / 2
        r_silica = 0.40323 / 2

        r = r - r_CG + r_silica

        '''
        Approximate the number of beads required to construct the nanoparticle,
        where N_approx = a(R/sigma)^2 + b(R/sigma) + c. The coefficients were
        obtained by fitting a second-order polynomial to data collected for the
        number of particle to compose a nanoparticle as a function of radius and
        bead diameter. This value is not exact, but provides a good start for the
        binary search algorithm.
        '''
        a = 9.4379
        b = 0.6826
        c = -1.3333
        N_approx = a * ((r / sigma) ** 2) + b * (r / sigma) + c
        if N_approx > 5e5 and not force_build:
            raise Exception('The bead size and nanoparticle radius provided '
                            'would result in a nanoparticle containing roughly '
                            '{} particles. Perhaps you made a typo. If not, please '
                            'rerun with the `force_build=True` argument.'
                            ''.format(N_approx))

        # Binary search algorithm to find maximum number of CG beads w/o overlaps
        min_points = max(N_approx - 500, 1)
        max_points = N_approx + 500
        opt_points = 0
        while opt_points == 0:
            mid = math.ceil((max_points + min_points) / 2)
            points = _fast_sphere_pattern(mid, r)
            points_high = _fast_sphere_pattern(mid+1, r)
            check = self._check_overlap(points, r_CG)
            check_high = self._check_overlap(points_high, r_CG)
            if check == False and check_high == True:
                points = _fast_sphere_pattern(mid, r)
                opt_points = mid
            elif check == True:
                max_points = mid - 1
            else:
                min_points = mid + 1

        for i, pos in enumerate(points):
            particle = mb.Compound(name="_CG", pos=pos)
            self.add(particle, "CG_{}".format(i))

    def _check_overlap(self, points, radius):
        """ Determines if there is any overlap for a set of spheres

        Parameters
        ----------
        points : array-like, shape=(n,3)
            Sphere locations
        radius : float
            Radius of spheres
        """
        dists = distance.cdist(points, points, 'euclidean')
        dists = dists[np.nonzero(dists)]

        return np.any(dists < 2.0 * radius)
