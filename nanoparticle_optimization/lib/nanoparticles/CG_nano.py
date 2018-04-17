from __future__ import division

import itertools
from math import ceil, isclose

import mbuild as mb
import numpy as np
from scipy.spatial import distance

from nanoparticle_optimization.utils.exceptions import BuildError


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
        occur to match the desired value. Note that currently `bvf` values are
        only supported up to a maximum value (that will be dependent on `radius`
        and `bead_diameter`. This is due to difficulty in estimating the bead
        volume fraction when regions of intersection exist that include more than
        two spheres.
    force_build : boolean, optional, default=False
        By default if a nanoparticle is expected to contain more than 5e5 beads
        the script will exit with an error (as this can lead to memory issues).
        Set this parameter to `True` to override this behavior.
    """
    def __init__(self, radius, bead_diameter, bvf=None, force_build=False):
        super(CG_nano, self).__init__()

        bead_radius = bead_diameter / 2

        '''
        The radius of silica is taken to be 0.40323nm * 0.5, where 0.40323nm
        is the LJ sigma for the Si-O interaction in the polysiloxane force field
        of Sun et al. See https://doi.org/10.1016/S1386-1425(97)00013-9.
        '''
        silica_radius = 0.201615

        '''
        The radius of the sphere on which to place the beads will be the desired
        radius, minus the bead radius, plus the radius of silica. This will
        result in a nanoparticle with a van der Waals radius of
        `radius` + 0.20615nm.
        '''
        shell_radius = radius - bead_radius + silica_radius

        '''
        Approximate the number of beads required to construct the nanoparticle,
        where N_approx = a(R/sigma)^2 + b(R/sigma) + c. The coefficients were
        obtained by fitting a second-order polynomial to data collected for the
        number of particle to compose a nanoparticle as a function of radius and
        bead diameter. This value is not exact, but provides a good start for the
        binary search algorithm.

        A similar procedure has been performed to estimate particle number for
        specified bead volume fractions.
        '''
        if not bvf:
            a = 9.4379
            b = 0.6826
            c = -1.3333
            r_over_s = shell_radius / bead_diameter
            N_approx = int(a * (r_over_s ** 2) + b * r_over_s + c)
        else:
            a = 25.7010
            b = -27.4485
            c = 22.4832
            r_over_s = shell_radius / bead_diameter
            N_approx = int(bvf * (a * (r_over_s ** 2) + b * r_over_s + c))
        if N_approx > 5e5 and not force_build:
            raise BuildError('The bead size and nanoparticle radius provided '
                             'would result in a nanoparticle containing roughly '
                             '{} particles. Perhaps you made a typo. If not, please '
                             'rerun with the `force_build=True` argument.'
                             ''.format(N_approx))

        if not bvf:
            '''
            Binary search algorithm to find maximum number of beads without
            overlaps
            '''
            min_points = max(N_approx - 500, 1)
            max_points = N_approx + 500
            opt_points = 0
            while opt_points == 0:
                mid = ceil((max_points + min_points) / 2)
                points = self._fast_sphere_pattern(mid, shell_radius)
                points_high = self._fast_sphere_pattern(mid+1, shell_radius)
                check = self._check_overlap(points, bead_radius)
                check_high = self._check_overlap(points_high, bead_radius)
                if check == False and check_high == True:
                    points = self._fast_sphere_pattern(mid, shell_radius)
                    opt_points = mid
                elif check == True:
                    max_points = mid - 1
                else:
                    min_points = mid + 1
        else:
            '''
            Binary search algorithm to find the required number of beads to
            satisfy the specified fractional bead volume
            '''
            min_points = max(N_approx - 500, 1)
            max_points = N_approx + 500
            opt_points = 0
            last = None
            while opt_points == 0:
                mid = ceil((max_points + min_points) / 2)
                if mid == last:
                    raise BuildError('Nanoparticle construction failed. This may '
                                     'be due to the specified `bvf` being too '
                                     'large or the absolute tolerance being too '
                                     'small. Try altering these values and running '
                                     'again.')
                points = self._fast_sphere_pattern(mid, shell_radius)
                test_bvf = self._calc_bvf(points, bead_radius, shell_radius)
                if round(test_bvf, 2) == round(bvf, 2):
                    points = self._fast_sphere_pattern(mid, shell_radius)
                    opt_points = mid
                elif test_bvf > bvf:
                    max_points = mid - 1
                else:
                    min_points = mid + 1
                last = mid

        for i, pos in enumerate(points):
            particle = mb.Compound(name="_CG", pos=pos)
            self.add(particle, "CG_{}".format(i))

    @staticmethod
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

    @staticmethod
    def _overlap_volume(radius, separation):
        if separation < 2 * radius:
            return (1/12 * np.pi * (4 * radius + separation) \
                    * (2 * radius - separation)**2)
        else:
            return 0

    @staticmethod
    def _intersected(positions, radius):
        """
        Determine by trilateration if three spheres intersect

        Adapted from:
        https://stackoverflow.com/questions/1406375/finding-intersection-points-between-3-spheres
        which was adapted from:
        https://en.wikipedia.org/wiki/Trilateration
        """
        P1 = positions[0]
        P2 = positions[1]
        P3 = positions[2]
        temp1 = P2 - P1
        e_x = temp1 / np.linalg.norm(temp1)
        temp2 = P3 - P1
        i = np.dot(e_x, temp2)
        temp3 = temp2 - i * e_x
        e_y = temp3 / np.linalg.norm(temp3)
        e_z = np.cross(e_x, e_y)
        d = np.linalg.norm(P2 - P1)
        j = np.dot(e_y, temp2)                                   
        x = d / 2
        y = (-2*i*x + i*i + j*j) / (2*j)
        temp4 = radius**2 - x*x - y*y
        if temp4 < 0:
            return False
        return True

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

    def _calc_bvf(self, points, bead_radius, shell_radius):
        """ Calculate the fractional bead volume within a spherical shell.

        Parameters
        ----------
        points : array-like, shape=(n,3)
            Sphere locations
        radius : float
            Radius of spheres
        """
        dists = distance.cdist(points, points, 'euclidean')

        '''
        Check for intersection. If three spheres intersect we cannot
        (easily) accurately determine the shared volume and therefore
        cannot calculate the bead volume fraction.
        '''
        for i, point_distances in enumerate(dists):
            overlaps = np.where(np.logical_and(np.less(point_distances,
                bead_radius * 2), np.not_equal(point_distances, 0)))[0]
            for combo in itertools.combinations(overlaps, 2):
                positions = [points[idx] for idx in combo + (i,)]
                if self._intersected(positions, bead_radius):
                    return 1.0

        dists = dists[np.nonzero(dists)]

        r_min = shell_radius - bead_radius
        r_max = shell_radius + bead_radius
        vol_shell = (4/3) * np.pi * (r_max**3 - r_min**3)
        vol_beads = len(points) * (4/3) * np.pi * bead_radius**3
        '''
        The total volume taken up by beads is the volume of all of the beads
        minus the volume of the bead intersections. Since each bead-bead
        distance will be present twice within `dists` we divide by two here
        to compensate.
        '''
        vol_overlap = np.sum([self._overlap_volume(bead_radius, dist)
                              for dist in dists]) / 2

        return (vol_beads - vol_overlap) / vol_shell
