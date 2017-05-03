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
    """Coarse-grained nanoparticle class. """
    def __init__(self, r=5.0, sigma=0.8):
        super(CG_nano, self).__init__()

        r_CG = sigma / 2
        r_silica = 0.40323 / 2

        # N_approx = a(R/sigma)^2 + b(R/sigma) + c
        # Not an exact number but very close
        a = 9.4379
        b = 0.6826
        c = -1.3333
        N_approx = a * ((r / sigma) ** 2) + b * (r / sigma) + c

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
                points = _fast_sphere_pattern(mid, r - r_CG + r_silica)
                opt_points = mid
            elif check == True:
                max_points = mid - 1
            else:
                min_points = mid + 1

        for i, pos in enumerate(points):
            particle = mb.Compound(name="CG", pos=pos)
            self.add(particle, "CG_{}".format(i))

    def _check_overlap(self, points, radius):
        """ Determines if there is any overlap for a set of spheres

        Args:
            points (np.ndarray (n,3)): sphere locations
            radius (float): radius of spheres
        """
        dists = distance.cdist(points, points, 'euclidean')
        dists = dists[np.nonzero(dists)]

        return np.any(dists < 2.0 * radius)


if __name__ == "__main__":
    cg_nano = CG_nano(r=8.0, sigma=0.6)
    cg_nano.save('test-fast.mol2', overwrite=True)
