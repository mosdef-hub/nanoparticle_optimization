from __future__ import division

import math

import mbuild as mb
import numpy as np
from scipy.spatial import distance

# Coarse-grained nanoparticle class
class CG_nano(mb.Compound):
    def __init__(self, r=5.0, sigma=0.8):
        super(CG_nano, self).__init__()

        r_CG = sigma / 2
        r_silica = (0.40323 * 0.8) / 2

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
            mask = mb.SpherePattern(mid)
            mask.scale(r)
            mask_high = mb.SpherePattern(mid + 1)
            mask_high.scale(r)
            check = self._check_overlap(mask, r_CG)
            check_high = self._check_overlap(mask_high, r_CG)
            if check == False and check_high == True:
                mask = mb.SpherePattern(mid)
                mask.scale(r - r_CG + r_silica)
                opt_points = mid
            elif check == True:
                max_points = mid - 1
            else:
                min_points = mid + 1

        for i, pos in enumerate(mask):
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
    cg_nano = CG_nano(r=3.0, sigma=0.9)
    cg_nano.save('test.mol2')
