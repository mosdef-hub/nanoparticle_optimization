from __future__ import division

import itertools
from math import ceil

import mbuild as mb
import numpy as np
from scipy.spatial import distance

from nanoparticle_optimization.lib.nanoparticles.CG_nano import CG_nano
from nanoparticle_optimization.utils.exceptions import BuildError


class CGSurface(mb.Compound):
    """
    Currently only for a single layer of beads.

    Currently only for square surfaces.
    """
    def __init__(self, bead_diameter, bvf, length=5):
        super(CGSurface, self).__init__()

        bead_radius = bead_diameter / 2
        silica_radius = 0.201615

        surface_volume = length * length * bead_diameter

        if not bvf:
            '''
            Binary search algorithm to find maximum number of beads without
            overlaps
            '''
            min_n = 1
            max_n = 100
            opt_n = 0
            while opt_n == 0:
                mid_n = ceil((max_n + min_n) / 2)
                lattice = _generate_lattice(n=mid_n,
                                            length=length)
                lattice_high = _generate_lattice(n=mid_n + 1,
                                                 length=length)
                check = self._check_overlap(lattice.xyz, bead_radius)
                check_high = self._check_overlap(lattice_high.xyz,
                                                 bead_radius)
                if check == False and check_high == True:
                    opt_n = mid_n
                elif check == True:
                    max_n = mid_n - 1
                else:
                    min_n = mid_n + 1
        else:
            '''
            Binary search algorithm to find the required number of beads
            to satisfy the specified fractional bead volume
            '''
            min_n = 1
            max_n = 50
            opt_n = 0
            last = None
            while opt_n == 0:
                mid_n = ceil((max_n + min_n) / 2)
                print(mid_n)
                if mid_n == last:
                    if test_bvf == 1.0:
                        lattice = self._generate_lattice(n=mid_n - 1,
                                                         length=length)
                        test_bvf = self._calc_bvf(lattice.xyz,
                                                  bead_radius,
                                                  surface_volume)
                    print('Could not achieve desired `bvf`. This surface '
                          'has a `bvf` of {:.3f}'.format(test_bvf))
                    break
                lattice = self._generate_lattice(n=mid_n,
                                                 length=length)
                test_bvf = self._calc_bvf(lattice.xyz, bead_radius,
                                          surface_volume)
                if round(test_bvf, 2) == round(bvf, 2):
                    opt_n = mid_n
                elif test_bvf > bvf:
                    max_n = mid_n - 1
                else:
                    min_n = mid_n + 1
                last = mid_n

        lattice.xyz[:, 2] -= (bead_radius + silica_radius)
        self.add(lattice)

    def _generate_lattice(self, n, length):
        spacing = length / n
        lattice_angles = [90, 90, 120]
        locations = [[0., 0., 0.]]
        basis = {'_CGN': locations}
        lattice = mb.Lattice(lattice_spacing=[spacing, spacing, 0.],
                             angles=lattice_angles,
                             lattice_points=basis)
        cgbead = mb.Compound(name='_CGN')
        cg_dict = {'_CGN': cgbead}
        surface = lattice.populate(compound_dict=cg_dict, x=n,
                                   y=n, z=1)
        # remap coordinates
        self._remap_coords(surface, length)

        return surface

    @staticmethod
    def _remap_coords(compound, length):
        for particle in compound:
            for i, coord in enumerate(particle.pos[:2]):
                if coord < 0.:
                    particle.pos[i] += length
                if coord >= length:
                    particle.pos[i] -= length

    def _check_overlap(self, points, radius):
        """ Determines if there is any overlap for a set of spheres

        We _should_ be accounting for periodic boundaries here; however,
        since all beads should be equally spaced, we'll know if there is
        overlap without accounting for them.

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

    def _calc_bvf(self, points, bead_radius, surface_volume):
        """
        Because we are constructing a lattice and only care about the
        minimum distance, we don't need to explicitly include the periodic
        boundaries in our distance calculation. We can find the minimum
        distance and from our lattice properties know how many pairs of
        particles should be separated by this distance.
        """
        dists = distance.cdist(points, points, 'euclidean')
        min_dist = np.min(dists)

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
                if CG_nano._intersected(positions, bead_radius):
                    return 1.0

        dists = dists[np.nonzero(dists)]
        vol_beads = len(points) * (4/3) * np.pi * bead_radius**3
        '''
        The total volume taken up by beads is the volume of all of the
        beads.
        '''
        vol_overlap = np.sum([CG_nano._overlap_volume(bead_radius, dist)
                              for dist in dists]) / 2
        print(vol_beads, vol_overlap, surface_volume)

        return (vol_beads - vol_overlap) / surface_volume

if __name__ == '__main__':
    bead_diameter = 0.6
    bvf = 0.5
    surface = CGSurface(bead_diameter=bead_diameter, bvf=bvf, length=5)
    surface.save('surface.mol2', overwrite=True)
