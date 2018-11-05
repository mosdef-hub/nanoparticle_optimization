from __future__ import division

import mbuild as mb
import numpy as np

from cgsurface import CGSurface
from nanoparticle_optimization.lib.chains import Alkane


class CGMonolayer(mb.Compound):
    def __init__(self, bead_diameter, bvf, chain_length,
                 chain_fidelity='UA', n_chains=100, length=5):
        super(CGMonolayer, self).__init__()

        silica_radius = 0.201615
        surface = CGSurface(bead_diameter=bead_diameter, bvf=bvf,
                            length=length)
        self.add(surface, 'surface')

        if chain_fidelity != 'UA':
            raise Exception("Currently only united-atom chains are "
                            "supported.")
        chain_prototype = Alkane(chain_length=chain_length,
                                 fidelity=chain_fidelity,
                                 cap_front=False)
        if n_chains ** 0.5 % 1 != 0:
            raise Exception("Number of chains must be a perfect square.")
        pattern = self._hexagonal_pattern(n_chains**0.5, length)
        port_separation = np.linalg.norm(chain_prototype['up'].pos - \
                                chain_prototype['up'].anchor.pos)
        chain_bead_radius = 0.395 / 2
        shift = silica_radius + chain_bead_radius - port_separation
        pattern[:, 2] += shift

        for position in pattern:
            port = mb.Port(anchor=self['surface'], orientation=[0, 0, 1],
                           separation=0)
            port.translate_to(position)
            chain = mb.clone(chain_prototype)
            mb.force_overlap(chain, chain['up'], port, add_bond=False)
            self.add(chain, 'chain[$]')

    @staticmethod
    def _hexagonal_pattern(n, length):
        spacing = length / n
        lattice_angles = [90, 90, 120]
        locations = [[0., 0., 0.]]
        basis = {'_CGN': locations}
        lattice = mb.Lattice(lattice_spacing=[spacing, spacing, 0.],
                             angles=lattice_angles,
                             lattice_points=basis)
        cgbead = mb.Compound(name='_CGN')
        cg_dict = {'_CGN': cgbead}
        lattice_pop = lattice.populate(compound_dict=cg_dict, x=n,
                                       y=n, z=1)
        # remap coordinates
        CGSurface._remap_coords(lattice_pop, length)

        return lattice_pop.xyz

if __name__ == '__main__':
    bead_diameter = 0.6
    bvf = 0.5
    chain_length = 18
    monolayer = CGMonolayer(bead_diameter=bead_diameter, bvf=bvf,
                            chain_length=chain_length, n_chains=900,
                            length=15)
    monolayer.save('monolayer.mol2', overwrite=True)
