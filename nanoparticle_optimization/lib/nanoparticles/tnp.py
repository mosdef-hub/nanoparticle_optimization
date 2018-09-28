from __future__ import division

import mbuild as mb
import numpy as np

from nanoparticle_optimization.lib.chains import Alkane
from nanoparticle_optimization import AA_nano, CG_nano


class TNP(mb.Compound):
    """An alkane-tethered, spherical nanoparticle

    Parameters
    ----------
    radius
    chain_density
    chain_length
    core_fidelity : str, optional, default='AA'
        Fidelity level for the nanoparticle core. Valid options are
        'AA' for an all-atom core and 'CG' for a coarse-grained core.
        If `core_fidelity`='CG' the `bead_diameter` and `bvf` parameters
        must also be provided.
    chain_fidelity : str, optional, default='UA'
        Fidelity level for the alkane chains. Valid options are 'UA' for
        united-atom and 'CG' for 3:1 mapped coarse-grained chains.
    bead_diameter
    bvf
    """
    def __init__(self, radius, chain_density, chain_length,
                 core_fidelity='AA', chain_fidelity='UA',
                 bead_diameter=None, bvf=None):
        super(TNP, self).__init__()

        if core_fidelity == 'AA':
            core_particles = ['Si', 'O']
            core = AA_nano(radius=radius)
        elif core_fidelity == 'CG':
            if not bead_diameter:
                raise Exception("`bead_diameter` must be defined when "
                                "`core_fidelity`='CG'")
            if not bvf:
                raise Exception("`bvf` must be defined when "
                                "`core_fidelity`='CG'")
            core_particles = ['_CGN']
            core = CG_nano(radius=radius, bead_diameter=bead_diameter,
                           bvf=bvf)
        else:
            raise Exception("`core_fidelity` must be either 'AA' or 'CG'.")

        self.add(core, 'core')

        chain_prototype = Alkane(chain_length=chain_length,
                                 fidelity=chain_fidelity,
                                 cap_front=False)

        surface_area = 4.0 * np.pi * radius ** 2.0
        n_chains = int(chain_density * surface_area)
        pattern = mb.SpherePattern(n_chains)
        pattern.scale(radius)

        for position in pattern.points:
            port = mb.Port(anchor=self['core'], orientation=position,
                           separation=radius)
            self['core'].add(port, "attachment_site[$]")

        chains, _ = pattern.apply_to_compound(chain_prototype,
            guest_port_name='up', host=self['core'])
        self.add(chains)

        self.label_rigid_bodies(rigid_particles=core_particles)
        for bond in self.bonds():
            if(bond[0].name in ['AA_nano', 'CG_nano'] or
                    bond[1].name in ['AA_nano', 'CG_nano']):
                if 'nano' in bond[0].name:
                    bond[1].rigid_id = 0
                if 'nano' in bond[1].name:
                    bond[0].rigid_id = 0
                self.remove_bond(bond)

if __name__ == "__main__":
    radius = 4
    chain_density = 1.0
    chain_length = 36
    bead_diameter = 0.6
    bvf = 0.4
    aa_tnp = TNP(radius=radius, chain_density=chain_density,
                 chain_length=chain_length, core_fidelity='AA',
                 chain_fidelity='UA')
    aa_tnp.save('aa-tnp.mol2', overwrite=True)
    '''
    ua_tnp = TNP(radius=radius, chain_density=chain_density,
                 chain_length=chain_length, core_fidelity='CG',
                 chain_fidelity='UA', bead_diameter=bead_diameter,
                 bvf=bvf)
    ua_tnp.save('ua-tnp.mol2', overwrite=True)
    cg_tnp = TNP(radius=radius, chain_density=chain_density,
                 chain_length=chain_length, core_fidelity='CG',
                 chain_fidelity='CG', bead_diameter=bead_diameter,
                 bvf=bvf)
    cg_tnp.save('cg-tnp.mol2', overwrite=True)
    '''
