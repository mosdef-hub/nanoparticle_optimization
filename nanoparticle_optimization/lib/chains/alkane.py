from __future__ import division

import mbuild as mb
from nanoparticle_optimization.lib.moieties import CH2UA, CH3UA, MMM, MME


class Alkane(mb.Compound):
    """An alkane chain

    Parameters
    ----------
    chain_length
    fidelity : str
        The fidelity of the alkane model. Valid options are 'UA' for
        united-atom and 'CG' for a 3:1 coarse-grained mapping.
    cap_front
    cap_end
    """
    def __init__(self, chain_length, fidelity, cap_front=True, cap_end=True):
        super(Alkane, self).__init__()

        if fidelity not in ['UA', 'CG']:
            raise Exception("`fidelity` must be either 'UA' or 'CG'.")
        if chain_length < 2:
            raise Exception("Chain length must be 2 or greater.")
        if fidelity == 'CG' and chain_length % 3 != 0:
            raise Exception("Chain length must be divisible by three for "
                            "use with `fidelity`='CG'.")
        if fidelity == 'CG' and chain_length == 3:
            raise Exception("Chain length must be greater than 3 for "
                            "use with `fidelity`='CG'.")

        if fidelity == 'UA':
            middle_bead = CH2UA()
            end_bead = CH3UA()
        if fidelity == 'CG':
            chain_length /= 3
            middle_bead = MMM()
            end_bead = MME()

        if not cap_front:
            chain_length += 1
        if not cap_end:
            chain_length += 1

        if chain_length > 2:
            chain = mb.Polymer(mb.clone(middle_bead), n=chain_length-2,
                              port_labels=('up', 'down'))
            self.add(chain, 'chain')

            if cap_front:
                self.add(mb.clone(end_bead), 'front_cap')
                mb.force_overlap(self['front_cap'], self['front_cap']['up'],
                                 self['chain']['up'])
            else:
                # Hoist port label to `Alkane` level
                self.add(chain['up'], 'up', containment=False)

            if cap_end:
                self.add(mb.clone(end_bead), 'end_cap')
                mb.force_overlap(self['end_cap'], self['end_cap']['up'],
                                 self['chain']['down'])
            else:
                # Hoist port label to `Alkane` level
                self.add(chain['down'], 'down', containment=False)

        else:
            self.add(mb.clone(end_bead), 'front_cap')
            self.add(mb.clone(end_bead), 'end_cap')
            mb.force_overlap(self['end_cap'], self['end_cap']['up'],
                             self['front_cap']['up'])
