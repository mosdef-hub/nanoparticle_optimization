from __future__ import division

import math

import mbuild as mb
import numpy as np
from scipy.spatial import distance

from .silica import Silica

class AA_nano(mb.Compound):
    def __init__(self, radius=2.0, O_layer=False):
        """Initialize an AA_nano object.
        
        Args:
            radius (float): Radius of the nanoparticle
        """ 
        super(AA_nano, self).__init__()

        single = Silica()

        O_buffer = 0.275
        # Replicate the bulk silica box if necessary
        rep = math.ceil(((radius+O_buffer)/single.periodicity[0])*2.0)
        rep = int(rep)
        bulk = mb.TiledCompound(tile=single,n_tiles=np.array([rep,rep,rep]),name="bulk_silica")
        bulk_name = []
        bulk_pos = []
        for child in bulk.children:
            for grandchild in child.children:
                bulk_name.append(grandchild.name)
                bulk_pos.append(grandchild.pos)
        bulk_center = [bulk.periodicity[i]/2.0 for i in range(3)]
        dists = distance.cdist(bulk_pos,[bulk_center],'euclidean')
        types = {'1':'Si','2':'O'}
        if O_layer:
            shell_buffer = 0.5
            for i, dist in enumerate(dists):
                if types[bulk_name[i]] == 'Si' and dist <= radius and dist > radius - shell_buffer:
                    particle = mb.Compound(name=types[bulk_name[i]], pos=bulk_pos[i])
                    self.add(particle, types[bulk_name[i]]+"_{}".format(i))
                elif types[bulk_name[i]] == 'O' and dist <= radius + O_buffer and dist > radius - shell_buffer:
                    particle = mb.Compound(name=types[bulk_name[i]], pos=bulk_pos[i])
                    self.add(particle, types[bulk_name[i]]+"_{}".format(i))
            self.generate_bonds(name_a='Si', name_b='O', dmin=0.0, dmax=0.20419)
            components = self.bond_graph.connected_components()
            major_component = max(components, key=len)
            for atom in list(self.particles()):
                if atom not in major_component:
                    dist = np.linalg.norm(atom.pos - bulk_center)
                    if atom.name == 'O' and dist > radius:
                        self.remove(atom)
            for i, dist in enumerate(dists):
                if dist <= radius - shell_buffer:
                    particle = mb.Compound(name=types[bulk_name[i]], pos=bulk_pos[i])
                    self.add(particle, types[bulk_name[i]]+"_{}".format(i))
            for bond in self.bonds():
                self.remove_bond(bond)
        else:
            for i, dist in enumerate(dists):
                if dist <= radius:
                    particle = mb.Compound(name=types[bulk_name[i]], pos=bulk_pos[i])
                    self.add(particle, types[bulk_name[i]]+"_{}".format(i))

if __name__ == "__main__":
    n = AA_nano(radius=9)
    n.save('AA-{}nm.mol2'.format(9))
