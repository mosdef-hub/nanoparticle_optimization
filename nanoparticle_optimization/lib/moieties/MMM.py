import numpy as np
import mbuild as mb

class MMM(mb.Compound):
    """Coarse-grained alkane bead containing a CH2-CH2-CH2 group"""
    def __init__(self):
        super(MMM, self).__init__()
        self.add(mb.Particle(name='_MMM'))
        
        self.add(mb.Port(anchor=self[0], separation=0.18), 'up')
        
        self.add(mb.Port(anchor=self[0], orientation=[0, -1, 0],
                         separation=0.18), 'down')
