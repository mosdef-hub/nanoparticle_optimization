import numpy as np
import mbuild as mb

class CH2UA(mb.Compound):
    """ """
    def __init__(self):
        super(CH2UA, self).__init__()
        self.add(mb.Particle(name='_CH2'))

        self.add(mb.Port(anchor=self[0], separation=0.075), 'up')

        self.add(mb.Port(anchor=self[0], orientation=[0, -1, 0], 
                         separation=0.075), 'down')
