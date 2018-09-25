import numpy as np
import mbuild as mb

class CH4UA(mb.Compound):
    """ """
    def __init__(self):
        super(CH4UA, self).__init__()
        self.add(mb.Particle(name='_CH4'))
