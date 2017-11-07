import numpy as np
import mbuild as mb

class UA_ch4(mb.Compound):
    """ """
    def __init__(self):
        super(UA_ch4, self).__init__()
        self.add(mb.Particle(name='_CH4'))
