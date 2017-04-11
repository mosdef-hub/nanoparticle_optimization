import numpy as np

def load(filename):
    target_data = np.loadtxt(filename)
    return Target(target_data[:,0], target_data[:,1])

class Target(object):
    def __init__(self, separations, potential):
        super(Target, self).__init__()

        self.separations = separations
        self.potential = potential

if __name__ == "__main__":
    pass
