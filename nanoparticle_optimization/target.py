import numpy as np

def load(filename):
    target_data = np.loadtxt(filename)
    try:
        target = Target(target_data[:,0], target_data[:,1], target_data[:,2])
    except IndexError:
        target = Target(target_data[:,0], target_data[:,1])
    return target

class Target(object):
    def __init__(self, separations, potential, error=None):
        super(Target, self).__init__()

        self.separations = separations
        self.potential = potential
        if error is None:
            self.error = np.zeros(potential.shape[0])
        else:
            self.error = error

if __name__ == "__main__":
    pass
