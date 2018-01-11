from __future__ import division

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mbuild import Box
import numpy as np

def visualize(compound):
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')

    X = compound.xyz[:,0]
    Y = compound.xyz[:,1]
    Z = compound.xyz[:,2]
    ax.scatter(X, Y, Z, c='orange', linewidth=2.0, s=350, edgecolors='black',
               alpha=1.0)

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(),
                          Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()
