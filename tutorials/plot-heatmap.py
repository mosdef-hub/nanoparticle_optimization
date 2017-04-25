from __future__ import division

import dill

import matplotlib.pyplot as plt
import numpy as np

optimization = dill.load(open('optimization.p', 'rb'))

x = optimization.grid[0]
x_spacing = x[1,0] - x[0,0]
x -= x_spacing/2
y = optimization.grid[1]
y_spacing = y[0,0] - y[0,1]
y -= y_spacing/2
residuals = np.log10(optimization.grid_residuals)

fig, ax = plt.subplots()

plt.pcolormesh(x, y, residuals, cmap='viridis_r')

for i, val in enumerate(optimization.grid[0][:-1,0]):
    for j, val2 in enumerate(optimization.grid[1][0,1:]):
        plt.text(val + x_spacing/2, val2 + y_spacing/2,
                 '{}'.format(round(optimization.grid_residuals[i,j], 2)),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=7, fontweight='bold')

plt.colorbar(shrink=0.75, label='Log(avg. residual)')
plt.xlabel('Epsilon, kcal/mol')
plt.ylabel('n')
#ax.set_xlim(0.5, 14.5)
#ax.set_ylim(11.15, 41.1)
#ax.set_aspect(10)
plt.tight_layout()
fig.savefig('heatmap.pdf')
