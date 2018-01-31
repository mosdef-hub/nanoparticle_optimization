import os
from pkg_resources import resource_filename

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

import nanoparticle_optimization as np_opt

resource_package = np_opt.__name__
rc_path = resource_filename(resource_package, os.path.join('utils', 'matplotlibrc'))
matplotlib.rc_file(rc_path)

sigma = np.arange(0.4, 2.1, 0.1)
sigma = [0.4, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
epsilon = [0.36192, 1.39136, 2.41813,
           2.87120, 3.40631, 4.11224, 4.43503, 5.29938, 5.68701, 6.71327, 6.96366,
           7.05099, 7.62263, 7.65790]
m = [4.34007, 4.83336, 5.13034,
     5.23675, 5.33795, 5.48890, 5.53134, 5.65830, 5.69174, 5.81852, 5.83916,
     5.84729, 5.88147, 5.88346]

fig = plt.figure(1, figsize=(8,8))
ax_epsilon = plt.subplot(211)
ax_m = plt.subplot(212)

ax_epsilon.plot(sigma, epsilon, linestyle='None', marker='o', color='blue',
                mfc='white', ms=12, mew=4)
ax_m.plot(sigma, m, linestyle='None', marker='o', color='blue', mfc='white', ms=12,
          mew=4)

'''
------------------
Linear Regressions
------------------
'''
'''
e_slope, e_intercept, e_r, _, _ = linregress(sigma, epsilon)
ax_epsilon.plot([0.5, 1.75],
                [e_slope * val + e_intercept for val in [0.5, 1.75]],
                linestyle='--', color='black', marker='None')
m_slope, m_intercept, m_r, _, _ = linregress(sigma, m)
ax_m.plot([0.5, 1.75],
          [m_slope * val + m_intercept for val in [0.5, 1.75]],
          linestyle='--', color='black', marker='None')
'''

plt.xlabel(r'$\sigma$')
ax_epsilon.set_ylabel(r'$\epsilon$')
ax_m.set_ylabel(r'$m$')
ax_epsilon.set_xticklabels([])
ax_epsilon.set_xlim([0.1, 2.1])
ax_m.set_xlim([0.1, 2.1])
plt.tight_layout()
plt.subplots_adjust(hspace=0.075)
plt.savefig('param-trends-trunc.pdf')
