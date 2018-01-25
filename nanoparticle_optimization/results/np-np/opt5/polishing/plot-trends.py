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

sigma = np.arange(0.2, 2.0, 0.1)
epsilon = [0.00995, 0.01083, 0.36192, 1.02372, 1.43429, 1.39136, 2.31170, 2.41813,
           2.87120, 3.40631, 4.11224, 4.43503, 5.29938, 5.68701, 6.71327, 6.96366,
           7.05099, 7.62263]
m = [3.00631, 2.75859, 4.34007, 4.79557, 4.97922, 4.83336, 5.18218, 5.13034,
     5.23675, 5.33795, 5.48890, 5.53134, 5.65830, 5.69174, 5.81852, 5.83916,
     5.84729, 5.88147]

sigma_big = np.arange(3.0, 5.0)
epsilon_big = [21.88478, 40.86000]
m_big = [7.26483, 8.29855]

fig = plt.figure(1, figsize=(8,8))
ax_epsilon = plt.subplot(211)
ax_m = plt.subplot(212)

ax_epsilon.plot(sigma, epsilon, linestyle='None', marker='o', color='blue',
                mfc='white', ms=12, mew=4)
ax_m.plot(sigma, m, linestyle='None', marker='o', color='blue', mfc='white', ms=12,
          mew=4)

ax_epsilon.plot(sigma_big, epsilon_big, linestyle='None', marker='o', color='black',
                mfc='white', ms=12, mew=4)
ax_m.plot(sigma_big, m_big, linestyle='None', marker='o', color='black',
          mfc='white', ms=12, mew=4)

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
ax_epsilon.set_xlim([0.1, 4.1])
ax_m.set_xlim([0.1, 4.1])
plt.tight_layout()
plt.subplots_adjust(hspace=0.075)
plt.savefig('param-trends.pdf')
