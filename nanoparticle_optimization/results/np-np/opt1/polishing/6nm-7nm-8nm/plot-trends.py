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

sigma = [0.8, 1.0, 1.2]
epsilon = [2.02985, 3.05085, 4.11373]
n = [40.82401, 36.12086, 31.84811]
m = [5.00241, 5.29527, 5.55859]

fig = plt.figure(1, figsize=(8,10))
ax_epsilon = plt.subplot(311)
ax_n = plt.subplot(312)
ax_m = plt.subplot(313)

sigma_extra = [0.6]
epsilon_extra = [1.13884]
n_extra = [32.48220]
m_extra = [4.73212]

ax_epsilon.plot(sigma, epsilon, linestyle='None', marker='o', color='blue',
                mfc='white', ms=12, mew=4)
ax_epsilon.plot(sigma_extra, epsilon_extra, linestyle='None', marker='o',
                color='red', mfc='white', ms=12, mew=4)
ax_n.plot(sigma, n, linestyle='None', marker='o', color='blue', mfc='white', ms=12,
          mew=4)
ax_n.plot(sigma_extra, n_extra, linestyle='None', marker='o', color='red',
          mfc='white', ms=12, mew=4)
ax_m.plot(sigma, m, linestyle='None', marker='o', color='blue', mfc='white', ms=12,
          mew=4)
ax_m.plot(sigma_extra, m_extra, linestyle='None', marker='o', color='red',
          mfc='white', ms=12, mew=4)

'''
------------------
Linear Regressions
------------------
'''
sigma = [val for i, val in enumerate(sigma)]
epsilon = [val for i, val in enumerate(epsilon)]
n = [val for i, val in enumerate(n)]
m = [val for i, val in enumerate(m)]
e_slope, e_intercept, e_r, _, _ = linregress(sigma, epsilon)
'''
ax_epsilon.plot([sigma[0], sigma[-1]],
                [e_slope * val + e_intercept for val in [sigma[0], sigma[-1]]],
                linestyle='--', color='black', marker='None')
'''
ax_epsilon.plot([0.5, 1.5],
                [e_slope * val + e_intercept for val in [0.5, 1.5]],
                linestyle='--', color='black', marker='None')
n_slope, n_intercept, n_r, _, _ = linregress(sigma, n)
'''
ax_n.plot([sigma[0], sigma[-1]],
          [n_slope * val + n_intercept for val in [sigma[0], sigma[-1]]],
          linestyle='--', color='black', marker='None')
'''
ax_n.plot([0.5, 1.5],
          [n_slope * val + n_intercept for val in [0.5, 1.5]],
          linestyle='--', color='black', marker='None')
m_slope, m_intercept, m_r, _, _ = linregress(sigma, m)
'''
ax_m.plot([sigma[0], sigma[-1]],
          [m_slope * val + m_intercept for val in [sigma[0], sigma[-1]]],
          linestyle='--', color='black', marker='None')
'''
ax_m.plot([0.5, 1.5],
          [m_slope * val + m_intercept for val in [0.5, 1.5]],
          linestyle='--', color='black', marker='None')

'''
------------------
2nd-order Fit
------------------
'''
e_coeff, e_resid, _, _, _ = np.polyfit(sigma, epsilon, 2, full=True)
'''
ax_epsilon.plot(np.linspace(sigma[0], sigma[-1], 10),
                [e_coeff[0] * val**2 + e_coeff[1] * val + e_coeff[2]
                for val in np.linspace(sigma[0], sigma[-1], 10)],
                linestyle='--', color='red', marker='None')
'''
'''
ax_epsilon.plot(np.linspace(0.25, 2.0, 10),
                [e_coeff[0] * val**2 + e_coeff[1] * val + e_coeff[2]
                for val in np.linspace(0.25, 2.0, 10)],
                linestyle='--', color='red', marker='None')

n_coeff, n_resid, _, _, _ = np.polyfit(sigma, n, 2, full=True)
ax_n.plot(np.linspace(0.25, 2.0, 10),
          [n_coeff[0] * val**2 + n_coeff[1] * val + n_coeff[2]
          for val in np.linspace(0.25, 2.0, 10)],
          linestyle='--', color='red', marker='None')

m_coeff, m_resid, _, _, _ = np.polyfit(sigma, m, 2, full=True)
ax_m.plot(np.linspace(0.25, 2.0, 10),
          [m_coeff[0] * val**2 + m_coeff[1] * val + m_coeff[2]
          for val in np.linspace(0.25, 2.0, 10)],
          linestyle='--', color='red', marker='None')
'''

plt.xlabel(r'$\sigma$, nm')
ax_epsilon.set_ylabel(r'$\epsilon$, kcal/mol')
ax_n.set_ylabel(r'$n$')
ax_m.set_ylabel(r'$m$')
ax_epsilon.set_xticklabels([])
ax_n.set_xticklabels([])
plt.tight_layout()
plt.subplots_adjust(hspace=0.075)
plt.savefig('param-trends.pdf')
