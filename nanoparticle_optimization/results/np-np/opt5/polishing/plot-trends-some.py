import os
from pkg_resources import resource_filename

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

import nanoparticle_optimization as np_opt

resource_package = np_opt.__name__
rc_path = resource_filename(resource_package, os.path.join('utils', 'matplotlibrc'))
matplotlib.rc_file(rc_path)

sigma_good = np.hstack((np.arange(0.9, 1.7, 0.1), np.arange(2.0, 5.0)))
epsilon_good = [2.41813, 2.87120, 3.40631, 4.11224, 4.43503, 5.29938, 5.68701,
                6.71327, 9.67500, 21.88478, 40.86000]
m_good = [5.13034, 5.23675, 5.33795, 5.48890, 5.53154, 5.65830, 5.69174, 5.81852,
          6.18750, 7.26483, 8.29855]

sigma_bad = [0.5, 0.6, 0.7, 0.8, 1.7, 1.8, 1.9]
epsilon_bad = [1.02372, 1.43429, 1.39136, 2.31170, 6.96366, 7.05099, 7.62263]
m_bad = [4.79557, 4.97922, 4.83336, 5.18218, 5.83916, 5.84729, 5.88147]

sigma_hi = [0.2, 0.3, 0.4]
epsilon_hi = [0.009949, 0.010828, 0.36192]
m_hi = [3.00631, 2.75859, 4.34007]

fig = plt.figure(1, figsize=(8,8))
ax_epsilon = plt.subplot(211)
ax_m = plt.subplot(212)

ax_epsilon.plot(sigma_good, epsilon_good, linestyle='None', marker='o', color='blue',
                mfc='white', ms=12, mew=4)
ax_m.plot(sigma_good, m_good, linestyle='None', marker='o', color='blue',
          mfc='white', ms=12, mew=4)

ax_epsilon.plot(sigma_bad, epsilon_bad, linestyle='None', marker='o', color='red',
                mfc='white', ms=12, mew=4)
ax_m.plot(sigma_bad, m_bad, linestyle='None', marker='o', color='red',
          mfc='white', ms=12, mew=4)

ax_epsilon.plot(sigma_hi, epsilon_hi, linestyle='None', marker='x', color='black',
                mfc='black', ms=12, mew=4)
ax_m.plot(sigma_hi, m_hi, linestyle='None', marker='x', color='black',
          mfc='black', ms=12, mew=4)

'''
----
Fits
----
'''
sigma_range = np.linspace(0.1, 5.5, 100)

m_slope, m_intercept, m_r, _, _ = linregress(sigma_good, m_good)
ax_m.plot(sigma_range,
          [m_slope * val + m_intercept for val in sigma_range],
          linestyle='--', color='black', marker='None')

sigma_good = np.hstack(([0.0], sigma_good))
epsilon_good = np.hstack(([0.0], epsilon_good))
e_fit = np.polyfit(sigma_good, epsilon_good, 2)
ax_epsilon.plot(sigma_range,
                [e_fit[0] * val**2 + e_fit[1] * val + e_fit[2]
                 for val in sigma_range],
                linestyle='--', color='black', marker='None')

plt.xlabel(r'$\sigma$')
ax_epsilon.set_ylabel(r'$\epsilon$')
ax_m.set_ylabel(r'$m$')
ax_epsilon.set_xticklabels([])
ax_epsilon.set_xlim([0.0, 5.6])
ax_m.set_xlim([0.0, 5.6])
plt.tight_layout()
plt.subplots_adjust(hspace=0.075)
plt.savefig('param-trends-some.pdf')
