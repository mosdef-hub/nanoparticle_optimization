from scipy.stats import linregress

sigma = [0.8, 1.0, 1.2]
epsilon = [2.0299, 3.0509, 4.1137]
n = [40.8240, 36.1209, 31.8481]
m = [5.0024, 5.2953, 5.5586]

epsilon_slope, epsilon_intercept, epsilon_r, _, _ = linregress(sigma, epsilon)
n_slope, n_intercept, n_r, _, _ = linregress(sigma, n)
m_slope, m_intercept, m_r, _, _ = linregress(sigma, m)

import pdb;pdb.set_trace()
