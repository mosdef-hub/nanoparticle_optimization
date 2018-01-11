import pkg_resources

import matplotlib.pyplot as plt
import mbuild as mb
import numpy as np
from scipy.stats import linregress

import nanoparticle_optimization
from nanoparticle_optimization import CG_nano, System, load_target


def calc_fit(U_target, U_cg):
    """
    Parameters
    ----------
    U_target : np.ndarray
    U_cg : np.adarray
    """

    error = sum(abs(U_target - U_cg))
    error /= sum(abs(U_target) + abs(U_cg))
    return 1 - error 

def test_mixed(forcefields, tag='cgff'):
    resource_package = nanoparticle_optimization.__name__

    for radius1 in [2, 3, 4, 6]:
        for radius2 in [3, 4, 6, 8]:
            if radius2 > radius1:
                resource_path = '/'.join(('utils',
                    'U_{}nm-{}nm_truncated.py'.format(int(radius1), int(radius2))))
                target = load(pkg_resources.resource_filename(resource_package,
                    resource_path))
                target.separations /= 10.0

                fig, ax = plt.subplots()
                ax.errorbar(target.separations, target.potential,
                    yerr=target.error, linestyle='None', marker='o',
                    color='black', label='Target')

                U_mins = []
                for forcefield in forcefields:
                    nano1 = CG_nano(radius1, sigma=forcefield['sigma'])
                    nano2 = CG_nano(radius2, sigma=forcefield['sigma'])
                    system = System(nano1, nano2)

                    U_cg = np.array([pot[0] for pot in 
                        system.calc_potential(forcefield, target.separations,
                        configurations=10, r_dependent_sampling=True)])

                    short_separations = np.linspace(target.separations[0] - 0.5,
                        target.separations[4], 50)
                    U_cg_short = np.array([pot[0] for pot in
                        system.calc_potential(forcefield, short_separations,
                        configurations=10, r_dependent_sampling=True)])

                    separations_full = np.hstack((target.separations,
                        short_separations))
                    U_cg_full = np.hstack((U_cg, U_cg_short))
                    U_mins.append(min(U_cg_full))

                    fit = calc_fit(target.potential, U_cg)

                    ax.plot(separations_full, U_cg_full, linestyle='-', marker='None',
                        label='CG, sigma={}nm, fit={}'.format(forcefield['sigma'],
                        round(fit, 3)))

                ax.set_xlim(radius1 + radius2 - 0.5, target.separations[-1])
                ax.set_ylim(min(min(target.potential), min(U_mins)), 0)
                ax.set_xlabel('r, nm')
                ax.set_ylabel('U, kcal/mol')
                plt.legend(loc='lower right')
                plt.tight_layout()
                fig.savefig('{}-R{}nm-R{}nm.pdf'.format(tag, radius1, radius2))

def test_fit(forcefields, radius, tag='cgff', cross=False, sigmas=[]):
    resource_package = nanoparticle_optimization.__name__

    if cross:
        resource_path = '/'.join(('utils',
            'U_{}nm_{}_truncated.txt'.format(int(radius), cross)))
        configs = 100
        rds = False
    else:
        resource_path = '/'.join(('utils',
            'U_{}nm_truncated.txt'.format(int(radius))))
        configs = 10
        rds = True
    target = load(pkg_resources.resource_filename(resource_package,
        resource_path))
    target.separations /= 10.0

    fig, ax = plt.subplots()

    ax.errorbar(target.separations, target.potential, yerr=target.error,
        linestyle='None', marker='o', color='black', label='Target')

    U_mins = []
    for i, forcefield in enumerate(forcefields):
        if not sigmas:
            nano = CG_nano(radius, sigma=forcefield['sigma'])
        else:
            nano = CG_nano(radius, sigma=sigmas[i])
        if cross:
            system = System(mb.Compound(pos=np.zeros(3)), nano)
        else:
            system = System(nano)

        U_cg = np.array([pot[0] for pot in system.calc_potential(forcefield,
            target.separations, configurations=configs, r_dependent_sampling=rds)])

        short_separations = np.linspace(target.separations[0] - 0.5,
            target.separations[4], 50)
        U_cg_short = np.array([pot[0] for pot in system.calc_potential(forcefield,
            short_separations, configurations=configs, r_dependent_sampling=rds)])

        separations_full = np.hstack((target.separations, short_separations))
        U_cg_full = np.hstack((U_cg, U_cg_short))
        U_mins.append(min(U_cg_full))

        fit = calc_fit(target.potential, U_cg)

        if sigmas:
            ax.plot(separations_full, U_cg_full, linestyle='-', marker='None',
                label='CG, sigma={}nm, fit={}'.format(sigmas[i],
                round(fit, 3)))
        else:
            ax.plot(separations_full, U_cg_full, linestyle='-', marker='None',
                label='CG, sigma={}nm, fit={}'.format(forcefield['sigma'],
                round(fit, 3)))

    if cross:
        xlim = radius + 0.1
        y_offset = 0.15
    else:
        xlim = 2 * radius - 0.5
        y_offset = 10
    ax.set_xlim(xlim, target.separations[-1])
    ax.set_ylim(min(min(target.potential), min(U_mins)) - y_offset, 0)
    ax.set_xlabel('r, nm')
    ax.set_ylabel('U, kcal/mol')
    plt.legend(loc='lower right')
    plt.tight_layout()
    if cross:
        fig.savefig('{}-R{}nm-{}.pdf'.format(tag, radius, cross))
    else:
        fig.savefig('{}-R{}nm.pdf'.format(tag, radius))

def test_sigma_regression(forcefields):
    sigmas = [forcefield['sigma'] for forcefield in forcefields]
    epsilons = [forcefield['epsilon'] for forcefield in forcefields]
    ns = [forcefield['n'] for forcefield in forcefields]
    ms = [forcefield['m'] for forcefield in forcefields]

    epsilon_slope, epsilon_intercept = linregress(sigmas, epsilons)
    n_slope, n_intercept = linregress(sigmas, ns)
    m_slope, m_intercept = linregress(sigmas, ms)

    return {'epsilon': tuple(epsilon_slope, epsilon_intercept),
            'n': tuple(n_slope, n_intercept), 'm': tuple(m_slope, m_intercept)}

def test_25nm(forcefields, tag='cgff', cross=False, sigmas=[]):
    resource_package = nanoparticle_optimization.__name__
    if cross:
        resource_path = '/'.join(('utils', 'U_25nm_{}.txt'.format(cross)))
    else:
        resource_path = '/'.join(('utils', 'U_25nm.txt'))
    target = load(pkg_resources.resource_filename(resource_package, resource_path))
    target.separations /= 10.0

    fig, ax = plt.subplots()
    ax.plot(target.separations, target.potential, linestyle='None', marker='x',
        color='black', label='All-atom')

    for i, forcefield in enumerate(forcefields):
        if not sigmas:
            nano = CG_nano(25.0, sigma=forcefield['sigma'])
        else:
            nano = CG_nano(25.0, sigma=sigmas[i])
        if cross:
            system = System(mb.Compound(pos=np.zeros(3)), nano)
            U_cg = np.array([pot[0] for pot in system.calc_potential(forcefield,
                target.separations, configurations=100)])
        else:
            system = System(nano)
            U_cg = np.array([pot[0] for pot in system.calc_potential(forcefield,
                target.separations, configurations=10, r_dependent_sampling=True)])
        fit = calc_fit(target.potential, U_cg)
        if sigmas:
            ax.plot(target.separations, U_cg, linestyle='-', marker='o',
                label='CG, sigma={}nm, fit={}'.format(sigmas[i], round(fit, 3)))
        else:
            ax.plot(target.separations, U_cg, linestyle='-', marker='o',
                label='CG, sigma={}nm, fit={}'.format(forcefield['sigma'],
                round(fit, 3)))

    if cross:
        ax.set_xlim(25 + 0.1, target.separations[-1] + 1)
    else:
        ax.set_xlim(50 - 0.5, target.separations[-1] + 1)
    ax.set_ylim(min(target.potential) - 0.25, 0)
    ax.set_xlabel('r, nm')
    ax.set_ylabel('U, kcal/mol')
    plt.legend(loc='lower right')
    plt.tight_layout()
    if cross:
        fig.savefig('{}-R25nm-{}.pdf'.format(tag, cross))
    else:
        fig.savefig('{}-R25nm.pdf'.format(tag))

def run_all(forcefields, tag='cgff', cross=False, sigmas=[]):
    if not cross:
        test_mixed(forcefields, tag)
    for radius in range(2, 10):
        test_fit(forcefields, radius, tag, cross=cross, sigmas=sigmas)
    test_25nm(forcefields, tag='cgff', cross=cross, sigmas=sigmas)
