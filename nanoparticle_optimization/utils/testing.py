from __future__ import division

import os
from pkg_resources import resource_filename

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mbuild as mb
import numpy as np
from scipy.stats import linregress

import nanoparticle_optimization as np_opt


resource_package = np_opt.__name__

def _square_plot(ax):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0) / abs(y1-y0))

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
    for radius1 in [4, 6]:
        for radius2 in [6, 8]:
            if radius2 > radius1:
                target_path = os.path.join('utils', 'target_data', 'np_np',
                    'mixed_size', 'truncated',
                    'U_{}nm-{}nm_truncated.txt'.format(int(radius1), int(radius2)))
                target = np_opt.load_target(resource_filename(resource_package,
                                                              target_path))
                target.separations /= 10

                fig, ax = plt.subplots()
                ax.errorbar(target.separations, target.potential,
                    yerr=target.error, linestyle='None', marker='o',
                    color='black', label='Atomistic')

                U_mins = []
                for forcefield in forcefields:
                    nano1 = np_opt.CG_nano(radius1, sigma=forcefield['sigma'])
                    nano2 = np_opt.CG_nano(radius2, sigma=forcefield['sigma'])
                    system = np_opt.System(nano1, nano2)

                    U_cg = np.array([pot[0] for pot in 
                        system.calc_potential(forcefield, target.separations,
                        configurations=10, r_dependent_sampling=True)])

                    short_separations = np.linspace(target.separations[0] - 0.5,
                        target.separations[4], 50)
                    U_cg_short = np.array([pot[0] for pot in
                        system.calc_potential(forcefield, short_separations,
                        configurations=10, r_dependent_sampling=True)])

                    separations_full = np.hstack((short_separations,
                        target.separations))
                    U_cg_full = np.hstack((U_cg_short, U_cg))
                    U_mins.append(min(U_cg_full))

                    fit = calc_fit(target.potential, U_cg)

                    sep_plot = []
                    U_plot = []
                    for i, U in enumerate(np.flip(U_cg_full, axis=0)[:-1]):
                        if U > np.flip(U_cg_full, axis=0)[i+1]:
                            U_plot.append(U)
                            sep_plot.append(np.flip(separations_full, axis=0)[i])
                        else:
                            break
                    sep_plot.append(sep_plot[-1] - (sep_plot[0] - sep_plot[1]))
                    U_plot.append(0)
                    sep_plot = np.flip(sep_plot, axis=0)
                    U_plot = np.flip(U_plot, axis=0)

                    ax.plot(sep_plot, U_plot, linestyle='-', marker='None',
                        label=r'$\sigma$'+'={}nm ({})'.format(forcefield['sigma'],
                        round(fit, 2)))

                ax.set_xlim(radius1 + radius2 - 0.5, target.separations[-1])
                ax.set_ylim(min(min(target.potential), min(U_mins)), 0)
                ax.set_xlabel('r, nm')
                ax.set_ylabel('U, kcal/mol')
                plt.legend(loc='lower right')
                _square_plot(ax)
                plt.tight_layout()
                fig.savefig('{}-R{}nm-R{}nm.pdf'.format(tag, radius1, radius2))

def test_fit(forcefields, radius, tag='cgff', cross=False, sigmas=None):
    if cross:
        if cross in ['CH2', 'CH3']:
            adir = 'united_atom'
        elif cross in ['MMM', 'MME']:
            adir = 'coarse_grained'
        target_path = os.path.join('utils', 'target_data', 'np_alkane', adir,
            'truncated', 'U_{}nm_{}_truncated.txt'.format(int(radius), cross))
        configs = 250
        rds = False
    else:
        target_path = os.path.join('utils', 'target_data', 'np_np', 'truncated',
            'U_{}nm_truncated.txt'.format(int(radius)))
        rds = True
    target = np_opt.load_target(resource_filename(resource_package, target_path))
    target.separations /= 10

    fig, ax = plt.subplots()

    ax.errorbar(target.separations, target.potential, yerr=target.error,
        linestyle='None', marker='o', color='black', label='Atomistic')

    U_mins = []
    for i, forcefield in enumerate(forcefields):
        if sigmas:
            nano = np_opt.CG_nano(radius, sigma=sigmas[i])
        else:
            if forcefield.sigma.value > 1.0:
                configs = 50
            else:
                configs = 10
            nano = np_opt.CG_nano(radius, sigma=forcefield['sigma'])
        if cross:
            system = np_opt.System(mb.Compound(pos=np.zeros(3)), nano)
        else:
            system = np_opt.System(nano)

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
            label = sigmas[i]
        else:
            label = forcefield['sigma']
        ax.plot(separations_full, U_cg_full, linestyle='-', marker='None',
            label=r'$\sigma$'+'={}nm ({})'.format(label, round(fit, 2)))

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
    _square_plot(ax)
    plt.tight_layout()
    if cross:
        fig.savefig('{}-R{}nm-{}.pdf'.format(tag, radius, cross))
    else:
        fig.savefig('{}-R{}nm.pdf'.format(tag, radius))

def test_linear_regression(forcefields):
    """Perform linear regression of Mie parameters

    Performs a linear regression of the Mie parameters epsilon, n, and m as a
    function of sigma.

    Parameters
    ----------
    forcefields : list of np_opt.Mie
    """
    sigma = [forcefield['sigma'] for forcefield in forcefields]
    epsilon = [forcefield['epsilon'] for forcefield in forcefields]
    n = [forcefield['n'] for forcefield in forcefields]
    m = [forcefield['m'] for forcefield in forcefields]

    fits = {}
    for param in ['epsilon', 'n', 'm']:
        yvals = [forcefield[param] for forcefield in forcefields]
        slope, intercept, r, p, std_err = linregress(sigma, yvals)
        fits[param] = tuple(slope, intercept, r)

    return fits

def test_25nm(forcefields, tag='cgff', cross=False, sigmas=None):
    """Test transferability to a nanoparticle with radius 25nm

    Parameters
    ----------
    forcefields : np_opt.Forcefield or list of np_opt.Forcefield
        Forcefield object(s) containing parameters to use to test transferability
    tag : str, optional, default='cgff'
        Filename tag to prepend to all plot PDFs
    cross : str, optional, default=False
        If not False, provides a short string indicating the Forcefield contains
        interactions for cross interactions with a different bead type. Valid
        options are 'CH2', 'CH3', 'MMM', and 'MME'.
    """
    if cross:
        if cross in ['CH2', 'CH3']:
            adir = 'united_atom'
        elif cross in ['MMM', 'MME']:
            adir = 'coarse_grained'
        target_path = os.path.join('utils', 'target_data', 'np_alkane', adir,
            'U_25nm_{}.txt'.format(cross))
    else:
        target_path = os.path.join('utils', 'target_data', 'np_np', 'U_25nm.txt')
    target = np_opt.load_target(resource_filename(resource_package, target_path))
    target.separations /= 10

    fig, ax = plt.subplots()
    ax.plot(target.separations, target.potential, linestyle='None', marker='x',
        color='black', label='Atomistic')

    for i, forcefield in enumerate(forcefields):
        if forcefield.sigma.value < 0.75:
            continue
        if sigmas:
            nano = np_opt.CG_nano(25.0, sigma=sigmas[i])
        else:
            nano = np_opt.CG_nano(25.0, sigma=forcefield['sigma'])
        if cross:
            system = np_opt.System(mb.Compound(pos=np.zeros(3)), nano)
            U_cg = np.array([pot[0] for pot in system.calc_potential(forcefield,
                target.separations, configurations=250)])
        else:
            system = np_opt.System(nano)
            if forcefield.sigma.value > 1.0:
                configs = 50
            else:
                configs = 10
            U_cg = np.array([pot[0] for pot in system.calc_potential(forcefield,
                target.separations, configurations=configs,
                r_dependent_sampling=True)])
        fit = calc_fit(target.potential, U_cg)
        if sigmas:
            label = sigmas[i]
        else:
            label = forcefield['sigma']
        ax.plot(target.separations, U_cg, linestyle='-', marker='o',
            label=r'$\sigma$'+'={}nm ({})'.format(label, round(fit, 2)))

    if cross:
        ax.set_xlim(25 + 0.1, target.separations[-1] + 1)
    else:
        ax.set_xlim(50 - 0.5, target.separations[-1] + 1)
    ax.set_ylim(min(target.potential) - 0.25, 0)
    ax.set_xlabel('r, nm')
    ax.set_ylabel('U, kcal/mol')
    plt.legend(loc='lower right')
    _square_plot(ax)
    plt.tight_layout()
    if cross:
        fig.savefig('{}-R25nm-{}.pdf'.format(tag, cross))
    else:
        fig.savefig('{}-R25nm.pdf'.format(tag))

def run_all(forcefields, tag='cgff', cross=False, rc_path=None, sigmas=None):
    """Execute all parameter tests

    Parameters
    ----------
    forcefields : np_opt.Forcefield or list of np_opt.Forcefield
        Forcefield object(s) containing parameters to use to test transferability
    tag : str, optional, default='cgff'
        Filename tag to prepend to all plot PDFs
    cross : str, optional, default=False
        If not False, provides a short string indicating the Forcefield contains
        interactions for cross interactions with a different bead type. Valid
        options are 'CH2', 'CH3', 'MMM', and 'MME'.
    rc_path : str, optional, default=None
        Path to a matplotlibrc file to load defaults for plots generated during
        testing. By default the matplotilbrc path in the 'utils' directory of the
        nanoparticle_optimization package will be used, however, if your current
        working directory contains a matplotlibrc file then that file will take
        precedence.
    """
    if not rc_path:
        rc_path = resource_filename(resource_package,
                                    os.path.join('utils', 'matplotlibrc'))
    matplotlib.rc_file(rc_path)

    if isinstance(forcefields, np_opt.Forcefield):
        forcefields = [forcefields]

    '''
    if not cross:
        test_mixed(forcefields, tag)
    '''

    '''
    for radius in range(2, 10):
        test_fit(forcefields, radius, tag, cross=cross, sigmas=sigmas)
    '''

    test_25nm(forcefields, tag=tag, cross=cross, sigmas=sigmas)
