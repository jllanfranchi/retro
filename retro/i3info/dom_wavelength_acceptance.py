#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Produce data tables containing DOM wavelength acceptance, the number-density
distribution of Cherenkov light, and the product of the two all as functions of
wavelength.

This allows for e.g. importance-sampling to produce light from the most useful
distributions and for reweighting photons generated by CLSim when one or both
of these has not been accounted for (either by sampling distribution or in
photon weights).
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'ALPHA', 'WL0', 'WL1',
    'get_phase_refr_index', 'cherenkov_dN_dXdwlen', 'dom_wavelength_acceptance'
]

import numpy as np

from icecube.clsim.GetIceCubeDOMAcceptance import I3Units, GetIceCubeDOMAcceptance
from icecube import icetray, dataclasses # pylint: disable=unused-import


ALPHA = 0.0072973525664
"""Fine-structure constant
https://en.wikipedia.org/wiki/Fine-structure_constant"""

WL0 = 260
"""Standard shortest wavelength to consider, in nm"""

WL1 = 680
"""Standard longest wavelength to consider, in nm"""


def get_phase_refr_index(wlen):
    """Get the phase refractive index for ice as a function of wavelength.

    See: https://wiki.icecube.wisc.edu/index.php/Refractive_index_of_ice#Numerical_values_for_ice
    or eqn 3 in https://arxiv.org/abs/hep-ex/0008001v1

    Parameters
    ----------
    wlen
        Wavelength in units of nm

    Returns
    -------
    n
        Phase refractive index.

    """
    # Convert wavelength to micrometers
    wl_um = wlen/1000
    return 1.55749 - 1.57988*wl_um + 3.99993*wl_um**2 - 4.68271*wl_um**3 + 2.09354*wl_um**4


def cherenkov_dN_dXdwlen(wlen, beta=1):
    """Number of photons per differential path length the Cherenkov emitter
    travels and per differential wavelength.

    You can simply multiply the result by a path length to "integrate out" path
    length since there is no x-dependence in the equation. However, since there
    is wavelength dependence in both the $lambda^-2$ factor and implicitly in
    the index of refraction, one would need to integrate over lambda more
    carefully.

    See, e.g., https://physics.stackexchange.com/a/105765

    Parameters
    ----------
    wlen
        Wavelength in units of nm

    beta
        Beta factor of particle emitting the Cherenkov light

    Returns
    -------
    dN_dXdwlen : float

    """
    return np.clip(
        np.pi * ALPHA / wlen**2 * (1 - 1 / (beta * get_phase_refr_index(wlen))**2),
        a_min=0,
        a_max=np.inf
    )


def dom_wavelength_acceptance(
        wlens=np.linspace(WL0, WL1, (WL1 - WL0)/5 + 1),
        beta=1
    ):
    """
    Parameters
    ----------
    wlens : iterable
        Wavelenghts, in nm

    beta : float
        Beta factor of particle emitting the Cherenkov light

    """
    dom_acceptance = GetIceCubeDOMAcceptance()
    acceptance = []
    cherenkov = []
    for wlen in wlens:
        acceptance.append(dom_acceptance.GetValue(wlen*I3Units.nanometer))
        cherenkov.append(cherenkov_dN_dXdwlen(wlen=wlen, beta=beta))
    combined = [a*c for a, c in zip(acceptance, cherenkov)]

    integral_combined = np.trapz(y=combined, x=wlens)
    integral_acceptance = np.trapz(y=acceptance, x=wlens)
    integral_cherenkov = np.trapz(y=cherenkov, x=wlens)
    combined /= integral_combined
    wavelength_combined = np.array(tuple(zip(wlens, combined)))
    wavelength_ckv_accept = np.array(tuple(zip(wlens, cherenkov, acceptance)))

    print('    wavelength (nm)  acceptance*cherenkov')
    print(wavelength_combined)
    print('integral(acceptance):           {:.8e}'.format(integral_acceptance))
    print('integral(Cherenkov):            {:.8e}'.format(integral_cherenkov))
    print('integral(acceptance*Cherenkov): {:.8e}'.format(integral_combined))
    print('fraction (combined. / cherenk.) = ', integral_combined/integral_cherenkov)

    header = (
        'Sampled DOM wavelength acceptance\n'
        'wavelength (nm), acceptance'
    )
    fpath = 'sampled_dom_wavelength_acceptance.csv'
    np.savetxt(fpath, wavelength_combined, delimiter=',', header=header)
    print('Saved sampled wavelength acceptance to "{}"'.format(fpath))

    header = (
        'Sampled Cherenkov distribution and DOM wavelength acceptance\n'
        'wavelength (nm),'
        ' Cherenkov light dN_dXdwlen,'
        ' IceCube DOM wavelength acceptance'
    )
    fpath = 'sampled_cherenkov_distr_and_dom_acceptance_vs_wavelength.csv'
    np.savetxt(fpath, wavelength_ckv_accept, delimiter=',', header=header)
    print('Saved wavelen + Ckv distr + DOM accept to "{}"'.format(fpath))


if __name__ == '__main__':
    dom_wavelength_acceptance()
