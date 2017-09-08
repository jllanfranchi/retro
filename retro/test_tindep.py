#!/usr/bin/env python
# coding: utf-8
# pylint: disable=wrong-import-position, invalid-name

"""
Create time- and DOM-independent "whole-detector" retro table.

Define a Cartesian grid that covers all of the IceCube fiducial volume, then
tabulate for each voxel the survival probability for photons coming from any
DOM at any time to reach that voxel. Also, tabulate the "average surviving
photon," defined by its x, y, and z components (which differs from the original
time- and DOM-dependent retro tables, wherein length, theta, and deltaphi are
used to characterize the average surviving photon).

Note that the length of the average surviving photon vector can be interpreted
as a measure of the directionality required for a photon to reach a DOM. I.e.,
if its length is 1, then only photons going exactly opposite that direction
will make it to a DOM (to within statistical and bin-size uncertainties used to
arrive at the average photon. If the length is _less_ than 1, then other
directions besides the average photon direction will be accepted, with
increasing likelihood as that length decreases towards 0.

The new table is in (x, y, z)--independent of time and DOM--and can be used to
scale the photons expected to reach any DOM at any time due to a hypothesis
that generates some number of photons (with an average direction / length) in
any of the voxel(s) of this table.
"""



from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
import cPickle as pickle
import os
from os.path import abspath, dirname, isfile, join
import sys
import time

import numpy as np
import pyfits

os.sys.path.append(dirname(dirname(abspath('__file__'))))
from retro import (powerspace, spherical_volume, extract_photon_info, pol2cart,
                   sph2cart)
from retro.shift_and_bin import shift_and_bin
from retro.sphbin2cartbin import sphbin2cartbin

from pisa.utils.hash import hash_obj
from pisa.utils.format import list2hrlist
from genericUtils import timediffstamp


def binmap_meta(r_max, r_power, n_rbins, n_costhetabins, n_phibins, binwidth,
                oversample, antialias):
    """Get metadata for spherical to Cartesian bin mapping, including the file
    name, hash string, and a dict with all of the parameters that contributed
    to these which can be passed via ``**binmap_kw`` to the `sphbin2cartbin`
    function.

    Parameters
    ----------
    r_max : float
        Maximum radius in Retro (t,r,theta)-binned DOM table (meters)

    r_power : float
        Binning in radial direction is regular in the inverse of this power.
        I.e., every element of `np.diff(r**(1/r_power))` is equal.

    n_rbins, n_costhetabins, n_phibins : int

    binwidth : float
        Cartesian bin widths, same in x, y, and z (meters)

    oversample : int
        Oversample factor, same in x, y, and z

    antialias : int
        Antialias factor

    Returns
    -------
    fname : string
        File name for the specified bin mapping

    binmap_hash : length-8 string
        Hex digits represented as a string.

    binmap_kw : dict
        The keyword args used for the hash.

    """
    binmap_kw = dict(
        r_max=r_max, r_power=r_power,
        n_rbins=n_rbins, n_costhetabins=n_costhetabins, n_phibins=n_phibins,
        x_bw=x_bw, y_bw=y_bw, z_bw=z_bw,
        x_oversample=x_oversample, y_oversample=y_oversample,
        z_oversample=z_oversample,
        antialias=antialias
    )
    binmap_hash = hash_obj(binmap_kw, hash_to='hex', full_hash=True)
    fname = (
        'sph2cart_binmap'
        '_%s'
        '_nr{n_rbins:d}_ncostheta{n_costhetabins:d}_nphi{n_phibins:d}'
        '_rmax{r_max:f}_rpwr{r_power}'
        '_bw{bw:.6f}'
        '_os{oversample:d}'
        '_aa{antialias:d}'
        '.pkl'.format(**binmap_kw)
    ) % binmap_hash
    return fname, binmap_hash, binmap_kw


def geom_meta(geom):
    """Hash on geometry. Note that the values are rounded to the nearest
    centimeter for hashing purposes. (Also, the values are converted to
    integers at this precision to eliminate any possible float32 / float64
    issues that could cause discrepancies in hash values for what we consider
    to be equal geometries.)

    Parameters
    ----------
    geom : shape (n_strings, n_depths, 3) numpy ndarray, dtype float32 or float64

    Returns
    -------
    hash_val : length-8 str
        Hex characters convert to a string of length 8

    """
    assert len(geom.shape) == 3
    assert geom.shape[2] == 3
    rounded_ints = np.round(geom * 100).astype(np.int)
    geom_hash = hash_obj(rounded_ints, hash_to='hex', full_hash=True)
    return geom_hash


def tdi_table_meta(binmap_hash, geom_hash, dom_tables_hash, times_str,
                   x_min, x_max, y_min, y_max, z_min, z_max, binwidth,
                   n_phibins, anisotropy):
    """Get metadata for a time- and DOM-independent Cartesian (x, y, z)-binned
    table.

    Parameters
    ----------
    binmap_hash : string
    geom_hash : string
    dom_tables_hash : string
    times_str : string
    x_lims, y_lims, z_lims : 2-tuples of floats
    binwidth : float
    n_phibins : int
    anisotropy : None or tuple

    Returns
    -------
    fbasename : string
    tdi_hash : string
    tdi_kw : string

    """
    tdi_kw = dict(
        geom_hash=geom_hash, binmap_hash=binmap_hash,
        dom_tables_hash=dom_tables_hash, times_str=times_str,
        x_min=x_lims[0], x_max=x_lims[1],
        y_min=y_lims[0], y_max=y_lims[1],
        z_min=z_lims[0], z_max=z_lims[1],
        binwidth=binwidth, nphi=n_phibins,
        anisotropy=anisotropy
    )
    hash_params = deepcopy(tdi_kw)
    for param in ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max']:
        rounded_int = int(np.round(hash_params[param]*100))
        hash_params[param] = rounded_int
        tdi_kw[param] = float(rounded_int) / 100
    hash_params['binwidth'] = int(np.round(hash_params['binwidth'] * 1e10))
    tdi_hash = hash_obj(hash_params, hash_to='hex', full_hash=True)
    if dom_tables_hash is None:
        dom_tables_hash = 'x'*16
    if anisotropy is None:
        anisotropy_str = '0'
    fbasename = (
        'retro_tdi_table'
        '_%s'
        '_binmap_{binmap_hash:s}'
        '_geom_{geom_hash:s}'
        '_dtbl_{dom_tables_hash:s}'
        '{times_str:s}'
        '_lims_x{x_lims[0]:.3f}_{x_lims[1]:.3f}_y{y_lims[0]:.3f}_{y_lims[1]:.3f}_z{z_lims[0]:.3f}_{z_lims[1]:.3f}'
        '_binwidth{binwidth:.9f}'
        '_nphi{nphi}'
        '_anisot_{anisotropy_str:s}'
        .format(**tdi_kw)
    ) % tdi_hash
    return fbasename, tdi_hash, tdi_kw


def generate_tdi_table(tables_dir, geom_fpath, dom_tables_hash,
                       r_max, r_power,
                       n_rbins, n_costhetabins, n_phibins, n_tbins,
                       x_lims, y_lims, z_lims,
                       binwidth, oversample, antialias,
                       strings=slice(None),
                       depths=slice(None),
                       times=slice(None),
                       recompute_binmap=False,
                       recompute_table=False):
    """Create a time- and DOM-independent Cartesian (x,y,z)-binned Retro
    tables (if it doesn't already exist or if the user requests that it be
    re-computed) and saves the tables to disk.

    The intermediate step of computing a bin mapping from polar (r, theta)
    coordinates for the source (t, r, theta)-binned DOM Retro tables is also
    performed if it hasn't already been saved to disk or if the user forces
    its recomputation; the result is stored to disk for future use.


    Parameters
    ----------
    tables_dir
    geom_fpath
    dom_tables_hash
    n_phibins : int
    x_lims, y_lims, z_lims : 2-tuples of floats
    binwidth : float
    oversample : int
    antialias : int
    strings : int, sequence, slice
        Select only these strings by indexing into the geom array

    depths : int, sequence, slice
        Select only these depth indices by indexing into the geom array

    times : int, sequence, slice
        Sum over only these times

    recompute_binmap : bool
        Force recomputation of bin mapping even if it already exists; existing
        file will be overwritten

    recompute_table : bool
        Force recomputation of table files even if the already exist; existing
        files will be overwritten


    Returns
    -------
    binmap_data : dict
        Contains 'ind_arrays', 'vol_arrays', and 'kwargs'

    binned_sp : shape (nx,ny,nz) numpy ndarray, dtype float32
        Survival probability table

    binned_px, binned_py, binned_pz : shape (nx,ny,nz) numpy ndarray, dtype float32
        Tables with average photon directionality, one each for x, y, and z
        components, respectively

    """
    if dom_tables_hash is None:
        dom_tables_hash = 'x'
        r_max = 400
        r_power = 2
        n_rbins = 200
        n_costhetabins = 40
        n_tbins = 300
    else:
        raise ValueError('Cannot handle non-None `dom_tables_hash`')

    nx = int(round((x_lims[1] - x_lims[0]) / binwidth))
    ny = int(round((y_lims[1] - y_lims[0]) / binwidth))
    nz = int(round((z_lims[1] - z_lims[0]) / binwidth))
    assert np.abs(x_lims[0] + nx * binwidth - x_lims[1]) < 1e-6
    assert np.abs(y_lims[0] + ny * binwidth - y_lims[1]) < 1e-6
    assert np.abs(z_lims[0] + nz * binwidth - z_lims[1]) < 1e-6

    xyz_shape = (nx, ny, nz)
    print('xyz_shape:', xyz_shape)

    geom = np.load(geom_fpath)

    depth_indices = np.atleast_1d(range(60)[depths])
    string_indices = np.atleast_1d(np.arange(86)[strings])

    subdet_doms = {'ic': [], 'dc': []}
    dc_strings = range(79, 86)
    for string_idx in string_indices:
        dom_coords = geom[string_idx:string_idx+1, depths, :]
        if string_idx in dc_strings:
            subdet_doms['dc'].append(dom_coords)
        else:
            subdet_doms['ic'].append(dom_coords)
    for subdet in subdet_doms.keys():
        dom_string_list = subdet_doms[subdet]
        if not dom_string_list:
            subdet_doms.pop(subdet)
        else:
            subdet_doms[subdet] = np.concatenate(dom_string_list, axis=0)
    geom = np.atleast_3d(geom[strings, depths, :])

    geom_hash = geom_meta(geom)

    r_edges = powerspace(0, r_max, n_rbins + 1, r_power)
    theta_edges = np.arccos(np.linspace(1, -1, n_costhetabins + 1))

    R, THETA = np.meshgrid(r_edges, theta_edges, indexing='ij')
    coords = []
    exact_vols = []
    for ri in range(n_rbins):
        subcoords = []
        sub_exact_vols = []
        for ti in range(int(np.ceil(n_costhetabins / 2.0))):
            rs = R[ri:ri+2, ti:ti+2]
            ts = THETA[ri:ri+2, ti:ti+2]
            bin_corner_coords = zip(rs.flat, ts.flat)
            dcostheta = np.abs(np.diff(np.cos([ts.max(), ts.min()])))
            exact_vol = spherical_volume(rmin=rs.max(), rmax=rs.min(),
                                         dcostheta=dcostheta, dphi=np.pi/2)
            sub_exact_vols.append(exact_vol)
        exact_vols.append(sub_exact_vols)
    exact_vols = np.array(exact_vols)

    binmap_fname, binmap_map_hash, binmap_kw = binmap_meta(
        r_max=r_max, r_power=r_power,
        n_rbins=n_rbins, n_costhetabins=n_costhetabins, n_phibins=n_phibins,
        binwidth=binwidth, oversample=oversample, antialias=antialias
    )
    fpath = join(tables_dir, fname)

    print('params:')
    print(binmap_kw)

    if not recompute_binmap and isfile(fpath):
        sys.stdout.write('Loading binmap from file\n  "%s"\n' % fpath)
        sys.stdout.flush()

        t0 = time.time()
        binmap_data = pickle.load(file(fpath, 'rb'))
        ind_arrays = binmap_data['ind_arrays']
        vol_arrays = binmap_data['vol_arrays']
        t1 = time.time()
        print('Time to load from pickle:', timediffstamp(t1 - t0))

    else:
        sys.stdout.write('Computing binmapping...\n')
        sys.stdout.flush()

        t0 = time.time()
        ind_arrays, vol_arrays = sphbin2cartbin(**binmap_kw)
        t1 = time.time()
        print('time to compute:', timediffstamp(t1 - t0))

        print('Writing binmapping to file\n  "%s"' % fpath)
        binmap_data = OrderedDict([
            ('kwargs', binmap_kw),
            ('ind_arrays', ind_arrays),
            ('vol_arrays', vol_arrays)
        ])
        pickle.dump(binmap_data, file(fpath, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        t2 = time.time()
        print('time to pickle the binmapping:', timediffstamp(t2 - t1))
    print('')

    binned_vol = np.sum([va.sum() for va in vol_arrays])
    exact_vol = spherical_volume(rmin=0, rmax=r_max, dcostheta=-1, dphi=np.pi/2)
    print('exact vol = %f, binned vol = %f (%e fract error)'
          % (exact_vol, binned_vol, (binned_vol-exact_vol)/exact_vol))

    ind_bin_vols = np.array([va.sum() for va in vol_arrays])
    fract_err = ind_bin_vols/exact_vols.flat - 1
    abs_fract_err = np.abs(fract_err)
    worst_abs_fract_err = np.max(abs_fract_err)
    flat_idx = np.where(abs_fract_err == worst_abs_fract_err)[0][0]
    r_idx, costheta_idx = divmod(flat_idx, int(np.ceil(n_costhetabins/2)))
    print('worst single-bin fract err: %e;'
          'r_idx=%d, costheta_idx=%d;'
          'binned vol=%e, exact vol=%e'
          % (worst_abs_fract_err, r_idx, costheta_idx, ind_bin_vols[flat_idx],
             exact_vols[r_idx, costheta_idx]))

    all_t_bins = list(range(n_tbins))
    remaining_t_bins = np.array(all_t_bins)[times].tolist()
    if all_t_bins == remaining_t_bins:
        times_str = ''
    else:
        times_str = '_tbins' + list2hrlist(remaining_t_bins)

    fbasename, tdi_hash, tdi_kw = tdi_table_meta(
        binmap_hash=binmap_hash, geom_hash=geom_hash, dom_tables_hash=None,
        times_str=times_str,
        x_min=x_lims[0], x_max=x_lims[1],
        y_min=y_lims[0], y_max=y_lims[1],
        z_min=z_lims[0], z_max=z_lims[1],
        binwidth=binwidth, nphi=n_phibins, anisotropy=anisotropy
    )

    names = [
        'survival_prob',
        'avg_photon_x',
        'avg_photon_y',
        'avg_photon_z'
    ]
    if not recompute_table:
        for name in names:
            fpath = join(tables_dir, fbasename + '_' + name + '.fits')
            if not isfile(fpath):
                print('could not find table, will (re)compute\n%s\n' % fpath)
                recompute_table = True
                break

    # TODO: bake this entire thing a function, and make these return values
    if not recompute_table:
        print('Loading Retro tdi table from disk')
        for name in names:
            fpath = join(tables_dir, fbasename + '_' + name + '.fits')
            with pyfits.open(fpath) as fits_file:
                tmp = fits_file[0].data
            if name == 'survival_prob':
                binned_sp = tmp
            elif name == 'avg_photon_x':
                binned_px = tmp
            elif name == 'avg_photon_y':
                binned_py = tmp
            elif name == 'avg_photon_z':
                binned_pz = tmp
            del tmp
        return binmap_data, binned_sp, binned_px, binned_py, binned_pz

    # Instantiate accumulation arrays
    binned_spv = np.zeros((nx*ny*nz), dtype=np.float64)
    binned_px_spv = np.zeros((nx*ny*nz), dtype=np.float64)
    binned_py_spv = np.zeros((nx*ny*nz), dtype=np.float64)
    binned_pz_spv = np.zeros((nx*ny*nz), dtype=np.float64)
    binned_one_minus_sp = np.ones((nx*ny*nz), dtype=np.float64)

    t00 = time.time()
    for subdet, subdet_dom_coords in subdet_doms.items():
        print('subdet:', subdet)
        print('subdet_dom_coords.shape:', subdet_dom_coords.shape)
        for rel_ix, depth_index in enumerate(depth_indices):
            print('depth_index:', depth_index)
            dom_coords = subdet_dom_coords[:, rel_ix, :]

            t0 = time.time()
            table_fname = (
                'retro_nevts1000'
                '_{subdet:s}'
                '_DOM{depth_index:d}'
                '_r_cz_t_angles'
                '.fits'.format(
                    subdet=subdet.upper(), depth_index=depth_index
                )
            )
            photon_info, bin_edges = extract_photon_info(
                fpath=join(tables_dir, table_fname),
                dom_depth_index=depth_index
            )
            t1 = time.time()
            print('time to load the retro table:', timediffstamp(t1 - t0))

            sp = photon_info.survival_prob[depth_index]
            plength = photon_info.length[depth_index]
            ptheta = photon_info.theta[depth_index]
            pdeltaphi = photon_info.deltaphi[depth_index]

            plength *= np.cos(pdeltaphi)
            pz = plength * np.cos(ptheta)
            prho = plength * np.sin(ptheta)

            t_indep_sp = 1 - np.prod(1 - sp[times], axis=0)

            mask = t_indep_sp != 0
            scale = 1 / sp.sum(axis=0)[mask]

            t_indep_pz = np.zeros_like(t_indep_sp)
            t_indep_prho = np.zeros_like(t_indep_sp)

            t_indep_pz[mask] = (pz[times] * sp[times]).sum(axis=0)[mask] * scale
            t_indep_prho[mask] = (prho[times] * sp[times]).sum(axis=0)[mask] * scale

            t2 = time.time()
            print('time to marginalize out time dim, 1 depth:', timediffstamp(t2 - t1))

            shift_and_bin(
                ind_arrays=ind_arrays,
                vol_arrays=vol_arrays,
                dom_coords=dom_coords,
                survival_prob=t_indep_sp,
                prho=t_indep_prho,
                pz=t_indep_pz,
                nr=n_rbins,
                ntheta=n_costhetabins,
                r_max=r_max,
                binned_spv=binned_spv,
                binned_px_spv=binned_px_spv,
                binned_py_spv=binned_py_spv,
                binned_pz_spv=binned_pz_spv,
                binned_one_minus_sp=binned_one_minus_sp,
                x_min=x_lims[0],
                y_min=y_lims[0],
                z_min=z_lims[0],
                x_max=x_lims[1],
                y_max=y_lims[1],
                z_max=z_lims[1],
                bandwidth=bandwidth,
                oversample=oversample,
                anisotropy=None
            )
            t3 = time.time()
            print('time to shift and bin:', timediffstamp(t3 - t2))
            print('')

    print('Total time to shift and bin:', timediffstamp(t3 - t00))
    print('')

    binned_sp = (1 - binned_one_minus_sp).reshape(xyz_shape)
    del binned_one_minus_sp

    mask = binned_spv != 0
    binned_px_spv[mask] /= binned_spv[mask]
    binned_py_spv[mask] /= binned_spv[mask]
    binned_pz_spv[mask] /= binned_spv[mask]
    del mask

    # Rename so as to not mislead
    binned_px = binned_px_spv.reshape(xyz_shape)
    binned_py = binned_py_spv.reshape(xyz_shape)
    binned_pz = binned_pz_spv.reshape(xyz_shape)
    del binned_px_spv, binned_py_spv, binned_pz_spv

    t4 = time.time()
    print('time to normalize histograms:', timediffstamp(t4 - t3))
    print('')

    arrays_names = [
        (binned_sp, 'survival_prob'),
        (binned_px, 'avg_photon_x'),
        (binned_py, 'avg_photon_y'),
        (binned_pz, 'avg_photon_z')
    ]
    for array, name in arrays_names:
        fname = '%s_%s.fits' % (fbasename, name)
        fpath = join(tables_dir, fname)
        hdulist = pyfits.HDUList([
            pyfits.PrimaryHDU(array.astype(np.float32)),
            pyfits.ImageHDU(xyz_shape),
            pyfits.ImageHDU(np.array([x_lims, y_lims, z_lims])),
            pyfits.ImageHDU(all_doms)
        ])
        print('Saving %s to file\n%s\n' % (name, fpath))
        hdulist.writeto(fpath, clobber=True)
    t5 = time.time()
    print('time to save tables to disk:', timediffstamp(t5 - t4))
    print('')

    print('TOTAL RUN TIME:', timediffstamp(t5 - t00))

    return binmap_data, binned_sp, binned_px, binned_py, binned_pz


def parse_args(description=__doc__):
    """Parse command line args"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--tables-dir',
        help='Path to eirectory containing Retro tables'
    )
    parser.add_argument(
        '--geom-fpath',
        help='Path to geometry NPY file'
    )
    #parser.add_argument(
    #    '--t-max', type=float,
    #    help='''Maximum time bin edge in the source (t,r,theta)-binnned DOM
    #    Retro tables (nanoseconds)'''
    #)
    #parser.add_argument(
    #    '--r-max', type=float,
    #    help='''Maximum radial bin edge in the source (t,r,theta)-binnned DOM
    #    Retro tables (meters)'''
    #)
    #parser.add_argument(
    #    '--r-power', type=float,
    #    help='''Power used for radial power-law binning in source
    #    (t,r,theta)-binned DOM Retro tables'''
    #)
    #parser.add_argument(
    #    '--n-rbins', type=int,
    #    help='''Number of radial bins used in source (t,r,theta)-binned DOM
    #    Retro tables'''
    #)
    #parser.add_argument(
    #    '--n-costhetabins', type=int,
    #    help='''Number of costheta bins used in source (t,r,theta)-binned DOM
    #    Retro tables'''
    #)
    #parser.add_argument(
    #    '--n-tbins', type=int,
    #    help='''Number of time bins used in source (t,r,theta)-binned DOM Retro
    #    tables'''
    #)
    parser.add_argument(
        '--x-lims', nargs=2, type=float,
        help='''Limits of the produced table in the x-direction (meters)'''
    )
    parser.add_argument(
        '--y-lims', nargs=2, type=float,
        help='''Limits of the produced table in the y-direction (meters)'''
    )
    parser.add_argument(
        '--z-lims', nargs=2, type=float,
        help='''Limits of the produced table in the z-direction (meters)'''
    )
    parser.add_argument(
        '--binwidth', type=float,
        help='''Binwidth in x, y, and z directions (meters). Must divide each
        of --x-lims, --y-lims, and --z-lims into an integral number of bins.'''
    )
    parser.add_argument(
        '--oversample', type=int,
        help='''Oversampling factor in the x-, y-, and z- directions (int >=
        1).'''
    )
    parser.add_argument(
        '--antialias', type=int,
        help='''Antialiasing factor (int between 1 and 50).'''
    )
    parser.add_argument(
        '--anisotropy', nargs='2', metavar='DIR MAG', required=False,
        default=None,
        help='''[NOT IMPLEMENTED] Simple ice anisotropy parameters to use: DIR
        for azimuthal direction of low-scattering axis (radians) and MAG for
        magnitude of anisotropy (unitless). If not specified, no anisotropy is
        modeled.'''
    )
    parser.add_argument(
        '--strings', type=str, nargs='+', required=False, default=None,
        help='''Only use these strings (indices start at 1, as per the IceCube
        convention). Specify a human-redable string, e.g. "80-86" to include
        only DeepCore strings, or "26-27,35-37,45-46,80-86" to include the
        IceCube strings that are considered to be part of DeepCore as well as
        "DeepCore-proper" strings. Note that spaces are acceptable.'''
    )
    parser.add_argument(
        '--depths', type=str, nargs='+', required=False, default=None,
        help='''Only use these depths, specified as indices with shallowest at
        0 and deepest at 59. Note that the actual depths of the DOMs depends
        upon whether the string is in DeepCore or not. Specify a human-redable
        string, e.g. "50-59" to include depths {50, 51, ..., 59}. Or one
        could specify "4-59:5" to use every fifth DOM on each string. Note that
        spaces are acceptable.'''
    )
    parser.add_argument(
        '--times', type=str, nargs='+', required=False, default=None,
        help='''Only use these times (specified as indices) from the source
        (t,r,theta)-binned Retro DOM tables. Specify as a human-readable
        sequence, similarly to --strings and --depths.'''
    )
    parser.add_argument(
        '--recompute-binmap', action='store_true',
        help='''Recompute the bin mapping even if the file exists; the existing
        file will be overwritten.'''
    )
    parser.add_argument(
        '--recompute-table', action='store_true',
        help='''Recompute the Retro time- and DOM-independent (TDI) table even
        if the corresponding files exist; these files will be overwritten.'''
    )
    kwargs = vars(parser.parse_args())
    return kwargs


if __name__ == '__main__':
    binmap_data, binned_sp, binned_px, binned_py, binned_pz = \
            generate_tdi_table(**parse_args())