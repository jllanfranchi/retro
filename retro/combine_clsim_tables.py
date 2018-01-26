#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, too-many-instance-attributes, too-many-locals

"""
Combine multiple Retro CLSim tables into a single table.
"""


from __future__ import absolute_import, division, print_function


__all__ = [
    'VALIDATE_KEYS', 'SUM_KEYS', 'ALL_KEYS',
    'combine_clsim_tables', 'parse_args', 'main'
]

__author__ = 'J.L. Lanfranchi'
__license__ = '''Copyright 2017 Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''


from argparse import ArgumentParser
from glob import glob
from os.path import abspath, basename, dirname, isfile, join, splitext
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    PARENT_DIR = dirname(dirname(abspath(__file__)))
    if PARENT_DIR not in sys.path:
        sys.path.append(PARENT_DIR)
from retro import COMPR_EXTENSIONS, expand, mkdir
from retro.table_readers import load_clsim_table_minimal


VALIDATE_KEYS = [
    'table_shape',
    'phase_refractive_index',
    'r_bin_edges',
    'costheta_bin_edges',
    't_bin_edges',
    'costhetadir_bin_edges',
    'deltaphidir_bin_edges'
] # yapf: disable
"""Values corresponding to these keys must match in all loaded tables"""

SUM_KEYS = [
    'table', 'n_photons'
] # yapf: disable
"""Sum together values corresponding to these keys in all tables"""

ALL_KEYS = VALIDATE_KEYS + SUM_KEYS
"""All keys expected to be in tables"""


def combine_clsim_tables(table_fpaths, outdir=None, overwrite=False):
    """Combine multiple CLSim-produced tables together into a single table.

    All tables specified must have the same binnings defined. Tables should
    also be produced using different random seeds; if corresponding metadata
    files can be found in the same directories as the CLSim tables, this will
    be enforced prior to loading and combining the actual tables together.

    Parameters
    ----------
    table_fpaths : string or iterable thereof
        Each string is glob-expanded

    outdir : string, optional
        Directory to which to save the combined table; if not specified, the
        resulting table will be returned but not saved to disk.

    Returns
    -------
    combined_table

    """
    # Get all input table filepaths, including glob expansion

    if isinstance(table_fpaths, basestring):
        table_fpaths = [table_fpaths]
    table_fpaths_tmp = []
    for fpath in table_fpaths:
        table_fpaths_tmp.extend(glob(expand(fpath)))
    table_fpaths = sorted(table_fpaths_tmp)

    print(
        'Found {} tables to combine:\n  {}'
        .format(len(table_fpaths), '\n  '.join(table_fpaths))
    )

    # Formulate output filenames and check if they exist

    output_fpaths = None
    if outdir is not None:
        outdir = expand(outdir)
        mkdir(outdir)
        output_fpaths = {k: join(outdir, k + '.npy') for k in ALL_KEYS}
        output_fpaths['source_tables'] = join(outdir, 'source_tables.txt')
        if not overwrite:
            for fpath in output_fpaths:
                if isfile(fpath):
                    raise IOError('File {} exists'.format(fpath))
        print(
            'Output files will be written to:\n  {}'
            .format('\n  '.join(output_fpaths.values()))
        )

    # Combine the tables

    combined_table = None
    for fpath in table_fpaths:
        table = load_clsim_table_minimal(fpath)

        if combined_table is None:
            combined_table = table
            continue

        if set(table.keys()) != set(SUM_KEYS + VALIDATE_KEYS):
            raise ValueError(
                'Table keys {} do not match expected keys {}'
                .format(sorted(table.keys()), sorted(ALL_KEYS))
            )
        for key in VALIDATE_KEYS:
            if not np.array_equal(table[key], combined_table[key]):
                raise ValueError('Unequal {} in file {}'.format(key, fpath))

        for key in SUM_KEYS:
            combined_table[key] += table[key]

        del table

    # Save the data to npy files on disk

    if outdir is not None:
        basenames = []
        for fpath in table_fpaths:
            base = basename(fpath)
            rootname, ext = splitext(base)
            if ext.lstrip('.') in COMPR_EXTENSIONS:
                base = rootname
            basenames.append(base)

        fpath = output_fpaths['source_tables']
        with open(fpath, 'w') as fobj:
            fobj.write('\n'.join(sorted(basenames)))
        print('Wrote "{}"'.format(fpath))

        for key in ALL_KEYS:
            fpath = output_fpaths[key]
            np.save(fptath, combined_table[key])
            print('Wrote "{}"'.format(fpath))

    return combined_table


def parse_args(description=__doc__):
    """Parse command line args.

    Returns
    -------
    args : Namespace

    """
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--table-fpaths', nargs='+', required=True,
        help='''Path(s) to Retro CLSim tables. Note that literal strings are
        glob-expanded.'''
    )
    parser.add_argument(
        '--outdir', required=True,
        help='''Directory to which to save the combined table. Defaults to same
        directory as the first file path specified by --table-fpaths.'''
    )
    return parser.parse_args()


def main():
    """Main function for calling combine_clsim_tables as a script"""
    args = parse_args()
    kwargs = vars(args)
    combine_clsim_tables(**kwargs)


if __name__ == '__main__':
    main()
