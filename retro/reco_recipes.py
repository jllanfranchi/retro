from __future__ import absolute_import, division, print_function

from copy import deepcopy
from collections import OrderedDict

import numpy as np


METHOD_DEPENDENCIES = dict(
    multinest=None,
    test=None,
    truth=None,
    crs=None,
    scipy=None,
    nlopt=None,
    skopt=None,
    fast=None,
    stopping_atm_muon_crs=None,
    crs_prefit=None,
    mn8d="crs_prefit",
    dn8d="crs_prefit",
)


def insert_dependencies(methods, recursion_level=0):
    """Insert reconstruction method dependencies if not already specified
    (before the dependent method).

    Note that `methods` is modified in place.

    Parameters
    ----------
    methods : list of strings
    recursion_level : int, optional

    Out
    ---
    methods

    """
    if recursion_level > 100:
        raise ValueError("Recursion limit reached")
    is_modified = False
    for method in deepcopy(methods):
        dependencies = METHOD_DEPENDENCIES[method]
        if not dependencies:
            continue
        for dependency in dependencies:
            if dependency not in methods:
                is_modified = True
                sys.stderr.write(
                    "Warning: reco method '{dep}' is required by method"
                    " '{meth}' but was not specified; `methods` is being"
                    " updated such that '{dep}' will be run before"
                    " '{meth}'.\n".format(dep=dependency, meth=method)
                )
                methods.insert(dependency, methods.index(method))
    if is_modified:
        insert_dependencies(methods, recursion_level=recursion_level + 1)





