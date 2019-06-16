#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Perform reconstructions
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    "METHOD_DEPENDENCIES",
    "LLH_FUDGE_SUMMAND",
    "reco",
    "reco_event",
    "validate_and_translate_methods_arg",
    "insert_method_dependencies",
    "generate_prior_func",
    "generate_loglike_func",
    "make_llhp",
    "make_estimate",
    "main",
]

__author__ = "J.L. Lanfranchi, P. Eller"
__license__ = """Copyright 2017-2018 Justin L. Lanfranchi and Philipp Eller

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
from os.path import abspath, dirname, isfile, join
import sys
import time

import numpy as np
from six import string_types

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import __version__, GarbageInputError, init_obj
from retro.hypo.discrete_cascade_kernels import SCALING_CASCADE_ENERGY
from retro.hypo.discrete_muon_kernels import pegleg_eval
from retro.priors import (
    EXT_IC,
    PRI_COSINE,
    PRI_TIME_RANGE,
    PRI_UNIFORM,
    PRISPEC_OSCNEXT_PREFIT_TIGHT,
    PRISPEC_OSCNEXT_CRS_MN,
    Bound,
    get_prior_func,
)
from retro.reco_optimizer_wrappers import (
    run_test,
    run_with_truth,
    run_crs,
    run_scipy,
    run_skopt,
    run_nlopt,
    run_dynesty,
    run_multinest,
)
from retro.retro_types import EVT_DOM_INFO_T, EVT_HIT_INFO_T, FitStatus
from retro.tables.pexp_5d import generate_pexp_and_llh_functions
from retro.utils.geom import rotate_points, add_vectors
from retro.utils.misc import sort_dict
from retro.utils.stats import estimate_from_llhp


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

LLH_FUDGE_SUMMAND = -1000


def reco(
    methods,
    events,
    dom_tables,
    tdi_tables,
    get_llh,
    report_every,
    redo_failed=False,
    redo_all=False,
    save_llhp=False,
    debug=False,
):
    """
    Setup tables, get events, run reconstructons on them, and optionally store
    results to disk.

    Note that "recipes" for different reconstructions are defined in the
    `Reco.run` method.

    Parameters
    ----------
    methods : string or iterable thereof
        Each must be one of `METHOD_DEPENDENCIES.keys()`

    events : iterable

    dom_tables

    tdi_tables

    get_llh

    redo_failed : bool, optional
        If `True`, reconstruct each event that either hasn't been
        reconstructed with each method (as usual), but also re-reconstruct
        events that have `fit_status` indicating a failure (i.e., all
        events will be reconstructed using a given method unless they have
        for that method `fit_status == FitStatus.OK`). Default is False.

    redo_all : bool, optional
        If `True`, reconstruct all events with all `methods`, regardless if
        they have been reconstructed with these methods previously.

    save_llhp : bool, optional
        Save likelihood values & corresponding parameter values within a
        LLH range of the max LLH (this takes up a lot of disk space and
        creats a lot of files; use with caution if running jobs en masse)

    debug : bool

    """
    start_time = time.time()

    methods = validate_and_translate_methods_arg(methods)
    print("Running {} reconstruction(s) on specified events".format(methods))

    debug = bool(debug)

    event_counter = 0
    for event in events:
        event.meta["prefix"] = join(
            event.meta["events_root"],
            "recos",
            "evt{}.".format(event.meta["event_idx"]),
        )
        event_counter += 1
        print(
            'Reconstructing event #{} (index {} in dir "{}") using method(s) {}'.format(
                event_counter,
                event.meta["event_idx"],
                event.meta["events_root"],
                methods,
            )
        )
        reco_event(
            event=event,
            methods=methods,
            dom_tables=dom_tables,
            get_llh=get_llh,
            redo_failed=redo_failed,
            redo_all=redo_all,
            save_llhp=save_llhp,
            report_every=report_every,
        )

    print("Total run time is {:.3f} s".format(time.time() - start_time))


def reco_event(
    methods,
    event,
    dom_tables,
    get_llh,
    redo_failed,
    redo_all,
    save_llhp,
    report_every,
):
    """Recipes for performing different kinds of reconstructions on a single
    event.

    Parameters
    ----------
    methods : str or iterable thereof
    event : OrderedDict
    dom_tables
    get_llh : callable
    redo_failed : bool
    redo_all : bool
    save_llhp : bool
    report_every : int >= 1

    """
    methods = validate_and_translate_methods_arg(methods)

    for method in methods:
        try:
            estimate_outf = join(
                event.meta["events_root"],
                "recos",
                "retro_{}.npy".format(method),
            )
            if isfile(estimate_outf):
                estimates = np.load(estimate_outf, mmap_mode="r")
                fit_status = estimates[event.meta["event_idx"]]["fit_status"]
                if fit_status != FitStatus.NotSet:
                    if redo_all:
                        print(
                            'Method "{}" already run on event; redoing'.format(
                                method
                            )
                        )
                    elif redo_failed and fit_status != FitStatus.OK:
                        print(
                            'Method "{}" already run on event but failed'
                            " previously; retrying".format(method)
                        )
                    else:
                        print(
                            'Method "{}" already run on event; skipping'.format(
                                method
                            )
                        )
                        continue

            if method in ("multinest", "test", "truth", "crs", "scipy", "nlopt", "skopt"):
                # -- simple 1-stage recos -- #

                hypo_handler = init_obj.setup_discrete_hypo(
                    cascade_kernel="scaling_aligned_one_dim",
                    track_kernel="pegleg",
                    track_time_step=1.0,
                )

                prior, priors_used = generate_prior_func(
                    event=event,
                    opt_param_names=hypo_handler.opt_param_names,
                    **PRISPEC_OSCNEXT_PREFIT_TIGHT
                )

                param_values = []
                log_likelihoods = []
                aux_values = []
                t_start = []
                loglike = generate_loglike_func(
                    event=event,
                    hypo_handler=hypo_handler,
                    dom_tables=dom_tables,
                    get_llh=get_llh,
                    param_values=param_values,
                    log_likelihoods=log_likelihoods,
                    aux_values=aux_values,
                    t_start=t_start,
                    report_every=report_every,
                )

                if method == "test":
                    run_info, fit_meta = run_test(
                        prior=prior,
                        hypo_handler=hypo_handler,
                        loglike=loglike,
                        seed=0,
                    )
                if method == "truth":
                    run_info, fit_meta = run_with_truth(
                        event=event,
                        hypo_handler=hypo_handler,
                        loglike=loglike,
                        prior=prior,
                    )
                elif method == "crs":
                    run_info, fit_meta = run_crs(
                        event=event,
                        prior=prior,
                        hypo_handler=hypo_handler,
                        loglike=loglike,
                        n_live=250,
                        max_iter=20000,
                        max_noimprovement=5000,
                        min_llh_std=0.1,
                        min_vertex_std=dict(x=1, y=1, z=1, time=3),
                        use_priors=False,
                        use_sobol=True,
                        seed=0,
                        report_every=report_every,
                    )
                elif method == "multinest":
                    run_info, fit_meta = run_multinest(
                        event=event,
                        prior=prior,
                        hypo_handler=hypo_handler,
                        loglike=loglike,
                        importance_sampling=True,
                        max_modes=1,
                        const_eff=True,
                        n_live=160,
                        evidence_tol=0.5,
                        sampling_eff=0.3,
                        max_iter=10000,
                        seed=0,
                        report_every=report_every,
                    )
                elif method == "scipy":
                    run_info, fit_meta = run_scipy(
                        event=event,
                        prior=prior,
                        hypo_handler=hypo_handler,
                        loglike=loglike,
                        method="differential_evolution",
                        eps=0.02,
                    )
                elif method == "nlopt":
                    run_info, fit_meta = run_nlopt(
                        event=event,
                        prior=prior,
                        hypo_handler=hypo_handler,
                        loglike=loglike,
                    )
                elif method == "skopt":
                    run_info, fit_meta = run_skopt(
                        event=event,
                        prior=prior,
                        hypo_handler=hypo_handler,
                        loglike=loglike,
                    )

                llhp = make_llhp(
                    all_param_names=hypo_handler.all_param_names,
                    event=event,
                    method=method,
                    log_likelihoods=log_likelihoods,
                    param_values=param_values,
                    aux_values=aux_values,
                    save=save_llhp,
                )
                make_estimate(
                    event=event,
                    method=method,
                    llhp=llhp,
                    remove_priors=True,
                    priors_used=priors_used,
                    run_info=run_info,
                    fit_meta=fit_meta,
                )

            elif method == "fast":
                hypo_handler = init_obj.setup_discrete_hypo(
                    cascade_kernel="scaling_aligned_point_ckv",
                    track_kernel="pegleg",
                    track_time_step=3.0,
                )

                prior, priors_used = generate_prior_func(
                    event=event,
                    opt_param_names=hypo_handler.opt_param_names,
                    **PRISPEC_OSCNEXT_PREFIT_TIGHT
                )

                param_values = []
                log_likelihoods = []
                aux_values = []
                t_start = []

                loglike = generate_loglike_func(
                    event=event,
                    hypo_handler=hypo_handler,
                    dom_tables=dom_tables,
                    get_llh=get_llh,
                    param_values=param_values,
                    log_likelihoods=log_likelihoods,
                    aux_values=aux_values,
                    t_start=t_start,
                    report_every=report_every,
                )

                run_info, fit_meta = run_crs(
                    event=event,
                    prior=prior,
                    hypo_handler=hypo_handler,
                    loglike=loglike,
                    n_live=160,
                    max_iter=10000,
                    max_noimprovement=1000,
                    min_llh_std=0.5,
                    min_vertex_std=dict(x=5, y=5, z=5, time=15),
                    use_priors=False,
                    use_sobol=True,
                    seed=0,
                    report_every=report_every,
                )

                llhp = make_llhp(
                    all_param_names=hypo_handler.all_param_names,
                    event=event,
                    method=method,
                    log_likelihoods=log_likelihoods,
                    param_values=param_values,
                    aux_values=aux_values,
                    save=save_llhp,
                )

                make_estimate(
                    event=event,
                    method=method,
                    llhp=llhp,
                    remove_priors=False,
                    run_info=run_info,
                    fit_meta=fit_meta,
                )

            elif method == "stopping_atm_muon_crs":
                hypo_handler = init_obj.setup_discrete_hypo(
                    track_kernel="stopping_table_energy_loss", track_time_step=3.0
                )

                prior, priors_used = generate_prior_func(
                    event=event,
                    opt_param_names=hypo_handler.opt_param_names,
                    x=dict(kind=PRI_UNIFORM, extents=EXT_IC["x"]),
                    y=dict(kind=PRI_UNIFORM, extents=EXT_IC["y"]),
                    z=dict(kind=PRI_UNIFORM, extents=EXT_IC["z"]),
                    time=dict(kind=PRI_TIME_RANGE),
                    track_zenith=dict(
                        kind=PRI_COSINE, extents=((0, Bound.ABS), (np.pi / 2, Bound.ABS))
                    ),
                )

                param_values = []
                log_likelihoods = []
                aux_values = []
                t_start = []

                loglike = generate_loglike_func(
                    event=event,
                    hypo_handler=hypo_handler,
                    dom_tables=dom_tables,
                    get_llh=get_llh,
                    param_values=param_values,
                    log_likelihoods=log_likelihoods,
                    aux_values=aux_values,
                    t_start=t_start,
                    report_every=report_every,
                )

                run_info, fit_meta = run_crs(
                    event=event,
                    prior=prior,
                    hypo_handler=hypo_handler,
                    loglike=loglike,
                    n_live=160,
                    max_iter=10000,
                    max_noimprovement=1000,
                    min_llh_std=0.,
                    min_vertex_std=dict(x=5, y=5, z=4, time=20),
                    use_priors=False,
                    use_sobol=True,
                    seed=0,
                    report_every=report_every,
                )

                llhp = make_llhp(
                    all_param_names=hypo_handler.all_param_names,
                    event=event,
                    method=method,
                    log_likelihoods=log_likelihoods,
                    param_values=param_values,
                    aux_values=aux_values,
                    save=save_llhp,
                )

                make_estimate(
                    event=event,
                    method=method,
                    llhp=llhp,
                    remove_priors=False,
                    run_info=run_info,
                    fit_meta=fit_meta,
                )

            elif method == "crs_prefit":
                hypo_handler = init_obj.setup_discrete_hypo(
                    cascade_kernel="scaling_aligned_point_ckv",
                    track_kernel="pegleg",
                    track_time_step=3.0,
                )

                prior, priors_used = generate_prior_func(
                    event=event,
                    opt_param_names=hypo_handler.opt_param_names,
                    **PRISPEC_OSCNEXT_PREFIT_TIGHT
                )

                param_values = []
                log_likelihoods = []
                aux_values = []
                t_start = []

                loglike = generate_loglike_func(
                    event=event,
                    hypo_handler=hypo_handler,
                    dom_tables=dom_tables,
                    get_llh=get_llh,
                    param_values=param_values,
                    log_likelihoods=log_likelihoods,
                    aux_values=aux_values,
                    t_start=t_start,
                    report_every=report_every,
                )

                run_info, fit_meta = run_crs(
                    event=event,
                    prior=prior,
                    hypo_handler=hypo_handler,
                    loglike=loglike,
                    n_live=160,
                    max_iter=10000,
                    max_noimprovement=1000,
                    min_llh_std=0.5,
                    min_vertex_std=dict(x=5, y=5, z=4, time=20),
                    use_priors=False,
                    use_sobol=True,
                    seed=0,
                    report_every=report_every,
                )

                llhp = make_llhp(
                    all_param_names=hypo_handler.all_param_names,
                    event=event,
                    method=method,
                    log_likelihoods=log_likelihoods,
                    param_values=param_values,
                    aux_values=aux_values,
                    save=save_llhp,
                )

                make_estimate(
                    event=event,
                    method=method,
                    llhp=llhp,
                    remove_priors=False,
                    run_info=run_info,
                    fit_meta=fit_meta,
                )

            elif method == "mn8d":
                hypo_handler = init_obj.setup_discrete_hypo(
                    cascade_kernel="scaling_aligned_one_dim",
                    track_kernel="pegleg",
                    track_time_step=1.0,
                )

                prior, priors_used = generate_prior_func(
                    event=event,
                    opt_param_names=hypo_handler.opt_param_names,
                    **PRISPEC_OSCNEXT_CRS_MN
                )

                param_values = []
                log_likelihoods = []
                aux_values = []
                t_start = []

                loglike = generate_loglike_func(
                    event=event,
                    hypo_handler=hypo_handler,
                    dom_tables=dom_tables,
                    get_llh=get_llh,
                    param_values=param_values,
                    log_likelihoods=log_likelihoods,
                    aux_values=aux_values,
                    t_start=t_start,
                    report_every=report_every,
                )

                run_info, fit_meta = run_multinest(
                    event=event,
                    prior=prior,
                    hypo_handler=hypo_handler,
                    loglike=loglike,
                    importance_sampling=True,
                    max_modes=1,
                    const_eff=True,
                    n_live=250,
                    evidence_tol=0.02,
                    sampling_eff=0.5,
                    max_iter=10000,
                    seed=0,
                    report_every=report_every,
                )

                llhp = make_llhp(
                    all_param_names=hypo_handler.all_param_names,
                    event=event,
                    method=method,
                    log_likelihoods=log_likelihoods,
                    param_values=param_values,
                    aux_values=aux_values,
                    save=save_llhp,
                )

                make_estimate(
                    event=event,
                    method=method,
                    llhp=llhp,
                    remove_priors=True,
                    priors_used=priors_used,
                    run_info=run_info,
                    fit_meta=fit_meta,
                )

            elif method == "dn8d":
                hypo_handler = init_obj.setup_discrete_hypo(
                    cascade_kernel="scaling_aligned_one_dim",
                    track_kernel="pegleg",
                    track_time_step=1.0,
                )

                prior, priors_used = generate_prior_func(
                    event=event,
                    opt_param_names=hypo_handler.opt_param_names,
                    return_cube=True,
                    **PRISPEC_OSCNEXT_CRS_MN
                )

                param_values = []
                log_likelihoods = []
                aux_values = []
                t_start = []

                loglike = generate_loglike_func(
                    event=event,
                    hypo_handler=hypo_handler,
                    dom_tables=dom_tables,
                    get_llh=get_llh,
                    param_values=param_values,
                    log_likelihoods=log_likelihoods,
                    aux_values=aux_values,
                    t_start=t_start,
                    report_every=report_every,
                )

                run_info, fit_meta = run_dynesty(
                    prior=prior,
                    hypo_handler=hypo_handler,
                    loglike=loglike,
                    n_live=100,
                    maxiter=2000,
                    maxcall=10000,
                    dlogz=0.1,
                )

                llhp = make_llhp(
                    all_param_names=hypo_handler.all_param_names,
                    event=event,
                    method=method,
                    log_likelihoods=log_likelihoods,
                    param_values=param_values,
                    aux_values=aux_values,
                    save=save_llhp,
                )

                make_estimate(
                    event=event,
                    method=method,
                    llhp=llhp,
                    remove_priors=True,
                    priors_used=priors_used,
                    run_info=run_info,
                    fit_meta=fit_meta,
                )

            else:
                raise ValueError("Unknown `Method` {}".format(method))

        except GarbageInputError as error:
            print(
                'ERROR: event idx {}, reco method {}: "{}"; ignoring'
                " and moving to next event".format(
                    event.meta["event_idx"], method, error
                )
            )


def validate_and_translate_methods_arg(methods):
    """Validation and translation for `methods` argument. Missing dependencies
    are also inserted.

    Parameters
    ----------
    methods : str or iterable thereof

    Returns
    -------
    methods : list of str

    """
    if isinstance(methods, string_types):
        methods = [methods]
    methods = list(methods)
    for method in methods:
        if method not in METHOD_DEPENDENCIES.keys():
            raise ValueError(
                'Unrecognized `method` "{}"; must be one of {}'.format(
                    method, METHOD_DEPENDENCIES.keys()
                )
            )
    if len(set(methods)) != len(methods):
        raise ValueError("Same reco(s) specified multiple times")
    insert_method_dependencies(methods)
    return methods


def insert_method_dependencies(methods, recursion_level=0):
    """Insert reconstruction method dependencies if not already specified
    (before the dependent method).

    Note that `methods` is modified in place.

    Parameters
    ----------
    methods : mutable sequence of strings
    recursion_level : int, optional

    Out
    ---
    methods

    """
    recursion_limit = 100
    if recursion_level > recursion_limit:
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
        insert_method_dependencies(methods, recursion_level=recursion_level + 1)


def generate_prior_func(
    event,
    opt_param_names,
    return_cube=False,
    debug=False,
    **kwargs
):
    """Generate the prior transform method `prior` and info `priors_used`
    for a given event. Optionally, plots the priors to current working
    directory if `debug` is True.

    Call, e.g., via:

        prior, priors_used = generate_prior_func(
            event=event,
            x=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT, extents=((-100, Bounds.REL), (100, Bounds.REL))),
            y=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT, extents=((-100, Bounds.REL), (100, Bounds.REL))),
            z=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT, extents=((-50, Bounds.REL), (50, Bounds.REL))),
            time=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT, extents=((-1000, Bounds.REL), (1000, Bounds.REL))),
            azimuth=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT),
            zenith=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT),
        )

    Parameters
    ----------
    event
    opt_param_names
    return_cube : bool
        if true, explicitly return the transformed cube
    **kwargs
        Prior definitions; anything unspecified falls back to a default
        (since all params must have priors, including ranges, for e.g.
        MultiNest and CRS).

    Returns
    -------
    prior : OrderedDict
    priors_used : OrderedDict

    """
    prior_funcs = []
    priors_used = OrderedDict()

    miscellany = []
    for dim_num, dim_name in enumerate(opt_param_names):
        spec = kwargs.get(dim_name, {})
        prior_func, prior_def, misc = get_prior_func(
            dim_num=dim_num, dim_name=dim_name, event=event, **spec
        )
        prior_funcs.append(prior_func)
        priors_used[dim_name] = prior_def
        miscellany.append(misc)

    def prior(cube, ndim=None, nparams=None):  # pylint: disable=unused-argument
        """Apply `prior_funcs` to the hypercube to map values from the unit
        hypercube onto values in the physical parameter space.

        The result overwrites the values in `cube`.

        Parameters
        ----------
        cube
        ndim
        nparams

        """
        for prior_func in prior_funcs:
            prior_func(cube)

        if return_cube:
            return cube

    if debug:
        # -- Plot priors and save to png's in current dir -- #
        import matplotlib as mpl
        mpl.use("agg", warn=False)
        import matplotlib.pyplot as plt

        n_opt_params = len(opt_param_names)
        rand = np.random.RandomState(0)
        cube = rand.rand(n_opt_params, int(1e5))
        prior(cube)

        nx = int(np.ceil(np.sqrt(n_opt_params)))
        ny = int(np.ceil(n_opt_params / nx))
        fig, axes = plt.subplots(ny, nx, figsize=(6 * nx, 4 * ny))
        axit = iter(axes.flat)
        for dim_num, dim_name in enumerate(opt_param_names):
            ax = next(axit)
            ax.hist(cube[dim_num], bins=100)
            misc = miscellany[dim_num]
            if "reco_val" in misc:
                ylim = ax.get_ylim()
                ax.plot([misc["reco_val"]] * 2, ylim, "k--", lw=1)
                ax.set_ylim(ylim)

            misc_strs = []
            if "reco" in misc:
                misc_strs.append(misc["reco"])
            if "reco_val" in misc:
                misc_strs.append("{:.2f}".format(misc["reco_val"]))
            if (
                "split_by_reco_param" in misc
                and misc["split_by_reco_param"] is not None
            ):
                misc_strs.append(
                    "split by {} = {:.2f}".format(
                        misc["split_by_reco_param"], misc["split_val"]
                    )
                )
            misc_str = ", ".join(misc_strs)
            ax.set_title(
                "{}: {} {}".format(
                    dim_name, priors_used[dim_name][0], misc_str
                )
            )
        for ax in axit:
            ax.axis("off")
        fig.tight_layout()
        plt_fpath_base = event.meta["prefix"] + "priors"
        fig.savefig(plt_fpath_base + ".png", dpi=120)

    return prior, priors_used


def generate_loglike_func(
    event,
    hypo_handler,
    dom_tables,
    get_llh,
    param_values,
    log_likelihoods,
    aux_values,
    t_start,
    report_every,
):
    """Generate the LLH callback function `loglike` for a given event.

    Parameters
    ----------
    event
    hypo_handler
    dom_tables
    get_llh : callable
    param_values : list
    log_likelihoods : list
    aux_values : list
    t_start : list
        Needs to be a list for `t_start` to be passed by reference (and
        therefore universally accessible within all methods that require
        knowing `t_start`).

    """
    # -- Variables to be captured by `loglike` closure -- #

    all_param_names = hypo_handler.all_param_names
    opt_param_names = hypo_handler.opt_param_names
    n_opt_params = hypo_handler.n_opt_params
    fixed_params = hypo_handler.fixed_params
    hits = event["hits"]
    hits_indexer = event["hits_indexer"]
    pegleg_muon_dt = hypo_handler.pegleg_kernel_kwargs.get("dt")
    pegleg_muon_const_e_loss = False
    dom_info = dom_tables.dom_info
    sd_idx_table_indexer = dom_tables.sd_idx_table_indexer
    if "truth" in event:
        truth = event["truth"]
        truth_info = OrderedDict(
            [
                ("x", truth["x"]),
                ("y", truth["y"]),
                ("z", truth["z"]),
                ("time", truth["time"]),
                ("zenith", truth["zenith"]),
                ("azimuth", truth["azimuth"]),
                ("track_azimuth", truth["track_azimuth"]),
                ("track_zenith", truth["track_zenith"]),
                ("track_energy", truth["track_energy"]),
                ("energy", truth["energy"]),
                ("cascade_energy", truth['total_cascade_energy']),
            ]
        )
        optional = [
            ("cscd_az", "total_cascade_azimuth"),
            ("cscd_zen", "total_cascade_zenith"),
            ("cscd_em_equiv_en", "total_cascade_em_equiv_energy"),
        ]
        for label, key in optional:
            if key in truth:
                truth_info[label] = truth[key]
    else:
        truth_info = None

    num_operational_doms = np.sum(dom_info["operational"])

    # Array containing only DOMs operational during the event & info
    # relevant to the hits these DOMs got (if any)
    event_dom_info = np.zeros(shape=num_operational_doms, dtype=EVT_DOM_INFO_T)

    # Array containing all relevant hit info for the event, including a
    # pointer back to the index of the DOM in the `event_dom_info` array
    event_hit_info = np.zeros(shape=hits.size, dtype=EVT_HIT_INFO_T)

    # Copy 'time' and 'charge' over directly; add 'event_dom_idx' below
    event_hit_info[["time", "charge"]] = hits[["time", "charge"]]

    # Must be a list, not tuple:
    copy_fields = [
        "sd_idx",
        "x",
        "y",
        "z",
        "quantum_efficiency",
        "noise_rate_per_ns",
    ]

    print("all noise rate %.5f" % np.nansum(dom_info["noise_rate_per_ns"]))
    print(
        "DOMs with zero or NaN noise %i"
        % np.count_nonzero(
            np.isnan(dom_info["noise_rate_per_ns"])
            | (dom_info["noise_rate_per_ns"] == 0)
        )
    )

    # Fill `event_{hit,dom}_info` arrays only for operational DOMs
    for dom_idx, this_dom_info in enumerate(dom_info[dom_info["operational"]]):
        this_event_dom_info = event_dom_info[dom_idx : dom_idx + 1]
        this_event_dom_info[copy_fields] = this_dom_info[copy_fields]
        sd_idx = this_dom_info["sd_idx"]
        this_event_dom_info["table_idx"] = sd_idx_table_indexer[sd_idx]

        # Copy any hit info from `hits_indexer` and total charge from
        # `hits` into `event_hit_info` and `event_dom_info` arrays
        this_hits_indexer = hits_indexer[hits_indexer["sd_idx"] == sd_idx]
        if len(this_hits_indexer) == 0:
            this_event_dom_info["hits_start_idx"] = 0
            this_event_dom_info["hits_stop_idx"] = 0
            this_event_dom_info["total_observed_charge"] = 0
            continue

        start = this_hits_indexer[0]["offset"]
        stop = start + this_hits_indexer[0]["num"]
        event_hit_info[start:stop]["event_dom_idx"] = dom_idx
        this_event_dom_info["hits_start_idx"] = start
        this_event_dom_info["hits_stop_idx"] = stop
        this_event_dom_info["total_observed_charge"] = np.sum(
            hits[start:stop]["charge"]
        )

    print("this evt. noise rate %.5f" % np.sum(event_dom_info["noise_rate_per_ns"]))
    print(
        "DOMs with zero noise: %i"
        % np.sum(event_dom_info["noise_rate_per_ns"] == 0)
    )
    # settings those to minimum noise
    noise = event_dom_info["noise_rate_per_ns"]
    mask = noise < 1e-7
    noise[mask] = 1e-7
    print("this evt. noise rate %.5f" % np.sum(event_dom_info["noise_rate_per_ns"]))
    print(
        "DOMs with zero noise: %i"
        % np.sum(event_dom_info["noise_rate_per_ns"] == 0)
    )
    print("min noise: ", np.min(noise))
    print("mean noise: ", np.mean(noise))

    assert np.sum(event_dom_info["quantum_efficiency"] <= 0) == 0, "negative QE"
    assert np.sum(event_dom_info["total_observed_charge"]) > 0, "no charge"
    assert np.isfinite(
        np.sum(event_dom_info["total_observed_charge"])
    ), "non-finite charge"

    def loglike(cube, ndim=None, nparams=None):  # pylint: disable=unused-argument
        """Get log likelihood values.

        Defined as a closure to capture particulars of the event and priors
        without having to pass these as parameters to the function.

        Note that this is called _after_ `prior` has been called, so `cube`
        already contains the parameter values scaled to be in their
        physical ranges.

        Parameters
        ----------
        cube
        ndim : int, optional
        nparams : int, optional

        Returns
        -------
        llh : float

        """
        t0 = time.time()
        if len(t_start) == 0:
            t_start.append(time.time())

        hypo = OrderedDict(list(zip(opt_param_names, cube)))

        generic_sources = hypo_handler.get_generic_sources(hypo)
        pegleg_sources = hypo_handler.get_pegleg_sources(hypo)
        scaling_sources = hypo_handler.get_scaling_sources(hypo)

        get_llh_retval = get_llh(
            generic_sources=generic_sources,
            pegleg_sources=pegleg_sources,
            scaling_sources=scaling_sources,
            event_hit_info=event_hit_info,
            event_dom_info=event_dom_info,
            pegleg_stepsize=1,
        )

        llh, pegleg_idx, scalefactor = get_llh_retval[:3]
        llh += LLH_FUDGE_SUMMAND
        aux_values.append(get_llh_retval[3:])

        assert np.isfinite(llh), "LLH not finite: {}".format(llh)
        # assert llh <= 0, "LLH positive: {}".format(llh)

        additional_results = []

        if hypo_handler.pegleg_kernel:
            pegleg_result = pegleg_eval(
                pegleg_idx=pegleg_idx,
                dt=pegleg_muon_dt,
                const_e_loss=pegleg_muon_const_e_loss,
                mmc=True,
            )
            additional_results.append(pegleg_result)

        if hypo_handler.scaling_kernel:
            additional_results.append(scalefactor * SCALING_CASCADE_ENERGY)

        result = (
            tuple(cube[:n_opt_params])
            + tuple(fixed_params.values())
            + tuple(additional_results)
        )
        param_values.append(result)

        log_likelihoods.append(llh)
        n_calls = len(log_likelihoods)
        t1 = time.time()

        if n_calls % report_every == 0:
            print("")
            if truth_info:
                msg = "truth:                "
                for key, val in zip(all_param_names, result):
                    try:
                        msg += " %s=%.1f" % (key, truth_info[key])
                    except KeyError:
                        pass
                print(msg)
            t_now = time.time()
            best_idx = np.argmax(log_likelihoods)
            best_llh = log_likelihoods[best_idx]
            best_p = param_values[best_idx]
            msg = "best llh = {:.3f} @ ".format(best_llh)
            for key, val in zip(all_param_names, best_p):
                msg += " %s=%.1f" % (key, val)
            print(msg)
            msg = "this llh = {:.3f} @ ".format(llh)
            for key, val in zip(all_param_names, result):
                msg += " %s=%.1f" % (key, val)
            print(msg)
            print("{} LLH computed".format(n_calls))
            print(
                "avg time per llh: {:.3f} ms".format(
                    (t_now - t_start[0]) / n_calls * 1000
                )
            )
            print("this llh took:    {:.3f} ms".format((t1 - t0) * 1000))
            print("")

        return llh

    return loglike


def make_llhp(
    all_param_names,
    event,
    method,
    log_likelihoods,
    param_values,
    aux_values,
    save,
):
    """Create a structured numpy array containing the reco information;
    also add derived dimensions, and optionally save to disk.

    Parameters
    ----------
    event : OrderedDict

    all_param_names : sequence of str

    method : str

    log_likelihoods : array

    param_values : array

    aux_values : array

    save : bool

    Returns
    -------
    llhp : length-n_llhp array of dtype llhp_t
        Note that llhp_t is derived from the defined parameter names.

    """
    # Setup LLHP dtype
    dim_names = list(all_param_names)

    # add derived quantities
    derived_dim_names = ["energy", "azimuth", "zenith"]
    if "cascade_d_zenith" in dim_names and "cascade_d_azimuth" in dim_names:
        derived_dim_names += ["cascade_zenith", "cascade_azimuth"]

    aux_names = ["zero_dllh", "lower_dllh", "upper_dllh"]

    all_dim_names = dim_names + derived_dim_names + aux_names

    llhp_t = np.dtype([(field, np.float32) for field in ["llh"] + all_dim_names])

    # dump
    llhp = np.zeros(shape=len(param_values), dtype=llhp_t)
    llhp["llh"] = log_likelihoods
    llhp[dim_names] = param_values

    llhp[aux_names] = aux_values

    # create derived dimensions
    if "energy" in derived_dim_names:
        if "track_energy" in dim_names:
            llhp["energy"] += llhp["track_energy"]
        if "cascade_energy" in dim_names:
            llhp["energy"] += llhp["cascade_energy"]

    if "cascade_d_zenith" in dim_names and "cascade_d_azimuth" in dim_names:
        # create cascade angles from delta angles
        rotate_points(
            p_theta=llhp["cascade_d_zenith"],
            p_phi=llhp["cascade_d_azimuth"],
            rot_theta=llhp["track_zenith"],
            rot_phi=llhp["track_azimuth"],
            q_theta=llhp["cascade_zenith"],
            q_phi=llhp["cascade_azimuth"],
        )

    if "track_zenith" in all_dim_names and "track_azimuth" in all_dim_names:
        if "cascade_zenith" in all_dim_names and "cascade_azimuth" in all_dim_names:
            # this resulting radius we won't need, but need to supply an array to
            # the function
            r_out = np.empty(shape=llhp.shape, dtype=np.float32)
            # combine angles:
            add_vectors(
                r1=llhp["track_energy"],
                theta1=llhp["track_zenith"],
                phi1=llhp["track_azimuth"],
                r2=llhp["cascade_energy"],
                theta2=llhp["cascade_zenith"],
                phi2=llhp["cascade_azimuth"],
                r3=r_out,
                theta3=llhp["zenith"],
                phi3=llhp["azimuth"],
            )
        else:
            # in this case there is no cascade angles
            llhp["zenith"] = llhp["track_zenith"]
            llhp["azimuth"] = llhp["track_azimuth"]

    elif "cascade_zenith" in all_dim_names and "cascade_azimuth" in all_dim_names:
        # in this case there are no track angles
        llhp["zenith"] = llhp["cascade_zenith"]
        llhp["azimuth"] = llhp["cascade_azimuth"]

    if save:
        fname = "retro_{}.llhp".format(method)
        # NOTE: since each array can have different length and numpy
        # doesn't handle "ragged" arrays nicely, forcing each llhp to be
        # saved to its own file
        llhp_outf = "{}{}.npy".format(event.meta["prefix"], fname)
        llh = llhp["llh"]
        cut_llhp = llhp[llh > np.max(llh) - 30]
        print(
            'Saving llhp within 30 LLH of max ({} llhp) to "{}"'.format(
                len(cut_llhp), llhp_outf
            )
        )
        np.save(llhp_outf, cut_llhp)

    return llhp


def make_estimate(
    event,
    method,
    llhp,
    remove_priors,
    priors_used=None,
    run_info=None,
    fit_meta=None,
):
    """Create estimate from llhp, attach result to `event`, and save to disk.

    Parameters
    ----------
    event : OrderedDict
    method : str
        Reconstruction method used
    llhp : length-n_llhp array of dtype llhp_t
    remove_priors : bool
        Remove effect of priors
    priors_used : OrderedDict, required if `remove_priors`
    run_info : mapping, optional
    fit_meta : mapping, optional

    Returns
    -------
    estimate : numpy array of struct dtype

    """
    estimate, _ = estimate_from_llhp(
        llhp=llhp,
        treat_dims_independently=False,
        use_prob_weights=True,
        priors_used=priors_used if remove_priors else None,
        meta=fit_meta,
    )

    # Test if the LLH would be positive without LLH_FUDGE_SUMMAND
    if estimate["max_llh"] > LLH_FUDGE_SUMMAND:
        sys.stderr.write(
            "\nWARNING: Would be positive LLH w/o LLH_FUDGE_SUMMAND: {}\n".format(
                estimate["max_llh"]
            )
        )
        if estimate.dtype.names and "fit_status" in estimate.dtype.names:
            if estimate["fit_status"] not in (FitStatus.OK, FitStatus.PositiveLLH):
                raise ValueError(
                    "Postive LLH *and* fit failed with fit_status = {!r}".format(
                        FitStatus(estimate["fit_status"])
                    )
                )
            estimate["fit_status"] = FitStatus.PositiveLLH

    # Place reco in current event in case another reco depends on it
    if "recos" not in event:
        event["recos"] = OrderedDict()
    event["recos"]["retro_" + method] = estimate

    estimate_outf = join(
        event.meta["events_root"],
        "recos",
        "retro_{}.npy".format(method),
    )
    if isfile(estimate_outf):
        estimates = np.load(estimate_outf, mmap_mode="r+")
        try:
            estimates[event.meta["event_idx"]] = estimate
        finally:
            # ensure file handle is not left open
            del estimates
    else:
        estimates = np.full(
            shape=event.meta["num_events"],
            fill_value=np.nan,
            dtype=estimate.dtype,
        )
        # Filling with nan doesn't set correct "fit_status"
        estimates["fit_status"] = FitStatus.NotSet
        estimates[event.meta["event_idx"]] = estimate
        np.save(estimate_outf, estimates)


def main(description=__doc__):
    """Script interface to Reco class and Reco.run(...) method"""
    parser = ArgumentParser(description=description)

    parser.add_argument(
        "--methods",
        required=True,
        choices=sorted(METHOD_DEPENDENCIES.keys()),
        nargs="+",
        help="""Method(s) to use for performing reconstructions; performed in
        order specified, so be sure to specify pre-fits / fits used as seeds
        first""",
    )
    parser.add_argument(
        "--redo-failed",
        action="store_true",
        help="""Whether to re-reconstruct events that have been reconstructed
        but have `fit_status` set to non-zero (i.e., not `FitStatus.OK`), in
        addition to reconstructing events with `fit_status` set to -1 (i.e.,
        `FitStatus.NotSet`)""",
    )
    parser.add_argument(
        "--redo-all",
        action="store_true",
        help="""Whether to reconstruct all events without existing
        reconstructions AND re-reconstruct all events that have existing
        reconstructions, regardless if their `fit_status` is OK or some form of
        failure""",
    )
    parser.add_argument(
        "--report-every",
        metavar="N-LLH-CALLS",
        default=100,
        type=int,
        help="""Report info every N-LLH-CALLS likelihood calls""",
    )
    parser.add_argument(
        "--save-llhp",
        action="store_true",
        help="Whether to save LLHP within 30 LLH of max-LLH to disk",
    )
    split_kw = init_obj.parse_args(
        dom_tables=True, tdi_tables=True, events=True, parser=parser
    )

    events_kw = sort_dict(split_kw.pop("events_kw"))
    dom_tables_kw = sort_dict(split_kw.pop("dom_tables_kw"))
    tdi_tables_kw = sort_dict(split_kw.pop("tdi_tables_kw"))
    reco_kw = sort_dict(split_kw.pop("other_kw"))

    # DEBUG
    print("split_kw:", split_kw)

    # We don't want to specify 'recos' explicitly so that new recos are
    # automatically found by `init_obj.get_events` function as they are created
    # by `reco` function
    events_kw.pop("recos", None)

    # Instantiate objects to use for reconstructions
    events = init_obj.get_events(**events_kw)
    dom_tables = init_obj.setup_dom_tables(**dom_tables_kw)
    tdi_tables, tdi_metas = init_obj.setup_tdi_tables(**tdi_tables_kw)
    _, get_llh, _ = generate_pexp_and_llh_functions(
        dom_tables=dom_tables,
        tdi_tables=tdi_tables,
        tdi_metas=tdi_metas,
    )

    reco(
        events=events,
        dom_tables=dom_tables,
        tdi_tables=tdi_tables,
        get_llh=get_llh,
        **reco_kw
    )


if __name__ == "__main__":
    main()
