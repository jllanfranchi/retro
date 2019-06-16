#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Wrapper functions for various optimizers (and simple non-optmization routines)
intended to be called in recipes defined in `retro.reco` module.
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    "CART_DIMS",
    "CRS_STOP_FLAGS",
    "print_non_fatal_exception",
    "get_multinest_meta",
    "run_test",
    "run_with_truth",
    "run_crs",
    "run_scipy",
    "run_skopt",
    "run_nlopt",
    "run_dynesty",
    "run_multinest",
]

from collections import OrderedDict
from os.path import abspath, dirname, isdir, join
from shutil import rmtree
import sys
from tempfile import mkdtemp
import time
import traceback

import numpy as np

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.retro_types import SPHER_T, FitStatus
from retro.utils.get_arg_names import get_arg_names
from retro.utils.geom import (
    fill_from_cart,
    fill_from_spher,
    reflect,
)


CART_DIMS = ("x", "y", "z", "time")

CRS_STOP_FLAGS = {
    0: "max iterations reached",
    1: "llh stddev below threshold",
    2: "no improvement",
    3: "vertex stddev below threshold",
}


def print_non_fatal_exception(event, method):
    """Print to stderr a detailed message about a failure in reconstruction
    that is non-fatal.

    Parameters
    ----------
    method : str
        The name of the function, e.g. "run_crs" or "run_multinest"

    """
    id_fields = ["run_id", "sub_run_id", "event_id", "sub_event_id"]
    id_str = ", ".join(
        "{} {}".format(f, event["header"][f]) for f in id_fields
    )
    sys.stderr.write(
        "ERROR! Reco function {method} failed on event index {idx} ({id_str}) in"
        ' path "{fpath}". Recording reco failure and continuing to next event)'
        "\n{tbk}\n".format(
            method=method,
            idx=event.meta["event_idx"],
            fpath=event.meta["events_root"],
            id_str=id_str,
            tbk="".join(traceback.format_exc()),
        )
    )


def get_multinest_meta(outputfiles_basename):
    """Get metadata from files that MultiNest writes to disk.

    Parameters
    ----------
    outputfiles_basename : str

    Returns
    -------
    fit_meta : OrderedDict
        Contains "logZ", "logZ_err" and, if importance nested sampling was run,
        "ins_logZ" and "ins_logZ_err"

    """
    fit_meta = OrderedDict()
    if isdir(outputfiles_basename):
        stats_fpath = join(outputfiles_basename, "stats.dat")
    else:
        stats_fpath = outputfiles_basename + "stats.dat"

    with open(stats_fpath, "r") as stats_f:
        stats = stats_f.readlines()

    logZ, logZ_err = None, None
    ins_logZ, ins_logZ_err = None, None

    for line in stats:
        if logZ is None and line.startswith("Nested Sampling Global Log-Evidence"):
            logZ, logZ_err = [float(x) for x in line.split(":")[1].split("+/-")]
        elif ins_logZ is None and line.startswith(
            "Nested Importance Sampling Global Log-Evidence"
        ):
            ins_logZ, ins_logZ_err = [float(x) for x in line.split(":")[1].split("+/-")]

    if logZ is not None:
        fit_meta["logZ"] = np.float32(logZ)
        fit_meta["logZ_err"] = np.float32(logZ_err)
    if ins_logZ is not None:
        fit_meta["ins_logZ"] = np.float32(ins_logZ)
        fit_meta["ins_logZ_err"] = np.float32(ins_logZ_err)

    return fit_meta


def run_test(prior, hypo_handler, loglike, seed):
    """Random sampling instead of an actual minimizer"""
    raise NotImplementedError("`run_test` not implemented")  # TODO
    t0 = time.time()

    kwargs = OrderedDict()
    for arg_name in get_arg_names(run_test):
        if arg_name in ("event", "prior", "hypo_handler", "loglike", "report_every"):
            continue
        kwargs[arg_name] = locals()[arg_name]

    rand = np.random.RandomState(seed=seed)
    for i in range(100):
        param_vals = rand.uniform(0, 1, hypo_handler.n_opt_params)
        prior(param_vals)
        llh = loglike(param_vals)
    run_info = OrderedDict([("method", "run_test"), ("kwargs", kwargs)])
    fit_meta = OrderedDict(
        [
            ("fit_status", np.int8(FitStatus.OK)),
            ("run_time", np.float32(time.time() - t0)),
        ]
    )
    return run_info, fit_meta


def run_with_truth(event, prior, hypo_handler, loglike, rand_dims=None, n_samples=10000, seed=0):
    """Run with all params set to truth except for the dimensions defined,
    which will be randomized.

    Parameters
    ----------
    event : OrderedDict
    prior : callable
    rand_dims : list, optional
        Dimensions to randomly sample; all not specified are set to truth

    n_samples : int
        Number of samples to draw

    """
    raise NotImplementedError("`run_with_truth` not implemented")  # TODO
    t0 = time.time()

    if rand_dims is None:
        rand_dims = []

    kwargs = OrderedDict()
    for arg_name in get_arg_names(run_with_truth):
        if arg_name in ("event", "prior", "hypo_handler", "loglike", "report_every"):
            continue
        kwargs[arg_name] = locals()[arg_name]

    truth = event["truth"]
    true_params = np.zeros(hypo_handler.n_opt_params)

    for i, name in enumerate(hypo_handler.opt_param_names):
        name = name.replace("cascade_", "total_cascade_")
        true_params[i] = truth[name]

    rand = np.random.RandomState(seed=seed)
    if len(rand_dims) > 1:
        for i in range(n_samples):
            rand_params = rand.uniform(0, 1, hypo_handler.n_opt_params)
            prior(rand_params)
            param_vals = np.zeros(hypo_handler.n_opt_params)
            param_vals[:] = true_params[:]
            param_vals[rand_dims] = rand_params[rand_dims]
            llh = loglike(param_vals)
    else:
        llh = loglike(true_params)

    run_info = OrderedDict([("method", "run_with_truth"), ("kwargs", kwargs)])
    fit_meta = OrderedDict(
        [
            ("fit_status", np.int8(FitStatus.OK)),
            ("run_time", np.float32(time.time() - t0)),
        ]
    )

    return run_info, fit_meta


def run_crs(
    event,
    prior,
    hypo_handler,
    loglike,
    n_live,
    max_iter,
    max_noimprovement,
    min_llh_std,
    min_vertex_std,
    use_priors,
    use_sobol,
    seed,
    report_every,
):
    """Implementation of the CRS2 algorithm, adapted to work with spherical
    coordinates (correct centroid calculation, reflection, and mutation).

    At the moment Cartesian (standard) parameters and spherical parameters
    are assumed to have particular names (i.e., spherical coordinates start
    with "az" and "zen"). Furthermore, all Cartesian coordinates must come
    first followed by the pairs of (azimuth, zenith) spherical coordinates;
    e.g., "az_1", "zen_1", "az_2", "zen_2", etc.

    Parameters
    ----------
    event : OrderedDict
    prior : callable
    hypo_handler
    n_live : int
        Number of live points
    max_iter : int
        Maximum iterations
    max_noimprovement : int
        Maximum iterations with no improvement of best point
    min_llh_std : float
        Break if stddev of llh values across all livepoints drops below
        this threshold
    min_vertex_std : mapping
        Break condition on stddev of Cartesian dimension(s) (x, y, z, and
        time). Keys are dimension names and values are the standard
        deviations for each dimension. All specified dimensions must drop
        below the specified stddevs for this break condition to be met.
    use_priors : bool
        Use priors during minimization; if `False`, priors are only used
        for sampling the initial distributions. Even if set to `True`,
        angles (azimuth and zenith) do not use priors while operating (only
        for generating the initial distribution)
    use_sobol : bool
        Use a Sobol sequence instead of numpy pseudo-random numbers. Seems
        to do slightly better (but only small differences observed in tests
        so far)
    seed : int
        Random seed
    report_every : int

    Returns
    -------
    run_info : OrderedDict

    Notes
    -----
    CRS2 [1] is a variant of controlled random search (CRS, a global
    optimizer) with faster convergence than CRS.

    Refrences
    ---------
    .. [1] P. Kaelo, M.M. Ali, "Some variants of the controlled random
       search algorithm for global optimization," J. Optim. Theory Appl.,
       130 (2) (2006), pp. 253-264.

    """
    t0 = time.time()

    if use_sobol:
        from sobol import i4_sobol

    rand = np.random.RandomState(seed=seed)

    # Record kwargs user supplied (after translation & standardization)
    kwargs = OrderedDict()
    for arg_name in get_arg_names(run_crs):
        if arg_name in ("event", "prior", "hypo_handler", "loglike", "report_every"):
            continue
        kwargs[arg_name] = locals()[arg_name]

    run_info = OrderedDict(
        [
            ("method", "run_crs"),
            ("method_description", "CRS2spherical+lm+sampling"),
            ("kwargs", kwargs),
        ]
    )

    n_opt_params = hypo_handler.n_opt_params
    # absolute minimum number of points necessary
    assert n_live > n_opt_params + 1

    # figure out which variables are Cartesian and which spherical
    opt_param_names = hypo_handler.opt_param_names
    cart_param_names = set(opt_param_names) & set(CART_DIMS)
    n_cart = len(cart_param_names)
    assert set(opt_param_names[:n_cart]) == cart_param_names
    n_spher_param_pairs = int((n_opt_params - n_cart) / 2)
    for sph_pair_idx in range(n_spher_param_pairs):
        az_param = opt_param_names[n_cart + sph_pair_idx * 2]
        zen_param = opt_param_names[n_cart + sph_pair_idx * 2 + 1]
        assert "az" in az_param, '"{}" not azimuth param'.format(az_param)
        assert "zen" in zen_param, '"{}" not zenith param'.format(zen_param)

    for dim in min_vertex_std.keys():
        if dim not in opt_param_names:
            raise ValueError('dim "{}" not being optimized'.format(dim))
        if dim not in cart_param_names:
            raise NotImplementedError(
                'dim "{}" stddev not computed, as stddev currently only'
                " computed for Cartesian parameters".format(dim)
            )

    # set standard reordering so subsequent calls with different input
    # ordering will create identical metadata
    min_vertex_std = OrderedDict(
        [(d, min_vertex_std[d]) for d in opt_param_names if d in min_vertex_std]
    )

    # storage for info about stddev, whether met, and when met; defaults
    # should indicate failure if not explicitly set elsewhere
    vertex_std = np.full(
        shape=1,
        fill_value=np.nan,
        dtype=[(d, np.float32) for d in min_vertex_std.keys()],
    )
    vertex_std_met = OrderedDict([(d, False) for d in min_vertex_std.keys()])
    vertex_std_met_at_iter = np.full(
        shape=1, fill_value=-1, dtype=[(d, np.int32) for d in min_vertex_std.keys()]
    )

    # default values (in case of failure and these don't get set elsewhere,
    # then these values will be returned)
    fit_status = FitStatus.GeneralFailure
    iter_num = 0
    stopping_flag = 0
    llh_std = np.nan
    no_improvement_counter = 0
    num_simplex_successes = 0
    num_mutation_successes = 0
    num_failures = 0

    # setup arrays to store points
    s_cart = np.zeros(shape=(n_live, n_cart))
    s_spher = np.zeros(shape=(n_live, n_spher_param_pairs), dtype=SPHER_T)
    llh = np.zeros(shape=(n_live,))

    def func(x):
        """Callable for minimizer"""
        if use_priors:
            param_vals = np.zeros_like(x)
            param_vals[:n_cart] = x[:n_cart]
            prior(param_vals)
            param_vals[n_cart:] = x[n_cart:]
        else:
            param_vals = x
        llh = loglike(param_vals)
        if np.isnan(llh):
            raise ValueError("llh is nan; params are {}".format(param_vals))
        if np.any(np.isnan(param_vals)):
            raise ValueError("params are nan: {}".format(param_vals))
        return -llh

    def create_x(x_cart, x_spher):
        """Patch Cartesian and spherical coordinates into one array"""
        # TODO: make proper
        x = np.empty(shape=n_opt_params)
        x[:n_cart] = x_cart
        x[n_cart + 1 :: 2] = x_spher["zen"]
        x[n_cart::2] = x_spher["az"]
        return x

    try:
        # generate initial population
        for i in range(n_live):
            # Sobol seems to do slightly better than pseudo-random numbers
            if use_sobol:
                # Note we start at seed=1 since for n_live=1 this puts the
                # first point in the middle of the range for all params (0.5),
                # while seed=0 produces all zeros (the most extreme point
                # possible, which will bias the distribution away from more
                # likely values).
                x, _ = i4_sobol(
                    dim_num=n_opt_params,  # number of dimensions
                    seed=i + 1,  # Sobol sequence number
                )
            else:
                x = rand.uniform(0, 1, n_opt_params)

            # Apply prior xforms to `param_vals` (contents are overwritten)
            param_vals = np.copy(x)
            prior(param_vals)

            # Always use prior-xformed angles
            x[n_cart:] = param_vals[n_cart:]

            # Only use xformed Cart params if NOT using priors during operation
            if not use_priors:
                x[:n_cart] = param_vals[:n_cart]

            # Break up into Cartesian and spherical coordinates
            s_cart[i] = x[:n_cart]
            s_spher[i]["zen"] = x[n_cart + 1 :: 2]
            s_spher[i]["az"] = x[n_cart::2]
            fill_from_spher(s_spher[i])
            llh[i] = func(x)

        best_llh = np.min(llh)
        no_improvement_counter = -1

        # optional bookkeeping
        num_simplex_successes = 0
        num_mutation_successes = 0
        num_failures = 0
        stopping_flag = 0

        # minimizer loop
        for iter_num in range(max_iter):
            if iter_num % report_every == 0:
                print(
                    "simplex: %i, mutation: %i, failed: %i"
                    % (num_simplex_successes, num_mutation_successes, num_failures)
                )

            # compute value for break condition 1
            llh_std = np.std(llh)

            # compute value for break condition 3
            for dim, cond in min_vertex_std.items():
                vertex_std[dim] = std = np.std(
                    s_cart[:, opt_param_names.index(dim)]
                )
                vertex_std_met[dim] = met = std < cond
                if met:
                    if vertex_std_met_at_iter[dim] == -1:
                        vertex_std_met_at_iter[dim] = iter_num
                else:
                    vertex_std_met_at_iter[dim] = -1

            # break condition 1
            if llh_std < min_llh_std:
                stopping_flag = 1
                break

            # break condition 2
            if no_improvement_counter > max_noimprovement:
                stopping_flag = 2
                break

            # break condition 3
            if len(min_vertex_std) > 0 and all(vertex_std_met.values()):
                stopping_flag = 3
                break

            new_best_llh = np.min(llh)

            if new_best_llh < best_llh:
                best_llh = new_best_llh
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            worst_idx = np.argmax(llh)
            best_idx = np.argmin(llh)

            # choose n_opt_params random points but not best
            choice = rand.choice(n_live - 1, n_opt_params, replace=False)
            choice[choice >= best_idx] += 1

            # Cartesian centroid
            centroid_cart = (
                np.sum(s_cart[choice[:-1]], axis=0) + s_cart[best_idx]
            ) / n_opt_params

            # reflect point
            new_x_cart = 2 * centroid_cart - s_cart[choice[-1]]

            # spherical centroid
            centroid_spher = np.zeros(n_spher_param_pairs, dtype=SPHER_T)
            centroid_spher["x"] = (
                np.sum(s_spher["x"][choice[:-1]], axis=0) + s_spher["x"][best_idx]
            ) / n_opt_params
            centroid_spher["y"] = (
                np.sum(s_spher["y"][choice[:-1]], axis=0) + s_spher["y"][best_idx]
            ) / n_opt_params
            centroid_spher["z"] = (
                np.sum(s_spher["z"][choice[:-1]], axis=0) + s_spher["z"][best_idx]
            ) / n_opt_params
            fill_from_cart(centroid_spher)

            # reflect point
            new_x_spher = np.zeros(n_spher_param_pairs, dtype=SPHER_T)
            reflect(s_spher[choice[-1]], centroid_spher, new_x_spher)

            if use_priors:
                outside = np.any(new_x_cart < 0) or np.any(new_x_cart > 1)
            else:
                outside = False

            if not outside:
                new_llh = func(create_x(new_x_cart, new_x_spher))

                if new_llh < llh[worst_idx]:
                    # found better point
                    s_cart[worst_idx] = new_x_cart
                    s_spher[worst_idx] = new_x_spher
                    llh[worst_idx] = new_llh
                    num_simplex_successes += 1
                    continue

            # mutation
            w = rand.uniform(0, 1, n_cart)
            new_x_cart2 = (1 + w) * s_cart[best_idx] - w * new_x_cart

            # first reflect at best point
            reflected_new_x_spher = np.zeros(n_spher_param_pairs, dtype=SPHER_T)
            reflect(new_x_spher, s_spher[best_idx], reflected_new_x_spher)

            new_x_spher2 = np.zeros_like(new_x_spher)

            # now do a combination of best and reflected point with weight w
            for dim in ("x", "y", "z"):
                w = rand.uniform(0, 1, n_spher_param_pairs)
                new_x_spher2[dim] = (1 - w) * s_spher[best_idx][
                    dim
                ] + w * reflected_new_x_spher[dim]
            fill_from_cart(new_x_spher2)

            if use_priors:
                outside = np.any(new_x_cart2 < 0) or np.any(new_x_cart2 > 1)
            else:
                outside = False

            if not outside:
                new_llh = func(create_x(new_x_cart2, new_x_spher2))

                if new_llh < llh[worst_idx]:
                    # found better point
                    s_cart[worst_idx] = new_x_cart2
                    s_spher[worst_idx] = new_x_spher2
                    llh[worst_idx] = new_llh
                    num_mutation_successes += 1
                    continue

            # if we get here no method was successful in replacing worst
            # point -> start over
            num_failures += 1

        print(CRS_STOP_FLAGS[stopping_flag])
        fit_status = FitStatus.OK

    except KeyboardInterrupt:
        raise

    except Exception:
        print_non_fatal_exception(event=event, method=run_info["method"])

    fit_meta = OrderedDict(
        [
            ("fit_status", np.int8(fit_status)),
            ("iterations", np.uint32(iter_num)),
            ("stopping_flag", np.int8(stopping_flag)),
            ("llh_std", np.float32(llh_std)),
            ("no_improvement_counter", np.uint32(no_improvement_counter)),
            ("vertex_std", vertex_std),  # already typed
            ("vertex_std_met_at_iter", vertex_std_met_at_iter),  # already typed
            ("num_simplex_successes", np.uint32(num_simplex_successes)),
            ("num_mutation_successes", np.uint32(num_mutation_successes)),
            ("num_failures", np.uint32(num_failures)),
            ("run_time", np.float32(time.time() - t0)),
        ]
    )

    return run_info, fit_meta


def run_scipy(event, prior, hypo_handler, loglike, method, eps):
    """Use an optimizer from scipy"""
    t0 = time.time()

    from scipy import optimize

    kwargs = OrderedDict()
    for arg_name in get_arg_names(run_scipy):
        if arg_name in ("event", "prior", "hypo_handler", "loglike", "report_every"):
            continue
        kwargs[arg_name] = locals()[arg_name]

    run_info = OrderedDict([("method", "run_scipy"), ("kwargs", kwargs)])

    # initial guess
    x0 = 0.5 * np.ones(shape=hypo_handler.n_opt_params)

    def func(x, *args):  # pylint: disable=unused-argument, missing-docstring
        param_vals = np.copy(x)
        prior(param_vals)
        llh = loglike(param_vals)
        del param_vals
        return -llh

    bounds = [(eps, 1 - eps)] * hypo_handler.n_opt_params
    settings = OrderedDict()
    settings["eps"] = eps

    fit_status = FitStatus.GeneralFailure
    try:
        if method == "differential_evolution":
            optimize.differential_evolution(func, bounds=bounds, popsize=100)
        else:
            optimize.minimize(
                func, x0, method=method, bounds=bounds, options=settings
            )
        fit_status = FitStatus.OK

    except KeyboardInterrupt:
        raise

    except Exception:
        print_non_fatal_exception(event=event, method=run_info["method"])

    fit_meta = OrderedDict(
        [
            ("fit_status", np.int8(fit_status)),
            ("run_time", np.float32(time.time() - t0)),
        ]
    )

    return run_info, fit_meta


def run_skopt(event, prior, hypo_handler, loglike):
    """Use an optimizer from scikit-optimize"""
    t0 = time.time()

    from skopt import gp_minimize  # , forest_minimize

    settings = OrderedDict(
        [
            ("acq_func", "EI"),  # acquisition function
            ("n_calls", 1000),  # number of evaluations of f
            ("n_random_starts", 5),  # number of random initialization
        ]
    )
    run_info = OrderedDict([("method", "run_skopt"), ("settings", settings)])

    # initial guess
    x0 = 0.5 * np.ones(shape=hypo_handler.n_opt_params)

    def func(x, *args):  # pylint: disable=unused-argument, missing-docstring
        param_vals = np.copy(x)
        prior(param_vals)
        llh = loglike(param_vals)
        del param_vals
        return -llh

    bounds = [(0, 1)] * hypo_handler.n_opt_params

    fit_status = FitStatus.GeneralFailure
    try:
        _ = gp_minimize(
            func,  # function to minimize
            bounds,  # bounds on each dimension of x
            x0=list(x0),
            **settings
        )
        fit_status = FitStatus.OK

    except KeyboardInterrupt:
        raise

    except Exception:
        print_non_fatal_exception(event=event, method=run_info["method"])

    fit_meta = OrderedDict(
        [
            ("fit_status", np.int8(fit_status)),
            ("run_time", np.float32(time.time() - t0)),
        ]
    )

    return run_info, fit_meta


def run_nlopt(event, prior, hypo_handler, loglike):
    """Use an optimizer from nlopt"""
    t0 = time.time()

    import nlopt

    def func(x, grad):  # pylint: disable=unused-argument, missing-docstring
        param_vals = np.copy(x)
        prior(param_vals)
        llh = loglike(param_vals)
        del param_vals
        return -llh

    # bounds
    lower_bounds = np.zeros(shape=hypo_handler.n_opt_params)
    upper_bounds = np.ones(shape=hypo_handler.n_opt_params)

    # for angles make bigger
    for i, name in enumerate(hypo_handler.opt_param_names):
        if "azimuth" in name:
            lower_bounds[i] = -0.5
            upper_bounds[i] = 1.5
        if "zenith" in name:
            lower_bounds[i] = -0.5
            upper_bounds[i] = 1.5

    # initial guess
    x0 = 0.5 * np.ones(shape=hypo_handler.n_opt_params)

    # stepsize
    dx = np.zeros(shape=hypo_handler.n_opt_params)
    for i in range(hypo_handler.n_opt_params):
        if "azimuth" in hypo_handler.opt_param_names[i]:
            dx[i] = 0.001
        elif "zenith" in hypo_handler.opt_param_names[i]:
            dx[i] = 0.001
        elif hypo_handler.opt_param_names[i] in ("x", "y"):
            dx[i] = 0.005
        elif hypo_handler.opt_param_names[i] == "z":
            dx[i] = 0.002
        elif hypo_handler.opt_param_names[i] == "time":
            dx[i] = 0.01

    # seed from several angles
    # opt = nlopt.opt(nlopt.LN_NELDERMEAD, hypo_handler.n_opt_params)
    opt = nlopt.opt(nlopt.GN_CRS2_LM, hypo_handler.n_opt_params)
    ftol_abs = 0.1
    # opt = nlopt.opt(nlopt.LN_PRAXIS, hypo_handler.n_opt_params)
    opt.set_lower_bounds([0.0] * hypo_handler.n_opt_params)
    opt.set_upper_bounds([1.0] * hypo_handler.n_opt_params)
    opt.set_min_objective(func)
    opt.set_ftol_abs(ftol_abs)

    settings = OrderedDict(
        [("method", opt.get_algorithm_name()), ("ftol_abs", np.float32(ftol_abs))]
    )

    run_info = OrderedDict([("method", "run_nlopt"), ("settings", settings)])

    fit_status = FitStatus.GeneralFailure
    try:
        # initial guess

        angles = np.linspace(0, 1, 3)
        angles = 0.5 * (angles[1:] + angles[:-1])

        for zen in angles:
            for az in angles:
                x0 = 0.5 * np.ones(shape=hypo_handler.n_opt_params)

                for i in range(hypo_handler.n_opt_params):
                    if "az" in hypo_handler.opt_param_names[i]:
                        x0[i] = az
                    elif "zen" in hypo_handler.opt_param_names[i]:
                        x0[i] = zen
                x = opt.optimize(x0)  # pylint: disable=unused-variable

        # local_opt = nlopt.opt(nlopt.LN_NELDERMEAD, hypo_handler.n_opt_params)
        # local_opt.set_lower_bounds([0.]*hypo_handler.n_opt_params)
        # local_opt.set_upper_bounds([1.]*hypo_handler.n_opt_params)
        # local_opt.set_min_objective(func)
        ##local_opt.set_ftol_abs(0.5)
        ##local_opt.set_ftol_abs(100)
        ##local_opt.set_xtol_rel(10)
        # local_opt.set_ftol_abs(1)
        # global
        # opt = nlopt.opt(nlopt.G_MLSL, hypo_handler.n_opt_params)
        # opt.set_lower_bounds([0.]*hypo_handler.n_opt_params)
        # opt.set_upper_bounds([1.]*hypo_handler.n_opt_params)
        # opt.set_min_objective(func)
        # opt.set_local_optimizer(local_opt)
        # opt.set_ftol_abs(10)
        # opt.set_xtol_rel(1)
        # opt.set_maxeval(1111)

        # opt = nlopt.opt(nlopt.GN_ESCH, hypo_handler.n_opt_params)
        # opt = nlopt.opt(nlopt.GN_ISRES, hypo_handler.n_opt_params)
        # opt = nlopt.opt(nlopt.GN_CRS2_LM, hypo_handler.n_opt_params)
        # opt = nlopt.opt(nlopt.GN_DIRECT_L_RAND_NOSCAL, hypo_handler.n_opt_params)
        # opt = nlopt.opt(nlopt.LN_NELDERMEAD, hypo_handler.n_opt_params)

        # opt.set_lower_bounds(lower_bounds)
        # opt.set_upper_bounds(upper_bounds)
        # opt.set_min_objective(func)
        # opt.set_ftol_abs(0.1)
        # opt.set_population([x0])
        # opt.set_initial_step(dx)

        # local_opt.set_maxeval(10)

        # x = opt.optimize(x0) # pylint: disable=unused-variable

        # polish it up
        # print('***************** polishing ******************')

        # dx = np.ones(shape=hypo_handler.n_opt_params) * 0.001
        # dx[0] = 0.1
        # dx[1] = 0.1

        # local_opt = nlopt.opt(nlopt.LN_NELDERMEAD, hypo_handler.n_opt_params)
        # lower_bounds = np.clip(np.copy(x) - 0.1, 0, 1)
        # upper_bounds = np.clip(np.copy(x) + 0.1, 0, 1)
        # lower_bounds[0] = 0
        # lower_bounds[1] = 0
        # upper_bounds[0] = 0
        # upper_bounds[1] = 0

        # local_opt.set_lower_bounds(lower_bounds)
        # local_opt.set_upper_bounds(upper_bounds)
        # local_opt.set_min_objective(func)
        # local_opt.set_ftol_abs(0.1)
        # local_opt.set_initial_step(dx)
        # x = opt.optimize(x)

        fit_status = FitStatus.OK

    except KeyboardInterrupt:
        raise

    except Exception:
        print_non_fatal_exception(event=event, method=run_info["method"])

    fit_meta = OrderedDict(
        [
            ("fit_status", np.int8(fit_status)),
            ("run_time", np.float32(time.time() - t0)),
            ("ftol_abs", np.float32(opt.get_ftol_abs())),
            ("ftol_rel", np.float32(opt.get_ftol_rel())),
            ("xtol_abs", np.float32(opt.get_xtol_abs())),
            ("xtol_rel", np.float32(opt.get_xtol_rel())),
            ("maxeval", np.float32(opt.get_maxeval())),
            ("maxtime", np.float32(opt.get_maxtime())),
            ("stopval", np.float32(opt.get_stopval())),
        ]
    )

    return run_info, fit_meta


def run_dynesty(
    prior,
    hypo_handler,
    loglike,
    n_live,
    maxiter,
    maxcall,
    dlogz
):
    """Setup and run Dynesty on an event.

    Parameters
    ----------

    Returns
    -------
    run_info : OrderedDict
        Metadata dict containing dynesty settings used and extra info returned by
        dynesty

    fit_meta : OrderedDict

    """
    import dynesty

    t0 = time.time()

    kwargs = OrderedDict()
    for arg_name in get_arg_names(run_dynesty):
        if arg_name in ("event", "prior", "hypo_handler", "loglike", "report_every"):
            continue
        kwargs[arg_name] = locals()[arg_name]


    dn_kwargs = OrderedDict(
        [
            ("ndim", hypo_handler.n_opt_params),
            ('nlive', n_live),
            (
                "periodic",
                [i for i, p in enumerate(hypo_handler.all_param_names) if 'az' in p.lower()],
            ),
        ]
    )

    sampler_kwargs = OrderedDict(
        [
            ('maxiter', maxiter),
            ('maxcall', maxcall),
            ('dlogz', dlogz),
        ]
    )

    run_info = OrderedDict(
        [
            ("method", "run_dynesty"),
            ("kwargs", kwargs),
            ("dn_kwargs", dn_kwargs),
            ("sampler_kwargs", sampler_kwargs),
        ]
    )

    fit_meta = OrderedDict()
    fit_meta["fit_status"] = np.int8(FitStatus.NotSet)
    sampler = dynesty.NestedSampler(
        loglikelihood=loglike,
        prior_transform=prior,
        method='unif',
        bound='single',
        update_interval=1,
        **dn_kwargs
    )
    print('sampler instantiated')
    sampler.run_nested(**sampler_kwargs)

    fit_meta["fit_status"] = np.int8(FitStatus.OK)
    fit_meta["run_time"] = np.float32(time.time() - t0)

    print(fit_meta)

    return run_info, fit_meta


def run_multinest(
    event,
    prior,
    hypo_handler,
    loglike,
    importance_sampling,
    max_modes,
    const_eff,
    n_live,
    evidence_tol,
    sampling_eff,
    max_iter,
    seed,
    report_every,
):
    """Setup and run MultiNest on an event.

    See the README file from MultiNest for greater detail on parameters
    specific to to MultiNest (parameters from `importance_sampling` on).

    Parameters
    ----------
    importance_sampling
    max_modes
    const_eff
    n_live
    evidence_tol
    sampling_eff
    max_iter
        Note that this limit is the maximum number of sample replacements
        and _not_ max number of likelihoods evaluated. A replacement only
        occurs when a likelihood is found that exceeds the minimum
        likelihood among the live points.
    seed
    report_every : int >= 1

    Returns
    -------
    run_info : OrderedDict
        Metadata dict containing MultiNest settings used and extra info returned by
        MultiNest

    fit_meta : OrderedDict

    """
    t0 = time.time()

    # Import pymultinest here; it's a less common dependency, so other
    # functions/constants in this module will still be import-able w/o it.
    import pymultinest

    kwargs = OrderedDict()
    for arg_name in get_arg_names(run_multinest):
        if arg_name in ("event", "prior", "hypo_handler", "loglike", "report_every"):
            continue
        kwargs[arg_name] = locals()[arg_name]

    mn_kwargs = OrderedDict(
        [
            ("n_dims", hypo_handler.n_opt_params),
            ("n_params", hypo_handler.n_params),
            ("n_clustering_params", hypo_handler.n_opt_params),
            (
                "wrapped_params",
                ["az" in p.lower() for p in hypo_handler.all_param_names],
            ),
            ("importance_nested_sampling", importance_sampling),
            ("multimodal", max_modes > 1),
            ("const_efficiency_mode", const_eff),
            ("n_live_points", n_live),
            ("evidence_tolerance", evidence_tol),
            ("sampling_efficiency", sampling_eff),
            ("null_log_evidence", -1e90),
            ("max_modes", max_modes),
            ("mode_tolerance", -1e90),
            ("seed", seed),
            ("log_zero", -1e100),
            ("max_iter", max_iter),
        ]
    )

    run_info = OrderedDict(
        [("method", "run_multinest"), ("kwargs", kwargs), ("mn_kwargs", mn_kwargs)]
    )

    fit_status = FitStatus.GeneralFailure
    tmpdir = mkdtemp()
    outputfiles_basename = join(tmpdir, "")
    mn_fit_meta = {}
    try:
        pymultinest.run(
            LogLikelihood=loglike,
            Prior=prior,
            verbose=True,
            outputfiles_basename=outputfiles_basename,
            resume=False,
            write_output=True,
            n_iter_before_update=report_every,
            **mn_kwargs
        )
        fit_status = FitStatus.OK
        mn_fit_meta = get_multinest_meta(outputfiles_basename=outputfiles_basename)

    except KeyboardInterrupt:
        raise

    except Exception:
        print_non_fatal_exception(event=event, method=run_info["method"])

    finally:
        rmtree(tmpdir)

    # TODO: If MultiNest fails in specific ways, set fit_status accordingly...

    fit_meta = OrderedDict(
        [
            ("fit_status", np.int8(fit_status)),
            ("logZ", np.float32(mn_fit_meta.pop("logZ", np.nan))),
            ("logZ_err", np.float32(mn_fit_meta.pop("logZ_err", np.nan))),
            ("ins_logZ", np.float32(mn_fit_meta.pop("ins_logZ", np.nan))),
            ("ins_logZ_err", np.float32(mn_fit_meta.pop("ins_logZ_err", np.nan))),
            ("run_time", np.float32(time.time() - t0)),
        ]
    )

    if mn_fit_meta:
        sys.stderr.write(
            "WARNING: Unrecorded MultiNest metadata: {}\n".format(
                ", ".join("{} = {}".format(k, v) for k, v in mn_fit_meta.items())
            )
        )

    return run_info, fit_meta
