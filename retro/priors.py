#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Prior definition generator and prior funcion generator to use for multinest
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    "PRI_UNIFORM",
    "PRI_LOG_UNIFORM",
    "PRI_ZEN_COSINE",
    "PRI_INTERP",
    "PRI_AZ_INTERP",
    "PRI_TIME_RANGE",
    "PRI_OSCNEXT_L5_V1_PREFIT",
    "PRI_OSCNEXT_L5_V1_CRS",
    "OSCNEXT_L5_V1_PRIORS",
    "get_point_estimate",
    "define_prior_from_prefit",
    "define_generic_prior",
    "get_prior_func",
]

__author__ = "J.L. Lanfranchi, P. Eller"
__license__ = """Copyright 2017 Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

from collections import OrderedDict
from copy import deepcopy
from os.path import abspath, basename, dirname, join
import sys

import enum
import numpy as np
from scipy import interpolate, stats
from six import string_types

RETRO_DIR = dirname(dirname(abspath(__file__)))
if __name__ == "__main__" and __package__ is None:
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import MissingOrInvalidPrefitError
from retro.retro_types import FitStatus
from retro.utils.misc import LazyLoader


PRI_UNIFORM = "retro_uniform_prior"
PRI_LOG_UNIFORM = "retro_loguniform_prior"
PRI_ZEN_COSINE = "retro_zen_cosine_prior"
PRI_INTERP = "retro_interpolated_prior"
PRI_AZ_INTERP = "retro_azimuth_interpolated_prior"
PRI_TIME_RANGE = "retro_time_range_prior"

PRI_OSCNEXT_L5_V1_PREFIT = "retro_oscnext_l5_v1_prefit_prior"
"""Priors from L5_SPEFit11 (and fallback to LineFit_DC) fits to oscNext level 5
(first version of processing, or v1) events. See
  retro/notebooks/plot_prior_reco_candidates.ipynb for the fitting process.
"""

PRI_OSCNEXT_L5_V1_CRS = "oscnext_l5_v1_crs"
"""Priors from CRS fits to oscNext level 5 (first version of processing, or v1)
events. See
  retro/notebooks/plot_prior_reco_candidates.ipynb for the fitting process.
"""


class Bound(enum.IntEnum):
    """Specify a boundary is absolute or relative"""

    ABS = 0
    REL = 1


EXT_TIGHT = dict(
    x=((-200, Bound.REL), (200, Bound.REL)),
    y=((-200, Bound.REL), (200, Bound.REL)),
    z=((-100, Bound.REL), (100, Bound.REL)),
    time=((-1000, Bound.REL), (1000, Bound.REL)),
    azimuth=((0, Bound.ABS), (2 * np.pi, Bound.ABS)),
    zenith=((0, Bound.ABS), (np.pi, Bound.ABS)),
)

EXT_MN = dict(
    x=((-300, Bound.REL), (300, Bound.REL)),
    y=((-300, Bound.REL), (300, Bound.REL)),
    z=((-400, Bound.REL), (400, Bound.REL)),
    time=((-1500, Bound.REL), (1500, Bound.REL)),
    azimuth=((0, Bound.ABS), (2 * np.pi, Bound.ABS)),
    zenith=((0, Bound.ABS), (np.pi, Bound.ABS)),
)

EXT_IC = dict(
    x=((-860, Bound.ABS), (870, Bound.ABS)),
    y=((-780, Bound.ABS), (770, Bound.ABS)),
    z=((-780, Bound.ABS), (790, Bound.ABS)),
)

EXT_DC = dict(
    x=((-150, Bound.ABS), (270, Bound.ABS)),
    y=((-210, Bound.ABS), (150, Bound.ABS)),
    z=((-770, Bound.ABS), (760, Bound.ABS)),
)

EXT_DC_SUBDUST = deepcopy(EXT_DC)
EXT_DC_SUBDUST["z"] = ((-610, Bound.ABS), (60, Bound.ABS))


PRISPEC_OSCNEXT_PREFIT_TIGHT = dict(
    x=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT, extents=EXT_TIGHT["x"]),
    y=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT, extents=EXT_TIGHT["y"]),
    z=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT, extents=EXT_TIGHT["z"]),
    time=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT, extents=EXT_TIGHT["time"]),
    azimuth=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT),
    zenith=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT),
)

PRISPEC_OSCNEXT_CRS_MN = dict(
    x=dict(kind=PRI_OSCNEXT_L5_V1_CRS, extents=EXT_MN["x"]),
    y=dict(kind=PRI_OSCNEXT_L5_V1_CRS, extents=EXT_MN["y"]),
    z=dict(kind=PRI_OSCNEXT_L5_V1_CRS, extents=EXT_MN["z"]),
    time=dict(kind=PRI_OSCNEXT_L5_V1_CRS, extents=EXT_MN["time"]),
    #azimuth=dict(kind=PRI_OSCNEXT_L5_V1_CRS),
    #zenith=dict(kind=PRI_OSCNEXT_L5_V1_CRS),
)


OSCNEXT_L5_V1_PRIORS = OrderedDict()
for _dim in ("time", "x", "y", "z", "azimuth", "zenith", "coszen"):
    OSCNEXT_L5_V1_PRIORS[_dim] = OrderedDict()
    for _reco in ("L5_SPEFit11", "LineFit_DC", "retro_crs_prefit"):
        OSCNEXT_L5_V1_PRIORS[_dim][_reco] = LazyLoader(
            datasource=join(
                RETRO_DIR,
                "retro_data",
                "priors",
                "{reco}_{dim}_neg_error.pkl".format(reco=_reco, dim=_dim),
            )
        )


def get_point_estimate(val, estimator, expect_scalar=True):
    """Retrieve a scalar for a reconstructed value.

    Allows for simple scalar recos, or for "estimates from LLH/Params" where a
    number of values are returned using different estimation techniques.

    Parameters
    ----------
    val : scalar, numpy array of struct dtype, or Mapping
    estimator : str
        Not used if `val` is a scalar, otherwise used to get field from numpy
        struct array or item from a Mapping
    expect_scalar : bool

    Returns
    -------
    scalar_val

    """
    # Convert a recarray to a simple array with struct dtype; convert scalar to 0-d array
    valarray = np.array(val)

    # If struct dtype, extract the point estimator
    if valarray.dtype.names:
        valarray = np.array(valarray[estimator])

    if expect_scalar:
        assert valarray.size == 1

        # Since we've forced it to be an array and we don't know the exact
        # dimensionality, use `flat` iterator and extract the first element of the
        # array
        return next(valarray.flat)

    return valarray


def define_prior_from_prefit(
    dim_name, event, priors, candidate_recos, point_estimator, extents=None
):
    """Define a prior from pre-fit(s). Priors are defined by the interpolation
    of KDE'd negative-error distribution for the pre-fits, and "fallback" fits
    can be defined in case one or more fits failed."""
    if isinstance(candidate_recos, string_types):
        candidate_recos = [candidate_recos]

    reco = None
    for candidate_reco in candidate_recos:
        try:
            fit_status = event["recos"][candidate_reco]["fit_status"]
        except (KeyError, ValueError):
            fit_status = FitStatus.OK
        if fit_status == FitStatus.OK:
            reco = candidate_reco
            break

    if reco is None:
        raise MissingOrInvalidPrefitError(
            "Couldn't find a valid prefit reco from among {}".format(candidate_recos)
        )

    # Remove "track_*", etc prefixes
    for prefix in ("track", "cascade"):
        if dim_name.startswith(prefix):
            dim_name = dim_name[len(prefix) :].lstrip("_")
            break

    try:
        reco_val = get_point_estimate(
            event["recos"][reco][dim_name], estimator=point_estimator
        )
    except (KeyError, ValueError):
        if dim_name == "coszen":
            reco_val = np.cos(
                get_point_estimate(
                    event["recos"][reco]["zenith"], estimator=point_estimator
                )
            )
        elif dim_name == "zenith":
            reco_val = np.arccos(
                get_point_estimate(
                    event["recos"][reco]["coszen"], estimator=point_estimator
                )
            )
        else:
            print('No dim "{}" in reco "{}"'.format(dim_name, reco))
            raise

    if not np.isfinite(reco_val):
        raise MissingOrInvalidPrefitError(
            'dim_name "{}", reco "{}": reco val = {}'.format(dim_name, reco, reco_val)
        )

    prior_info = priors[dim_name][reco].data
    prior_sha256 = priors[dim_name][reco].sha256
    prior_fname = basename(priors[dim_name][reco].datasource)

    split_by_reco_param = prior_info["metadata"]["split_by_reco_param"]
    if split_by_reco_param is None:
        split_val = None
    else:
        if split_by_reco_param == "coszen":
            split_val = np.cos(event["recos"][reco]["zenith"])
        else:
            split_val = event["recos"][reco][split_by_reco_param]

        if not np.isfinite(split_val):
            raise MissingOrInvalidPrefitError(
                'Reco "{}", split val "{}" = {}'.format(
                    reco, split_by_reco_param, split_val
                )
            )

    pri = None
    for edges, pri_ in prior_info["dists"].items():
        if split_by_reco_param is None:
            pri = pri_
            break

        if edges[0] <= split_val <= edges[1]:
            pri = pri_
            break

    if pri is None:
        raise ValueError(
            '`split_by_reco_param` "{}" value={} outside binned ranges?: {}'.format(
                split_by_reco_param, split_val, prior_info["dists"].keys()
            )
        )

    xvals = pri["x"] + reco_val

    if extents is None:
        low = np.min(xvals)
        high = np.max(xvals)
    else:
        (low, low_bound_kind), (high, high_bound_kind) = extents
        low = low if low_bound_kind == Bound.ABS else reco_val + low
        high = high if high_bound_kind == Bound.ABS else reco_val + high
        # extra correction for bias in LineFit_DC's z reco
        if (reco, dim_name) == ("LineFit_DC", "z"):
            if low_bound_kind == Bound.REL:
                low -= 15
            if high_bound_kind == Bound.REL:
                high -= 15

    basic_pri_kind = PRI_AZ_INTERP if "azimuth" in dim_name else PRI_INTERP

    prior_def = dict(
        kind=basic_pri_kind,
        params=(reco, reco_val, prior_sha256, xvals, pri["pdf"], low, high),
    )

    metadata = deepcopy(prior_info["metadata"])
    metadata["prior_file_name"] = prior_fname
    metadata["prior_file_sha256"] = prior_sha256[:10]
    metadata["reco_val"] = reco_val
    metadata["split_val"] = split_val

    return prior_def, metadata


def define_generic_prior(kind, extents, kwargs):
    """Create prior definition for a `kind` that exists in `scipy.stats.distributions`.

    Parameters
    ----------
    kind : str
        Must be a continuous distribution in `scipy.stats.distributions`

    extents : sequence of two 2-tuples
        Format should be .. ::

            ((lower_bound, lower_bound_kind), (upper_bound, upper_bound_kind))

        with {lower,upper}_bound scalars and {lower,upper}_bound_kind of type
        `Bound`

    kwargs : Mapping
        Must contain keys for any `shapes` (shape parameters) taken by the
        distribution as well as "loc" and "scale" (which are required for all
        distributions).

    Returns
    -------
    prior_def : tuple
        As defined/used in `retro.priors.get_prior_func`; e.g., formatted as ::

            (kind, (arg0, arg1, ..., argN, low, high))

        where some distributions do not have any args, and thus  will only have
        `low` and `high`

    """
    dist = getattr(stats.distributions, kind)
    (low, low_bound_kind), (high, high_bound_kind) = extents
    if not low_bound_kind == high_bound_kind == Bound.ABS:
        raise ValueError(
            "Only absolute bound (`Bound.ABS`) allowed for"
            " `scipy.stats.distributions` priors (`kind` = {})".format(kind)
        )

    # Parameters to the dist excluding "loc" and "scale" are listed in
    # dist.shapes and must be specified first
    if dist.shapes:
        args = []
        for shape_param in dist.shapes:
            args.append(kwargs[shape_param])
    else:
        args = []

    if "loc" in kwargs:
        args.append(kwargs["loc"])
    if "scale" in kwargs:
        args.append(kwargs["scale"])

    prior_def = (kind, tuple(args + [low, high]))

    return prior_def


def generate_prior_and_pdf_funcs(
    dim_num,
    dim_name,
    kind,
    kind_kwargs,
    extents,
):
    """Generate prior function given a prior definition and the actual event

    Parameters
    ----------
    dim_num : int
        the cube dimension number from multinest

    dim_name : str
        parameter name

    kind : str, optional
        If not provided, uniform-like priors and as-large-as-possible bounds
        are set automatically (e.g., `dim_name="zenith"` yields a cosine prior
        defined in [0, pi] which yields samples uniform in cosine-zenith space)

    center_relative_to : str, scalar, or None; optional
        If a string is provided, `eval` it to retrieve the point at which to
        center the distribution. E.g., .. ::

            center_relative_to="event['L5_SPEFit11']['x'] if event['L5_SPEFit11']['fit_status'] == FitStatus.OK else event['LineFit_DC']['x']"

        If a scalar is provided, that value will be used directly to shift the
        distribution. E.g., .. ::

            center_relative_to=35.2

        If None is passed (the default), the distribution will not be shifted.
        Note that `None` should be passed for prior kinds `PRI_OSCNEXT_*` and
        other KDE-derived pariors since these priors compute how to center
        their distributions from the `event` directly.

    extents : sequence of two 2-tuples, optional
        If not provided, seemingly sensible defaults are defined automatically.
        E.g., if `dim_name="zenith"` and whether or not a prior `kind` is
        defined, if `extents` is None, `extents` are set to [0, pi]).

        If provided, format should be .. ::

            ((lower_bound, lower_bound_kind), (upper_bound, upper_bound_kind))

        with `{lower,upper}_bound` scalar values and
        `{lower,upper}_bound_kind` of type `Bound` (e.g. Bound.ABS or
        Bound.REL). If any bound is relative, `extents_relative_to` must be
        specified.

        For example, .. ::

            ((-25, Bound.REL), (100, Bound.ABS))

    extents_relative_to : str or scalar, or 2-sequence thereof; or None; required if any `extents` are relative
        Specify similarly to `center_relative_to`, but this value or values determine
        which point(s) relative bound(s) are centered about.

    event : event dict or None, required sometimes
        Required if

            * KDE-derived priors are used
            * `center_relative_to` is specified and/or any bound is specified
              as Bound.REL

    kwargs : mapping, required if prior takes additional arguments
        Additional arguments to prior `kind`. E.g., `kind="norm"` requires `loc`
        and `scale`. If any bound is relative or `dist_is_rel=True`,  specify
        an eval-able string for `realative_to` to make the
        `relative_to="event['

    Returns
    -------
    prior_func : callable that takes one scalar in [0, 1]
        Generate samples according to the prior in the range of `extents` by
        passing uniformly-distributed samples in the domain [0, 1] to
        `prior_func`

    prior_pdf_func : ufunc-like callable that takes a scalar or ndarray argument
        Sample the prior function's probability density function at values in
        `extents`. Use this to remove bias due to the prior by weighting
        samples by, e.g., .. ::

            >>> samples = np.array([prior_func(x) for x in np.linspace(0, 1, 1000)])
            >>> probs = prior_pdf_func(samples)
            >>> hist, edges = np.histogram(samples, weights=1/probs)

        where the resulting `hist` should be roughly uniform.

    misc : OrderedDict
        Metadata fully describing the prior for purposes of result provenance
        (i.e., save this information to disk)

    """
    # -- Re-cast priors as something simple to make a func out of -- #

    misc = OrderedDict()

    if hasattr(stats.distributions, kind):
        prior_def = define_generic_prior(kind, extents, kwargs)
    else:
        prior_def = (kind, (low, high))
        raise ValueError(
            'Unhandled or invalid prior "{}" for dim_name "{}"'.format(kind, dim_name)
        )

    # -- Create prior function -- #

    # pylint: disable=unused-argument, missing-docstring

    kind, prior_args = prior_def

    if kind == PRI_UNIFORM:
        low, high = prior_args
        width = high - low

        def prior_func(cube, n=dim_num, width=width, low=low):
            cube[n] = cube[n] * width + low

        def prior_pdf_func(samples, area_norm=width):
            return np.full_like(samples, fill_value=1/area_norm)

    elif kind == PRI_LOG_UNIFORM:
        low, high = prior_args
        log_low = np.log(low)
        log_width = np.log(high) - log_low

        def prior_func(cube, n=dim_num, log_width=log_width, log_low=log_low):
            cube[n] = np.exp(cube[n] * log_width + log_low)

        def prior_pdf_func(samples, area_norm=log_width):
            return 1 / (samples * area_norm)

    elif kind == PRI_ZEN_COSINE:
        zen_low, zen_high = prior_args
        cz_low = np.cos(zen_high)
        cz_high = np.cos(zen_low)
        cz_diff = cz_high - cz_low

        def prior_func(cube, n=dim_num, cz_low=cz_low, cz_diff=cz_diff):
            x = (cz_diff * cube[n]) + cz_low
            cube[n] = np.arccos(x)

        def prior_pdf_func(samples):
            return 0.5 * np.sin(samples)

    elif kind in (PRI_INTERP, PRI_AZ_INTERP):
        x, pdf, low, high = prior_args[-4:]

        if (
            kind == PRI_AZ_INTERP
            and not np.isclose(x.max() - x.min(), high - low)
            or kind == PRI_INTERP
            and (x.min() > low or x.max() < high)
        ):
            print(
                'Dim "{}", prior kind "{}" `x` range = [{}, {}] does not cover'
                " [low, high] range = [{}, {}]".format(
                    dim_name, kind, x.min(), x.max(), low, high
                )
            )

        if kind == PRI_AZ_INTERP:
            if not (np.isclose(low, 0) and np.isclose(high, 2 * np.pi)):
                raise ValueError("az range [low, high) must be [0, 2pi)")

            # Ensure x covers exactly the same distance as (low, high) defines
            highlow_range = high - low
            x = x.min() + (x - x.min()) * highlow_range / (x.max() - x.min())

            # Compute cumulative distribution function (cdf) via trapezoidal-rule
            # integration
            cdf = np.array([np.trapz(x=x[:n], y=pdf[:n]) for n in range(1, len(x) + 1)])
            # Ensure first value in cdf is exactly 0
            cdf -= cdf[0]
            # Ensure last value in cdf is exactly 1
            cdf /= cdf[-1]

            # Create smooth spline interpolator for ppf (inverse of cdf)
            ppf_interp = interpolate.UnivariateSpline(x=cdf, y=x, ext="raise", s=0)

            def prior_func(cube, n=dim_num, ppf_interp=ppf_interp):
                cube[n] = ppf_interp(cube[n]) % (2 * np.pi)

        else:
            # If x covers _more_ than the allowed [low, high] range, resample the
            # pdf in the allowed range (expected to occur for binned zenith and
            # coszen error distributions)
            if x.min() < low or x.max() > high:
                x_orig = x
                pdf_orig = pdf
                x = np.linspace(
                    start=max(low, x_orig.min()),
                    stop=min(high, x_orig.max()),
                    num=len(x_orig),
                ).squeeze()
                pdf = np.interp(x=x, xp=x_orig, fp=pdf_orig)
                integral = np.trapz(x=x, y=pdf)
                try:
                    pdf /= integral
                except:
                    print("low, high:", low, high)
                    print("x_orig.min, x_orig.max:", x_orig.min(), x_orig.max())
                    print("len(x):", len(x))
                    print("x.shape:", x.shape)
                    print("x:", x)
                    print("x_orig.shape:", x_orig.shape)
                    print("x_orig:", x_orig)
                    print("pdf_orig.shape:", pdf_orig.shape)
                    print("pdf.shape:", pdf.shape)
                    print("integral.shape:", integral.shape)
                    raise

            # Compute cumulative distribution function (cdf) via trapezoidal-rule
            # integration
            cdf = np.array([np.trapz(x=x[:n], y=pdf[:n]) for n in range(1, len(x) + 1)])
            # Ensure first value in cdf is exactly 0
            cdf -= cdf[0]
            # Ensure last value in cdf is exactly 1
            cdf /= cdf[-1]

            # Create smooth spline interpolator for ppf (inverse of cdf)
            ppf_interp = interpolate.UnivariateSpline(x=cdf, y=x, ext="raise", s=0)

            def prior_func(cube, n=dim_num, ppf_interp=ppf_interp):
                cube[n] = ppf_interp(cube[n])

            # Create smooth spline interpolator for pdf
            pdf_interp = interpolate.UnivariateSpline(x=x, y=pdf, ext="raise", s=0)

            def prior_pdf_func(samples, pdf_interp=pdf_interp):
                return pdf_interp(samples)

        # Create smooth spline interpolator for pdf
        pdf_interp = interpolate.UnivariateSpline(x=x, y=pdf, ext="raise", s=0)

        def prior_pdf_func(samples, pdf_interp=pdf_interp):
            return pdf_interp(samples)

    elif hasattr(stats.distributions, kind):
        dist_args = prior_args[:-2]
        low, high = prior_args[-2:]
        frozen_dist = getattr(stats.distributions, kind)(*dist_args)

        range_low, r_high = frozen_dist.cdf([low, high])
        range_width = float(np.abs(range_high - range_low))

        def prior_func(
            cube,
            frozen_dist_ppf=frozen_dist.ppf,
            dim_num=dim_num,
            low=low,
            high=high,
            range_low=range_low,
            range_width=range_width,
        ):
            cube[dim_num] = np.clip(
                frozen_dist_ppf(cube[dim_num] * range_width + range_low),
                a_min=low,
                a_max=high,
            )

        def prior_pdf_func(samples, frozen_dist_pdf=frozen_dist.pdf, area_norm=r_width):
            return frozen_dist_pdf(samples) / area_norm

    else:
        raise NotImplementedError('Prior "{}" not implemented.'.format(kind))

    return prior_func, prior_pdf_func, misc
