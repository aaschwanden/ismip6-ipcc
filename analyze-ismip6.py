#!/usr/bin/env python

# Copyright (C) 2019-20 Andy Aschwanden

from netCDF4 import Dataset as NC
import numpy as np
import os
import pylab as plt
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from glob import glob
from pathlib import Path


def trend_estimator(x, y):
    """
    Trend estimator

    Simultaneous estimation of bias, trend, annual, semi-annual and
    161-day sinusoid (alias period S2 tide errors).

    Parameters
    ----------
    x, y : array_like, x must have units "years"

    Returns
    -------
    x : ndarray
    The solution (or the result of the last iteration for an unsuccessful
    call).
    cov_x : ndarray
    Uses the fjac and ipvt optional outputs to construct an
    estimate of the jacobian around the solution.  ``None`` if a
    singular matrix encountered (indicates very flat curvature in
    some direction).  This matrix must be multiplied by the
    residual standard deviation to get the covariance of the
    parameter estimates -- see curve_fit.
    infodict : dict
    a dictionary of optional outputs with the key s::

        - 'nfev' : the number of function calls
        - 'fvec' : the function evaluated at the output
        - 'fjac' : A permutation of the R matrix of a QR
                 factorization of the final approximate
                 Jacobian matrix, stored column wise.
                 Together with ipvt, the covariance of the
                 estimate can be approximated.
        - 'ipvt' : an integer array of length N which defines
                 a permutation matrix, p, such that
                 fjac*p = q*r, where r is upper triangular
                 with diagonal elements of nonincreasing
                 magnitude. Column j of p is column ipvt(j)
                 of the identity matrix.
        - 'qtf'  : the vector (transpose(q) * fvec).

    mesg : str
    A string message giving information about the cause of failure.
    ier : int
    An integer flag.  If it is equal to 1, 2, 3 or 4, the solution was
    found.  Otherwise, the solution was not found. In either case, the
    optional output variable 'mesg' gives more information.

    Notes
    -----
    Code snipplet provided by Anthony Arendt, March 13, 2011.
    Uses scipy.optimize.leastsq, see documentation of
    scipy.optimize.leastsq for details.
    """

    try:
        from scipy import optimize
    except:
        print("scipy.optimize not found. Please install.")
        exit(1)

    def fitfunc(p, x):
        return (
            p[0]
            + p[1] * x
            + p[2] * np.cos(2.0 * np.pi * (x - p[3]) / 1.0)
            + p[4] * np.cos(2.0 * np.pi * (x - p[5]) / 0.5)
            + p[6] * np.cos(2.0 * np.pi * (x - p[7]) / 0.440794)
        )

    def errfunc(p, x, y):
        return fitfunc(p, x) - y

    p0 = [0.0, -80.0, 40.0, 0.0, 10.0, 0.0, 1.0, 0.0]

    return optimize.leastsq(errfunc, p0[:], args=(x, y), full_output=1)


def set_size(w, h, ax=None):
    """ w, h: width, height in inches """

    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def plot_prognostic(out_filename, df):

    # Dataframe with Year 2100 only for histogram
    m_df = df[df["Time"] == 2100]

    fig, ax = plt.subplots(1, 2, sharey="col", figsize=[6.2, 3.1], gridspec_kw=dict(width_ratios=[3, 1]))
    fig.subplots_adjust(wspace=0.05)
    [
        ax[0].plot(g[-1]["Time"], g[-1]["SLE (cm)"], color="0.5", alpha=0.5, linewidth=0.5)
        for g in df.groupby(by=["Group", "Model", "Exp"])
    ]
    # sns.lineplot(x="Time", y="SLE (cm)", data=df, hue="Group", palette="tab20", linewidth=0.5, ax=ax[0])
    sns.histplot(m_df, y="SLE (cm)", stat="density", bins=np.linspace(0, 20, 21), kde=True, color="0.5", ax=ax[1])
    sns.despine(ax=ax[1], left=True, bottom=True)
    ax[0].set_ylim(-1, 20)
    ax[1].set_ylim(-1, 20)
    ax[0].set_xlim(proj_start, proj_end)
    ax[1].set_xlabel(None)
    ax[1].set_ylabel(None)
    ax[1].axes.xaxis.set_visible(False)
    ax[1].axes.yaxis.set_visible(False)
    ax[0].set_xlabel("Year")
    ax[0].set_ylabel("SLE contribution (cm)")
    fig.savefig(out_filename, bbox_inches="tight")


def plot_historical(out_filename, df, grace):

    signal_color = "0.75"
    trend_color = "0.5"

    grace_mass = grace.mass.values - np.interp(proj_start, grace.time, grace.mass)
    grace_sigma = grace.sigma.values
    x = grace.time[np.logical_and(grace.time > hist_start, grace.time < hist_end)]
    y = grace_mass[np.logical_and(grace.time > hist_start, grace.time < hist_end)]
    p = trend_estimator(x, y)[0]
    grace_bias = p[0]
    grace_trend = p[1]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.fill_between(grace.time, grace_mass - 2 * grace_sigma, grace_mass + 2 * grace_sigma, color="#9ecae1")
    ax.plot(grace.time, grace_mass, ":", color="#3182bd", linewidth=0.4)
    (l_g,) = ax.plot(
        [hist_start, proj_start],
        [grace_bias + grace_trend * hist_start, grace_bias + grace_trend * proj_start],
        color="#2171b5",
        linewidth=1.0,
    )

    def plot_signal(g):
        x = g[-1]["Time"]
        y = g[-1]["Mass (Gt)"]
        return ax.plot(x, y, color=signal_color, linewidth=0.5)

    def plot_trend(g):
        x = g[-1]["Time"]
        y = g[-1]["Mass (Gt)"]
        p = trend_estimator(x, y)[0]
        return print(p[0], p[1])

    [plot_signal(g) for g in df.groupby(by=["Group", "Model", "Exp"])]
    [plot_trend(g) for g in df.groupby(by=["Group", "Model", "Exp"])]

    set_size(6, 3)

    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative mass change since 2015 (Gt)")

    ax.set_xlim(left=2006, right=2020)
    ax.set_ylim(-2000, 4000)
    fig.savefig(out_filename, bbox_inches="tight")


basedir = "v7_CMIP5_pub"

ctrl_files = []
for path in Path(basedir).rglob("*_mm_*_ctrl_proj.nc"):
    ctrl_files.append(path)

hist_files = []
for path in Path(basedir).rglob("*_mm_*_historical.nc"):
    hist_files.append(path)

files = []

hist_start = 2008
hist_end = 2014
proj_start = hist_end + 1
proj_end = 2100

# Ideally, we would get the time axis from the netCDF files, but some contributions have wrong meta data

# Historical time differs from model to model
# Projection time is same for all model
proj_time = np.arange(proj_start, proj_end + 1)
tolerance = 0.25

domain = {"GIS": "greenland_mass_200204_202008.txt"}

for d, data in domain.items():
    print(f"Analyzing {d}")

    grace = pd.read_csv(data, header=30, delim_whitespace=True, skipinitialspace=True, names=["time", "mass", "sigma"])

    dfs = []
    for path in Path(basedir).rglob("*_mm_cr_*.nc"):
        files.append(path)
        # Experiment
        nc = NC(path)
        exp_sle = nc.variables["sle"][:]
        # For comparison with GRACE, we use grounded ice mass, converted to Gt
        exp_mass = nc.variables["limgr"][:] / 1e12

        f = path.name.split(f"scalars_mm_cr_{d}_")[-1].split(".nc")[0].split("_")
        if len(f) == 3:
            group, model, exp = f
        else:
            g1, g2, model, exp = f
            group = f"{g1}-{g2}"

        ctrl_file = [m for m in ctrl_files if model in m.name][0]
        hist_file = [m for m in hist_files if model in m.name][0]

        # The last entry of the historical and the first entry of the projection are the same

        # Projection
        nc_ctrl = NC(ctrl_file)
        ctrl_sle = nc_ctrl.variables["sle"][:] - nc_ctrl.variables["sle"][0]
        ctrl_mass = (nc_ctrl.variables["limgr"][:] - nc_ctrl.variables["limgr"][0]) / 1e12

        # Historical
        nc_hist = NC(hist_file)
        hist_sle = nc_hist.variables["sle"][:-1] - nc_hist.variables["sle"][-1]
        hist_mass = (nc_hist.variables["limgr"][:-1] - nc_hist.variables["limgr"][-1]) / 1e12

        # We need to add the CTRL run to all simulations
        proj_sle = exp_sle + ctrl_sle
        proj_mass = exp_mass + ctrl_mass

        hist_time = -np.arange(len(hist_sle))[::-1] + hist_end
        m_time = np.hstack((hist_time, proj_time))
        m_sle = -np.hstack((hist_sle, proj_sle)) * 100
        m_mass = np.hstack((hist_mass, proj_mass))
        n = len(m_time)
        dfs.append(
            pd.DataFrame(
                data=np.hstack(
                    [
                        m_time.reshape(-1, 1),
                        m_sle.reshape(-1, 1),
                        m_mass.reshape(-1, 1),
                        np.repeat(group, n).reshape(-1, 1),
                        np.repeat(model, n).reshape(-1, 1),
                        np.repeat(exp, n).reshape(-1, 1),
                    ]
                ),
                columns=["Time", "SLE (cm)", "Mass (Gt)", "Group", "Model", "Exp"],
            )
        )
    df = pd.concat(dfs)
    df = df.astype({"Time": float, "SLE (cm)": float, "Mass (Gt)": float, "Model": str, "Exp": str})

    plot_historical(f"{d}_historical.pdf", df, grace)
    # plot_prognostic(f"{d}_prognostic.pdf", df)
