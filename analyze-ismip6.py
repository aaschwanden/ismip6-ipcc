#!/usr/bin/env python

# Copyright (C) 2019-20 Andy Aschwanden

from netCDF4 import Dataset as NC
import numpy as np
import os
import pylab as plt
import pandas as pd
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

#%% End of trend_estimator, start of plotting functions

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
    """
    Plot model projections
    """

    # min/max for binning and setting the axis bounds
    xmin = np.floor(df[df["Time"] == 2100]["SLE (cm)"].min())
    xmax = np.ceil(df[df["Time"] == 2100]["SLE (cm)"].max())

    # Dataframe with Year 2100 only for histogram
    p_df = df[(df["Time"] == 2100) & (df["Meet_Threshold"] == True)]
    f_df = df[(df["Time"] == 2100) & (df["Meet_Threshold"] == False)]

    fig, ax = plt.subplots(1, 2, sharey="col", figsize=[6.2, 2.0], 
                           num='prognostic_all', clear=True,
                           gridspec_kw=dict(width_ratios=[20, 1]))
    fig.subplots_adjust(wspace=0.025)

    def plot_signal(g):
        if g[-1]["Meet_Threshold"].any() == True:
            signal_color = "#74c476"
        else:
            signal_color = "0.5"

        return ax[0].plot(g[-1]["Time"], g[-1]["SLE (cm)"], color=signal_color, linewidth=0.5)

    # Plot each model response by grouping
    [plot_signal(g) for g in df.groupby(by=["Group", "Model", "Exp"])]

    ## Historgram plots
    # sns.histplot(
    #     f_df,
    #     y="SLE (cm)",
    #     bins=np.linspace(-2, 20, 23),
    #     kde=True,
    #     fill=False,
    #     color="0.5",
    #     ax=ax[1],
    # )
    # sns.histplot(
    #     p_df,
    #     y="SLE (cm)",
    #     bins=np.linspace(-2, 20, 23),
    #     kde=True,
    #     fill=False,
    #     color="#238b45",
    #     ax=ax[1],
    # )
    # sns.kdeplot(
    #     data=df[df["Time"] == 2100],
    #     y="SLE (cm)",
    #     hue="Meet_Threshold",
    #     palette=["0.5", "#238b45"],
    #     common_norm=False,
    #     clip=[xmin, xmax],
    #     linewidth=0.75,
    #     legend=False,
    #     ax=ax[1],
    # )

    ## Boxplot
    sns.boxplot(
        data=df[df["Time"] == 2100],
        x="Meet_Threshold",
        y="SLE (cm)",
        hue="Meet_Threshold",
        palette=["0.5", "#238b45"],
        width=0.8,
        linewidth=0.75,
        fliersize=0.40,
        ax=ax[1],
    )
    sns.despine(ax=ax[1], left=True, bottom=True)
    try:
        ax[1].get_legend().remove()
    except:
        pass
    ax[0].set_ylim(xmin, xmax)
    ax[1].set_ylim(xmin, xmax)
    ax[0].set_xlim(proj_start, proj_end)
    ax[1].set_xlabel(None)
    ax[1].set_ylabel(None)
    ax[1].axes.xaxis.set_visible(False)
    ax[1].axes.yaxis.set_visible(False)
    ax[0].set_xlabel("Year")
    ax[0].set_ylabel("SLE contribution (cm)")
    fig.savefig(out_filename, bbox_inches="tight")


def plot_prognostic_uaf(out_filename, df):
    """
    As above, but single out UAF
    """

    xmin = np.floor(df[df["Time"] == 2100]["SLE (cm)"].min())
    xmax = np.ceil(df[df["Time"] == 2100]["SLE (cm)"].max())

    is_df = df
    is_df["Is_UAF"] = False
    is_df.loc[is_df["Group"] == "UAF", "Is_UAF"] = True

    fig, ax = plt.subplots(1, 2, sharey="col", figsize=[6.2, 2.0], 
                           num='prognostic_uaf', clear=True,
                           gridspec_kw=dict(width_ratios=[20, 1]))
    fig.subplots_adjust(wspace=0.025)

    def plot_signal(g):
        if g[-1]["Meet_Threshold"].any() == True:
            signal_color = "#bdd7e7"
        else:
            signal_color = "0.5"

        return ax[0].plot(g[-1]["Time"], g[-1]["SLE (cm)"], color=signal_color, linewidth=0.5)

    [plot_signal(g) for g in is_df.groupby(by=["Group", "Model", "Exp"])]

    sns.boxplot(
        data=is_df[is_df["Time"] == 2100],
        x="Is_UAF",
        y="SLE (cm)",
        hue="Is_UAF",
        palette=["0.5", "#bdd7e7"],
        width=0.8,
        linewidth=0.75,
        fliersize=0.40,
        ax=ax[1],
    )
    sns.despine(ax=ax[1], left=True, bottom=True)
    try:
        ax[1].get_legend().remove()
    except:
        pass
    ax[0].set_ylim(xmin, xmax)
    ax[1].set_ylim(xmin, xmax)
    ax[0].set_xlim(proj_start, proj_end)
    ax[1].set_xlabel(None)
    ax[1].set_ylabel(None)
    ax[1].axes.xaxis.set_visible(False)
    ax[1].axes.yaxis.set_visible(False)
    ax[0].set_xlabel("Year")
    ax[0].set_ylabel("SLE contribution (cm)")
    fig.savefig(out_filename, bbox_inches="tight")


def plot_historical(out_filename, df, grace, model_trends):
    """
    Plot historical simulations, their trend, along with
    the GRACE signal and trend.
    """

    fig = plt.figure(num='historical', clear=True)
    ax = fig.add_subplot(111)

    def plot_signal(g):
        # plt.pause(0.1)
        # ax.set_title(str(g[0]))
        # input(str(g[0]) + "  press")
        m_df = g[-1]
        x = m_df["Time"]
        y = m_df["Mass (Gt)"]
        if m_df["Meet_Threshold"].values[0]:
            signal_color = "#74c476"
        else:
            signal_color = "0.75"

        return ax.plot(x, y, 
                       color=signal_color, 
                       linewidth=0.5)

    def plot_trend(model_trend):
        if np.abs(1 - model_trend / grace_trend) < tolerance:
            trend_color = "#238b45"
        else:
            trend_color = "0.5"

        return ax.plot(
            [hist_start, proj_start],
            # [model_trend * (hist_start - proj_start), 0],
            [0, -model_trend * (hist_start - proj_start)], # zero at hist_start
            color=trend_color,
            linewidth=0.75,
        )


    # Plot GRACE
    ax.fill_between(
        grace["Time"], grace["Mass"] - 2 * grace["Sigma"], grace["Mass"] + 2 * grace["Sigma"], color="#9ecae1"
    )
    ax.plot(grace["Time"], grace["Mass"], ":", color="#3182bd", linewidth=0.5)
    (l_g,) = ax.plot(
        [hist_start, proj_start],
        # [grace_bias + grace_trend * hist_start, 0], 
        [0, grace_bias + grace_trend * proj_start], # zero at hist_start
        color="#2171b5",
        linewidth=1.0,
        label='GRACE mass changes',
    )

    plt.legend()
    set_size(6, 2)

    ax.set_xlabel("Year")
    # ax.set_ylabel("Cumulative mass change\nsince 2015 (Gt)")
    ax.set_ylabel("Cumulative mass change\nsince 2007 (Gt)")

    ax.set_xlim(left=2001, right=2021)
    ax.set_ylim(-2000, 4000)
    
    [plot_signal(g) for g in df.groupby(by=["Group", "Model", "Exp"])]
    [plot_trend(row[-1]["Trend (Gt/yr)"]) for row in model_trends.iterrows()]
    
    
    fig.savefig(out_filename, bbox_inches="tight")

def plot_trends(out_filename, df):
    """
    Create plot of the historical trends of models vs. GRACE
    """

    fig, ax = plt.subplots(num='trend_plot', clear=True)    
    # sns.kdeplot(data=model_trends['Trend (Gt/yr)'])
    sns.histplot(data=model_trends, x='Trend (Gt/yr)', bins=np.arange(-350, 300, 50),
                 ax=ax, label='Model trend')
    sns.rugplot(data=model_trends['Trend (Gt/yr)'], ax=ax)
    sns.rugplot(data=[grace_trend], ax=ax)
    ax.set_xlabel('2007-2015 Mass Loss Trend (Gt/yr)')
    fig.legend(bbox_to_anchor=(0.65, .77, .01, .01), loc='lower left')
    
    ax.annotate('GRACE trend',
            xy=(grace_trend, 0.5), xycoords='data',
            xytext=(grace_trend, 4.7), textcoords='data',
            arrowprops=dict(arrowstyle='fancy', facecolor='C1', edgecolor='none',
                            mutation_scale=25),
            bbox=dict(fc='1', edgecolor='none'),
            horizontalalignment='center', verticalalignment='top', fontsize=12
            )

    fig.savefig(out_filename, bbox_inches="tight")

#%% End of plotting function definitions, start of analysis

# Where the ISMIP6 simulations reside
basedir = "v7_CMIP5_pub"

ctrl_files = []
for path in Path(basedir).rglob("*_mm_*_ctrl_proj.nc"):
    ctrl_files.append(path)

hist_files = []
for path in Path(basedir).rglob("*_mm_*_historical.nc"):
    hist_files.append(path)

files = []

hist_start = 2007
hist_end = 2014
proj_start = hist_end + 1
proj_end = 2100

# Ideally, we would get the time axis from the netCDF files, but some contributions (UAF!) have wrong meta data

# Historical time differs from model to model
# Projection time is same for all model
proj_time = np.arange(proj_start, proj_end + 1)

# tolarance within which we consider the model trend to be "pass" / "ok". This is arbitrary.
tolerance = 0.25

# Greenland only though this could easily be extended to Antarctica
domain = {"GIS": "greenland_mass_200204_202008.txt"}

for d, data in domain.items():
    print(f"Analyzing {d}")

    # Load the GRACE data
    grace = pd.read_csv(data, header=30, delim_whitespace=True, skipinitialspace=True, names=["Time", "Mass", "Sigma"])
    # # Normalize GRACE signal to the starting date of the projection
    # grace["Mass"] -= np.interp(proj_start, grace["Time"], grace.Mass)
    # Normalize GRACE signal to the starting date of the historical period
    grace["Mass"] -= np.interp(hist_start, grace["Time"], grace.Mass)

    # Get the GRACE trend
    grace_hist_df = grace[(grace["Time"] >= hist_start) & (grace["Time"] <= proj_start)]
    x = grace_hist_df["Time"]
    y = grace_hist_df["Mass"][(grace["Time"] > hist_start) & (grace["Time"] <= proj_start)]
    p = trend_estimator(x, y)[0]
    grace_bias = p[0]
    grace_trend = p[1]
    # # re-standardize baseline mass change to be zero at hist_start
    # grace["Mass"] += grace_trend * (proj_start - hist_start)


    # Now read model output from each of the ISMIP6 files. The information we 
    # need is in the file names, not the metadate so this is no fun.
        # Approach is to read each dataset into a dataframe, then concatenate all
        #   dataframes into one Arch dataframe that contains all model runs.
        # Resulting dataframe consists of both historical and projected changes
    dfs = []
    for path in Path(basedir).rglob("*_mm_cr_*.nc"):
        files.append(path)
        # Experiment
        nc = NC(path)
        exp_sle = nc.variables["sle"][:]
        # For comparison with GRACE, we use grounded ice mass, converted to Gt
        exp_mass = nc.variables["limgr"][:] / 1e12

        f = path.name.split(f"scalars_mm_cr_{d}_")[-1].split(".nc")[0].split("_")
        # This is ugly, because of "ITLS_PIK"
        if len(f) == 3:
            group, model, exp = f
        else:
            g1, g2, model, exp = f
            group = f"{g1}_{g2}"

        # Find the coressponding CTRL Historical simulations
        ctrl_file = [m for m in ctrl_files if (f"{group}_{model}" in m.name)][0]
        hist_file = [m for m in hist_files if (f"{group}_{model}" in m.name)][0]

        # The last entry of the historical and the first entry of the projection are the same

        # Projection
        nc_ctrl = NC(ctrl_file)
        ctrl_sle = nc_ctrl.variables["sle"][:] - nc_ctrl.variables["sle"][0]
        ctrl_mass = (nc_ctrl.variables["limgr"][:] - nc_ctrl.variables["limgr"][0]) / 1e12

        # Historical
        nc_hist = NC(hist_file)
        hist_sle = nc_hist.variables["sle"][:-1] - nc_hist.variables["sle"][-1]
        hist_mass = (nc_hist.variables["limgr"][:-1] - nc_hist.variables["limgr"][-1]) / 1e12

        # We need to add the CTRL run to all projections
        proj_sle = exp_sle + ctrl_sle
        proj_mass = exp_mass + ctrl_mass

        # Historical simulations start at different years since initialization was left
        # up to the modelers
        hist_time = -np.arange(len(hist_sle))[::-1] + hist_end

        # Let's add the data to the main DataFrame
        m_time = np.hstack((hist_time, proj_time))
        m_sle = -np.hstack((hist_sle, proj_sle)) * 100
        m_mass = np.hstack((hist_mass, proj_mass))
        
        # Change baseline for mass changes to 2007, hist_start
        try:
            i_hist_start = np.where(m_time == hist_start)[0][0]
            m_sle -= m_sle[i_hist_start] 
            m_mass -= m_mass[i_hist_start] 
        except IndexError: #For the UAF simulations which start in 2008
            # Do linear extrapolation of the UAF trends back to 2007.
            hist_inds = (m_time>=hist_start) & (m_time<=proj_start)
            p = trend_estimator(m_time[hist_inds], m_mass[hist_inds])[0]
            model_bias = p[0]
            model_trend = p[1]
            mass_offset = model_trend * (proj_start - hist_start)
            m_mass += mass_offset 

            p = trend_estimator(m_time[hist_inds], m_sle[hist_inds])[0]
            model_bias = p[0]
            model_trend = p[1]
            sle_offset = model_trend * (proj_start - hist_start)
            m_sle += sle_offset 

        
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
        # End of working with each model run individually (the path for-loop)
        
    # Concatenate all DataFrames and convert object types
    df = pd.concat(dfs)
    df = df.astype({"Time": float, "SLE (cm)": float, "Mass (Gt)": float, "Model": str, "Exp": str})

    # Add a boolean whether the Model is within the trend tolerance
    pass_dfs = []
    fail_dfs = []
    groups = []
    models = []
    exps = []
    trends = []
    for g in df.groupby(by=["Group", "Model", "Exp"]):
        m_df = g[-1][(g[-1]["Time"] >= hist_start) & (g[-1]["Time"] <= proj_start)]
        x = m_df["Time"]
        y = m_df["Mass (Gt)"]
        p = trend_estimator(x, y)[0]
        model_bias = p[0]
        model_trend = p[1]
        groups.append(g[0][0])
        models.append(g[0][1])
        exps.append(g[0][2])
        trends.append(model_trend)

        if np.abs(1 - model_trend / grace_trend) <= tolerance:
            pass_dfs.append(g[-1])
        else:
            fail_dfs.append(g[-1])
    fail_df = pd.concat(fail_dfs)
    fail_df["Meet_Threshold"] = False
    pass_df = pd.concat(pass_dfs)
    pass_df["Meet_Threshold"] = True
    df = pd.concat([fail_df, pass_df])
    model_trends = pd.DataFrame(
        data=np.hstack(
            [
                np.array(groups).reshape(-1, 1),
                np.array(models).reshape(-1, 1),
                np.array(exps).reshape(-1, 1),
                np.array(trends).reshape(-1, 1),
            ]
        ),
        columns=["Group", "Model", "Exp", "Trend (Gt/yr)"],
    )
    model_trends = model_trends.astype({"Group": str, "Model": str, "Exp": str, "Trend (Gt/yr)": float})
    model_trends = model_trends.groupby(by=["Group", "Model"]).mean().reset_index()

    plot_historical(f"{d}_historical.pdf", df, grace, model_trends)
    plot_prognostic(f"{d}_prognostic.pdf", df)
    plot_prognostic_uaf(f"{d}_prognostic_uaf.pdf", df)
    plot_trends(f"{d}_trends.pdf", df)
