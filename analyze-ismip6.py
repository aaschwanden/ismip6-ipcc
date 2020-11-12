#!/usr/bin/env python

# Copyright (C) 2019-20 Andy Aschwanden

from glob import glob
import matplotlib.lines as mlines
from netCDF4 import Dataset as NC
import numpy as np
import os
import pylab as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import statsmodels.api as sm


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


def plot_historical_fluxes(out_filename, df, grace):
    def plot_signal(g):
        if g[-1]["Meet_Threshold"].any() == True:
            signal_color = "#74c476"
        else:
            signal_color = "0.5"
        return ax.plot(g[-1]["Time"], g[-1]["Mass (Gt)"], color=signal_color, linewidth=0.75)

    def plot_smb(g):
        if g[-1]["Meet_Threshold"].any() == True:
            signal_color = "#74c476"
        else:
            signal_color = "0.5"
        return ax.plot(g[-1]["Time"], g[-1]["SMB (Gt)"], color=signal_color, linewidth=0.5, linestyle="dotted")

    def plot_d(g):
        if g[-1]["Meet_Threshold"].any() == True:
            signal_color = "#74c476"
        else:
            signal_color = "0.5"
        return ax.plot(g[-1]["Time"], g[-1]["D (Gt)"], color=signal_color, linewidth=0.5, linestyle="dashed")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Plot each model response by grouping
    [plot_smb(g) for g in df.groupby(by=["Group", "Model", "Exp"])]
    [plot_d(g) for g in df.groupby(by=["Group", "Model", "Exp"])]
    [plot_signal(g) for g in df.groupby(by=["Group", "Model", "Exp"])]
    grace_line = ax.plot(
        grace["Time"], grace["Mass (Gt)"], "-", color="#3182bd", linewidth=1, label="GRACE mass changes"  # ":",
    )

    l_mb = mlines.Line2D([], [], color="k", linewidth=0.5, linestyle="solid", label="MB")
    l_smb = mlines.Line2D([], [], color="k", linewidth=0.5, linestyle="dotted", label="SMB")
    l_d = mlines.Line2D([], [], color="k", linewidth=0.5, linestyle="dashed", label="D")
    legend_1 = ax.legend(handles=[l_mb, l_smb, l_d], loc="upper left")
    legend_1.get_frame().set_linewidth(0.0)
    legend_1.get_frame().set_alpha(0.0)

    good_models = mlines.Line2D([], [], color="#74c476", linewidth=0.5, label="Model trends within 25% of GRACE")
    all_models = mlines.Line2D([], [], color="0.75", linewidth=0.5, label="Other model trends")
    legend_2 = ax.legend(handles=[grace_line[0], good_models, all_models], loc="lower left")
    legend_2.get_frame().set_linewidth(0.0)
    legend_2.get_frame().set_alpha(0.0)
    # Pylab automacially removes first legend when legend is called a second time.
    # Add legend 1 back
    ax.add_artist(legend_1)

    ax.set_xlabel("Year")
    ax.set_ylabel(f"Cumulative mass change\nsince {hist_start} (Gt)")

    ax.set_xlim(hist_start, proj_start)
    fig.savefig(out_filename, bbox_inches="tight")

    ax.set_xlim(hist_start, proj_start)
    out_filename = f"{d}_fluxes_historical_2008_2030.pdf"
    fig.savefig(out_filename, bbox_inches="tight")


def plot_prognostic(out_filename, df):
    """
    Plot model projections
    """

    # min/max for binning and setting the axis bounds
    xmin = np.floor(df[df["Time"] >= proj_start]["SLE (cm)"].min())
    xmax = np.ceil(df[df["Time"] >= proj_start]["SLE (cm)"].max())

    # Dataframe with Year 2100 only for histogram
    p_df = df[(df["Time"] == 2100) & (df["Meet_Threshold"] == True)]
    f_df = df[(df["Time"] == 2100) & (df["Meet_Threshold"] == False)]

    fig, ax = plt.subplots(
        1,
        2,
        sharey="col",
        figsize=[6.2, 2.0],
        num="prognostic_all",
        clear=True,
        gridspec_kw=dict(width_ratios=[20, 1]),
    )
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


def plot_historical_as19(out_filename, df, grace, as19):
    def plot_signal(g):

        return ax.plot(g[-1]["Year"], g[-1]["Mass (Gt)"] - g[-1]["Mass (Gt)"].iloc[7], color="0.75", linewidth=0.5)

    fig = plt.figure(num="historical", clear=True)
    ax = fig.add_subplot(111)

    # [plot_signal(g) for g in as19.groupby(by=["RCP", "Experiment"])]

    # Plot GRACE and model results
    ax.fill_between(
        grace["Time"],
        grace["Mass (Gt)"] - 2 * grace["Sigma (Gt)"],
        grace["Mass (Gt)"] + 2 * grace["Sigma (Gt)"],
        color="#9ecae1",
    )

    grace_line = ax.plot(
        grace["Time"], grace["Mass (Gt)"], "-", color="#3182bd", linewidth=1, label="GRACE mass changes"  # ":",
    )

    as19_median = as19[as19["RCP"] == rcp].groupby(by="Year").quantile(0.50)
    as19_median = as19_median - as19_median.iloc[7]
    (l_as19_50,) = ax.plot(
        as19_median.index, as19_median["Mass (Gt)"], linewidth=1, color="0.0", label="AS19 LES median"
    )
    as19_5 = as19[as19["RCP"] == rcp].groupby(by="Year").quantile(0.05)
    as19_5 = as19_5 - as19_5.iloc[7]
    (l_as19_5,) = ax.plot(as19_5.index, as19_5["Mass (Gt)"], linewidth=0.5, color="0.30", label="AS19 LES 5th")
    as19_95 = as19[as19["RCP"] == rcp].groupby(by="Year").quantile(0.95)
    as19_95 = as19_95 - as19_95.iloc[7]
    (l_as19_95,) = ax.plot(as19_95.index, as19_95["Mass (Gt)"], linewidth=0.5, color="0.60", label="AS19 LES 95th")
    ax.legend(handles=[grace_line[0], l_as19_50, l_as19_5, l_as19_95], loc="upper right")
    set_size(6, 2)

    ax.set_xlabel("Year")
    ax.set_ylabel(f"Cumulative mass change\nsince {hist_start} (Gt)")

    ax.set_xlim(left=2005, right=2020)
    ax.set_ylim(-2000, 4000)

    fig.savefig(out_filename, bbox_inches="tight")


def plot_prognostic_w_as19(out_filename, df, as19):
    """
    Plot model projections
    """

    xmin = 0
    xmax = 45
    fig, ax = plt.subplots(
        1,
        2,
        sharey="col",
        figsize=[6.2, 2.0],
        num="prognostic_all",
        clear=True,
        gridspec_kw=dict(width_ratios=[20, 1]),
    )
    fig.subplots_adjust(wspace=0.025)

    def plot_signal(g):
        # if g[-1]["Meet_Threshold"].any() == True:
        #     signal_color = "#74c476"
        # else:
        #     signal_color = "0.5"
        signal_color = "0.5"

        return ax[0].plot(g[-1]["Time"], g[-1]["SLE (cm)"], color=signal_color, linewidth=0.1)

    # Plot each model response by grouping
    # [plot_signal(g) for g in df.groupby(by=["Group", "Model", "Exp"])]

    for rcp, rcp_col, rcp_col_shade in zip(
        ["85", "26"],
        [
            "#990002",
            "#003466",
        ],
        ["#F4A582", "#4393C3"],
    ):
        m_df = df[df["RCP"] == rcp]
        median = m_df.groupby(by=["Time"]).quantile(0.50)
        pctl5 = m_df.groupby(by=["Time"]).quantile(0.05)
        pctl25 = m_df.groupby(by=["Time"]).quantile(0.25)
        pctl75 = m_df.groupby(by=["Time"]).quantile(0.75)
        pctl95 = m_df.groupby(by=["Time"]).quantile(0.95)
        ax[0].plot(median.index, median["SLE (cm)"], linewidth=0.75, color=rcp_col)
        l90 = ax[0].fill_between(pctl95.index, pctl5["SLE (cm)"], pctl95["SLE (cm)"], color=rcp_col_shade)
        ax[0].plot(pctl5.index, pctl5["SLE (cm)"], linewidth=0.5, color=rcp_col)
        ax[0].plot(pctl95.index, pctl95["SLE (cm)"], linewidth=0.5, color=rcp_col)
    ## Boxplot
    sns.boxplot(
        data=as19[(as19["Year"] == 2100) & (as19["RCP"] != 45)],
        x="RCP",
        y="SLE (cm)",
        hue="RCP",
        palette=[
            "#4393C3",
            "#F4A582",
        ],
        whis=[5, 95],
        fliersize=0,
        width=1.0,
        linewidth=0.75,
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


def plot_prognostic_w_scaling(out_filename, df, model_trends, grace_trend):
    """
    Plot model projections with corrections
    """

    model_trends["trend_scalar"] = grace_trend / model_trends["Trend (Gt/yr)"]
    super_df = pd.merge(df, model_trends)
    super_df["scaled_SLE"] = super_df["SLE (cm)"] * super_df["trend_scalar"]

    # min/max for binning and setting the axis bounds
    ymin = -10  # np.floor(df[df["Time"] == 2100]["SLE (cm)"].min())
    ymax = 40  # np.ceil(df[df["Time"] == 2100]["SLE (cm)"].max())

    # Dataframe with Year 2100 only for histogram
    p_df = df[(df["Time"] == 2100) & (df["Meet_Threshold"] == True)]
    f_df = df[(df["Time"] == 2100) & (df["Meet_Threshold"] == False)]
    bp_super_df = super_df[(super_df["Time"] == 2100)]

    fig, ax = plt.subplots(
        2,
        2,
        sharey="col",
        figsize=[9.4, 5.5],
        num="prognostic_all_scaled",
        clear=True,
        gridspec_kw=dict(width_ratios=[20, 1]),
    )
    fig.subplots_adjust(wspace=0.025)

    def plot_signal(g):
        if g[-1]["Meet_Threshold"].any() == True:
            signal_color = "#74c476"
        else:
            signal_color = "0.5"
        # ax[1,0].set_xlim(2007, 2020)
        # ax[1,0].set_ylim(-2, 2)
        # plt.pause(0.1)
        return ax[0, 0].plot(g[-1]["Time"], g[-1]["SLE (cm)"], color=signal_color, linewidth=0.5)

    def plot_signal_scaled(g):
        if g[-1]["Meet_Threshold"].any() == True:
            signal_color = "#74c476"
        else:
            signal_color = "0.5"
        # ax[1,0].set_xlim(2007, 2020)
        # ax[1,0].set_ylim(-2, 2)
        # plt.pause(0.1)

        # print('good')
        # if g[-1]["Time"].any() == 1995.0:
        #     return

        return ax[1, 0].plot(g[-1]["Time"], g[-1]["scaled_SLE"], color=signal_color, linewidth=0.5)

    # Plot each model response by grouping
    [plot_signal_scaled(g) for g in super_df.groupby(by=["Group", "Model", "Exp"])]
    [plot_signal(g) for g in df.groupby(by=["Group", "Model", "Exp"])]

    ## Boxplot
    sns.boxplot(
        data=super_df[super_df["Time"] == 2100],
        x="Meet_Threshold",
        y="scaled_SLE",
        hue="Meet_Threshold",
        palette=["0.5", "#238b45"],
        width=0.8,
        linewidth=0.75,
        fliersize=0.40,
        ax=ax[1, 1],
    )
    sns.despine(ax=ax[1, 1], left=True, bottom=True)
    try:
        ax[1, 1].get_legend().remove()
    except:
        pass

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
        ax=ax[0, 1],
    )
    sns.despine(ax=ax[0, 1], left=True, bottom=True)
    try:
        ax[0, 1].get_legend().remove()
    except:
        pass

    ax[0, 0].text(2012, 30, "Goelzer et al., 2020, ISMIP6 projections", fontsize=12)
    ax[1, 0].text(2012, 30, "Projections linearly scaled to match 2007-2015 GRACE observations", fontsize=12)

    [ax.tick_params("x", top=True) for ax in ax[:, 0]]
    [ax.tick_params("y", right=True) for ax in ax[:, 0]]
    [ax.set_ylim(ymin, ymax) for ax in ax[:, 0]]
    [ax.set_ylim(ymin, ymax) for ax in ax[:, 1]]
    # [ax.set_xlim(hist_start, proj_end) for ax in ax[:,0]]
    [ax.set_xlim(hist_start, 2100) for ax in ax[:, 0]]
    [ax.set_xlabel(None) for ax in ax[:, 1]]
    [ax.set_ylabel(None) for ax in ax[:, 1]]
    ax[0, 0].set_xticklabels("")
    [ax.axes.xaxis.set_visible(False) for ax in ax[:, 1]]
    [ax.axes.yaxis.set_visible(False) for ax in ax[:, 1]]
    ax[1, 0].set_xlabel("Year")
    [ax.set_ylabel("SLE contribution (cm)") for ax in ax[:, 0]]

    fig.savefig(out_filename, bbox_inches="tight")


def plot_historical(out_filename, df, grace, model_trends):
    """
    Plot historical simulations, their trend, along with
    the GRACE signal and trend.
    """

    fig = plt.figure(num="historical", clear=True)
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

        return ax.plot(x, y, color=signal_color, linewidth=0.5)

    def plot_trend(model_trend):
        if np.abs(1 - model_trend / grace_trend) < tolerance:
            trend_color = "#238b45"
        else:
            trend_color = "0.5"

        return ax.plot(
            [hist_start, proj_start],
            # [model_trend * (hist_start - proj_start), 0],
            [0, -model_trend * (hist_start - proj_start)],  # zero at hist_start
            color=trend_color,
            linewidth=0.75,
        )

    # Plot GRACE and model results
    ax.fill_between(
        grace["Time"],
        grace["Mass (Gt)"] - 2 * grace["Sigma (Gt)"],
        grace["Mass (Gt)"] + 2 * grace["Sigma (Gt)"],
        color="#9ecae1",
    )

    [plot_signal(g) for g in df.groupby(by=["Group", "Model", "Exp"])]
    # [plot_trend(row[-1]["Trend (Gt/yr)"]) for row in model_trends.iterrows()]

    grace_line = ax.plot(
        grace["Time"], grace["Mass (Gt)"], "-", color="#3182bd", linewidth=1, label="GRACE mass changes"  # ":",
    )

    # (l_g,) = ax.plot(  # GRACE trend
    #     [hist_start, proj_start],
    #     # [grace_bias + grace_trend * hist_start, 0],
    #     [0, grace_bias + grace_trend * proj_start],  # zero at hist_start
    #     color="#2171b5",
    #     linewidth=1.0,
    # )

    good_models = mlines.Line2D([], [], color="#74c476", linewidth=0.5, label="Model trends within 25% of GRACE")
    all_models = mlines.Line2D([], [], color="0.75", linewidth=0.5, label="Other model trends")
    plt.legend(handles=[grace_line[0], good_models, all_models], loc="upper right")
    set_size(6, 2)

    ax.set_xlabel("Year")
    ax.set_ylabel(f"Cumulative mass change\nsince {hist_start} (Gt)")

    ax.set_xlim(left=2000, right=2030)
    ax.set_ylim(-3000, 5000)

    fig.savefig(out_filename, bbox_inches="tight")


def plot_trends(out_filename, df):
    """
    Create plot of the historical trends of models vs. GRACE
    """

    fig, ax = plt.subplots(num="trend_plot", clear=True)
    # sns.kdeplot(data=model_trends['Trend (Gt/yr)'])
    sns.histplot(
        data=model_trends,
        x="Trend (Gt/yr)",
        hue="Meet_Threshold",
        palette=["0.5", "#238b45"],
        bins=np.arange(-350, 300, 50),
        ax=ax,
        label="Model trend",
    )
    sns.rugplot(
        data=model_trends,
        x="Trend (Gt/yr)",
        hue="Meet_Threshold",
        palette=["0.5", "#238b45"],
        ax=ax,
        legend=False,
    )
    # sns.rugplot(data=[grace_trend], ax=ax)
    ax.set_xlabel(f"{hist_start}-{proj_start} Mass Loss Trend (Gt/yr)")
    # fig.legend(bbox_to_anchor=(0.65, .77, .01, .01), loc='lower left')

    ax.set_xlim(-400, 300)
    ax.set_ylim(0, 6.8)

    # Plot dashed line for GRACE
    ax.fill_between
    ax.plot([grace_trend, grace_trend], [0, 6.8], "--", color="#3182bd", linewidth=3)
    ax.text(-330, 2.8, "Observed GRACE trend", rotation=90, fontsize=12)

    # # Plot arrow for GRACE
    # ax.annotate('GRACE trend',
    #         xy=(grace_trend, 0.5), xycoords='data',
    #         xytext=(grace_trend, 4.7), textcoords='data',
    #         arrowprops=dict(arrowstyle='fancy', facecolor='C1', edgecolor='none',
    #                         mutation_scale=25),
    #         bbox=dict(fc='1', edgecolor='none'),
    #         horizontalalignment='center', verticalalignment='top', fontsize=12
    #         )

    fig.savefig(out_filename, bbox_inches="tight")


#%% End of plotting function definitions, start of analysis


secpera = 3.15569259747e7

# Where the ISMIP6 simulations reside
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
    grace = pd.read_csv(
        data, header=30, delim_whitespace=True, skipinitialspace=True, names=["Time", "Mass (Gt)", "Sigma (Gt)"]
    )
    # Normalize GRACE signal to the starting date of the projection
    grace["Mass (Gt)"] -= np.interp(proj_start, grace["Time"], grace["Mass (Gt)"])

    # Get the GRACE trend
    grace_time = (grace["Time"] >= hist_start) & (grace["Time"] <= proj_start)
    grace_hist_df = grace[grace_time]
    x = grace_hist_df["Time"]
    y = grace_hist_df["Mass (Gt)"][(grace["Time"] >= hist_start) & (grace["Time"] <= proj_start)]
    s = grace_hist_df["Sigma (Gt)"][(grace["Time"] >= hist_start) & (grace["Time"] <= proj_start)]
    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()
    p = ols.params
    grace_bias = p[0]
    grace_trend = p[1]

    as19 = pd.read_csv("as19//aschwanden_et_al_2019_les.gz")
    as19["SLE (cm)"] = -as19["Mass (Gt)"] / 362.5 / 10

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
        exp_smb = nc.variables["smb"][:] / 1e12 * secpera

        f = path.name.split(f"scalars_mm_cr_{d}_")[-1].split(".nc")[0].split("_")
        # This is ugly, because of "ITLS_PIK"
        if len(f) == 3:
            group, model, exp = f
        else:
            g1, g2, model, exp = f
            group = f"{g1}_{g2}"

        if exp in ["exp07"]:
            rcp = 26
        else:
            rcp = 85
        # Find the coressponding CTRL Historical simulations
        ctrl_file = [m for m in ctrl_files if (f"{group}_{model}" in m.name)][0]
        hist_file = [m for m in hist_files if (f"{group}_{model}" in m.name)][0]

        # The last entry of the historical and the first entry of the projection are the same

        # Projection
        nc_ctrl = NC(ctrl_file)
        ctrl_sle = nc_ctrl.variables["sle"][:]
        ctrl_mass = nc_ctrl.variables["limgr"][:] / 1e12
        ctrl_smb = nc_ctrl.variables["smb"][:] / 1e12 * secpera

        # Historical
        nc_hist = NC(hist_file)
        hist_sle = nc_hist.variables["sle"][:-1] - nc_hist.variables["sle"][-1]
        hist_mass = (nc_hist.variables["limgr"][:-1] - nc_hist.variables["limgr"][-1]) / 1e12
        hist_smb = nc_hist.variables["smb"][:-1] / 1e12 * secpera

        # proj_sle = exp_sle + ctrl_sle
        # proj_mass = exp_mass + ctrl_mass
        # proj_smb = exp_smb + ctrl_smb
        proj_sle = exp_sle
        proj_mass = exp_mass
        proj_smb = exp_smb

        # Historical simulations start at different years since initialization was left
        # up to the modelers
        hist_time = -np.arange(len(hist_sle))[::-1] + hist_end

        # Let's add the data to the main DataFrame
        m_time = np.hstack((hist_time, proj_time))
        m_sle = -np.hstack((hist_sle, proj_sle)) * 100
        m_sle -= np.interp(proj_start, m_time, m_sle)
        m_mass = np.hstack((hist_mass, proj_mass))
        m_mass -= np.interp(proj_start, m_time, m_mass)
        m_smb = np.cumsum(np.hstack((hist_smb, proj_smb)))
        m_smb -= np.interp(proj_start, m_time, m_smb)
        m_d = m_mass - m_smb
        n = len(m_time)
        dfs.append(
            pd.DataFrame(
                data=np.hstack(
                    [
                        m_time.reshape(-1, 1),
                        m_sle.reshape(-1, 1),
                        m_mass.reshape(-1, 1),
                        m_smb.reshape(-1, 1),
                        m_d.reshape(-1, 1),
                        np.repeat(group, n).reshape(-1, 1),
                        np.repeat(model, n).reshape(-1, 1),
                        np.repeat(exp, n).reshape(-1, 1),
                        np.repeat(rcp, n).reshape(-1, 1),
                    ]
                ),
                columns=["Time", "SLE (cm)", "Mass (Gt)", "SMB (Gt)", "D (Gt)", "Group", "Model", "Exp", "RCP"],
            )
        )
        # End of working with each model run individually (the path for-loop)

    # Concatenate all DataFrames and convert object types
    df = pd.concat(dfs)
    df = df.astype(
        {
            "Time": float,
            "SLE (cm)": float,
            "Mass (Gt)": float,
            "SMB (Gt)": float,
            "D (Gt)": float,
            "Model": str,
            "Exp": str,
            "RCP": str,
        }
    )

    # Add a boolean whether the Model is within the trend tolerance
    pass_dfs = []
    fail_dfs = []
    groups = []
    models = []
    exps = []
    trends = []
    sigmas = []
    # meet_thresh = []
    for g in df.groupby(by=["Group", "Model", "Exp"]):
        m_df = g[-1][(g[-1]["Time"] >= hist_start) & (g[-1]["Time"] <= proj_start)]
        hist_inds = (m_time >= hist_start) & (m_time <= proj_start)
        x = m_df["Time"]
        y = m_df["Mass (Gt)"]
        X = sm.add_constant(x)
        ols = sm.OLS(y, X).fit()
        p = ols.params
        model_bias = p[0]
        model_trend = p[1]
        model_trend_sigma = ols.bse[-1]
        groups.append(g[0][0])
        models.append(g[0][1])
        exps.append(g[0][2])
        trends.append(model_trend)
        sigmas.append(model_trend_sigma)
        # meet_thresh.append(np.abs(1 - model_trend / grace_trend) <= tolerance)
        if np.abs(1 - model_trend / grace_trend) <= tolerance:
            pass_dfs.append(g[-1])
        else:
            fail_dfs.append(g[-1])
    fail_df = pd.concat(fail_dfs)
    fail_df["Meet_Threshold"] = False
    if len(pass_dfs) == 0:  # in case no models pass the threshold...
        pass_df = pd.DataFrame()
    else:
        pass_df = pd.concat(pass_dfs)
        pass_df["Meet_Threshold"] = True
    df = pd.concat([fail_df, pass_df])
    df = df.astype(
        {
            "Time": float,
            "SLE (cm)": float,
            "Mass (Gt)": float,
            "SMB (Gt)": float,
            "D (Gt)": float,
            "Model": str,
            "Exp": str,
        }
    ).reset_index(drop=True)
    model_trends = pd.DataFrame(
        data=np.hstack(
            [
                np.array(groups).reshape(-1, 1),
                np.array(models).reshape(-1, 1),
                np.array(exps).reshape(-1, 1),
                np.array(trends).reshape(-1, 1),
                np.array(sigmas).reshape(-1, 1),
                # np.array(meet_thresh).reshape(-1, 1),
            ]
        ),
        columns=["Group", "Model", "Exp", "Trend (Gt/yr)", "Sigma (Gt/yr)"],  # "Meet_Threshold"],
    )
    model_trends = model_trends.astype(
        {"Group": str, "Model": str, "Exp": str, "Trend (Gt/yr)": float, "Sigma (Gt/yr)": float}
    )
    model_trends = model_trends.groupby(by=["Group", "Model"]).mean(numeric_only=False).reset_index()
    model_trends["Meet_Threshold"] = np.abs(1 - model_trends["Trend (Gt/yr)"] / grace_trend) <= tolerance
    # Create unique ID column Group-Model
    model_trends["Group-Model"] = model_trends["Group"] + "-" + model_trends["Model"]

    # %% Compare historical trend with final outcome of sea level equivalent
    final = df[df["Time"] == 2100].copy()
    final["Group-Model"] = final["Group"] + "-" + final["Model"]
    # Select those experimental results from experiments 05, 06, and 08,
    #   which are the RCP8.5 experiments, with medium Slater ocean sensitivity
    #   and MIROC5, NorESM and HadGEM2-ES GCMs

    final_rcp85core = final.loc[final["Exp"].isin(["exp05", "exp06", "exp08"])].copy()
    final_rcp85core = final_rcp85core.groupby(["Group-Model"]).mean()  # Take the mean of the three models
    final_rcp85core = final_rcp85core.merge(
        model_trends, on="Group-Model"
    )  # Merge the 2100 mass loss with the observed trend

    # fig, ax = plt.subplots()  # num="SLE v Trend", clear=True)
    # final_rcp85core.plot(
    #     x="Trend (Gt/yr)",
    #     y="SLE (cm)",
    #     kind="scatter",
    #     ax=ax,
    # )
    # ax.set_xlabel(f"{hist_start}-{proj_start} Mass Loss Trend (Gt/yr)")
    # ax.set_ylabel("2100 SLE Mass Loss (cm)")
    # # Label the scatter plot with the Group-Model names
    # for i in range(len(final_rcp85core)):
    #     ax.text(final_rcp85core['Trend (Gt/yr)'].values[i],
    #             final_rcp85core['SLE (cm)'].values[i],
    #             final_rcp85core['Group-Model'].values.tolist()[i],
    #             )
    # ax.grid('on')

    # %% Build Final Plots
    # h_df = df[(df["Time"] >= hist_start) & (df["Time"] <= proj_start)]
    # plot_historical_fluxes(f"{d}_fluxes_historical.pdf", h_df, grace)
    # plot_historical(f"{d}_historical.pdf", df, grace, model_trends)
    # plot_prognostic(f"{d}_prognostic.pdf", df)
    # plot_trends(f"{d}_trends.pdf", df)
    # plot_prognostic_w_scaling(f"{d}_prognostic_scaled.pdf", df, model_trends, grace_trend)
    plot_prognostic_w_as19(f"{d}_prognostic_w_as19.pdf", df, as19)
    plot_historical_as19(f"{d}_historical_as19.pdf", df, grace, as19)
