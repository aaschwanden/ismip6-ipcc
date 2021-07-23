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

import matplotlib as mpl
import matplotlib.cm as cmx
import matplotlib.colors as colors

from datetime import datetime
import time


def set_size(w, h, ax=None):
    """w, h: width, height in inches"""

    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def toYearFraction(date):
    def sinceEpoch(date):  # returns seconds since epoch
        return time.mktime(date.timetuple())

    s = sinceEpoch

    year = date.year
    startOfThisYear = datetime(year=year, month=1, day=1)
    startOfNextYear = datetime(year=year + 1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed / yearDuration

    return date.year + fraction


def plot_sle_pdfs(out_filename, df, as19):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.kdeplot(data=df, x="SLE (cm)", hue="RCP", hue_order=["26", "85"], palette=["#003466", "#990002"], ax=ax)
    sns.kdeplot(
        data=as19, x="SLE (cm)", hue="RCP", hue_order=[26, 85], palette=["#003466", "#990002"], fill=True, ax=ax
    )

    set_size(4.7, 2)
    fig.savefig(out_filename, bbox_inches="tight")


def plot_historical_partitioning_cumulative(out_filename, df, mou19):
    def plot_smb(g):
        return ax.plot(
            g[-1]["Year"],
            g[-1]["Cumulative surface mass balance anomaly (Gt)"],
            color=simulated_signal_color,
            linewidth=simulated_signal_lw,
            linestyle="solid",
        )

    def plot_d(g):
        return ax.plot(
            g[-1]["Year"],
            g[-1]["Cumulative ice dynamics anomaly (Gt)"],
            color=simulated_signal_color,
            linewidth=simulated_signal_lw,
            linestyle="dotted",
        )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    [plot_smb(g) for g in df.groupby(by=["Group", "Model", "Exp"])]
    [plot_d(g) for g in df.groupby(by=["Group", "Model", "Exp"])]

    ax.fill_between(
        mou19["Year"],
        0.95 * mou19["Cumulative surface mass balance anomaly (Gt)"],
        1.05 * mou19["Cumulative surface mass balance anomaly (Gt)"],
        color=mouginot_sigma_color,
        linewidth=mouginot_signal_lw,
        linestyle="solid",
    )
    ax.fill_between(
        mou19["Year"],
        0.96 * mou19["Cumulative ice dynamics anomaly (Gt)"],
        1.04 * mou19["Cumulative ice dynamics anomaly (Gt)"],
        color=mouginot_sigma_color,
        linewidth=mouginot_signal_lw,
        linestyle="dotted",
    )

    ax.plot(
        mou19["Year"],
        mou19["Cumulative surface mass balance anomaly (Gt)"],
        color=mouginot_signal_color,
        linewidth=mouginot_signal_lw,
        linestyle="solid",
    )
    ax.plot(
        mou19["Year"],
        mou19["Cumulative ice dynamics anomaly (Gt)"],
        color=mouginot_signal_color,
        linewidth=mouginot_signal_lw,
        linestyle="dotted",
    )

    ax.axvline(proj_start, color="k", linestyle="dashed", linewidth=grace_signal_lw)
    ax.axhline(0, color="k", linestyle="dotted", linewidth=grace_signal_lw)
    ax.text(2014.75, -7000, "Historical Period", ha="right")
    ax.text(2015.25, -7000, "Projection Period", ha="left")

    l_smb = mlines.Line2D([], [], color="k", linewidth=0.5, linestyle="solid", label="SMB")
    l_d = mlines.Line2D([], [], color="k", linewidth=0.5, linestyle="dotted", label="D")
    l_mou19 = mlines.Line2D(
        [], [], color=mouginot_signal_color, linewidth=0.5, linestyle="solid", label="Observed (Mouginot et al, 2019)"
    )
    l_simulated = mlines.Line2D(
        [], [], color=simulated_signal_color, linewidth=0.5, linestyle="solid", label="Simulated"
    )
    legend_1 = ax.legend(handles=[l_smb, l_d], loc="upper left")
    legend_1.get_frame().set_linewidth(0.0)
    legend_1.get_frame().set_alpha(0.0)

    legend_2 = ax.legend(handles=[l_mou19, l_simulated], loc="upper right")
    legend_2.get_frame().set_linewidth(0.0)
    legend_2.get_frame().set_alpha(0.0)

    # Pylab automacially removes first legend when legend is called a second time.
    # Add legend 1 back
    ax.add_artist(legend_1)

    ax.set_xlabel("Year")
    ax.set_ylabel(f"Cumulative flux\since{proj_start} (Gt)")

    ax.set_xlim(2000, 2025)
    ymin = -7500
    ymax = 7500
    ax.set_ylim(ymin, ymax)

    set_size(4.7, 2)
    fig.savefig(out_filename, bbox_inches="tight")


def plot_historical_partitioning(out_filename, df, mou19, man):
    def plot_smb(g):
        return ax.plot(
            g[-1]["Year"],
            g[-1]["Rate of surface mass balance anomaly (Gt/yr)"],
            color=simulated_signal_color,
            linewidth=simulated_signal_lw,
            linestyle="solid",
        )

    def plot_d(g):
        return ax.plot(
            g[-1]["Year"],
            g[-1]["Rate of ice dynamics anomaly (Gt/yr)"],
            color=simulated_signal_color,
            linewidth=simulated_signal_lw,
            linestyle="dashed",
        )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    [plot_smb(g) for g in df.groupby(by=["Group", "Model", "Exp"])]
    [plot_d(g) for g in df.groupby(by=["Group", "Model", "Exp"])]

    ax.fill_between(
        imbie["Year"],
        imbie["Rate of surface mass balance anomaly (Gt/yr)"]
        - 1 * imbie["Rate of surface mass balance anomaly uncertainty (Gt/yr)"],
        imbie["Rate of surface mass balance anomaly (Gt/yr)"]
        + 1 * imbie["Rate of surface mass balance anomaly uncertainty (Gt/yr)"],
        color=imbie_sigma_color,
        alpha=0.5,
        linewidth=0,
    )

    ax.fill_between(
        imbie["Year"],
        imbie["Rate of ice dynamics anomaly (Gt/yr)"] - 1 * imbie["Rate of ice dynamics anomaly uncertainty (Gt/yr)"],
        imbie["Rate of ice dynamics anomaly (Gt/yr)"] + 1 * imbie["Rate of ice dynamics anomaly uncertainty (Gt/yr)"],
        color=imbie_sigma_color,
        alpha=0.5,
        linewidth=0,
    )

    ax.fill_between(
        mou19["Year"],
        0.95 * mou19["Rate of surface mass balance anomaly (Gt/yr)"],
        1.05 * mou19["Rate of surface mass balance anomaly (Gt/yr)"],
        color=mouginot_sigma_color,
        alpha=0.5,
        linewidth=0,
    )
    ax.fill_between(
        mou19["Year"],
        0.96 * mou19["Rate of ice dynamics anomaly (Gt/yr)"],
        1.04 * mou19["Rate of ice dynamics anomaly (Gt/yr)"],
        color=mouginot_sigma_color,
        alpha=0.5,
        linewidth=0,
    )

    ax.plot(
        mou19["Year"],
        mou19["Rate of surface mass balance anomaly (Gt/yr)"],
        color=mouginot_signal_color,
        linewidth=mouginot_signal_lw,
        linestyle="solid",
    )
    ax.plot(
        mou19["Year"],
        mou19["Rate of ice dynamics anomaly (Gt/yr)"],
        color=mouginot_signal_color,
        linewidth=mouginot_signal_lw,
        linestyle="dashed",
    )

    ax.fill_between(
        man["Year"],
        -man["Discharge [Gt yr-1]"] - 1 * man["Discharge Error [Gt yr-1]"],
        -man["Discharge [Gt yr-1]"] + 1 * man["Discharge Error [Gt yr-1]"],
        color=mankoff_sigma_color,
        alpha=0.5,
        linewidth=0,
    )

    ax.plot(
        man["Year"],
        -man["Discharge [Gt yr-1]"],
        color=mankoff_signal_color,
        linewidth=mouginot_signal_lw,
        linestyle="dashed",
    )

    ax.plot(
        imbie["Year"],
        imbie["Rate of surface mass balance anomaly (Gt/yr)"],
        color=imbie_signal_color,
        linewidth=imbie_signal_lw,
        linestyle="solid",
    )
    ax.plot(
        imbie["Year"],
        imbie["Rate of ice dynamics anomaly (Gt/yr)"],
        color=imbie_signal_color,
        linewidth=imbie_signal_lw,
        linestyle="dashed",
    )

    ax.axvline(proj_start, color="k", linestyle="dashed", linewidth=grace_signal_lw)
    ax.axhline(0, color="k", linestyle="dotted", linewidth=grace_signal_lw)

    l_smb = mlines.Line2D([], [], color="k", linewidth=0.5, linestyle="solid", label="SMB")
    l_d = mlines.Line2D([], [], color="k", linewidth=0.5, linestyle="dashed", label="D")
    l_mou19 = mlines.Line2D(
        [],
        [],
        color=mouginot_signal_color,
        linewidth=0.5,
        linestyle="solid",
        label="Reconstructed (Mouginot et al, 2019)",
    )
    l_imbie = mlines.Line2D(
        [], [], color=imbie_signal_color, linewidth=0.5, linestyle="solid", label="Reconstructed (IMBIE)"
    )
    l_man = mlines.Line2D(
        [],
        [],
        color=mankoff_signal_color,
        linewidth=0.5,
        linestyle="solid",
        label="Reconstructed (Mankoff et al, 2020)",
    )
    l_simulated = mlines.Line2D(
        [], [], color=simulated_signal_color, linewidth=0.5, linestyle="solid", label="Simulated"
    )
    legend_1 = ax.legend(
        handles=[l_smb, l_d], loc="upper left", bbox_to_anchor=(0.1, 0.01, 0, 0), bbox_transform=plt.gcf().transFigure
    )
    legend_1.get_frame().set_linewidth(0.0)
    legend_1.get_frame().set_alpha(0.0)

    legend_2 = ax.legend(
        handles=[l_simulated, l_mou19, l_imbie, l_man],
        loc="upper left",
        bbox_to_anchor=(0.30, 0.01, 0, 0),
        bbox_transform=plt.gcf().transFigure,
    )
    #    legend_2 = ax.legend(handles=[l_mou19, l_man, l_simulated], loc="upper right")
    legend_2.get_frame().set_linewidth(0.0)
    legend_2.get_frame().set_alpha(0.0)

    # Pylab automacially removes first legend when legend is called a second time.
    # Add legend 1 back
    ax.add_artist(legend_1)

    ax.set_xlabel("Year")
    ax.set_ylabel(f"Flux (Gt/yr)")

    ax.set_xlim(2000, 2014)
    ymin = -750
    ymax = 750
    ax.set_ylim(ymin, ymax)

    set_size(3.2, 2)
    fig.savefig(out_filename, bbox_inches="tight")


def plot_prognostic(out_filename, df, as19):
    """
    Plot model projections
    """

    # min/max for binning and setting the axis bounds
    xmin = np.floor(df[df["Year"] >= proj_start]["SLE (cm)"].min())
    xmax = np.ceil(df[df["Year"] >= proj_start]["SLE (cm)"].max())

    xmin = -5
    xmax = 16

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

    def plot_signal(g, *kwargs):
        if g[-1]["RCP"].any() == "85":
            signal_color = rcp_col_dict[85]
        else:
            signal_color = rcp_col_dict[26]

        return ax[0].plot(g[-1]["Year"], g[-1]["SLE (cm)"], color=signal_color, linewidth=0.5, alpha=0.5, *kwargs)

    # Plot each model response by grouping
    [plot_signal(g) for g in df.groupby(by=["Group", "Model", "Exp"])]

    ## Boxplot
    sns.kdeplot(
        data=df[df["Year"] == 2100],
        y="SLE (cm)",
        hue="RCP",
        hue_order=["26", "85"],
        palette=[rcp_col_dict[26], rcp_col_dict[85]],
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


def plot_historical(out_filename, df, grace, mou19, imbie, model_trends):
    """
    Plot historical simulations and the GRACE signal.
    """

    def plot_signal(g):
        m_df = g[-1]
        x = m_df["Year"]
        y = m_df["Cumulative ice sheet mass change (Gt)"]

        return ax.plot(x, y, color=simulated_signal_color, linewidth=simulated_signal_lw)

    xmin = 2000
    xmax = 2025
    ymin = -3000
    ymax = 4000

    fig = plt.figure(num="historical", clear=True)
    ax = fig.add_subplot(111)

    [plot_signal(g) for g in df.groupby(by=["Group", "Model", "Exp"])]

    # Plot GRACE
    ax.fill_between(
        mou19["Year"],
        (1 - 0.057) * mou19["Cumulative ice sheet mass change (Gt)"],
        (1 + 0.057) * mou19["Cumulative ice sheet mass change (Gt)"],
        color=mouginot_sigma_color,
        alpha=0.5,
        linewidth=0,
    )
    mou19_line = ax.plot(
        mou19["Year"],
        mou19["Cumulative ice sheet mass change (Gt)"],
        "-",
        color=mouginot_signal_color,
        linewidth=mouginot_signal_lw,
        label="Observed (Mouginot et al, 2019)",
    )

    ax.fill_between(
        imbie["Year"],
        imbie["Cumulative ice sheet mass change (Gt)"]
        - 1 * imbie["Cumulative ice sheet mass change uncertainty (Gt)"],
        imbie["Cumulative ice sheet mass change (Gt)"]
        + 1 * imbie["Cumulative ice sheet mass change uncertainty (Gt)"],
        color=imbie_sigma_color,
        alpha=0.5,
        linewidth=0,
    )
    imbie_line = ax.plot(
        imbie["Year"],
        imbie["Cumulative ice sheet mass change (Gt)"],
        "-",
        color=imbie_signal_color,
        linewidth=imbie_signal_lw,
        label="Observed (IMBIE)",
    )

    ax.fill_between(
        grace["Year"],
        grace["Cumulative ice sheet mass change (Gt)"]
        - 1 * grace["Cumulative ice sheet mass change uncertainty (Gt)"],
        grace["Cumulative ice sheet mass change (Gt)"]
        + 1 * grace["Cumulative ice sheet mass change uncertainty (Gt)"],
        color=grace_sigma_color,
        alpha=0.5,
        linewidth=0,
    )
    grace_line = ax.plot(
        grace["Year"],
        grace["Cumulative ice sheet mass change (Gt)"],
        "-",
        color=grace_signal_color,
        linewidth=grace_signal_lw,
        label="Observed (GRACE)",
    )
    ax.axvline(proj_start, color="k", linestyle="dashed", linewidth=grace_signal_lw)
    ax.axhline(0, color="k", linestyle="dotted", linewidth=grace_signal_lw)
    ax.text(2014.75, 3000, "Historical Period", ha="right")
    ax.text(2015.25, 3000, "Projection Period", ha="left")

    model_line = mlines.Line2D([], [], color=simulated_signal_color, linewidth=simulated_signal_lw, label="Simulated")

    legend = ax.legend(handles=[grace_line[0], mou19_line[0], imbie_line[0], model_line], loc="lower left")
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    ax.set_xlabel("Year")
    ax.set_ylabel(f"Cumulative mass change\nsince {proj_start} (Gt)")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax_sle = ax.twinx()
    ax_sle.set_ylabel(f"Contribution to sea-level \nsince {proj_start} (cm SLE)")
    ax_sle.set_ylim(-ymin * gt2cmSLE, -ymax * gt2cmSLE)

    set_size(5, 2)

    fig.savefig(out_filename, bbox_inches="tight")


def plot_trends(out_filename, df):
    """
    Create plot of the historical trends of models vs. GRACE
    """

    fig, ax = plt.subplots(num="trend_plot", clear=True)
    sns.histplot(
        data=model_trends,
        x="Trend (Gt/yr)",
        palette=[simulated_signal_color],
        color=simulated_signal_color,
        bins=np.arange(-350, 300, 50),
        ax=ax,
        label="Model trend",
    )
    sns.rugplot(
        data=model_trends,
        x="Trend (Gt/yr)",
        palette=[simulated_signal_color],
        ax=ax,
        legend=False,
    )
    ax.set_xlabel(f"{hist_start}-{proj_start} Mass Change Trend (Gt/yr)")

    ax.set_xlim(-400, 200)
    ymin = 0
    ymax = 6.8
    ax.set_ylim(0, 6.8)

    # Plot dashed line for GRACE
    ax.fill_between(
        [grace_trend - grace_trend_stderr, grace_trend + grace_trend_stderr],
        [ymin, ymin],
        [ymax, ymax],
        color=grace_sigma_color,
    )
    ax.axvline(grace_trend, linestyle="solid", color=grace_signal_color, linewidth=1)
    ax.axvline(grace_trend - grace_trend_stderr, linestyle="dotted", color=grace_signal_color, linewidth=1)
    ax.axvline(grace_trend + grace_trend_stderr, linestyle="dotted", color=grace_signal_color, linewidth=1)
    ax.axvline(grace_trend, linestyle="solid", color=grace_signal_color, linewidth=1)
    ax.text(-340, 2.2, "Observed (GRACE)", rotation=90, fontsize=12)

    ax.axvline(0, linestyle="--", color="k", linewidth=1)

    set_size(3.2, 2.4)
    fig.savefig(out_filename, bbox_inches="tight")


#%% End of plotting function definitions, start of analysis

fontsize = 8
lw = 0.65
aspect_ratio = 0.35
markersize = 2

params = {
    "axes.linewidth": 0.25,
    "lines.linewidth": lw,
    "axes.labelsize": fontsize,
    "font.size": fontsize,
    "xtick.direction": "in",
    "xtick.labelsize": fontsize,
    "xtick.major.size": 2.5,
    "xtick.major.width": 0.25,
    "ytick.direction": "in",
    "ytick.labelsize": fontsize,
    "ytick.major.size": 2.5,
    "ytick.major.width": 0.25,
    "legend.fontsize": fontsize,
    "lines.markersize": markersize,
    "font.size": fontsize,
}

plt.rcParams.update(params)


secpera = 3.15569259747e7

grace_signal_lw = 0.75
mouginot_signal_lw = 0.75
imbie_signal_lw = 0.75
simulated_signal_lw = 0.3
grace_signal_color = "#084594"
grace_sigma_color = "#9ecae1"
mouginot_signal_color = "#a63603"
mouginot_sigma_color = "#fdbe85"
imbie_signal_color = "#005a32"
imbie_sigma_color = "#a1d99b"
mankoff_signal_color = "#54278f"
mankoff_sigma_color = "#bcbddc"
simulated_signal_color = "0.7"

gt2cmSLE = 1.0 / 362.5 / 10.0

rcp_list = ["26", "45", "85"]
rcp_col_dict = {"CTRL": "k", 85: "#990002", 45: "#5492CD", 26: "#003466"}

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
domain = {"GIS": "grace/greenland_mass_200204_202008.txt"}

remove_ctrl = True

for d, data in domain.items():
    print(f"Analyzing {d}")

    # Load the GRACE data
    grace = pd.read_csv(
        data,
        header=30,
        delim_whitespace=True,
        skipinitialspace=True,
        names=["Year", "Cumulative ice sheet mass change (Gt)", "Cumulative ice sheet mass change uncertainty (Gt)"],
    )
    # Normalize GRACE signal to the starting date of the projection
    grace["Cumulative ice sheet mass change (Gt)"] -= np.interp(
        proj_start, grace["Year"], grace["Cumulative ice sheet mass change (Gt)"]
    )

    # Get the GRACE trend
    grace_time = (grace["Year"] >= hist_start) & (grace["Year"] <= proj_start)
    grace_hist_df = grace[grace_time]
    x = grace_hist_df["Year"]
    y = grace_hist_df["Cumulative ice sheet mass change (Gt)"]
    s = grace_hist_df["Cumulative ice sheet mass change uncertainty (Gt)"]
    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()
    p = ols.params
    grace_bias = p[0]
    grace_trend = p[1]
    grace_trend_stderr = ols.bse[1]

    as19 = pd.read_csv("as19/aschwanden_et_al_2019_les_2008_norm.csv.gz")
    as19["SLE (cm)"] = -as19["Mass (Gt)"] / 362.5 / 10
    as19 = as19.astype({"RCP": int, "Experiment": int})

    imbie_df = pd.read_excel(
        "imbie/imbie_dataset_greenland_dynamics-2020_02_28.xlsx", sheet_name="Greenland Ice Mass", engine="openpyxl"
    )
    imbie = imbie_df[
        [
            "Year",
            "Cumulative ice sheet mass change (Gt)",
            "Cumulative ice sheet mass change uncertainty (Gt)",
            "Cumulative surface mass balance anomaly (Gt)",
            "Cumulative surface mass balance anomaly uncertainty (Gt)",
            "Cumulative ice dynamics anomaly (Gt)",
            "Cumulative ice dynamics anomaly uncertainty (Gt)",
            "Rate of mass balance anomaly (Gt/yr)",
            "Rate of ice dynamics anomaly (Gt/yr)",
            "Rate of mass balance anomaly uncertainty (Gt/yr)",
            "Rate of ice dyanamics anomaly uncertainty (Gt/yr)",
        ]
    ].rename(
        columns={
            "Rate of mass balance anomaly (Gt/yr)": "Rate of surface mass balance anomaly (Gt/yr)",
            "Rate of mass balance anomaly uncertainty (Gt/yr)": "Rate of surface mass balance anomaly uncertainty (Gt/yr)",
            "Rate of ice dyanamics anomaly uncertainty (Gt/yr)": "Rate of ice dynamics anomaly uncertainty (Gt/yr)",
        }
    )

    for v in [
        "Cumulative ice sheet mass change (Gt)",
        "Cumulative ice dynamics anomaly (Gt)",
        "Cumulative surface mass balance anomaly (Gt)",
    ]:
        imbie[v] -= imbie[imbie["Year"] == proj_start][v].values

    s = imbie[(imbie["Year"] >= 1980) & (imbie["Year"] < 1990)]
    mass_mean = s["Cumulative ice sheet mass change (Gt)"].mean() / (1990 - 1980)
    smb_mean = s["Cumulative surface mass balance anomaly (Gt)"].mean() / (1990 - 1980)
    imbie[f"Rate of surface mass balance anomaly (Gt/yr)"] += 2 * 1964 / 10
    imbie[f"Rate of ice dynamics anomaly (Gt/yr)"] -= 2 * 1964 / 10

    mou19_df = pd.read_excel(
        "mouginot/pnas.1904242116.sd02.xlsx", sheet_name="(2) MB_GIS", header=8, usecols="B,AR:BJ", engine="openpyxl"
    )
    mou19_d = mou19_df.iloc[7]
    mou19_smb = mou19_df.iloc[19]
    mou19_mass = mou19_df.iloc[41]
    mou19 = pd.DataFrame(
        data=np.hstack(
            [
                mou19_df.columns[1::].values.reshape(-1, 1),
                mou19_mass.values[1::].reshape(-1, 1),
                np.cumsum(mou19_smb.values[1::]).reshape(-1, 1),
                -np.cumsum(mou19_d.values[1::]).reshape(-1, 1),
                mou19_smb.values[1::].reshape(-1, 1),
                -mou19_d.values[1::].reshape(-1, 1),
            ]
        ),
        columns=[
            "Year",
            "Cumulative ice sheet mass change (Gt)",
            "Cumulative surface mass balance anomaly (Gt)",
            "Cumulative ice dynamics anomaly (Gt)",
            "Rate of surface mass balance anomaly (Gt/yr)",
            "Rate of ice dynamics anomaly (Gt/yr)",
        ],
    )
    mou19 = mou19.astype(
        {
            "Year": float,
            "Cumulative ice sheet mass change (Gt)": float,
            "Cumulative surface mass balance anomaly (Gt)": float,
            "Cumulative ice dynamics anomaly (Gt)": float,
            "Rate of surface mass balance anomaly (Gt/yr)": float,
            "Rate of ice dynamics anomaly (Gt/yr)": float,
        }
    )

    # Normalize
    for v in [
        "Cumulative ice sheet mass change (Gt)",
        "Cumulative ice dynamics anomaly (Gt)",
        "Cumulative surface mass balance anomaly (Gt)",
    ]:
        mou19[v] -= mou19[mou19["Year"] == proj_start][v].values

    man_d = pd.read_csv("mankoff/GIS_D.csv", parse_dates=[0])
    man_d["Year"] = [toYearFraction(d) for d in man_d["Date"]]
    man_d = man_d.astype({"Discharge [Gt yr-1]": float})
    man_err = pd.read_csv("mankoff/GIS_err.csv", parse_dates=[0])
    man_err["Year"] = [toYearFraction(d) for d in man_err["Date"]]
    man_err = man_err.astype({"Discharge Error [Gt yr-1]": float})
    man = pd.merge(man_d, man_err, on="Year").drop(columns=["Date_x", "Date_y"])

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

        # Per email with Heiko on Nov. 13, 2020, stick with just the exp projections alone, without adding back the ctrl projections
        """
        from Heiko:
        "The solution that we chose for ISMIP6 is therefore to remove the ctrl_proj from the projections
        and communicate the numbers as such, i.e. SL contribution for additional forcing after 2014. 
        In our (strong) opinion, the results should never be communicated uncorrected."
        
        Also, point of reference from Goelzer et al., 2020, the ctrl simulations represent mass change
        with the SMB fixed to 1960-1989 levels (no anomaly in SMB) and no change in ice sheet mask.
        So ctrl after the historical spinup represents an abrupt return to an earlier SMB forcing in 2015.
        """

        if remove_ctrl:
            proj_sle = exp_sle
            proj_mass = exp_mass
        else:
            proj_sle = exp_sle + ctrl_sle
            proj_mass = exp_mass + ctrl_mass

        proj_smb = exp_smb

        # Historical simulations start at different years since initialization was left
        # up to the modelers
        hist_time = -np.arange(len(hist_sle))[::-1] + hist_end

        # Let's add the data to the main DataFrame
        m_time = np.hstack((hist_time, proj_time))
        m_sle = -np.hstack((hist_sle, proj_sle)) * 100
        m_sle -= np.interp(proj_start, m_time, m_sle)
        m_mass = np.hstack((hist_mass, proj_mass))
        m_smb = np.cumsum(np.hstack((hist_smb, proj_smb)))
        m_smb -= np.interp(proj_start, m_time, m_smb)
        m_d = m_mass - m_smb
        m_mass_rate = np.gradient(np.hstack((hist_mass, proj_mass)))
        m_smb_rate = np.hstack((hist_smb, proj_smb))
        m_d_rate = m_mass_rate - m_smb_rate
        m_mass -= np.interp(proj_start, m_time, m_mass)

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
                        m_mass_rate.reshape(-1, 1),
                        m_smb_rate.reshape(-1, 1),
                        m_d_rate.reshape(-1, 1),
                        np.repeat(group, n).reshape(-1, 1),
                        np.repeat(model, n).reshape(-1, 1),
                        np.repeat(exp, n).reshape(-1, 1),
                        np.repeat(rcp, n).reshape(-1, 1),
                    ]
                ),
                columns=[
                    "Year",
                    "SLE (cm)",
                    "Cumulative ice sheet mass change (Gt)",
                    "Cumulative surface mass balance anomaly (Gt)",
                    "Cumulative ice dynamics anomaly (Gt)",
                    "Rate of ice sheet mass change (Gt/yr)",
                    "Rate of surface mass balance anomaly (Gt/yr)",
                    "Rate of ice dynamics anomaly (Gt/yr)",
                    "Group",
                    "Model",
                    "Exp",
                    "RCP",
                ],
            )
        )
        # End of working with each model run individually (the path for-loop)

    # Concatenate all DataFrames and convert object types
    df = pd.concat(dfs)
    df = df.astype(
        {
            "Year": float,
            "SLE (cm)": float,
            "Cumulative ice sheet mass change (Gt)": float,
            "Cumulative surface mass balance anomaly (Gt)": float,
            "Cumulative ice dynamics anomaly (Gt)": float,
            "Rate of ice sheet mass change (Gt/yr)": float,
            "Rate of surface mass balance anomaly (Gt/yr)": float,
            "Rate of ice dynamics anomaly (Gt/yr)": float,
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
    for g in df.groupby(by=["Group", "Model", "Exp"]):
        m_df = g[-1][(g[-1]["Year"] >= hist_start) & (g[-1]["Year"] <= proj_start)]
        x = m_df["Year"]
        y = m_df["Cumulative ice sheet mass change (Gt)"]
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
            "Year": float,
            "SLE (cm)": float,
            "Cumulative ice sheet mass change (Gt)": float,
            "Cumulative surface mass balance anomaly (Gt)": float,
            "Cumulative ice dynamics anomaly (Gt)": float,
            "Rate of ice sheet mass change (Gt/yr)": float,
            "Rate of surface mass balance anomaly (Gt/yr)": float,
            "Rate of ice dynamics anomaly (Gt/yr)": float,
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

    plot_prognostic(f"{d}_prognostic.pdf", df, as19)
    plot_sle_pdfs(f"{d}_sle_pdf_2100.pdf", df[df["Year"] == 2100], as19[(as19["Year"] == 2100) & (as19["RCP"] != 45)])
    plot_historical_partitioning_cumulative(f"{d}_historical_partitioning_cumulative.pdf", df, mou19)
    plot_historical_partitioning(f"{d}_historical_partitioning.pdf", df, mou19, man)
    plot_historical(f"{d}_historical.pdf", df, grace, mou19, imbie, model_trends)
    plot_trends(f"{d}_trends.pdf", df)

    def iqr(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        return IQR

    f = as19[as19["Year"] == 2100]
    f.groupby(by=["RCP"]).apply(iqr, "SLE (cm)")

    df[df["Year"] == 2100].groupby(by=["RCP"]).apply(iqr, "SLE (cm)")
