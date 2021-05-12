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
    xmin = np.floor(df[df["Year"] >= proj_start]["SLE (cm)"].min())
    xmax = np.ceil(df[df["Year"] >= proj_start]["SLE (cm)"].max())

    # Dataframe with Year 2100 only for histogram
    p_df = df[(df["Year"] == 2100) & (df["Meet_Threshold"] == True)]
    f_df = df[(df["Year"] == 2100) & (df["Meet_Threshold"] == False)]

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

        return ax[0].plot(g[-1]["Year"], g[-1]["SLE (cm)"], color=signal_color, linewidth=0.5)

    # Plot each model response by grouping
    [plot_signal(g) for g in df.groupby(by=["Group", "Model", "Exp"])]

    ## Boxplot
    sns.boxplot(
        data=df[df["Year"] == 2100],
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

        return ax[0].plot(g[-1]["Year"], g[-1]["SLE (cm)"], color=signal_color, linewidth=0.1)

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
        median = m_df.groupby(by=["Year"]).quantile(0.50)
        pctl5 = m_df.groupby(by=["Year"]).quantile(0.05)
        pctl25 = m_df.groupby(by=["Year"]).quantile(0.25)
        pctl75 = m_df.groupby(by=["Year"]).quantile(0.75)
        pctl95 = m_df.groupby(by=["Year"]).quantile(0.95)
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


def plot_historical(out_filename, df, df_ctrl, grace):
    """
    Plot historical simulations and observations.
    """

    def plot_signal(g):
        m_df = g[-1]
        x = m_df["Year"]
        y = m_df["Mass (Gt)"]

        return ax.plot(x, y, color=simulated_signal_color, linewidth=simulated_signal_lw)

    xmin = 2005
    xmax = 2025
    ymin = -20000
    ymax = 1000

    fig = plt.figure(num="historical", clear=True, figsize=[6.2, 2.5])
    ax = fig.add_subplot(111)

    # [plot_signal(g) for g in df.groupby(by=["Experiment"])]

    as19_median = df.drop(columns=["interval"]).groupby(by="Year").quantile(0.50)
    as19_std = df.groupby(by="Year").std()
    as19_low = df.drop(columns=["interval"]).groupby(by="Year").quantile(0.05)
    as19_high = df.drop(columns=["interval"]).groupby(by="Year").quantile(0.95)

    as19_ctrl_median = df_ctrl.groupby(by="Year").quantile(0.50)

    as19_ci = ax.fill_between(
        as19_median.index,
        as19_low["Cumulative ice sheet mass change (Gt)"],
        as19_high["Cumulative ice sheet mass change (Gt)"],
        color="0.5",
        alpha=0.50,
        linewidth=0.0,
        zorder=10,
        label="AS19 90% c.i.",
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

    l_es_median = ax.plot(
        as19_median.index,
        as19_median["Cumulative ice sheet mass change (Gt)"],
        color="k",
        linewidth=grace_signal_lw,
        label="Median(Ensemble)",
    )
    l_ctrl_median = ax.plot(
        as19_ctrl_median.index,
        as19_ctrl_median["Cumulative ice sheet mass change (Gt)"],
        color="k",
        linewidth=grace_signal_lw,
        linestyle="dotted",
        label="Median(CTRL)",
    )

    grace_line = ax.plot(
        grace["Year"],
        grace["Cumulative ice sheet mass change (Gt)"],
        "-",
        color=grace_signal_color,
        linewidth=grace_signal_lw,
        label="Observed (GRACE)",
    )
    ax.axhline(0, color="k", linestyle="dotted", linewidth=grace_signal_lw)

    model_line = mlines.Line2D(
        [], [], color=simulated_signal_color, linewidth=simulated_signal_lw, label="Simulated (AS19)"
    )

    legend = ax.legend(handles=[grace_line[0], l_ctrl_median[0], l_es_median[0], as19_ci], loc="lower left")
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    ax.set_xlabel("Year")
    ax.set_ylabel(f"Cumulative mass change\nsince {proj_start} (Gt)")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax_sle = ax.twinx()
    ax_sle.set_ylabel(f"Contribution to sea-level \nsince {proj_start} (cm SLE)")
    ax_sle.set_ylim(-ymin * gt2cmSLE, -ymax * gt2cmSLE)

    fig.savefig(out_filename, bbox_inches="tight")


def plot_projection(out_filename, df, df_ctrl, grace):
    """
    Plot historical simulations and observations.
    """

    def plot_signal(g):
        m_df = g[-1]
        x = m_df["Year"]
        y = m_df["SLE (cm)"]

        return ax[0].plot(x, y, color=simulated_signal_color, linewidth=simulated_signal_lw)

    xmin = 2020
    xmax = 2100
    ymin = 0
    ymax = 40

    fig, ax = plt.subplots(
        1,
        2,
        sharey="col",
        figsize=[6.2, 2.5],
        num="prognostic_all",
        clear=True,
        gridspec_kw=dict(width_ratios=[20, 1.5]),
    )
    fig.subplots_adjust(wspace=0.025)

    # [plot_signal(g) for g in df.groupby(by=["Experiment"])]

    as19_median = df.drop(columns=["interval"]).groupby(by="Year").quantile(0.50)
    as19_std = df.groupby(by="Year").std()
    as19_low = df.drop(columns=["interval"]).groupby(by="Year").quantile(0.05)
    as19_high = df.drop(columns=["interval"]).groupby(by="Year").quantile(0.95)

    as19_ctrl_median = df_ctrl.groupby(by="Year").quantile(0.50)

    as19_ci = ax[0].fill_between(
        as19_median.index,
        as19_low["SLE (cm)"],
        as19_high["SLE (cm)"],
        color="0.5",
        alpha=0.50,
        linewidth=0.0,
        zorder=10,
        label="AS19 90% c.i.",
    )

    l_es_median = ax[0].plot(
        as19_median.index,
        as19_median["SLE (cm)"],
        color="k",
        linewidth=grace_signal_lw,
        label="Median(Ensemble)",
    )
    l_ctrl_median = ax[0].plot(
        as19_ctrl_median.index,
        as19_ctrl_median["SLE (cm)"],
        color="k",
        linewidth=grace_signal_lw,
        linestyle="dotted",
        label="Median(CTRL)",
    )

    def plot_signal(g):
        m_df = g[-1]
        x = m_df["Year"]
        y = m_df["SLE (cm)"]

        return ax[0].plot(x, y, color=simulated_signal_color, linewidth=simulated_signal_lw)

    cmap = sns.color_palette("mako_r", n_colors=4)
    l_bs = []
    for beta in [3, 2, 1]:
        m_df = df[df[f"Mass Trend {beta}-sigma (Gt/yr)"] == True]
        median = m_df.groupby(by="Year").median()
        l = ax[0].plot(
            median.index,
            median["SLE (cm)"],
            color=cmap[beta],
            linewidth=grace_signal_lw,
            label=f"{beta}-sigma",
        )
        l_bs.append(l[0])

    model_line = mlines.Line2D(
        [], [], color=simulated_signal_color, linewidth=simulated_signal_lw, label="Simulated (AS19)"
    )

    legend = ax[0].legend(handles=[l_ctrl_median[0], l_es_median[0], as19_ci], loc="upper left")
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    legend_2 = ax[0].legend(handles=l_bs, title="Calibrated", loc="upper center")
    legend_2.get_frame().set_linewidth(0.0)
    legend_2.get_frame().set_alpha(0.0)

    ax[0].add_artist(legend)

    ax[0].set_xlabel("Year")
    ax[0].set_ylabel(f"Cumulative mass change\nsince {proj_start} (Gt)")

    ax[0].set_xlim(xmin, xmax)
    ax[0].set_ylim(ymin, ymax)

    df_2100 = df[df["Year"] == 2100]
    sns.kdeplot(
        data=df_2100,
        #        y="SLE (cm)",
        y="SLE (cm)",
        ax=ax[1],
        color="k",
    )
    for beta in [3, 2, 1]:
        sns.kdeplot(
            data=df_2100[df_2100[f"Mass Trend {beta}-sigma (Gt/yr)"] == True],
            y="SLE (cm)",
            ax=ax[1],
            color=cmap[beta],
        )

    sns.despine(ax=ax[1], left=True, bottom=True)
    try:
        ax[1].get_legend().remove()
    except:
        pass
    ax[1].set_xlabel(None)
    ax[1].set_ylabel(None)
    ax[1].axes.xaxis.set_visible(False)
    ax[1].axes.yaxis.set_visible(False)

    fig.savefig(out_filename, bbox_inches="tight")


# def plot_projection(out_filename, df, df_ctrl, grace):
#     """
#     Plot historical simulations and observations.
#     """

#     def plot_signal(g, rcp):
#         m_df = g[-1]
#         x = m_df["Year"]
#         y = m_df["Mass (Gt)"]

#         return ax.plot(x, y, color="0.6", alpha=0.3, linewidth=simulated_signal_lw)

#     def plot_ctrl(g, rcp, ls):
#         m_df = g[-1]
#         x = m_df["Year"]
#         y = m_df["Cumulative ice sheet mass change (Gt)"]

#         return ax.plot(x, y, color=rcp_col_dict[rcp], alpha=1.0, linewidth=grace_signal_lw, linestyle=ls)

#     xmin = 2005
#     xmax = 2025
#     ymin = -150000
#     ymin = -12000
#     ymax = 1000

#     fig = plt.figure(num="historical", clear=True)
#     ax = fig.add_subplot(111)

#     for rcp in [26, 45, 85]:
#         pdf = df[df["RCP"] == rcp]
#         q_16 = pdf.drop(columns=["Experiment"]).groupby(by=["Year"]).quantile(0.16).reset_index()
#         q_84 = pdf.drop(columns=["Experiment"]).groupby(by=["Year"]).quantile(0.84).reset_index()
#         q_50 = pdf.drop(columns=["Experiment"]).groupby(by=["Year"]).quantile(0.50).reset_index()
#         ax.fill_between(
#             q_16["Year"],
#             q_16["Cumulative ice sheet mass change (Gt)"],
#             q_84["Cumulative ice sheet mass change (Gt)"],
#             color=rcp_shade_col_dict[rcp],
#             alpha=0.5,
#             linewidth=1.0,
#         )
#         pdf_ctrl = df_ctrl[df_ctrl["RCP"] == rcp]
#         [plot_ctrl(g, rcp, "dashed") for g in pdf_ctrl.groupby(by=["Experiment"])]
#         x = q_50["Year"]
#         y = q_50["Cumulative ice sheet mass change (Gt)"]
#         ax.plot(x, y, color=rcp_col_dict[rcp], alpha=1.0, linewidth=grace_signal_lw, linestyle="solid")

#     as19_mean = df.groupby(by="Year").mean()
#     as19_std = df.groupby(by="Year").std()
#     as19_low = df.groupby(by="Year").quantile(0.05)
#     as19_high = df.groupby(by="Year").quantile(0.95)
#     as19_median = df.groupby(by="Year").quantile(0.50)

#     # as19_ci = ax.fill_between(
#     #     as19_mean.index,
#     #     as19_low["Cumulative ice sheet mass change (Gt)"],
#     #     as19_high["Cumulative ice sheet mass change (Gt)"],
#     #     color="0.0",
#     #     alpha=0.30,
#     #     linewidth=0.0,
#     #     zorder=10,
#     #     label="Simulated (AS19) 90% c.i.",
#     # )

#     ax.fill_between(
#         grace["Year"],
#         grace["Cumulative ice sheet mass change (Gt)"]
#         - 1 * grace["Cumulative ice sheet mass change uncertainty (Gt)"],
#         grace["Cumulative ice sheet mass change (Gt)"]
#         + 1 * grace["Cumulative ice sheet mass change uncertainty (Gt)"],
#         color=grace_sigma_color,
#         alpha=0.5,
#         linewidth=0,
#     )
#     grace_line = ax.plot(
#         grace["Year"],
#         grace["Cumulative ice sheet mass change (Gt)"],
#         "-",
#         color=grace_signal_color,
#         linewidth=grace_signal_lw,
#         label="Observed (GRACE)",
#     )
#     ax.axhline(0, color="k", linestyle="dotted", linewidth=grace_signal_lw)

#     model_line = mlines.Line2D(
#         [], [], color=simulated_signal_color, linewidth=simulated_signal_lw, label="Simulated (AS19)"
#     )

#     legend = ax.legend(handles=[grace_line[0]], loc="lower left")
#     legend.get_frame().set_linewidth(0.0)
#     legend.get_frame().set_alpha(0.0)

#     ax.set_xlabel("Year")
#     ax.set_ylabel(f"Cumulative mass change\nsince {proj_start} (Gt)")

#     ax.set_xlim(xmin, xmax)
#     ax.set_ylim(ymin, ymax)
#     ax_sle = ax.twinx()
#     ax_sle.set_ylabel(f"Contribution to sea-level \nsince {proj_start} (cm SLE)")
#     ax_sle.set_ylim(-ymin * gt2cmSLE, -ymax * gt2cmSLE)

#     set_size(5, 2.5)

#     fig.savefig(out_filename, bbox_inches="tight")


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
    ax.text(-330, 2.8, "Grace GRACE trend", rotation=90, fontsize=12)

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


def calculate_trend(df, x_var, y_var, y_units):

    x = x_var
    y = f"{y_var} ({y_units})"
    y_var_trend = f"{y_var} Trend ({y_units}/yr)"
    y_var_sigma = f"{y_var} Trend Error ({y_units}/yr)"
    r_df = df.groupby(by=["RCP", "Experiment"]).apply(trend_f, x, y)
    r_df = r_df.reset_index().rename({0: y_var_trend, 1: y_var_sigma}, axis=1)

    return r_df


def trend_f(df, x_var, y_var):
    m_df = df[(df[x_var] >= calibration_start) & (df[x_var] <= calibration_end)]
    x = m_df[x_var]
    y = m_df[y_var]
    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()
    p = ols.params
    bias = p[0]
    trend = p[1]
    trend_sigma = ols.bse[-1]

    return pd.Series([trend, trend_sigma])


#%% End of plotting function definitions, start of analysis


secpera = 3.15569259747e7

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


grace_signal_lw = 1.0
mouginot_signal_lw = 0.75
imbie_signal_lw = 0.75
simulated_signal_lw = 0.1
grace_signal_color = "#238b45"
grace_sigma_color = "#a1d99b"
mouginot_signal_color = "#a63603"
mouginot_sigma_color = "#fdbe85"
imbie_signal_color = "#005a32"
imbie_sigma_color = "#a1d99b"
simulated_signal_color = "0.7"

gt2cmSLE = 1.0 / 362.5 / 10.0

rcp_list = [26, 85]
rcp_dict = {26: "RCP 2.6", 45: "RCP 4.5", 85: "RCP 8.5"}
rcp_col_dict = {85: "#990002", 45: "#5492CD", 26: "#003466"}
rcp_shade_col_dict = {85: "#F4A582", 45: "#92C5DE", 26: "#4393C3"}
model_ls_dict = {"Model Uncertainty (ISMIP6)": "solid", "Parametric Uncertainty (AS19)": "dashed"}


gt2cmSLE = 1.0 / 362.5 / 10.0


proj_start = 2008
calibration_start = 2010
calibration_end = 2020

# Greenland only though this could easily be extended to Antarctica
domain = {"GIS": "grace/greenland_mass_200204_202102.txt"}

for d, data in domain.items():
    print(f"Analyzing {d}")

    as19 = pd.read_csv("as19//aschwanden_et_al_2019_les_2008_norm.csv.gz")
    as19 = as19[as19["Year"] <= 2100]
    as19["Cumulative ice sheet mass change (Gt)"] = as19["Mass (Gt)"]
    as19["SLE (cm)"] = -as19["Mass (Gt)"] / 362.5 / 10

    grace = pd.read_csv(
        "grace/greenland_mass_200204_202102.txt",
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
    grace_time = (grace["Year"] >= calibration_start) & (grace["Year"] <= calibration_end)
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

    mass_trend = calculate_trend(as19, "Year", "Mass", "Gt")
    mass_trend["interval"] = pd.arrays.IntervalArray.from_arrays(
        mass_trend["Mass Trend (Gt/yr)"] - mass_trend["Mass Trend Error (Gt/yr)"],
        mass_trend["Mass Trend (Gt/yr)"] + mass_trend["Mass Trend Error (Gt/yr)"],
    )

    for beta in [1, 2, 3]:
        # This should NOT be the standard error of the OLS regression:
        mass_trend[f"Mass Trend {beta}-sigma (Gt/yr)"] = mass_trend["interval"].array.overlaps(
            pd.Interval(grace_trend - beta * grace_trend_stderr, grace_trend + beta * grace_trend_stderr)
        )
    as19 = pd.merge(as19, mass_trend, on=["RCP", "Experiment"])

    as19_ctrl = pd.read_csv("as19//aschwanden_et_al_2019_ctrl.csv.gz")
    as19_ctrl["Cumulative ice sheet mass change (Gt)"] = as19_ctrl["Mass (Gt)"]
    as19_ctrl["SLE (cm)"] = -as19_ctrl["Mass (Gt)"] / 362.5 / 10
    as19_ctrl = as19_ctrl[(as19_ctrl["Experiment"] == "CTRL") & (as19_ctrl["Resolution (m)"] == 900)]
    # h_df = df[(df["Year"] >= hist_start) & (df["Year"] <= proj_start)]
    # plot_historical_fluxes(f"{d}_fluxes_historical.pdf", df, mou19)
    plot_historical(f"{d}_PISM_calibration_historical.pdf", as19, as19_ctrl, grace)
    plot_projection(f"{d}_PISM_calibration_projection.pdf", as19, as19_ctrl, grace)
