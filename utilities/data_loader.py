import numpy as np
import pandas as pd
import os
import requests
import netrc
import getpass
from urllib.parse import urlparse
from urllib.request import HTTPError
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
from glob import glob
from netCDF4 import Dataset as NC
from pathlib import Path

from .helper import hist_start, hist_end, proj_start, proj_end, proj_time, secpera

URS_URL = "https://urs.earthdata.nasa.gov"


def get_username():
    username = ""

    # For Python 2/3 compatibility:
    try:
        do_input = raw_input  # noqa
    except NameError:
        do_input = input

    while not username:
        try:
            username = do_input("Earthdata username: ")
        except KeyboardInterrupt:
            quit()
    return username


def get_password():
    password = ""
    while not password:
        try:
            password = getpass.getpass("password: ")
        except KeyboardInterrupt:
            quit()
    return password


def get_credentials():
    """Get user credentials from .netrc or prompt for input."""
    credentials = None
    errprefix = ""
    try:
        info = netrc.netrc()
        username, account, password = info.authenticators(urlparse(URS_URL).hostname)
        errprefix = "netrc error: "
    except Exception as e:
        if not ("No such file" in str(e)):
            print("netrc error: {0}".format(str(e)))
        username = None
        password = None

    if not username:
        username = get_username()
        password = get_password()

    return (username, password)


def load_imbie():
    """
    Loading the IMBIE Greenland data set downloaded from
    http://imbie.org/wp-content/uploads/2012/11/imbie_dataset_greenland_dynamics-2020_02_28.xlsx

    """
    imbie_df = pd.read_excel(
        "http://imbie.org/wp-content/uploads/2012/11/imbie_dataset_greenland_dynamics-2020_02_28.xlsx",
        sheet_name="Greenland Ice Mass",
        engine="openpyxl",
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

    return imbie


def load_mouginot():
    """
    Load the Mouginot et al (2019) data set
    """
    mou19_df = pd.read_excel(
        "https://www.pnas.org/highwire/filestream/860129/field_highwire_adjunct_files/2/pnas.1904242116.sd02.xlsx",
        sheet_name="(2) MB_GIS",
        header=8,
        usecols="B,AR:BJ",
        engine="openpyxl",
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

    return mou19


def load_grace():
    grace = pd.read_csv(
        "https://podaac-tools.jpl.nasa.gov/drive/files/allData/tellus/L4/ice_mass/RL06/v02/mascon_CRI/greenland_mass_200204_202101.txt",
        header=30,
        delim_whitespace=True,
        skipinitialspace=True,
        names=["Year", "Cumulative ice sheet mass change (Gt)", "Cumulative ice sheet mass change uncertainty (Gt)"],
    )
    # Normalize GRACE signal to the starting date of the projection
    grace["Cumulative ice sheet mass change (Gt)"] -= np.interp(
        proj_start, grace["Year"], grace["Cumulative ice sheet mass change (Gt)"]
    )

    return grace


def load_ismip6():
    outpath = "."
    url = "https://zenodo.org/record/3939037/files/v7_CMIP5_pub.zip"

    with urlopen(url) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(outpath)

    ismip6_filename = "ismip6_gris_ctrl_removed.csv.gz"
    if os.path.isfile(ismip6_filename):
        df = pd.read_csv(ismip6_filename)
    else:
        print(f"{ismip6_filename} not found locally. Downloading.")
        ismip6_to_csv(basedir="v7_CMIP5_pub", ismip6_filename)
        df = pd.read_csv(ismip6_filename)
    return df


def ismip6_to_csv(basedir, ismip6_filename):
    # Now read model output from each of the ISMIP6 files. The information we
    # need is in the file names, not the metadate so this is no fun.
    # Approach is to read each dataset into a dataframe, then concatenate all
    #   dataframes into one Arch dataframe that contains all model runs.
    # Resulting dataframe consists of both historical and projected changes

    ctrl_files = []
    for path in Path(basedir).rglob("*_mm_*_ctrl_proj.nc"):
        ctrl_files.append(path)

    hist_files = []
    for path in Path(basedir).rglob("*_mm_*_historical.nc"):
        hist_files.append(path)

    files = []
    dfs = []
    for path in Path(basedir).rglob("*_mm_cr_*.nc"):
        files.append(path)
        # Experiment
        nc = NC(path)
        exp_sle = nc.variables["sle"][:]
        # For comparison with GRACE, we use grounded ice mass, converted to Gt
        exp_mass = nc.variables["limgr"][:] / 1e12
        exp_smb = nc.variables["smb"][:] / 1e12 * secpera

        f = path.name.split(f"scalars_mm_cr_GIS_")[-1].split(".nc")[0].split("_")
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
    df.to_csv(ismip6_filename, compression="gzip")
