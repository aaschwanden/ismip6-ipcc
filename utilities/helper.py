import numpy as np

secpera = 3.15569259747e7
hist_start = 2008
hist_end = 2014
proj_start = hist_end + 1
proj_end = 2100
proj_time = np.arange(proj_start, proj_end + 1)

ais_exp_dict = {
    "exp01": "open",
    "exp02": "open",
    "exp03": "open",
    "exp04": "open",
    "exp05": "std",
    "exp06": "std",
    "exp07": "std",
    "exp08": "std",
    "exp09": "std",
    "exp10": "std",
    "exp11": "open",
    "exp12": "std",
    "exp13": "std",
    "expA1": "open",
    "expA2": "open",
    "expA3": "open",
    "expA4": "open",
    "expA5": "std",
    "expA6": "std",
    "expA7": "std",
    "expA8": "std",
}
