import inspect
from pathlib import Path

import numpy as np


def img2jansky(image, header):
    return (
        4
        * image
        * np.log(2)
        * np.power(header["CDELT1"], 2)
        / (np.pi * header["BMIN"] * header["BMAJ"])
    )


def get_header_info(hdr):
    keys = ["RA", "DEC", "FREQ"]

    values = {}
    for k, v in hdr.items():
        if v in keys:
            values[v] = hdr[f"CRVAL{k[-1]}"]

    values["DATE-OBS"] = hdr["DATE-OBS"]
    values["TELESCOP"] = hdr["TELESCOP"]
    values["INSTRUME"] = hdr["INSTRUME"]

    return values


def pvar(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    var_name = str([k for k, v in callers_local_vars if v is var][0])
    print(f"{var_name}:\t{var}\n")


def pshape(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    var_name = str([k for k, v in callers_local_vars if v is var][0])
    print(f"{var_name}.shape:\t{var.shape}\n")


def rmtree(root: Path):
    """Recursively remove directories and files
    starting from a root directory.

    Parameters
    ----------
    root : Path
        Root path of the directories you want to delete.
    """
    for p in root.iterdir():
        if p.is_dir():
            rmtree(p)
        else:
            p.unlink()

    root.rmdir()
