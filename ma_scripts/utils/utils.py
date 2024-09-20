import inspect

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from colorspacious import cspace_converter
from colour import Color
from matplotlib.colors import LinearSegmentedColormap


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


def shifted_colormap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Parameters
    ----------
    cmap : The matplotlib colormap to be altered
    start : float
        Offset from lowest point in the colormap's range.
        Defaults to 0.0 (no lower offset). Should be between
        0.0 and `midpoint`.
    midpoint : float
        The new center of the colormap. Defaults to
        0.5 (no shift). Should be between 0.0 and 1.0. In
        general, this should be  1 - vmax / (vmax + abs(vmin))
    stop : float
        Offset from highest point in the colormap's range.
        Defaults to 1.0 (no upper offset). Should be between
        `midpoint` and 1.0.

    Returns
    -------
    newcmap : matplotlib.cm
        Shifted colormap.
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)

    return newcmap


def custom_cmap(color_list: list) -> LinearSegmentedColormap:
    """Creates a colormap to a given list of colors.
    Also provides a preview of the colormap via plt.imshow().

    Parameters:
    -----------
    ramp_colors: list
        List of hexadecimal color codes with either the
        leading number sign (#) or no leading number sign
        at all.

    Returns:
    --------
    cmap: LinearSegmentedColormap
        A colormap containing the given colors.
    """

    list_lengths = {len(c) for c in color_list}

    if next(iter(list_lengths)) == 6 and len(list_lengths) == 1:
        color_list = ["#" + c for c in color_list]
    elif len(list_lengths) != 1:
        raise ValueError(
            "Please a hexadecimal color code format with EITHER the"
            " leading number sign or no leading number sign at all."
        )

    cmap = LinearSegmentedColormap.from_list(
        "my_list", [Color(c1).rgb for c1 in color_list]
    )

    _cmap = np.repeat(np.arange(0, len(color_list), 0.01)[None, ...], 10, axis=0)

    fig, ax = plt.subplots(figsize=(15, 1))
    ax.imshow(_cmap, cmap=cmap, interpolation="nearest", origin="lower")
    ax.set(xticks=[], yticks=[])

    return cmap


def test_cmap(cmap) -> None:
    x = np.linspace(0.0, 1.0, 1000)

    fig, ax = plt.subplots()

    rgb = cmap(x)[np.newaxis, :, :3]
    lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)

    y_ = lab[0, :, 0]
    c_ = x

    ax.scatter(x, y_, c=c_, cmap=cmap, s=300, linewidths=0.0)
    ax.set_ylabel("Lightness $L^*$", fontsize=12)
    ax.set_xticks([])


def pvar(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    var_name = str([k for k, v in callers_local_vars if v is var][0])
    print(f"{var_name}:\t{var}\n")


def pshape(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    var_name = str([k for k, v in callers_local_vars if v is var][0])
    print(f"{var_name}.shape:\t{var.shape}\n")


def rmtree(root):
    for p in root.iterdir():
        if p.is_dir():
            rmtree(p)
        else:
            p.unlink()

    root.rmdir()
