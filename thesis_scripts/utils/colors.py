import matplotlib.pyplot as plt
import numpy as np
from colorspacious import cspace_converter
from colour import Color
from matplotlib.colors import LinearSegmentedColormap


def shifted_colormap(
    cmap: LinearSegmentedColormap,
    start: float = 0.0,
    midpoint: float = 0.5,
    stop: float = 1.0,
    name="shiftedcmap",
) -> LinearSegmentedColormap:
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Parameters
    ----------
    cmap : LinearSegmentedColormap
        The matplotlib colormap to be altered
    start : float, optional
        Offset from lowest point in the colormap's range.
        Defaults to 0.0 (no lower offset). Should be between
        0.0 and `midpoint`. Default: 0.0
    midpoint : float, optional
        The new center of the colormap. Defaults to
        0.5 (no shift). Should be between 0.0 and 1.0. In
        general, this should be  1 - vmax / (vmax + abs(vmin))
        Default: 0.5
    stop : float, optional
        Offset from highest point in the colormap's range.
        Defaults to 1.0 (no upper offset). Should be between
        `midpoint` and 1.0. Default: 1.0
    name : str, optional
        Name for the shifted colormap. Default: 'shiftedcmap'

    Returns
    -------
    cmap : LinearSegmentedColormap
        Shifted colormap.
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    index = np.linspace(start, stop, 257)

    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for i, si in zip(index, shift_index):
        r, g, b, a = cmap(i)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    cmap = LinearSegmentedColormap(name, cdict)

    return cmap


def custom_cmap(color_list: list) -> LinearSegmentedColormap:
    """Creates a colormap to a given list of colors.
    Also provides a preview of the colormap via plt.imshow().

    Parameters
    ----------
    color_list : list
        List of hexadecimal color codes with either the
        leading number sign (#) or no leading number sign
        at all.

    Returns
    -------
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


def test_cmap(cmap: LinearSegmentedColormap) -> None:
    x = np.linspace(0.0, 1.0, 1000)

    fig, ax = plt.subplots()

    rgb = cmap(x)[np.newaxis, :, :3]
    lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)

    y_ = lab[0, :, 0]
    c_ = x

    ax.scatter(x, y_, c=c_, cmap=cmap, s=300, linewidths=0.0)
    ax.set_ylabel("Lightness $L^*$", fontsize=12)
    ax.set_xticks([])
