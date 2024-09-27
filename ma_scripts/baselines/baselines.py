import astropy.units as u
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from astropy.coordinates import EarthLocation

from pyvisgen.layouts.layouts import get_array_layout


def plot_baselines(
    layout: str,
    save_path: str = None,
    projection_thresh: float = 20,
    earth_lon: float = -80,
    earth_lat: float = 20,
    station_color: str = "#eb6359",
    baselines_color="#111e30",
    fig=None,
    ax=None,
) -> tuple:
    """Plots baselines and stations of a given antenna
    array across the globe.

    Parameters
    ----------
    layout : str
        Layout, i.e. array name.
    save_path : str, optional
        Path to output file.
    projection_thresh : float, optional
        Projection threshold for orthographic projection.
        Default: 20
    earth_lon : float, optional
        Longitude viewing angle. Default: -80
    earth_lat : float, optional
        Latitude viewing angle. Default: 20
    station_color : str, optional
        Color for antenna positions. Default: #eb6359
    baselines_color : str, optional
        Color for baselines. Default: #111e30
    fig : matplotlib.figure.figure
        Matplotlib figure object. Default: None
    ax : matplotlib.axis.Axis
        Matplotlib axis object. Default: None
    """
    layout = get_array_layout(layout, writer=True)

    stations = EarthLocation.from_geocentric(
        layout["X"], layout["Y"], layout["Z"], unit=u.m
    ).to_geodetic()

    if not any((fig, ax)):
        orthographic = ccrs.Orthographic(earth_lon, earth_lat)
        orthographic.threshold = orthographic.threshold * projection_thresh

        fig, ax = plt.subplots(subplot_kw={"projection": orthographic})

    ax.coastlines(zorder=1, linewidth=0.5)
    ax.stock_img()
    ax.plot(
        stations.lon,
        stations.lat,
        marker=".",
        color=station_color,
        ls="none",
        zorder=3,
        transform=ccrs.Geodetic(),
    )

    for i in range(len(stations.lon)):
        for j in range(len(stations.lat)):
            if i <= j:
                ax.plot(
                    [stations.lon[i].value, stations.lon[j].value],
                    [stations.lat[i].value, stations.lat[j].value],
                    ls="-",
                    color=baselines_color,
                    linewidth=1,
                    zorder=2,
                    transform=ccrs.Geodetic(),
                )

    ax.set_global()

    if save_path:
        fig.savefig(save_path)

    return fig, ax
