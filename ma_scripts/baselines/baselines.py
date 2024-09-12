import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from astropy.coordinates import EarthLocation
import astropy.units as u

from pyvisgen.layouts.layouts import get_array_layout


def plot_baselines(
    layout: str,
    save_path: str=None,
    projection_thresh: float=20,
    earth_lon: float=-80,
    earth_lat: float=20,
    station_color: str="#eb6359",
    baselines_color="#111e30",
    fig=None,
    ax=None,
) -> tuple:
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
        transform=ccrs.Geodetic()
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
                    transform=ccrs.Geodetic()
                )
    
    ax.set_global()

    if save_path:
        fig.savefig(save_path)

    return fig, ax
        