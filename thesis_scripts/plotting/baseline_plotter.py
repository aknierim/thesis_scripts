from dataclasses import dataclass, fields
from pathlib import Path

import astropy.units as un
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from astropy.coordinates import ITRS, EarthLocation, SkyCoord
from astropy.time import Time
from cartopy.feature.nightshade import Nightshade
from tqdm.notebook import tqdm

from pyvisgen.layouts.layouts import get_array_layout
from pyvisgen.simulation.observation import Observation
from pyvisgen.simulation.visibility import vis_loop
from thesis_scripts.baselines import plot_baselines


@dataclass
class SimData:
    uu: torch.tensor
    vv: torch.tensor
    src_lon: torch.tensor
    src_lat: torch.tensor
    obstimes: torch.tensor

    def __getitem__(self, i):
        return SimData(*[getattr(self, f.name)[i] for f in fields(self)])


class BaselinePlotter:
    @classmethod
    def from_params(
        cls,
        array_layout: str | Path | pd.DataFrame,
        source: str,
        obs_start: str,
        obs_length: int,
        frames: int,
        *,
        num_scans: int = 40,
        scan_sep: int = 120,
        int_time: int = 15,
        ref_freq: float = 15239875000.0,
        frequency_offsets=[0],
        bandwidths=[32000000],
        fov: float = 0.3,
        image_size: int = 512,
        corrupted: bool = True,
        device: str = "cuda",
        sensitivity_cut: float = 1e-6,
        sky_image: torch.tensor = None,
        subplots: str = "EVS",
        start_end_only: bool = False,
    ) -> None:
        cls = cls()
        cls.array_layout = array_layout
        cls.source = source
        cls.obs_start = obs_start
        cls.obs_length = obs_length
        cls.frames = frames
        cls.sky_image = sky_image
        cls.subplots = subplots
        cls.start_end_only = start_end_only

        layout = get_array_layout(array_layout, writer=True)
        cls.stations = cls._get_stations(layout)
        cls.n_base = cls._num_base(layout)

        cls.src_crd = cls._get_source()

        cls.obs_kwargs = {
            "src_ra": cls.src_crd.ra,
            "src_dec": cls.src_crd.dec,
            "start_time": Time(obs_start).datetime,
            "scan_duration": obs_length * 60,
            "num_scans": num_scans,
            "scan_separation": scan_sep,
            "integration_time": int_time,
            "ref_frequency": ref_freq,
            "frequency_offsets": frequency_offsets,
            "bandwidths": bandwidths,
            "fov": fov,
            "image_size": image_size,
            "array_layout": array_layout,
            "corrupted": corrupted,
            "device": device,
            "dense": False,
            "sensitivity_cut": sensitivity_cut,
        }

        cls.obs = cls._observe()

        cls.init_plot()

        return cls

    def __call__(self):
        print(f"""
            Plotting
            {50 * '='}
            Observation
            -----------
            Observatory: {self.array_layout:>37}
            Target: {self.source:>42}
            Coordinates:
                RA:  {self.src_crd.ra:>37}
                DEC: {self.src_crd.dec:>37}
            Observation start: {self.obstimes[0].iso:>31}
            Observation end: {self.obstimes[-1].iso:>33}
            Observation length: {self.obs_length:>30}
            Number of baselines: {self.n_base:> 29}
            {50 * '-'}

            Animation/Plotting
            ------------------
            Number of frames: {self.frames:>32}
            Frame skip: {self.frame_skip:>36}
            xlim: {self.xlim}
            ylim: {self.ylim}
            {50 * '-'}
        """)

    def _get_stations(self, layout):
        return EarthLocation.from_geocentric(
            layout["X"], layout["Y"], layout["Z"], unit=un.m
        ).to_geodetic()

    def _get_source(self):
        if isinstance(self.source, str):
            src_crd = SkyCoord.from_name(self.source)
        elif isinstance(self.source, tuple):
            src_crd = SkyCoord(*self.source, unit="deg")
        else:
            raise TypeError("'source' has to be of type str or tuple!")

        return src_crd

    def _num_base(self, stations):
        return len(stations) ** 2 - len(stations)

    def _observe(self):
        obs = Observation(**self.obs_kwargs)
        self.num_base = obs.num_baselines * 2

        if (self.sky_image) and ("S" in self.subplots):
            match self.sky_image.shape[0]:
                case 512:
                    batch_size = 1000
                case 1024:
                    batch_size = 250
                case 2048:
                    batch_size = 40
                case _:
                    batch_size = 1

            self.vis = vis_loop(
                obs=obs,
                SI=self.sky_image,
                mode="full",
                noisy=0,
                show_progress=True,
                batch_size=batch_size,
            )

            uu = self.vis.u
            vv = self.vis.v
            obstimes = Time(self.vis.date.cpu(), format="jd")

        else:
            # valid_subset = obs.baselines.get_valid_subset(obs.num_baselines, obs.device)
            # obstimes = Time(valid_subset.date.cpu(), format="jd")
            #
            # uu = valid_subset.u_valid
            # vv = valid_subset.v_valid
            bas = obs.baselines
            obstimes = Time(bas.time / (60 * 60 * 24), format="mjd")
            self.valid = bas.valid.reshape(-1, self.num_base).bool()
            uu = bas.u
            vv = bas.v

        src_crd = self.src_crd.transform_to(frame=ITRS(obstime=obstimes))

        return SimData(
            uu,
            vv,
            src_crd.spherical.lon,
            src_crd.spherical.lat,
            obstimes,
        )

    def init_plot(self, sampled_color: str = "#111e30") -> None:
        self.sampled_color = sampled_color
        self.obstimes = self.obs.obstimes[:: self.num_base]
        self.frame_skip = int(self.obstimes.shape[0] / self.frames)
        per_subplot_kw = {}

        if "E" in self.subplots:
            projection = ccrs.NearsidePerspective(
                central_longitude=self.obs.src_lon[0],
                central_latitude=self.obs.src_lat[0],
                satellite_height=1e10,
            )

            per_subplot_kw.update({"E": {"projection": projection}})

        if "V" in self.subplots:
            per_subplot_kw.update({"V": {"box_aspect": 1}})

        if "S" in self.subplots:
            per_subplot_kw.update({"S": {"box_aspect": 1}})

        self.fig, self.ax = plt.subplot_mosaic(
            self.subplots,
            figsize=(len(self.subplots) * 5.1, 5),
            per_subplot_kw=per_subplot_kw,
            layout="constrained",
        )

        if "E" in self.subplots:
            plot_baselines(
                "vlba",
                projection_thresh=20,
                earth_lon=self.obs.src_lon[0],
                earth_lat=self.obs.src_lat[0],
                fig=self.fig,
                ax=self.ax["E"],
            )
            self.ax["E"].scatter(
                self.obs.src_lon[0],
                self.obs.src_lat[0],
                transform=ccrs.Geodetic(),
                color="#fbbe23",
                zorder=10,
                label="Projected source position",
            )

            self.ax["E"].add_feature(Nightshade(self.obstimes[0].datetime, alpha=0.2))

            self.ax["E"].set_title(
                f"{self.obstimes[0].iso} (UTC)",
                fontfamily="monospace",
                fontname="Fira Mono",
            )

            self.ax["E"].legend(loc="upper left", bbox_to_anchor=(0, -0.02))

        if "V" in self.subplots:
            self.u = self.obs.uu.cpu().reshape(-1, self.num_base)
            self.v = self.obs.vv.cpu().reshape(-1, self.num_base)
            self.u[~self.valid] = np.NaN
            self.v[~self.valid] = np.NaN

            self.scat1 = self.ax["V"].scatter(
                self.u[0],
                self.v[0],
                s=10,
                c=self.sampled_color,
                label="Sampled $(u,v)$ data",
            )
            self.scat2 = self.ax["V"].scatter(
                self.u[0],
                self.v[0],
                s=10,
                c="#94ab71",
                label="Current $(u,v)$ data",
            )
            self.scat3 = self.ax["V"].scatter(
                -self.u[0], -self.v[0], s=10, c=self.sampled_color
            )
            self.scat4 = self.ax["V"].scatter(-self.u[0], -self.v[0], s=10, c="#94ab71")

            xlim = np.abs(self.u.min() + 0.1 * self.u.min())
            ylim = np.abs(self.v.min() + 0.1 * self.v.min())

            self.ax["V"].set(
                xlabel=r"$u \:/\: \lambda$",
                ylabel=r"$v \:/\: \lambda$",
                xlim=(-xlim, xlim),
                ylim=(-ylim, ylim),
            )

            self.ax["V"].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0))

        if (self.sky_image) and ("S" in self.subplots):
            self.ax["S"].imshow()

        self.fig.suptitle(str(self.source))

    def _update(self, i):
        if "E" in self.subplots:
            self.ax["E"].cla()
            self.ax["E"].set_title(
                f"{self.obs.obstimes[i].iso} (UTC)",
                fontfamily="monospace",
                fontname="Fira Mono",
            )
            self.ax["E"].projection = ccrs.NearsidePerspective(
                central_longitude=self.obs.src_lon[i],
                central_latitude=self.obs.src_lat[i],
                satellite_height=1e10,
            )

            plot_baselines(
                self.array_layout,
                projection_thresh=20,
                earth_lon=self.obs.src_lon[i],
                earth_lat=self.obs.src_lat[i],
                fig=self.fig,
                ax=self.ax["E"],
            )
            self.ax["E"].add_feature(
                Nightshade(self.obs.obstimes[i].datetime, alpha=0.2)
            )
            self.ax["E"].scatter(
                self.obs.src_lon[i],
                self.obs.src_lat[i],
                transform=ccrs.Geodetic(),
                color="#fbbe23",
                zorder=10,
                label="Projected source position",
            )
            self.ax["E"].legend(loc="upper left", bbox_to_anchor=(0, -0.02))

        if "V" in self.subplots:
            if i > 0:
                self.scat1.remove()
                self.scat3.remove()
                self.scat1 = self.ax["V"].scatter(
                    self.u[: i - 1], self.v[: i - 1], s=10, c=self.sampled_color
                )
                self.scat3 = self.ax["V"].scatter(
                    -self.u[: i - 1], -self.v[: i - 1], s=10, c=self.sampled_color
                )

            self.scat2.remove()
            self.scat4.remove()
            self.scat2 = self.ax["V"].scatter(self.u[i], self.v[i], s=10, c="#94ab71")
            self.scat4 = self.ax["V"].scatter(-self.u[i], -self.v[i], s=10, c="#94ab71")

    def plot(self):
        if not self.start_end_only:
            for i in tqdm(np.arange(1, self.frames, 4)):
                i = i * self.frame_skip
                self._update(i)
                self.fig.savefig(
                    f"test/{str(self.source)}-{int(i / 4)}.png",
                    dpi=150,
                )
        else:
            self._update(0)
            self.fig.savefig(
                f"test/{str(self.source)}-start.png",
                dpi=150,
            )
            self._update(-1)
            self.fig.savefig(
                f"test/{str(self.source)}-end.png",
                dpi=150,
            )
