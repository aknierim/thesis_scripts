import datetime
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from matplotlib.colors import SymLogNorm
from tqdm import tqdm

from ma_scripts.utils import Layout
from pyvisgen.fits import writer
from pyvisgen.simulation.observation import Observation
from pyvisgen.simulation.visibility import vis_loop
from radiotools.measurements import Measurement


def _rmtree(root):
    for p in root.iterdir():
        if p.is_dir():
            _rmtree(p)
        else:
            p.unlink()

    root.rmdir()


class CreateMeasurement:
    def __init__(
        self,
        fits_file: str,
        start: int,
        end: int,
        step: int,
        mode: str = "delta",
        polarisation: str = "circular",
        fov_correction=None,
        batch_size=100,
    ):
        torch._dynamo.config.suppress_errors = True
        torch._logging.set_logs(
            dynamo=logging.CRITICAL, aot=logging.CRITICAL, inductor=logging.CRITICAL
        )

        self.start = start
        self.end = end
        self.step = step
        self.mode = mode
        self.polarisation = polarisation
        self.batch_size = batch_size

        hdu = fits.open(fits_file)
        hdr = hdu[0].header

        sky = hdu[0].data[0, 0, ...]
        self.sky = torch.from_numpy(sky.astype(np.float64))
        self.sky = self.sky[np.newaxis, ...]

        self.src_ra = hdr["OBSRA"]
        self.src_dec = hdr["OBSDEC"]
        self.obs_time = datetime.datetime.strptime(hdr["DATE-OBS"], "%Y-%m-%d")
        self.obs_freq = hdr["CRVAL3"]

        self.layout_name = hdu[0].header["TELESCOP"].lower()
        self.layout = Layout.from_pyvisgen(
            f"~/MA/pyvisgen/pyvisgen/layouts/{self.layout_name}.txt"
        )

        res = self.layout.get_max_resolution(self.obs_freq)
        self.fov = res * sky.shape[0]

        if fov_correction:
            self.fov *= fov_correction

    def plot_sky(self):
        fig, ax = plt.subplots()

        im = ax.imshow(self.sky[0, ...], origin="lower", norm=SymLogNorm(0.005))
        fig.colorbar(im, ax=ax)

    def visibilities(self, delta, amp_ratio):
        obs = Observation(
            src_ra=self.src_ra,
            src_dec=self.src_dec,
            start_time=self.obs_time,
            scan_duration=60,
            num_scans=200,
            scan_separation=1600,
            integration_time=15,
            ref_frequency=self.obs_freq,
            frequency_offsets=[0],
            bandwidths=[64e8],
            fov=self.fov,
            image_size=self.sky.shape[-1],
            array_layout=self.layout_name,
            corrupted=False,
            device="cuda:0",
            dense=False,
            sensitivity_cut=1e-6,
            polarisation=self.polarisation,
            pol_kwargs={
                "delta": delta,
                "amp_ratio": amp_ratio,
                "random_state": 42,
            },
            field_kwargs={"scale": [self.sky.min().item(), self.sky.max().item()]},
        )

        vis = vis_loop(
            obs=obs,
            SI=self.sky,
            mode="grid",
            noisy=0,
            show_progress=True,
            batch_size=self.batch_size,
        )
        self.vis_list.append(vis)
        self.obs_list.append(obs)

        BASE = "/scratch/aknierim/MA"

        for s in tqdm(["I", "Q", "U", "V"]):
            fits_path = (
                BASE + f"/fits/{self.polarisation}.{s}.{self.mode}_d_{delta}.fits"
            )
            ms_path = (
                BASE
                + f"/measurement_sets/{self.polarisation}.{s}.{self.mode}_d_{delta}.ms"
            )

            if Path(fits_path).is_file():
                Path(fits_path).unlink()
                print(f"Removing {fits_path}...")

            if Path(ms_path).is_dir():
                _rmtree(Path(ms_path))
                print(f"Removing {ms_path}...")

            hdu_list = writer.create_hdu_list(vis, obs)
            hdu_list.writeto(fits_path, overwrite=True)

            ms = Measurement.from_fits(fits_path)
            ms.save_as_ms(ms_path)

    def create(self):
        self.vis_list = []
        self.obs_list = []

        if self.mode == "delta":
            for delta in np.arange(self.start, self.end, self.step):
                self.visibilities(delta=delta, amp_ratio=0.5)
        elif self.mode == "amp":
            for amp in np.arange(self.start, self.end, self.step):
                self.visibilities(delta=45, amp_ratio=amp)

        return self.vis_list, self.obs_list
