import datetime
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from matplotlib.colors import SymLogNorm
from tqdm import tqdm

from ma_scripts.utils import Layout, rmtree
from pyvisgen.fits import writer
from pyvisgen.simulation.observation import Observation
from pyvisgen.simulation.visibility import vis_loop
from radiotools.measurements import Measurement


class CreateMeasurement:
    def __init__(
        self,
        fits_file: str,
        start: int,
        end: int,
        step: int,
        mode: str = "delta",
        polarisation: str | None = "circular",
        fov_correction: float | None = None,
        batch_size: int = 100,
        device: str = "cuda",
    ) -> None:
        """Creates a measurement simulation using pyvisgen and saves
        it to a measurement set.

        Parameters
        ----------
        fits : str
            Path to an input FITS file.
        start : int
            Start value for delta or the amplitudes. This will create
            simulations in a loop with step size 'step' until 'end'
            is reached.
        end : int
            End value for delta or the amplitudes.
        step : int
            Step size for delta/amplitudes loop.
        mode : str, optional
            Wether to vary phase delay delta or the amplitudes.
            Available values are 'delta', and 'amp'. Default: 'delta'
        polarisation : str or None, optional
            Type of polarisation to simulate. Available values are
            'circular', 'linear', or None. If None is passed,
            no polarisation is applied. Default: 'circular'.
        fov_correction : float or None, optional
            Optional correction factor, e.g. 1.3 for the fov.
            If None is passed, nothing happens. Default: None
        batch_size : int,  optional
            Number of batches for the simulation. Default: 100
        device : str, optional
            Device to run the simulation on. Default: 'cuda'
        """
        torch._dynamo.config.suppress_errors = True
        torch._logging.set_logs(
            dynamo=logging.CRITICAL, aot=logging.CRITICAL, inductor=logging.CRITICAL
        )
        self.fits_file = fits_file
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

        self.device = device

    def plot_sky(self):
        """Plots the sky image extracted from the FITS file."""
        fig, ax = plt.subplots()

        im = ax.imshow(self.sky[0, ...], origin="lower", norm=SymLogNorm(0.005))
        fig.colorbar(im, ax=ax)

    def _visibilities(self, delta: float, amp_ratio: float):
        """Computes the visibilities using the pyvisgen Observation class
        and the vis_loop function.

        Parameters
        ----------
        delta : float
            Value for phase angle delta.
        amp_ratio : float
            Amplitude ratio.
        """
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
            device=self.device,
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

        for s in tqdm(["I", "Q", "U", "V"]):
            fits_path = (
                self.BASE_PATH
                + f"/fits/{self.polarisation}.{s}.{self.mode}_d_{delta}.fits"
            )
            ms_path = (
                self.BASE_PATH
                + f"/measurement_sets/{self.polarisation}.{s}.{self.mode}_d_{delta}.ms"
            )

            if Path(fits_path).is_file():
                Path(fits_path).unlink()
                print(f"Removing {fits_path}...")

            if Path(ms_path).is_dir():
                rmtree(Path(ms_path))
                print(f"Removing {ms_path}...")

            hdu_list = writer.create_hdu_list(vis, obs)
            hdu_list.writeto(fits_path, overwrite=True)

            ms = Measurement.from_fits(fits_path)
            ms.save_as_ms(ms_path)

    def create(self, base_path: str, sim: bool = True) -> tuple[list, list]:
        """Creates the measurement set.

        Parameters
        ----------
        base_path : str
            Base directory where subdirectories for data
            are located. The structure is the following:

                base_path/
                 |
                 |- measurement_sets/
                 |- fits/

        sim : bool, optional
            If set to False, no simulation is applied and
            the measurement set is created directly from
            the FITS file. Default: True

        Returns
        -------
        vis_list : list
            List of visibility dataclass objects.
        obs_list : list
            List of observation dataclass objects.
        """
        self.vis_list = []
        self.obs_list = []
        self.BASE_PATH = base_path

        if not sim:
            ms_path = (
                self.BASE_PATH + f"/measurement_sets/{Path(self.fits_file).stem}.ms"
            )
            print("Simulation set to false. Creating measurement set...")
            print(self.fits_file)
            ms = Measurement.from_fits(self.fits_file)
            ms.save_as_ms(ms_path)

            return 0

        if self.mode == "delta":
            for delta in np.arange(self.start, self.end, self.step):
                self._visibilities(delta=delta, amp_ratio=0.5)
        elif self.mode == "amp":
            for amp in np.arange(self.start, self.end, self.step):
                self._visibilities(delta=45, amp_ratio=amp)

        return self.vis_list, self.obs_list
