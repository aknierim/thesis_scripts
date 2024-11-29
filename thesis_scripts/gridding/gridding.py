"""Taken and slightly adapted from our radiotools package."""

from pathlib import Path

import numpy as np
import torch
from numpy.typing import ArrayLike
from numpy import AxisError
from scipy.constants import c

from thesis_scripts.io import FitsReader


def create_attributes(
    uu: torch.tensor,
    vv: torch.tensor,
    vis_data: "Visibilities",
    freq: float,
    fov: float,
    img_size: int,
) -> tuple:
    """
    Calculates the mask (UV coverage) and the dirty image.

    Parameters
    ----------
    uu : array_like
        The U baseline coordinates in units of wavelength
    vv : array_like
        The U baseline coordinates in units of wavelength
    vis_data : pyvisgen.simulation.visibility.Visibilities
        pyvisgen visibility dataclass object.
    freq : float
        Frequency of the observation.
    fov : float
        Field of view.
    img_size : int
        Image size.

    Returns
    -------
    samps : torch.tensor
        Tensor of samples.
    mask : torch.tensor
        Tensor of the grid mask.
    mask_real : torch.tensor
        Tensor of the real part of the grid mask.
    mask_imag : torch.tensor
        Tensor of the imaginary part of the grid mask.
    dirty_img : torch.tensor
        Tensor of the dirty image.
    """

    u = uu * freq / c
    v = vv * freq / c

    real = vis_data.real
    imag = vis_data.imag

    samps = np.array(
        [
            torch.cat([-u, u]),
            torch.cat([-v, v]),
            torch.cat([real, real]),
            torch.cat([imag, -imag]),
        ]
    )

    N = img_size

    fov = np.deg2rad(fov * 3600)

    delta_l = fov / N
    delta = (N * delta_l) ** (-1)

    bins = (
        torch.arange(start=-(N / 2) * delta, end=(N / 2 + 1) * delta, step=delta)
        - delta / 2
    )

    mask, *_ = np.histogram2d(samps[0], samps[1], bins=[bins, bins], density=False)
    mask[mask == 0] = 1

    mask_real, x_edges, y_edges = np.histogram2d(
        samps[0], samps[1], bins=[bins, bins], weights=samps[2], density=False
    )
    mask_imag, x_edges, y_edges = np.histogram2d(
        samps[0], samps[1], bins=[bins, bins], weights=samps[3], density=False
    )
    mask_real /= mask
    mask_imag /= mask

    mask = mask
    mask_real = mask_real
    mask_imag = mask_imag
    dirty_img = np.abs(
        np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(mask_real + 1j * mask_imag)))
    )

    return samps, mask, mask_real, mask_imag, dirty_img


class Gridder:
    @classmethod
    def from_params(
        cls,
        uu: ArrayLike,
        vv: ArrayLike,
        freq_bands: ArrayLike,
        visibilities: ArrayLike,
        img_size: int,
        fov: float,
    ) -> tuple:
        cls = cls()
        return cls.gridder()

    @classmethod
    def from_fits(cls, file_path: str | Path, img_size: int, fov: float) -> tuple:
        cls = cls()

        reader = FitsReader(file_path)
        data = reader.get_uv_data()
        _, freq_bands = reader.get_freq_data()

        try:
            u = data["UU"]
        except KeyError:
            u = data["UU--"]
        try:
            v = data["VV"]
        except KeyError:
            v = data["VV--"]

        vis = data["DATA"]

        return cls.gridder(u, v, freq_bands, vis, img_size, fov)

    def gridder(
        cls,
        uu: ArrayLike,
        vv: ArrayLike,
        freq_bands: ArrayLike,
        visibilities: ArrayLike,
        img_size: int,
        fov: float,
    ) -> tuple:
        u = np.array([uu * np.array(freq) for freq in freq_bands]).ravel()
        v = np.array([vv * np.array(freq) for freq in freq_bands]).ravel()

        try:
            stokes_I = (
                np.squeeze((visibilities[..., 0, 0] + 1j * visibilities[..., 0, 1]))
                .swapaxes(0, 1)
                .ravel()
            )
        except AxisError:
            stokes_I = (
                np.squeeze((visibilities[..., 0, 0] + 1j * visibilities[..., 0, 1]))
                .ravel()
            )

        samps = np.array(
            [
                np.concatenate([-u, u]),
                np.concatenate([-v, v]),
                np.concatenate([stokes_I.real, stokes_I.real]),
                np.concatenate([stokes_I.imag, -stokes_I.imag]),
            ]
        )

        fov = np.deg2rad(fov / 3600)
        delta_l = fov / img_size
        delta = (img_size * delta_l) ** (-1)

        bins = (
            np.arange(
                start=-(img_size / 2) * delta,
                stop=((img_size + 1) / 2) * delta,
                step=delta,
            )
            - delta / 2
        )

        mask, *_ = np.histogram2d(samps[0], samps[1], bins=[bins, bins], density=False)
        mask[mask == 0] = 1

        mask_real, *_ = np.histogram2d(
            samps[0], samps[1], bins=[bins, bins], weights=samps[2], density=False
        )
        mask_imag, *_ = np.histogram2d(
            samps[0], samps[1], bins=[bins, bins], weights=samps[3], density=False
        )

        mask_real /= mask
        mask_imag /= mask
        dirty_image = np.abs(
            np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(mask_real + 1j * mask_imag)))
        )

        return mask_real, mask_imag, dirty_image
