"""Taken and slightly adapted from our radiotools package."""

import numpy as np
import torch
from scipy.constants import c


def create_attributes(uu, vv, vis_data, freq, fov, img_size):
    """
    Calculates the mask (UV coverage) and the dirty image.

    Parameters
    ----------
    uu: array_like
        The U baseline coordinates in units of wavelength
    vv: array_like
        The U baseline coordinates in units of wavelength
    stokes_i: array_like
        The Stokes I parameters of the measurement
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
