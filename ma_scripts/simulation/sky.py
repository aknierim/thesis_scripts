import torch
import tqdm
import numpy as np
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve2d

def create_sky(
    size,
    shift=10,
    intensity=1.0,
    random_state=42,
    n_sources=10,
    mode="RC",
) -> torch.tensor:
    sky = torch.zeros((size, size))

    if "loop" in mode:
        rng = np.random.default_rng(random_state)

        x = torch.from_numpy(rng.integers(low=0, high=size, size=n_sources))
        y = torch.from_numpy(rng.integers(low=0, high=size, size=n_sources))
        intensity = torch.from_numpy(rng.random(n_sources))
        sky[x,y] = intensity

        x_stddev = torch.from_numpy(rng.uniform(0, 4, size=n_sources))
        y_stddev = torch.from_numpy(rng.uniform(0, 4, size=n_sources))
        theta = torch.from_numpy(rng.uniform(0, 180, size=n_sources))

        temp = size / 200
        for i in tqdm(range(n_sources)):
            xi, yi = x[i].item(), y[i].item()
            kernel = Gaussian2DKernel(x_stddev[i].item(), y_stddev[i].item(), theta[i].item()).array
            sky[xi - int(temp):xi + int(temp), yi - int(temp):yi + int(temp)] = torch.from_numpy(
                convolve2d(sky[xi - int(temp):xi + int(temp), yi - int(temp):yi + int(temp)], kernel, mode="same")
            )

    if "R" in mode:
        rng = np.random.default_rng(random_state)

        x = torch.from_numpy(rng.integers(low=0, high=size, size=n_sources))
        y = torch.from_numpy(rng.integers(low=0, high=size, size=n_sources))
        intensity = torch.from_numpy(rng.random(n_sources))

        sky[x,y] = intensity
    if "grid" in mode:
        center = int(size / 2)
        sky[center, center] = intensity
        sky[center + shift, center] = intensity
        sky[center + shift, center + shift] = intensity
        sky[center - shift, center] = intensity
        sky[center - shift, center - shift] = intensity
        sky[center + shift, center - shift] = intensity
        sky[center, center - shift] = intensity
        sky[center, center + shift] = intensity
        sky[center - shift, center + shift] = intensity

    if "c" in mode:
        kernel = Gaussian2DKernel(4).array
        sky = torch.from_numpy(
            convolve2d(sky, kernel, mode="same")
        )

    if "C" in mode:
        rng = np.random.default_rng(random_state)

        x_stddev = rng.uniform(0, 10)
        y_stddev = rng.uniform(0, 10)
        theta = rng.uniform(0, 180)

        kernel = Gaussian2DKernel(x_stddev, y_stddev, theta).array
        sky = torch.from_numpy(
            convolve2d(sky, kernel, mode="same")
        )
    return sky