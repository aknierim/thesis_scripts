import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.io.fits.hdu.image import PrimaryHDU
from matplotlib.axes import Axes
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from numpy.typing import ArrayLike


def show_beam(
    img: ArrayLike, hdu: PrimaryHDU, ax: Axes, true_img: ArrayLike = None
) -> None:
    """Plots beam shape in the lower left corner of the plot.

    Parameters
    ----------
    img : array_like
        Image data.
    hdu : PrimaryHDU
        Primary HDU of the data.
    ax : matplotlib.axes.Axes
        Axis to plot on.
    true_img : array_like, optional
        True sky image. If passed, computes a correction
        factor for the beam size by taking the ratio of
        true sky and input image sizes. Default: None
    """
    size_correction = 1

    if true_img is not None:
        img_shape = np.asarray(img.shape[-2:])
        true_shape = np.asarray(true_img.shape)
        size_correction = true_shape / img_shape
        size_correction = size_correction.prod()

    cell_size = size_correction * hdu[0].header["CDELT1"]

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = xlim[0] + (xlim[1] - xlim[0]) * 0.1
    y = ylim[0] + (ylim[1] - ylim[0]) * 0.1

    ax.add_patch(
        Ellipse(
            (x, y),
            width=hdu[0].header["BMIN"] / cell_size,
            height=hdu[0].header["BMAJ"] / cell_size,
            angle=hdu[0].header["BPA"],
            facecolor="lightgrey",
        )
    )


class Skyplot:
    def __init__(
        self,
        models: dict,
        images: dict,
        true_sky: ArrayLike,
        hdus: dict,
        true_sky_hdu: PrimaryHDU,
        lims: dict = {},
        ts_lims: dict = {},
        imshow_kwargs: dict = {
            "norm": SymLogNorm(0.005, vmin=0, vmax=1),
            "cmap": "inferno",
            "origin": "lower",
        },
        cont_kwargs: dict = {
            "colors": "white",
            "norm": SymLogNorm(0.005),
            "linewidths": 0.8,
        },
        neg_cont_kwargs: dict = {
            "norm": SymLogNorm(0.005),
            "colors": "lightgrey",
            "linestyles": "dashed",
            "linewidths": 0.8,
        },
    ) -> None:
        """Initializes common variables for plots.

        Parameters
        ----------
        models : dict
            Dictionary containing model data mapped to keys I, Q, U, or V.
        images : dict
            Dictionary containing image data mapped to keys I, Q, U, or V.
        true_sky : array_like
            True sky image data.
        hdus : dict
            Dictionary containing primary HDUs mapped to keys I, Q, U, or V.
        true_sky_hdu : PrimaryHDU
            True sky primary HDU.
        lims : dict, optional
            x- and y-limits for images. Default: {}
        ts_lims : dict, optional
            x- and y-limits for true sky image. Default: {}
        imshow_kwargs : dict, optional
            Keyword arguments for matplotlib.axes.Axes.imshow.
        cont_kwargs : dict, optional
            Keyword arguments for matplotlib.axes.Axes.contour.
        neg_cont_kwargs : dict, optional
            Keyword arguments for matplotlib.axes.Axes.contour of
            negative values, see plots released by MOJAVE.
        """
        self.models: dict = models

        self.images: dict = images
        self.true_sky = true_sky

        self.hdus: dict = hdus
        self.true_sky_hdu = true_sky_hdu

        self.lims = lims
        self.ts_lims = ts_lims
        self.imshow_kwargs = imshow_kwargs
        self.cont_kwargs = cont_kwargs
        self.neg_cont_kwargs = neg_cont_kwargs

        RR = self.images["I"] + self.images["V"]
        LL = self.images["I"] - self.images["V"]
        LR = self.images["Q"] + 1j * self.images["U"]
        RL = self.images["Q"] - 1j * self.images["U"]

        self.visibilities = [RR, RL, LR, LL]

    def _common_opts(
        self, im: ArrayLike, fig: Figure, ax: Axes, opt_lbl: str = ""
    ) -> None:
        """Sets common options for plots.

        Parameters
        ----------
        im : array_like
            Image data for colorbar.
        fig : matplotlib.figure.Figure
            Matplotlib figure object.
        ax : matplotlib.axes.Axes
            Matplotlib axes object.
        opt_lbl : str, optional
            Optional, additional label passed to ax.annotate.
            Default: ""
        """
        fig.colorbar(
            im,
            ax=ax,
            pad=-0.02,
            label="Intensity $/\;\mathrm{Jy}\cdot\mathrm{Beam}^{-1}$",
        )

        anchor = ax.get_window_extent()
        ax.annotate(
            ax.get_label() + opt_lbl,
            (0.05, 0.95),
            va="top",
            ha="left",
            xycoords=anchor,
            fontsize=18,
            color="white",
        )

    def model(self, imshow_kwargs: dict = {}) -> tuple[Figure, Axes]:
        """Plots model data initialized in Skyplot.__init__().

        Parameters
        ----------
        imshow_kwargs : dict, optional
            Dictionary containing keyword arguments for imshow.
            Overwrites imshow_kwargs set in Skyplot.__init__().
            Default: {}

        Returns
        -------
        fig : matplotlib.figure.Figure
            Matplotlib figure object.
        ax : matplotlib.axes.Axes
            Matplotlib axes object.
        """
        _imshow_kwargs = self.imshow_kwargs
        if imshow_kwargs:
            _imshow_kwargs = imshow_kwargs

        fig, axs = plt.subplot_mosaic("IQ;UV", layout="constrained", figsize=(10, 7))

        for ax, model in zip(axs.values(), self.models.values()):
            im = ax.imshow(model[0, 0, ...], **_imshow_kwargs)
            ax.set(**self.lims)

            self._common_opts(im, fig, ax, opt_lbl=" (model)")

        return fig, axs

    def iquv(
        self, imshow_kwargs: dict = {}, cont_kwargs: dict = {}
    ) -> tuple[Figure, Axes]:
        """Plots IQUV image data initialized in Skyplot.__init__().

        Parameters
        ----------
        imshow_kwargs : dict, optional
            Dictionary containing keyword arguments for imshow.
            Overwrites imshow_kwargs set in Skyplot.__init__().
            Default: {}
        cont_kwargs : dict, optional
            Dictionary containing keyword arguments for contourm plots.
            Overwrites cont_kwargs set in Skyplot.__init__().
            Default: {}

        Returns
        -------
        fig : matplotlib.figure.Figure
            Matplotlib figure object.
        ax : matplotlib.axes.Axes
            Matplotlib axes object.
        """
        _imshow_kwargs = self.imshow_kwargs
        _cont_kwargs = self.cont_kwargs
        if imshow_kwargs:
            _imshow_kwargs = imshow_kwargs
        if cont_kwargs:
            _cont_kwargs = cont_kwargs

        fig, axs = plt.subplot_mosaic(
            [["I", "Q", "True Sky", "True Sky"], ["U", "V", "True Sky", "True Sky"]],
            layout="constrained",
            figsize=(19, 7),
        )

        ax_imgs = []
        for comp in "IQUV":
            im = axs[comp].imshow(self.images[comp], **_imshow_kwargs)
            ax_imgs.append(im)

            axs[comp].set(**self.lims)

            show_beam(
                self.images[comp], self.hdus[comp], ax=axs[comp], true_img=self.true_sky
            )

            try:
                axs[comp].contour(
                    self.images[comp],
                    levels=np.geomspace(
                        self.images[comp].max() / 1e2, self.images[comp].max(), 5
                    ),
                    **_cont_kwargs,
                )
            except Exception as e:
                warnings.warn(str(e))
                continue

        im_true = axs["True Sky"].imshow(
            self.true_sky, origin="lower", cmap="inferno", norm=SymLogNorm(0.005)
        )
        axs["True Sky"].contour(
            self.true_sky,
            levels=np.geomspace(self.true_sky.max() / 6e2, self.true_sky.max(), 10),
            **_cont_kwargs,
        )
        axs["True Sky"].set(**self.ts_lims)
        ax_imgs.append(im_true)
        show_beam(
            self.true_sky, self.true_sky_hdu, ax=axs["True Sky"], true_img=self.true_sky
        )

        for ax, im in zip(axs.values(), ax_imgs):
            fig.colorbar(
                im,
                pad=-0.01,
                label="Flux Intensity $/\;\mathrm{Jy}\cdot\mathrm{Beam}^{-1}$",
            )

            anchor = ax.get_window_extent()
            ax.annotate(
                ax.get_label(),
                (0.05, 0.95),
                va="top",
                ha="left",
                xycoords=anchor,
                fontsize=18,
                color="white",
            )

        return fig, axs

    def rrll(
        self, imshow_kwargs: dict = {}, cont_kwargs: dict = {}
    ) -> tuple[Figure, Axes]:
        """Plots RR, LL, RL, LL image data initialized
        in Skyplot.__init__().

        Parameters
        ----------
        imshow_kwargs : dict, optional
            Dictionary containing keyword arguments for imshow.
            Overwrites imshow_kwargs set in Skyplot.__init__().
            Default: {}
        cont_kwargs : dict, optional
            Dictionary containing keyword arguments for contourm plots.
            Overwrites cont_kwargs set in Skyplot.__init__().
            Default: {}

        Returns
        -------
        fig : matplotlib.figure.Figure
            Matplotlib figure object.
        ax : matplotlib.axes.Axes
            Matplotlib axes object.
        """

        _imshow_kwargs = self.imshow_kwargs
        _cont_kwargs = self.cont_kwargs
        if imshow_kwargs:
            _imshow_kwargs = imshow_kwargs
        if cont_kwargs:
            _cont_kwargs = cont_kwargs

        fig, axs = plt.subplot_mosaic(
            [["RR", "RL"], ["LR", "LL"]], layout="constrained", figsize=(11, 8)
        )

        for ax, vis in zip(axs.values(), self.visibilities):
            im = ax.imshow(vis.real, **_imshow_kwargs)

            ax.set(**self.lims)

            try:
                ax.contour(
                    vis.real,
                    levels=np.geomspace(vis.max() / 1e2, vis.max(), 10),
                    **_cont_kwargs,
                )
            except ValueError as e:
                warnings.warn(str(e))
                continue

            fig.colorbar(
                im,
                pad=-0.01,
                label="Flux Intensity $/\;\mathrm{Jy}\cdot\mathrm{Beam}^{-1}$",
            )

            anchor = ax.get_window_extent()
            ax.annotate(
                ax.get_label(),
                (0.05, 0.95),
                va="top",
                ha="left",
                xycoords=anchor,
                fontsize=14,
                color="white",
            )

        return fig, axs
