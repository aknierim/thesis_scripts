from astropy.io import fits


class FitsReader:
    def __init__(self, file):
        """Handle fits files simulated with the pyvisgen package.

        Parameters
        ----------
        file: str
            Path to a fits file.

        """
        self.file = file
        print(self.file)

    def get_uv_data(self):
        with fits.open(self.file) as hdul:
            uv_data = hdul[0].data

        return uv_data

    def get_freq_data(
        self,
    ):
        with fits.open(self.file) as hdul:
            base_freq = hdul[0].header["CRVAL4"]
            freq_offsets = hdul[self._find_FQ_table(hdul)].data["IF FREQ"]
            freq_bands = base_freq + freq_offsets

        return base_freq, freq_bands.flatten()

    def _find_FQ_table(self, hdul):
        try:
            index = hdul.index_of("AIPS FQ")
        except KeyError:
            index = hdul.index_of("FQ")

        return index

    def __call__(self):
        with fits.open(self.file) as hdul:
            fits_file = hdul

        return fits_file.info()
