"""pytest tests for module noise_estimation.py"""
import os
from pathlib import Path

from astropy.io import fits

ROOT = Path(os.path.realpath(__file__)).parents[1]
DATA = fits.getdata(ROOT / "data" / "grs-test_field.fits")


def test_get_signal_ranges():
    from gausspyplus.preparation.determine_intervals import get_signal_ranges

    spectrum = DATA[:, 31, 40]
    signal_ranges = get_signal_ranges(spectrum=spectrum, rms=0.1)
    assert signal_ranges == [[142, 207], [212, 252]]


def test_get_signal_ranges_without_signal():
    """If no signal is found, remove_intervals (e.g. noise spikes) must still be excluded from the returned
    ranges so they are not used in the goodness-of-fit calculations (behavior of v0.2)."""
    import numpy as np

    from gausspyplus.preparation.determine_intervals import get_signal_ranges

    spectrum = np.random.default_rng(0).normal(0, 0.1, 400)

    assert get_signal_ranges(spectrum=spectrum, rms=0.1) == []
    assert get_signal_ranges(spectrum=spectrum, rms=0.1, remove_intervals=[[100, 120]]) == [[0, 100], [120, 400]]


if __name__ == "__main__":
    test_get_signal_ranges()
