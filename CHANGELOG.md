# GaussPy+ Changelog

## 0.3.0 (unreleased)

A major modernization and restructuring of the package, based on the extensive refactoring work by Manuel Riener (2021–2023) and continued in this fork. Decomposition results are consistent with v0.2: the full pipeline was validated against the v0.2 code on the GRS test fields (identical component counts; fit parameters agree to numerical precision; see the compatibility notes below for the two intentional behavior changes).

### Package restructuring (breaking change: import paths)

The flat module layout was reorganized into subpackages. Users need to update their imports:

| v0.2 | v0.3 |
|---|---|
| `gausspyplus.prepare` | `gausspyplus.preparation.prepare` |
| `gausspyplus.decompose` | `gausspyplus.decomposition.decompose` |
| `gausspyplus.spatial_fitting` | `gausspyplus.spatial_fitting.spatial_fitting` |
| `gausspyplus.finalize` | `gausspyplus.processing.finalize` |
| `gausspyplus.training` / `gausspyplus.training_set` | `gausspyplus.training.training` / `gausspyplus.training.training_set` |
| `gausspyplus.plotting` | `gausspyplus.plotting.plotting` |
| `gausspyplus.config_file` | `gausspyplus.definitions.config_file` |
| `gausspyplus.gausspy_py3.AGD_decomposer` | `gausspyplus.decomposition.agd_decomposer` |
| `gausspyplus.gausspy_py3.gradient_descent` | `gausspyplus.training.gradient_descent` |
| `gausspyplus.utils.spectral_cube_functions` | `gausspyplus.processing.spectral_cube_functions` |
| `gausspyplus.utils.noise_estimation` / `determine_intervals` | `gausspyplus.preparation.noise_estimation` / `determine_intervals` |

The `example` directory was renamed to `tutorials`; documentation files moved to `docs`.

### Modernized packaging and dependencies

* Supports Python 3.10–3.13 (v0.2 targeted Python 3.5).
* Compatible with the current scientific Python stack: numpy >= 1.24 (including numpy 2.x), scipy >= 1.10, astropy >= 5.3, lmfit >= 1.2, matplotlib >= 3.7, networkx >= 2.8.
* Packaging with Poetry (`pyproject.toml`); the legacy `astropy_helpers`/`setup.py` build was removed.
* New pytest suite covering the full pipeline (preparation, decomposition, improved fitting, both spatial refitting phases, training, and unit tests) on the GRS test fields.
* Code modernization throughout: type hints, dataclasses for settings and fit models (`Model`, `Spectrum`), black formatting, pre-commit hooks.

### Fixed

* Phase-2 spatially coherent refitting crashed with an `IndexError` on fields containing spectra without fit solutions (wrong indices passed to `np.delete`). Also fixed the incorrect determination of the mask for differing numbers of components.
* The training stage crashed on Python >= 3.13 (frame-locals introspection incompatible with PEP 667).
* The `save_initial_guesses` setting was effectively always enabled (the method of the same name shadowed the settings attribute), and explicitly setting it to `True` raised a `TypeError`. The method is now called `save_initial_guesses_to_file` and the setting works as documented (default `False`).
* `get_signal_ranges` buffered signal intervals with an accelerating pad (i × `pad_channels` in iteration i), overshooting the `min_channels` target; the padding is now constant per iteration.
* Refits that reproduced the same fit to float precision were misclassified as changed (exact `==` comparison of FWHM values) and repeatedly re-refit; the comparison now uses `np.isclose`, removing redundant refit iterations.
* Decomposing dictionaries in the plain GaussPy format (without an `"index"` key) raised a `KeyError`.
* Constructing `plotting.Figure()` without the `max_rows_per_figure` argument was broken (a dead `cached_property` clashed with the dataclass field of the same name).
* Replaced APIs removed from numpy/scipy (`np.float`, `np.int`, `np.in1d`, `scipy.ndimage.filters`).

### Changed

* Renamed for clarity (all internal call sites updated):
  * `GaussPyPrepare.calculate_rms_noise` → `prepare_spectrum` (it returns a fully prepared spectrum)
  * `check_if_intervals_contain_signal` → `get_intervals_with_significant_signal`
  * `area_of_gaussian` → `integrated_area_under_gaussian_curve`
  * `determine_peaks(peak=...)` → `determine_peaks(peak_type=...)`
  * `GaussPyTraining.gpy_dirpath` → `dirpath_gpy` (consistent with all other stages)
* Removed unused helper functions `get_slice_parameters`, `transform_coordinates_to_pixel`, and `combine_fields` from `spectral_cube_functions`.

### Compatibility notes

* **Pickle formats are unchanged**: files produced with v0.2 (prepared, decomposed, and training-set pickles) remain fully readable, and v0.3 outputs keep the same dictionary keys and value formats.
* Two intentional behavior changes relative to v0.2, both validated on the GRS test fields:
  1. Signal ranges can be slightly tighter due to the constant-padding fix in `get_signal_ranges` (the v0.2 accelerating buffer overshot `min_channels` by up to ~50%). The effect on fit parameters is at the 1e-6 level; component counts are unaffected.
  2. The spatially coherent refitting performs fewer redundant refit iterations due to the `np.isclose` comparison; final fits are unchanged.

## 0.2 (2020-05-19)

* Established compatibility with Python 3.8.2, Numpy 1.18.4, lmfit 1.0.1,  astropy 4.0.1, matplotlib 3.2.1.

* Added option that uses flagged neighbors in the spatial refitting routine.
If the 'use_all_neighors' parameter is set to True, flagged neighbors are used as refit solutions in case the refit was not possible with fit solutions from unflagged neighbors. See Appendix A.3 in Riener+ 2019b for more details.

* Added option to restrict spatial refitting routine to a subset of the data.
This is only meant for testing purposes, if users would like to check the effects of parameter settings in the spatially coherent refitting routines without running it on the entire data set (which can be time-consuming).
The subset of the data can be indicated with the 'pixel_range' keyword and has to be supplied as a dictionary. For example, 'pixel_range = {'y': [10, 20], 'x': [5, 10]}' restricts the spatial refitting to a subset of 50 spectra located within ``10 <= y < 20`` and ``5 <= x < 10``.

* Added the `finalize.py` module for producing tables of the final decomposition results. The following code gives an example on how to produce a table of the decomposition results from a dataset that was split into individual subcubes:

```python
import os
from gausspyplus.processing.finalize import Finalize

for subcube_nr in range(1, total_nr_of_subcubes + 1):
    filename = '{}{}'.format(ppv_cube_name, subcube_nr)

    fin = Finalize(config_file='gausspy+.ini')
    fin.path_to_pickle_file = os.path.join(
        dirpath_gpy, 'gpy_prepared', '{}.pickle'.format(filename))
    fin.path_to_decomp_file = os.path.join(
        dirpath_gpy, 'gpy_decomposed', '{}_g+_fit_fin_sf-p2.pickle'.format(filename))
    fin.dirpath_table = os.path.join(dirpath_gpy, 'gpy_tables')
    fin.dct_params = {'mean_separation': 4., '_w_start': 2 / 3}
    fin.subcube_nr = subcube_nr
    fin.finalize_dct()
    fin.make_table()
```

* Added try/except blocks to catch errors in the GaussPy decomposition.
If errors are introduced in the GaussPy decomposition step the index of the spectrum with the corresponding error is printed in the terminal. The spectrum causing the error is replaced with None in the fit results dictionary.

* Added safeguard to prevent eternal loop in ``get_signal_ranges``.

* Added flux preserving mode in ``gausspyplus.processing.spectral_cube_functions.spatial_smoothing`` if ``reproject=True``.

* Introduced ``max_ncomps`` parameter, which enforces a maximum number of fit components per spectrum.

* Added function to create a default file structure similar to the GRS test field scripts in the example directory.

* Removed HDF5 dependency.

* Many small improvements and bugfixes.


## 0.1 (2019-06-01)

* Initial release of gausspyplus.
