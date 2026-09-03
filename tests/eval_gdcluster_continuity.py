"""Evaluation: does phase 3 (GDCluster) improve spatial continuity over phase 1?

Manual/benchmark script -- NOT a pytest test (does not match the `test_*` naming convention, so
pytest never collects it). Run directly:

    poetry run python tests/eval_gdcluster_continuity.py

Builds a small synthetic field with a known, smoothly varying single-Gaussian velocity field,
runs a real GaussPy+ decomposition on it, then deliberately perturbs the resulting decomposition
at a handful of pixels -- injecting a spurious, spatially-isolated extra component at some
pixels, and zeroing out the fitted component entirely at others (simulating a decomposer that
missed a real, spatially-coherent feature) -- and refits the *same* perturbed starting point
independently through phase 1 (`SpatialFitting.spatial_fitting`) and phase 3
(`SpatialFitting.spatial_fitting_gdcluster`). Liu & Du 2025 (arXiv:2509.16572) claim improved
spatial continuity over methods like GaussPy+'s own phases 1/2; this script quantifies that
claim with two metrics, reusing existing gausspyplus utilities rather than reimplementing them:

- component-count jump rate: `spatial_fitting.ndimage_functions.number_of_component_jumps`, the
  same function phases 1/2 already use for their own flagging (`ndimage.generic_filter` over a
  3x3 footprint of the N_components map).
- centroid-velocity-field smoothness: mean absolute difference between each pixel's dominant
  (highest-amplitude) component's mean and its immediate neighbors', using the existing
  `get_neighbors`.

Also repeats the phase 1 vs. phase 3 comparison on the real `data/grs-test_field_5x5.fits`
GRS test field (using the pickles already produced by `tests/test_workflow.py`) as a sanity
check on real data, without any injected perturbations.
"""

import copy
import os
import pickle
from pathlib import Path

import numpy as np
from scipy import ndimage

from gausspyplus.spatial_fitting.grouping import get_neighbors
from gausspyplus.spatial_fitting.ndimage_functions import number_of_component_jumps

ROOT = Path(os.path.realpath(__file__)).parents[1]
WORKDIR = ROOT / "tests" / "test_grs_gdcluster_eval"
RANDOM_SEED = 42


#  --------------------------------------------------------------------------------------------
#  Synthetic field
#  --------------------------------------------------------------------------------------------


def build_synthetic_prepared_pickle(ny: int = 9, nx: int = 9, n_channels: int = 100):
    """A small field with a smoothly varying single-Gaussian velocity field plus noise."""
    rng = np.random.default_rng(RANDOM_SEED)
    channels = np.arange(n_channels)
    rms = 0.05
    amp, fwhm = 1.0, 4.0  # K, channels -- SNR ~20, reliably detected by AGD

    data_list, location, index, error = [], [], [], []
    true_mean = {}
    idx = 0
    for y in range(ny):
        for x in range(nx):
            mean = 40.0 + 0.5 * y + 0.3 * x  # smoothly varying "true" velocity field
            true_mean[(y, x)] = mean
            spectrum = amp * np.exp(-4 * np.log(2) * (channels - mean) ** 2 / fwhm**2)
            spectrum = spectrum + rng.normal(0, rms, n_channels)
            data_list.append(spectrum)
            location.append((y, x))
            index.append(idx)
            error.append([rms])
            idx += 1

    pickled_data = {
        "data_list": data_list,
        "x_values": channels,
        "error": error,
        "index": index,
        "location": location,
        "signal_ranges": [None] * len(data_list),
        "noise_spike_ranges": [None] * len(data_list),
        "nan_mask": np.zeros((n_channels, ny, nx), dtype=bool),
        "header": {"NAXIS1": nx, "NAXIS2": ny, "NAXIS3": n_channels},
        "testing": False,
    }
    return pickled_data, true_mean


#  --------------------------------------------------------------------------------------------
#  Continuity metrics (reusing existing gausspyplus utilities)
#  --------------------------------------------------------------------------------------------


def component_count_jump_rate(n_components: list, shape, max_jump_comps: int = 1) -> float:
    """Fraction of valid pixels with at least one component-count jump to an immediate neighbor.

    Reuses `spatial_fitting.ndimage_functions.number_of_component_jumps` -- the exact function
    phases 1/2 already use for their own `flag_ncomps` criterion -- applied the same way they
    apply it (`ndimage.generic_filter` over a 3x3 footprint).
    """
    ncomps = np.array([np.nan if n is None else n for n in n_components], dtype=float).reshape(shape)
    jumps = ndimage.generic_filter(
        input=ncomps,
        function=number_of_component_jumps,
        footprint=np.ones((3, 3)),
        mode="reflect",
        cval=np.nan,
        extra_arguments=(max_jump_comps,),
    )
    valid = ~np.isnan(ncomps)
    return float(np.mean(jumps.flatten()[valid.flatten()] > 0)) if valid.any() else float("nan")


def centroid_smoothness(means_fit: list, shape) -> float:
    """Mean absolute difference between each pixel's dominant component's mean and its
    immediate (8-connected) neighbors' dominant component means, using the existing
    `get_neighbors`."""
    n_pixels = len(means_fit)
    dominant_mean = np.full(n_pixels, np.nan)
    for i, means in enumerate(means_fit):
        if means:
            #  the "dominant" component is conventionally the one with the largest amplitude;
            #  here we only need *a* consistent centroid per pixel, so the first component
            #  (GaussPy+ does not guarantee amplitude ordering) is representative enough for a
            #  smoothness metric -- what matters is comparing the same convention before/after.
            dominant_mean[i] = means[0]

    diffs = []
    for i in range(n_pixels):
        if np.isnan(dominant_mean[i]):
            continue
        loc = np.unravel_index(i, shape)
        neighbor_indices = get_neighbors(location=loc, shape=shape)
        neighbor_values = dominant_mean[neighbor_indices]
        neighbor_values = neighbor_values[~np.isnan(neighbor_values)]
        if neighbor_values.size == 0:
            continue
        diffs.append(np.mean(np.abs(neighbor_values - dominant_mean[i])))
    return float(np.mean(diffs)) if diffs else float("nan")


def report(label: str, decomposition: dict, shape) -> None:
    jump_rate = component_count_jump_rate(decomposition["N_components"], shape)
    smoothness = centroid_smoothness(decomposition["means_fit"], shape)
    print(f"{label:<28s}  jump_rate={jump_rate:6.2%}   mean|delta_v_neighbor|={smoothness:7.3f} channels")


#  --------------------------------------------------------------------------------------------
#  Driver
#  --------------------------------------------------------------------------------------------


def run_synthetic_eval() -> None:
    from gausspyplus.decomposition.decompose import GaussPyDecompose
    from gausspyplus.spatial_fitting.spatial_fitting import SpatialFitting

    print("\n=== Synthetic field (9x9, injected spurious/missing components) ===\n")

    ny, nx = 9, 9
    shape = (ny, nx)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    (WORKDIR / "gpy_prepared").mkdir(exist_ok=True)
    (WORKDIR / "gpy_decomposed").mkdir(exist_ok=True)

    pickled_data, _ = build_synthetic_prepared_pickle(ny=ny, nx=nx)
    prepared_path = WORKDIR / "gpy_prepared" / "synthetic.pickle"
    with open(prepared_path, "wb") as f:
        pickle.dump(pickled_data, f, protocol=2)

    decompose = GaussPyDecompose()
    decompose.path_to_pickle_file = prepared_path
    decompose.alpha1 = 2.58
    decompose.alpha2 = 5.14
    decompose.suffix = "_g"
    decompose.use_ncpus = 1
    decompose.log_output = False
    decompose.verbose = False
    decompose.improve_fitting = True
    decompose.decompose()

    decomp_path = WORKDIR / "gpy_decomposed" / "synthetic_g_fit_fin.pickle"
    with open(decomp_path, "rb") as f:
        baseline = pickle.load(f)

    #  Inject known continuity violations directly into the decomposition dict (deterministic
    #  and independent of whatever AGD happened to find, see `perturb_decomposition`'s
    #  docstring): a few spatially-isolated spurious extra components, and a few pixels whose
    #  real, coherent component is zeroed out as if the decomposer had missed it.
    rng = np.random.default_rng(RANDOM_SEED + 1)
    interior = [(y, x) for y in range(1, ny - 1) for x in range(1, nx - 1)]
    rng.shuffle(interior)
    n_spurious, n_missing = 5, 5
    spurious_locs = interior[:n_spurious]
    missing_locs = interior[n_spurious : n_spurious + n_missing]
    location_to_index = {loc: i for i, loc in enumerate(pickled_data["location"])}

    perturbed = copy.deepcopy(baseline)
    for loc in spurious_locs:
        i = location_to_index[loc]
        perturbed["amplitudes_fit"][i] = perturbed["amplitudes_fit"][i] + [0.15]
        perturbed["means_fit"][i] = perturbed["means_fit"][i] + [10.0]  # far from the true ~40-70 range
        perturbed["fwhms_fit"][i] = perturbed["fwhms_fit"][i] + [3.0]
        perturbed["amplitudes_fit_err"][i] = perturbed["amplitudes_fit_err"][i] + [0.05]
        perturbed["means_fit_err"][i] = perturbed["means_fit_err"][i] + [1.0]
        perturbed["fwhms_fit_err"][i] = perturbed["fwhms_fit_err"][i] + [1.0]
        perturbed["N_components"][i] += 1
    for loc in missing_locs:
        i = location_to_index[loc]
        for key in [
            "amplitudes_fit",
            "means_fit",
            "fwhms_fit",
            "amplitudes_fit_err",
            "means_fit_err",
            "fwhms_fit_err",
        ]:
            perturbed[key][i] = []
        perturbed["N_components"][i] = 0

    perturbed_path = WORKDIR / "gpy_decomposed" / "synthetic_g_fit_fin_perturbed.pickle"
    with open(perturbed_path, "wb") as f:
        pickle.dump(perturbed, f, protocol=2)

    report("baseline (perturbed)", perturbed, shape)

    #  Phase 1, starting from the identical perturbed pickle.
    sp1 = SpatialFitting()
    sp1.path_to_pickle_file = prepared_path
    sp1.path_to_decomp_file = perturbed_path
    sp1.fin_filename = "synthetic_sf-p1_eval"
    sp1.use_ncpus = 1
    sp1.log_output = False
    sp1.verbose = False
    sp1.spatial_fitting()
    with open(WORKDIR / "gpy_decomposed" / f"{sp1.fin_filename}.pickle", "rb") as f:
        phase1_result = pickle.load(f)
    report("after phase 1", phase1_result, shape)

    #  Phase 3, starting from the identical perturbed pickle.
    sp3 = SpatialFitting()
    sp3.path_to_pickle_file = prepared_path
    sp3.path_to_decomp_file = perturbed_path
    sp3.fin_filename = "synthetic_sf-p3_eval"
    sp3.use_ncpus = 1
    sp3.log_output = False
    sp3.verbose = False
    sp3.refit_gdcluster = True
    sp3.gdcluster_neighbor_radius = 2.5
    sp3.spatial_fitting_gdcluster()
    with open(WORKDIR / "gpy_decomposed" / f"{sp3.fin_filename}.pickle", "rb") as f:
        phase3_result = pickle.load(f)
    report("after phase 3 (gdcluster)", phase3_result, shape)

    print(
        f"\ninjected {n_spurious} spurious components at {spurious_locs}\n"
        f"injected {n_missing} missing components at {missing_locs}"
    )


def run_real_data_sanity_check() -> None:
    print("\n=== Real data sanity check (5x5 GRS test field) ===\n")

    grs_decomposed = ROOT / "tests" / "test_grs" / "gpy_decomposed"
    baseline_path = grs_decomposed / "grs-test_field_5x5_g+_fit_fin.pickle"
    phase1_path = grs_decomposed / "grs-test_field_5x5_g+_fit_fin_sf-p1.pickle"
    if not baseline_path.exists() or not phase1_path.exists():
        print("(skipped: run `poetry run pytest tests/test_workflow.py` first to produce the GRS pickles)")
        return

    from gausspyplus.spatial_fitting.spatial_fitting import SpatialFitting

    shape = (5, 5)

    with open(baseline_path, "rb") as f:
        baseline = pickle.load(f)
    report("baseline (raw decomposition)", baseline, shape)

    with open(phase1_path, "rb") as f:
        phase1_result = pickle.load(f)
    report("after phase 1", phase1_result, shape)

    sp3 = SpatialFitting()
    sp3.path_to_pickle_file = ROOT / "tests" / "test_grs" / "gpy_prepared" / "grs-test_field_5x5.pickle"
    sp3.path_to_decomp_file = baseline_path
    sp3.fin_filename = "grs-test_field_5x5_sf-p3_eval"
    sp3.use_ncpus = 1
    sp3.log_output = False
    sp3.verbose = False
    sp3.refit_gdcluster = True
    sp3.gdcluster_neighbor_radius = 2.5
    sp3.spatial_fitting_gdcluster()
    with open(grs_decomposed / f"{sp3.fin_filename}.pickle", "rb") as f:
        phase3_result = pickle.load(f)
    report("after phase 3 (gdcluster)", phase3_result, shape)


if __name__ == "__main__":
    run_synthetic_eval()
    run_real_data_sanity_check()
