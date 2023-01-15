import os
import pickle
import shutil
from pathlib import Path

from astropy.io import fits

import numpy as np

ROOT = Path(os.path.realpath(__file__)).parents[1]
filepath = Path(ROOT / "data" / "grs-test_field_10x10.fits")
DATA = fits.getdata(filepath)

# TODO: Delete created pickle files after running the tests; otherwise intermediate-stage pickle files can still be
#  used if an intermediate stage fails and later tests seem to pass even though they should not


def is_not_none(x):
    return x is not None


def _remove_old_files():
    for dirname in ["gpy_prepared", "gpy_decomposed"]:
        if (path_files := ROOT / "tests" / "test_grs" / dirname).exists():
            for file in [
                filepath
                for filepath in path_files.iterdir()
                if filepath.is_file() and filepath.stem.startswith("grs-test_field_10x10")
            ]:
                file.unlink()


# @pytest.mark.skip(reason="Temporarily disabled to make tests run quicker")
def test_prepare_cube():
    from gausspyplus.preparation.prepare import GaussPyPrepare

    _remove_old_files()

    prepare = GaussPyPrepare()
    prepare.path_to_file = str(filepath)
    prepare.dirpath_gpy = str(ROOT / "tests" / "test_grs")
    prepare.use_ncpus = 1
    prepare.log_output = False
    prepare.verbose = False
    prepare.prepare_cube()
    with open(ROOT / "tests" / f"test_grs/gpy_prepared/{filepath.stem}.pickle", "rb") as pfile:
        data_prepared = pickle.load(pfile)

    expected_values = [
        0.10315973929242594,
        10.212814189950166,
    ]
    actual_values = [
        prepare.average_rms,
        sum(rms[0] for rms in data_prepared["error"] if rms[0] is not None),
    ]
    assert np.allclose(expected_values, actual_values)


# @pytest.mark.skip(reason="Temporarily disabled to make tests run quicker")
def test_decompose_cube_gausspy():
    from gausspyplus.decomposition.decompose import GaussPyDecompose

    decompose = GaussPyDecompose()
    decompose.path_to_pickle_file = str(ROOT / "tests" / f"test_grs/gpy_prepared/{filepath.stem}.pickle")
    decompose.alpha1 = 2.58
    decompose.alpha2 = 5.14
    decompose.suffix = "_g"
    decompose.use_ncpus = 1
    decompose.log_output = False
    decompose.verbose = False
    decompose.improve_fitting = False
    decompose.decompose()
    with open(
        ROOT / "tests" / f"test_grs/gpy_decomposed/{filepath.stem}_g_fit_fin.pickle",
        "rb",
    ) as pfile:
        data_decomposed = pickle.load(pfile)

    expected_values = [
        181,
        2821.8647475612247,
        302.183306677499,
    ]
    actual_values = [
        sum(filter(is_not_none, data_decomposed["N_components"])),
        sum(map(sum, filter(is_not_none, data_decomposed["fwhms_fit"]))),
        sum(map(sum, filter(is_not_none, data_decomposed["fwhms_fit_err"]))),
    ]
    assert np.allclose(expected_values, actual_values)

    decompose.improve_fitting = True
    decompose.suffix = "_g+"
    decompose.decompose()
    with open(
        ROOT / "tests" / f"test_grs/gpy_decomposed/{filepath.stem}_g+_fit_fin.pickle",
        "rb",
    ) as pfile:
        data_decomposed_gplus = pickle.load(pfile)

    expected_values = [
        272,
        4210.279681986788,
        572.5082893844038,
        14.174536904795435,
        117.25885765551081,
        -46522.959923033035,
        107,
        70,
    ]
    actual_values = [
        sum(filter(is_not_none, data_decomposed_gplus["N_components"])),
        sum(map(sum, filter(is_not_none, data_decomposed_gplus["fwhms_fit"]))),
        sum(map(sum, filter(is_not_none, data_decomposed_gplus["fwhms_fit_err"]))),
        sum(filter(is_not_none, data_decomposed_gplus["pvalue"])),
        sum(filter(is_not_none, data_decomposed_gplus["best_fit_rchi2"])),
        sum(filter(is_not_none, data_decomposed_gplus["best_fit_aicc"])),
        sum(map(sum, filter(is_not_none, data_decomposed_gplus["log_gplus"]))),
        sum(len(x) for x in data_decomposed_gplus["log_gplus"] if x is not None and bool(x)),
    ]
    assert np.allclose(expected_values, actual_values)

    # TODO: test a new decomposition round with n_max_comps


# @pytest.mark.skip(reason="Temporarily disabled to make tests run quicker")
def test_spatial_fitting_phase_1():
    from gausspyplus.spatial_fitting.spatial_fitting import SpatialFitting

    sp = SpatialFitting()
    sp.path_to_pickle_file = str(ROOT / "tests" / f"test_grs/gpy_prepared/{filepath.stem}.pickle")
    #  Filepath to the pickled dictionary of the decomposition results
    sp.path_to_decomp_file = str(ROOT / "tests" / f"test_grs/gpy_decomposed/{filepath.stem}_g+_fit_fin.pickle")
    sp.refit_blended = True
    sp.refit_neg_res_peak = True
    sp.refit_broad = True
    sp.flag_residual = True
    sp.refit_residual = True
    sp.refit_ncomps = True
    sp.use_ncpus = 1
    sp.log_output = True
    sp.verbose = True
    sp.spatial_fitting()

    with open(
        ROOT / "tests" / f"test_grs/gpy_decomposed/{filepath.stem}_g+_fit_fin_sf-p1.pickle",
        "rb",
    ) as pfile:
        data_spatial_fitted_phase_1 = pickle.load(pfile)

    # TODO: The spatial refitting seems to refit the spectra, because the values are slightly changed; this makes no
    #  difference to the results but might prolong the whole spatial refitting unnecessarily, because refit_iteration
    #  is still increased -> it's better in such cases to compare whether the number of components or fit values have
    #  changed substantially with np.allclose -> if not, the values from the previous iteration should be kept
    # TODO: check whether refit_iteration tracks the number of how often a spectrum has been refit
    expected_values = [
        1,
        272,
        4210.279292558555,
        572.5082530900359,
        0,
    ]
    actual_values = [
        sp.refitting_iteration,
        sum(filter(is_not_none, data_spatial_fitted_phase_1["N_components"])),
        sum(map(sum, filter(is_not_none, data_spatial_fitted_phase_1["fwhms_fit"]))),
        sum(map(sum, filter(is_not_none, data_spatial_fitted_phase_1["fwhms_fit_err"]))),
        sum(data_spatial_fitted_phase_1["refit_iteration"]),
    ]
    assert np.allclose(expected_values, actual_values)


# @pytest.mark.skip(reason="Temporarily disabled to make tests run quicker")
def test_spatial_fitting_phase_2():
    from gausspyplus.spatial_fitting.spatial_fitting import SpatialFitting

    sp = SpatialFitting()
    sp.path_to_pickle_file = str(ROOT / "tests" / f"test_grs/gpy_prepared/{filepath.stem}.pickle")
    #  Filepath to the pickled dictionary of the decomposition results
    sp.path_to_decomp_file = str(ROOT / "tests" / f"test_grs/gpy_decomposed/{filepath.stem}_g+_fit_fin_sf-p1.pickle")
    sp.refit_blended = False
    sp.refit_neg_res_peak = False
    sp.refit_broad = False
    sp.refit_residual = False
    sp.refit_ncomps = True
    sp.use_ncpus = 1
    sp.log_output = True
    sp.verbose = True
    sp.spatial_fitting(continuity=True)

    with open(
        ROOT / "tests" / f"test_grs/gpy_decomposed/{filepath.stem}_g+_fit_fin_sf-p2.pickle",
        "rb",
    ) as pfile:
        data_spatial_fitted_phase_2 = pickle.load(pfile)

    expected_values = [
        6,
        272,
        4210.279292558555,
        572.5082530900359,
        0,
    ]
    actual_values = [
        sp.refitting_iteration,
        sum(filter(is_not_none, data_spatial_fitted_phase_2["N_components"])),
        sum(map(sum, filter(is_not_none, data_spatial_fitted_phase_2["fwhms_fit"]))),
        sum(map(sum, filter(is_not_none, data_spatial_fitted_phase_2["fwhms_fit_err"]))),
        sum(data_spatial_fitted_phase_2["refit_iteration"]),
    ]
    assert np.allclose(expected_values, actual_values)


if __name__ == "__main__":
    test_prepare_cube()
    test_decompose_cube_gausspy()
    test_spatial_fitting_phase_1()
    test_spatial_fitting_phase_2()
