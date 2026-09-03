"""pytest tests for module spatial_fitting/gdcluster.py (Phase 3 / GDCluster helper functions)"""

import numpy as np

from gausspyplus.spatial_fitting.gdcluster import (
    line_center_separation,
    similarity,
    match_one_neighbor,
    get_neighbors_within_radius,
    _bounds_from_matches,
    _widen_degenerate_bounds,
    _median_group_values,
    build_weak_wide_prior,
)
from gausspyplus.definitions.spectrum import Spectrum


def test_line_center_separation_and_similarity_identical_components():
    #  Identical central and neighboring components: mu_sep = 0 -> similarity = 1.
    mu_sep = line_center_separation(
        mean_central=10.0,
        sigma_central=2.0,
        mean_neighbor=10.0,
        sigma_neighbor=2.0,
        mean_error_central=0.1,
        r_vdiff=1.22,
    )
    assert mu_sep == 0.0
    assert similarity(mu_sep) == 1.0


def test_line_center_separation_matches_eq_13():
    #  Liu & Du 2025, Eq. 13: mu_sep = |mu_c - mu_o| / (r_vdiff * min(sigma_c, sigma_o) + mu_e_c)
    mu_sep = line_center_separation(
        mean_central=10.0,
        sigma_central=3.0,
        mean_neighbor=13.0,
        sigma_neighbor=2.0,
        mean_error_central=0.5,
        r_vdiff=1.22,
    )
    expected = 3.0 / (1.22 * 2.0 + 0.5)
    assert np.isclose(mu_sep, expected)


def test_similarity_boundary_at_mu_sep_equals_one():
    assert similarity(1.0) == 0.0
    assert similarity(0.999999) > 0.0
    assert similarity(1.5) == 0.0


def test_similarity_eq_14_formula():
    for mu_sep in [0.0, 0.2, 0.5, 0.9]:
        assert np.isclose(similarity(mu_sep), 1 - mu_sep**2)


def test_match_one_neighbor_recovers_permuted_identical_components():
    #  Three central components, three neighboring components that are the same physical
    #  components in a different order: the one-to-one assignment should recover the correct
    #  permutation with similarity == 1 for every matched pair.
    central_means = [10.0, 20.0, 30.0]
    central_sigmas = [2.0, 2.0, 2.0]
    central_mean_errors = [0.2, 0.2, 0.2]
    neighbor_means = [30.0, 10.0, 20.0]
    neighbor_sigmas = [2.0, 2.0, 2.0]

    row_ind, col_ind, sims = match_one_neighbor(
        central_means=central_means,
        central_sigmas=central_sigmas,
        central_mean_errors=central_mean_errors,
        neighbor_means=neighbor_means,
        neighbor_sigmas=neighbor_sigmas,
        r_vdiff=1.22,
    )
    assert len(row_ind) == 3
    assert np.all(np.isclose(sims, 1.0))
    #  central index 0 (mean 10) should match neighbor index 1 (mean 10), etc.
    expected_col_for_row = {0: 1, 1: 2, 2: 0}
    for r, c in zip(row_ind, col_ind):
        assert c == expected_col_for_row[r]


def test_match_one_neighbor_excludes_far_pair_via_zero_similarity():
    #  One good match, one neighbor component far away from any central component -- the
    #  assignment is still forced to pair it with something, but the similarity must be 0 (not
    #  a "real" match), so callers filtering on similarity > 0 correctly exclude it.
    central_means = [10.0]
    central_sigmas = [1.0]
    central_mean_errors = [0.1]
    neighbor_means = [10.0, 500.0]
    neighbor_sigmas = [1.0, 1.0]

    row_ind, col_ind, sims = match_one_neighbor(
        central_means=central_means,
        central_sigmas=central_sigmas,
        central_mean_errors=central_mean_errors,
        neighbor_means=neighbor_means,
        neighbor_sigmas=neighbor_sigmas,
        r_vdiff=1.22,
    )
    #  min(n_central, n_neighbor) == 1 pair returned
    assert len(row_ind) == 1
    #  the good match (neighbor index 0) is the one selected, with similarity close to 1
    assert col_ind[0] == 0
    assert sims[0] > 0.99


def test_match_one_neighbor_empty_inputs():
    row_ind, col_ind, sims = match_one_neighbor(
        central_means=[],
        central_sigmas=[],
        central_mean_errors=[],
        neighbor_means=[1.0],
        neighbor_sigmas=[1.0],
        r_vdiff=1.22,
    )
    assert row_ind.size == 0 and col_ind.size == 0 and sims.size == 0

    row_ind, col_ind, sims = match_one_neighbor(
        central_means=[1.0],
        central_sigmas=[1.0],
        central_mean_errors=[0.1],
        neighbor_means=[],
        neighbor_sigmas=[],
        r_vdiff=1.22,
    )
    assert row_ind.size == 0 and col_ind.size == 0 and sims.size == 0


def test_get_neighbors_within_radius_is_circular_not_square():
    #  At radius=1.0 only the 4 orthogonal neighbors (distance 1) should be included; the 4
    #  diagonal neighbors (distance sqrt(2) ~= 1.414) must be excluded, unlike a square block.
    shape = (5, 5)
    location = (2, 2)
    neighbors_r1 = get_neighbors_within_radius(location=location, shape=shape, radius=1.0)
    assert sorted(neighbors_r1.tolist()) == sorted(
        [
            np.ravel_multi_index((1, 2), shape),
            np.ravel_multi_index((3, 2), shape),
            np.ravel_multi_index((2, 1), shape),
            np.ravel_multi_index((2, 3), shape),
        ]
    )

    #  At radius=1.5 the diagonal neighbors (distance ~1.414) are included too -> full 8-neighborhood.
    neighbors_r15 = get_neighbors_within_radius(location=location, shape=shape, radius=1.5)
    assert len(neighbors_r15) == 8
    assert set(neighbors_r1.tolist()).issubset(set(neighbors_r15.tolist()))


def test_get_neighbors_within_radius_respects_shape_bounds():
    #  Corner pixel: out-of-bounds neighbors must not appear regardless of radius.
    neighbors = get_neighbors_within_radius(location=(0, 0), shape=(3, 3), radius=1.5)
    assert len(neighbors) == 3  # (0,1), (1,0), (1,1) are the only in-bounds pixels within radius 1.5


def test_bounds_from_matches_widens_to_include_own_value():
    #  Own value inside the matched range: bounds equal the matched min/max.
    assert _bounds_from_matches([1.0, 2.0, 3.0], own_value=2.0) == (1.0, 3.0)
    #  Own value outside the matched range: bounds widen to include it.
    assert _bounds_from_matches([1.0, 2.0, 3.0], own_value=5.0) == (1.0, 5.0)
    assert _bounds_from_matches([1.0, 2.0, 3.0], own_value=-1.0) == (-1.0, 3.0)


def test_widen_degenerate_bounds_only_touches_degenerate_input():
    #  Non-degenerate bounds pass through unchanged.
    assert _widen_degenerate_bounds(1.0, 2.0) == (1.0, 2.0)
    #  min == max must be widened so lmfit never sees an unusable "min == max" parameter bound.
    lo, hi = _widen_degenerate_bounds(5.0, 5.0)
    assert lo < 5.0 < hi
    #  Degenerate at exactly zero (e.g. an all-zero matched group) must still widen.
    lo, hi = _widen_degenerate_bounds(0.0, 0.0)
    assert lo < 0.0 < hi


def test_median_group_values_never_returns_degenerate_bounds():
    #  A group whose members are all numerically identical would otherwise produce amp/mean/fwhm
    #  bounds with min == max, which lmfit rejects ("Parameter 'pN' has min == max") -- this was
    #  observed on real data (see the phase 3 notebook run).
    identical_group = [(2.0, 42.0, 5.0) for _ in range(7)]

    inferred = _median_group_values(identical_group, min_comp=6, mean_separation=2.0, fwhm_separation=4.0)

    assert len(inferred) == 1
    component = inferred[0]
    for key in ["amp_bounds", "mean_bounds", "fwhm_bounds"]:
        lo, hi = component[key]
        assert hi > lo, f"{key} is degenerate: {component[key]}"


def test_median_group_values_min_comp_threshold():
    #  Two velocity clusters of unmatched neighboring components: one with 7 members (> min_comp
    #  of 6, should be added), one with 4 members (<= min_comp, should not).
    big_group = [(1.0 + 0.01 * i, 50.0 + 0.1 * i, 4.0) for i in range(7)]
    small_group = [(1.0, 100.0, 4.0) for _ in range(4)]
    unmatched = big_group + small_group

    inferred = _median_group_values(unmatched, min_comp=6, mean_separation=2.0, fwhm_separation=4.0)

    assert len(inferred) == 1
    component = inferred[0]
    assert 49.0 < component["mean"] < 51.0
    assert np.isclose(component["amp"], np.median([c[0] for c in big_group]))
    assert component["mean_bounds"][0] <= component["mean"] <= component["mean_bounds"][1]


def test_median_group_values_empty_input():
    assert _median_group_values([], min_comp=6, mean_separation=2.0, fwhm_separation=4.0) == []


def test_build_weak_wide_prior_bounds():
    spectrum = Spectrum(intensity_values=np.zeros(100), channels=np.arange(100), rms_noise=0.1)
    prior = build_weak_wide_prior(spectrum=spectrum, snr=3.0, min_fwhm=3.0, amp_max_factor=0.3, fwhm_min_factor=4.0)
    amp_min, amp_max = prior["amp_bounds"]
    fwhm_min, fwhm_max = prior["fwhm_bounds"]
    assert amp_min == 0.0
    assert amp_max == 0.3 * 3.0 * 0.1
    assert amp_min <= prior["amp_ini"] <= amp_max
    assert fwhm_min == 4.0 * 3.0
    assert fwhm_max is None
    assert fwhm_min <= prior["fwhm_ini"]
