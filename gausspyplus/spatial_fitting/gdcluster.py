"""Phase 3 spatially coherent refitting: GDCluster-style consensus refinement.

Implements the fit-refinement method of Liu & Du 2025 (GDCluster; arXiv:2509.16572), Sect. 2.2,
as an opt-in third phase of gausspyplus's spatially coherent refitting, complementary to phases
1/2 (Riener+2019, implemented in ``spatial_fitting.py``). Phases 1/2 are flag-triggered:
whole-solution swaps are only accepted if they improve AICc/quality flags. This phase is
consensus-driven and per-component: it is applied unconditionally to every sightline. Each
fitted Gaussian component is matched against its spatial neighbors' components (Eq. 13-14 of
the paper), components without enough neighbor support are dropped, components the neighbors
agree exist but are missing centrally are added, and the (possibly adjusted) component set is
refit once with parameter bounds taken from the matched neighbors -- the new fit is always
accepted (no AICc gate).

Deviations from the paper (verified against the arXiv PDF directly, not the abstract):

1. Matching operates on gausspyplus's *final fitted* components (``amplitudes_fit`` /
   ``means_fit`` / ``fwhms_fit``), not GDCluster's own pre-fit derivative-spectroscopy initial
   guesses (Sect. 2.1). gausspyplus does not persist per-sightline initial guesses at the
   spatial-fitting stage; this is the same input phases 1/2 already operate on.
2. ``mu_e_c`` (the central component's line-center uncertainty in Eq. 13) is the fitted mean
   uncertainty ``means_fit_err`` (lmfit stderr), with a configurable fallback
   (``gdcluster_mean_error_fallback``) when it is missing or zero -- not Eq. 12's analytic
   noise-propagation error for the derivative-spectroscopy estimate, which has no equivalent in
   gausspyplus's lmfit-based fitting.
3. The paper's Gaussian is parameterized by standard deviation ``sigma`` (their Eq. 1), while
   gausspyplus stores/fits FWHM throughout. All similarity/bound arithmetic below converts
   FWHM -> sigma via ``CONVERSION_STD_TO_FWHM`` and back.
4. The neighborhood ("angular radius of twice the angular resolution") is a circular disk in
   pixel space, not the square/rectangular block ``get_neighbors`` returns natively; see
   ``get_neighbors_within_radius``.
5. The paper does not specify how an "inferred Gaussian" for a missing component (Sect. 2.2) is
   computed from the matched neighbors, nor how components from different neighboring LoSs are
   grouped together first. Unmatched neighbor components are grouped by velocity via the
   existing ``group_fit_solutions`` (see ``spatial_fitting/grouping.py``), and each qualifying
   group's *median* amp/mean/fwhm is used -- deliberately different from that module's
   mean-based ``determine_average_values``, which remains untouched and used only by phases 1/2.
6. Parameter bounds for surviving/added central components are the min/max of the matched
   neighbors' values, per the paper -- widened to also include the central component's own
   current value so it is never outside its own fit bounds (an implementation necessity, not a
   deviation from the paper's substance). Bounds that still degenerate to a single point (every
   matched value numerically identical) are widened by a small epsilon, since lmfit rejects a
   parameter whose min equals its max.
7. If the refit yields zero valid components (all filtered out by the standard quality checks),
   or if there is nothing to fit at all (no surviving/added/prior components), the previous fit
   for that spectrum is left unchanged.
8. Numeric defaults for the optional weak/wide HI prior (``gdcluster_insert_weak_wide_prior``)
   are not specified by the paper (which only says "small upper bound for A and large lower
   bound for sigma"); see ``build_weak_wide_prior``.
9. One gpy+ "standard guardrail" is kept despite the "always accept" design (point 7 above): a
   refit is rejected -- the previous fit is retained -- if it introduces or worsens negative
   residual features (``N_neg_res_peak``), the same diagnostic ``gp_plus.check_for_negative_residual``
   computes for every fit and that phases 1/2's own ``_choose_new_fit`` already guards against via
   their flag comparison. A negative residual trough means the fitted model oversubtracts real
   flux -- a correctness defect, not a structural choice GDCluster's spatial consensus should be
   allowed to trade off. No other flag (blended components, rchi2, residual normality) is
   guarded this way; those remain "always accept" per the paper's design.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

from gausspyplus.decomposition.gaussian_functions import CONVERSION_STD_TO_FWHM
from gausspyplus.definitions.spectrum import Spectrum
from gausspyplus.spatial_fitting.grouping import get_neighbors, group_fit_solutions

if TYPE_CHECKING:
    from gausspyplus.spatial_fitting.spatial_fitting import SpatialFitting


def line_center_separation(
    mean_central: float,
    sigma_central: float,
    mean_neighbor: float,
    sigma_neighbor: float,
    mean_error_central: float,
    r_vdiff: float,
) -> float:
    """Relative line-center separation between a central and a neighboring Gaussian component.

    Liu & Du 2025 (arXiv:2509.16572), Eq. 13:
    mu_sep = |mu_c - mu_o| / (r_vdiff * min(sigma_c, sigma_o) + mu_e_c)

    :param sigma_central: Standard deviation (not FWHM) of the central component.
    :param sigma_neighbor: Standard deviation (not FWHM) of the neighboring component.
    :param mean_error_central: Line-center uncertainty of the central component (mu_e_c).
    :param r_vdiff: Hyper-parameter r_v_diff (default 1.22, "Rayleigh criterion").
    """
    return abs(mean_central - mean_neighbor) / (r_vdiff * min(sigma_central, sigma_neighbor) + mean_error_central)


def similarity(mu_sep: float) -> float:
    """Similarity between two Gaussian components from their relative line-center separation.

    Liu & Du 2025, Eq. 14: s = 1 - mu_sep**2 if mu_sep <= 1, else 0.
    """
    return 1 - mu_sep**2 if mu_sep <= 1 else 0.0


def match_one_neighbor(
    central_means: Sequence[float],
    central_sigmas: Sequence[float],
    central_mean_errors: Sequence[float],
    neighbor_means: Sequence[float],
    neighbor_sigmas: Sequence[float],
    r_vdiff: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the one-to-one assignment between the components of a central and one neighboring LoS.

    Liu & Du 2025, Sect. 2.2: "we construct a similarity matrix from Equation 14 and solve a
    one-to-one assignment problem. The objective is to maximize the sum of the selected
    similarities" -- done here per (central, single neighboring LoS) pair. Maximizing
    sum(similarity) is equivalent to minimizing sum(-similarity), which is what
    ``linear_sum_assignment`` solves directly.

    :return: (central_indices, neighbor_indices, similarities) for every pair returned by the
        assignment solver -- including pairs with zero similarity, i.e. not a real match.
        Callers should filter on ``similarities > 0``.
    """
    n_central, n_neighbor = len(central_means), len(neighbor_means)
    if n_central == 0 or n_neighbor == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([])

    similarity_matrix = np.zeros((n_central, n_neighbor))
    for c in range(n_central):
        for o in range(n_neighbor):
            mu_sep = line_center_separation(
                mean_central=central_means[c],
                sigma_central=central_sigmas[c],
                mean_neighbor=neighbor_means[o],
                sigma_neighbor=neighbor_sigmas[o],
                mean_error_central=central_mean_errors[c],
                r_vdiff=r_vdiff,
            )
            similarity_matrix[c, o] = similarity(mu_sep)

    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
    return row_ind, col_ind, similarity_matrix[row_ind, col_ind]


def get_neighbors_within_radius(location: Tuple[int, int], shape: Tuple[int, int], radius: float) -> np.ndarray:
    """Indices of all valid neighboring pixels within a circular radius (in pixels).

    Reuses ``get_neighbors`` (a square/rectangular neighborhood out to ``ceil(radius)`` pixels,
    ``gausspyplus/spatial_fitting/grouping.py``, unmodified) and filters to a circular disk,
    matching Liu & Du 2025's "angular radius" (Sect. 2.2) rather than a square block.
    """
    n_neighbors = int(np.ceil(radius))
    indices, coordinates = get_neighbors(
        location=location,
        shape=shape,
        n_neighbors=n_neighbors,
        return_indices=True,
        return_coordinates=True,
    )
    if indices.size == 0:
        return indices
    distances = np.linalg.norm(coordinates - np.array(location), axis=1)
    return indices[distances <= radius]


def _bounds_from_matches(matched_values: Sequence[float], own_value: float) -> Tuple[float, float]:
    """Min/max of matched neighbor values, widened to include the component's own current value.

    Liu & Du 2025, Sect. 2.2, step 5: parameter bounds are "the range defined by the minimum and
    maximum values of the corresponding parameters of the kinematically close neighbors".
    Widening to include ``own_value`` is an implementation necessity so the current/initial value
    is always within its own least-squares bounds (lmfit requires min <= value <= max); it does
    not change the paper's substance since ``own_value`` is the fit's own starting point.
    """
    values = list(matched_values) + [own_value]
    return min(values), max(values)


def _widen_degenerate_bounds(lo: float, hi: float, min_width: float = 1e-6) -> Tuple[float, float]:
    """Ensure a non-zero-width bound interval; lmfit requires ``min < max`` for every parameter.

    Bounds derived purely from matched-neighbor (and inferred-group) values can degenerate to a
    single point -- e.g. every matched neighbor and the component's own value are numerically
    identical -- which lmfit rejects with "Parameter 'pN' has min == max" (observed in practice
    on real data). Widening by a small epsilon in that case never changes a normal,
    non-degenerate bound.
    """
    if hi > lo:
        return lo, hi
    eps = max(abs(lo) * 1e-6, min_width)
    return lo - eps, hi + eps


def _median_group_values(
    unmatched_components: List[Tuple[float, float, float]],
    min_comp: int,
    mean_separation: float,
    fwhm_separation: float,
) -> List[Dict]:
    """Group unmatched neighboring components by velocity and infer missing central components.

    Liu & Du 2025, Sect. 2.2: "if the number of neighbors is greater than min_comp, a Gaussian
    inferred from neighbors is added to the central LoS." The paper does not specify how
    components from different neighboring LoSs are grouped together, nor how the inferred
    Gaussian's parameters are computed. Unmatched components (each a (amp, mean, fwhm) tuple
    from one neighboring LoS) are grouped by mean position with the existing
    ``group_fit_solutions`` (used unmodified; its own mean-based ``determine_average_values`` is
    not used here), and the *median* of each qualifying group is taken as the inferred
    component's initial guess; the group's min/max supplies its fit bounds (step 5).

    :return: One dict per inferred component with keys 'amp'/'mean'/'fwhm' (medians, fwhm in
        FWHM units) and 'amp_bounds'/'mean_bounds'/'fwhm_bounds' (group min/max).
    """
    if not unmatched_components:
        return []

    amps, means, fwhms = (np.array(x) for x in zip(*unmatched_components))
    sort_order = np.argsort(means)
    amps, means, fwhms = amps[sort_order], means[sort_order], fwhms[sort_order]

    grouped = group_fit_solutions(
        amps_tot=amps,
        means_tot=means,
        fwhms_tot=fwhms,
        split_fwhm=False,
        mean_separation=mean_separation,
        fwhm_separation=fwhm_separation,
    )

    inferred = []
    for group in grouped.values():
        group_amps = np.asarray(group["amps"])
        # Liu & Du 2025: components are added only if their support is *greater than* min_comp
        # (the removal criterion in step 3 uses "*smaller than* min_comp" for the complementary
        # direction) -- both boundaries are taken literally from the paper's wording.
        if len(group_amps) <= min_comp:
            continue
        group_means = np.asarray(group["means"])
        group_fwhms = np.asarray(group["fwhms"])
        amp_bounds = _widen_degenerate_bounds(max(0.0, float(np.min(group_amps))), float(np.max(group_amps)))
        mean_bounds = _widen_degenerate_bounds(float(np.min(group_means)), float(np.max(group_means)))
        fwhm_bounds = _widen_degenerate_bounds(max(0.0, float(np.min(group_fwhms))), float(np.max(group_fwhms)))
        inferred.append(
            {
                "amp": float(np.median(group_amps)),
                "mean": float(np.median(group_means)),
                "fwhm": float(np.median(group_fwhms)),
                "amp_bounds": list(amp_bounds),
                "mean_bounds": list(mean_bounds),
                "fwhm_bounds": list(fwhm_bounds),
            }
        )
    return inferred


def build_weak_wide_prior(
    spectrum: Spectrum,
    snr: float,
    min_fwhm: float,
    amp_max_factor: float,
    fwhm_min_factor: float,
) -> Dict:
    """A single weak, wide Gaussian prior component.

    Liu & Du 2025, Sect. 2.2: "several weak and wide Gaussians, with a small upper bound for A
    and a large lower bound for sigma, can be manually inserted as priors before the fitting
    process ... We recommend adding at least one Gaussian to every LoS of pervasive HI emission,
    and no additional components for CO data." Numeric bounds are not specified by the paper;
    ``amp_max_factor``/``fwhm_min_factor`` (settings ``gdcluster_weak_wide_prior_*``) are
    configurable.
    """
    amp_max = amp_max_factor * snr * spectrum.rms_noise
    fwhm_min = fwhm_min_factor * min_fwhm
    return {
        "amp_ini": 0.5 * amp_max,
        "mean_ini": float(np.mean(spectrum.channels)),
        "fwhm_ini": 1.5 * fwhm_min,
        "amp_bounds": [0.0, amp_max],
        "mean_bounds": [float(spectrum.channels[0]), float(spectrum.channels[-1])],
        "fwhm_bounds": [fwhm_min, None],
    }


def refit_spectrum_gdcluster(self: "SpatialFitting", index: int, i: int) -> List:
    """Phase 3 (GDCluster-style) refit of a single spectrum against its spatial neighbors.

    Implements Liu & Du 2025 (arXiv:2509.16572), Sect. 2.2, steps 1-7 (see module docstring for
    the full list of deviations from the paper). Unlike phases 1/2, the new fit is accepted if
    it succeeds without an AICc/flag-based accept/reject gate -- with one exception (deviation
    9): a refit that introduces or worsens negative residual features is rejected, keeping the
    previous fit.

    Uses the same call-signature convention as ``SpatialFitting.refit_spectrum_phase_1/2`` (an
    unbound function taking ``self`` explicitly) so it can be dispatched the same way by
    ``parallel_processing.py``.

    :param index: Index ('index_fit' keyword) of the spectrum that will be refit.
    :param i: List index into ``self.locations_refit`` for this spectrum.
    :return: [index, fit_results, neighbor_indices, is_refit_attempted], matching the phase 1/2
        result-tuple convention: ``fit_results`` is ``None`` if the spectrum was left unchanged.
    """
    location = self.locations_refit[i]
    neighbor_indices = get_neighbors_within_radius(
        location=location, shape=self.shape, radius=self.gdcluster_neighbor_radius
    )
    neighbor_indices = np.array(
        [idx for idx in neighbor_indices if self.decomposition["N_components"][idx]],
        dtype=int,
    )

    if neighbor_indices.size == 0:
        return [index, None, neighbor_indices, False]

    central_amps = list(self.decomposition["amplitudes_fit"][index] or [])
    central_means = list(self.decomposition["means_fit"][index] or [])
    central_fwhms = list(self.decomposition["fwhms_fit"][index] or [])
    central_mean_errs_raw = list(self.decomposition["means_fit_err"][index] or [])
    central_sigmas = [fwhm / CONVERSION_STD_TO_FWHM for fwhm in central_fwhms]
    central_mean_errs = [err if err else self.gdcluster_mean_error_fallback for err in central_mean_errs_raw]

    matched_neighbor_params: Dict[int, List[Tuple[float, float, float]]] = {c: [] for c in range(len(central_means))}
    unmatched_neighbor_components: List[Tuple[float, float, float]] = []

    for n_idx in neighbor_indices:
        n_amps = self.decomposition["amplitudes_fit"][n_idx]
        n_means = self.decomposition["means_fit"][n_idx]
        n_fwhms = self.decomposition["fwhms_fit"][n_idx]
        n_sigmas = [fwhm / CONVERSION_STD_TO_FWHM for fwhm in n_fwhms]

        row_ind, col_ind, sims = match_one_neighbor(
            central_means=central_means,
            central_sigmas=central_sigmas,
            central_mean_errors=central_mean_errs,
            neighbor_means=n_means,
            neighbor_sigmas=n_sigmas,
            r_vdiff=self.gdcluster_r_vdiff,
        )

        matched_neighbor_col_indices = set()
        for c, o, s in zip(row_ind, col_ind, sims):
            if s <= 0:
                continue
            matched_neighbor_params[c].append((n_amps[o], n_means[o], n_sigmas[o]))
            matched_neighbor_col_indices.add(o)

        for o in range(len(n_means)):
            if o not in matched_neighbor_col_indices:
                unmatched_neighbor_components.append((n_amps[o], n_means[o], n_fwhms[o]))

    fit_components: Dict[str, Dict] = {}

    #  Step 3: remove central components with too little neighbor support; keep the rest with
    #  bounds from step 5 ("smaller than min_comp" is removed, so ">= min_comp" is kept).
    for c in range(len(central_means)):
        matches = matched_neighbor_params[c]
        if len(matches) < self.gdcluster_min_comp:
            continue
        match_amps, match_means, match_sigmas = zip(*matches)
        amp_min, amp_max = _bounds_from_matches(match_amps, central_amps[c])
        mean_min, mean_max = _bounds_from_matches(match_means, central_means[c])
        sigma_min, sigma_max = _bounds_from_matches(match_sigmas, central_sigmas[c])
        amp_bounds = _widen_degenerate_bounds(max(0.0, amp_min), amp_max)
        mean_bounds = _widen_degenerate_bounds(mean_min, mean_max)
        fwhm_bounds = _widen_degenerate_bounds(
            max(0.0, sigma_min * CONVERSION_STD_TO_FWHM), sigma_max * CONVERSION_STD_TO_FWHM
        )
        fit_components[str(len(fit_components) + 1)] = {
            "amp_ini": central_amps[c],
            "mean_ini": central_means[c],
            "fwhm_ini": central_fwhms[c],
            "amp_bounds": list(amp_bounds),
            "mean_bounds": list(mean_bounds),
            "fwhm_bounds": list(fwhm_bounds),
        }

    #  Step 4: add components the neighbors agree exist but are missing centrally.
    for inferred in _median_group_values(
        unmatched_components=unmatched_neighbor_components,
        min_comp=self.gdcluster_min_comp,
        mean_separation=self.mean_separation,
        fwhm_separation=self.fwhm_separation,
    ):
        fit_components[str(len(fit_components) + 1)] = {
            "amp_ini": inferred["amp"],
            "mean_ini": inferred["mean"],
            "fwhm_ini": inferred["fwhm"],
            "amp_bounds": inferred["amp_bounds"],
            "mean_bounds": inferred["mean_bounds"],
            "fwhm_bounds": inferred["fwhm_bounds"],
        }

    spectrum = Spectrum(
        intensity_values=self.data[index],
        channels=self.channels,
        rms_noise=self.errors[index][0],
        signal_intervals=self.signal_intervals[index],
        noise_spike_intervals=self.noise_spike_intervals[index],
    )

    #  Step 7 (optional): pre-insert a weak, wide prior component, unconditionally.
    if self.gdcluster_insert_weak_wide_prior:
        fit_components[str(len(fit_components) + 1)] = build_weak_wide_prior(
            spectrum=spectrum,
            snr=self.snr,
            min_fwhm=self.min_fwhm,
            amp_max_factor=self.gdcluster_weak_wide_prior_amp_max_factor,
            fwhm_min_factor=self.gdcluster_weak_wide_prior_fwhm_min_factor,
        )

    if not fit_components:
        #  Everything was removed as noise and nothing qualified to be added: leave unchanged.
        return [index, None, neighbor_indices, False]

    #  Step 6: refit the original (unsmoothed) spectrum once, with the adjusted component set
    #  and bounds. Reuses the same lmfit + quality-control routine as phases 1/2.
    fit_results = self._gaussian_fitting(spectrum=spectrum, fit_components=fit_components)

    #  Deviation 9 / gpy+ standard guardrail: never accept a refit that introduces or worsens
    #  negative residual features, even though phase 3 otherwise always accepts a successful
    #  refit. Mirrors the diagnostic phases 1/2's own `_choose_new_fit` already guards against.
    if fit_results is not None:
        n_neg_res_before = self.decomposition["N_neg_res_peak"][index] or 0
        if fit_results["N_neg_res_peak"] > n_neg_res_before:
            fit_results = None

    return [index, fit_results, neighbor_indices, True]
