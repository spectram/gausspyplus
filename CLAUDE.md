# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Fork (spectram/gausspyplus) of [mriener/gausspyplus](https://github.com/mriener/gausspyplus): a fully automated Gaussian decomposition package for emission-line spectra (radio/mm astronomy), built on the Autonomous Gaussian Decomposition (AGD) algorithm from GaussPy. See Riener et al. 2019 for the science.

## Branch landscape (critical context)

All local branches fork from the same commit on `master` (a171796); none contains another. The goal of current work is to port local changes onto `refactor` and publish it.

- **`master`** — upstream v0.2-era code. Flat module layout (`gausspyplus/decompose.py`, `plotting.py`, …), legacy `astropy_helpers`/`setup.py` build, no real test suite (`gausspyplus/tests/test_example.py` is a stub).
- **`mainmod`** — current working branch: `master` + 3 small fixes (removed deprecated `np.int`, removed `overwrite=` kwarg from `fig.savefig`, renamed `save_initial_guesses` method to avoid clashing with the boolean attribute of the same name) + committed test-output pickles under `tests/test_grs/`.
- **`musecubes`** (= `merge-to-musecubes`) — `master` + ~11 commits of feature work: plotting enhancements (custom plot names, component-ratio annotations, mean-velocity lines), spatial_fitting phase-2 `np.delete` index bug fix, its own `save_initial_guesses` conflict fix, MUSE cube support snippets.
- **`refactor`** — 278 commits by upstream author Manuel Riener (2021–2023) modernizing the package, plus one local `setup` commit bumping dependency pins (astropy 5.3.4, numpy 1.22, scipy 1.9.2). Poetry build (`pyproject.toml`, python `>=3.9,<3.11`), real pytest suite, black (line length 119), pre-commit, type hints, dataclasses.

### Old → new module mapping (for porting changes to `refactor`)

| `master`/`mainmod` | `refactor` |
|---|---|
| `gausspyplus/decompose.py` | `gausspyplus/decomposition/decompose.py` |
| `gausspyplus/gausspy_py3/AGD_decomposer.py` | `gausspyplus/decomposition/agd_decomposer.py` |
| `gausspyplus/gausspy_py3/gradient_descent.py` | `gausspyplus/training/gradient_descent.py` |
| `gausspyplus/plotting.py` | `gausspyplus/plotting/plotting.py` |
| `gausspyplus/prepare.py` | `gausspyplus/preparation/prepare.py` |
| `gausspyplus/spatial_fitting.py` | `gausspyplus/spatial_fitting/spatial_fitting.py` |
| `gausspyplus/training.py`, `training_set.py` | `gausspyplus/training/` |
| `gausspyplus/config_file.py` | `gausspyplus/definitions/config_file.py` |
| `gausspyplus/finalize.py` | `gausspyplus/processing/finalize.py` |
| `example/` | `tutorials/` |

On `refactor`, per-spectrum fit bookkeeping lives in the `Model` dataclass (`gausspyplus/definitions/model.py`); settings/defaults are dataclasses in `gausspyplus/definitions/definitions.py`.

## Commands

### On `refactor`

```bash
poetry install          # supports Python >=3.10,<3.14 (numpy 2.x / scipy 1.15 / astropy 6.1 stack)
poetry run pytest tests/                       # full suite (~20 s)
poetry run pytest tests/test_workflow.py -k prepare   # single test
```

Tests read `data/grs-test_field_5x5.fits` (and `_10x10`) and write/clean outputs under `tests/test_grs/`. Golden-value assertions use `assert_stats_match`: integer stats (component counts) exact, float aggregates with rtol — fit results drift slightly between numpy/scipy/lmfit versions, and the iterative spatial refit amplifies this. If goldens need regenerating after a dependency bump, verify counts stay consistent and document old values in a comment. Formatting: `black` with line length 119 (enforced via pre-commit).

### On `master`/`mainmod`/`musecubes`

No meaningful automated tests. Verification is done by running the pipeline scripts in `example/` (`step_1-training_set--grs.py` … `step_6-spatial_refitting-p2--grs.py`) or the tutorial notebooks against the GRS test field data.

## Architecture: the decomposition pipeline

The package is a staged pipeline; each stage reads/writes pickled dicts (keys like `data_list`, `index`, `error`) in `gpy_prepared/`, `gpy_decomposed/` directories:

1. **Training set creation** (`training_set.py`) — extract spectra with known/fitted components.
2. **Training** (`training.py` + `gradient_descent.py`) — learn the two smoothing parameters (`alpha1`, `alpha2`) for AGD via gradient descent.
3. **Prepare** (`prepare.py`, noise estimation utilities) — estimate per-spectrum rms noise, mask channels, package a FITS cube into the pickle format.
4. **Decompose** (`decompose.py` → AGD decomposer + `gp_plus` improved-fitting routines) — initial Gaussian guesses via derivative spectroscopy, then quality-controlled least-squares fits (lmfit), flagging blended components / negative residuals.
5. **Spatial refitting** (`spatial_fitting.py`, phases 1 and 2) — refit spectra flagged as inconsistent with neighbors, enforcing spatial coherence.
6. **Finalize** (`finalize.py`) — assemble results into output tables/cubes.

Multiprocessing is funneled through `parallel_processing.py` (module-level state, `init`/`func` pattern — beware when refactoring). All stage classes share the config mechanism: attributes settable directly or via a `.ini` config file (`config_file.py` / `definitions/`).

**Settings-attribute shadowing hazard:** stage classes inherit settings as dataclass fields (`SettingsDefault`, `SettingsDecomposition`, …) from `definitions/definitions.py`. A method or property whose name equals a settings field shadows the field default via the MRO (see the `save_initial_guesses` fix, commit 471327f). When adding methods to stage classes, check the name against the settings dataclasses.

## TODO triage (2026-07; 114 `# TODO:` comments, all upstream author's notes)

Plan of record for working through them — **tiers 1–2 before publishing, the rest deferred**:

**Tier 1 — correctness: RESOLVED 2026-07.** Outcomes (details in the commit messages):
- Training stage was broken on Python >= 3.13 by the frame-locals hack in `objective_function` (PEP 667), **not** by the ragged-array warning (that root cause was already gone). Fixed; `tests/test_training.py` added. Caveat: training spawns a fresh multiprocessing Pool per objective evaluation — with the spawn start method (macOS default) a full 500-iteration training run is slow.
- `decompose.py` cached properties: path-derived ones are plain properties now; `logger`/`pickled_data` stay cached, so one `GaussPyDecompose` instance == one input file (settings like `improve_fitting` may be toggled between runs).
- `parallel_processing` decompose worker now tolerates pickles without an `"index"` key (plain-GaussPy format).
- `mask_out_ranges` default was already immutable (`None`); stale TODO dropped.
- Interval buffering: constant-pad bugfix validated on all 125 GRS spectra (68 get tighter signal ranges, old accelerating pad overshot `min_channels`) — **release-note this**. Restored a dropped v0.2 edge case: no-signal spectra still exclude `remove_intervals` from goodness-of-fit ranges.
- All four spatial-fitting refit-loop questions answered: the if/elif dispatch, skip condition, and weight filter are correct (now documented in place); `is_successful_refit` was renamed `is_refit_attempted` (it feeds the "Tried to refit" statistic; success == `fit_results is not None`).

Also fixed: `.gitignore` contained `*_*.py` / `*_*.ipynb`, silently ignoring any new Python file with an underscore in its name (e.g. new test modules).

**Tier 2 — safe mechanical cleanups: RESOLVED 2026-07.** Dead code deleted (three unused
`spectral_cube_functions` helpers, commented-out mp path); the dedup TODO was stale (already deduplicated);
`np.isclose` now used for the broad-flag FWHM comparison (eliminates re-refitting of float-identical fits —
refit_iteration goldens updated); `split_params` deliberately kept over `np.split` (preserves list pickle format).

**Tier 3 — homogenization chores: RESOLVED 2026-07.** Outcomes:
- Renamed `GaussPyTraining.gpy_dirpath` → `dirpath_gpy` (every other stage already used `dirpath_gpy`; the old
  name silently ignored the `dirpath_gpy` value that tutorial step 2 sets).
- Renames applied: `check_if_intervals_contain_signal` → `get_intervals_with_significant_signal`,
  `area_of_gaussian` → `integrated_area_under_gaussian_curve`, `GaussPyPrepare.calculate_rms_noise` →
  `prepare_spectrum`, `determine_peaks(peak=...)` → `peak_type=`.
- Removed the dead `Figure.max_rows_per_figure` cached_property (name-clashed with the dataclass field and was
  always shadowed; also fixed `Figure()` construction without that parameter).
- **Deliberately NOT changed (documented in place): the pickle formats.** rms stays a one-element list per
  spectrum, intervals stay plain lists, and training-set files keep 'fwhms'/'means'/'amplitudes' vs the
  decomposition files' '*_fit' keys — changing any of these would break compatibility with previously generated
  pickle files.

**Tier 4 — deferred science/algorithm questions (document, don't block release):**
- `decomposition/agd_decomposer.py:66` — derivative normalization (`np.diff(gauss2, 2) / dv**2`?).
- `decomposition/fit_quality_checks.py:174` — provenance of `separation_factor = 0.8493218` (likely the component-separation criterion from Riener+ 2019; document rather than change).
- Bootstrapped parameter errors (`gaussian_functions.py:95`, `gp_plus.py:57`), median vs mean grouping (`spatial_fitting/grouping.py:216`), training-set rchi2 limits (`training_set.py:53, 257`), `perform_final_fit=False` in training (`gradient_descent.py:87`).
- Recomputation caching in the spatial refit loop (`spatial_fitting.py:613, 619, 1497`) — performance only.

**Tier 5 — test debt:**
- `tests/integration_test.py:15` — stale intermediate pickles can mask failures across test ordering; add per-test cleanup/fixtures.
- No test covers the training stage (relevant to the Tier-1 numpy 2.x ragged-array risk).
- Untested `n_max_comps` decomposition round (`integration_test.py:135`), `refit_iteration` semantics (`integration_test.py:163-167`, `test_workflow.py:200`).
