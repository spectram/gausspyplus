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

**Tier 1 — correctness, resolve/verify before publish:**
- `decomposition/decompose.py:65` — `cached_property` prevented `improve_fitting` toggling on a reused instance; `fitting` was already downgraded to `@property` as a workaround, but other cached properties (`dirpath`, `pickled_data`, …) still make instance reuse fragile.
- `training/training.py:1` — ragged-nested-sequence warning from GaussPy training; under numpy 2.x this is an **error**, not a warning, so the training stage likely breaks (untested by the suite — no training test exists).
- `parallel_processing/parallel_processing.py:72` — unhandled missing/None `idx` keyword in the decompose worker.
- `definitions/definitions.py:273` — `mask_out_ranges` mutable-default concern (currently `default=None`, so verify then drop the comment).
- `preparation/determine_intervals.py:132` — refactored interval buffering intentionally changed results vs. v0.2 (bugfix); validate on a real cube before publishing so the change is a release note, not a surprise.
- `spatial_fitting/spatial_fitting.py:855, 887, 693, 1719` — open questions in refit-loop logic (if/elif correctness, `is_successful_refit` semantics, neighbor-grouping condition, duplicate weight check).

**Tier 2 — safe mechanical cleanups (do as one commit each):**
- Delete dead code: `processing/spectral_cube_functions.py:483, 1145, 1799` (three unused functions), `parallel_processing/parallel_processing.py:177` (unused alternative multiprocessing path).
- Deduplicate: `decomposition/gaussian_functions.py:101` (identical function in `agd_decomposer`).
- Small robustness: `spatial_fitting/flags.py:115` (use `np.isclose`), `definitions/model.py:54` (`np.split`).

**Tier 3 — homogenization chores (batch together; touch many call sites):**
- `dirpath_gpy` vs `gpy_dirpath` naming (`decomposition/decompose.py:30`, `preparation/prepare.py:89`).
- rms as list-of-list → scalar (`preparation/prepare.py:201`, `training/training_set.py:137`).
- Return ranges as `np.ndarray` (`preparation/determine_intervals.py:83`, `preparation/noise_estimation.py:67`).
- Pickle-dict key homogenization between training set and decomposition (`plotting/plotting.py:252`).
- Renames flagged throughout (`gaussian_functions.py:12`, `noise_estimation.py:108-109`, `determine_intervals.py:84, 161`, `prepare.py:297`, `plotting.py:302`).

**Tier 4 — deferred science/algorithm questions (document, don't block release):**
- `decomposition/agd_decomposer.py:66` — derivative normalization (`np.diff(gauss2, 2) / dv**2`?).
- `decomposition/fit_quality_checks.py:174` — provenance of `separation_factor = 0.8493218` (likely the component-separation criterion from Riener+ 2019; document rather than change).
- Bootstrapped parameter errors (`gaussian_functions.py:95`, `gp_plus.py:57`), median vs mean grouping (`spatial_fitting/grouping.py:216`), training-set rchi2 limits (`training_set.py:53, 257`), `perform_final_fit=False` in training (`gradient_descent.py:87`).
- Recomputation caching in the spatial refit loop (`spatial_fitting.py:613, 619, 1497`) — performance only.

**Tier 5 — test debt:**
- `tests/integration_test.py:15` — stale intermediate pickles can mask failures across test ordering; add per-test cleanup/fixtures.
- No test covers the training stage (relevant to the Tier-1 numpy 2.x ragged-array risk).
- Untested `n_max_comps` decomposition round (`integration_test.py:135`), `refit_iteration` semantics (`integration_test.py:163-167`, `test_workflow.py:200`).
