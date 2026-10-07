# DynaSurv project check — 7 October 2026

## Scope and conclusion

Reviewed the active local working tree at Git HEAD `e9dcb87`, including existing
uncommitted HPO/configuration changes. The corrected survival, recommendation
and semisynthetic protocols are implemented, but the local artifacts do not
establish a completed corrected experiment cycle. This was a source,
configuration and artifact-inventory review, not a new scientific validation
or an inspection of remote cluster jobs.

The maintained entry point is [skills/README.md](../skills/README.md). Historical
September audits and numerical reports retain their original evidence; this
report supersedes their descriptions of *current implementation* where the code
has since changed.

## Component state

| Component | Verified current behavior | Local evidence boundary |
|---|---|---|
| Survival model | Learned static-state projections; training-only scaling/grid maximum; development validation separated from temporal test; both IPCW Brier weight arrays supplied; checkpoint manifests | One run retains a protocol-v2 manifest; its six checkpoint files were removed during cleanup. No training or checkpoint metric evaluation was performed here |
| Recommendation | Propensities over all observed classes; missing patient support yields no eligible choices; strict member provenance; one eligible arm is not a confident comparison | Only one active local run, below the configured two-member minimum; historical 40-model outputs are not current validation |
| Semisynthetic validation | Four-arm prefix DGP; original-patient grouped splits; fixed checkpoint rule; exact expected/latent factual scoring | No corrected `eval_v2_*` metadata found under local `models/` or `reports/` |
| HPO | Local v5 runner maximizes validation CI minus calibration gap; separate calibration stopping; fixed 100 intervals; four independent cluster workers | No local `studies/` directory or default exported winner; cluster launch not exercised |

The saved local run is
`models/HR+HER2-/4lines/23092026_150314_seed_1837416121/`. Its manifest records
3,371 train, 843 validation and 1,977 test patients, no separate early-stop
subset in final-training mode, and 100 intervals. These are stored metadata
counts, not a reconstruction of the present cohort or proof of fit quality.
No patient identifiers were copied into this report.

## Execution blockers and implementation gaps

1. **Default winning configuration is absent locally.** Training, recommendation
   and semisynthetic training/evaluation default to
   `configs/hpo_v3/best_config.json`. The strict loader requires an existing file.
   HPO can generate it without a previous winner, or those CLIs accept an
   explicit compatible `--model-config`. The default smoke command therefore
   cannot currently run with its defaults.
2. **HPO version/path and resume behavior need care.** Protocol v5 stores local
   and cluster studies under `studies/hpo_v5/`, but both export to
   `configs/hpo_v3/`. `load_if_exists=True` resumes fixed names without comparing
   current configuration/data/source against stored `protocol_details`.
   A later export can replace the shared winner. Export checks a trial protocol
   attribute, not the full identity of all executions in a resumed study.
3. **Old HPO safety/operations claims no longer match source.** The removed
   `check_shared_storage.py`, cross-node probe, preflight, launcher ownership
   lock, heartbeat, recovery CLI and database options are not part of the
   current minimal launcher. Journal file locking remains. Interrupted RUNNING
   trials block export; there is no current recovery option. This inspection
   does not establish whether the cluster journal filesystem works reliably.
4. **Recommendation plotting has not caught up with decision statuses.**
   `data_analysis/recommendation_mix.py::mix_shares` handles only `confident`,
   `undecided` and `no_support`. New `only_supported_option` rows would be
   omitted from the stacked bars. Its default report kind is `bestCALIB`,
   while the active recommendation config defaults to `val_loss`.
5. **Existing output/resource conventions differ from repository guidance.**
   The semisynthetic evaluator writes per-run CSVs beside checkpoints in
   `models/semisynthetic_v2/`; aggregates go to `reports/`. The sweep launcher
   requests four hours rather than the repository default of 24. These were
   documented, not silently changed as part of a documentation task.
6. **Some source comments and legacy paths remain historical.**
   `semisynthetic/drivers.py` still explains dynamic static-feature copies
   by saying the model ignores static inputs. The active model now uses them.
   Independent MH/multihead/residual implementations and legacy workflows have
   not been comprehensively re-audited here.

## Interpretation limits

The repaired code does not retroactively validate earlier numerical results or
undo prior inspection of the test cohort. The recommendation ensemble measures
agreement between model predictions; its confidence label does not establish
an individual treatment benefit. The semisynthetic evaluator checks current-line
choices on real observed histories, not full trajectories produced by following
the ensemble policy.

Training-only scaling and time-grid fitting are implemented, but the treatment
vocabulary still comes from the merged cohort. Upstream imputation and feature
timestamps were not re-audited. Real-data censoring assumptions, assignment-model
probability quality and causal identification are not established by the checks
below. No new performance comparison or methodological hypothesis is proposed.

## Documentation changes

- Populated the previously empty survival-model and semisynthetic component files.
- Added an HPO component document and a component index.
- Updated the recommendation document with current execution/provenance rules,
  distinguished historical figures, and repaired its relative source links.
- Rewrote the stale HPO run guide and eligibility-filter explanation.
- Updated README and CLAUDE guidance; repaired AGENTS component paths, added HPO
  maintenance mapping, and clarified the environment/smoke command.
- Linked historical state/audit reports to the current component documentation.

This report was written alongside the HPO, model and documentation changes and
later updated to reflect checkpoint cleanup and protocol v5. It does not include
a rerun of training or HPO. The default-named presentation was left out of
version control. `CLAUDE.md` is ignored by the existing rule, so its local
guidance is not part of these commits.

## Verification

- Parsed all **68** Python files under `src/` and `scripts/` with `ast.parse`:
  no syntax errors.
- Loaded real-data and semisynthetic TOML sections using the repository's strict
  typed configuration classes; loaded and validated the DGP config.
- Executed HPO `--help` in the existing `dynasurv_env`: imports succeeded and
  the parser exposes only `--export-only` besides standard help.
- Inspected artifact paths and aggregate manifest counts without loading patient
  data tables or evaluating model predictions.
- Checked all 74 local links and balanced code fences in the ten current
  documentation files; no broken links were found.
- `bash -n` passed for the HPO, seed-training and semisynthetic sweep launchers.
  The edited tracked documentation passes `git diff --check`. The full working
  tree check still flags a pre-existing blank line at EOF in `.gitignore`, which
  this review leaves unchanged.

The environment's Python executable was used directly because `mamba run`
could not write its cache lock inside the sandbox. Matplotlib used a temporary
cache during HPO import. No package installation was needed. Deprecated tests,
training, new HPO trials, remote jobs and manuscript compilation were not run.
