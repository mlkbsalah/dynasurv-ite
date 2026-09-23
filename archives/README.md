# Historical artifacts — not eligible for current model selection

`legacy_pre_v2_2026-09-23/artifacts/` preserves the original relative paths of
the old model runs, evaluations, Optuna databases, logs and their winning configs.
These used protocols predating the data/model corrections and must not be mixed
with new runs. `manifest.json` records every file's size and SHA-256, move status,
and the result of post-move verification. No files are deleted or overwritten.

To recover an artifact, copy its path beneath `artifacts/` to a separate historical
analysis directory. To restore its original location, first ensure that location
does not exist, then move that specific artifact back. Do not restore legacy
configs or runs over new ones. Do not execute an archived checkpoint from an
untrusted source: PyTorch checkpoints may contain pickle objects.

The archive utility defaults to a dry-run and protects modern/ambiguous runs;
if a subtype tree mixes old and new, only verified legacy run directories move.
The newer `23092026_150314_seed_1837416121` run was preserved. Other worktrees and remote
cluster files are not modified. A later `make sync` can pull old remote runs back;
check the source before syncing. Keep this archive out of training input paths.

New HPO studies use `studies/hpo_v3/`; their selected configuration is exported to
`configs/hpo_v3/best_config.json`. This file is deliberately absent until a new
study has completed. Fresh real and semi-synthetic training must use corrected
data protocols, not the archived winning configurations.
