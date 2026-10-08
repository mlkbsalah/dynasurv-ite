# HPO run guide

Reviewed **8 October 2026** against the current working tree. The runner uses
HPO protocol **v4 validation-loss** and the corrected **v2 data protocol**. New
trials default to LayerNorm MLPs in a distinct `hpo_v4_val_loss` study namespace, away
from historical BatchNorm v4 and CI-minus-calibration v5 trials. Pass
`--mlp-normalization batch` to search BatchNorm MLPs in a separate namespace.
The previous v3 guide's configurable launch, storage
probe, preflight, recovery and metric-selection commands no longer apply.

The [component document](../../skills/hyperparameter_optimization.md) describes
the objective, search space, provenance and remaining implementation limits.

## Cluster launch

From the shared project on the cluster:

```bash
cd /home/m-ben-salah/repos/dynasurv-ite
sbatch slurm/RunHPO.sh
```

For BatchNorm, submit from the same directory:

```bash
sbatch --export=ALL,HPO_NORMALIZATION=batch slurm/RunHPO.sh
```

The launcher passes the same normalization mode to all workers and to the
final export step. Its default remains LayerNorm.

The launcher requests partition `ai`, two nodes, two H100 GPUs per node,
four tasks (two per node), eight CPUs per task, **16G memory per node**, and
24 hours. `srun` binds one GPU to each task. Each task independently trains
Optuna trials; this is not distributed training of one model.

It binds `/home/m-ben-salah/repos/dynasurv-ite` to `/workspace` inside
`dynasurv.sif`, then runs `hyperopt/run_optuna.py` from `/workspace/scripts`
with `PYTHONPATH=/workspace/src`. The repository, data, container and shared
journal location must be available to both nodes. The current launcher does
not perform the old cross-node storage probe or GPU preflight.

There are 100 additional trial attempts per launch, divided into 25 per worker.
GPU training uses BF16 mixed precision. Optuna minimizes the best validation-epoch
`val_loss`; pruning and early stopping monitor the same loss. Early stopping has
patience 10 and training has a maximum of 100 epochs. CI, IBS and calibration
gap remain trial diagnostics. These constants
are in the Python source; old `N_TRIALS`, `METRIC`, `STUDY_TAG`, `DRY_RUN` and
`HPO_STORAGE_BACKEND` overrides have no effect in this launcher.

After all workers succeed, the launcher invokes `--export-only` once in the
same Slurm job environment. If workers fail, `set -e` prevents that export.

## Local launch

From the repository root with data and the existing environment available:

```bash
mamba activate dynasurv_env
python3 scripts/hyperopt/run_optuna.py --help
python3 scripts/hyperopt/run_optuna.py
```

For a local BatchNorm HPO run, use
`python3 scripts/hyperopt/run_optuna.py --mlp-normalization batch`. To export
its completed winner later, add `--export-only` to that command.

The second command starts 100 real trial attempts on one device. Local mode
chooses CUDA, then MPS, then CPU and automatically exports after optimization.
No prior winning JSON is required. The runner reads `configs/config.toml` and
resolves its data path relative to the config directory.

To export an existing completed local study without fitting:

```bash
python3 scripts/hyperopt/run_optuna.py --export-only
```

Study selection is automatic: `SLURM_JOB_ID` selects the cluster journal;
without it, `--export-only` reads the local journal. There is no CLI selector
for another study or journal.

## Outputs

| Output | Path |
|---|---|
| Local study | `studies/hpo_v4_val_loss/local/study.journal` (`dynasurv_hpo_v4_val_loss_local`) |
| Cluster study | `studies/hpo_v4_val_loss/cluster/study.journal` (`dynasurv_hpo_v4_val_loss_cluster`) |
| Winning configuration, both modes | `configs/hpo_v3/best_config.json` |
| Exported provenance | `configs/hpo_v3/best_config.provenance.json` |
| BatchNorm local/cluster study | `studies/hpo_v4_val_loss_batchnorm/{local,cluster}/study.journal` |
| BatchNorm winner and provenance | `configs/hpo_v4_batchnorm/best_config.json` and `best_config.provenance.json` |

The `hpo_v3` export directory is retained by the current downstream defaults;
it does not mean the study uses objective v3. Local and cluster exports target
the same file and replace the earlier v5 winner; save its config and provenance
before launching if needed. Trials store exact model settings, scores and best-epoch
attributes in the journal. **No trial checkpoints are written.** Refit the
winner with `TrainDynasurvCausal.py` to obtain model weights and recommendation
support state.

The exported architecture records `mlp_normalization`. Refit a BatchNorm
winner from the `scripts/` directory with:

```bash
python3 TrainDynasurvCausal.py --model-config ../configs/hpo_v4_batchnorm/best_config.json
```

Older configs without this field continue to select LayerNorm. Keep the
matching `--mlp-normalization` setting when exporting either study.

Export rejects unresolved RUNNING trials, the absence of a finite completed
trial, or a winning trial with the wrong protocol attribute. It writes the
configuration and provenance using atomic file replacements.

## Resume and validation limits

`load_if_exists=True` resumes a fixed study name and adds another per-launch
budget. Recorded `protocol_details` are not compared with current data/config
or source before resuming. The previous fingerprint gate, launcher ownership
lock, heartbeat, stale-trial recovery and database backend options are absent.
Journal storage retains exclusive-create locking, but interrupted locks/trials
have no recovery CLI in the current runner.

The earlier v5 winner is available locally; no v4 validation-loss trial or
cluster job was run during this change. Source validity does not demonstrate
multi-node storage reliability or a completed v4 tuning result.
