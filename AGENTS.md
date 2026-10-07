# Repository Guidelines

## Project Structure & Module Organization

DynaSurv is a Python framework for multi-line survival analysis and treatment recommendation with causal inference. Core code lives in `src/CausalSurv/`: `data/` prepares cohorts, `model/` implements Lightning models, `metrics/` defines losses, `evaluation/` computes survival metrics, `recommendation/` assembles treatment recommendations, and `semisynthetic/` supports benchmarks. Executable workflows live in `scripts/`, with Optuna tooling in `scripts/hyperopt/` and cluster launchers in `slurm/`. Use `configs/` for experiment settings. Analysis and publication material lives in `notebooks/`, `data_analysis/`, `reports/`, and `latex/`.

## Build, Test, and Development Commands

Use the locally defined mamba environment `dynasurv_env`:

- `mamba activate dynasurv_env`
- `cd scripts && python3 TrainDynasurvCausal.py --fast_dev_run`: requires `configs/hpo_v3/best_config.json` or an explicit `--model-config`. Currently caps training at three full epochs; it is not Lightning's one-batch fast-dev mode. Use only to detect execution errors, not to interpret model performance.
- `pre-commit run --all-files`: run the pinned Ruff lint/fix and formatting hooks.
- `make build-docker`: build the Linux AMD64 image and export `dynasurv.tar`. Used for submitting container job on HPC.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for functions and variables, and `PascalCase` for classes. Follow surrounding module conventions and use typed configuration dataclasses in `src/CausalSurv/config.py`. Ruff enables error, undefined-name, and import-order checks (`E`, `F`, `I`), with `E501` ignored. Hooks may modify staged files; review and restage their changes.

## Testing Guidelines

Ignore testing; `test/` is deprecated.

## Commit & Pull Request Guidelines

Follow the history's prefixes: `fix:`, `feat:`, `docs:`, and `chore:`. Keep subjects specific and imperative.

## Data & Experiment Hygiene

Keep patient data, credentials, checkpoints, and generated studies out of commits. Preserve cohort splits and experiment provenance. Experiment results (CSVs, parquet, plots, pdfs) that are asked should be placed in the `reports/` folder under the convenient subfolder. This does not apply for model checkpoints that are in `models/`

## Citation and fact checking

Do not hallucinate citations if asked to look for citations, make sure to search for DOI of citation and verify the paper before adding citation.
For every suggested idea or hypothesis accompany it with 2 DOI links.
When analyzing plots or numerical results explain clearly and in simple terms the drawn results, and critically challenge every claim by stating where it could fail.

## HPC cluster description and slurm guidelines

The cluster is constituted of 2 GPU nodes housing 2 H100 GPUs each. There is one single partition on the cluster named `ai`.

### Guidelines

- Keep scripts minimal do not add options unless asked to.
- Set by default the jobs to run for 24h
- Set memory to 16G
- Set cpu per task to 8
- include the following lines for specification

```{bash}
PROJECT_DIR="/home/m-ben-salah/repos/dynasurv-ite"
CONTAINER_NAME="dynasurv.sif"
SIF="$PROJECT_DIR/$CONTAINER_NAME"

time apptainer exec --nv --bind "$PROJECT_DIR:/workspace" "$SIF" \
    bash -c "cd /workspace/scripts && PYTHONPATH=/workspace/src python3 <script name>"
```

## Project maintenance and updates

Every time there is a modification in one or more of the pipeline components: the causal survival model, the recommendation algorithm, the semisynthetic validation or hyperparameter optimization, make sure to modify the corresponding skill according to the table:

|component|skill|
|---|---|
|recommendation|`skills/recommendation.md`|
|semisynthetic validation|`skills/semisynthetic_validation.md`|
|survival_model|`skills/survival_model.md`|
|hyperparameter optimization|`skills/hyperparameter_optimization.md` and `scripts/hyperopt/README.md`|

These paths are relative to the repository root. Start with `skills/README.md`
for component status and documentation links. Keep historical experiment reports
distinct from current implementation documentation.
