#!/bin/bash
#SBATCH --job-name=DynaSurvHPO4
#SBATCH --output=%x.%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:2
#SBATCH --partition=ai
#SBATCH --signal=B:USR1@120

# Four independent one-GPU trial streams, TWO per node, NOT DDP.
# From the SHARED repository root: sbatch slurm/RunHPO.sh
# Default storage is one shared NFSv3+ journal; no database service is needed.
# Optional: HPO_STORAGE_BACKEND=rdb with OPTUNA_STORAGE_URL for PostgreSQL/MySQL.
# Never pass credentials in CLI arguments or enable shell tracing.
# N_TRIALS=100 is TOTAL additional attempts, divided across the four workers.
# PROJECT_DIR, DATA_DIR, SIF, N_TRIALS, MAX_EPOCHS, PATIENCE, METRIC, PRECISION, THREADS,
# STUDY_TAG, STUDY_NAME and heartbeat settings may be overridden in the environment.
# DRY_RUN=1 bash slurm/RunHPO.sh prints commands without touching files.
set +x
set -euo pipefail
umask 077

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
DATA_DIR=${DATA_DIR:-"$PROJECT_DIR/data"}
SIF=${SIF:-"$PROJECT_DIR/dynasurv.sif"}
HPO_STORAGE_BACKEND=${HPO_STORAGE_BACKEND:-nfs-journal}
STUDY_TAG=${STUDY_TAG:-hpo_v3}
STUDY_NAME=${STUDY_NAME:-dynasurv_$STUDY_TAG}
N_TRIALS=${N_TRIALS:-100}
MAX_EPOCHS=${MAX_EPOCHS:-100}
PATIENCE=${PATIENCE:-15}
METRIC=${METRIC:-average_ci}
PRECISION=${PRECISION:-bf16-mixed}
THREADS=${THREADS:-${SLURM_CPUS_PER_TASK:-4}}
HEARTBEAT_INTERVAL=${HEARTBEAT_INTERVAL:-60}
HEARTBEAT_GRACE_PERIOD=${HEARTBEAT_GRACE_PERIOD:-300}
case "$STUDY_TAG" in
    *[!a-zA-Z0-9_-]*|'') echo "STUDY_TAG must be a simple directory name" >&2; exit 1 ;;
esac
RUN_DIR="$PROJECT_DIR/studies/$STUDY_TAG"
case "$HPO_STORAGE_BACKEND" in
    nfs-journal) STORAGE_ARGS=(--storage "/workspace/studies/$STUDY_TAG/study.journal" --shared-journal) ;;
    rdb) STORAGE_ARGS=(--storage-url-env OPTUNA_STORAGE_URL) ;;
    *) echo 'HPO_STORAGE_BACKEND must be nfs-journal or rdb' >&2; exit 1 ;;
esac
COMMON=(python3 /workspace/scripts/hyperopt/run_optuna.py
    --config /workspace/configs/config.toml --data-dir /hpo_data
    "${STORAGE_ARGS[@]}" --study-name "$STUDY_NAME"
    --artifact-dir "/workspace/studies/$STUDY_TAG/trials"
    --out "/workspace/configs/$STUDY_TAG/best_config.json"
    --n-trials "$N_TRIALS" --workers 4 --nodes 2 --n-jobs 1
    --max-epochs "$MAX_EPOCHS" --patience "$PATIENCE"
    --metric "$METRIC" --accelerator gpu --precision "$PRECISION"
    --heartbeat-interval "$HEARTBEAT_INTERVAL" --heartbeat-grace-period "$HEARTBEAT_GRACE_PERIOD"
    --threads "$THREADS" --num-workers 0)
SRUN=(srun --nodes=2 --ntasks=4 --ntasks-per-node=2
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}"
    --gpus-per-task=h100:1 --gpu-bind=single:1 --kill-on-bad-exit=1)
CONTAINER=(apptainer exec --nv --bind "$PROJECT_DIR:/workspace" --bind "$DATA_DIR:/hpo_data:ro"
    --pwd /workspace --env PYTHONPATH=/workspace/src "$SIF")
PROBE=(python3 /workspace/scripts/hyperopt/check_shared_storage.py
    --directory "/workspace/studies/$STUDY_TAG/storage_checks/${SLURM_JOB_ID:-dry-run}"
    --workers 4 --nodes 2
    --launch-token-file "/workspace/studies/$STUDY_TAG/.launcher.lock/job_id"
    --launch-token "${SLURM_JOB_ID:-dry-run}")

if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf 'Study: %s; total additional trials: %s; 2 nodes x 2 H100s; storage: %s\n' "$STUDY_NAME" "$N_TRIALS" "$HPO_STORAGE_BACKEND"
    printf 'Container: '; printf '%q ' "${CONTAINER[@]}"; printf '\n'
    if [[ "$HPO_STORAGE_BACKEND" == nfs-journal ]]; then
        printf 'Storage probe init: '; printf '%q ' "${PROBE[@]}" --mode init; printf '\n'
        printf 'Storage probe workers: '; printf '%q ' "${SRUN[@]}" --time=00:05:00 apptainer exec '...' "${PROBE[@]}" --mode worker; printf '\n'
        printf 'Storage probe verify: '; printf '%q ' "${PROBE[@]}" --mode verify; printf '\n'
    fi
    printf 'Initialize: '; printf '%q ' "${COMMON[@]}" --worker-id 0 --init-only; printf '\n'
    printf 'Preflight: '; printf '%q ' "${SRUN[@]}" apptainer exec '...' "${COMMON[@]}" --preflight-only; printf '\n'
    printf 'Workers: '; printf '%q ' "${SRUN[@]}" apptainer exec '...' "${COMMON[@]}"; printf '\n'
    printf 'Export: '; printf '%q ' "${COMMON[@]}" --worker-id 0 --export-only; printf '\n'
    exit 0
fi
if [[ "$HPO_STORAGE_BACKEND" == rdb && -z ${OPTUNA_STORAGE_URL:-} ]]; then
    echo 'RDB mode requires OPTUNA_STORAGE_URL for PostgreSQL/MySQL. Default NFS journal mode needs no database.' >&2
    exit 1
fi
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Submit this launcher with sbatch' >&2; exit 1; }
[[ ${SLURM_NTASKS:-0} == 4 && ${SLURM_JOB_NUM_NODES:-0} == 2 ]] || {
    echo 'This launcher requires four tasks on TWO nodes, two tasks/GPUs per node' >&2; exit 1;
}
for shared_path in "$PROJECT_DIR" "$DATA_DIR" "$SIF"; do
    [[ "$shared_path" == /* && "$shared_path" != *:* && "$shared_path" != *,* ]] || {
        echo 'PROJECT_DIR, DATA_DIR and SIF must be absolute paths without colons or commas (Apptainer bind syntax).' >&2; exit 1;
    }
done
[[ -f "$SIF" && -d "$PROJECT_DIR/src/CausalSurv" && -d "$DATA_DIR" ]] || {
    echo "Check shared PROJECT_DIR, DATA_DIR ($DATA_DIR), and SIF ($SIF)" >&2; exit 1;
}
mkdir -p "$RUN_DIR"
LOCK_DIR="$RUN_DIR/.launcher.lock"
mkdir "$LOCK_DIR" || {
    echo "Another job or a stale lock owns $LOCK_DIR. Check the old job and its job_id marker before removing a stale lock." >&2
    exit 1
}
worker_pid=''
unlock() {
    code=$?
    trap - EXIT
    # Study state is already persistent. Remove only this job's marker/lock,
    # never trial artifacts, probe journals or the journal's own storage lock.
    if [[ -f "$LOCK_DIR/job_id" ]]; then
        unlink "$LOCK_DIR/job_id"
    fi
    rmdir "$LOCK_DIR"
    exit "$code"
}
stop_workers() {
    if [[ -n "$worker_pid" ]]; then
        kill -TERM "$worker_pid" 2>/dev/null || true
        wait "$worker_pid" || true
    fi
    exit 143
}
trap unlock EXIT
trap stop_workers USR1 TERM INT
printf '%s\n' "$SLURM_JOB_ID" > "$LOCK_DIR/job_id"
COMMON+=(--launch-token-file "/workspace/studies/$STUDY_TAG/.launcher.lock/job_id"
    --launch-token "$SLURM_JOB_ID")
# Explicit environment forwarding keeps credentials out of argv and logs.
if [[ "$HPO_STORAGE_BACKEND" == rdb ]]; then
    export OPTUNA_STORAGE_URL
    export APPTAINERENV_OPTUNA_STORAGE_URL="$OPTUNA_STORAGE_URL"
fi
export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 WANDB_MODE=disabled
if [[ "$HPO_STORAGE_BACKEND" == nfs-journal ]]; then
    # A separate, disposable study tests real cross-node visibility/locking.
    # It cannot contribute trials to the HPO budget or produce a model winner.
    "${CONTAINER[@]}" "${PROBE[@]}" --mode init
    "${SRUN[@]}" --time=00:05:00 "${CONTAINER[@]}" "${PROBE[@]}" --mode worker &
    worker_pid=$!
    wait "$worker_pid"
    worker_pid=''
    "${CONTAINER[@]}" "${PROBE[@]}" --mode verify
fi
# Explicit recovery is allowed only after this job owns the study lock and all
# prior workers have stopped. Never automatically convert active trials to FAIL.
if [[ ${RECOVER_STALE:-0} == 1 ]]; then
    "${CONTAINER[@]}" "${COMMON[@]}" --worker-id 0 --fail-stale-running
fi
"${CONTAINER[@]}" "${COMMON[@]}" --worker-id 0 --init-only
# Both nodes verify storage/data/device/protocol and the shared per-job output marker
# before any worker creates a trial. Global SLURM_PROCID gives worker IDs 0..3.
"${SRUN[@]}" "${CONTAINER[@]}" "${COMMON[@]}" --preflight-only &
worker_pid=$!
wait "$worker_pid"
worker_pid=''
"${SRUN[@]}" "${CONTAINER[@]}" "${COMMON[@]}" &
worker_pid=$!
wait "$worker_pid"
worker_pid=''
# One exporter after every worker succeeds; no export on worker failure.
"${CONTAINER[@]}" "${COMMON[@]}" --worker-id 0 --export-only
