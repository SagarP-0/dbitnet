#!/bin/bash
# CPU batch job. EDIT every CHANGE_ME using your cluster's documented procedure; the script refuses to
# run while any placeholder remains. No GPU is requested or used. Nothing is installed.
#SBATCH --job-name=aesdist-cpu
#SBATCH --partition=CHANGE_ME
#SBATCH --account=CHANGE_ME
#SBATCH --qos=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=CHANGE_ME
#SBATCH --mem=CHANGE_ME
#SBATCH --time=CHANGE_ME
#SBATCH --signal=B:USR1@300        # warn 5 min before wall-time; train.py stops at the next batch (best effort)
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
if grep -q "CHANGE_ME" "${BASH_SOURCE[0]}"; then
  echo "ERROR: placeholders (CHANGE_ME) remain in this script. Edit it first." >&2; exit 2
fi

# ---- user-editable settings (user-owned paths only) -------------------------------------------
PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR to the aes_dbitnet directory}"
export NDGOHR_BASELINE_DIR="${NDGOHR_BASELINE_DIR:?set to a read-only checkout of the original NDGohr repo}"
export AESDIST_OUT_DIR="${AESDIST_OUT_DIR:?set to a user-owned, persistent output directory}"
ANACONDA_MODULE="${ANACONDA_MODULE:?name of the Anaconda module on your cluster}"
CONDA_ENV="${CONDA_ENV:-aes-cryptanalysis}"
RUN_ARGS=("$@")                    # e.g. smoke   |   train --experiment r3_2B --arch dbitnet --mode smoke
# -----------------------------------------------------------------------------------------------

module load "$ANACONDA_MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "job=$SLURM_JOB_ID host=$(hostname) date=$(date -Is)"
echo "partition=${SLURM_JOB_PARTITION:-?} cpus=${SLURM_CPUS_PER_TASK} mem=${SLURM_MEM_PER_NODE:-?}"
python --version
python - <<'PY'
import tensorflow as tf
print("tensorflow", tf.__version__, "| visible GPUs (will be hidden in cpu mode):", [g.name for g in tf.config.list_physical_devices("GPU")])
PY
python "$PROJECT_DIR/scripts/run_experiment.py" env-report > "$AESDIST_OUT_DIR/env_info_job${SLURM_JOB_ID}.json" || true

# --device cpu hides GPUs from TensorFlow, so a GPU can never be used by accident.
python "$PROJECT_DIR/scripts/run_experiment.py" "${RUN_ARGS[@]:-smoke}" --device cpu
