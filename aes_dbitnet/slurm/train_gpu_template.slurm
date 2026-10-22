#!/bin/bash
# ======================================================================================================
# GPU TEMPLATE - NOT RUNNABLE AS-IS. CONDITIONAL on administrator-approved GPU allocation syntax.
# The GPU request lines below are deliberately absent: do not guess --gres / --gpus / constraint syntax.
# Obtain the supported request form from your cluster documentation or support, insert it at the marked
# place, and test with a short job first. Do not use this to work around scheduler restrictions.
# ======================================================================================================
#SBATCH --job-name=aesdist-gpu
#SBATCH --partition=CHANGE_ME          # the GPU partition name from official docs
#SBATCH --account=CHANGE_ME
#SBATCH --qos=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=CHANGE_ME
#SBATCH --mem=CHANGE_ME
#SBATCH --time=CHANGE_ME
# <<< INSERT ADMINISTRATOR-APPROVED GPU RESOURCE REQUEST LINE(S) HERE >>>
#SBATCH --signal=B:USR1@300
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
if grep -q "CHANGE_ME" "${BASH_SOURCE[0]}"; then echo "ERROR: placeholders remain." >&2; exit 2; fi
if ! grep -q "^#SBATCH.*\(gres\|gpus\)" "${BASH_SOURCE[0]}" && ! grep -q "APPROVED_GPU_SYNTAX_CONFIRMED=1" "${BASH_SOURCE[0]}"; then
  echo "ERROR: no GPU request present. Insert the approved syntax and add the line APPROVED_GPU_SYNTAX_CONFIRMED=1 in a comment." >&2; exit 2
fi

PROJECT_DIR="${PROJECT_DIR:?}"; export NDGOHR_BASELINE_DIR="${NDGOHR_BASELINE_DIR:?}"; export AESDIST_OUT_DIR="${AESDIST_OUT_DIR:?}"
ANACONDA_MODULE="${ANACONDA_MODULE:?}"; CONDA_ENV="${CONDA_ENV:?GPU-capable env (see requirements-gpu.txt)}"
RUN_ARGS=("$@")

module load "$ANACONDA_MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
# NOTE: CUDA_VISIBLE_DEVICES is set by SLURM for the allocation; this script never overrides it.
echo "job=$SLURM_JOB_ID host=$(hostname)"; env | grep -E '^(SLURM_.*GPU|CUDA_VISIBLE_DEVICES)' || echo "no GPU-related SLURM env vars present"
nvidia-smi -L || echo "nvidia-smi unavailable (informational only; allocation is NOT inferred from it)"
# --device gpu: run_experiment.py refuses unless SLURM_JOB_ID is set AND TensorFlow itself reports a GPU.
# It never falls back to CPU silently; a non-zero exit means the GPU was unusable.
python "$PROJECT_DIR/scripts/run_experiment.py" "${RUN_ARGS[@]:-smoke}" --device gpu
