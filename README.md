# aes_dbitnet — DBitNet vs NDGohr on reduced-round AES-128 (skeleton + CPU smoke stage)

**Status.** This is stage 1: project skeleton, validation tests, CPU smoke pipeline. **No benchmark result exists yet.**
The notebook (`aes_dbitnet_benchmark.ipynb`), full dataset generation, multi-seed training, the 2^16 WKRP run,
and the complexity study are deliberately not started. Metrics printed by the smoke test come from ~4k-example toy
runs and are *not* results.

## Layout
```
config/experiments.yaml     all settings (modes, protocols, architectures, experiment list)
src/aesdist/                library (baseline_adapter, datagen, bits, models, train, checkpoint, evaluate,
                            wkrp, partial_decrypt, validation, reference_aes, complexity, ...)
scripts/run_experiment.py   non-interactive entry point (env-report | smoke | gen-data | train)
scripts/env_report.py       read-only package report for your conda env
slurm/train_cpu.slurm       CPU job (placeholders must be filled)
slurm/train_gpu_template.slurm   GPU TEMPLATE - conditional, no GPU syntax included
tests/                      64 tests; docs/  baseline findings, dependency report
```
Outputs (data, runs, results, smoke report) go only under `AESDIST_OUT_DIR` / `--out-dir`.
The original NDGohr files are read from `NDGOHR_BASELINE_DIR` and never modified.

## Setup (user-local only)
1. Put the original repo files in a read-only directory; `export NDGOHR_BASELINE_DIR=/path/to/it`.
2. In your existing env: `python scripts/env_report.py > env_report.txt` and send it for review.
3. `pip install --dry-run -r requirements-dev.txt` (changes nothing). Install only after the conflict review in
   `docs/DEPENDENCY_REPORT.md`, preferably into a cloned/new user env.
4. `python -m pytest tests -q` (about 1 minute on 1 core in the dev sandbox).

## CPU smoke test
Interactive on an *approved allocation* (not the login node):
`python scripts/run_experiment.py smoke --out-dir $AESDIST_OUT_DIR`  → writes `$AESDIST_OUT_DIR/smoke/smoke_report.json`.
Batch: fill every `CHANGE_ME` in `slurm/train_cpu.slurm` using your cluster's documented procedure, set
`PROJECT_DIR`, `NDGOHR_BASELINE_DIR`, `AESDIST_OUT_DIR`, `ANACONDA_MODULE`, then `sbatch slurm/train_cpu.slurm smoke`.
Jobs are never submitted from code.

## Staged use (all opt-in)
`gen-data --experiment r3_2B --mode smoke` → `train --experiment r3_2B --arch dbitnet --mode smoke`.
`--mode full` is refused without `--full-experiment`. Resume is the default; `--fresh` refuses to touch an existing
partial run unless `--overwrite` (old run is moved aside, not deleted). Completed runs are never retrained.

## Interactive Jupyter (only if your cluster permits it; notebook not yet delivered)
Inside an allocation: `jupyter lab --no-browser --ip=127.0.0.1 --port=<PORT>`. From your machine:
`ssh -L <PORT>:<compute-node>:<PORT> <you>@<login-host>`, then open `http://127.0.0.1:<PORT>`. Never bind to
`0.0.0.0`. If interactive notebooks are not allowed, the same code runs through `run_experiment.py` in batch.

## GPU
Not enabled. Use `slurm/train_gpu_template.slurm` only after obtaining the administrator-approved request syntax.
`--device gpu` refuses unless `SLURM_JOB_ID` is set and TensorFlow itself sees a GPU (no silent CPU fallback; tested).

## Known limitations
- SIGTERM/SIGUSR1 handling is best-effort (partial epoch discarded); SIGKILL/node loss cannot be caught but the last
  atomic checkpoint stays valid.
- Resume equality (1 epoch + resume == uninterrupted, weights within 1e-5) was verified on 1-core CPU in the sandbox only;
  multi-threaded/GPU runs may differ in low-order bits.
- Pretrained `.h5` handling is filename parsing + structural checks only; no pretrained model has been loaded.
- The absolute-subkey attack is not implemented yet (only its validated candidate mapping); the difference-based WKRP
  engine exists and is tested on mechanics with a stand-in scorer and a tiny untrained-ish model.
