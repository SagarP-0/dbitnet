# Dependency and Python compatibility report

## What was and was not checked
- **Not checked:** your `aes-cryptanalysis` environment. I have no access to your cluster, so I cannot read its
  installed packages. Run `python scripts/env_report.py > env_report.txt` inside that env (no heavy imports, safe on a
  login node; add `--check-tf` only inside an allocation) and send me the output. Nothing below should be installed
  until that output has been compared with the pins.
- **Checked (development sandbox, not your cluster):** CPython 3.10.20, `tensorflow-cpu==2.15.1` → resolved
  keras 2.15.0, numpy 1.26.4, h5py 3.16.0, protobuf 4.25.9, ml-dtypes 0.3.2. All 64 tests and the 79-check smoke test pass there.
- Your reported Python is 3.10.22; I cannot verify that string. `env_report.py` prints the real one.

## Framework choice
The baseline is Keras/TensorFlow (`tensorflow.keras`, `.h5` models, `MirroredStrategy`). Both architectures are built in
the same framework so the comparison is like-for-like. PyTorch is not used and not needed. scikit-learn is not needed
(metrics are in numpy). The baseline's original TF version is unknown (kernel metadata says Python 3.7.10).

| Package | Pin | Why |
|---|---|---|
| tensorflow-cpu | 2.15.1 | Python 3.10 supported; last release where `tf.keras` is Keras 2 (legacy `.h5` loads); TF ≥ 2.16 switches to Keras 3 |
| numpy | >=1.23.5,<2 | TF 2.15 constraint |
| h5py | >=3.10 | `.h5` model files |
| pyyaml | >=6 | config |
| pytest, pycryptodome | dev only | tests; third-party AES cross-check |

## Safe procedure for your account (no changes to shared installs)
1. `conda activate aes-cryptanalysis`, then `python scripts/env_report.py` and `pip check` (read-only).
2. `pip install --dry-run -r requirements-cpu.txt` — shows what *would* change, installs nothing (pip ≥ 22.2).
3. If the dry run would upgrade/downgrade existing packages you care about, don't touch that env: clone it
   (`conda create --name aes-dbitnet --clone aes-cryptanalysis`, user-local) and install there, or create a fresh
   user env. Both stay in your home directory.
4. Never mix `tensorflow` and `tensorflow-cpu` in one environment; pick one.
Likely conflicts to look for: an existing `tensorflow`/`keras` ≥ 2.16, `numpy` 2.x, a `protobuf` outside 3.20.3–4.x, or a
GPU `tensorflow` wheel when you wanted CPU. I cannot tell which apply until I see the report.

## GPU / CUDA (conditional)
Not selected yet, deliberately. Options once GPU allocation syntax is confirmed: `tensorflow[and-cuda]==2.15.1`
(user-space CUDA 12.x libraries from pip; several GB, check quota) or a cluster CUDA module matching TF 2.15's CUDA/cuDNN
build. TF 2.15 on H100 (sm_90) is **unverified** here. `requirements-gpu.txt` is a stub with these notes. The code
refuses GPU mode unless a SLURM allocation exists *and* TensorFlow itself reports a GPU.

## Baseline incompatibilities handled
Hard-coded 5-GPU `MirroredStrategy`; `CUDA_VISIBLE_DEVICES` override and import-time execution in the attack script;
unseedable `os.urandom`; relative paths; int64 AES outputs. The original files are never edited or imported except
`AES_128_batch.py` (side-effect free, loaded read-only without writing bytecode).
