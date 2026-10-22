#!/usr/bin/env python
"""Report installed packages WITHOUT importing heavy libraries (safe on a login node).

Run inside the user-level conda env:  python scripts/env_report.py > env_report.txt
Add --check-tf to also import TensorFlow and list visible devices (do this inside an allocation).
Nothing is installed or modified.
"""
import json
import platform
import sys
from importlib import metadata

WANT = ["tensorflow", "tensorflow-cpu", "tensorflow-gpu", "keras", "numpy", "h5py", "scipy", "scikit-learn", "pandas",
        "matplotlib", "pyyaml", "jupyter", "jupyterlab", "notebook", "ipykernel", "nbformat", "nbconvert", "pytest",
        "pycryptodome", "cryptography", "protobuf", "ml-dtypes", "torch", "nvidia-cudnn-cu12", "nvidia-cuda-runtime-cu12"]


def main():
    out = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "packages": {}}
    for name in WANT:
        try:
            out["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out["packages"][name] = None
    if "--check-tf" in sys.argv:
        import tensorflow as tf
        out["tf_import"] = tf.__version__
        out["tf_visible_gpus"] = [g.name for g in tf.config.list_physical_devices("GPU")]
        out["tf_built_with_cuda"] = tf.test.is_built_with_cuda()
    print(json.dumps(out, indent=2))
    print("\nAlso useful (run yourself, read-only):\n  pip list --format=freeze\n  pip check\n  pip install --dry-run -r requirements-cpu.txt")


if __name__ == "__main__":
    main()
