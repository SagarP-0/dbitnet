"""Load the ORIGINAL NDGohr AES code read-only, without modifying or copying it.

The original directory is located via the `path` argument or the
NDGOHR_BASELINE_DIR environment variable. Bytecode writing is disabled while
importing so that nothing is written into the original directory.
"""
import importlib.util
import os
import sys
from pathlib import Path

from .io_utils import sha256_file

EXPECTED_FILES = [
    "AES_128_batch.py",
    "main_net.py",
    "16bit_AES_key_recovery_attack.py",
    "test_something.ipynb",
]
_CACHE = {}


class BaselineNotFound(RuntimeError):
    pass


def resolve_baseline_dir(path=None):
    p = path or os.environ.get("NDGOHR_BASELINE_DIR")
    if not p:
        raise BaselineNotFound(
            "Baseline directory not set. Export NDGOHR_BASELINE_DIR to a directory "
            "containing the original AES_128_batch.py (read-only is fine)."
        )
    p = Path(p).expanduser().resolve()
    if not (p / "AES_128_batch.py").is_file():
        raise BaselineNotFound(f"AES_128_batch.py not found in {p}")
    return p


def baseline_hashes(path=None):
    p = resolve_baseline_dir(path)
    return {n: sha256_file(p / n) for n in EXPECTED_FILES if (p / n).is_file()}


def load_aes(path=None):
    """Return the original AES_128_batch module object (cached)."""
    p = resolve_baseline_dir(path)
    key = str(p)
    if key in _CACHE:
        return _CACHE[key]
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("ndgohr_AES_128_batch", p / "AES_128_batch.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # original has an `if __name__ == "__main__"` guard
    finally:
        sys.dont_write_bytecode = old
    _CACHE[key] = mod
    return mod
