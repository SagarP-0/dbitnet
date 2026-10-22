"""Atomic writes and hashing helpers (no dependencies beyond numpy)."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def dumps(obj, indent=2):
    return json.dumps(obj, indent=indent, sort_keys=True, default=_json_default)


def atomic_write(path, writer, mode="wb"):
    """Write via temp file in the same directory, fsync, then os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, mode) as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path, obj):
    atomic_write(path, lambda f: f.write(dumps(obj)), mode="w")


def atomic_save_npy(path, arr):
    atomic_write(path, lambda f: np.save(f, arr, allow_pickle=False))


def atomic_save_npz(path, **arrays):
    atomic_write(path, lambda f: np.savez(f, **arrays))


def cleanup_tmp(directory):
    """Remove leftover *.tmp files from interrupted atomic writes."""
    n = 0
    for p in Path(directory).glob("*.tmp"):
        p.unlink()
        n += 1
    return n


def sha256_arrays(arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def sha256_file(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def config_hash(obj, n=16):
    return hashlib.sha256(dumps(obj, indent=None).encode()).hexdigest()[:n]
