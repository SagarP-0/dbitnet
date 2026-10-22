"""Atomic, self-validating checkpoints (single .npz: weights, optimizer vars, JSON metadata).

Stored values are plain numpy arrays in layer order, so a checkpoint is only valid for an
identical architecture; this is enforced through `arch_hash` and shape checks on load.
"""
import json
import platform
import sys

import numpy as np

from .io_utils import atomic_save_npz, config_hash, sha256_arrays


class CheckpointError(RuntimeError):
    pass


def _opt_vars(opt):
    v = opt.variables
    return list(v() if callable(v) else v)


def framework_versions():
    import tensorflow as tf
    return {"python": sys.version.split()[0], "tensorflow": tf.__version__, "keras": getattr(tf.keras, "__version__", "?"),
            "numpy": np.__version__, "platform": platform.platform()}


def save_checkpoint(path, model, opt, meta):
    w = [np.asarray(a) for a in model.get_weights()]
    o = [np.asarray(v.numpy()) for v in _opt_vars(opt)] if opt is not None else []
    meta = dict(meta)
    meta["n_weights"], meta["n_opt"] = len(w), len(o)
    meta["sha256"] = sha256_arrays(w + o)
    meta["versions"] = framework_versions()
    arrays = {f"w_{i}": a for i, a in enumerate(w)}
    arrays.update({f"o_{i}": a for i, a in enumerate(o)})
    arrays["meta_json"] = np.frombuffer(json.dumps(meta, default=str).encode(), dtype=np.uint8)
    atomic_save_npz(path, **arrays)


def read_checkpoint(path, expect_arch_hash=None):
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta_json"]).decode())
            w = [z[f"w_{i}"] for i in range(meta["n_weights"])]
            o = [z[f"o_{i}"] for i in range(meta["n_opt"])]
    except Exception as e:  # truncated / corrupt / missing key
        raise CheckpointError(f"cannot read checkpoint {path}: {e}") from e
    if sha256_arrays(w + o) != meta["sha256"]:
        raise CheckpointError(f"checksum mismatch in {path}")
    if expect_arch_hash is not None and meta.get("arch_hash") != expect_arch_hash:
        raise CheckpointError("checkpoint architecture/config does not match the current run")
    return w, o, meta


def restore(model, opt, w, o):
    cur = model.get_weights()
    if len(cur) != len(w) or any(a.shape != b.shape for a, b in zip(cur, w)):
        raise CheckpointError("weight shapes differ from current model")
    model.set_weights(w)
    if opt is not None and o:
        vs = _opt_vars(opt)
        if len(vs) != len(o):
            raise CheckpointError(f"optimizer variable count differs ({len(vs)} vs {len(o)}); was opt.build() called?")
        for v, a in zip(vs, o):
            v.assign(a)


def arch_hash(model_cfg, loss, optimizer_cfg):
    return config_hash({"model": model_cfg, "loss": loss, "opt": optimizer_cfg})
