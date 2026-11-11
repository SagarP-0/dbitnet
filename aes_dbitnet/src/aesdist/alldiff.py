"""DBitNet over all 256 plaintext differences (byte 0), 3-round AES, final MixColumns on/off.

Plaintext pair: P1 = P0 ^ [d,0,...,0]. "enabled"/"disabled" = final_mc of aes_encrypt:
  enabled : ARK0 ; 2 x [SB SR MC ARK] ; SB SR MC ARK3
  disabled: ARK0 ; 2 x [SB SR MC ARK] ; SB SR ARK3          (standard final round)
Rounds 1 and 2 contain MixColumns in both variants.

Layout: <root>/{mixcolumns_enabled,mixcolumns_disabled}/diff_XX/ (run files + result.json),
        <root>/data/<variant>/ (DatasetStore dirs), <root>/summaries/, <root>/experiment_metadata.json
"""
import json
import os
import shutil
import socket
import time
import traceback
from pathlib import Path

import numpy as np

from . import experiment as X
from .io_utils import atomic_write_json, config_hash, dumps

NAME = "dbitnet_all_differences"
VARIANTS = ("enabled", "disabled")
RESULT_KEYS = ("schema", "status", "difference_dec", "mixcolumns", "seed", "signature", "test", "content_hash")
TRANSFORMS = {
    "enabled": "ARK0; 2x[SB,SR,MC,ARK]; SB,SR,MC,ARK3",
    "disabled": "ARK0; 2x[SB,SR,MC,ARK]; SB,SR,ARK3",
}


class RunMismatch(RuntimeError):
    pass


class Locked(RuntimeError):
    pass


def root_dir(out_dir, mode):
    return Path(out_dir) / ("all_input_differences_3round" + ("" if mode == "full" else f"_{mode}"))


def diff_dir(root, variant, d):
    return Path(root) / f"mixcolumns_{variant}" / f"diff_{d:02X}"


def final_mc(cfg, variant):
    return bool(cfg["all_differences"]["mixcolumns"][variant])


def make_store(cfg, variant, d, mode, seed, root):
    from .datagen import DatasetStore
    sec, m, ds = cfg["all_differences"], cfg["modes"][mode], cfg["dataset"]
    return DatasetStore(Path(root) / "data" / f"mixcolumns_{variant}", nr=sec["nr"], final_mc=final_mc(cfg, variant),
                        n=m["n"], chunk_size=m["chunk_size"], seed=seed, diff=d,
                        fractions=tuple(ds["fractions"]), paired=True)


def make_tcfg(cfg, variant, d, mode, seed, protocol=None):
    sec = cfg["all_differences"]
    c = {**cfg, "experiments": {"alldiff": {"nr": sec["nr"], "final_mc": final_mc(cfg, variant), "bytes": sec["bytes"]}}}
    t = X.make_tcfg(c, "alldiff", sec["architecture"], protocol or sec["protocol"], mode, seed)
    t.update(diff=d, mixcolumns=variant)
    return t


def signature(t, store):
    return config_hash({"train": t, "data": store.cfg})


class DiffLock:
    def __init__(self, ddir):
        self.path = Path(ddir) / ".lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise Locked(f"{self.path} exists: {self.path.read_text().strip()} (another job, or a killed one: --force-unlock)")
        with os.fdopen(fd, "w") as f:
            f.write(f"{socket.gethostname()} pid={os.getpid()} job={os.environ.get('SLURM_JOB_ID', '-')} t={int(time.time())}")
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def load_result(ddir, sig=None, log=print):
    """Valid completed result.json (parsable, schema keys, content hash, optional signature) or None."""
    p = Path(ddir) / "result.json"
    if not p.is_file():
        return None
    try:
        r = json.loads(p.read_text())
        body = {k: v for k, v in r.items() if k != "content_hash"}
        ok = all(k in r for k in RESULT_KEYS) and r["status"] == "completed" and r["content_hash"] == config_hash(body)
        if ok and sig is not None and r["signature"] != sig:
            raise RunMismatch(f"{p}: result was produced with a different configuration (use --overwrite to redo)")
        if ok:
            return r
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    bad = p.with_name(f"result.json.invalid-{int(time.time())}")
    p.rename(bad)
    log(f"[alldiff] invalid result.json moved to {bad.name}; experiment treated as incomplete")
    return None


def best_threshold(y, p):
    """Accuracy-maximising threshold over a quantile grid (grid as in run_diff.py), chosen on validation data."""
    cand = np.quantile(p, np.linspace(0.01, 0.99, 197))
    return float(cand[int(np.argmax([np.mean((p > t) == y) for t in cand]))])


def run_diff_config(cfg, aes, variant, d, mode, seed, root, protocol=None, overwrite=False, purge=False,
                    log=print, stop=None, max_epochs_this_call=None):
    """Train/resume + evaluate one (variant, difference). Returns 'skipped' | 'completed' | 'partial' | 'interrupted'."""
    import tensorflow as tf
    from . import evaluate as E, train as T
    from .metrics import binary_metrics
    from .models import param_counts

    ddir = diff_dir(root, variant, d)
    t = make_tcfg(cfg, variant, d, mode, seed, protocol)
    store = make_store(cfg, variant, d, mode, seed, root)
    sig = signature(t, store)
    if overwrite and ddir.exists():
        shutil.move(str(ddir), str(ddir.with_name(ddir.name + f".superseded-{int(time.time())}")))
    if load_result(ddir, sig, log):
        return "skipped"
    with DiffLock(ddir):
        meta_p = ddir / "run_meta.json"
        if meta_p.is_file() and json.loads(meta_p.read_text()).get("signature") != sig:
            raise RunMismatch(f"{ddir}: existing run uses a different configuration (use --overwrite to restart)")
        atomic_write_json(meta_p, {"signature": sig, "train": t, "data": store.cfg})
        store.generate(aes, log=lambda *_: None)
        data = {s: store.load_split(s) for s in ("train", "val", "test")}
        res = T.train_model(t, data, ddir, resume=True, log=log, stop=stop, max_epochs_this_call=max_epochs_this_call)
        if res["status"] not in ("completed", "already_completed"):
            return res["status"]

        model, meta = T.load_best(t, ddir)
        idx, mi = t["byte_idx"], t["model"]["input_mode"]
        mt, pt = E.evaluate_split(model, *data["test"], idx, mi, t["eval_batch"])
        thr = best_threshold(data["val"][1], E.predict_probs(model, *data["val"], idx, mi, t["eval_batch"]))
        mc_ = binary_metrics(data["test"][1], pt, thr)
        bench = E.benchmark_inference(model, data["test"][0], idx, mi, batch=min(5000, len(data["test"][1])))
        hist = json.loads((ddir / "history.json").read_text())
        keep = ("accuracy", "tpr", "tnr", "precision", "bce", "mse", "tp", "tn", "fp", "fn", "n", "threshold")
        rec = {
            "schema": 1, "status": "completed", "difference_hex": f"{d:02X}", "difference_dec": d,
            "plaintext_difference": [d] + [0] * 15, "rounds": t["nr"], "mixcolumns": variant,
            "final_mc": t["final_mc"], "transforms": TRANSFORMS[variant], "seed": seed, "mode": mode,
            "protocol": t["protocol"], "signature": sig, "params": param_counts(model),
            "epochs_run": len(hist), "best_epoch": meta.get("best_epoch"),
            "train_loss_final": hist[-1]["train_loss"], "val_loss_final": hist[-1]["val_loss"],
            "val_loss_best": min(h["val_loss"] for h in hist), "val_acc_best": max(h["val_acc"] for h in hist),
            "test": {k: mt[k] for k in keep},
            "test_calibrated": {**{k: mc_[k] for k in keep}, "chosen_on": "validation"},
            "acc_z_vs_chance": (mt["accuracy"] - 0.5) / np.sqrt(0.25 / mt["n"]),
            "train_time_s": sum(h["train_time_s"] for h in hist), "eval_wall_s": mt["eval_wall_s_incl_bit_encoding"],
            "inference": bench, "dataset_dir": str(store.dir), "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "tensorflow": tf.__version__,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        rec["content_hash"] = config_hash(rec)
        atomic_write_json(ddir / "result.json", rec)
        (ddir / "failure.json").unlink(missing_ok=True)
        if purge:
            store.purge()
        return "completed"


def write_metadata(cfg, mode, seed, protocol, root):
    sec = cfg["all_differences"]
    t = make_tcfg(cfg, "enabled", 0, mode, seed, protocol)
    meta = {"name": NAME, "mode": mode, "seed": seed, "rounds": sec["nr"], "difference_byte": sec["byte_pos"],
            "difference_definition": "P1 = P0 xor [d,0,...,0], d in 0..255; same key for both plaintexts",
            "negative_pairs": "P1 replaced by an independent random plaintext under the same key (original procedure)",
            "variants": {v: {"final_mc": final_mc(cfg, v), "transforms": TRANSFORMS[v]} for v in VARIANTS},
            "data_streams": "plaintexts, keys, labels shared by all differences and both variants for a given seed",
            "model": t["model"], "protocol": t["protocol"], "loss": t["loss"], "lr": t["lr"], "schedule": t["schedule"],
            "epochs": t["epochs"], "batch_size": t["batch_size"], "dataset": cfg["modes"][mode], "fractions": cfg["dataset"]["fractions"]}
    p = Path(root) / "experiment_metadata.json"
    if not p.is_file() or json.loads(p.read_text()) != json.loads(dumps(meta)):
        atomic_write_json(p, meta)


def preflight(aes):
    """Cheap correctness gate before any heavy work."""
    from . import validation as V
    res = V.tables_check(aes) + V.fips197_checks(aes) + V.reduced_round_checks(aes, rounds=(3,), n=16)
    bad = [n for n, ok, _ in res if not ok]
    if bad:
        raise RuntimeError(f"AES preflight failed: {bad}")


def run_all(cfg, aes, out_dir, start, end, variants, seed, mode, protocol=None, overwrite=False, purge=False,
            force_unlock=False, log=print, stop=None, max_epochs_this_call=None):
    """Loop d (outer) x variant (inner). One failure never stops the loop; a stop signal does.
    Returns (exit_code, status_by_key): 0 all ok, 1 some failed, 3 interrupted."""
    import tensorflow as tf
    root = root_dir(out_dir, mode)
    root.mkdir(parents=True, exist_ok=True)
    write_metadata(cfg, mode, seed, protocol, root)
    if force_unlock:
        for v in variants:
            for d in range(start, end + 1):
                (diff_dir(root, v, d) / ".lock").unlink(missing_ok=True)
    status = {}
    for d in range(start, end + 1):
        for v in variants:
            key = f"{v}/{d:02X}"
            t0 = time.time()
            try:
                status[key] = run_diff_config(cfg, aes, v, d, mode, seed, root, protocol, overwrite, purge, log, stop,
                                              max_epochs_this_call)
            except Exception as e:
                status[key] = "failed"
                log(f"[alldiff] FAILED {key}: {e!r}")
                atomic_write_json(diff_dir(root, v, d) / "failure.json", {"error": repr(e), "trace": traceback.format_exc(),
                                                                          "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
            tf.keras.backend.clear_session()
            log(f"[alldiff] {key}: {status[key]} ({time.time() - t0:.0f}s)")
            if status[key] == "interrupted" or (stop is not None and stop.requested):
                log("[alldiff] stop requested; rerun the same command to resume")
                return 3, status
    return (1 if "failed" in status.values() else 0), status
