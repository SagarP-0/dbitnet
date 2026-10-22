"""Shared experiment logic used by run_experiment.py, smoke.py and (later) the notebook."""
import copy
import json
import os
from pathlib import Path

import numpy as np

from .conventions import resolve_truncation
from .datagen import DatasetStore


def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def out_dir_from(arg=None):
    return Path(arg or os.environ.get("AESDIST_OUT_DIR") or "./outputs").expanduser().resolve()


def arch_kwargs(cfg, arch_name):
    a = copy.deepcopy(cfg["architectures"][arch_name])
    if "base" in a:
        base = copy.deepcopy(cfg["architectures"][a.pop("base")])
        base.update(a)
        a = base
    return a


def make_store(cfg, exp, mode, out_dir):
    m = cfg["modes"][mode]
    e = cfg["experiments"][exp]
    d = cfg["dataset"]
    return DatasetStore(Path(out_dir) / "data", nr=e["nr"], final_mc=e["final_mc"], n=m["n"], chunk_size=m["chunk_size"],
                        seed=d["data_seed"], diff=d["diff"], fractions=tuple(d["fractions"]))


def make_tcfg(cfg, exp, arch_name, protocol, mode, seed):
    e = cfg["experiments"][exp]
    m = cfg["modes"][mode]
    p = cfg["protocols"][protocol]
    idx = resolve_truncation(e["bytes"])
    ak = arch_kwargs(cfg, arch_name)
    arch = "ndgohr" if arch_name.startswith("ndgohr") else "dbitnet"
    model = {"arch": arch, "bits": 8 * len(idx), **ak}
    return {"model": model, "byte_idx": idx, "loss": p["loss"], "lr": p["lr"], "min_lr": p["min_lr"],
            "schedule": p["schedule"], "best_metric": p["best_metric"], "epochs": m["epochs"],
            "batch_size": m["batch_size"], "seed": int(seed), "early_stopping": None,
            "snapshot_every": 0, "eval_batch": 10000, "experiment": exp, "arch_name": arch_name,
            "protocol": protocol, "mode": mode, "nr": e["nr"], "final_mc": e["final_mc"]}


def run_id(t):
    return f"{t['experiment']}__{t['arch_name']}__{t['protocol']}__seed{t['seed']}__{t['mode']}"


def run_one(cfg, exp, arch_name, protocol, mode, seed, out_dir, resume=True, overwrite=False, log=print, stop=None,
            max_epochs_this_call=None):
    """Train (or resume) one run, evaluate best-val weights on the held-out test split, benchmark inference."""
    from . import evaluate as E, train as T
    store = make_store(cfg, exp, mode, out_dir)
    if not store.is_complete():
        raise RuntimeError(f"dataset incomplete: run gen-data first (missing/corrupt chunks: {store.verify(deep=False)})")
    data = {s: store.load_split(s) for s in ("train", "val", "test")}
    t = make_tcfg(cfg, exp, arch_name, protocol, mode, seed)
    rdir = Path(out_dir) / "runs" / run_id(t)
    res = T.train_model(t, data, rdir, resume=resume, overwrite=overwrite, log=log, stop=stop,
                        max_epochs_this_call=max_epochs_this_call)
    if res["status"] not in ("completed", "already_completed"):
        return res
    model, meta = T.load_best(t, rdir)
    mode_in = t["model"]["input_mode"]
    m, p = E.evaluate_split(model, *data["test"], t["byte_idx"], mode_in, batch=t["eval_batch"])
    b = E.benchmark_inference(model, data["test"][0], t["byte_idx"], mode_in, batch=min(5000, len(data["test"][1])))
    hist = json.loads((rdir / "history.json").read_text())
    row = {"run_id": run_id(t), "nr": t["nr"], "bytes": t["byte_idx"], "arch": arch_name, "protocol": protocol,
           "seed": seed, "mode": mode, "best_epoch": meta.get("best_epoch"), "epochs_run": len(hist),
           "train_time_s_total": sum(h["train_time_s"] for h in hist),
           "train_time_s_per_epoch_mean": float(np.mean([h["train_time_s"] for h in hist])),
           "test": m, "inference": b, "dataset_dir": str(store.dir)}
    (Path(out_dir) / "results").mkdir(parents=True, exist_ok=True)
    from .io_utils import atomic_write_json
    atomic_write_json(Path(out_dir) / "results" / f"{run_id(t)}.json", row)
    return row
