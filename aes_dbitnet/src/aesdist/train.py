"""Training loop with atomic latest/best checkpoints and exact resume (custom loop, TF 2.15 Keras).

State persisted every epoch in latest.npz: weights, Adam variables, epoch, global step,
scheduler state, shuffle-RNG state, early-stopping state, history, config, versions.
best.npz holds the best-validation checkpoint (separate file, never overwritten by latest).

Honest limits: SIGTERM/SIGUSR1 are caught and the loop stops at the next BATCH boundary,
discarding the partial epoch (resume re-runs that epoch). SIGKILL / node failure cannot be
caught; the previous epoch's checkpoint remains valid because writes are atomic.
"""
import math
import shutil
import signal
import time
from pathlib import Path

import numpy as np

from . import checkpoint as ck
from .bits import iter_batches
from .io_utils import atomic_write_json
from .metrics import binary_metrics


class RunExistsError(RuntimeError):
    pass


class StopFlag:
    """Set by signal handlers (wall-time warning); polled once per batch."""
    def __init__(self):
        self.requested = False
        self.signum = None

    def install(self):
        def h(signum, _frame):
            self.requested, self.signum = True, signum
        for s in (signal.SIGTERM, signal.SIGUSR1):
            try:
                signal.signal(s, h)
            except (ValueError, OSError):
                pass
        return self


def lr_for_epoch(sched, epoch, total_epochs, base_lr, min_lr, state):
    if sched == "constant":
        return base_lr
    if sched == "cosine":      # closed form of the epoch index => stateless, trivially resumable
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * epoch / max(1, total_epochs)))
    if sched == "cyclic":      # original NDGohr: cyclic_lr(10, 0.002, 0.0001)
        n, hi, lo = 10, 0.002, 0.0001
        return lo + ((n - 1) - epoch % n) / (n - 1) * (hi - lo)
    if sched == "plateau":
        return state["plateau_lr"]
    raise ValueError(sched)


def _predict(model, ct, y, idx, mode, bs):
    ps = [np.asarray(model(x, training=False)).reshape(-1) for x, _ in iter_batches(ct, y, idx, mode, bs)]
    return np.concatenate(ps)


def train_model(tcfg, data, run_dir, resume=True, overwrite=False, log=print, stop=None, max_epochs_this_call=None):
    """tcfg keys: model(dict), byte_idx, input_mode, loss('bce'|'mse'), lr, min_lr, epochs, batch_size,
    schedule, seed, best_metric('val_loss'|'val_acc'), early_stopping({patience,min_delta}|None),
    snapshot_every(int|0), eval_batch.  data: dict(train=(ct,y), val=(ct,y)).
    Returns dict(status, history, best, train_time_s)."""
    import tensorflow as tf
    from .models import build_model, param_counts

    run_dir = Path(run_dir)
    latest, best_p, status_p = run_dir / "latest.npz", run_dir / "best.npz", run_dir / "status.json"
    if status_p.is_file():
        import json
        st = json.loads(status_p.read_text())
        if st.get("state") == "completed" and not overwrite:
            log(f"[train] {run_dir.name}: already completed; not retraining")
            return {"status": "already_completed", **st}
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        if not overwrite:
            raise RunExistsError(f"{run_dir} exists; use resume=True or overwrite=True")
        aside = run_dir.with_name(run_dir.name + f".superseded-{int(time.time())}")
        shutil.move(str(run_dir), str(aside))
        log(f"[train] moved previous run to {aside}")
    run_dir.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(int(tcfg["seed"]))
    model = build_model(tcfg["model"])
    loss_name = tcfg["loss"]
    lf = tf.keras.losses.BinaryCrossentropy() if loss_name == "bce" else tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam(learning_rate=tcfg["lr"])
    opt.build(model.trainable_variables)
    ahash = ck.arch_hash(tcfg["model"], loss_name, {"opt": "adam", "lr": tcfg["lr"], "sched": tcfg["schedule"]})
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(tcfg["seed"]), 0x7A1])))
    state = {"epoch": 0, "global_step": 0, "history": [], "train_time_s": 0.0,
             "sched": {"plateau_lr": tcfg["lr"], "plateau_wait": 0, "plateau_best": float("inf")},
             "es": {"wait": 0, "best": float("inf")}, "best_value": None}
    if resume and latest.is_file():
        w, o, meta = ck.read_checkpoint(latest, expect_arch_hash=ahash)
        ck.restore(model, opt, w, o)
        state = meta["state"]
        rng.bit_generator.state = meta["rng_state"]
        log(f"[train] resumed from epoch {state['epoch']} (global step {state['global_step']})")
    atomic_write_json(run_dir / "config.json", {**tcfg, "arch_hash": ahash, "params": param_counts(model)})
    atomic_write_json(status_p, {"state": "running", "epoch": state["epoch"]})

    @tf.function(reduce_retracing=True)
    def step(x, y):
        with tf.GradientTape() as tape:
            p = model(x, training=True)
            l = lf(y, p)
            tot = l + tf.add_n(model.losses) if model.losses else l
        g = tape.gradient(tot, model.trainable_variables)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return l

    idx, mode, bs = tcfg["byte_idx"], tcfg["model"]["input_mode"], int(tcfg["batch_size"])
    (tct, ty), (vct, vy) = data["train"], data["val"]
    higher_better = tcfg.get("best_metric", "val_loss") == "val_acc"
    ran, interrupted = 0, False
    while state["epoch"] < tcfg["epochs"]:
        if max_epochs_this_call is not None and ran >= max_epochs_this_call:
            break
        ep = state["epoch"]
        lr = lr_for_epoch(tcfg["schedule"], ep, tcfg["epochs"], tcfg["lr"], tcfg.get("min_lr", 1e-5), state["sched"])
        opt.learning_rate.assign(lr)
        t0 = time.perf_counter(); tot, cnt, steps = 0.0, 0, 0
        for x, y in iter_batches(tct, ty, idx, mode, bs, rng=rng):
            if stop is not None and stop.requested:
                interrupted = True
                break
            l = float(step(x, y)); tot += l * len(x); cnt += len(x); steps += 1
        if interrupted:
            log("[train] stop requested: discarding partial epoch; last completed checkpoint stays valid")
            break
        train_time = time.perf_counter() - t0
        v0 = time.perf_counter()
        vp = _predict(model, vct, vy, idx, mode, int(tcfg.get("eval_batch", 10000)))
        vm = binary_metrics(vy, vp)
        val_time = time.perf_counter() - v0
        vloss = vm["bce"] if loss_name == "bce" else vm["mse"]
        rec = {"epoch": ep + 1, "lr": lr, "train_loss": tot / cnt, "val_loss": vloss, "val_acc": vm["accuracy"],
               "val_tpr": vm["tpr"], "val_tnr": vm["tnr"], "train_time_s": train_time, "val_time_s": val_time}
        state["history"].append(rec)
        state["epoch"] += 1; state["global_step"] += steps; state["train_time_s"] += train_time
        ran += 1
        # scheduler / early-stopping / best bookkeeping
        sch = state["sched"]
        if vloss < sch["plateau_best"] - 1e-6:
            sch["plateau_best"], sch["plateau_wait"] = vloss, 0
        else:
            sch["plateau_wait"] += 1
            if tcfg["schedule"] == "plateau" and sch["plateau_wait"] >= tcfg.get("plateau_patience", 2):
                sch["plateau_lr"] = max(tcfg.get("min_lr", 1e-5), sch["plateau_lr"] * 0.5); sch["plateau_wait"] = 0
        monitored = vm["accuracy"] if higher_better else vloss
        is_best = state["best_value"] is None or (monitored > state["best_value"] if higher_better else monitored < state["best_value"])
        if is_best:
            state["best_value"] = monitored
        es = tcfg.get("early_stopping")
        stop_early = False
        if es:
            if vloss < state["es"]["best"] - es.get("min_delta", 0.0):
                state["es"] = {"wait": 0, "best": vloss}
            else:
                state["es"]["wait"] += 1
                stop_early = state["es"]["wait"] >= es["patience"]
        meta = {"state": state, "rng_state": rng.bit_generator.state, "arch_hash": ahash, "config": tcfg, "epoch": state["epoch"]}
        ck.save_checkpoint(latest, model, opt, meta)
        if is_best:
            ck.save_checkpoint(best_p, model, None, {**meta, "best_epoch": state["epoch"]})
        if tcfg.get("snapshot_every") and state["epoch"] % tcfg["snapshot_every"] == 0:
            ck.save_checkpoint(run_dir / f"epoch_{state['epoch']:03d}.npz", model, opt, meta)
        atomic_write_json(run_dir / "history.json", state["history"])
        log(f"[train] epoch {state['epoch']}/{tcfg['epochs']} lr={lr:.2e} loss={rec['train_loss']:.4f} "
            f"val_loss={vloss:.4f} val_acc={vm['accuracy']:.4f} ({train_time:.1f}s)")
        if stop_early:
            log("[train] early stopping")
            break
    done = state["epoch"] >= tcfg["epochs"] or (tcfg.get("early_stopping") and state["es"]["wait"] >= tcfg["early_stopping"]["patience"])
    final = {"state": "completed" if (done and not interrupted) else ("interrupted" if interrupted else "partial"),
             "epoch": state["epoch"], "train_time_s": state["train_time_s"], "best_value": state["best_value"]}
    atomic_write_json(status_p, final)
    return {"status": final["state"], "history": state["history"], "train_time_s": state["train_time_s"],
            "model": model, "run_dir": str(run_dir), **{k: final[k] for k in ("epoch", "best_value")}}


def load_best(tcfg, run_dir):
    """Return a model with the best-validation weights loaded."""
    from .models import build_model
    model = build_model(tcfg["model"])
    ahash = ck.arch_hash(tcfg["model"], tcfg["loss"], {"opt": "adam", "lr": tcfg["lr"], "sched": tcfg["schedule"]})
    w, _, meta = ck.read_checkpoint(Path(run_dir) / "best.npz", expect_arch_hash=ahash)
    ck.restore(model, None, w, [])
    return model, meta
