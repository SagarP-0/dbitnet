"""Consolidated summaries and plots for the all-differences experiment (plots need matplotlib)."""
import csv
import json
from pathlib import Path

import numpy as np

from .alldiff import VARIANTS, diff_dir, load_result
from .io_utils import atomic_write, atomic_write_json

COLS = ("accuracy", "tpr", "tnr", "precision")


def collect(root, variant):
    """256 entries: result dict for valid completed runs, else {'status': ...}."""
    out = []
    for d in range(256):
        dd = diff_dir(root, variant, d)
        r = load_result(dd, log=lambda *_: None)
        if r:
            out.append(r)
        elif (dd / "failure.json").is_file():
            out.append({"difference_dec": d, "status": "failed"})
        elif (dd / "latest.npz").is_file():
            out.append({"difference_dec": d, "status": "incomplete"})
        else:
            out.append({"difference_dec": d, "status": "missing"})
    return out


def _ranked(rows, k=5):
    done = sorted((r for r in rows if r["status"] == "completed"), key=lambda r: r["test"]["accuracy"])
    f = lambda r: {"diff": r["difference_hex"], "acc": r["test"]["accuracy"], "tpr": r["test"]["tpr"], "tnr": r["test"]["tnr"]}
    return {"lowest": [f(r) for r in done[:k]], "highest": [f(r) for r in done[::-1][:k]]}


def _flat(r):
    if r["status"] != "completed":
        return {"status": r["status"]}
    t, c = r["test"], r["test_calibrated"]
    return {"status": "completed", "acc": t["accuracy"], "tpr": t["tpr"], "tnr": t["tnr"], "precision": t["precision"],
            "acc_calibrated": c["accuracy"], "thr_calibrated": c["threshold"], "val_acc_best": r["val_acc_best"],
            "val_loss_best": r["val_loss_best"], "train_loss_final": r["train_loss_final"],
            "acc_z_vs_chance": r["acc_z_vs_chance"], "train_time_s": r["train_time_s"],
            "inference_ms_single": r["inference"]["latency_ms_median"],
            "inference_samples_per_s": r["inference"]["throughput_samples_per_s"]}


def summarize(root, plots=True, log=print):
    root = Path(root)
    sdir = root / "summaries"
    rows = {v: collect(root, v) for v in VARIANTS}
    for v in VARIANTS:
        counts = {s: sum(r["status"] == s for r in rows[v]) for s in ("completed", "incomplete", "failed", "missing")}
        sig = [r["difference_hex"] for r in rows[v] if r["status"] == "completed" and r["acc_z_vs_chance"] > 3]
        atomic_write_json(sdir / f"mixcolumns_{v}.json", {"variant": v, "counts": counts, "extremes_by_test_accuracy": _ranked(rows[v]),
                                                          "diffs_with_acc_z_over_3": sig, "results": rows[v]})
        log(f"[summary] {v}: {counts}")
    comp = []
    for d in range(256):
        e, n = (_flat(rows[v][d]) for v in VARIANTS)
        row = {"diff_hex": f"{d:02X}", "diff_dec": d, "enabled": e, "disabled": n}
        if e["status"] == n["status"] == "completed":
            row["delta_acc_enabled_minus_disabled"] = e["acc"] - n["acc"]
        comp.append(row)
    atomic_write_json(sdir / "comparison.json", comp)
    fields = ["acc", "tpr", "tnr", "precision", "acc_calibrated", "val_acc_best", "acc_z_vs_chance", "train_time_s"]
    head = ["diff_hex", "diff_dec"] + [f"{v}_{f}" for v in VARIANTS for f in ["status"] + fields] + ["delta_acc_enabled_minus_disabled"]
    def w(f):
        wr = csv.writer(f)
        wr.writerow(head)
        for r in comp:
            wr.writerow([r["diff_hex"], r["diff_dec"]] + [r[v].get(k, "") for v in VARIANTS for k in ["status"] + fields]
                        + [r.get("delta_acc_enabled_minus_disabled", "")])
    atomic_write(sdir / "comparison.csv", w, mode="w")
    if plots:
        try:
            make_plots(comp, sdir / "plots")
        except ImportError:
            log("[summary] matplotlib not installed: JSON/CSV written, plots skipped")
    return comp


def make_plots(comp, pdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pdir = Path(pdir)
    pdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
    xs = np.arange(256)
    ticks = list(range(0, 256, 16))
    val = lambda v, k: np.array([r[v].get(k, np.nan) if r[v]["status"] == "completed" else np.nan for r in comp], dtype=float)
    style = {"enabled": ("#c0392b", "o", "final MixColumns enabled"), "disabled": ("#1f6fb4", "s", "final MixColumns disabled")}

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(pdir / f"{name}.{ext}", dpi=200)
        plt.close(fig)

    def axes_fmt(ax, label, chance=True):
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:02X}" for t in ticks])
        ax.set_xlim(-2, 257)
        ax.set_xlabel("input difference (byte 0, hex)")
        ax.set_ylabel(label)
        if chance:
            ax.axhline(0.5, color="grey", lw=0.8, ls="--")

    for key, label in (("acc", "test accuracy"), ("tpr", "TPR"), ("tnr", "TNR")):
        fig, ax = plt.subplots(figsize=(9, 3.4))
        for v, (c, m, lab) in style.items():
            ax.plot(xs, val(v, key), m, ms=2.5, color=c, label=lab, alpha=0.85)
        axes_fmt(ax, label)
        ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
        save(fig, f"{key}_vs_difference")

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1, 1]})
    for ax, v in zip(axs[:2], VARIANTS):
        c, m, lab = style[v]
        ax.bar(xs, val(v, "acc") - 0.5, bottom=0.5, color=c, width=1.0)
        axes_fmt(ax, "test accuracy")
        ax.set_title(lab, fontsize=10)
    axs[2].bar(xs, np.nan_to_num(val("enabled", "acc") - val("disabled", "acc")), color="#555", width=1.0)
    axes_fmt(axs[2], "accuracy: enabled - disabled", chance=False)
    axs[2].axhline(0, color="grey", lw=0.8)
    axs[2].set_title("paired difference", fontsize=10)
    save(fig, "enabled_vs_disabled")
