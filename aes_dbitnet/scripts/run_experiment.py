#!/usr/bin/env python
"""Non-interactive entry point. Subcommands: env-report | smoke | gen-data | train | all-diffs | summarize-diffs.

`--experiment dbitnet_all_differences` without a subcommand means all-diffs (256 differences x final MixColumns on/off).

Safe defaults: 'full' mode refuses to run without --full-experiment. This script never submits
SLURM jobs, never installs packages, never edits CUDA/SLURM settings.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", nargs="?", choices=["env-report", "smoke", "gen-data", "train", "all-diffs", "summarize-diffs"])
    ap.add_argument("--config", default=str(ROOT / "config" / "experiments.yaml"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--baseline-dir", default=None, help="original NDGohr repo dir (read-only); or NDGOHR_BASELINE_DIR")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--full-experiment", action="store_true", help="required to run --mode full")
    ap.add_argument("--experiment", help="key under experiments: in the config")
    ap.add_argument("--arch", default="dbitnet", help="ndgohr | dbitnet | dbitnet_concat1")
    ap.add_argument("--protocol", default=None, help="training protocol (default: controlled; all-diffs: from config)")
    ap.add_argument("--start", type=int, default=0, help="all-diffs: first difference (0-255)")
    ap.add_argument("--end", type=int, default=255, help="all-diffs: last difference, inclusive")
    ap.add_argument("--mixcolumns", choices=["enabled", "disabled", "both"], default="both", help="all-diffs: final-round MixColumns variant")
    ap.add_argument("--purge-data", action="store_true", help="all-diffs: delete dataset chunks after a run completes")
    ap.add_argument("--force-unlock", action="store_true", help="all-diffs: remove stale .lock files in the selected range")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--resume", dest="resume", action="store_true", default=True)
    g.add_argument("--fresh", dest="resume", action="store_false", help="start fresh; fails if a partial run exists unless --overwrite")
    ap.add_argument("--overwrite", action="store_true", help="with --fresh: move an existing run aside and restart")
    a = ap.parse_args(argv)

    if a.cmd is None:
        if a.experiment != "dbitnet_all_differences":
            ap.error("a subcommand is required (or --experiment dbitnet_all_differences)")
        a.cmd = "all-diffs"
    if a.baseline_dir:
        os.environ["NDGOHR_BASELINE_DIR"] = a.baseline_dir
    if a.mode == "full" and not a.full_experiment:
        ap.error("--mode full requires --full-experiment (deliberate opt-in)")

    from aesdist import experiment as X
    out = X.out_dir_from(a.out_dir)

    if a.cmd == "env-report":
        from aesdist.envinfo import collect_env
        from aesdist.io_utils import dumps
        print(dumps(collect_env()))
        return 0
    cfg = X.load_config(a.config)
    if a.cmd == "smoke":
        from aesdist.envinfo import configure_device
        from aesdist.smoke import run_smoke
        return 0 if run_smoke(cfg, out, device=a.device).ok else 1
    if a.cmd in ("all-diffs", "summarize-diffs"):
        from aesdist import alldiff
        root = alldiff.root_dir(out, a.mode)
        if a.cmd == "summarize-diffs":
            from aesdist.alldiff_summary import summarize
            summarize(root, plots=not a.no_plots)
            return 0
        if not 0 <= a.start <= a.end <= 255:
            ap.error("need 0 <= --start <= --end <= 255")
        from aesdist.baseline_adapter import load_aes
        from aesdist.envinfo import configure_device
        from aesdist.train import StopFlag
        m = cfg["modes"][a.mode]
        print(f"MODE={a.mode}: n={m['n']} epochs={m['epochs']} batch={m['batch_size']} -> {root}", flush=True)
        print("device:", configure_device(a.device), flush=True)
        aes = load_aes()
        alldiff.preflight(aes)
        variants = alldiff.VARIANTS if a.mixcolumns == "both" else (a.mixcolumns,)
        code, status = alldiff.run_all(cfg, aes, out, a.start, a.end, variants, a.seed, a.mode, a.protocol,
                                       overwrite=a.overwrite, purge=a.purge_data, force_unlock=a.force_unlock,
                                       stop=StopFlag().install())
        from collections import Counter
        print("summary:", dict(Counter(status.values())), "exit", code)
        return code
    if not a.experiment or a.experiment not in cfg["experiments"]:
        ap.error(f"--experiment required; choices: {list(cfg['experiments'])}")
    from aesdist.baseline_adapter import load_aes
    from aesdist.envinfo import collect_env, configure_device
    from aesdist.io_utils import atomic_write_json
    print("device:", configure_device(a.device))
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "env_info.json", collect_env())
    if a.cmd == "gen-data":
        st = X.make_store(cfg, a.experiment, a.mode, out)
        print("dataset dir:", st.dir, "| estimate:", __import__("aesdist.datagen", fromlist=["x"]).estimate_resources(st.cfg["n"], st.cfg["chunk_size"]))
        st.generate(load_aes())
        st.write_metadata()
        print("verify:", st.verify() or "all chunks OK")
        return 0
    if a.cmd == "train":
        from aesdist.train import StopFlag
        row = X.run_one(cfg, a.experiment, a.arch, a.protocol or "controlled", a.mode, a.seed, out, resume=a.resume,
                        overwrite=a.overwrite, stop=StopFlag().install())
        print(row if not isinstance(row, dict) or "test" not in row else {k: row["test"][k] for k in ("accuracy", "tpr", "tnr")})
        return 0


if __name__ == "__main__":
    sys.exit(main())
