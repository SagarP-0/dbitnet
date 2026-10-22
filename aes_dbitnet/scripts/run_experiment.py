#!/usr/bin/env python
"""Non-interactive entry point. Subcommands: env-report | smoke | gen-data | train.

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
    ap.add_argument("cmd", choices=["env-report", "smoke", "gen-data", "train"])
    ap.add_argument("--config", default=str(ROOT / "config" / "experiments.yaml"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--baseline-dir", default=None, help="original NDGohr repo dir (read-only); or NDGOHR_BASELINE_DIR")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--full-experiment", action="store_true", help="required to run --mode full")
    ap.add_argument("--experiment", help="key under experiments: in the config")
    ap.add_argument("--arch", default="dbitnet", help="ndgohr | dbitnet | dbitnet_concat1")
    ap.add_argument("--protocol", default="controlled")
    ap.add_argument("--seed", type=int, default=0)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--resume", dest="resume", action="store_true", default=True)
    g.add_argument("--fresh", dest="resume", action="store_false", help="start fresh; fails if a partial run exists unless --overwrite")
    ap.add_argument("--overwrite", action="store_true", help="with --fresh: move an existing run aside and restart")
    a = ap.parse_args(argv)

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
        row = X.run_one(cfg, a.experiment, a.arch, a.protocol, a.mode, a.seed, out, resume=a.resume,
                        overwrite=a.overwrite, stop=StopFlag().install())
        print(row if not isinstance(row, dict) or "test" not in row else {k: row["test"][k] for k in ("accuracy", "tpr", "tnr")})
        return 0


if __name__ == "__main__":
    sys.exit(main())
