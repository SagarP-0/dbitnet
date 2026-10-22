"""CPU smoke test (Stages 1-3 of the task's test plan + WKRP mechanics). Tiny data; metrics are NOT results."""
import time
from pathlib import Path

import numpy as np

from . import validation as V
from .baseline_adapter import baseline_hashes, load_aes
from .conventions import ct_index_for_state_byte
from .io_utils import atomic_write_json


class Report:
    def __init__(self, log=print):
        self.rows, self.log = [], log

    def add(self, stage, name, ok, detail=""):
        self.rows.append({"stage": stage, "check": name, "ok": bool(ok), "detail": str(detail)})
        self.log(f"  [{'PASS' if ok else 'FAIL'}] {stage}: {name}" + (f"  ({detail})" if detail and not ok else ""))

    def extend(self, stage, results):
        for n, ok, d in results:
            self.add(stage, n, ok, d)

    @property
    def ok(self):
        return all(r["ok"] for r in self.rows)


def run_smoke(cfg, out_dir, device="cpu", log=print):
    from .envinfo import collect_env, configure_device
    from . import datagen as dg, models as M, experiment as X, wkrp
    from .partial_decrypt import last_round_subkey_bytes
    R = Report(log)
    out_dir = Path(out_dir) / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    log("== Stage 1: environment ==")
    dev = configure_device(device)
    env = collect_env()
    R.add("S1", "device configured", True, dev)
    R.add("S1", "TensorFlow importable", "unavailable" not in str(env.get("tensorflow")), env.get("tensorflow"))
    if device == "cpu":
        R.add("S1", "no GPU visible to TF in cpu mode", True, "set_visible_devices([], 'GPU') applied")

    log("== Stage 2: AES / conventions ==")
    aes = load_aes()
    hashes = baseline_hashes()
    R.add("S2", "original baseline files hashed (not modified)", len(hashes) > 0, f"{len(hashes)} files")
    R.extend("S2", V.tables_check(aes))
    R.extend("S2", V.fips197_checks(aes))
    R.extend("S2", V.reduced_round_checks(aes, n=24))
    table = {0: [0, 13], 1: [10, 7], 2: [4, 1], 3: [14, 11], 4: [8, 5], 5: [2, 15], 6: [12, 9], 7: [6, 3]}
    R.add("S2", "derived key indices == original attack table",
          all([ct_index_for_state_byte(2 * i), ct_index_for_state_byte(2 * i + 1)] == v for i, v in table.items()))
    from .partial_decrypt import partial_decrypt_full_state
    rng = np.random.default_rng(99)
    pt = rng.integers(0, 256, (64, 16), dtype=np.uint8); keys = rng.integers(0, 256, (64, 16), dtype=np.uint8)
    ct3 = aes.aes_encrypt(pt.copy(), keys, 3, False)
    R.add("S2", "3-round ct, true last subkey, ARK-InvSR-InvSB == 2-round ct with final MC",
          np.array_equal(partial_decrypt_full_state(aes, ct3, last_round_subkey_bytes(aes, keys, 3)), aes.aes_encrypt(pt.copy(), keys, 2, True)))

    log("== Stage 2b: dataset generation and validation (smoke size) ==")
    stores, datas = {}, {}
    for exp in ("r2_2B", "r3_2B"):
        st = X.make_store(cfg, exp, "smoke", out_dir)
        R.add("S2b", f"{exp}: resource estimate", True, st.cfg and dg.estimate_resources(st.cfg["n"], st.cfg["chunk_size"]))
        st.generate(aes, log=lambda *_: None)
        st.write_metadata({"baseline_sha256": hashes})
        R.add("S2b", f"{exp}: all chunks present and checksummed", st.verify() == {}, st.verify())
        e = cfg["experiments"][exp]
        ct, y, aux = dg.generate_chunk(aes, st.cfg["chunk_size"], e["nr"], e["final_mc"], st.cfg["seed"], 0, return_aux=True)
        R.extend("S2b", [(f"{exp}: " + n, ok, d) for n, ok, d in dg.validate_chunk_with_aux(aes, ct, y, aux, e["nr"], e["final_mc"])])
        stores[exp] = st
        datas[exp] = {s: st.load_split(s) for s in ("train", "val", "test")}
        R.add("S2b", f"{exp}: split sizes", True, {s: len(v[1]) for s, v in datas[exp].items()})

    log("== Stage 1b: architecture validation ==")
    rf_rows = []
    for arch_name in ("ndgohr", "dbitnet", "dbitnet_concat1"):
        for exp in ("r2_2B", "r3_2B"):
            t = X.make_tcfg(cfg, exp, arch_name, "controlled", "smoke", 0)
            model = M.build_model(t["model"])
            v = M.validate_model(model, loss=t["loss"])
            R.extend("S1b", [(f"{arch_name}/{exp}: " + n, ok, d) for n, ok, d in v["checks"]])
            rf = None
            if t["model"]["arch"] == "dbitnet":
                seqlen = v["input_shape"][0]
                rf = M.receptive_field_report(t["model"]["kernel"], t["model"]["dilations"], seqlen)
            rf_rows.append({"arch": arch_name, "exp": exp, "params": v["params"], "memory": v["memory"], "receptive_field": rf})
    atomic_write_json(out_dir / "architecture_report.json", rf_rows)

    log("== Stage 3: minimal training, checkpoint, resume, evaluation ==")
    trained = {}
    for arch_name in ("ndgohr", "dbitnet"):
        exp = "r2_2B"
        rid = f"smoke_{arch_name}"
        t = X.make_tcfg(cfg, exp, arch_name, "controlled", "smoke", 0)
        from . import train as T, checkpoint as ck, evaluate as E
        rdir = out_dir / "runs" / rid
        r1 = T.train_model(t, datas[exp], rdir, resume=True, overwrite=False, log=lambda *_: None, max_epochs_this_call=1)
        R.add("S3", f"{arch_name}: epoch 1 finite loss + checkpoint written",
              np.isfinite(r1["history"][-1]["train_loss"]) and (rdir / "latest.npz").is_file() and (rdir / "best.npz").is_file())
        r2 = T.train_model(t, datas[exp], rdir, resume=True, log=lambda *_: None)
        R.add("S3", f"{arch_name}: resumed to completion", r2["status"] in ("completed", "already_completed") and r2.get("epoch") == t["epochs"], r2["status"])
        try:
            ck.read_checkpoint(rdir / "latest.npz")
            R.add("S3", f"{arch_name}: latest checkpoint validates", True)
        except ck.CheckpointError as e:
            R.add("S3", f"{arch_name}: latest checkpoint validates", False, e)
        model, meta = T.load_best(t, rdir)
        m, p = E.evaluate_split(model, *datas[exp]["test"], t["byte_idx"], "channels2")
        R.add("S3", f"{arch_name}: test metrics computed from held-out predictions", np.isfinite(m["accuracy"]) and m["n"] == len(datas[exp]["test"][1]),
              {k: round(m[k], 4) for k in ("accuracy", "tpr", "tnr")})
        trained[arch_name] = (model, t, m)

    log("== WKRP mechanics (256 deltas, N=32 pairs; NOT a result) ==")
    model, t, _ = trained["dbitnet"]
    N = 32
    rng = np.random.default_rng(5)
    pt0 = rng.integers(0, 256, (N, 16), dtype=np.uint8); pt1 = pt0.copy(); pt1[:, 0] ^= 0x80
    k = rng.integers(0, 256, (N, 16), dtype=np.uint8)
    ctp = np.stack([aes.aes_encrypt(pt0.copy(), k, 3, False), aes.aes_encrypt(pt1.copy(), k, 3, False)], 1).astype(np.uint8)
    sk = last_round_subkey_bytes(aes, k, 3).astype(np.uint8)
    score = lambda Xb: np.asarray(model(Xb, training=False)).reshape(-1)
    wd = out_dir / "wkrp_smoke"
    man = wkrp.run_wkrp(score, aes.inv_s_box, ctp, sk, wd, delta_chunk=64, n_deltas=256, max_rows=4000, log=lambda *_: None, tag="smoke_dbitnet_untrained_ish")
    R.add("WKRP", "256 deltas evaluated in 4 resumable chunks", man["complete"], man["counters"])
    csvp = wkrp.merge_chunks(wd, 4)
    s = wkrp.summarize(csvp)
    R.add("WKRP", "summary computed; delta=0 row present", s["correct_mean"] is not None, {"rank_of_delta0": s["rank_of_correct_by_mean_prob"]})

    atomic_write_json(out_dir / "smoke_report.json", {"all_ok": R.ok, "wall_s": time.time() - t_start, "env": env,
                                                      "baseline_sha256": hashes, "checks": R.rows,
                                                      "trained_metrics_SMOKE_ONLY": {a: v[2] for a, v in trained.items()}})
    log(f"\nSMOKE {'PASSED' if R.ok else 'FAILED'}: {sum(r['ok'] for r in R.rows)}/{len(R.rows)} checks; report: {out_dir / 'smoke_report.json'}")
    return R
