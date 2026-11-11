import json
import os

import numpy as np
import pytest

from aesdist import alldiff as A
from aesdist import datagen as dg
from aesdist import reference_aes as ref
from aesdist.experiment import load_config

ROOT = os.path.join(os.path.dirname(__file__), "..")


@pytest.fixture(scope="module")
def cfg():
    c = load_config(os.path.join(ROOT, "config", "experiments.yaml"))
    c["modes"]["smoke"] = {"n": 2000, "chunk_size": 200, "epochs": 2, "batch_size": 250, "seeds": [0]}
    return c


@pytest.mark.parametrize("d", [0, 1, 0x80, 0xFF])
@pytest.mark.parametrize("variant", A.VARIANTS)
def test_difference_construction_and_classes(aes, cfg, d, variant):
    mc = A.final_mc(cfg, variant)
    ct, y, aux = dg.generate_chunk(aes, 400, 3, mc, 0, 0, diff=d, return_aux=True, paired=True)
    pos = y == 1
    delta = np.zeros(16, dtype=np.uint8)
    delta[0] = d
    assert np.all((aux["pt0"] ^ aux["pt1"])[pos] == delta)                       # P1 = P0 ^ [d,0,...,0]
    assert int(pos.sum()) == 200 and (aux["pt0"] ^ aux["pt1"])[~pos].any(axis=1).all()
    res = dg.validate_chunk_with_aux(aes, ct, y, aux, 3, mc, diff=d)
    assert all(ok for _, ok, _ in res), res
    if d == 0:                                                                   # identical plaintexts -> identical ciphertexts
        assert np.array_equal(ct[pos, 0], ct[pos, 1])


def test_round_count_and_exact_mixcolumns_distinction(aes):
    rng = np.random.default_rng(1)
    pt = rng.integers(0, 256, (12, 16), dtype=np.uint8)
    key = rng.integers(0, 256, (12, 16), dtype=np.uint8)
    on = aes.aes_encrypt(pt.copy(), key, 3, True).astype(np.uint8)
    off = aes.aes_encrypt(pt.copy(), key, 3, False).astype(np.uint8)
    for i in range(12):
        k3 = ref.expand_key(list(key[i]), 3)[3]
        assert list(on[i]) == ref.encrypt(list(pt[i]), list(key[i]), 3, True)     # 3 rounds, final MC
        assert list(off[i]) == ref.encrypt(list(pt[i]), list(key[i]), 3, False)   # 3 rounds, standard final round
        # the variants differ only by MixColumns before the last AddRoundKey
        assert [a ^ b for a, b in zip(on[i], k3)] == ref._mix_columns([a ^ b for a, b in zip(off[i], k3)])
        assert list(off[i]) != ref.encrypt(list(pt[i]), list(key[i]), 2, False)   # not a 2-round cipher
        assert list(off[i]) != ref.encrypt(list(pt[i]), list(key[i]), 4, False)


def test_paired_streams_share_everything_but_the_cipher(aes):
    ct_on, y_on, a_on = dg.generate_chunk(aes, 200, 3, True, 7, 1, diff=5, return_aux=True, paired=True)
    ct_off, y_off, a_off = dg.generate_chunk(aes, 200, 3, False, 7, 1, diff=5, return_aux=True, paired=True)
    assert np.array_equal(y_on, y_off) and all(np.array_equal(a_on[k], a_off[k]) for k in a_on)
    assert not np.array_equal(ct_on, ct_off)
    _, y9, a9 = dg.generate_chunk(aes, 200, 3, True, 7, 1, diff=9, return_aux=True, paired=True)   # other difference
    assert np.array_equal(y_on, y9) and np.array_equal(a_on["pt0"], a9["pt0"]) and np.array_equal(a_on["keys"], a9["keys"])
    neg = y_on == 0
    assert np.array_equal(a_on["pt1"][neg], a9["pt1"][neg])                      # negatives identical across differences
    assert not np.array_equal(a_on["pt1"][~neg], a9["pt1"][~neg])
    # opt-in only: the unpaired stream still depends on final_mc, and unpaired dataset hashes are unchanged
    u_on = dg.generate_chunk(aes, 200, 3, True, 7, 1)
    u_off = dg.generate_chunk(aes, 200, 3, False, 7, 1)
    assert not np.array_equal(u_on[1], u_off[1])
    assert "paired_mc_stream" not in dg.DatasetStore("/tmp/x", 3, True, 2000, 200, 7).cfg


def test_input_dimensions(cfg):
    tf = pytest.importorskip("tensorflow")
    from aesdist.models import build_model, receptive_field
    for v in A.VARIANTS:
        t = A.make_tcfg(cfg, v, 0x80, "smoke", 0)
        assert t["model"]["arch"] == "dbitnet" and t["model"]["bits"] == 128 and t["byte_idx"] == list(range(16))
        assert t["loss"] == "mse" and t["schedule"] == "cyclic" and t["final_mc"] == (v == "enabled")
        m = build_model(t["model"])
        assert tuple(m.input_shape) == (None, 128, 2) and tuple(m.output_shape) == (None, 1)
        assert receptive_field(3, t["model"]["dilations"]) == 63
    f = load_config(os.path.join(ROOT, "config", "experiments.yaml"))["modes"]["full"]
    assert (f["n"], f["epochs"], f["batch_size"]) == (1_000_000, 20, 5000)       # untouched full protocol


@pytest.fixture()
def quiet():
    return lambda *_: None


def test_run_skip_resume_corruption_failure_and_interrupt(aes, cfg, tmp_path, quiet, monkeypatch):
    pytest.importorskip("tensorflow")
    from aesdist import train as T
    out = tmp_path
    root = A.root_dir(out, "smoke")
    # partial run (1 of 2 epochs) for both variants, then resume to completion
    for v in A.VARIANTS:
        assert A.run_diff_config(cfg, aes, v, 128, "smoke", 0, root, log=quiet, max_epochs_this_call=1) == "partial"
        assert A.load_result(A.diff_dir(root, v, 128), log=quiet) is None          # partial output is not "completed"
    for v in A.VARIANTS:
        assert A.run_diff_config(cfg, aes, v, 128, "smoke", 0, root, log=quiet) == "completed"
    rp = A.diff_dir(root, "enabled", 128) / "result.json"
    r = json.loads(rp.read_text())
    assert r["epochs_run"] == 2 and r["status"] == "completed" and r["difference_hex"] == "80" and r["mixcolumns"] == "enabled"
    assert 0 <= r["test"]["accuracy"] <= 1 and r["inference"]["latency_ms_median"] > 0
    for f in ("latest.npz", "best.npz", "history.json", "config.json", "status.json", "run_meta.json"):
        assert (A.diff_dir(root, "enabled", 128) / f).is_file()
    # completed experiments are skipped, nothing rewritten
    m0 = rp.stat().st_mtime_ns
    assert A.run_diff_config(cfg, aes, "enabled", 128, "smoke", 0, root, log=quiet) == "skipped"
    assert rp.stat().st_mtime_ns == m0
    # truncated result.json is detected, set aside, and rebuilt from the finished training without retraining
    rp.write_text(rp.read_text()[:120])
    lp = A.diff_dir(root, "enabled", 128) / "latest.npz"
    l0 = lp.stat().st_mtime_ns
    assert A.run_diff_config(cfg, aes, "enabled", 128, "smoke", 0, root, log=quiet) == "completed"
    assert list(rp.parent.glob("result.json.invalid-*")) and lp.stat().st_mtime_ns == l0
    assert A.load_result(rp.parent, log=quiet) is not None
    # tampered but parsable result is rejected by the content hash
    d = json.loads(rp.read_text()); d["test"]["accuracy"] = 0.999; rp.write_text(json.dumps(d))
    assert A.load_result(rp.parent, log=quiet) is None
    A.run_diff_config(cfg, aes, "enabled", 128, "smoke", 0, root, log=quiet)
    # a different configuration must not silently reuse the directory
    cfg2 = json.loads(json.dumps(cfg)); cfg2["modes"]["smoke"]["epochs"] = 3
    with pytest.raises(A.RunMismatch):
        A.run_diff_config(cfg2, aes, "enabled", 128, "smoke", 0, root, log=quiet)
    # failure isolation: difference 1 fails, difference 2 still runs; exit code reports the failure
    orig = dg.DatasetStore.generate
    def boom(self, aes_, *a, **k):
        if self.cfg["diff"] == 1:
            raise OSError("disk full (injected)")
        return orig(self, aes_, *a, **k)
    monkeypatch.setattr(dg.DatasetStore, "generate", boom)
    code, st = A.run_all(cfg, aes, out, 1, 2, ("disabled",), 0, "smoke", log=quiet)
    assert code == 1 and st == {"disabled/01": "failed", "disabled/02": "completed"}
    assert (A.diff_dir(root, "disabled", 1) / "failure.json").is_file()
    monkeypatch.setattr(dg.DatasetStore, "generate", orig)
    # interrupt: stop flag halts the whole loop (no further differences), and a rerun resumes
    stop = T.StopFlag(); stop.requested = True
    code, st = A.run_all(cfg, aes, out, 3, 4, ("disabled",), 0, "smoke", log=quiet, stop=stop)
    assert code == 3 and list(st) == ["disabled/03"]
    code, st = A.run_all(cfg, aes, out, 3, 4, ("disabled",), 0, "smoke", log=quiet)
    assert code == 0 and st == {"disabled/03": "completed", "disabled/04": "completed"}
    # stale lock handling
    lock = A.diff_dir(root, "disabled", 5) / ".lock"
    lock.parent.mkdir(parents=True); lock.write_text("dead job")
    assert A.run_all(cfg, aes, out, 5, 5, ("disabled",), 0, "smoke", log=quiet)[1] == {"disabled/05": "failed"}
    assert A.run_all(cfg, aes, out, 5, 5, ("disabled",), 0, "smoke", log=quiet, force_unlock=True)[1] == {"disabled/05": "completed"}
    # metadata + summaries (+ plots when matplotlib exists)
    meta = json.loads((root / "experiment_metadata.json").read_text())
    assert meta["variants"]["enabled"]["final_mc"] is True and meta["variants"]["disabled"]["final_mc"] is False
    from aesdist.alldiff_summary import summarize
    comp = summarize(root, plots=False, log=quiet)
    sd = root / "summaries"
    assert {p.name for p in sd.iterdir()} >= {"mixcolumns_enabled.json", "mixcolumns_disabled.json", "comparison.json", "comparison.csv"}
    en = json.loads((sd / "mixcolumns_enabled.json").read_text())
    assert en["counts"]["completed"] == 1 and en["counts"]["missing"] == 255
    assert "delta_acc_enabled_minus_disabled" in comp[128]
    pytest.importorskip("matplotlib")
    summarize(root, plots=True, log=quiet)
    assert (sd / "plots" / "acc_vs_difference.png").is_file() and (sd / "plots" / "enabled_vs_disabled.pdf").is_file()
