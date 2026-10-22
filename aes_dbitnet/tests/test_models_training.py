import json
import os

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
tf.config.set_visible_devices([], "GPU")

from aesdist import checkpoint as ck
from aesdist import datagen as dg
from aesdist import models as M
from aesdist import train as T
from aesdist.evaluate import benchmark_inference, evaluate_split
from aesdist.metrics import binary_metrics


def test_ndgohr_param_counts_match_hand_derivation():
    # Hand derivation from main_net.make_resnet (see docs): 29,777 (bits=16), 144,465 (bits=128)
    for bits, total in ((16, 29777), (128, 144465)):
        pc = M.param_counts(M.build_ndgohr(bits))
        assert pc["total"] == total and pc["non_trainable"] == 608, pc


def test_ndgohr_flat_and_channels2_are_numerically_identical():
    a = M.build_ndgohr(16, "flat"); b = M.build_ndgohr(16, "channels2")
    b.set_weights(a.get_weights())
    rng = np.random.default_rng(0)
    flat = rng.integers(0, 2, (32, 32)).astype(np.float32)
    ch = flat.reshape(32, 2, 16).transpose(0, 2, 1)
    assert np.allclose(a(flat, training=False), b(ch, training=False), atol=1e-6)


def test_receptive_field():
    assert M.receptive_field(3, [1, 2, 4, 8, 16]) == 63
    assert M.receptive_field(5, [1, 2, 4, 8, 16]) == 125
    r = M.receptive_field_report(3, [1, 2, 4, 8, 16], 16)
    assert r["rf_covers_sequence"] and r["layers_with_dilation_ge_seq_len"] == [16]
    assert not M.receptive_field_report(3, [1, 2, 4, 8, 16], 128)["rf_covers_sequence"]


@pytest.mark.parametrize("arch,mode,bits", [("dbitnet", "channels2", 16), ("dbitnet", "concat1", 16),
                                            ("dbitnet", "channels2", 128), ("ndgohr", "channels2", 16)])
def test_architecture_validation(arch, mode, bits):
    model = M.build_model({"arch": arch, "bits": bits, "input_mode": mode})
    v = M.validate_model(model)
    assert all(ok for _, ok, _ in v["checks"]), v["checks"]
    assert v["params"]["total"] == v["params"]["trainable"] + v["params"]["non_trainable"]


def test_dbitnet_config_options_and_errors():
    m = M.build_dbitnet(32, kernel=5, filters=64, residual=True, pool="flatten", second_dense=True)
    assert tuple(m.output_shape) == (None, 1)
    with pytest.raises(ValueError):
        M.build_dbitnet(16, n_layers=5, dilations=[1, 2])
    with pytest.raises(ValueError):
        M.build_dbitnet(16, input_mode="flat")


def test_metrics_exact():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([.9, .8, .2, .1, .6, .4, .3, .2])
    m = binary_metrics(y, p)
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (2, 1, 4, 1)
    assert m["accuracy"] == 6 / 8 and m["tpr"] == 2 / 3 and m["tnr"] == 4 / 5 and m["precision"] == 2 / 3


@pytest.fixture(scope="module")
def aes_module():
    from aesdist.baseline_adapter import BaselineNotFound, load_aes
    try:
        return load_aes()
    except BaselineNotFound as e:
        pytest.skip(str(e))


def _tcfg(arch="dbitnet", epochs=3, **kw):
    cfg = {"model": {"arch": arch, "bits": 16, "input_mode": "channels2"}, "byte_idx": [0, 1], "loss": "bce",
           "lr": 1e-3, "min_lr": 1e-5, "epochs": epochs, "batch_size": 250, "schedule": "cosine", "seed": 3,
           "best_metric": "val_loss", "early_stopping": None, "snapshot_every": 0, "eval_batch": 1000}
    cfg.update(kw)
    return cfg


def _data(aes, tmp_path):
    st = dg.DatasetStore(tmp_path / "ds", nr=2, final_mc=True, n=3000, chunk_size=300, seed=2)
    st.generate(aes, log=lambda *_: None)
    return {s: st.load_split(s) for s in ("train", "val", "test")}


def test_train_checkpoint_resume_exact(aes_module, tmp_path):
    data = _data(aes_module, tmp_path)
    cfg = _tcfg(epochs=3)
    quiet = lambda *_: None
    # uninterrupted reference run
    r_full = T.train_model(cfg, data, tmp_path / "full", log=quiet)
    assert r_full["status"] == "completed" and len(r_full["history"]) == 3
    assert all(np.isfinite(h["train_loss"]) and np.isfinite(h["val_loss"]) for h in r_full["history"])
    # interrupted after 1 epoch, then resumed
    r1 = T.train_model(cfg, data, tmp_path / "part", log=quiet, max_epochs_this_call=1)
    assert r1["status"] == "partial" and r1["epoch"] == 1
    r2 = T.train_model(cfg, data, tmp_path / "part", resume=True, log=quiet)
    assert r2["status"] == "completed" and r2["epoch"] == 3
    wf = ck.read_checkpoint(tmp_path / "full" / "latest.npz")[0]
    wp = ck.read_checkpoint(tmp_path / "part" / "latest.npz")[0]
    maxdiff = max(float(np.max(np.abs(a - b))) for a, b in zip(wf, wp))
    assert maxdiff < 1e-5, f"resume diverged from uninterrupted run: {maxdiff}"
    hf, hp = [h["val_loss"] for h in r_full["history"]], [h["val_loss"] for h in r2["history"]]
    assert np.allclose(hf, hp, atol=1e-5)
    # best vs latest are separate files; best loads and evaluates
    assert (tmp_path / "part" / "best.npz").is_file() and (tmp_path / "part" / "latest.npz").is_file()
    model, meta = T.load_best(cfg, tmp_path / "part")
    m, p = evaluate_split(model, *data["test"], cfg["byte_idx"], "channels2")
    assert m["n"] == 300 and np.isfinite(m["bce"]) and len(p) == 300
    b = benchmark_inference(model, data["test"][0], cfg["byte_idx"], "channels2", batch=300, n_single=20, repeats=2)
    assert b["throughput_samples_per_s"] > 0 and b["latency_ms_median"] > 0


def test_completed_run_not_overwritten_and_mismatch_rejected(aes_module, tmp_path):
    data = _data(aes_module, tmp_path)
    quiet = lambda *_: None
    cfg = _tcfg(epochs=1)
    T.train_model(cfg, data, tmp_path / "r", log=quiet)
    again = T.train_model(cfg, data, tmp_path / "r", log=quiet)
    assert again["status"] == "already_completed"
    # a partially-trained run directory must not be silently overwritten by a fresh run
    cfg2 = _tcfg(epochs=2, model={"arch": "dbitnet", "bits": 16, "input_mode": "channels2", "filters": 64})
    T.train_model(_tcfg(epochs=2), data, tmp_path / "m", log=quiet, max_epochs_this_call=1)
    with pytest.raises(T.RunExistsError):
        T.train_model(_tcfg(epochs=2), data, tmp_path / "m", resume=False, log=quiet)
    # ... unless overwrite is explicit, in which case the old run is moved aside, not deleted
    T.train_model(_tcfg(epochs=1), data, tmp_path / "m", resume=False, overwrite=True, log=quiet)
    assert list(tmp_path.glob("m.superseded-*"))
    T.train_model(_tcfg(epochs=2), data, tmp_path / "m2", log=quiet, max_epochs_this_call=1)
    with pytest.raises(ck.CheckpointError):
        T.train_model(cfg2, data, tmp_path / "m2", resume=True, log=quiet)


def test_corrupt_checkpoint_detected(aes_module, tmp_path):
    data = _data(aes_module, tmp_path)
    T.train_model(_tcfg(epochs=1), data, tmp_path / "c", log=lambda *_: None)
    p = tmp_path / "c" / "latest.npz"
    p.write_bytes(p.read_bytes()[:500])
    with pytest.raises(ck.CheckpointError):
        ck.read_checkpoint(p)


def test_stop_flag_discards_partial_epoch(aes_module, tmp_path):
    data = _data(aes_module, tmp_path)
    stop = T.StopFlag(); stop.requested = True
    r = T.train_model(_tcfg(epochs=2), data, tmp_path / "s", log=lambda *_: None, stop=stop)
    assert r["status"] == "interrupted" and r["epoch"] == 0
    r2 = T.train_model(_tcfg(epochs=2), data, tmp_path / "s", resume=True, log=lambda *_: None)
    assert r2["status"] == "completed"
