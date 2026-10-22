import numpy as np
import pytest

from aesdist.partial_decrypt import last_round_subkey_bytes, ct_indices
from aesdist import wkrp


def make_data(aes, n=64, seed=21):
    rng = np.random.default_rng(seed)
    pt0 = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    pt1 = pt0.copy(); pt1[:, 0] ^= 0x80
    keys = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    ct = np.stack([aes.aes_encrypt(pt0.copy(), keys, 3, False), aes.aes_encrypt(pt1.copy(), keys, 3, False)], 1).astype(np.uint8)
    return ct, last_round_subkey_bytes(aes, keys, 3).astype(np.uint8), keys, pt0, pt1


def test_delta_zero_reproduces_true_2round_state_and_nonzero_does_not(aes):
    ct, sk, keys, pt0, pt1 = make_data(aes)
    cidx = ct_indices([0, 1])
    ct_sel, ts = ct[:, :, cidx], sk[:, cidx]
    X = wkrp.wkrp_inputs(aes.inv_s_box, ct_sel, ts, np.array([[0, 0], [1, 0], [0, 7]], dtype=np.uint8))
    N = ct.shape[0]
    # expected: 2-round (with final MC) ciphertexts of both plaintexts, bytes 0 and 1, as bits
    e0 = aes.aes_encrypt(pt0.copy(), keys, 2, True)[:, :2].astype(np.uint8)
    e1 = aes.aes_encrypt(pt1.copy(), keys, 2, True)[:, :2].astype(np.uint8)
    exp = np.unpackbits(np.stack([e0, e1], 1), axis=2).transpose(0, 2, 1).astype(np.float32)
    assert X.shape == (3 * N, 16, 2)
    assert np.array_equal(X[:N], exp)
    assert not np.array_equal(X[N:2 * N], exp) and not np.array_equal(X[2 * N:], exp)


def test_resume_equals_uninterrupted_and_counters(aes, tmp_path):
    ct, sk, *_ = make_data(aes, n=16)
    def score(X):   # deterministic stand-in scorer (NOT a distinguisher); tests mechanics only
        return 1.0 / (1.0 + np.exp(-(X.reshape(len(X), -1).sum(1) - 16) / 5.0))
    kw = dict(score_fn=score, inv_s_box=None)
    run = lambda d, **k: wkrp.run_wkrp(score, aes.inv_s_box, ct, sk, d, delta_chunk=256, n_deltas=1024,
                                       max_rows=1000, log=lambda *_: None, **k)
    full = tmp_path / "full"; part = tmp_path / "part"
    m_full = run(full)
    assert m_full["complete"] and m_full["counters"]["model_evals"] == 1024 * 16
    m1 = run(part, stop_after_chunks=2)
    assert not m1["complete"] and len(m1["completed"]) == 2
    m2 = run(part)
    assert m2["complete"]
    a = wkrp.merge_chunks(full, 4).read_text(); b = wkrp.merge_chunks(part, 4).read_text()
    assert a == b
    s = wkrp.summarize(full / "wkrp_results.csv")
    assert s["n_candidates"] == 1024 and s["n_examples"] == 16
    with pytest.raises(RuntimeError):                      # different data must not reuse the directory
        wkrp.run_wkrp(score, aes.inv_s_box, ct[::-1].copy(), sk[::-1].copy(), full, delta_chunk=256, n_deltas=1024, log=lambda *_: None)
