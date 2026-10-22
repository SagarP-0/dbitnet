import numpy as np
import pytest

from aesdist import datagen as dg
from aesdist.bits import encode_bits, to_model_input, iter_batches
from aesdist.conventions import DIFF


def test_chunk_determinism_and_independence(aes):
    a = dg.generate_chunk(aes, 200, 3, False, seed=5, chunk_idx=2)
    b = dg.generate_chunk(aes, 200, 3, False, seed=5, chunk_idx=2)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    c = dg.generate_chunk(aes, 200, 3, False, seed=6, chunk_idx=2)
    d = dg.generate_chunk(aes, 200, 3, False, seed=5, chunk_idx=3)
    assert not np.array_equal(a[0], c[0]) and not np.array_equal(a[0], d[0])
    e = dg.generate_chunk(aes, 200, 3, True, seed=5, chunk_idx=2)   # final_mc is part of the stream key
    assert not np.array_equal(a[1], e[1]) or not np.array_equal(a[0], e[0])


@pytest.mark.parametrize("nr,mc", [(2, True), (2, False), (3, False)])
def test_labels_differential_and_roundtrip(aes, nr, mc):
    ct, y, aux = dg.generate_chunk(aes, 400, nr, mc, 1, 0, return_aux=True)
    assert ct.shape == (400, 2, 16) and ct.dtype == np.uint8 and y.dtype == np.uint8
    res = dg.validate_chunk_with_aux(aes, ct, y, aux, nr, mc)
    assert all(ok for _, ok, _ in res), res
    assert int(y.sum()) == 200  # exactly balanced


@pytest.mark.parametrize("idx", [[0, 13], [0], [0, 1, 2, 3], list(range(16)), [3, 1]])
def test_encode_bits_matches_original_convert_to_binary(aes, idx):
    ct, y = dg.generate_chunk(aes, 100, 3, False, 9, 0)
    sel = np.hstack((ct[:, 0][:, idx], ct[:, 1][:, idx]))      # exactly what the original hstack does
    ref = aes.convert_to_binary(sel, len(idx))
    assert np.array_equal(encode_bits(ct, idx), ref)


def test_input_modes_layout(aes):
    ct, _ = dg.generate_chunk(aes, 20, 3, False, 9, 0)
    flat = encode_bits(ct, [0, 13]); nb = 2
    ch = to_model_input(flat, nb, "channels2")
    cc = to_model_input(flat, nb, "concat1")
    assert ch.shape == (20, 16, 2) and cc.shape == (20, 32, 1)
    # channel 0 holds C0 bits, channel 1 holds C1 bits, positions aligned
    assert np.array_equal(ch[:, :, 0], flat[:, :16]) and np.array_equal(ch[:, :, 1], flat[:, 16:])
    assert np.array_equal(cc[:, :16, 0], flat[:, :16]) and np.array_equal(cc[:, 16:, 0], flat[:, 16:])
    # matches the original Reshape((2,ws)) + Permute((2,1))
    assert np.array_equal(ch, flat.reshape(20, 2, 16).transpose(0, 2, 1))


def test_encode_rejects_bad_input():
    with pytest.raises(ValueError):
        encode_bits(np.zeros((4, 2, 16), dtype=np.int64), [0])
    with pytest.raises(ValueError):
        encode_bits(np.zeros((4, 16), dtype=np.uint8), [0])


def test_split_plan_and_store_lifecycle(aes, tmp_path):
    st = dg.DatasetStore(tmp_path, nr=3, final_mc=False, n=2000, chunk_size=200, seed=1)
    assert st.splits.count("train") == 8 and st.splits.count("val") == 1 and st.splits.count("test") == 1
    assert len(st.verify()) == st.n_chunks                       # nothing generated yet
    st.generate(aes, log=lambda *_: None, stop_after=4)          # simulate interruption
    assert len(st.verify()) == 6 and not st.is_complete()
    first = {k: (st._paths(k)[0].stat().st_mtime_ns) for k in range(4)}
    st.generate(aes, log=lambda *_: None)                        # resume
    assert st.verify() == {} and st.is_complete()
    assert all(st._paths(k)[0].stat().st_mtime_ns == first[k] for k in range(4)), "valid chunks must not be rewritten"
    # splits: disjoint, correct sizes, balanced
    sizes = {}
    for s in ("train", "val", "test"):
        ct, y = st.load_split(s)
        sizes[s] = len(y)
        assert int(y.sum()) * 2 == len(y)
    assert sizes == {"train": 1600, "val": 200, "test": 200}
    # corruption detection: truncate one chunk file, flip a byte in another
    p0 = st._paths(2)[0]; p0.write_bytes(p0.read_bytes()[:100])
    arr = np.load(st._paths(5)[0]); arr[0, 0, 0] ^= 1; np.save(st._paths(5)[0], arr)
    assert st.verify() == {2: "corrupt", 5: "corrupt"}
    keep = st._paths(0)[0].stat().st_mtime_ns
    st.generate(aes, log=lambda *_: None)                        # regenerates only the bad ones
    assert st.verify() == {} and st._paths(0)[0].stat().st_mtime_ns == keep
    with pytest.raises(ValueError):
        dg.assign_splits(5)                                      # cannot make 3 non-empty 80/10/10 splits? 4/0/1
