import numpy as np
import pytest

from aesdist import reference_aes as ref
from aesdist import validation as V


def _assert_all(results):
    bad = [(n, d) for n, ok, d in results if not ok]
    assert not bad, bad


def test_tables_match_mathematical_definition(aes):
    _assert_all(V.tables_check(aes))


def test_fips197_vectors_and_key_schedule(aes):
    _assert_all(V.fips197_checks(aes))


def test_reduced_round_and_final_mc_conventions(aes):
    _assert_all(V.reduced_round_checks(aes))


def test_full_aes_vs_pycryptodome(aes):
    Crypto = pytest.importorskip("Crypto.Cipher.AES")
    rng = np.random.default_rng(7)
    pt = rng.integers(0, 256, (40, 16), dtype=np.uint8)
    keys = rng.integers(0, 256, (40, 16), dtype=np.uint8)
    got = aes.aes_encrypt(pt.copy(), keys, 10, False)
    for i in range(40):
        exp = Crypto.new(keys[i].tobytes(), Crypto.MODE_ECB).encrypt(pt[i].tobytes())
        assert got[i].astype(np.uint8).tobytes() == exp  # original returns int64


def test_final_mc_flag_changes_output_only_in_final_round(aes):
    rng = np.random.default_rng(3)
    pt = rng.integers(0, 256, (8, 16), dtype=np.uint8)
    k = rng.integers(0, 256, (8, 16), dtype=np.uint8)
    assert not np.array_equal(aes.aes_encrypt(pt.copy(), k, 2, False), aes.aes_encrypt(pt.copy(), k, 2, True))


def test_decrypt_roundtrip_reduced_rounds(aes):
    rng = np.random.default_rng(5)
    pt = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    k = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    for nr in (2, 3):
        for mc in (False, True):
            ct = aes.aes_encrypt(pt.copy(), k, nr, mc)
            assert np.array_equal(aes.aes_decrypt(ct, k, nr, mc), pt)
