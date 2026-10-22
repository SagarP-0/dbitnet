"""Executable validation checks shared by pytest and the CPU smoke test.

Every function returns a list of (name, ok, detail). Nothing here is a literature value:
each check recomputes something and compares.
"""
import numpy as np

from . import reference_aes as ref

# FIPS-197 Appendix B (cipher example) and Appendix C.1 (AES-128). Public test vectors.
FIPS_B = dict(
    key="2b7e151628aed2a6abf7158809cf4f3c",
    pt="3243f6a8885a308d313198a2e0370734",
    ct="3925841d02dc09fbdc118597196a0b32",
    # state at the start of round 2 (after round 1), Appendix B
    after_round1="a49c7ff2689f352b6b5bea43026a5049",
    rk1="a0fafe1788542cb123a339392a6c7605",
    rk10="d014f9a8c9ee2589e13f0cc8b6630ca6",
)
FIPS_C1 = dict(
    key="000102030405060708090a0b0c0d0e0f",
    pt="00112233445566778899aabbccddeeff",
    ct="69c4e0d86a7b0430d8cdb78070b4c55a",
)


def _hex(a):
    """Hex of an array whose values are bytes. NOTE: the original aes_encrypt returns int64
    (its s_box table is an int64 array), so values must be cast before tobytes()."""
    a = np.asarray(a)
    assert a.min() >= 0 and a.max() < 256
    return a.astype(np.uint8).ravel().tobytes().hex()


def _b(hexstr):
    return np.frombuffer(bytes.fromhex(hexstr), dtype=np.uint8)


def tables_check(aes):
    sb = np.array(ref.SBOX, dtype=np.uint8)
    isb = np.array(ref.INV_SBOX, dtype=np.uint8)
    return [
        ("original s_box == S-box generated from GF(2^8) definition", bool(np.array_equal(aes.s_box.astype(np.uint8), sb)), ""),
        ("original inv_s_box == generated inverse", bool(np.array_equal(aes.inv_s_box.astype(np.uint8), isb)), ""),
    ]


def fips197_checks(aes):
    out = []
    for name, v in (("FIPS-197 App.B", FIPS_B), ("FIPS-197 App.C.1", FIPS_C1)):
        pt, key = _b(v["pt"])[None, :], _b(v["key"])[None, :]
        ct = aes.aes_encrypt(pt.copy(), key, 10, False)
        out.append((f"{name}: original aes_encrypt(nr=10, no final MC) == vector",
                    _hex(ct[0]) == v["ct"], _hex(ct[0])))
        out.append((f"{name}: pure-Python reference == vector",
                    bytes(ref.encrypt(list(pt[0]), list(key[0]), 10)).hex() == v["ct"], ""))
        dec = aes.aes_decrypt(ct, key, 10, False)
        out.append((f"{name}: original decrypt(encrypt(pt)) == pt", _hex(dec[0]) == v["pt"], ""))
    # key schedule (App. A.1 / B round keys). round_keys[r][n][row][col]; flat idx = 4*col+row
    key = _b(FIPS_B["key"])[None, :]
    rks = aes.key_expansion(key, 10)
    flat = lambda r: _hex(rks[r][0].transpose(1, 0).reshape(16))  # -> index 4*col+row
    out.append(("key schedule: round key 1 (FIPS-197 A.1)", flat(1) == FIPS_B["rk1"], flat(1)))
    out.append(("original aes_encrypt output dtype (documented quirk)", True, str(aes.aes_encrypt(_b(FIPS_B["pt"])[None, :].copy(), key, 2, False).dtype)))
    out.append(("key schedule: round key 10 (FIPS-197 A.1)", flat(10) == FIPS_B["rk10"], flat(10)))
    out.append(("reference: state after round 1 (FIPS-197 App.B)",
                bytes(ref.encrypt(list(_b(FIPS_B["pt"])), list(key[0]), 10, stop_after_round=1)).hex() == FIPS_B["after_round1"], ""))
    return out


def reduced_round_checks(aes, rounds=(1, 2, 3, 4), n=48, seed=1234):
    """Original aes_encrypt vs independent reference for reduced rounds, both final-MC settings."""
    rng = np.random.default_rng(seed)
    pt = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    keys = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    out = []
    for nr in rounds:
        for mc in (False, True):
            got = aes.aes_encrypt(pt.copy(), keys, nr, mc)
            exp = np.array([ref.encrypt(list(pt[i]), list(keys[i]), nr, mc) for i in range(n)], dtype=np.uint8)
            out.append((f"reduced-round nr={nr} final_mc={mc}: original == reference ({n} random vectors)",
                        bool(np.array_equal(got, exp)), ""))
    # key-schedule prefix property (needed for 3-round ct -> 2-round state partial decryption)
    k3 = aes.key_expansion(keys, 3)
    k2 = aes.key_expansion(keys, 2)
    out.append(("key_expansion(nr=3)[:3] == key_expansion(nr=2)", bool(np.array_equal(k3[:3], k2)), ""))
    # differential on byte 0 survives plaintext XOR convention: ARK0 is linear
    return out


def all_ok(results):
    return all(ok for _, ok, _ in results)
