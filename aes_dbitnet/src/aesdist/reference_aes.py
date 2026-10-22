"""Independent, slow, pure-Python AES-128 reference (byte lists, no numpy).

Written separately from the original implementation: the S-box is GENERATED from its
mathematical definition (GF(2^8) inverse + affine map), not copied from the original
tables. It supports reduced rounds and the optional final MixColumns so that the
original aes_encrypt(..., nr, flag) can be cross-checked. It is validated against the
FIPS-197 vectors in validation.py / tests. "Independent" here means independently
written, not third-party; tests additionally compare full AES-128 to PyCryptodome.

Byte order: flat list of 16 bytes, index i = 4*col + row (same as input byte order).
"""


def _xtime(a):
    a <<= 1
    return (a ^ 0x11B) & 0xFF if a & 0x100 else a


def gmul(a, b):
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        a = _xtime(a)
        b >>= 1
    return p


def _make_sbox():
    sbox = []
    for x in range(256):
        inv = 0
        if x:
            inv = 1
            for _ in range(254):  # x^254 = x^-1 in GF(2^8)
                inv = gmul(inv, x)
        s = res = inv
        for _ in range(4):
            s = ((s << 1) | (s >> 7)) & 0xFF
            res ^= s
        sbox.append(res ^ 0x63)
    return sbox


SBOX = _make_sbox()
INV_SBOX = [0] * 256
for _i, _v in enumerate(SBOX):
    INV_SBOX[_v] = _i


def expand_key(key, nr):
    """Return list of nr+1 round keys, each a flat 16-byte list (index 4*col+row)."""
    assert len(key) == 16
    w = [list(key[4 * i: 4 * i + 4]) for i in range(4)]
    rcon = 1
    for i in range(4, 4 * (nr + 1)):
        t = list(w[i - 1])
        if i % 4 == 0:
            t = t[1:] + t[:1]
            t = [SBOX[b] for b in t]
            t[0] ^= rcon
            rcon = _xtime(rcon)
        w.append([w[i - 4][j] ^ t[j] for j in range(4)])
    return [sum(w[4 * r: 4 * r + 4], []) for r in range(nr + 1)]


def _sub(s):
    return [SBOX[b] for b in s]


def _shift_rows(s):
    out = [0] * 16
    for c in range(4):
        for r in range(4):
            out[4 * c + r] = s[4 * ((c + r) % 4) + r]
    return out


def _mix_columns(s):
    out = [0] * 16
    for c in range(4):
        a = s[4 * c: 4 * c + 4]
        out[4 * c + 0] = gmul(a[0], 2) ^ gmul(a[1], 3) ^ a[2] ^ a[3]
        out[4 * c + 1] = a[0] ^ gmul(a[1], 2) ^ gmul(a[2], 3) ^ a[3]
        out[4 * c + 2] = a[0] ^ a[1] ^ gmul(a[2], 2) ^ gmul(a[3], 3)
        out[4 * c + 3] = gmul(a[0], 3) ^ a[1] ^ a[2] ^ gmul(a[3], 2)
    return out


def _xor(a, b):
    return [x ^ y for x, y in zip(a, b)]


def encrypt(pt, key, nr=10, final_mc=False, stop_after_round=None):
    """nr-round AES: ARK0, (nr-1) x [SB SR MC ARK], then SB SR [MC if final_mc] ARK.

    stop_after_round=k (1 <= k < nr) returns the state after full round k (used for the
    FIPS-197 Appendix B intermediate check).
    """
    rk = expand_key(key, nr)
    s = _xor(list(pt), rk[0])
    for r in range(1, nr):
        s = _xor(_mix_columns(_shift_rows(_sub(s))), rk[r])
        if stop_after_round == r:
            return s
    s = _shift_rows(_sub(s))
    if final_mc:
        s = _mix_columns(s)
    return _xor(s, rk[nr])
