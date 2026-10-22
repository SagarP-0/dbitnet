"""Documented conventions, derived from AES_128_batch.py (see docs/BASELINE_INSPECTION.md).

Byte index convention (aes_encrypt: reshape(n,4,4).transpose(0,2,1) and inverse):
    flat byte index i  <->  state row = i % 4, column = i // 4      (FIPS-197 column-major)
Bit convention (convert_to_binary): MSB first within a byte;
    flat layout = [C0 selected bytes (MSB..LSB each) | C1 selected bytes].
"""
DIFF = 0x80          # plaintext difference value
DIFF_BYTE = 0        # byte 0 (row 0, col 0); pt1[:, 0] ^= 0x80 in generate_train_data
NUM_BYTES = 16

# Named truncations. Source status is stated explicitly; nothing here is "assumed equal".
TRUNCATION_PRESETS = {
    "full":   {"bytes": list(range(16)), "source": "commented-out `ct = hstack(ct0, ct1)` branch of generate_train_data"},
    "col0":   {"bytes": [0, 1, 2, 3],    "source": "commented-out 'one column' branch: ct[:, i*4:(i+1)*4], i=0"},
    "row0":   {"bytes": [0, 4, 8, 12],   "source": "commented-out 'one row' branch: range(16)[i:i+13:4], i=0"},
    "2B_r2":  {"bytes": [0, 1],          "source": "commented-out 2-byte branch ct[:, i*2:(i+1)*2], i=0; attack file pairs 2-round model i with bytes [2i,2i+1]"},
    "2B_r3":  {"bytes": [0, 13],         "source": "active branch + main_net.__main__ + test_something.ipynb use data_index=[0,13]"},
    "1B":     {"bytes": [0],             "source": "commented-out 1-byte branch (index is a free choice 0..15; 0 is a CONFIG CHOICE)"},
}


def resolve_truncation(spec):
    """spec: preset name or explicit list of byte indices. Returns a validated list."""
    if isinstance(spec, str):
        if spec not in TRUNCATION_PRESETS:
            raise ValueError(f"unknown truncation preset {spec!r}")
        idx = list(TRUNCATION_PRESETS[spec]["bytes"])
    else:
        idx = [int(i) for i in spec]
    if not idx or len(set(idx)) != len(idx) or any(i < 0 or i >= NUM_BYTES for i in idx):
        raise ValueError(f"invalid byte index list {idx}")
    return idx


def state_pos(i):
    """(row, col) of flat byte index i."""
    return i % 4, i // 4


def flat_index(row, col):
    return 4 * col + row


def ct_index_for_state_byte(i):
    """Ciphertext/last-subkey byte index that feeds state byte i after InvShiftRows.

    Forward ShiftRows: ct(r, c) = state(r, (c + r) % 4)  =>  state(r, c') = ct(r, (c' - r) % 4).
    Reproduces the attack file's key_index table, e.g. state bytes [0,1] -> [0,13].
    """
    r, c = state_pos(i)
    return flat_index(r, (c - r) % 4)
