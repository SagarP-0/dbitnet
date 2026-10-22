"""Ciphertext-pair -> bit tensor encodings.

Stored ciphertext pairs have shape (m, 2, 16) uint8 (axis 1: C0, C1).

encode_bits  -> (m, 2*8*nb) uint8, flat layout [C0 bits | C1 bits], MSB-first. This is
                bit-for-bit the layout of the original convert_to_binary (tested).
to_model_input modes:
  "flat"      (m, 2*bits)    original NDGohr input (model does Reshape((2,bits))+Permute)
  "channels2" (m, bits, 2)   channel 0 = C0 bit j, channel 1 = C1 bit j   [PRIMARY]
  "concat1"   (m, 2*bits, 1) single channel, C0 bits followed by C1 bits   [ablation]
where bits = 8 * (number of retained bytes).
"""
import numpy as np


def encode_bits(ct, byte_idx):
    ct = np.asarray(ct)
    if ct.dtype != np.uint8 or ct.ndim != 3 or ct.shape[1] != 2 or ct.shape[2] != 16:
        raise ValueError(f"expected uint8 (m,2,16), got {ct.dtype} {ct.shape}")
    idx = [int(i) for i in byte_idx]
    sel = ct[:, :, idx]                                  # (m, 2, nb)
    bits = np.unpackbits(sel, axis=2)                    # (m, 2, 8*nb), MSB first per byte
    return bits.reshape(ct.shape[0], 2 * 8 * len(idx))


def to_model_input(flat, nb, mode):
    m = flat.shape[0]
    if flat.shape[1] != 16 * nb:
        raise ValueError("flat width does not match number of bytes")
    if mode == "flat":
        return flat
    if mode == "channels2":
        return flat.reshape(m, 2, 8 * nb).transpose(0, 2, 1)
    if mode == "concat1":
        return flat[:, :, None]
    raise ValueError(f"unknown input mode {mode!r}")


def input_shape(nb, mode):
    return {"flat": (16 * nb,), "channels2": (8 * nb, 2), "concat1": (16 * nb, 1)}[mode]


def iter_batches(ct, y, byte_idx, mode, batch_size, rng=None, drop_last=False):
    """Yield (x float32, y float32 (b,1)). Bits are expanded per batch to bound memory."""
    n = ct.shape[0]
    order = rng.permutation(n) if rng is not None else np.arange(n)
    nb = len(byte_idx)
    for s in range(0, n, batch_size):
        ids = order[s: s + batch_size]
        if drop_last and len(ids) < batch_size:
            break
        x = to_model_input(encode_bits(ct[ids], byte_idx), nb, mode).astype(np.float32)
        yield x, y[ids].astype(np.float32)[:, None]
