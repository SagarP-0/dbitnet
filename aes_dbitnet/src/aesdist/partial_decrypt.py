"""Last-round partial decryption for the 3-round -> 2-round-distinguisher attack setting.

Original procedure (16bit_AES_key_recovery_attack.py::attack_with_one_nd), applied to a
3-round ciphertext produced WITHOUT final MixColumns (aes_encrypt(..., flag=False)):
    state = ct ; AddRoundKey(guess) ; InvShiftRows ; InvSubBytes        (no InvMixColumns)
The result equals the output of the 2-round cipher WITH final MixColumns
(aes_encrypt(..., nr=2, flag=True)) when the guess is the true last round key. This is
tested, not assumed (tests/test_partial_decrypt.py).

Because SubBytes/AddRoundKey act bytewise and InvShiftRows only permutes bytes, state
byte i depends on exactly ONE ciphertext byte and ONE subkey byte:
    state[i] = InvSBox[ ct[j] ^ k[j] ],   j = ct_index_for_state_byte(i)
so scoring a 2-byte distinguisher needs 2 byte-lookups per ciphertext, not a full-state
decryption.

Two candidate conventions (kept separate on purpose):
  * ABSOLUTE (original attack): enumerate guesses g in [0, 2^16): g1 = g & 0xFF ->
    subkey byte key_index[0], g2 = g >> 8 -> subkey byte key_index[1].
  * DIFFERENCE (WKRP, wkrp.py): guess = true_subkey ^ delta; requires the true key and is
    an analysis tool, not an attack.
"""
import numpy as np

from .conventions import ct_index_for_state_byte


def ct_indices(state_bytes):
    return [ct_index_for_state_byte(i) for i in state_bytes]


def last_round_subkey_bytes(aes, keys, nr):
    """Last round key bytes in flat index order (index 4*col+row), shape (n,16)."""
    rk = aes.key_expansion(keys, nr)[nr]          # (n, row, col)
    idx = np.arange(16)
    return rk[:, idx % 4, idx // 4]


def partial_decrypt_full_state(aes, ct, subkey_bytes):
    """Original ARK -> InvSR -> InvSB on the whole state. ct, subkey_bytes: (n,16)."""
    n = ct.shape[0]
    st = ct.reshape(n, 4, 4).transpose(0, 2, 1)
    sk = subkey_bytes.reshape(n, 4, 4).transpose(0, 2, 1)
    st = aes.add_round_key(st, sk)
    st = aes.inv_shift_rows(st)
    st = aes.inv_sub_bytes(st)
    return st.transpose(0, 2, 1).reshape(n, 16)


def partial_decrypt_bytes(inv_sbox, ct_sel, k_sel):
    """Bytewise partial decryption of selected ciphertext bytes (any broadcastable shapes)."""
    return inv_sbox[np.bitwise_xor(ct_sel, k_sel)]


def absolute_candidate_to_bytes(g):
    """Original packing: kg1 + (kg2 << 8)  ->  (kg1, kg2)."""
    return g & 0xFF, g >> 8
