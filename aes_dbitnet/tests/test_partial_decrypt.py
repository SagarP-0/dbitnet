import numpy as np
import pytest

from aesdist.conventions import ct_index_for_state_byte
from aesdist.partial_decrypt import (absolute_candidate_to_bytes, last_round_subkey_bytes,
                                     partial_decrypt_bytes, partial_decrypt_full_state, ct_indices)


@pytest.fixture()
def data(aes):
    rng = np.random.default_rng(11)
    n = 200
    pt0 = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    pt1 = pt0.copy(); pt1[:, 0] ^= 0x80
    keys = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    return pt0, pt1, keys


def test_true_key_partial_decrypt_equals_2round_with_final_mc(aes, data):
    pt0, pt1, keys = data
    ct3 = aes.aes_encrypt(pt0.copy(), keys, 3, False)
    sk = last_round_subkey_bytes(aes, keys, 3)
    got = partial_decrypt_full_state(aes, ct3, sk)
    exp = aes.aes_encrypt(pt0.copy(), keys, 2, True)
    assert np.array_equal(got, exp)


def test_last_round_subkey_bytes_consistent_with_ciphertext(aes, data):
    pt0, _, keys = data
    ct3 = aes.aes_encrypt(pt0.copy(), keys, 3, False)
    sk = last_round_subkey_bytes(aes, keys, 3)
    # Independent route: ct ^ sk must equal ShiftRows(SubBytes(round-2 output)) viewed bytewise,
    # i.e. InvSubBytes of the result at index ct_index(i) equals the 2-round state byte i.
    st2 = aes.aes_encrypt(pt0.copy(), keys, 2, True)
    for i in range(16):
        j = ct_index_for_state_byte(i)
        assert np.array_equal(aes.inv_s_box[ct3[:, j] ^ sk[:, j]], st2[:, i])


@pytest.mark.parametrize("blk", range(8))
def test_bytewise_shortcut_equals_full_state_for_all_8_blocks(aes, data, blk):
    pt0, _, keys = data
    ct3 = aes.aes_encrypt(pt0.copy(), keys, 3, False)
    sk = last_round_subkey_bytes(aes, keys, 3)
    full = partial_decrypt_full_state(aes, ct3, sk)
    sb = [2 * blk, 2 * blk + 1]
    cidx = ct_indices(sb)
    short = partial_decrypt_bytes(aes.inv_s_box, ct3[:, cidx], sk[:, cidx])
    assert np.array_equal(short, full[:, sb])


def test_original_absolute_candidate_indexing_reproduced(aes, data):
    """Re-implements the ORIGINAL attack's guess_key placement (without importing that script)."""
    pt0, _, keys = data
    ct3 = aes.aes_encrypt(pt0.copy(), keys[:1].repeat(len(pt0), 0), 3, False)   # one fixed key, many pts
    key1 = keys[:1]
    last = aes.key_expansion(key1, 3)[-1].copy()           # (1,4,4) row,col  == ks[-1]
    key_index = [0, 13]; data_index = 0; byte_num = 2
    for g in (0, 1, 0x1234, 0xFFFF, 0xA55A):
        kg1, kg2 = absolute_candidate_to_bytes(g)
        gk = last.copy()
        gk[0, key_index[0] % 4, key_index[0] // 4] = kg1
        gk[0, key_index[1] % 4, key_index[1] // 4] = kg2
        n = ct3.shape[0]
        s = ct3.reshape(n, 4, 4).transpose((0, 2, 1))
        s = aes.add_round_key(s, np.tile(gk, (n, 1, 1)))
        s = aes.inv_shift_rows(s)
        s = aes.inv_sub_bytes(s)
        s = s.transpose((0, 2, 1)).reshape(-1, 16)[:, data_index * byte_num:(data_index + 1) * byte_num]
        # ours
        k_sel = np.tile(np.array([kg1, kg2], dtype=np.uint8), (n, 1))
        ours = partial_decrypt_bytes(aes.inv_s_box, ct3[:, key_index], k_sel)
        assert np.array_equal(s, ours)
    # sanity: the true bytes reproduce the 2-round(MC) state; a wrong guess does not
    true = last_round_subkey_bytes(aes, key1, 3)[0, key_index]
    st2 = aes.aes_encrypt(pt0.copy(), key1.repeat(len(pt0), 0), 2, True)[:, :2]
    good = partial_decrypt_bytes(aes.inv_s_box, ct3[:, key_index], true[None, :])
    bad = partial_decrypt_bytes(aes.inv_s_box, ct3[:, key_index], (true ^ np.array([1, 0], dtype=np.uint8))[None, :])
    assert np.array_equal(good, st2) and not np.array_equal(bad, st2)
