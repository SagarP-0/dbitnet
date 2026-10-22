from aesdist.conventions import (ct_index_for_state_byte, resolve_truncation, state_pos, flat_index)
import pytest

# Table copied from 16bit_AES_key_recovery_attack.py (file_and_key_index): model i <-> key_index
ATTACK_TABLE = {0: [0, 13], 1: [10, 7], 2: [4, 1], 3: [14, 11], 4: [8, 5], 5: [2, 15], 6: [12, 9], 7: [6, 3]}


def test_flat_index_roundtrip():
    for i in range(16):
        assert flat_index(*state_pos(i)) == i


@pytest.mark.parametrize("i", range(8))
def test_derived_key_indices_match_original_attack_table(i):
    assert [ct_index_for_state_byte(2 * i), ct_index_for_state_byte(2 * i + 1)] == ATTACK_TABLE[i]


def test_inv_shiftrows_mapping_is_a_permutation():
    assert sorted(ct_index_for_state_byte(i) for i in range(16)) == list(range(16))


def test_truncation_validation():
    assert resolve_truncation("2B_r3") == [0, 13]
    assert resolve_truncation([3, 1]) == [3, 1]
    for bad in ([], [0, 0], [16], [-1], "nope"):
        with pytest.raises(ValueError):
            resolve_truncation(bad)
