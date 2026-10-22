"""Operation-count bookkeeping for the key-recovery complexity analysis (skeleton).

Deliberately does NOT assert 2^21.585. The original notebook cell computes
log2(2^16 * 3 * 2 * 8) = 21.585; the meaning of each factor is an UNVERIFIED reading:
  3 = ciphertext pairs per trial (n=3 in attack_with_dual_NDs call), 2 = ciphertexts per pair,
  8 = number of 2-byte blocks / distinguishers.  Whether this counts partial decryptions,
  model evaluations, or something else, and whether 3 pairs achieve any stated success
  probability, must come from an instrumented, repeated attack run (later stage).
"""
import math


def op_counts(n_pairs, n_blocks=1, candidates=1 << 16, structures=1):
    model_evals = candidates * n_pairs * n_blocks * structures
    return {
        "candidates_per_block": candidates,
        "blocks": n_blocks,
        "pairs_per_structure": n_pairs,
        "structures": structures,
        "chosen_plaintexts_per_trial": 2 * n_pairs * structures,
        "model_evals": model_evals,
        "byte_pair_partial_decryptions": 2 * model_evals,   # one per ciphertext
        "log2_model_evals": math.log2(model_evals),
        "log2_ct_partial_decryptions": math.log2(2 * model_evals),
    }
