import subprocess
import sys
from pathlib import Path

import pytest

from aesdist import complexity
from aesdist.pretrained import parse_filename

ROOT = Path(__file__).resolve().parents[1]

# Filenames exactly as they appear in the uploaded main_net/test_something/attack files (not the .h5 files themselves)
NAMES = [
    ("AES16_data_index_[0, 13]_best_model_3r_depth5_num_epochs20_acc_0.5042799711227417.h5", 2, [0, 13], 3),
    ("AES128_data_index_None_best_model_3r_depth5_num_epochs20_acc_0.5303980112075806.h5", 16, None, 3),
    ("AES128_data_index_None_best_model_2r_depth5_num_epochs20_acc_1.0.h5", 16, None, 2),
    ("AES16_data_index_0_best_model_2r_depth5_num_epochs20_acc_0.9980679750442505.h5", 2, 0, 2),
    ("AES16_data_index_7_best_model_2r_depth5_num_epochs20_acc_0.9980049729347229.h5", 2, 7, 2),
]


@pytest.mark.parametrize("name,nbytes,idx,nr", NAMES)
def test_parse_pretrained_filenames(name, nbytes, idx, nr):
    p = parse_filename("./3_round_good_trained_nets/" + name)
    assert p["bytes_per_ciphertext"] == nbytes and p["data_index"] == idx and p["rounds"] == nr
    assert 0.0 <= p["acc_in_filename"] <= 1.0


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_filename("model_final.h5")


def test_op_counts_do_not_assume_2_pow_21_585():
    c = complexity.op_counts(n_pairs=3, n_blocks=8)
    assert c["model_evals"] == 2 ** 16 * 3 * 8 and c["chosen_plaintexts_per_trial"] == 6
    assert abs(c["log2_ct_partial_decryptions"] - 21.585) < 1e-3   # = the notebook's number, under one reading


def test_full_mode_requires_explicit_flag():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "train", "--mode", "full",
                        "--experiment", "r3_2B"], capture_output=True, text=True)
    assert r.returncode == 2 and "--full-experiment" in r.stderr


def test_gpu_mode_refuses_outside_slurm_and_without_gpu(monkeypatch):
    pytest.importorskip("tensorflow")
    from aesdist.envinfo import configure_device
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="SLURM"):
        configure_device("gpu")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(RuntimeError, match="no GPU|sees no GPU"):
        configure_device("gpu")        # sandbox/CPU-only: must refuse, never fall back silently
