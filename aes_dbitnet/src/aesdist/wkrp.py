"""Difference-based Wrong Key Response Profile (WKRP), chunked and resumable.

Setting: 3-round AES (no final MC), plaintext difference 0x80 on byte 0, 2-round 2-byte
distinguisher on state bytes `state_bytes` (default [0,1] -> ciphertext/subkey bytes
[0,13]). For candidate difference delta=(d_a,d_b) the examined subkey guess of example i is
    guess_i = true_subkey_i ^ delta          (delta = 0  <=>  the correct key)
This is an ANALYSIS of the distinguisher, not the original absolute-subkey attack
(which enumerates absolute values for one fixed key; see partial_decrypt.py).

delta_id packing mirrors the original `kg1 + (kg2 << 8)`: delta_id = d_a | (d_b << 8)
(a "packing" only; the semantics are differences).

Per delta we report population statistics (ddof=0) over the N examples:
    mean_prob, std_prob              of the sigmoid score
    mean_llr,  std_llr               of log2(p/(1-p)) with p clipped to [EPS, 1-EPS]
Counters: model_evals = D*N ; partial decryptions (byte-pair lookups) = 2*D*N ;
chosen plaintexts = 2*N (shared by all candidates).
"""
import csv
import json
from pathlib import Path

import numpy as np

from .io_utils import atomic_write, atomic_write_json, cleanup_tmp, sha256_arrays
from .partial_decrypt import ct_indices, partial_decrypt_bytes

EPS = 1e-7
COLUMNS = ["delta_id", "d_a", "d_b", "n", "mean_prob", "std_prob", "mean_llr", "std_llr", "is_zero"]


def wkrp_inputs(inv_sbox, ct_sel, true_sub_sel, deltas, input_mode="channels2"):
    """Build distinguisher inputs for D deltas x N examples (delta-major).

    ct_sel: (N, 2, nbytes) selected ciphertext bytes (axis1 = C0,C1)
    true_sub_sel: (N, nbytes) true subkey bytes at the same positions
    deltas: (D, nbytes) uint8
    Returns array (D*N, bits, 2) for channels2 or (D*N, 2*bits, 1) for concat1.
    """
    D, N, nb = deltas.shape[0], ct_sel.shape[0], ct_sel.shape[2]
    guess = true_sub_sel[None, :, :] ^ deltas[:, None, :]                  # (D,N,nb)
    st = partial_decrypt_bytes(inv_sbox, ct_sel[None, :, :, :], guess[:, :, None, :])  # (D,N,2,nb)
    st = np.ascontiguousarray(st.reshape(D * N, 2, nb).astype(np.uint8))
    bits = np.unpackbits(st, axis=2)                                        # (DN,2,8nb)
    if input_mode == "channels2":
        return bits.transpose(0, 2, 1).astype(np.float32)
    if input_mode == "concat1":
        return bits.reshape(D * N, 16 * nb)[:, :, None].astype(np.float32)
    raise ValueError(input_mode)


def _stats(p, N):
    p = np.asarray(p, dtype=np.float64).reshape(-1, N)
    pc = np.clip(p, EPS, 1 - EPS)
    llr = np.log2(pc / (1 - pc))
    return p.mean(1), p.std(1, ddof=0), llr.mean(1), llr.std(1, ddof=0)


def run_wkrp(score_fn, inv_sbox, ct_pairs, true_last_subkey, out_dir, state_bytes=(0, 1),
             delta_chunk=1024, max_rows=200_000, input_mode="channels2", n_deltas=1 << 16,
             stop_after_chunks=None, log=print, tag=""):
    """Evaluate all deltas in resumable chunks. Returns manifest dict.

    score_fn(X float32) -> probabilities, shape (rows,) or (rows,1).
    ct_pairs: (N,2,16) 3-round ciphertext pairs (real pairs); true_last_subkey: (N,16).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cleanup_tmp(out_dir)
    cidx = ct_indices(state_bytes)
    nb = len(cidx)
    if nb != 2:
        raise ValueError("this WKRP enumerates 2^16 deltas: exactly two bytes")
    ct_sel = np.ascontiguousarray(ct_pairs[:, :, cidx])
    ts = np.ascontiguousarray(true_last_subkey[:, cidx])
    N = ct_sel.shape[0]
    cfg = dict(N=N, state_bytes=list(state_bytes), ct_indices=cidx, delta_chunk=delta_chunk, n_deltas=n_deltas,
               input_mode=input_mode, data_sha256=sha256_arrays([ct_pairs, true_last_subkey]),
               std_convention="population (ddof=0)", llr_clip=EPS, tag=tag)
    mpath = out_dir / "wkrp_manifest.json"
    if mpath.is_file():
        man = json.loads(mpath.read_text())
        if man["config"] != cfg:
            raise RuntimeError("WKRP manifest config differs (different data/model/settings); use a new out_dir")
    else:
        man = {"config": cfg, "completed": [], "counters": {"model_evals": 0, "byte_pair_partial_decryptions": 0}}
    n_chunks = (n_deltas + delta_chunk - 1) // delta_chunk
    done = set(man["completed"])
    ran = 0
    for k in range(n_chunks):
        cpath = out_dir / f"chunk_{k:05d}.csv"
        if k in done and cpath.is_file():
            continue
        if stop_after_chunks is not None and ran >= stop_after_chunks:
            break
        ids = np.arange(k * delta_chunk, min((k + 1) * delta_chunk, n_deltas), dtype=np.int64)
        rows = []
        sub = max(1, max_rows // N)
        for s in range(0, len(ids), sub):
            di = ids[s: s + sub]
            deltas = np.stack([di & 0xFF, di >> 8], axis=1).astype(np.uint8)
            X = wkrp_inputs(inv_sbox, ct_sel, ts, deltas, input_mode)
            p = np.asarray(score_fn(X)).reshape(-1)
            if p.shape[0] != X.shape[0]:
                raise RuntimeError("score_fn returned wrong number of scores")
            mp, sp, ml, sl = _stats(p, N)
            for j, d in enumerate(di):
                rows.append([int(d), int(d & 0xFF), int(d >> 8), N, mp[j], sp[j], ml[j], sl[j], int(d == 0)])
        def w(f, rows=rows):
            wr = csv.writer(f)
            wr.writerow(COLUMNS)
            wr.writerows(rows)
        atomic_write(cpath, w, mode="w")
        man["completed"] = sorted(done | {k})
        done.add(k)
        man["counters"]["model_evals"] += len(ids) * N
        man["counters"]["byte_pair_partial_decryptions"] += 2 * len(ids) * N
        atomic_write_json(mpath, man)          # manifest after chunk => interrupted runs resume
        ran += 1
        log(f"[wkrp] chunk {k + 1}/{n_chunks} done")
    man["complete"] = len(done) == n_chunks
    return man


def merge_chunks(out_dir, n_chunks, dest="wkrp_results.csv"):
    out_dir = Path(out_dir)
    rows = []
    for k in range(n_chunks):
        with open(out_dir / f"chunk_{k:05d}.csv") as f:
            r = list(csv.reader(f))
        rows += r[1:]
    def w(f):
        wr = csv.writer(f)
        wr.writerow(COLUMNS)
        wr.writerows(rows)
    atomic_write(out_dir / dest, w, mode="w")
    return out_dir / dest


def summarize(csv_path):
    """Descriptive WKRP summary. Treats 'correct key has highest mean' as a hypothesis to test."""
    a = np.genfromtxt(csv_path, delimiter=",", names=True)
    mean = a["mean_prob"]
    z = a["is_zero"] == 1
    wrong = mean[~z]
    order = np.argsort(-mean)
    rank0 = int(np.where(a["delta_id"][order] == 0)[0][0]) + 1 if z.any() else None
    return dict(
        n_candidates=int(len(a)), n_examples=int(a["n"][0]),
        correct_mean=float(mean[z][0]) if z.any() else None,
        correct_std=float(a["std_prob"][z][0]) if z.any() else None,
        wrong_mean_of_means=float(wrong.mean()), wrong_std_of_means=float(wrong.std(ddof=0)),
        wrong_max_mean=float(wrong.max()),
        peak_to_baseline=float(mean[z][0] - wrong.mean()) if z.any() else None,
        contrast_in_wrong_sigma=float((mean[z][0] - wrong.mean()) / wrong.std(ddof=0)) if z.any() and wrong.std() > 0 else None,
        rank_of_correct_by_mean_prob=rank0,
        note="descriptive; z-like contrast is not a calibrated significance level",
    )
