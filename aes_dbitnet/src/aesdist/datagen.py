"""Seeded, chunked, resumable dataset generation around the ORIGINAL AES code.

Differences from the original generate_train_data (all deliberate, all documented):
  * Randomness comes from numpy PCG64 streams derived from (seed, nr, final_mc, chunk),
    NOT os.urandom (which cannot be seeded). os.urandom is never patched.
  * Labels are EXACTLY balanced inside every chunk (original: Bernoulli(0.5) per example).
  * Negative pairs follow the original procedure: pt1 replaced by a fresh independent
    random plaintext under the SAME key; positives: pt1 = pt0 ^ (0x80 at byte 0).
  * One independent random key per example (as in the original).
  * Whole 16-byte ciphertext pairs are stored (uint8, 32 B/example); truncation to
    1/2/4/16 bytes happens at load time, so all truncations of one (rounds, final_mc)
    configuration share identical examples, labels and splits.
  * Splits are made by whole chunks (train/val/test chunk sets are disjoint); since every
    example has its own key and plaintext, no information is shared across chunks.
"""
import json
from pathlib import Path

import numpy as np

from .conventions import DIFF
from .io_utils import atomic_save_npy, atomic_write_json, cleanup_tmp, config_hash, sha256_arrays

DOMAIN_TAG = 0xAE5

MANIFEST = "manifest.json"
BYTES_PER_EXAMPLE = 33  # 32 ciphertext bytes + 1 label byte


def chunk_rng(seed, nr, final_mc, chunk_idx, stream=0):
    ss = np.random.SeedSequence([int(seed), int(nr), int(bool(final_mc)), int(chunk_idx), int(stream), DOMAIN_TAG])
    return np.random.Generator(np.random.PCG64(ss))


def generate_chunk(aes, m, nr, final_mc, seed, chunk_idx, diff=DIFF, return_aux=False):
    """Return (ct (m,2,16) uint8, y (m,) uint8[, aux]) for one chunk. Deterministic."""
    if m % 2:
        raise ValueError("chunk size must be even for exact balance")
    rng = chunk_rng(seed, nr, final_mc, chunk_idx)
    y = np.zeros(m, dtype=np.uint8)
    y[: m // 2] = 1
    rng.shuffle(y)
    pt0 = rng.integers(0, 256, (m, 16), dtype=np.uint8)
    pt1 = pt0.copy()
    pt1[:, 0] ^= np.uint8(diff)
    neg = y == 0
    pt1[neg] = rng.integers(0, 256, (int(neg.sum()), 16), dtype=np.uint8)
    keys = rng.integers(0, 256, (m, 16), dtype=np.uint8)
    ct0 = aes.aes_encrypt(pt0.copy(), keys, nr, bool(final_mc))
    ct1 = aes.aes_encrypt(pt1.copy(), keys, nr, bool(final_mc))
    ct = np.stack([ct0, ct1], axis=1).astype(np.uint8)
    if return_aux:
        return ct, y, dict(pt0=pt0, pt1=pt1, keys=keys)
    return ct, y


def validate_chunk_with_aux(aes, ct, y, aux, nr, final_mc, diff=DIFF):
    """Independent checks: decrypt with the original aes_decrypt and test the differential."""
    res = []
    d0 = aes.aes_decrypt(ct[:, 0].copy(), aux["keys"], nr, bool(final_mc))
    d1 = aes.aes_decrypt(ct[:, 1].copy(), aux["keys"], nr, bool(final_mc))
    res.append(("decrypt(ct0) == pt0", bool(np.array_equal(d0, aux["pt0"])), ""))
    res.append(("decrypt(ct1) == pt1", bool(np.array_equal(d1, aux["pt1"])), ""))
    xor = aux["pt0"] ^ aux["pt1"]
    pos = y == 1
    expect = np.zeros(16, dtype=np.uint8)
    expect[0] = diff
    res.append(("label 1: pt0^pt1 == 0x80 at byte 0, 0 elsewhere", bool(np.all(xor[pos] == expect)), ""))
    res.append(("label 0: pt0^pt1 != fixed difference (all examples)", bool(np.all(np.any(xor[~pos] != expect, axis=1))), ""))
    res.append(("class balance exact", int(pos.sum()) * 2 == len(y), f"n1={int(pos.sum())} n={len(y)}"))
    res.append(("dtypes uint8", ct.dtype == np.uint8 and y.dtype == np.uint8, ""))
    return res


def assign_splits(n_chunks, fractions=(0.8, 0.1, 0.1)):
    """Deterministic chunk->split plan. Raises if a split would be empty."""
    n_train = int(round(fractions[0] * n_chunks))
    n_val = int(round(fractions[1] * n_chunks))
    n_test = n_chunks - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(f"{n_chunks} chunks cannot be split {fractions} with non-empty splits")
    return ["train"] * n_train + ["val"] * n_val + ["test"] * n_test


def estimate_resources(n, chunk_size):
    per_chunk_peak = chunk_size * (16 * 6 + 32 * 3 + 1)  # pt0/pt1/keys + ct + transient AES states (rough upper bound)
    return {
        "disk_bytes": int(n * BYTES_PER_EXAMPLE),
        "peak_ram_generation_bytes_rough": int(per_chunk_peak * 3),
        "note": "RAM figure is a coarse upper-bound heuristic, not a measurement.",
    }


class DatasetStore:
    """On-disk chunked dataset with manifest, integrity verification and selective regeneration."""

    def __init__(self, root, nr, final_mc, n, chunk_size, seed, diff=DIFF, fractions=(0.8, 0.1, 0.1)):
        if n % chunk_size:
            raise ValueError("n must be a multiple of chunk_size")
        self.cfg = dict(nr=int(nr), final_mc=bool(final_mc), n=int(n), chunk_size=int(chunk_size),
                        seed=int(seed), diff=int(diff), fractions=list(fractions),
                        label_scheme="exact_balance_per_chunk", rng="numpy PCG64 SeedSequence",
                        negative_scheme="independent random pt1, same key (original procedure)")
        self.n_chunks = n // chunk_size
        self.splits = assign_splits(self.n_chunks, fractions)
        self.dir = Path(root) / f"ds_r{nr}_mc{int(bool(final_mc))}_s{seed}_n{n}_c{chunk_size}_{config_hash(self.cfg, 8)}"

    # --- paths / manifest
    def _paths(self, k):
        return self.dir / f"ct_{k:05d}.npy", self.dir / f"y_{k:05d}.npy"

    def _read_manifest(self):
        p = self.dir / MANIFEST
        if p.is_file():
            return json.loads(p.read_text())
        return {"config": self.cfg, "chunks": {}}

    def _write_manifest(self, man):
        atomic_write_json(self.dir / MANIFEST, man)

    # --- integrity
    def chunk_status(self, k, deep=True, man=None):
        man = man or self._read_manifest()
        rec = man["chunks"].get(str(k))
        ctp, yp = self._paths(k)
        if rec is None or not ctp.is_file() or not yp.is_file():
            return "missing"
        try:
            ct = np.load(ctp, allow_pickle=False)
            y = np.load(yp, allow_pickle=False)
        except Exception:
            return "corrupt"
        m = self.cfg["chunk_size"]
        if ct.shape != (m, 2, 16) or ct.dtype != np.uint8 or y.shape != (m,) or y.dtype != np.uint8:
            return "corrupt"
        if int(y.sum()) * 2 != m:
            return "corrupt"
        if deep and sha256_arrays([ct, y]) != rec["sha256"]:
            return "corrupt"
        return "ok"

    def verify(self, deep=True):
        man = self._read_manifest()
        return {k: s for k in range(self.n_chunks) if (s := self.chunk_status(k, deep, man)) != "ok"}

    # --- generation
    def generate(self, aes, chunks=None, log=print, stop_after=None):
        """Generate missing/corrupt chunks only; valid chunks are never rewritten."""
        self.dir.mkdir(parents=True, exist_ok=True)
        cleanup_tmp(self.dir)
        man = self._read_manifest()
        if man["config"] != self.cfg:
            raise RuntimeError("existing manifest config differs from requested config")
        todo = list(range(self.n_chunks)) if chunks is None else list(chunks)
        done_now = 0
        for k in todo:
            if self.chunk_status(k, deep=True, man=man) == "ok":
                continue
            if stop_after is not None and done_now >= stop_after:
                break
            ct, y = generate_chunk(aes, self.cfg["chunk_size"], self.cfg["nr"], self.cfg["final_mc"],
                                   self.cfg["seed"], k, self.cfg["diff"])
            ctp, yp = self._paths(k)
            atomic_save_npy(ctp, ct)
            atomic_save_npy(yp, y)
            man["chunks"][str(k)] = {"sha256": sha256_arrays([ct, y]), "split": self.splits[k], "m": int(len(y))}
            self._write_manifest(man)  # manifest updated after each chunk => resumable
            done_now += 1
            log(f"[datagen] chunk {k + 1}/{self.n_chunks} written ({self.splits[k]})")
        return done_now

    def is_complete(self):
        return not self.verify(deep=False)

    def load_split(self, split):
        bad = self.verify(deep=True)
        if bad:
            raise RuntimeError(f"dataset has missing/corrupt chunks: {bad}")
        ids = [k for k in range(self.n_chunks) if self.splits[k] == split]
        ct = np.concatenate([np.load(self._paths(k)[0]) for k in ids])
        y = np.concatenate([np.load(self._paths(k)[1]) for k in ids])
        return ct, y

    def write_metadata(self, extra=None):
        meta = dict(self.cfg)
        meta.update(chunks=self.n_chunks, split_plan=self.splits, resources=estimate_resources(self.cfg["n"], self.cfg["chunk_size"]))
        if extra:
            meta.update(extra)
        atomic_write_json(self.dir / "dataset_metadata.json", meta)
