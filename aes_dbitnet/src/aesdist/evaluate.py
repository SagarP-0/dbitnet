"""Held-out evaluation and inference benchmarking."""
import time

import numpy as np

from .bits import iter_batches, to_model_input, encode_bits
from .metrics import binary_metrics
from .models import param_counts


def predict_probs(model, ct, y, byte_idx, mode, batch=10000):
    out = [np.asarray(model(x, training=False)).reshape(-1) for x, _ in iter_batches(ct, y, byte_idx, mode, batch)]
    return np.concatenate(out)


def evaluate_split(model, ct, y, byte_idx, mode, batch=10000, threshold=0.5):
    t0 = time.perf_counter()
    p = predict_probs(model, ct, y, byte_idx, mode, batch)
    m = binary_metrics(y, p, threshold)
    m["eval_wall_s_incl_bit_encoding"] = time.perf_counter() - t0
    m.update({f"params_{k}": v for k, v in param_counts(model).items()})
    return m, p


def benchmark_inference(model, ct, byte_idx, mode, batch=5000, n_single=200, repeats=5):
    """Latency (batch=1) and throughput (batch=`batch`) of model(x, training=False) on pre-encoded tensors.

    Bit encoding is excluded. Hardware/threads are NOT controlled here: report them with the numbers.
    """
    nb = len(byte_idx)
    x_all = to_model_input(encode_bits(ct[: max(batch, n_single)], byte_idx), nb, mode).astype(np.float32)
    x1 = x_all[:1]
    for _ in range(10):
        model(x1, training=False)
    lat = []
    for i in range(n_single):
        t0 = time.perf_counter(); model(x_all[i:i + 1], training=False); lat.append(time.perf_counter() - t0)
    xb = x_all[:batch]
    model(xb, training=False)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); model(xb, training=False); ts.append(time.perf_counter() - t0)
    lat = np.array(lat) * 1e3
    return {"latency_ms_median": float(np.median(lat)), "latency_ms_p95": float(np.percentile(lat, 95)),
            "throughput_batch": int(len(xb)), "throughput_samples_per_s": float(len(xb) / np.median(ts)),
            "n_single_calls": n_single, "repeats": repeats}
