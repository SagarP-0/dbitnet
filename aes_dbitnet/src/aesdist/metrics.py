"""Binary-classification metrics computed directly from held-out predictions (numpy only)."""
import numpy as np


def binary_metrics(y_true, p, threshold=0.5):
    y = np.asarray(y_true).astype(np.int64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    pred = (p > threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    n = len(y); n1 = tp + fn; n0 = tn + fp
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    out = dict(
        n=n, threshold=threshold, tp=tp, tn=tn, fp=fp, fn=fn,
        accuracy=(tp + tn) / n,
        tpr=tp / n1 if n1 else float("nan"),
        tnr=tn / n0 if n0 else float("nan"),
        precision=tp / (tp + fp) if (tp + fp) else float("nan"),
        bce=float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
        mse=float(np.mean((y - p) ** 2)),
    )
    # statistic printed by the original evaluation notebook
    if n1 and n0:
        med = np.median(p[y == 1])
        out["pct_random_above_real_median"] = float(100 * np.mean(p[y == 0] > med))
    return out
