"""NDGohr (faithful re-implementation of main_net.make_resnet) and configurable DBitNet.

Both are built with tf.keras (TF 2.15 / Keras 2 API). TensorFlow is imported lazily.

NDGohr (main_net.py::make_resnet, called with depth=5, reg_param=1e-5):
  [Reshape((2,bits)) + Permute((2,1)) when input_mode='flat']
  Conv1D(16,k=1) -> BN -> ReLU -> 5 x { Conv1D(16,k=3)-BN-ReLU-Conv1D(16,k=3)-BN-ReLU, Add(shortcut) }
  -> Flatten -> Dense64-BN-ReLU -> Dense64-BN-ReLU -> Dense(1,sigmoid); L2(1e-5) on all kernels.
  No dilation. input_mode='channels2' feeds the already-permuted (bits,2) tensor: the
  Reshape+Permute pair has no weights, so the two variants are numerically identical (tested).

DBitNet (task spec, configurable): stacked Conv1D(filters, kernel, dilation_rate=d, padding='same')
  + BN + ReLU (optional residual), then GAP or Flatten, Dense(64)-BN-ReLU [x2], Dense(1,sigmoid).
  Primary input: (bits, 2) with C0/C1 in separate channels. Ablation: (2*bits, 1) concatenated.
"""
import numpy as np

from .bits import input_shape


def receptive_field(kernel, dilations):
    """Theoretical RF of stacked stride-1 dilated convs: 1 + (k-1) * sum(d)."""
    return 1 + (kernel - 1) * int(sum(dilations))


def receptive_field_report(kernel, dilations, seq_len):
    rf = receptive_field(kernel, dilations)
    return {"receptive_field": rf, "sequence_length": seq_len, "rf_covers_sequence": rf >= seq_len,
            "layers_with_dilation_ge_seq_len": [d for d in dilations if d >= seq_len]}


def _keras():
    import tensorflow as tf
    return tf.keras


def build_ndgohr(bits, input_mode="channels2", num_filters=16, ks=3, depth=5, d1=64, d2=64, reg=1e-5):
    k = _keras(); L = k.layers; l2 = k.regularizers.l2
    if input_mode == "flat":
        inp = L.Input(shape=(2 * bits,))
        x = L.Permute((2, 1))(L.Reshape((2, bits))(inp))
    elif input_mode == "channels2":
        inp = L.Input(shape=(bits, 2)); x = inp
    else:
        raise ValueError("NDGohr supports input_mode 'flat' or 'channels2'")
    x = L.Conv1D(num_filters, kernel_size=1, padding="same", kernel_regularizer=l2(reg))(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    shortcut = x
    for _ in range(depth):
        c = L.Conv1D(num_filters, kernel_size=ks, padding="same", kernel_regularizer=l2(reg))(shortcut)
        c = L.Activation("relu")(L.BatchNormalization()(c))
        c = L.Conv1D(num_filters, kernel_size=ks, padding="same", kernel_regularizer=l2(reg))(c)
        c = L.Activation("relu")(L.BatchNormalization()(c))
        shortcut = L.Add()([shortcut, c])
    x = L.Flatten()(shortcut)
    x = L.Activation("relu")(L.BatchNormalization()(L.Dense(d1, kernel_regularizer=l2(reg))(x)))
    x = L.Activation("relu")(L.BatchNormalization()(L.Dense(d2, kernel_regularizer=l2(reg))(x)))
    out = L.Dense(1, activation="sigmoid", kernel_regularizer=l2(reg))(x)
    return k.Model(inp, out, name="ndgohr")


def build_dbitnet(bits, input_mode="channels2", n_layers=5, kernel=3, dilations=None, filters=32,
                  residual=False, pool="gap", dense_units=64, second_dense=False, reg=0.0):
    k = _keras(); L = k.layers
    dilations = list(dilations) if dilations is not None else [1, 2, 4, 8, 16][:n_layers]
    if len(dilations) != n_layers:
        raise ValueError(f"need {n_layers} dilation rates, got {dilations}")
    if input_mode not in ("channels2", "concat1"):
        raise ValueError("DBitNet supports input_mode 'channels2' or 'concat1'")
    shp = input_shape(bits // 8, input_mode)
    inp = L.Input(shape=shp)
    x = inp
    kr = k.regularizers.l2(reg) if reg else None
    for i, d in enumerate(dilations):
        h = L.Conv1D(filters, kernel, dilation_rate=d, padding="same", use_bias=False, kernel_regularizer=kr)(x)
        h = L.Activation("relu")(L.BatchNormalization()(h))
        x = L.Add()([x, h]) if (residual and i > 0) else h   # first layer changes channel count
    if pool == "gap":
        x = L.GlobalAveragePooling1D()(x)
    elif pool == "flatten":
        x = L.Flatten()(x)
    else:
        raise ValueError(pool)
    for _ in range(2 if second_dense else 1):
        x = L.Activation("relu")(L.BatchNormalization()(L.Dense(dense_units, kernel_regularizer=kr)(x)))
    out = L.Dense(1, activation="sigmoid", kernel_regularizer=kr)(x)
    return k.Model(inp, out, name="dbitnet")


def build_model(cfg):
    """cfg: {'arch': 'ndgohr'|'dbitnet', 'bits': int, 'input_mode': ..., **arch kwargs}"""
    cfg = dict(cfg)
    arch = cfg.pop("arch")
    if arch == "ndgohr":
        return build_ndgohr(**cfg)
    if arch == "dbitnet":
        return build_dbitnet(**cfg)
    raise ValueError(arch)


def param_counts(model):
    tr = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    nt = int(sum(int(np.prod(w.shape)) for w in model.non_trainable_weights))
    return {"total": tr + nt, "trainable": tr, "non_trainable": nt}


def validate_model(model, loss="bce", batch=8, seed=0):
    """Architecture validation: shapes, finite weights, finite loss/grads on synthetic data, memory estimate.

    The synthetic step runs on a CLONE so BatchNorm statistics of `model` are untouched.
    """
    import tensorflow as tf
    checks = []
    shape = tuple(model.input_shape[1:])
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, (batch,) + shape).astype(np.float32)
    y = rng.integers(0, 2, (batch, 1)).astype(np.float32)
    out = model(x, training=False)
    checks.append(("output shape == (batch,1)", tuple(out.shape) == (batch, 1), str(tuple(out.shape))))
    checks.append(("parameters finite", all(np.isfinite(w.numpy()).all() for w in model.weights), ""))
    clone = tf.keras.models.clone_model(model)
    lf = tf.keras.losses.BinaryCrossentropy() if loss == "bce" else tf.keras.losses.MeanSquaredError()
    with tf.GradientTape() as tape:
        p = clone(x, training=True)
        l = lf(y, p)
        if clone.losses:
            l = l + tf.add_n(clone.losses)
    grads = tape.gradient(l, clone.trainable_variables)
    checks.append(("loss finite", bool(np.isfinite(float(l))), f"{float(l):.4f}"))
    checks.append(("gradients exist and are finite", all(g is not None and np.isfinite(g.numpy()).all() for g in grads), ""))
    pc = param_counts(model)
    act = 0
    for layer in model.layers:
        try:
            shp = layer.output_shape
            if isinstance(shp, list):
                shp = shp[0]
            act += int(np.prod([s for s in shp[1:] if s]))
        except Exception:
            pass
    mem = {"param_bytes_fp32": pc["total"] * 4,
           "adam_training_state_bytes_fp32_approx": pc["trainable"] * 4 * 4,   # weights+grads+2 Adam slots
           "activation_floats_per_sample_forward": act,
           "activation_bytes_per_sample_fp32_approx": act * 4,
           "note": "coarse estimate: sum of layer output sizes; training keeps activations for backprop"}
    return {"checks": checks, "params": pc, "memory": mem, "input_shape": shape}
