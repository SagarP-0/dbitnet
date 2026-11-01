import argparse
import json
import os
import subprocess
import sys
import time
from os import urandom
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import (Activation, Add, BatchNormalization, Conv1D,
                                     Dense, Flatten, Input, Permute, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

print("TensorFlow Version:", tf.__version__)

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=1, help="First difference value (0-255)")
parser.add_argument("--end", type=int, default=255, help="Last difference value, inclusive (0-255)")
parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
args = parser.parse_args()
args.output_dir = os.path.abspath(args.output_dir)

# --- Resolve repo path relative to this script, then enter it ---
REPO_URL = "https://github.com/CryptAnalystDesigner/differential-neural-cryptanalysis-on-aes.git"
REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream_repo")

if not os.path.exists(os.path.join(REPO_DIR, "AES_128_batch.py")):
    raise FileNotFoundError(
        f"AES_128_batch.py not found in repository: {REPO_DIR}"
    )

os.chdir(REPO_DIR)
sys.path.insert(0, os.getcwd())

assert os.path.exists('AES_128_batch.py'), "AES_128_batch.py not found in current directory."
print("Repository loaded successfully.")

from AES_128_batch import aes_encrypt, convert_to_binary

NR = 3
BYTE_POS = 0
BYTE_INDICES = list(range(16))
USE_CT_DIFF = True

N_TRAIN = 2 * 10**6
N_VAL   = 10**6
N_CALIB = 2 * 10**5
N_TEST  = 10**6
CHUNK   = 10**6
EPOCHS  = 20
BATCH   = 5000
DEPTH   = 5

os.makedirs(args.output_dir, exist_ok=True)

def gen_chunk(n, byte_idx, use_ct_diff, pt_diff):
    Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
    pt0 = np.frombuffer(urandom(16 * n), dtype=np.uint8).reshape(-1, 16)
    pt1 = pt0.copy()
    for b, v in pt_diff.items():
        pt1[:, b] ^= v
    k0 = int(np.sum(Y == 0))
    pt1[Y == 0] = np.frombuffer(urandom(16 * k0), dtype=np.uint8).reshape(-1, 16)
    keys = np.frombuffer(urandom(16 * n), dtype=np.uint8).reshape(-1, 16)
    ct0 = aes_encrypt(pt0, keys, NR, False)
    ct1 = aes_encrypt(pt1, keys, NR, False)
    ct = np.hstack((ct0[:, byte_idx], ct1[:, byte_idx]))
    X = convert_to_binary(ct, len(byte_idx))
    if use_ct_diff:
        half = X.shape[1] // 2
        X = np.hstack((X, X[:, :half] ^ X[:, half:]))
    return X, Y

def gen_data(n, byte_idx, use_ct_diff, pt_diff):
    Xs, Ys = [], []
    for i in range(0, n, CHUNK):
        X, Y = gen_chunk(min(CHUNK, n - i), byte_idx, use_ct_diff, pt_diff)
        Xs.append(X); Ys.append(Y)
    return np.concatenate(Xs), np.concatenate(Ys)

def make_resnet(word_size, num_blocks, num_filters=16, d1=64, d2=64, ks=3, depth=5, reg_param=1e-5):
    inp = Input(shape=(num_blocks * word_size,))
    x = Permute((2, 1))(Reshape((num_blocks, word_size))(inp))
    x = Conv1D(num_filters, 1, padding='same', kernel_regularizer=l2(reg_param))(x)
    x = Activation('relu')(BatchNormalization()(x))
    shortcut = x
    for _ in range(depth):
        c = Conv1D(num_filters, ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
        c = Activation('relu')(BatchNormalization()(c))
        c = Conv1D(num_filters, ks, padding='same', kernel_regularizer=l2(reg_param))(c)
        c = Activation('relu')(BatchNormalization()(c))
        shortcut = Add()([shortcut, c])
    x = Flatten()(shortcut)
    x = Activation('relu')(BatchNormalization()(Dense(d1, kernel_regularizer=l2(reg_param))(x)))
    x = Activation('relu')(BatchNormalization()(Dense(d2, kernel_regularizer=l2(reg_param))(x)))
    out = Dense(1, activation='sigmoid', kernel_regularizer=l2(reg_param))(x)
    return Model(inp, out)

def cyclic_lr(num_epochs, high_lr, low_lr):
    return lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)

def metrics(y_true, p, thr):
    y_pred = (p.flatten() > thr).astype(np.uint8)
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    return {"thr": float(thr), "Acc": (TP + TN) / len(y_true),
            "TPR": TP / max(TP + FN, 1), "TNR": TN / max(TN + FP, 1)}

def best_threshold(y, p):
    p = p.flatten()
    cand = np.quantile(p, np.linspace(0.01, 0.99, 197))
    accs = [np.mean((p > t).astype(np.uint8) == y) for t in cand]
    return cand[int(np.argmax(accs))]

def run_one(diff_int):
    diff_tag = f"b0_0x{diff_int:02X}"
    pt_diff = {BYTE_POS: diff_int}
    out_file = os.path.join(args.output_dir, f"{diff_tag}.json")

    # Skip if already completed (built-in fault tolerance / resumable)
    if os.path.exists(out_file):
        print(f"Result for {diff_tag} already exists. Skipping.", flush=True)
        return

    t0 = time.time()
    word_size = 8 * len(BYTE_INDICES)
    num_blocks = 3 if USE_CT_DIFF else 2

    X, Y = gen_data(N_TRAIN, BYTE_INDICES, USE_CT_DIFF, pt_diff)
    Xv, Yv = gen_data(N_VAL, BYTE_INDICES, USE_CT_DIFF, pt_diff)

    net = make_resnet(word_size=word_size, num_blocks=num_blocks, depth=DEPTH)
    net.compile(optimizer='adam', loss='mse', metrics=['acc'])

    ckpt = f"/tmp/ckpt_{diff_tag}_{os.getpid()}.weights.h5"
    net.fit(X, Y, epochs=EPOCHS, batch_size=BATCH, validation_data=(Xv, Yv),
            callbacks=[LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001)),
                       ModelCheckpoint(ckpt, monitor='val_loss', save_best_only=True, save_weights_only=True)],
            verbose=0)

    net.load_weights(ckpt)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    del X, Y, Xv, Yv

    Xc, Yc = gen_data(N_CALIB, BYTE_INDICES, USE_CT_DIFF, pt_diff)
    thr_opt = best_threshold(Yc, net.predict(Xc, batch_size=10000, verbose=0))
    del Xc, Yc

    Xt, Yt = gen_data(N_TEST, BYTE_INDICES, USE_CT_DIFF, pt_diff)
    pt = net.predict(Xt, batch_size=10000, verbose=0)

    res = {
        "difference": diff_tag,
        "thr_0.5": metrics(Yt, pt, 0.5),
        "thr_calibrated": metrics(Yt, pt, thr_opt)
    }

    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)

    # Free memory/graph state before the next difference
    del net, Xt, Yt, pt
    tf.keras.backend.clear_session()

    print(f"Finished {diff_tag} in {time.time() - t0:.0f}s. "
          f"Acc(0.5)={res['thr_0.5']['Acc']:.4f} Acc(cal)={res['thr_calibrated']['Acc']:.4f}",
          flush=True)

# --- Loop over all differences ---
for d in range(args.start, args.end + 1):
    try:
        run_one(d)
    except Exception as e:
        print(f"FAILED 0x{d:02X}: {e!r}", flush=True)
        tf.keras.backend.clear_session()

# --- Combine all per-difference JSONs into one summary file ---
summary = []
for d in range(256):
    p = os.path.join(args.output_dir, f"b0_0x{d:02X}.json")
    if os.path.exists(p):
        with open(p) as f:
            summary.append(json.load(f))
with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"All done. {len(summary)} results combined in {args.output_dir}/all_results.json")