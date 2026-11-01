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

# --- Parse args first (resolve output dir to an absolute path before chdir) ---
parser = argparse.ArgumentParser()
parser.add_argument("--diff_val", type=int, required=True, help="Difference value (0-255)")
parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
args = parser.parse_args()
args.output_dir = os.path.abspath(args.output_dir)

# --- Clone the repository (skipped if already present) and enter it ---
REPO_URL = "https://github.com/CryptAnalystDesigner/differential-neural-cryptanalysis-on-aes.git"
#REPO_DIR = "differential-neural-cryptanalysis-on-aes"
REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream_repo")

#if not os.path.exists(os.path.join(REPO_DIR, "AES_128_batch.py")):
#    subprocess.run(["git", "clone", "-q", REPO_URL, REPO_DIR], check=True)
#if not os.path.exists(os.path.join(REPO_DIR, "AES_128_batch.py")):
#   if os.path.exists(REPO_DIR):
#        raise FileNotFoundError(
#            f"Repository directory exists but AES_128_batch.py is missing: {REPO_DIR}"
#        )
#    subprocess.run(["git", "clone", "-q", REPO_URL, REPO_DIR], check=True)
if not os.path.exists(os.path.join(REPO_DIR, "AES_128_batch.py")):
    raise FileNotFoundError(
        f"AES_128_batch.py not found in repository: {REPO_DIR}"
    )

os.chdir(REPO_DIR)
sys.path.insert(0, os.getcwd())
#changed this stuff above

assert os.path.exists('AES_128_batch.py'), "AES_128_batch.py not found in current directory."
print("Repository loaded successfully.")

from AES_128_batch import aes_encrypt, convert_to_binary

NR = 3
BYTE_POS = 0
DIFF_INT = args.diff_val
DIFF_TAG = f"b0_0x{DIFF_INT:02X}"
PT_DIFF = {BYTE_POS: DIFF_INT}
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
out_file = os.path.join(args.output_dir, f"{DIFF_TAG}.json")

# Skip if already completed (built-in fault tolerance)
if os.path.exists(out_file):
    print(f"Result for {DIFF_TAG} already exists. Skipping.")
    sys.exit(0)

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

# --- Pipeline Execution ---
word_size = 8 * len(BYTE_INDICES)
num_blocks = 3 if USE_CT_DIFF else 2

X, Y = gen_data(N_TRAIN, BYTE_INDICES, USE_CT_DIFF, PT_DIFF)
Xv, Yv = gen_data(N_VAL, BYTE_INDICES, USE_CT_DIFF, PT_DIFF)

net = make_resnet(word_size=word_size, num_blocks=num_blocks, depth=DEPTH)
net.compile(optimizer='adam', loss='mse', metrics=['acc'])


#ckpt = f"/tmp/ckpt_{DIFF_TAG}_{os.getpid()}.weights.h5"
ckpt = os.path.join(args.output_dir, f"{DIFF_TAG}.weights.h5")
net.fit(X, Y, epochs=EPOCHS, batch_size=BATCH, validation_data=(Xv, Yv),
        callbacks=[LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001)),
                   ModelCheckpoint(ckpt, monitor='val_loss', save_best_only=True, save_weights_only=True)],
        verbose=0)

net.load_weights(ckpt)
#if os.path.exists(ckpt):
#    os.remove(ckpt)
del X, Y, Xv, Yv

Xc, Yc = gen_data(N_CALIB, BYTE_INDICES, USE_CT_DIFF, PT_DIFF)
thr_opt = best_threshold(Yc, net.predict(Xc, batch_size=10000, verbose=0))
del Xc, Yc

Xt, Yt = gen_data(N_TEST, BYTE_INDICES, USE_CT_DIFF, PT_DIFF)
pt = net.predict(Xt, batch_size=10000, verbose=0)

res = {
    "difference": DIFF_TAG,
    "thr_0.5": metrics(Yt, pt, 0.5),
    "thr_calibrated": metrics(Yt, pt, thr_opt)
}

with open(out_file, "w") as f:
    json.dump(res, f, indent=2)

print(f"Finished {DIFF_TAG}. Results written to {out_file}.")
