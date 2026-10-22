# Baseline inspection (NDGohr) and decisions

**Scope of inspection.** Only the four uploaded files were read: `AES_128_batch.py`, `main_net.py`,
`16bit_AES_key_recovery_attack.py`, `test_something.ipynb`. The GitHub repository, its README, any
requirements file, and every `.h5` model were **not** accessed. Their SHA-256 hashes are recorded by the smoke
test (`smoke_report.json`) so later runs can prove the baseline was not modified.

Legend: **[T]** confirmed by an executed test in this project (run in the development sandbox, CPython 3.10.20,
TF 2.15.1 CPU), **[S]** read from source only, **[?]** open.

## AES (`AES_128_batch.py`)
| Item | Finding | Status |
|---|---|---|
| Round convention (`aes_encrypt`) | ARK(rk0); `nr-1` × [SB, SR, MC, ARK]; final SB, SR, *optional* MC (`flag`), ARK(rk[nr]) | [T] vs independent reference, nr=1..4, both `flag` values |
| Standard AES-128 at nr=10, `flag=False` | FIPS-197 App. B and C.1 vectors reproduced; matches PyCryptodome on 40 random vectors | [T] |
| Key schedule (`key_expansion`) | Round keys 1 and 10 of FIPS-197 A.1 reproduced; `key_expansion(nr=3)[:3] == key_expansion(nr=2)` | [T] |
| S-box / inverse S-box tables | Equal to tables generated from the GF(2^8) definition | [T] |
| Byte order | flat index `i = 4*col + row` (input is column-major) | [T] |
| Final-MC flag | `flag=True` keeps MixColumns in the last round (non-standard). Source comment: original 2-round distinguishers were trained with it | [S] comment; consistency [T] (below) |
| **Output dtype** | `aes_encrypt`/`aes_decrypt` return **int64** (the `s_box` table is int64), not uint8. Values are correct bytes; they must be cast before `.tobytes()`/storage. This project casts to uint8 at the storage boundary | [T] |
| ShiftRows comment | Comment says "to the right"; the code is the standard left rotation by row index (FIPS-correct) | [T] via FIPS vectors |
| Differential | `pt1[:,0] ^= 0x80` → byte 0 (row 0, col 0), MSB | [T] |
| Labels (`generate_train_data`) | `Y = urandom(n) & 1` (Bernoulli, not exact balance). Y=1: real pair. Y=0: `pt1` replaced with fresh random plaintext under the *same key* (so negatives are *not* random ciphertexts) | [S] |
| Keys | One fresh random key per example | [S] |
| Bit encoding (`convert_to_binary`) | MSB-first; layout `[C0 bits | C1 bits]` | [T] encode_bits is bit-identical |
| Active truncation branch | `ct0[:, data_index]` with **list** `data_index` (e.g. `[0,13]`), 2 bytes. Integer `data_index` breaks it. All other truncations exist only as commented-out code | [S] |
| Randomness | `os.urandom` only → cannot be seeded; **not** monkeypatched here | [S] |

## Network and training (`main_net.py`)
`make_resnet(depth=5, reg_param=1e-5)`: flat input → `Reshape((2,word_size))` → `Permute((2,1))` → Conv1D(16,k=1)-BN-ReLU →
5 residual blocks (2×Conv1D(16,k=3)-BN-ReLU + Add) → Flatten → Dense64-BN-ReLU ×2 → Dense(1,sigmoid); L2(1e-5) everywhere; **no dilation**.
`word_size` = bits per ciphertext (16 for 2 bytes, 128 for 16 bytes). Hence the baseline input is already `(bits, 2)`
with C0/C1 as channels — identical to the primary DBitNet representation chosen here.
- Parameters: **29,777** (608 non-trainable) at `word_size=16`; **144,465** at 128. Hand-derived before coding, **[T]** confirmed by `model.count_params()`.
- Loss **MSE**; optimizer `'adam'` with `LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))`; batch `1000 × n_GPUs` (5 hard-coded GPUs → 5000); 20 epochs; 10^7 train / 10^6 validation examples, freshly generated; checkpoint on best `val_loss`. [S]
- The saved file is renamed with `max(val_acc)` over all epochs; the weights inside are the best-`val_loss` epoch. Filename accuracy may therefore not describe the weights. [S]
- Equivalence of this project's `ndgohr` model with the original flat-input pipeline (Reshape+Permute): outputs identical to 1e-6. [T]

## Evaluation notebook (`test_something.ipynb`)
Threshold 0.5; accuracy/TPR/TNR/MSE and "% random above real median". Stored output (authors' environment, **not reproduced here**): 3-round `[0,13]` model, acc 0.504012, TPR 0.9919, TNR 0.0159 on 10^6 fresh samples, i.e. near chance.

## Key recovery (`16bit_AES_key_recovery_attack.py`)
- Partial decryption: ARK(guess) → InvSR → InvSB (no InvMC) on a 3-round no-final-MC ciphertext. **[T]** with the true last subkey this equals `aes_encrypt(nr=2, flag=True)` exactly — this confirms the 2-round models are meant to be trained with final MC, and the 3-round key schedule prefix is shared.
- Ciphertext/subkey byte positions for 2-byte block `i` (state bytes `[2i,2i+1]`) derived from ShiftRows and **[T]** equal to the attack file's `key_index` table for all 8 blocks (e.g. `[0,13]`, `[10,7]`).
- A 2-byte distinguisher needs only two bytes of each ciphertext and the two matching subkey bytes: `state[i] = InvSBox[ct[j] ^ k[j]]` **[T]** equal to the full-state route.
- Enumerates **absolute** values `g ∈ [0,2^16)` (`kg1 = g & 0xFF` → `key_index[0]`, `kg2 = g >> 8` → `key_index[1]`) with the other subkey bytes set to the *true* ones; score = mean of `log2(p/(1-p))` (unclipped), survival threshold `c=7`. [S]
- `n=3` pairs per structure; retry guard `num > 2**0` allows up to 2 structures (6 or 12 plaintexts per trial); `t=1` per `data_index`, i.e. 8 separate trials. [S]
- Not importable safely: executes at import and sets `CUDA_VISIBLE_DEVICES="0"`. It is never imported or run by this project.

## Differences between this project's controlled data and the original generator
Seeded PCG64 streams instead of `os.urandom`; **exactly balanced** labels per chunk; whole-chunk train/val/test splits with independent keys per example (no key reuse across or within splits); ciphertext pairs stored once per `(rounds, final_mc)` and truncated at load time so all truncations and both architectures see identical examples; 10^6 examples (800k train) vs. the baseline's 10^7.

## Decisions recorded (from the project owner)
1. Exact 50/50 balance for new experiments. 2. NDGohr and DBitNet trained on identical data/splits under a shared "controlled" protocol (BCE, Adam 1e-3, cosine, equal epochs); NDGohr's original recipe (MSE, cyclic LR) available as a *sensitivity* protocol; pretrained models reported separately. 3. Primary DBitNet input `(batch, bits, 2)`; `(batch, 2·bits, 1)` as ablation (`dbitnet_concat1`). 4. No monkeypatching of `os.urandom`. 5. Absolute-key attack and difference-based WKRP kept separate. 6. FIPS-197 and convention validation precede training. 7. Skeleton + tests + CPU smoke first; no automatic full runs.

## Open items
- [?] Pretrained `.h5` files: exact filenames, TF/Keras version they load under, whether filename accuracy matches the saved weights, and whether the 2-round models were all trained with final MC — needs the files.
- [?] 3-round "row/column" 4-byte case: source has both a column and a row variant commented out; config defaults to column 0 (`[0,1,2,3]`), row 0 available.
- [?] 3-round 1-byte index (no source precedent) and 2-round 1-byte index: config choice, index 0.
- [?] Meaning of the factors in `log2(2^16·3·2·8)=21.585` and the success probability of 3 pairs: must come from instrumented repeated attack runs, not this formula.
- [?] README/requirements of the upstream repo (not accessed).
- Design note: default DBitNet (filters 32, GAP head) has **15,553** params at 16 bits vs NDGohr's 29,777 — a parameter-count confound to be reported (and ablated via `filters`/`pool`) in the comparison. For 16-bit inputs the dilation-16 layer exceeds the sequence length (only zero-padded context), so the dilation schedule must be treated as an ablation, not an assumption.
