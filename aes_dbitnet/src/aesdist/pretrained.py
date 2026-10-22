"""Pretrained NDGohr .h5 handling: filename parsing and compatibility checks ONLY.

No model is evaluated or claimed compatible here; the original files were not available
when this module was written. A pretrained model has a different training history (10^7
training examples, MSE loss, cyclic LR, 5-GPU MirroredStrategy) than the controlled runs.
"""
import ast
import re

PATTERN = re.compile(r"^AES(?P<ws>\d+)_data_index_(?P<idx>.+?)_best_model_(?P<nr>\d+)r_depth(?P<depth>\d+)"
                     r"_num_epochs(?P<ep>\d+)_acc_(?P<acc>[0-9.]+?)\.h5$")


def parse_filename(name):
    m = PATTERN.match(name.split("/")[-1])
    if not m:
        raise ValueError(f"unrecognised pretrained model filename: {name}")
    raw = m["idx"]
    idx = None if raw == "None" else ast.literal_eval(raw)
    ws = int(m["ws"])
    return {"word_size_bits": ws, "bytes_per_ciphertext": ws // 8, "data_index_raw": raw, "data_index": idx,
            "rounds": int(m["nr"]), "depth": int(m["depth"]), "epochs": int(m["ep"]),
            "acc_in_filename": float(m["acc"]),
            "caveat": "filename accuracy = max(val_acc) over epochs, but the saved weights are the best val_loss epoch "
                      "(ModelCheckpoint monitor='val_loss' + rename in main_net.py); they may differ."}


def check_compat(model, parsed):
    """Structural compatibility with the flat-input NDGohr layout. Returns list of (name, ok, detail)."""
    ws = parsed["word_size_bits"]
    shp = tuple(model.input_shape[1:])
    return [("input is flat vector of 2*word_size bits", shp == (2 * ws,), f"model input {shp}, expected {(2 * ws,)}"),
            ("single sigmoid output", tuple(model.output_shape[1:]) == (1,), str(model.output_shape))]
