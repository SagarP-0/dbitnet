#!/usr/bin/env python3
"""
Collect classification metrics from all_input_differences_3round.

Expected directory layout:
    all_input_differences_3round/
        mixcolumns_disabled/
            diff_00/result.json
            diff_01/result.json
            ...
        mixcolumns_enabled/
            diff_00/result.json
            diff_01/result.json
            ...

The output is grouped by input difference, with separate entries for
MixColumns enabled/disabled and raw/calibrated test metrics.
"""

import argparse
import json
from pathlib import Path

CONDITIONS = ("mixcolumns_disabled", "mixcolumns_enabled")
METRICS = ("accuracy", "tpr", "tnr")


def load_metrics(result_path: Path, condition: str, folder_name: str) -> tuple[str, dict]:
    """Read one result.json and return its difference and selected metrics."""
    with result_path.open("r", encoding="utf-8") as file:
        result = json.load(file)

    # Prefer the difference recorded in result.json; fall back to diff_XX.
    difference = result.get("difference_hex")
    if difference is None:
        difference = folder_name.removeprefix("diff_")
    difference = str(difference).upper().removeprefix("0X").zfill(2)

    extracted = {
        "mixcolumns": result.get("mixcolumns", condition.removeprefix("mixcolumns_")),
        "status": result.get("status"),
    }

    # Keep only the three requested metrics from each evaluation section.
    for section in ("test", "test_calibrated"):
        source = result.get(section) or {}
        extracted[section] = {
            metric: source.get(metric)
            for metric in METRICS
        }

    return difference, extracted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract TPR, TNR and accuracy from differential-cryptanalysis results."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/home/sagar/Documents/MiniProject/"
            "all_input_differences_3round"
        ),
        help="Root directory containing mixcolumns_disabled/enabled.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: ROOT/summary_metrics.json).",
    )
    args = parser.parse_args()

    root = args.root.expanduser()
    output_path = args.output or root / "summary_metrics.json"

    # Difference -> condition -> extracted result.
    summary = {}
    missing_directories = []
    missing_results = []
    errors = []

    for condition in CONDITIONS:
        condition_dir = root / condition
        if not condition_dir.is_dir():
            missing_directories.append(str(condition_dir))
            continue

        # Each difference is expected to have its own diff_XX directory.
        difference_dirs = sorted(
            (p for p in condition_dir.glob("diff_*") if p.is_dir()),
            key=lambda p: p.name.upper(),
        )

        for diff_dir in difference_dirs:
            result_path = diff_dir / "result.json"
            if not result_path.is_file():
                missing_results.append(str(result_path))
                continue

            try:
                difference, extracted = load_metrics(
                    result_path, condition, diff_dir.name
                )
                summary.setdefault(difference, {})[condition] = extracted
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                errors.append(f"{result_path}: {exc}")

    # Sort differences numerically (00, 01, ..., FF).
    summary = dict(
        sorted(summary.items(), key=lambda item: int(item[0], 16))
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"Saved {len(summary)} input differences to: {output_path}")
    print(f"Result files processed: {sum(len(v) for v in summary.values())}")
    print(f"Missing condition directories: {len(missing_directories)}")
    print(f"Missing result.json files: {len(missing_results)}")
    print(f"Files with errors: {len(errors)}")

    for item in missing_directories + missing_results + errors:
        print(f"WARNING: {item}")


if __name__ == "__main__":
    main()
