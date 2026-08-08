#!/usr/bin/env python3
# ==============================================================================
# train_cnn.py — CNN Experiment (FFT / HighPass / LowPass)
# Usage:  cd Project && python train_cnn.py
#
# Output per preprocessing → Processing/<filter>/CNN/output/
#   metrics.json | confusion_matrix.png | classification_report.txt
#   train_report.txt | training_curves.png | model.pth
# ==============================================================================

import sys
from pathlib import Path

# Allow import from same directory
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    ALL_CLASSES, LABELS_CSV, SEED,
    load_rows, split_data,
    load_existing_result, run_single_experiment, print_model_summary,
)

MODEL_NAME    = "CNN"
PREPROCESSINGS = ["FFT", "HighPass", "LowPass"]


def main():
    print("=" * 60)
    print(f"  {MODEL_NAME} — Defect Sound Classification Experiment")
    print(f"  Preprocessings: {PREPROCESSINGS}")
    print("=" * 60)

    rows = load_rows(LABELS_CSV)
    print(f"\n  Total samples : {len(rows)}")

    classes      = ALL_CLASSES
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_rows, test_rows = split_data(rows, test_size=0.2)
    print(f"  Train: {len(train_rows)} | Test: {len(test_rows)}")

    all_results = []
    for i, pp in enumerate(PREPROCESSINGS, 1):
        existing = load_existing_result(pp, MODEL_NAME)
        if existing:
            all_results.append(existing)
            print(f"\n[{i}/{len(PREPROCESSINGS)}] ⏩ {pp} + {MODEL_NAME} — already done "
                  f"(F1={existing['metrics']['macro_f1']:.4f}), skipping")
            continue
        print(f"\n[{i}/{len(PREPROCESSINGS)}]", end="")
        result = run_single_experiment(pp, MODEL_NAME, train_rows, test_rows, class_to_idx, classes)
        if result:
            all_results.append(result)

    print_model_summary(MODEL_NAME, all_results)


if __name__ == "__main__":
    main()
