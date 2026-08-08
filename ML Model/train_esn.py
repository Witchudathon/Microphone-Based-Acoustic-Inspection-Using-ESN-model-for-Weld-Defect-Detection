#!/usr/bin/env python3
# ==============================================================================
# train_esn.py — ESN (CNN+Reservoir) Experiment (FFT / HighPass / LowPass)
# Usage:  cd Project && python train_esn.py
#
# Output per preprocessing → Processing/<filter>/ESN/output/
#   metrics.json | confusion_matrix.png | classification_report.txt
#   train_report.txt | training_curves.png | model.pth
#
# Best model (highest F1) → ../models/best_esn_<preprocessing>.pth
# ==============================================================================

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import (
    ALL_CLASSES, LABELS_CSV, SEED,
    load_rows, split_data,
    load_existing_result, run_single_experiment, print_model_summary,
)

MODEL_NAME     = "ESN"
PREPROCESSINGS = ["FFT", "HighPass", "LowPass"]

# Best model is saved here (root models/ folder)
MODELS_DIR = Path(__file__).parent.parent / "models"


def save_best_to_models(all_results):
    """Copy the best preprocessing variant's model.pth to the models/ folder."""
    if not all_results:
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best = max(all_results, key=lambda x: x["metrics"]["macro_f1"])
    pp   = best["preprocessing"]
    src  = Path("Processing") / pp / MODEL_NAME / "output" / "model.pth"

    if not src.exists():
        print(f"\n  ⚠ Source model not found: {src}")
        return

    dst = MODELS_DIR / f"best_esn_{pp.lower()}.pth"
    shutil.copy2(src, dst)

    print(f"\n  ★ Best ESN variant : {pp}  (F1={best['metrics']['macro_f1']:.4f})")
    print(f"  💾 Saved to models : {dst}")

    # Also write a quick summary text next to the model
    summary_path = MODELS_DIR / "best_esn_info.txt"
    with open(summary_path, "w") as f:
        f.write("Best ESN Model Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Preprocessing : {pp}\n")
        f.write(f"Accuracy      : {best['metrics']['accuracy']:.4f}\n")
        f.write(f"Macro F1      : {best['metrics']['macro_f1']:.4f}\n")
        f.write(f"Macro Precision: {best['metrics']['macro_precision']:.4f}\n")
        f.write(f"Macro Recall  : {best['metrics']['macro_recall']:.4f}\n")
        f.write("\nPer-class F1:\n")
        for cls, v in best["metrics"].get("per_class", {}).items():
            f.write(f"  {cls:<12}: {v['f1']:.4f}\n")
        f.write(f"\nSaved as: {dst.name}\n")
    print(f"  📋 Info saved  → {summary_path}")

    # All results comparison
    compare_path = MODELS_DIR / "esn_experiment_compare.txt"
    with open(compare_path, "w") as f:
        f.write("ESN Experiment — All Preprocessing Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"  {'Preprocessing':<12} {'Accuracy':>10} {'F1':>10}\n")
        f.write("  " + "-" * 36 + "\n")
        for r in sorted(all_results, key=lambda x: -x["metrics"]["macro_f1"]):
            m = r["metrics"]
            marker = " ★" if r["preprocessing"] == pp else ""
            f.write(f"  {r['preprocessing']:<12} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}{marker}\n")
    print(f"  📊 Compare log  → {compare_path}")


def main():
    print("=" * 60)
    print(f"  {MODEL_NAME} — Defect Sound Classification Experiment")
    print(f"  Preprocessings: {PREPROCESSINGS}")
    print(f"  Best model → {MODELS_DIR}/")
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
    save_best_to_models(all_results)


if __name__ == "__main__":
    main()
