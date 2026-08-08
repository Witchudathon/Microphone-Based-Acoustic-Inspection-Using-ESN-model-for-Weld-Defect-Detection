import csv
from collections import Counter
from pathlib import Path

LABELS_PATH = Path("dataset/labels.csv")

counter = Counter()

with LABELS_PATH.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        counter[row["class_name"]] += 1

print("=== File count per class (from labels.csv) ===")
for cls, count in counter.items():
    print(f"{cls:>10}: {count}")
