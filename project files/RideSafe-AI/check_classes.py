from pathlib import Path
from collections import Counter

labels_dir = Path(
    r"C:\AI Projects\HelmetViolationSystem\datasets\helmet_final\train\labels"
)

counts = Counter()

for label_file in labels_dir.glob("*.txt"):
    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            class_id = int(line.split()[0])
            counts[class_id] += 1

print("=" * 50)
print("TRAINING DATASET CLASS COUNT")
print("=" * 50)

print(f"Helmet (0):     {counts[0]}")
print(f"No Helmet (1):  {counts[1]}")
print(f"Total objects:  {sum(counts.values())}")

print("=" * 50)