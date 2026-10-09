from collections import Counter
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("split,expected_per_class", [("train", 30), ("val", 10)])
def test_synthetic_yolo_dataset_is_balanced_and_paired(split, expected_per_class):
    image_dir = PROJECT_ROOT / "data" / "images" / split
    label_dir = PROJECT_ROOT / "data" / "labels" / split
    images = {path.stem for path in image_dir.glob("*.jpg")}
    labels = {path.stem for path in label_dir.glob("*.txt")}
    assert images == labels

    counts = Counter()
    for label_path in label_dir.glob("*.txt"):
        lines = label_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        fields = lines[0].split()
        assert len(fields) == 5
        class_id = int(fields[0])
        coordinates = [float(value) for value in fields[1:]]
        assert class_id in range(7)
        assert all(0.0 <= value <= 1.0 for value in coordinates)
        counts[class_id] += 1

    assert counts == Counter({class_id: expected_per_class for class_id in range(7)})
