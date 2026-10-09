"""Evaluate denomination inference on an independent photographed-note dataset."""

import argparse
import json
from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend import main  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def evaluate(dataset_dir):
    dataset_dir = Path(dataset_dir).resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if main.PYTORCH_MODEL is None and main.MODEL is None:
        raise RuntimeError("No denomination model loaded")

    class_names = list(main.CLASSES.values())
    confusion = {actual: {predicted: 0 for predicted in [*class_names, "unknown"]} for actual in class_names}
    unreadable = []
    total = 0

    for actual in class_names:
        class_dir = dataset_dir / actual
        if class_dir.is_dir():
            candidates = class_dir.rglob("*")
        else:
            # Also support generated flat datasets whose filenames contain the
            # class token, such as val_100_rupees_0.jpg.
            candidates = (
                path for path in dataset_dir.iterdir()
                if f"_{actual}_" in path.name
            )
        for image_path in sorted(candidates):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                unreadable.append(str(image_path))
                continue
            result = main.perform_inference(image)
            predicted = result["class_name"]
            if predicted not in confusion[actual]:
                predicted = "unknown"
            confusion[actual][predicted] += 1
            total += 1

    if total == 0:
        raise RuntimeError("No readable evaluation images found in the expected class folders")

    per_class = {}
    for class_name in class_names:
        true_positive = confusion[class_name][class_name]
        false_negative = sum(confusion[class_name].values()) - true_positive
        false_positive = sum(confusion[other][class_name] for other in class_names if other != class_name)
        precision = safe_div(true_positive, true_positive + false_positive)
        recall = safe_div(true_positive, true_positive + false_negative)
        per_class[class_name] = {
            "support": sum(confusion[class_name].values()),
            "precision": precision,
            "recall": recall,
            "f1": safe_div(2 * precision * recall, precision + recall),
        }

    correct = sum(confusion[name][name] for name in class_names)
    return {
        "dataset": str(dataset_dir),
        "warning": "Independent real photographs are required; synthetic results are not deployment evidence.",
        "images_evaluated": total,
        "unreadable_images": unreadable,
        "accuracy": safe_div(correct, total),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Directory containing one folder per denomination")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    report = evaluate(args.dataset)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main_cli()
