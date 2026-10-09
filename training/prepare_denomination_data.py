"""Build reproducible, group-aware manifests without modifying source images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURRENT_ROOT = PROJECT_ROOT / "dataset" / "archive" / "data" / "data" / "real"
LEGACY_ROOT = PROJECT_ROOT / "dataset" / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "prepared_denomination"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}
CLASS_NAMES = {
    0: "10_rupees",
    1: "20_rupees",
    2: "50_rupees",
    3: "100_rupees",
    4: "200_rupees",
    5: "500_rupees",
    6: "2000_rupees",
}
CURRENT_FOLDERS = {"10": 0, "20": 1, "50": 2, "100": 3, "200": 4, "500": 5, "2000": 6}
LEGACY_FOLDERS = {"ten": 0, "twenty": 1, "fifty": 2, "hundred": 3, "fivehundred": 5}


def image_files(folder: Path) -> list[Path]:
    return sorted(p.resolve() for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def legacy_group(path: Path) -> str:
    """Keep images from the same camera/date collection session together."""
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            make = str(exif.get(271, "unknown")).strip() or "unknown"
            model = str(exif.get(272, "unknown")).strip() or "unknown"
            captured = str(exif.get(36867, exif.get(306, "unknown"))).strip()
            day = captured[:10] if captured else "unknown"
    except Exception:
        return f"legacy_unreadable_{path.stem}"
    if make == model == day == "unknown":
        # Numeric filenames are sequential capture exports; blocks reduce adjacent-frame leakage.
        try:
            block = (int(path.stem) - 1) // 10
            return f"legacy_unknown_block_{block:04d}"
        except ValueError:
            return f"legacy_unknown_{path.stem}"
    return f"legacy_{make}_{model}_{day}".replace(" ", "_").lower()


def build_rows(current_root: Path, legacy_root: Path) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for folder, class_id in CURRENT_FOLDERS.items():
        for path in image_files(current_root / folder):
            rows.append({
                "path": str(path), "class_id": class_id, "class_name": CLASS_NAMES[class_id],
                "source": "current_real", "group_id": f"current_{class_id}_{path.stem.lower()}",
            })
    for folder, class_id in LEGACY_FOLDERS.items():
        for path in image_files(legacy_root / folder):
            rows.append({
                "path": str(path), "class_id": class_id, "class_name": CLASS_NAMES[class_id],
                "source": "legacy", "group_id": f"{class_id}_{legacy_group(path)}",
            })
    return rows


def assign_splits(rows: list[dict[str, str | int]], seed: int) -> None:
    rng = random.Random(seed)
    by_class: dict[int, dict[str, list[dict[str, str | int]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_class[int(row["class_id"])][str(row["group_id"])].append(row)

    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    for groups in by_class.values():
        ordered = list(groups.items())
        rng.shuffle(ordered)
        ordered.sort(key=lambda item: len(item[1]), reverse=True)
        total = sum(len(items) for _, items in ordered)
        counts = Counter()
        for _, items in ordered:
            split = min(ratios, key=lambda name: counts[name] / max(total * ratios[name], 1))
            for row in items:
                row["split"] = split
            counts[split] += len(items)


def write_manifests(rows: list[dict[str, str | int]], output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["path", "class_id", "class_name", "source", "group_id", "split"]
    for name, selected in [("all", rows)] + [(split, [r for r in rows if r["split"] == split]) for split in ("train", "val", "test")]:
        with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
    digest = hashlib.sha256((output_dir / "all.csv").read_bytes()).hexdigest()
    summary = Counter((str(r["split"]), str(r["class_name"]), str(r["source"])) for r in rows)
    with (output_dir / "SUMMARY.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"seed={seed}\nmanifest_sha256={digest}\ntotal={len(rows)}\n")
        for key, count in sorted(summary.items()):
            handle.write(f"{key[0]},{key[1]},{key[2]}={count}\n")
    print(f"Prepared {len(rows)} image records in {output_dir}")
    print(f"Manifest SHA-256: {digest}")
    for split in ("train", "val", "test"):
        counts = Counter(str(r["class_name"]) for r in rows if r["split"] == split)
        print(f"{split}: {sum(counts.values())} images | {dict(sorted(counts.items()))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-root", type=Path, default=CURRENT_ROOT)
    parser.add_argument("--legacy-root", type=Path, default=LEGACY_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for root in (args.current_root, args.legacy_root):
        if not root.is_dir():
            raise SystemExit(f"Dataset directory not found: {root}")
    rows = build_rows(args.current_root.resolve(), args.legacy_root.resolve())
    if not rows:
        raise SystemExit("No supported images found")
    assign_splits(rows, args.seed)
    write_manifests(rows, args.output_dir.resolve(), args.seed)


if __name__ == "__main__":
    main()
