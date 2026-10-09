import json
from pathlib import Path

import cv2
import pytest

from backend import main


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SMOKE_CASES = json.loads((Path(__file__).parent / "smoke_images.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", SMOKE_CASES, ids=lambda case: case["expected_class"])
def test_shipped_smoke_image_is_decodable_and_returns_contract(case):
    image_path = PROJECT_ROOT / case["path"]
    image = cv2.imread(str(image_path))
    assert image is not None

    result = main.perform_inference(image)
    assert result["class_id"] in range(-1, 7)
    assert result["class_name"] in {*main.CLASSES.values(), "unknown"}
    assert 0.0 <= float(result["confidence"]) <= 1.0
    assert result["friendly_name"]
    assert result["message"]


def test_at_least_one_denomination_model_is_available():
    assert main.PYTORCH_MODEL is not None or main.MODEL is not None
