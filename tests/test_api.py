from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

from backend import main


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIENT = TestClient(main.app)


def test_health_reports_all_model_states():
    response = CLIENT.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert payload["version"] == main.APP_VERSION
    assert payload["denomination_engine"] in {"pytorch", "classic_cv", "unavailable"}
    assert isinstance(payload["pytorch_model_loaded"], bool)
    assert isinstance(payload["classic_model_loaded"], bool)
    assert isinstance(payload["authenticity_model_loaded"], bool)
    assert isinstance(payload["authenticity_experimental_enabled"], bool)
    assert payload["supported_classes"] == list(main.CLASSES.values())


def test_liveness_and_security_headers():
    response = CLIENT.get("/api/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive", "version": main.APP_VERSION}
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["cache-control"] == "no-store"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_non_image_media_type_is_rejected():
    response = CLIENT.post(
        "/api/detect",
        files={"file": ("not-an-image.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415
    assert response.json()["detail"]["error_code"] == "unsupported_media_type"


def test_invalid_image_bytes_are_rejected():
    response = CLIENT.post(
        "/api/detect",
        files={"file": ("broken.jpg", b"not an image", "image/jpeg")},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error_code"] == "invalid_image"


def test_compressed_size_limit_is_enforced(monkeypatch):
    monkeypatch.setattr(main, "MAX_IMAGE_BYTES", 4)
    response = CLIENT.post(
        "/api/detect",
        files={"file": ("large.jpg", b"12345", "image/jpeg")},
    )
    assert response.status_code == 413
    assert response.json()["detail"]["error_code"] == "image_too_large"


def test_pixel_limit_is_enforced(monkeypatch):
    image_path = PROJECT_ROOT / "data" / "images" / "val" / "val_10_rupees_0.jpg"
    monkeypatch.setattr(main, "MAX_IMAGE_PIXELS", 1)
    with pytest.raises(main.ImageValidationError) as error:
        main.decode_image_bytes(image_path.read_bytes())
    assert error.value.error_code == "too_many_pixels"
    assert error.value.status_code == 413


def test_upload_response_preserves_bbox(monkeypatch):
    image_path = PROJECT_ROOT / "data" / "images" / "val" / "val_10_rupees_0.jpg"
    assert cv2.imread(str(image_path)) is not None

    monkeypatch.setattr(
        main,
        "perform_inference",
        lambda _image: {
            "class_id": 0,
            "class_name": "10_rupees",
            "friendly_name": "Ten Rupees",
            "confidence": 0.9,
            "message": "smoke test",
            "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
        },
    )

    response = CLIENT.post(
        "/api/detect",
        files={"file": (image_path.name, image_path.read_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["bbox"] == {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
