import json

from scripts.doctor import inspect_project


def test_doctor_finds_models_and_smoke_assets():
    report = inspect_project()
    assert report["models"]["currency_classifier_pytorch.pth"]["present"]
    assert report["models"]["currency_classifier.pkl"]["present"]
    assert report["smoke_images"] == {"cases": 7, "missing": []}
    assert report["missing_project_files"] == []
    json.dumps(report)
