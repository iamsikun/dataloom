"""Official reproduction must reject missing or substituted benchmark data."""

import importlib.util
from pathlib import Path

import pytest

from dataloom.datasets.adult import _synthetic_stand_in


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_adult.py"
    spec = importlib.util.spec_from_file_location("prepare_adult", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CACHE_FILE", tmp_path / "adult.parquet")
    return module


def test_check_only_requires_cache(preparation):
    with pytest.raises(FileNotFoundError, match="cache is missing"):
        preparation.prepare_adult(check_only=True)


def test_synthetic_cache_is_rejected(preparation):
    _synthetic_stand_in().to_parquet(preparation.CACHE_FILE)
    with pytest.raises(RuntimeError, match="not the genuine UCI schema"):
        preparation.prepare_adult(check_only=True)


def test_download_failure_never_creates_synthetic_cache(preparation, monkeypatch):
    def unavailable():
        raise OSError("network unavailable")

    monkeypatch.setattr(preparation, "_download_adult", unavailable)
    with pytest.raises(OSError, match="network unavailable"):
        preparation.prepare_adult()
    assert not preparation.CACHE_FILE.exists()
