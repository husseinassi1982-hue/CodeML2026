"""Test fixtures: the app on a fresh temporary data folder, and fake photos."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config  # noqa: E402,F401
from database.database import reset_state  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A test client for the app, with records, patients and key file in an empty tmp_path."""
    from fastapi.testclient import TestClient
    from main import app

    monkeypatch.setenv("DAYONE_KEY", "")  # use a key file inside tmp_path
    reset_state(tmp_path)
    return TestClient(app)


@pytest.fixture
def photo():
    """Factory of fake photo uploads (filename, bytes, type); each call gives different bytes."""
    n = {"i": 0}

    def make():  # distinct bytes each time, so they aren't flagged as duplicates
        n["i"] += 1
        return ("page.jpg", f"fake-jpeg-{n['i']}".encode(), "image/jpeg")
    return make
