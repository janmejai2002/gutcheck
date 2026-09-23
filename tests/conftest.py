import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


@pytest.fixture
def gutcheck_home(tmp_path, monkeypatch):
    """Isolated cache/config dir so tests never touch the real ~/.cache/gutcheck."""
    monkeypatch.setenv("GUTCHECK_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("GUTCHECK_LEASE_CONFIG", raising=False)
    monkeypatch.delenv("GUTCHECK_GATEWAY_CONFIG", raising=False)
    return tmp_path / "home"


def model_available(name="laya-en"):
    home = os.environ.get("GUTCHECK_HOME", os.path.join(os.path.expanduser("~"), ".cache", "gutcheck"))
    return os.path.exists(os.path.join(home, "models", name, "gutcheck.json"))


requires_model = pytest.mark.skipif(
    not (os.environ.get("GUTCHECK_TEST_MODEL") and model_available()),
    reason="set GUTCHECK_TEST_MODEL=1 with laya-en installed to run model tests")
