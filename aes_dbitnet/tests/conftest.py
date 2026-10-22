import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def aes():
    from aesdist.baseline_adapter import BaselineNotFound, load_aes
    try:
        return load_aes()
    except BaselineNotFound as e:
        pytest.skip(f"original NDGohr AES_128_batch.py not available: {e}")
