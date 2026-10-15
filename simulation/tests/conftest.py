import os
import sys
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from grhd.cfgio import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "config" / "base.yaml")


@pytest.fixture(scope="session")
def small_cfg():
    """Test-only configuration: m=5, k=2 so that k <= mC2 holds (V1)."""
    c = load_config(ROOT / "config" / "base.yaml")
    c["model"]["m"] = 5
    c["model"]["k"] = 2
    c["model"]["r_H"] = 6
    return c
