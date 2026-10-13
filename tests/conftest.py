import pytest

from qcad import config

requires_data = pytest.mark.skipif(
    not (config.DATA_ROOT / "bottle").is_dir(),
    reason=f"dataset not found: {config.DATA_ROOT}",
)
requires_split = pytest.mark.skipif(
    not config.SPLIT_FILE.exists(), reason="split file not found",
)
