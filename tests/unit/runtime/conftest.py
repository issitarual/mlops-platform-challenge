from pathlib import Path

import pytest


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    """project/, data/ and output/ directories as the runner sees them."""
    dirs = {name: tmp_path / name for name in ("project", "data", "output")}
    for d in dirs.values():
        d.mkdir()
    (dirs["data"] / "data.csv").write_text(
        "a,b,y\n" + "".join(f"{i},{i * 2},{i * 3 + 1}\n" for i in range(20))
    )
    return dirs
