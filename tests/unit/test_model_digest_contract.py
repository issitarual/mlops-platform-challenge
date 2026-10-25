"""Host and serving compute the same model hash, or no registered model would ever load."""

from pathlib import Path

from mlship.adapters.hashing import sha256_untrusted_tree
from mlship.domain.runs import tree_digest
from mlship_runtime.serving.engines.model_files import model_files_digest


def test_host_and_serving_agree_on_the_model_hash(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "MLmodel").write_text("flavors: {}\n")
    (tmp_path / "nested" / "weights.bin").write_bytes(bytes(range(256)))
    host = tree_digest(sha256_untrusted_tree(tmp_path, 10**6, 100))
    assert model_files_digest(tmp_path) == host
