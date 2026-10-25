"""``mlship init``: a new project that passes ``mlship validate --strict`` (ADR-021)."""

from pathlib import Path

from mlship.domain.errors import MlshipError
from mlship.services.ports import ProjectTemplate


class ScaffoldService:
    def __init__(self, template: ProjectTemplate) -> None:
        self._template = template

    def create(self, target: Path, *, force: bool = False) -> list[Path]:
        """Write every template file; refuse, listing all of them, if any already exists."""
        if target.exists() and not target.is_dir():
            raise MlshipError(f"{target} is a file, not a folder.")
        files = self._template.files()
        links = sorted(name for name in files if (target / name).is_symlink())
        if links:
            raise MlshipError(
                f"{', '.join(links)} in {target} are symbolic links; nothing was written.",
                hint="Remove the links first: mlship never writes through a link.",
            )
        existing = sorted(name for name in files if (target / name).exists())
        if existing and not force:
            raise MlshipError(
                f"{target} already has {', '.join(existing)}; nothing was written.",
                hint="Choose an empty folder, or pass --force to overwrite these files.",
            )
        created: list[Path] = []
        for name, content in files.items():
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            created.append(path)
        return created
