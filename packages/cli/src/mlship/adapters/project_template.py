"""``ProjectTemplate`` shipped inside the package (``templates/project``)."""

from collections.abc import Mapping
from importlib import resources
from importlib.resources.abc import Traversable

_SUFFIX = ".tmpl"
_DOT_PREFIX = "dot-"


class PackagedProjectTemplate:
    """``train.py.tmpl`` becomes ``train.py``; ``dot-gitignore.tmpl`` becomes ``.gitignore``.

    The suffix keeps template code out of the package's own modules, linters and type
    checks; the prefix keeps dotfiles from being dropped when the package is built.
    """

    def files(self) -> Mapping[str, bytes]:
        root = resources.files("mlship.adapters.templates").joinpath("project")
        return dict(_walk(root, ""))


def _walk(node: Traversable, prefix: str) -> list[tuple[str, bytes]]:
    found: list[tuple[str, bytes]] = []
    for child in sorted(node.iterdir(), key=lambda c: c.name):
        if child.name == "__pycache__":
            continue
        name = child.name.removesuffix(_SUFFIX)
        if name.startswith(_DOT_PREFIX):
            name = "." + name.removeprefix(_DOT_PREFIX)
        path = f"{prefix}{name}"
        if child.is_dir():
            found.extend(_walk(child, f"{path}/"))
        else:
            found.append((path, child.read_bytes()))
    return found
