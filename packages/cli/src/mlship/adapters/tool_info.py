"""Which mlship produced a run: version plus the git commit of its source (ADR-019).

``git_dirty`` tells whether the tool's own packages had uncommitted changes: a run made
with uncommitted tool code cannot be reproduced from the commit alone. Installed
without git (or outside a checkout), the commit is reported as ``unknown``.
"""

import subprocess
from importlib import metadata

from mlship.adapters.process import Runner, find_executable, run
from mlship.adapters.runtime_packager import RUNTIME_DIST, SDK_DIST, source_dir
from mlship.domain.errors import MlshipError
from mlship.domain.runs import ToolInfo

CLI_DIST = "mlship"
UNKNOWN = "unknown"


class GitToolInfo:
    def __init__(self, runner: Runner = subprocess.run) -> None:
        self._runner = runner

    def info(self) -> ToolInfo:
        version = metadata.version(CLI_DIST)
        try:
            sources = [source_dir(dist) for dist in (CLI_DIST, RUNTIME_DIST, SDK_DIST)]
        except MlshipError:
            return ToolInfo(version, UNKNOWN, None)
        try:
            git = find_executable("git", hint="")
        except MlshipError:
            return ToolInfo(version, UNKNOWN, None)
        head = run([git, "rev-parse", "HEAD"], sources[0], self._runner)
        status = run(
            [git, "status", "--porcelain", "--", *map(str, sources)], sources[0], self._runner
        )
        if head.returncode != 0 or status.returncode != 0:
            return ToolInfo(version, UNKNOWN, None)
        return ToolInfo(version, head.stdout.strip(), bool(status.stdout.strip()))
