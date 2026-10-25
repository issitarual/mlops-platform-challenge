"""Run listing and promotion by explicit alias (never "latest"; ADR-015)."""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from mlship.domain.errors import MlshipError
from mlship.domain.runs import Promotion, RunRecord, is_valid_alias
from mlship.services.ports import ModelRegistry, RunRepository


class RegistryService:
    def __init__(
        self,
        run_repository: RunRepository,
        registry: ModelRegistry,
        actor: Callable[[], str],
    ) -> None:
        self._runs = run_repository
        self._registry = registry
        self._actor = actor  # who is promoting: OS user locally, IAM principal in production

    def list_runs(self, model_name: str, limit: int) -> Sequence[RunRecord]:
        self._runs.check_available()
        return self._runs.list_recent(model_name, limit)

    def promote(self, model_name: str, version: int, alias: str) -> Promotion:
        """Point ``alias`` at ``version``, recording who did it, when and what it replaced.

        No metric comparison yet: the champion/challenger gate is ADR-013 (P2).
        """
        if not is_valid_alias(alias):
            raise MlshipError(
                f"{alias!r} is not a valid alias.",
                hint="Use lowercase letters, digits, '-' or '_' (e.g. champion, challenger). "
                "'latest' is reserved: serving always follows an explicit alias.",
            )
        self._runs.check_available()
        target = self._registry.get_version(model_name, version)  # fails if missing
        previous = self._registry.get_alias(model_name, alias)
        promotion = Promotion(alias, target, previous)
        if not promotion.changed:
            return promotion
        self._registry.set_alias(
            model_name,
            alias,
            version,
            tags={
                f"mlship.{alias}.promoted_by": self._actor(),
                f"mlship.{alias}.promoted_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
                f"mlship.{alias}.replaced_version": str(previous.version) if previous else "none",
            },
        )
        return promotion
