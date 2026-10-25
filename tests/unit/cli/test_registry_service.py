import pytest
from fakes import FakeModelRegistry, FakeRunRepository

from mlship.domain.errors import MlshipError, ModelNotFoundError, TrackingUnavailableError
from mlship.services.registry_service import RegistryService


def _service(registry: FakeModelRegistry, runs: FakeRunRepository | None = None) -> RegistryService:
    return RegistryService(runs or FakeRunRepository(), registry, actor=lambda: "laura")


def _registry_with(versions: int) -> FakeModelRegistry:
    registry = FakeModelRegistry()
    for number in range(versions):
        registry.register("demo", f"run{number}", f"sha{number}")
    return registry


def test_first_promotion_records_who_when_and_what_it_replaced() -> None:
    registry = _registry_with(1)
    result = _service(registry).promote("demo", 1, "champion")
    assert result.changed and result.previous is None
    assert registry.aliases[("demo", "champion")] == 1
    tags = registry.version_tags[("demo", 1)]
    assert tags["mlship.champion.promoted_by"] == "laura"
    assert tags["mlship.champion.replaced_version"] == "none"
    assert "mlship.champion.promoted_at" in tags


def test_promotion_moves_the_alias_and_reports_the_previous_version() -> None:
    registry = _registry_with(2)
    service = _service(registry)
    service.promote("demo", 1, "champion")
    result = service.promote("demo", 2, "champion")
    assert result.previous is not None and result.previous.version == 1
    assert registry.aliases[("demo", "champion")] == 2


def test_promoting_the_current_holder_changes_nothing() -> None:
    registry = _registry_with(1)
    service = _service(registry)
    service.promote("demo", 1, "champion")
    registry.version_tags.clear()
    assert not service.promote("demo", 1, "champion").changed
    assert registry.version_tags == {}


def test_missing_version_is_a_clear_error() -> None:
    with pytest.raises(ModelNotFoundError, match="no version 7"):
        _service(_registry_with(1)).promote("demo", 7, "champion")


@pytest.mark.parametrize("alias", ["latest", "Champion", "v1!", ""])
def test_reserved_or_malformed_aliases_are_rejected(alias: str) -> None:
    with pytest.raises(MlshipError, match="not a valid alias"):
        _service(_registry_with(1)).promote("demo", 1, alias)


def test_unreachable_tracking_is_reported_before_touching_the_registry() -> None:
    service = _service(_registry_with(1), FakeRunRepository(available=False))
    with pytest.raises(TrackingUnavailableError):
        service.promote("demo", 1, "champion")
