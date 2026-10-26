from mlship.adapters.http_probe import UrllibProbe


def test_only_http_is_ever_opened() -> None:
    probe = UrllibProbe(timeout_seconds=0.5)
    assert not probe.answers("file:///etc/passwd")
    assert not probe.answers("ftp://127.0.0.1/x")


def test_nothing_listening_is_false_not_an_error() -> None:
    assert not UrllibProbe(timeout_seconds=0.5).answers("http://127.0.0.1:9/api/health")
