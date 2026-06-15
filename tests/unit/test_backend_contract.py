"""The factory and every backend satisfy the Layer-1 contract."""
from kempnerpulse.reader import Backend, BackendKind, ReaderConfig, make_backend
from kempnerpulse.reader.dcgmi import DcgmiBackend
from kempnerpulse.reader.prometheus import PrometheusBackend
from kempnerpulse.reader.replay import ReplayBackend


def test_make_backend_returns_expected_types():
    assert isinstance(make_backend(ReaderConfig(backend=BackendKind.DCGMI)), DcgmiBackend)
    assert isinstance(make_backend(ReaderConfig(backend=BackendKind.PROMETHEUS)), PrometheusBackend)
    assert isinstance(make_backend(ReaderConfig(backend=BackendKind.REPLAY)), ReplayBackend)


def test_backends_satisfy_protocol():
    for backend in (DcgmiBackend(), PrometheusBackend(), ReplayBackend()):
        assert isinstance(backend, Backend)


def test_caps_report_their_kind():
    assert DcgmiBackend().caps.kind is BackendKind.DCGMI
    assert PrometheusBackend().caps.kind is BackendKind.PROMETHEUS
    assert ReplayBackend().caps.kind is BackendKind.REPLAY
    # dcgmi advertises its known field set
    assert "DCGM_FI_PROF_SM_ACTIVE" in DcgmiBackend().caps.fields
