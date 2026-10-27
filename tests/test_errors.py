import httpx
import pytest

from harness.errors import Ambiguous, NotApplied, Rejected, request


def client_returning(status: int) -> httpx.Client:
    return httpx.Client(
        base_url="http://sim",
        transport=httpx.MockTransport(lambda req: httpx.Response(status, json={"status": status})),
    )


def client_raising(exc: Exception) -> httpx.Client:
    def handler(req):
        raise exc

    return httpx.Client(base_url="http://sim", transport=httpx.MockTransport(handler))


def test_success_returns_json():
    assert request(client_returning(201), "POST", "/x") == {"status": 201}


@pytest.mark.parametrize(
    "status, error",
    [
        (429, NotApplied),
        (500, Ambiguous),
        (502, Ambiguous),
        (503, Ambiguous),
        (504, Ambiguous),
        (400, Rejected),
        (404, Rejected),
        (409, Rejected),
        (412, Rejected),
        (422, Rejected),
    ],
)
def test_status_classification(status, error):
    with pytest.raises(error) as info:
        request(client_returning(status), "POST", "/x")
    assert info.value.status == status
    assert info.value.body == {"status": status}


@pytest.mark.parametrize(
    "exc, error",
    [
        (httpx.ConnectError("refused"), NotApplied),
        (httpx.ConnectTimeout("connect timeout"), NotApplied),
        (httpx.ReadTimeout("read timeout"), Ambiguous),
        (httpx.WriteTimeout("write timeout"), Ambiguous),
        (httpx.ReadError("connection reset"), Ambiguous),
        (httpx.RemoteProtocolError("server disconnected"), Ambiguous),
    ],
)
def test_transport_classification(exc, error):
    with pytest.raises(error):
        request(client_raising(exc), "POST", "/x")
