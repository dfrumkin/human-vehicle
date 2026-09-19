"""Tests for the Gemini backend's own decisions, against a stub client.

No network and no API key. What the SDK does with a request is not under test; what this code puts
in one is -- above all `store=False`, which inverts the API's default and whose regression would
silently retain 55 days of logs describing the people in the footage.
"""

from pathlib import Path
from typing import Any

import pytest

from human_vehicle.vlm import GeminiBackend, sum_usage

SCHEMA: dict[str, Any] = {"type": "object", "properties": {"interactions": {"type": "array"}}}


class _StubState:
    def __init__(self, name: str) -> None:
        self.name = name


class _StubHandle:
    def __init__(self, name: str, state: str = "ACTIVE") -> None:
        self.name = name
        self.uri = f"https://files/{name}"
        self.mime_type = "video/mp4"
        self.state = _StubState(state)
        self.error = "the upload was rejected"


class _StubUsage:
    total_input_tokens = 1000
    total_output_tokens = 200
    total_thought_tokens = 400
    total_tokens = 1600


class _StubInteraction:
    output_text = '{"interactions": []}'
    usage = _StubUsage()


class _StubFiles:
    """The four `client.files` calls the backend makes, with the states it will see recorded."""

    def __init__(self, states: list[str] | None = None) -> None:
        self.states = list(states or ["ACTIVE"])
        self.uploaded: list[str] = []
        self.deleted: list[str] = []

    def upload(self, *, file: str) -> _StubHandle:
        self.uploaded.append(file)
        return _StubHandle(f"files/{Path(file).stem}", self.states.pop(0))

    def get(self, *, name: str) -> _StubHandle:
        return _StubHandle(name, self.states.pop(0) if self.states else "ACTIVE")

    def delete(self, *, name: str) -> None:
        self.deleted.append(name)


class _StubInteractions:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> _StubInteraction:
        self.requests.append(request)
        return _StubInteraction()


class _StubClient:
    def __init__(self, states: list[str] | None = None) -> None:
        self.files = _StubFiles(states)
        self.interactions = _StubInteractions()


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    """A stand-in file. Nothing decodes it -- the stub client only ever sees its path."""
    path = tmp_path / "annotated.mp4"
    path.write_bytes(b"not really an mp4")
    return path


def test_every_request_switches_off_retention(clip: Path) -> None:
    """`store=False` inverts the API's default, which keeps interaction logs for 55 days.

    This is the one test that has to exist: a regression here has no visible symptom, and what
    would be retained is a model's description of the people in someone's footage.
    """
    client = _StubClient()
    backend = GeminiBackend(client=client)

    backend.generate(clip, "a prompt", window=None, schema=SCHEMA)
    backend.generate(clip, "a prompt", window=(4.0, 10.0), schema=SCHEMA)

    assert [request["store"] for request in client.interactions.requests] == [False, False]


def test_a_window_becomes_the_requests_offsets(clip: Path) -> None:
    """Without these the whole clip is sent for every window: a plausible, wholly wrong run."""
    client = _StubClient()
    backend = GeminiBackend(client=client, fps=4.0)

    backend.generate(clip, "a prompt", window=(4.0, 10.5), schema=SCHEMA)

    processing = client.interactions.requests[0]["input"][0]["processing"]
    # Decimal seconds with an "s" suffix, alongside fps, rather than numbers or milliseconds.
    assert processing["start_offset"] == "4s"
    assert processing["end_offset"] == "10.5s"
    assert processing["fps"] == 4.0


def test_a_whole_clip_request_carries_no_offsets(clip: Path) -> None:
    client = _StubClient()
    backend = GeminiBackend(client=client)

    backend.generate(clip, "a prompt", window=None, schema=SCHEMA)

    processing = client.interactions.requests[0]["input"][0]["processing"]
    assert "start_offset" not in processing and "end_offset" not in processing


def test_the_schema_and_prompt_reach_the_request(clip: Path) -> None:
    """The caller's schema is what constrains the output, so it has to arrive unchanged."""
    client = _StubClient()
    backend = GeminiBackend(client=client, model_id="gemini-3.8-flash", seed=7)

    backend.generate(clip, "the prompt text", window=None, schema=SCHEMA)

    request = client.interactions.requests[0]
    assert request["model"] == "gemini-3.8-flash"
    assert request["response_format"]["schema"] == SCHEMA
    assert request["response_format"]["mime_type"] == "application/json"
    assert request["generation_config"]["seed"] == 7
    assert request["input"][1] == {"type": "text", "text": "the prompt text"}


def test_a_clip_is_uploaded_once_and_reused(clip: Path) -> None:
    """The reason this uses the File API at all: a windowed run is one upload, seven requests."""
    client = _StubClient()
    backend = GeminiBackend(client=client)

    for window in ((0.0, 6.0), (4.0, 10.0), (8.0, 14.0)):
        backend.generate(clip, "a prompt", window=window, schema=SCHEMA)

    assert client.files.uploaded == [str(clip)]
    assert len(client.interactions.requests) == 3


def test_cleanup_deletes_what_the_backend_uploaded(clip: Path) -> None:
    client = _StubClient()
    backend = GeminiBackend(client=client)
    backend.generate(clip, "a prompt", window=None, schema=SCHEMA)

    deleted = backend.delete_uploads()

    assert deleted == client.files.deleted == ["files/annotated"]
    # A second cleanup has nothing left to do, and uploading again is then a fresh upload.
    assert backend.delete_uploads() == []


def test_an_upload_that_failed_raises(clip: Path) -> None:
    """A dead upload must end the call rather than be handed to the model as a live file."""
    backend = GeminiBackend(client=_StubClient(states=["PROCESSING", "FAILED"]))

    with pytest.raises(RuntimeError, match="upload failed"):
        backend.upload(clip, timeout_s=1.0, poll_s=0.0)


def test_an_upload_that_never_becomes_active_gives_up(clip: Path) -> None:
    """Waiting only for ACTIVE hangs forever, which in a notebook is a cell that never returns."""
    backend = GeminiBackend(client=_StubClient(states=["PROCESSING"] * 10))

    with pytest.raises(TimeoutError):
        backend.upload(clip, timeout_s=-1.0, poll_s=0.0)


def test_usage_bills_thinking_at_the_output_rate(clip: Path) -> None:
    """Thinking is reported outside `total_output_tokens` but billed like it, so a cost that
    ignored it would under-report by roughly 3x at thinking_level="high"."""
    client = _StubClient()
    backend = GeminiBackend(client=client)

    response = backend.generate(clip, "a prompt", window=None, schema=SCHEMA)

    assert response.usage["billable_output_tokens"] == 600.0
    assert response.usage["estimated_cost_usd"] == pytest.approx(1000 / 1e6 * 0.75 + 600 / 1e6 * 3.75)


def test_the_config_records_what_ran() -> None:
    """A record names the model and the SDK shape it was written against, not what was intended."""
    backend = GeminiBackend(model_id="gemini-3.8-flash", fps=4.0, thinking_level="high", seed=3)

    assert backend.config["model_id"] == "gemini-3.8-flash"
    assert backend.config["sdk_version"]
    assert backend.slug == "3.8-flash__fps4__low__think-high__seed3"
    # The tolerance a windowed run validates edges against is this backend's sampling interval.
    assert backend.tolerance_s == 0.25


def test_a_backend_without_a_key_only_fails_when_it_is_used(monkeypatch: pytest.MonkeyPatch, clip: Path) -> None:
    """Construction stays free of credentials, so the no-API checks can run without one."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    backend = GeminiBackend()

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        backend.generate(clip, "a prompt", window=None, schema=SCHEMA)


def test_usage_totals_skip_what_a_failed_call_never_reported() -> None:
    """A key one call is missing counts as zero there rather than vanishing from the total."""
    assert sum_usage([{"total_tokens": 10.0, "estimated_cost_usd": 0.5}, {"total_tokens": 4.0}, {}]) == {
        "estimated_cost_usd": 0.5,
        "total_tokens": 14.0,
    }
