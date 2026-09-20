"""Tests for the end-to-end script.

What the stages do is covered by the other test modules, and running a detector or a model here is
not on the table, so these cover the decisions the script itself makes: which clips it finds, which
backend it builds, where it writes, and what it does when a clip fails.

The batch tests substitute the stages with stubs on the module, which is what lets a whole run
happen in milliseconds with no weights, no credential and no network.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from human_vehicle import pipeline
from human_vehicle.interactions import InteractionRun, ReportedInteraction, WindowRun
from human_vehicle.tracking import Category
from human_vehicle.vlm import GeminiBackend, QwenBackend


def _args(*argv: str) -> Any:
    """Parse a command line the way `main` does, without running anything."""
    return pipeline.build_parser().parse_args(["in.mp4", "out", *argv])


# --- Input discovery -------------------------------------------------------------------------


def test_a_file_input_is_the_one_clip(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.touch()

    assert pipeline.discover_videos(clip, tmp_path / "out") == [clip]


def test_a_folder_input_is_every_mp4_in_it_in_name_order(tmp_path: Path) -> None:
    for name in ("b.mp4", "a.mp4", "notes.txt"):
        (tmp_path / name).touch()

    found = pipeline.discover_videos(tmp_path, tmp_path.parent / "out")

    assert [path.name for path in found] == ["a.mp4", "b.mp4"]


def test_clips_under_the_output_folder_are_not_picked_up(tmp_path: Path) -> None:
    """Writing the output inside the input folder must not feed a previous run back in.

    An annotated clip re-annotated produces a run that looks entirely normal and answers a
    different question, so this cannot be left to the reader noticing.
    """
    (tmp_path / "clip.mp4").touch()
    output = tmp_path / "out" / "annotated"
    output.mkdir(parents=True)
    (output / "clip__yolo26x.mp4").touch()

    found = pipeline.discover_videos(tmp_path, tmp_path / "out")

    assert [path.name for path in found] == ["clip.mp4"]


def test_a_folder_with_no_clips_fails_naming_the_pattern(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"no \*\.mp4 files in"):
        pipeline.discover_videos(tmp_path, tmp_path / "out")


def test_an_input_that_does_not_exist_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no such file or folder"):
        pipeline.discover_videos(tmp_path / "missing", tmp_path / "out")


# --- Detector input size ---------------------------------------------------------------------


@pytest.mark.parametrize(("height", "expected"), [(288, 640), (1079, 640), (1080, 1280), (2160, 1280)])
def test_auto_imgsz_steps_up_at_1080(height: int, expected: int) -> None:
    """The one arithmetic rule the script owns, checked at its boundary."""
    assert pipeline.imgsz_for(height) == expected


# --- Backend construction --------------------------------------------------------------------


def test_gemini_is_the_default_backend_at_its_own_defaults() -> None:
    """Nothing given means the backend's own defaults, not a copy of them kept here."""
    backend = pipeline.build_backend(_args())

    assert isinstance(backend, GeminiBackend)
    assert (backend.model_id, backend.fps, backend.resolution, backend.thinking_level, backend.seed) == (
        "gemini-3.8-flash",
        2.0,
        "low",
        "medium",
        1,
    )


def test_gemini_settings_are_passed_through() -> None:
    backend = pipeline.build_backend(_args("--fps", "4", "--resolution", "high", "--seed", "7"))

    assert isinstance(backend, GeminiBackend)
    assert (backend.fps, backend.resolution, backend.seed) == (4.0, "high", 7)


def test_local_needs_a_model_id_and_says_why() -> None:
    """`QwenBackend` has no default id because the right one differs per machine; nor has this."""
    with pytest.raises(ValueError, match="--vlm local needs --vlm-model"):
        pipeline.build_backend(_args("--vlm", "local"))


def test_local_builds_the_qwen_backend() -> None:
    backend = pipeline.build_backend(_args("--vlm", "local", "--vlm-model", "Qwen/Qwen3.5-9B", "--seed", "3"))

    assert isinstance(backend, QwenBackend)
    assert (backend.model_id, backend.seed, backend.max_new_tokens) == ("Qwen/Qwen3.5-9B", 3, 8192)


def test_a_gemini_setting_under_local_is_refused_rather_than_ignored() -> None:
    """Silently dropping it would look exactly like a run that honoured it."""
    with pytest.raises(ValueError, match=r"--thinking-level, --fps|--fps, --thinking-level"):
        pipeline.build_backend(_args("--vlm", "local", "--vlm-model", "x", "--fps", "4", "--thinking-level", "high"))


def test_a_local_setting_under_gemini_is_refused() -> None:
    with pytest.raises(ValueError, match="--max-new-tokens is a --vlm local setting"):
        pipeline.build_backend(_args("--max-new-tokens", "4096"))


# --- Windowing -------------------------------------------------------------------------------


def test_windowing_defaults_to_eight_seconds_every_four() -> None:
    assert pipeline.resolve_window(_args()) == (8.0, 4.0)


def test_whole_clip_asks_for_no_windows() -> None:
    assert pipeline.resolve_window(_args("--whole-clip")) is None


def test_whole_clip_with_a_windowing_flag_is_refused() -> None:
    with pytest.raises(ValueError, match="--window-s is a windowing setting"):
        pipeline.resolve_window(_args("--whole-clip", "--window-s", "6"))


def test_a_stride_longer_than_the_window_is_refused_before_anything_runs() -> None:
    """`find_interactions` refuses this too, but only after a clip has been tracked and rendered."""
    with pytest.raises(ValueError, match="would leave stretches of a clip unseen"):
        pipeline.resolve_window(_args("--window-s", "4", "--stride-s", "8"))


# --- A whole batch, with the stages stubbed ----------------------------------------------------


class _StubBackend(GeminiBackend):
    """A Gemini backend that calls nothing and remembers whether its uploads were cleaned up.

    A subclass rather than a separate object because `_delete_uploads` asks whether the backend is
    a `GeminiBackend` -- the local one uploads nothing and must not be asked.
    """

    def __init__(self) -> None:
        super().__init__(client=object())
        self.deleted = False

    def delete_uploads(self) -> list[str]:
        self.deleted = True
        return ["files/stub"]


def _run(**overrides: Any) -> InteractionRun:
    """A run record with one interaction in one window, as a windowed call would return."""
    window = WindowRun(window_index=0, start_s=0.0, end_s=8.0, prompt="")
    interaction = ReportedInteraction(
        person_ids=["P1"],
        vehicle_ids=["V1"],
        person_description="person in a red jacket",
        vehicle_description="white sedan",
        interaction="opens the driver-side door",
        start_time_s=1.0,
        evidence_time_s=2.0,
        end_time_s=3.0,
        confidence=0.9,
        window_index=0,
    )
    window.interactions = [interaction]
    fields: dict[str, Any] = {
        "source": "annotated.mp4",
        "clip_id": "clip__yolo26x__tracktrack__reid-none__imgsz640__buf3",
        "duration_s": 8.0,
        "prompt_version": "a1",
        "backend_config": {},
        "backend_slug": "3.8-flash__fps2__low__think-medium__seed1",
        "started_at": "2026-09-20T10:00:00.000+00:00",
        "window_s": 8.0,
        "stride_s": 4.0,
        "windows": [window],
        "interactions": [interaction],
    }
    return InteractionRun(**{**fields, **overrides})


@pytest.fixture
def stubbed_stages(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the four stages with stubs, and record what `find_interactions` was asked for."""
    calls: dict[str, Any] = {}

    def fake_track_video(source: Path, **kwargs: Any) -> Any:
        calls["imgsz"] = kwargs["imgsz"]
        return source

    def fake_relabel(tracks: Any) -> tuple[Any, dict[Any, str]]:
        return tracks, {(Category.PERSON, 1): "P1", (Category.VEHICLE, 1): "V1"}

    def fake_render(tracks: Any, output_path: Path, **_: Any) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"annotated")
        return output

    def fake_find(video: Path, backend: Any, **kwargs: Any) -> InteractionRun:
        calls["window"] = (kwargs.get("window_s"), kwargs.get("stride_s"))
        calls["annotated"] = Path(video)
        return _run() if kwargs.get("window_s") else _run(window_s=None, stride_s=None)

    # The labels the clip is taken to have drawn. Supporting the stub run's "P1"/"V1" by default,
    # so a test that cares about the unverified path is the one that empties this.
    calls["times"] = {"P1": [2.0], "V1": [2.0]}

    monkeypatch.setattr(pipeline, "label_times", lambda tracks: calls["times"])
    monkeypatch.setattr(pipeline, "probe_stream", lambda source: (640, 480, "30/1"))
    monkeypatch.setattr(pipeline, "track_video", fake_track_video)
    monkeypatch.setattr(pipeline, "relabel_tracks", fake_relabel)
    monkeypatch.setattr(pipeline, "render_tracked_video", fake_render)
    monkeypatch.setattr(pipeline, "config_slug", lambda tracks: "yolo26x__tracktrack__reid-none__imgsz640__buf3")
    monkeypatch.setattr(pipeline, "find_interactions", fake_find)
    return calls


def test_a_windowed_run_writes_the_annotated_clip_and_both_records(
    tmp_path: Path, stubbed_stages: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The naming is the contract anyone reading the output depends on."""
    monkeypatch.setattr(pipeline, "build_backend", lambda args: _StubBackend())
    clip = tmp_path / "clip.mp4"
    clip.touch()
    output = tmp_path / "out"

    assert pipeline.main([str(clip), str(output)]) == 0

    slug = "yolo26x__tracktrack__reid-none__imgsz640__buf3"
    tag = "3.8-flash__fps2__low__think-medium__seed1__w8s4__a1__20260920-100000-000"
    assert (output / "annotated" / f"clip__{slug}.mp4").is_file()
    record = output / "interactions" / f"clip__{slug}" / f"{tag}.json"
    merged = record.with_name(f"{tag}__merged.json")
    assert json.loads(record.read_text())["clip_id"] == f"clip__{slug}"
    assert json.loads(merged.read_text())["merged_interaction_count"] == 1
    # The model reads the annotated render, never the source.
    assert stubbed_stages["annotated"] == output / "annotated" / f"clip__{slug}.mp4"


def test_whole_clip_passes_no_windowing_and_writes_no_merged_file(
    tmp_path: Path, stubbed_stages: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "build_backend", lambda args: _StubBackend())
    clip = tmp_path / "clip.mp4"
    clip.touch()
    output = tmp_path / "out"

    assert pipeline.main([str(clip), str(output), "--whole-clip"]) == 0

    assert stubbed_stages["window"] == (None, None)
    assert not list((output / "interactions").rglob("*__merged.json"))


def test_one_clip_failing_costs_only_itself_and_the_exit_code(
    tmp_path: Path, stubbed_stages: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch must come back with an account of every clip, not a traceback from the first."""
    backend = _StubBackend()
    monkeypatch.setattr(pipeline, "build_backend", lambda args: backend)

    def explode_on_the_first(source: Path, **kwargs: Any) -> Any:
        if source.stem == "a":
            raise RuntimeError("could not open")
        return source

    monkeypatch.setattr(pipeline, "track_video", explode_on_the_first)
    for name in ("a.mp4", "b.mp4"):
        (tmp_path / name).touch()
    output = tmp_path / "out"

    assert pipeline.main([str(tmp_path), str(output)]) == 1

    assert not list(output.glob("annotated/a__*.mp4"))
    assert list(output.glob("annotated/b__*.mp4"))
    # The footage is on someone else's machine until this happens, and a raised stage is exactly
    # when it is easiest to skip.
    assert backend.deleted


def test_uploads_are_deleted_even_when_the_batch_cannot_finish(
    tmp_path: Path, stubbed_stages: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """KeyboardInterrupt is not an Exception, so it leaves the loop -- and must still clean up."""
    backend = _StubBackend()
    monkeypatch.setattr(pipeline, "build_backend", lambda args: backend)
    monkeypatch.setattr(pipeline, "run_pipeline", _raise_interrupt)
    clip = tmp_path / "clip.mp4"
    clip.touch()

    with pytest.raises(KeyboardInterrupt):
        pipeline.main([str(clip), str(tmp_path / "out")])

    assert backend.deleted


def _raise_interrupt(*_: Any, **__: Any) -> list[pipeline.ClipResult]:
    raise KeyboardInterrupt


def test_a_run_whose_calls_all_failed_is_a_failed_clip(
    tmp_path: Path, stubbed_stages: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The clip completed, but nothing was answered, and the exit code has to say so."""
    monkeypatch.setattr(pipeline, "build_backend", lambda args: _StubBackend())
    monkeypatch.setattr(
        pipeline, "find_interactions", lambda video, backend, **kwargs: _run(error="all 1 call(s) failed")
    )
    clip = tmp_path / "clip.mp4"
    clip.touch()

    assert pipeline.main([str(clip), str(tmp_path / "out")]) == 1


def test_a_bad_argument_exits_before_any_clip_is_touched(tmp_path: Path) -> None:
    """Exit 2, argparse's own code for a bad command line, and nothing tracked."""
    clip = tmp_path / "clip.mp4"
    clip.touch()

    with pytest.raises(SystemExit) as exit_info:
        pipeline.main([str(clip), str(tmp_path / "out"), "--window-s", "4", "--stride-s", "8"])

    assert exit_info.value.code == 2


def test_labels_the_clip_does_not_support_are_reported_on_stderr(
    tmp_path: Path, stubbed_stages: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run whose ids were invented still succeeds; the warning is how anyone finds out."""
    monkeypatch.setattr(pipeline, "build_backend", lambda args: _StubBackend())
    stubbed_stages["times"] = {}  # the tracker drew nothing, so no reported label can hold up
    clip = tmp_path / "clip.mp4"
    clip.touch()
    output = tmp_path / "out"

    assert pipeline.main([str(clip), str(output)]) == 0

    assert "2 unsupported label(s) in 1 record(s): P1 x1, V1 x1" in capsys.readouterr().err
    # Quarantined, not deleted: the record still says what the model claimed.
    record = json.loads(next((output / "interactions").rglob("*[!d].json")).read_text())
    assert record["interactions"][0]["person_ids"] == []
    assert record["interactions"][0]["unverified_person_ids"] == ["P1"]


def test_nothing_is_said_when_every_label_holds_up(
    tmp_path: Path, stubbed_stages: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(pipeline, "build_backend", lambda args: _StubBackend())
    clip = tmp_path / "clip.mp4"
    clip.touch()

    assert pipeline.main([str(clip), str(tmp_path / "out")]) == 0

    assert "unsupported" not in capsys.readouterr().err
