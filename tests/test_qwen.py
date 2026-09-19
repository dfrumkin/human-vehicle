"""Tests for the local Qwen backend, against stub runtimes.

Nothing here downloads weights or runs inference. The backend is exercised with a stand-in runtime,
and each adapter with stand-in library calls -- which is what lets both adapters be covered on a
machine where only one of their libraries is installed.

The bias is towards failures that are *silent*: a time shifted by the wrong amount, a video the
model never saw, a prompt that quietly re-enabled thinking. Each of those produces a confident wrong
answer rather than an error, so a test is the only thing that would notice.
"""

import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from human_vehicle.interactions import ClipInteractions, find_interactions, probe_duration
from human_vehicle.vlm import (
    LOCAL_FPS,
    PIXELS_PER_FRAME,
    TEMPORAL_PATCH_SIZE,
    MlxRuntime,
    QwenBackend,
    TorchRuntime,
    repair_json,
    shift_times,
    trim_segment,
)
from tests.conftest import MakeVideo

SCHEMA: dict[str, Any] = {"type": "object", "properties": {"interactions": {"type": "array"}}}


def _interaction(**overrides: Any) -> dict[str, Any]:
    """One well-formed interaction, as a model on a segment's own clock would report it."""
    return {
        "person_ids": ["P1"],
        "vehicle_ids": ["V1"],
        "person_description": "person in a red jacket",
        "vehicle_description": "white sedan",
        "interaction": "opens the driver-side door",
        "start_time_s": 1.0,
        "evidence_time_s": 2.0,
        "end_time_s": 3.0,
        "confidence": 0.9,
        **overrides,
    }


class _StubRuntime:
    """Records what the backend asked of it and returns a canned answer."""

    name = "stub"

    def __init__(self, text: str = "", *, error: Exception | None = None) -> None:
        self._text = text
        self._error = error
        self.calls: list[dict[str, Any]] = []
        self.videos: list[Path] = []

    @property
    def version(self) -> str | None:
        return "0.0.0"

    @property
    def device(self) -> str | None:
        return "stub-device"

    def generate(
        self, video: Path, prompt: str, *, frames: int, max_new_tokens: int, seed: int
    ) -> tuple[str, dict[str, float]]:
        if self._error is not None:
            raise self._error
        # The path is recorded *and* the bytes are read now: a temporary segment is deleted when
        # generate returns, so a test that looked afterwards would find nothing.
        self.videos.append(video)
        self.calls.append(
            {
                "video": video,
                "exists": video.is_file(),
                "duration": probe_duration(video) if video.is_file() else None,
                "prompt": prompt,
                "frames": frames,
                "max_new_tokens": max_new_tokens,
                "seed": seed,
            }
        )
        return self._text, {"total_input_tokens": 10.0, "total_output_tokens": 5.0}


@pytest.fixture
def clip(make_video: MakeVideo) -> Path:
    """A real, decodable clip: the trimming tests need one, and ffmpeg is a prerequisite anyway.

    Twenty seconds at 6 fps, which is long enough to hold a window well away from the clip start --
    the only way to tell a correct trim from one that returned the beginning of the file.
    """
    return make_video("clip.mp4", width=64, height=48, fps="6", frames=120)


# --- What the model is shown ------------------------------------------------------------------


def test_a_window_is_trimmed_and_a_whole_clip_is_not(clip: Path) -> None:
    """A window becomes its own short file; a whole-clip call hands over the source untouched."""
    runtime = _StubRuntime(json.dumps({"interactions": []}))
    backend = QwenBackend("Qwen/Qwen3.5-9B", runtime=runtime)

    backend.generate(clip, "prompt", window=None, schema=SCHEMA)
    assert runtime.calls[0]["video"] == clip

    backend.generate(clip, "prompt", window=(2.0, 6.0), schema=SCHEMA)
    assert runtime.calls[1]["video"] != clip
    assert runtime.calls[1]["exists"], "the segment must exist while the runtime is reading it"
    assert runtime.calls[1]["duration"] == pytest.approx(4.0, abs=0.2)


def test_the_segment_is_removed_even_when_the_runtime_fails(clip: Path) -> None:
    """A failed call must not leave the trimmed file behind."""
    runtime = _StubRuntime(error=RuntimeError("out of memory"))
    backend = QwenBackend("Qwen/Qwen3.5-9B", runtime=runtime)

    with pytest.raises(RuntimeError):
        backend.generate(clip, "prompt", window=(2.0, 6.0), schema=SCHEMA)


def test_the_shape_reaches_the_prompt_as_an_example_not_a_schema(clip: Path) -> None:
    """Nothing constrains a local model's output, so the shape has to be in the words.

    It must not be the JSON Schema itself. Handed 2 KB of `$defs`, Qwen3.5-9B answered with a
    tidied copy of the schema -- valid JSON, no `interactions` key, the whole call recorded as a
    response error. This is that regression: the prompt carries the field names and a worked
    example, and no schema machinery for the model to mistake for the answer.
    """
    real = ClipInteractions.model_json_schema()
    runtime = _StubRuntime(json.dumps({"interactions": []}))
    QwenBackend("Qwen/Qwen3.5-9B", runtime=runtime).generate(clip, "find them", window=None, schema=real)

    asked = runtime.calls[0]["prompt"]
    assert "find them" in asked
    assert "$defs" not in asked, "the schema's own machinery is what the model copied back"
    for field in ("person_ids", "vehicle_ids", "start_time_s", "confidence"):
        assert field in asked
    assert '{"interactions": [{' in asked, "an example of the answer, not a description of it"


def test_an_unrecognised_schema_falls_back_to_the_raw_schema(clip: Path) -> None:
    """Worse than an example, but never wrong: a shape this cannot read still reaches the model."""
    odd: dict[str, Any] = {"type": "object", "properties": {"whatever": {"type": "string"}}}
    runtime = _StubRuntime(json.dumps({"interactions": []}))
    QwenBackend("Qwen/Qwen3.5-9B", runtime=runtime).generate(clip, "find them", window=None, schema=odd)

    asked = runtime.calls[0]["prompt"]
    assert json.loads(asked[asked.index("{") :]) == odd


def test_frames_follow_the_segment_not_the_clip(clip: Path) -> None:
    """The frame count sizes the MLX pixel budget, so it must describe what was actually sent."""
    runtime = _StubRuntime(json.dumps({"interactions": []}))
    backend = QwenBackend("Qwen/Qwen3.5-9B", runtime=runtime)

    backend.generate(clip, "prompt", window=None, schema=SCHEMA)
    backend.generate(clip, "prompt", window=(2.0, 6.0), schema=SCHEMA)

    assert runtime.calls[0]["frames"] == round(probe_duration(clip) * LOCAL_FPS)
    assert runtime.calls[1]["frames"] == round(4.0 * LOCAL_FPS)


# --- Trimming ---------------------------------------------------------------------------------


def _frame_at(video: Path, seconds: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
    decoded, frame = capture.read()
    capture.release()
    assert decoded, f"could not read {video} at {seconds}s"
    return frame


def _difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left.astype(np.int16) - right.astype(np.int16))))


def test_a_trimmed_segment_starts_at_the_window(clip: Path, tmp_path: Path) -> None:
    """The segment's t=0 must *be* the window start, not merely its length be right.

    Duration alone would pass a cut that returned the first N seconds of the clip -- which is the
    likeliest way to get this wrong, and would put a constant offset into every corrected time in
    every windowed run. So this compares pictures: `make_video` builds from ffmpeg's `testsrc`,
    whose frame changes continuously, and the second assertion is the one that fails on a cut that
    ignored `-ss`.
    """
    segment = trim_segment(clip, (4.0, 8.0), tmp_path / "segment.mp4")

    assert probe_duration(segment) == pytest.approx(4.0, abs=0.2)
    first = _frame_at(segment, 0.0)
    assert _difference(first, _frame_at(clip, 4.0)) < 12.0, "segment should begin at the window start"
    assert _difference(first, _frame_at(clip, 0.0)) > 12.0, "segment must not begin at the clip start"


def test_a_trim_of_the_wrong_length_is_rejected(clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad cut fails the call rather than shifting every time in the run."""
    real = subprocess.run

    def _short(args: Any, **kwargs: Any) -> Any:
        # Take two seconds however many were asked for.
        patched = [("2.0" if index and args[index - 1] == "-t" else value) for index, value in enumerate(args)]
        return real(patched, **kwargs)

    monkeypatch.setattr(subprocess, "run", _short)
    with pytest.raises(RuntimeError, match="every corrected timestamp would be wrong"):
        trim_segment(clip, (0.0, 6.0), tmp_path / "segment.mp4")


# --- Putting the times back -------------------------------------------------------------------


def test_window_times_are_shifted_onto_the_clips_clock(clip: Path) -> None:
    """The model answers about the segment; what comes back must be on the clip's clock."""
    runtime = _StubRuntime(json.dumps({"interactions": [_interaction()]}))
    response = QwenBackend("Qwen/Qwen3.5-9B", runtime=runtime).generate(
        clip, "prompt", window=(12.0, 18.0), schema=SCHEMA
    )

    reported = json.loads(response.text)["interactions"][0]
    assert (reported["start_time_s"], reported["evidence_time_s"], reported["end_time_s"]) == (13.0, 14.0, 15.0)


def test_a_whole_clip_call_shifts_nothing(clip: Path) -> None:
    runtime = _StubRuntime(json.dumps({"interactions": [_interaction()]}))
    response = QwenBackend("Qwen/Qwen3.5-9B", runtime=runtime).generate(clip, "prompt", window=None, schema=SCHEMA)

    assert json.loads(response.text)["interactions"][0]["start_time_s"] == 1.0


def test_shifted_times_are_not_clamped() -> None:
    """A time past the segment stays past it; the validator decides, not the backend."""
    payload = {"interactions": [_interaction(end_time_s=6.4)]}
    assert shift_times(payload, 12.0)["interactions"][0]["end_time_s"] == pytest.approx(18.4)


@pytest.mark.parametrize(
    "payload",
    [
        "not an object",
        {"no_interactions_key": True},
        {"interactions": "not a list"},
        {"interactions": ["not an object"]},
        {"interactions": [{"start_time_s": "soon"}]},
    ],
)
def test_the_shifter_never_raises(payload: Any) -> None:
    """Whatever shape arrives, it passes through for the validator to reject.

    `find_interactions` records `raw_text` and `usage` only after `generate` returns, so a shifter
    that raised would turn the model's malformed output into a backend failure and lose both.
    """
    shift_times(payload, 12.0)


def test_a_non_numeric_time_survives_to_the_validator(clip: Path) -> None:
    runtime = _StubRuntime(json.dumps({"interactions": [_interaction(start_time_s="soon")]}))
    response = QwenBackend("Qwen/Qwen3.5-9B", runtime=runtime).generate(
        clip, "prompt", window=(12.0, 18.0), schema=SCHEMA
    )

    reported = json.loads(response.text)["interactions"][0]
    assert reported["start_time_s"] == "soon"
    assert reported["end_time_s"] == 15.0, "the numeric siblings still move"


# --- Repairing what an unconstrained model says -----------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"interactions": []}', {"interactions": []}),
        ('```json\n{"interactions": []}\n```', {"interactions": []}),
        ('```\n{"interactions": []}\n```', {"interactions": []}),
        ('Here is the answer:\n{"interactions": []}', {"interactions": []}),
    ],
)
def test_repair_makes_an_answer_parseable(raw: str, expected: Any) -> None:
    text, payload = repair_json(raw)
    assert payload == expected
    assert json.loads(text) == expected


def test_prose_with_braces_is_left_to_fail() -> None:
    """Step 3's documented limit: the slice spans both and does not parse.

    Recorded as malformed with the model's own words rather than rescued by a scanner.
    """
    text, payload = repair_json('I considered {a, b} and concluded:\n{"interactions": []}')
    assert payload is None
    assert text.startswith("I considered")


def test_a_repaired_answer_is_what_gets_shifted(clip: Path) -> None:
    """Repair and the shift compose: a fenced answer still comes back on the clip's clock."""
    runtime = _StubRuntime(f"```json\n{json.dumps({'interactions': [_interaction()]})}\n```")
    response = QwenBackend("Qwen/Qwen3.5-9B", runtime=runtime).generate(
        clip, "prompt", window=(12.0, 18.0), schema=SCHEMA
    )

    assert json.loads(response.text)["interactions"][0]["start_time_s"] == 13.0


# --- What the backend says about itself --------------------------------------------------------


def test_a_model_id_is_required() -> None:
    with pytest.raises(ValueError, match="model_id is required"):
        QwenBackend("")


def test_max_new_tokens_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        QwenBackend("Qwen/Qwen3.5-9B", max_new_tokens=0, runtime=_StubRuntime())


def test_tolerance_is_one_temporal_patch() -> None:
    """Two frames are one moment to this model, so the slack is coarser than the sampling rate."""
    backend = QwenBackend("Qwen/Qwen3.5-9B", runtime=_StubRuntime())
    assert backend.tolerance_s == TEMPORAL_PATCH_SIZE / LOCAL_FPS == 1.0


def test_the_slug_keeps_the_model_variant() -> None:
    """9B and 27B must not name the same run: the slug is how a record says which model ran."""
    nine = QwenBackend("Qwen/Qwen3.5-9B", runtime=_StubRuntime()).slug
    twenty_seven = QwenBackend("mlx-community/Qwen3.5-27B-4bit", runtime=_StubRuntime()).slug

    assert nine.startswith("Qwen3.5-9B")
    assert twenty_seven.startswith("Qwen3.5-27B-4bit")
    assert nine != twenty_seven


def test_config_records_what_ran_and_what_it_did() -> None:
    config = QwenBackend("Qwen/Qwen3.5-9B", runtime=_StubRuntime()).config

    assert config["model_id"] == "Qwen/Qwen3.5-9B"
    assert config["runtime"] == "stub"
    assert config["fps"] == LOCAL_FPS
    # A saved record should say for itself which caveats apply to it.
    assert config["schema_in_prompt"] and config["response_repaired"] and config["times_shifted"]


def test_config_and_slug_survive_a_runtime_that_cannot_load() -> None:
    """`find_interactions` reads both outside the try that catches a failing generate.

    A config that raised -- or that forced a load to answer -- would turn one recorded window
    failure into a lost run.
    """

    class _Unloadable:
        name = "torch"

        @property
        def version(self) -> str | None:
            return None

        @property
        def device(self) -> str | None:
            return None

        def generate(self, *args: Any, **kwargs: Any) -> tuple[str, dict[str, float]]:
            raise ImportError("transformers is not installed")

    backend = QwenBackend("Qwen/Qwen3.5-9B", runtime=_Unloadable())
    assert backend.config["runtime_version"] is None
    assert backend.config["device"] is None
    assert backend.slug


def test_usage_totals_alongside_a_gemini_run(clip: Path) -> None:
    """A local call costs nothing per token, and the zero keeps mixed totals meaningful."""
    runtime = _StubRuntime(json.dumps({"interactions": []}))
    response = QwenBackend("Qwen/Qwen3.5-9B", runtime=runtime).generate(clip, "p", window=None, schema=SCHEMA)

    assert response.usage["estimated_cost_usd"] == 0.0
    assert response.usage["total_input_tokens"] == 10.0


# --- Driven by find_interactions ---------------------------------------------------------------


def test_a_windowed_run_is_told_the_segments_duration(clip: Path) -> None:
    """The prompt must describe the window as a clip in its own right, not as part of a longer one."""
    runtime = _StubRuntime(json.dumps({"interactions": []}))
    backend = QwenBackend("Qwen/Qwen3.5-9B", runtime=runtime)

    find_interactions(clip, backend, window_s=4.0, stride_s=4.0, clip_id="c")

    asked = runtime.calls[0]["prompt"]
    assert "The clip is 4.0 seconds long." in asked
    assert "segment from" not in asked, "a corrected backend must not also be asked to add an offset"


def test_a_runtime_that_cannot_load_is_recorded_not_raised(clip: Path) -> None:
    """One bad window must not cost the run its record."""

    class _Unloadable:
        name = "torch"
        version = None
        device = None

        def generate(self, *args: Any, **kwargs: Any) -> tuple[str, dict[str, float]]:
            raise ImportError("transformers is not installed")

    run = find_interactions(clip, QwenBackend("Qwen/Qwen3.5-9B", runtime=_Unloadable()), clip_id="c")

    assert run.error is not None
    assert "ImportError" in (run.windows[0].error or "")
    assert run.backend_config["model_id"] == "Qwen/Qwen3.5-9B"
    assert run.backend_slug


def test_schema_invalid_json_is_recorded_as_the_models_output(clip: Path) -> None:
    """Parseable but wrong-shaped output is the model's failure, not the backend's.

    If the shifter raised on it, the window would be recorded as a backend error and the text and
    usage that were paid for would be gone.
    """
    runtime = _StubRuntime(json.dumps({"something_else": []}))
    run = find_interactions(clip, QwenBackend("Qwen/Qwen3.5-9B", runtime=runtime), clip_id="c")

    assert run.windows[0].raw_text == json.dumps({"something_else": []})
    assert run.windows[0].usage["total_input_tokens"] == 10.0
    assert "no 'interactions' list" in (run.windows[0].error or "")


def test_shifted_times_pass_the_windowed_validator(clip: Path) -> None:
    """End to end: a segment-local answer arrives as an accepted, clip-global interaction."""
    runtime = _StubRuntime(json.dumps({"interactions": [_interaction()]}))
    run = find_interactions(
        clip, QwenBackend("Qwen/Qwen3.5-9B", runtime=runtime), window_s=4.0, stride_s=4.0, clip_id="c"
    )

    assert not run.malformed
    first = run.interactions[0]
    assert (first.start_time_s, first.end_time_s) == (1.0, 3.0)
    later = [item for item in run.interactions if item.window_index > 0]
    assert later and later[0].start_time_s > 3.0, "a later window's times must be offset"


# --- The adapters' own calls -------------------------------------------------------------------
#
# These assert what each adapter hands its library, with the library faked. Neither test imports
# mlx_vlm or transformers, so both adapters are covered wherever the suite runs -- which matters
# because only one of them is installable on any given machine, and the other is the one nobody can
# check by hand.


class _Recorder:
    """Captures the kwargs of whatever it stands in for."""

    def __init__(self, result: Any = None) -> None:
        self.result = result
        self.kwargs: dict[str, Any] = {}
        self.args: tuple[Any, ...] = ()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.args, self.kwargs = args, kwargs
        return self.result


class _MlxResult:
    text = '{"interactions": []}'
    prompt_tokens = 120
    generation_tokens = 30


def _mlx_runtime() -> tuple[MlxRuntime, _Recorder, _Recorder, _Recorder]:
    from human_vehicle.vlm import MlxEntryPoints

    template, generate, seed = _Recorder("<formatted>"), _Recorder(_MlxResult()), _Recorder()
    runtime = MlxRuntime(
        "mlx-community/Qwen3.5-9B-4bit",
        entry_points=MlxEntryPoints(
            load=lambda model_id: ("model", "processor"),
            generate=generate,
            apply_chat_template=template,
            load_config=lambda model_id: {"model_type": "qwen3_5"},
            seed=seed,
        ),
    )
    return runtime, template, generate, seed


def test_the_mlx_adapter_switches_thinking_off_and_decodes_greedily(tmp_path: Path) -> None:
    """Thinking would eat the token budget before the JSON was written; sampling would break repeats.

    Qwen ships sampling defaults in its own generation config, so declining to pass anything is not
    the same as asking for greedy.
    """
    runtime, template, generate, seed = _mlx_runtime()
    runtime.generate(tmp_path / "seg.mp4", "prompt", frames=12, max_new_tokens=512, seed=7)

    assert template.kwargs["enable_thinking"] is False
    assert generate.kwargs["enable_thinking"] is False
    assert generate.kwargs["temperature"] == 0.0
    assert generate.kwargs["seed"] == 7
    assert seed.args == (7,)


def test_the_mlx_adapter_sizes_the_whole_video_pixel_budget(tmp_path: Path) -> None:
    """mlx's `max_pixels` covers the whole video, so a per-frame figure would shrink every frame."""
    runtime, template, _, _ = _mlx_runtime()
    runtime.generate(tmp_path / "seg.mp4", "prompt", frames=12, max_new_tokens=512, seed=1)

    assert template.kwargs["max_pixels"] == PIXELS_PER_FRAME * 12
    assert template.kwargs["fps"] == LOCAL_FPS


def test_the_mlx_adapter_sends_the_video(tmp_path: Path) -> None:
    """The model is shown video, not stills -- on both the template and the generate call."""
    runtime, template, generate, _ = _mlx_runtime()
    segment = tmp_path / "seg.mp4"
    _, usage = runtime.generate(segment, "prompt", frames=12, max_new_tokens=512, seed=1)

    assert template.kwargs["video"] == str(segment)
    assert generate.kwargs["video"] == str(segment)
    assert template.kwargs["num_images"] == 0
    assert usage["total_input_tokens"] == 120.0


class _FakeTensor:
    def __init__(self, rows: int, columns: int) -> None:
        self.shape = (rows, columns)


class _FakeInputs(dict[str, Any]):
    def __init__(self, prompt_length: int) -> None:
        super().__init__({"input_ids": _FakeTensor(1, prompt_length)})
        self.moved_to: Any = None

    def to(self, device: Any) -> "_FakeInputs":
        self.moved_to = device
        return self


class _FakeProcessor:
    def __init__(self, prompt_length: int = 100) -> None:
        self.template = _Recorder(_FakeInputs(prompt_length))
        self.decoded: Any = None

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        return self.template(*args, **kwargs)

    def decode(self, tokens: Any, **kwargs: Any) -> str:
        self.decoded = tokens
        return '{"interactions": []}'


def _torch_runtime(prompt_length: int = 100, total_length: int = 130) -> tuple[TorchRuntime, Any, Any, Any]:
    from human_vehicle.vlm import TorchEntryPoints

    processor = _FakeProcessor(prompt_length)
    generate = _Recorder([list(range(total_length))])

    class _FakeModel:
        def generate(self, **kwargs: Any) -> Any:
            return generate(**kwargs)

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class _FakeTorch:
        cuda = _FakeCuda()
        bfloat16 = "bfloat16"
        manual_seed = _Recorder()
        use_deterministic_algorithms = _Recorder()

    load_video = _Recorder((np.zeros((12, 8, 8, 3), dtype=np.uint8), "source-metadata"))
    metadata = _Recorder("decoded-metadata")
    runtime = TorchRuntime(
        "Qwen/Qwen3.5-9B",
        entry_points=TorchEntryPoints(
            torch=_FakeTorch(),
            load_model=lambda model_id, **kwargs: _FakeModel(),
            load_processor=lambda model_id: processor,
            load_video=load_video,
            video_metadata=metadata,
        ),
    )
    return runtime, processor, load_video, metadata


def test_the_torch_adapter_switches_thinking_off_and_decodes_greedily(tmp_path: Path) -> None:
    runtime, processor, _, _ = _torch_runtime()
    runtime.generate(tmp_path / "seg.mp4", "prompt", frames=12, max_new_tokens=512, seed=7)

    assert processor.template.kwargs["enable_thinking"] is False
    assert runtime._entry_points.torch.manual_seed.args == (7,)  # pyright: ignore[reportPrivateUsage]


def test_the_torch_adapter_decodes_only_the_completion(tmp_path: Path) -> None:
    """`generate` returns prompt and completion concatenated, and the prompt holds the schema.

    Decoding the whole sequence would let the brace-slice repair pick the schema out of the prompt
    and return it as the model's answer: valid JSON, entirely the wrong content.
    """
    runtime, processor, _, _ = _torch_runtime(prompt_length=100, total_length=130)
    _, usage = runtime.generate(tmp_path / "seg.mp4", "prompt", frames=12, max_new_tokens=512, seed=1)

    assert processor.decoded == list(range(100, 130)), "only the generated tokens may be decoded"
    assert usage["total_input_tokens"] == 100.0
    assert usage["total_output_tokens"] == 30.0


def test_the_torch_adapter_decodes_video_itself_with_opencv(tmp_path: Path) -> None:
    """The processor's own fetching hardcodes torchcodec, which this project does not declare.

    What reaches the processor is still a video -- decoded frames with their temporal patching
    intact -- and the metadata must describe *that* array, not the source, or the processor samples
    again from the wrong rate and indexes past the end.
    """
    runtime, processor, load_video, metadata = _torch_runtime()
    runtime.generate(tmp_path / "seg.mp4", "prompt", frames=12, max_new_tokens=512, seed=1)

    assert load_video.kwargs["backend"] == "opencv"
    assert load_video.kwargs["fps"] == LOCAL_FPS
    assert metadata.kwargs["total_num_frames"] == 12
    assert metadata.kwargs["fps"] == LOCAL_FPS
    assert processor.template.kwargs["video_metadata"] == ["decoded-metadata"]
    # The per-frame cap the reference implementation applies; without it a long video costs far
    # more tokens than it should.
    assert processor.template.kwargs["cap_pixels_per_frame"] is True

    content = processor.template.args[0][0]["content"]
    assert [item["type"] for item in content] == ["video", "text"]
