"""The vision-language model this project asks its questions of, and the seam it sits behind.

The question -- the prompt, the schema, the validation, the windowing -- belongs to the caller, and
a caller reaches a model only through `VlmBackend`. This module owns the model: its knobs, and
whatever that particular model needs in order to answer in the schema it is handed. A backend whose
runtime cannot constrain output has to put the shape into the prompt itself, which is why the schema
description and the JSON repair live here. Another model means another class here satisfying the
protocol.

There are two implementations. `GeminiBackend` is written against the `interactions.create` API and
records the installed `google-genai` version in its `config`. `QwenBackend` runs Qwen3.5 locally,
through MLX on Apple Silicon and through transformers elsewhere; which one is installed is settled
in `pyproject.toml` by platform marker, so nothing here probes for a runtime.

Both are shown **video**, never a sequence of stills, sampled at 2 frames a second.
"""

import json
import os
import platform
import re
import subprocess
import tempfile
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

from google import genai

from human_vehicle.device import select_device
from human_vehicle.video import probe_duration, require_binary

# A window of the clip, in seconds on the clip's own clock: (start, end).
Window = tuple[float, float]


@dataclass(frozen=True, slots=True)
class VlmResponse:
    """What one call returned: the model's text, and what it cost.

    `usage` is flat and numeric -- token counts, an estimated price -- so that totalling several
    windows is a plain sum with no per-provider knowledge. A provider's own breakdown (Gemini
    reports tokens per input modality) is not carried; the totals it adds up to are.
    """

    text: str
    usage: Mapping[str, float]


class VlmBackend(Protocol):
    """A vision-language model that can be shown a clip, or a segment of one, and asked a question.

    The protocol is deliberately small: the prompt and the schema are handed in, so a second
    implementation supplies a model and nothing else.
    """

    @property
    def tolerance_s(self) -> float:
        """How finely this backend can place a moment in time.

        The windowed validator allows this much slack at a window's edges. It belongs to the backend
        because only the backend knows how finely its model resolves time.

        For a backend whose model sees every sampled frame separately this is the sampling interval.
        Where the model folds frames together -- Qwen's vision encoder pools two consecutive frames
        into one temporal position -- it is correspondingly coarser, and reporting the sampling rate
        instead would claim a precision the model does not have.
        """
        ...

    @property
    def segment_local_prompt(self) -> bool:
        """Whether a window's prompt should be written on the segment's own clock.

        False -- the usual case -- means the model is told it is seeing part of a longer clip and
        asked for times on that clip's clock, which is what a window's prompt normally says.

        True means the opposite: the caller should describe the window as if it were a clip in its
        own right, because this backend shows the model exactly that. **It does not change
        what `generate` returns.** Such a backend corrects the times itself and still hands back
        clip-global ones, so validation is unaffected and needs no matching branch. Reading this as
        "reports segment-local times" and adding one would double-shift every window.
        """
        ...

    @property
    def config(self) -> Mapping[str, Any]:
        """What ran, for the run record. JSON scalars only: the record is written to disk."""
        ...

    @property
    def slug(self) -> str:
        """A filename fragment naming that configuration, as `tracking.config_slug` does."""
        ...

    def generate(self, video: Path, prompt: str, *, window: Window | None, schema: dict[str, Any]) -> VlmResponse:
        """Answer `prompt` about `video`, or about `window` of it, as JSON.

        `schema` is the shape the answer must take, and `text` must come back as JSON and nothing
        else -- the caller hands it straight to `json.loads`, with no fence stripping and no brace
        matching.

        A backend whose runtime can constrain output to `schema` gets that for free. One that cannot
        -- an unconstrained local model will happily wrap its answer in prose or a ```json fence --
        **must strip that itself before returning**. The failure otherwise is not a few malformed
        records: `json.loads` raises, and the whole call is recorded as an error, losing every good
        record in it. The stripping belongs here, with the model whose habit it is; putting it in
        the shared parser would also let a constrained backend's regression pass unnoticed.

        Times in the answer are expected on the **clip's** clock even when `window` is set -- the
        prompt says so, and the caller validates against it.
        """
        ...


# Price per 1M tokens, (input, output), paid tier, through 2026-12-31. Thinking bills as output.
# Keyed by model, because a cost recorded at another model's rates is not an estimate but a wrong
# number: `GeminiBackend` refuses an id that is not in here rather than write one.
PRICES: Mapping[str, tuple[float, float]] = {"gemini-3.8-flash": (0.75, 3.75)}

DEFAULT_MODEL_ID = "gemini-3.8-flash"


def sum_usage(usages: list[Mapping[str, float]]) -> dict[str, float]:
    """Add up the usage of several calls, key by key.

    Works for any backend because a `VlmResponse.usage` is flat and numeric by contract. A key
    missing from one call counts as zero there, so a failed call contributes nothing rather than
    removing the key from the total.
    """
    keys = {key for usage in usages for key in usage}
    return {key: sum(usage.get(key, 0.0) for usage in usages) for key in sorted(keys)}


class GeminiBackend:
    """Gemini, through the File API and `interactions.create`.

    The clip is uploaded once and reused for every call, which is the reason to use the File API at
    all: a windowed run is one upload and seven requests, where inline data would re-transmit the
    whole clip seven times.

    **Uploads outlive the call.** Whoever holds a backend owns what it has uploaded until they call
    `delete_uploads`; Google removes them on its own after 48 hours. This is not a context manager
    on purpose -- the point of the upload is to be reused across calls and across a whole session of
    prompt iteration, and a backend that deleted on leaving a scope would re-upload every run.

    `store=False` goes on every request. The API's own default retains the prompt, the media
    reference and the output -- which describe the people in the footage -- for 55 days.
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        fps: float = 2.0,
        resolution: str = "low",
        thinking_level: str = "medium",
        seed: int = 1,
        max_output_tokens: int = 8192,
        client: Any = None,
    ) -> None:
        """The sampling knobs are Gemini's own: set here, never part of the question being asked.

        `fps` defaults to 2.0 rather than the documented 1.0: a 9-second clip sampled nine times can
        miss a two-second interaction between frames. `resolution="low"` is 70 tokens a frame and is
        identical to "medium" for video.

        `client` is for tests. Left None, a client is built from `GEMINI_API_KEY` on first use --
        lazily, so constructing a backend needs no credential and no network.

        An unpriced `model_id` is refused here, before anything is spent: `usage` carries an
        estimated cost into the run record, and a cost worked out at some other model's rates would
        read as real. Adding a model means adding its rates to `PRICES`.
        """
        if model_id not in PRICES:
            raise ValueError(f"no price recorded for {model_id!r}; add its rates to vlm.PRICES")

        self.model_id = model_id
        self.fps = fps
        self.resolution = resolution
        self.thinking_level = thinking_level
        self.seed = seed
        self.max_output_tokens = max_output_tokens
        self._client = client
        self._uploads: dict[str, Any] = {}

    @property
    def client(self) -> Any:
        if self._client is None:
            if not os.environ.get("GEMINI_API_KEY"):
                raise RuntimeError("GEMINI_API_KEY is not set; put it in the repository's .env file")
            self._client = genai.Client()
        return self._client

    @property
    def tolerance_s(self) -> float:
        """One sampling interval: the finest this model can localise a moment at `fps`."""
        return 1.0 / self.fps

    @property
    def segment_local_prompt(self) -> bool:
        """False: Gemini is shown the whole upload with offsets, and answers on the clip's clock."""
        return False

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "backend": "gemini",
            "model_id": self.model_id,
            "fps": self.fps,
            "resolution": self.resolution,
            "thinking_level": self.thinking_level,
            "seed": self.seed,
            "max_output_tokens": self.max_output_tokens,
            # This backend is written against the interactions.create shape, which is new enough
            # that a later SDK could move it. A record should say what it ran against.
            "sdk_version": version("google-genai"),
        }

    @property
    def slug(self) -> str:
        model_short = self.model_id.replace("gemini-", "")
        return f"{model_short}__fps{self.fps:g}__{self.resolution}__think-{self.thinking_level}__seed{self.seed}"

    def upload(self, video: Path, *, timeout_s: float = 180.0, poll_s: float = 2.0) -> Any:
        """Upload `video` once per backend and reuse the handle.

        Polling is bounded and checks for failure. Waiting only for ACTIVE hangs forever on an
        upload that will never reach it, which looks like a call that simply never returns -- the
        one failure mode hardest to notice.
        """
        key = str(video)
        if key in self._uploads:
            return self._uploads[key]

        handle = self.client.files.upload(file=str(video))
        deadline = time.monotonic() + timeout_s
        while True:
            state = getattr(handle.state, "name", str(handle.state))
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise RuntimeError(f"upload failed for {video.name}: {getattr(handle, 'error', None)}")
            if state != "PROCESSING":
                raise RuntimeError(f"unexpected file state {state!r} for {video.name}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"{video.name} still {state} after {timeout_s:.0f}s; giving up")
            time.sleep(poll_s)
            handle = self.client.files.get(name=handle.name)

        self._uploads[key] = handle
        return handle

    def delete_uploads(self) -> list[str]:
        """Remove everything this backend uploaded, and return what was deleted.

        Failures are swallowed deliberately: a file already gone or expired is the expected case at
        cleanup time, and it must not stop the rest from being removed.
        """
        deleted: list[str] = []
        for key, handle in list(self._uploads.items()):
            try:
                self.client.files.delete(name=handle.name)
                deleted.append(handle.name)
            except Exception:  # already gone, expired, or unreachable
                pass
            self._uploads.pop(key, None)
        return deleted

    def generate(self, video: Path, prompt: str, *, window: Window | None, schema: dict[str, Any]) -> VlmResponse:
        """One `interactions.create` call, with output constrained to `schema`."""
        handle = self.upload(video)
        processing: dict[str, Any] = {"type": "static", "fps": self.fps}
        if window is not None:
            # Offsets live inside `processing`, alongside fps, and are decimal seconds with an "s"
            # suffix -- strings, not numbers: `StaticMediaProcessingParam` declares both as `str`.
            # Getting this wrong sends the whole clip for every window, which is a
            # plausible-looking, wholly wrong run rather than a visible failure.
            processing["start_offset"] = f"{window[0]:g}s"
            processing["end_offset"] = f"{window[1]:g}s"

        request: dict[str, Any] = {
            "model": self.model_id,
            "input": [
                {
                    "type": "video",
                    "uri": handle.uri,
                    "mime_type": handle.mime_type,
                    "processing": processing,
                    "resolution": self.resolution,
                },
                {"type": "text", "text": prompt},
            ],
            # One format, not a list of them: the parameter takes either
            # (`Union[ResponseFormatParam, List[ResponseFormatParam]]`), and this call wants exactly
            # one kind of output.
            "response_format": {"type": "text", "mime_type": "application/json", "schema": schema},
            "generation_config": {
                "thinking_level": self.thinking_level,
                "seed": self.seed,
                "max_output_tokens": self.max_output_tokens,
            },
            # Inverts the API's default, which retains the prompt, the media reference and the
            # output for 55 days. A regression here has no visible symptom, which is why a test
            # asserts it on every request this backend sends.
            "store": False,
        }

        interaction = self.client.interactions.create(**request)
        return VlmResponse(text=interaction.output_text, usage=_usage(interaction, self.model_id))


def _usage(interaction: Any, model_id: str) -> dict[str, float]:
    """Token counts and an estimated price for one call, at `model_id`'s own rates.

    Thinking bills at the output rate but is reported *outside* `total_output_tokens`
    (total == input + output + thought), so billing `total_output_tokens` alone under-reports by
    roughly 3x at thinking_level="high". `billable_output_tokens` is the number to read.
    """
    usage = getattr(interaction, "usage", None)
    if usage is None:
        return {}

    counts = {
        name: float(getattr(usage, name, 0) or 0)
        for name in ("total_input_tokens", "total_output_tokens", "total_thought_tokens", "total_tokens")
    }
    price_in, price_out = PRICES[model_id]
    billable_out = counts["total_output_tokens"] + counts["total_thought_tokens"]
    cost = counts["total_input_tokens"] / 1e6 * price_in + billable_out / 1e6 * price_out
    return {**counts, "billable_output_tokens": billable_out, "estimated_cost_usd": round(cost, 6)}


# --- Local Qwen3.5 ----------------------------------------------------------------------------

# Frames a second the local backend asks its runtime for. Fixed rather than a parameter: it is
# mlx-vlm's own default and GeminiBackend's, so both backends see the same rate with nobody
# choosing it.
LOCAL_FPS = 2.0

# Qwen3.5's vision encoder folds this many consecutive frames into one temporal position
# (`vision_config.temporal_patch_size` on Qwen/Qwen3.5-9B). Two frames at LOCAL_FPS become one
# moment as far as the model is concerned, which is what `tolerance_s` has to report.
TEMPORAL_PATCH_SIZE = 2

# Pixel budget per sampled frame, before the runtime's own patching. Multiplied by the frame count
# on the MLX path, whose `max_pixels` is a budget for the *whole* video rather than per frame --
# handing it a per-frame figure directly shrinks a twelve-frame window to about 160x160 each.
PIXELS_PER_FRAME = 640 * 480

# How far a trimmed segment's duration may sit from the window it was cut for. Generous enough for
# frame-boundary rounding at the lowest frame rate this project's footage uses (6 fps, so 0.167 s),
# tight enough that a keyframe-rounded cut cannot pass.
TRIM_TOLERANCE_S = 0.2

# How the schema being answered names a time, and so how `time_fields` finds one in it.
TIME_SUFFIX = "_time_s"

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


class LocalRuntime(Protocol):
    """One local inference runtime, reduced to what `QwenBackend` asks of it.

    Small on purpose: the backend owns trimming, prompting, repair and the time shift, so a runtime
    only has to turn a video and a prompt into text. That is what lets both adapters be tested on a
    machine where only one of their libraries is installed.
    """

    @property
    def name(self) -> str:
        """Which runtime this is, for the run record."""
        ...

    @property
    def version(self) -> str | None:
        """The library version, or None before anything has been loaded."""
        ...

    @property
    def device(self) -> str | None:
        """The accelerator in use, or None before anything has been loaded."""
        ...

    def generate(
        self, video: Path, prompt: str, *, frames: int | None, max_new_tokens: int, seed: int
    ) -> tuple[str, dict[str, float]]:
        """Answer `prompt` about `video`, returning the generated text and its token counts.

        `frames` is how many the runtime is expected to sample, or None where the caller does not
        know. A runtime that needs the count -- the MLX path, to size a whole-video pixel budget --
        works it out from `video` itself when it is not given one, so the cost of finding out falls
        only where the number is actually read. Only the generated tokens may be decoded into the
        text.
        """
        ...


def _strip_fence(text: str) -> str | None:
    """The contents of a Markdown fence, or None when there is not one."""
    match = _FENCE.match(text)
    return match.group(1) if match else None


def repair_json(text: str) -> tuple[str, Any | None]:
    """Make an unconstrained model's answer parseable, returning the text and what it parsed to.

    Three steps, stopping at the first that yields JSON: the text as it stands, then with a
    Markdown fence stripped, then the span from the first brace to the last. Where none parse the
    original text comes back with None, so the caller's parser raises on it and the call is
    recorded with the model's own words rather than something tidied.

    The brace slice is a last resort with a known limit: prose *containing* braces before the answer
    spans both and fails. That is left to fail loudly rather than fixed with a scanner -- with
    thinking switched off and only generated tokens decoded, the first two steps are what actually
    happens.
    """
    candidates = [text]
    fenced = _strip_fence(text)
    if fenced is not None:
        candidates.append(fenced)
    opening, closing = text.find("{"), text.rfind("}")
    if opening != -1 and closing > opening:
        candidates.append(text[opening : closing + 1])

    for candidate in candidates:
        try:
            return candidate, json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return text, None


_JSON_PLACEHOLDERS: Mapping[str, Any] = {
    "string": "...",
    "number": 0.0,
    "integer": 0,
    "boolean": True,
}


def _placeholder(spec: Mapping[str, Any]) -> Any:
    """A stand-in value of the right shape for one property."""
    kind = spec.get("type")
    if kind == "array":
        return [_placeholder(spec.get("items") or {})]
    return _JSON_PLACEHOLDERS.get(str(kind), "...")


def _item_fields(schema: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]] | None:
    """The name of the schema's list property and the properties of what it holds.

    One walk -- array property, `$ref` into `$defs`, the item's `properties` -- shared by everything
    that has to know the answer's shape, so no two of them can read it differently. None for a
    schema this does not recognise; each caller decides what to do about that.
    """
    definitions = schema.get("$defs") or {}
    properties = schema.get("properties") or {}
    key = next((name for name, spec in properties.items() if spec.get("type") == "array"), None)
    if key is None:
        return None

    reference = (properties[key].get("items") or {}).get("$ref", "")
    item = definitions.get(reference.rsplit("/", 1)[-1]) or {}
    fields = item.get("properties") or {}
    return (key, fields) if fields else None


def describe_schema(schema: Mapping[str, Any]) -> str:
    """Tell an unconstrained model the answer's shape, as fields and a worked example.

    **Not the JSON Schema itself.** Handed 2 KB of `$defs` and `properties`, a 9B model answers with
    a tidied copy of the schema: valid JSON, no `interactions` key, and the whole call recorded as a
    response error. An example of the answer is unambiguous where a description of the answer is
    not, and it costs a fifth of the tokens.

    Derived from the schema rather than written out, so it cannot drift from what the caller
    actually validates. Falls back to the raw schema for a shape this does not recognise, which is
    worse but never wrong.
    """
    walked = _item_fields(schema)
    if walked is None:
        return f"Return JSON matching this schema:\n{json.dumps(schema)}"
    key, fields = walked

    lines = [
        f'Return a JSON object with one key, "{key}", holding a list of objects.',
        "Return the objects themselves -- not this description of them, and not a schema.",
        "",
        "Each object has exactly these keys:",
    ]
    for name, spec in fields.items():
        kind = spec.get("type", "string")
        shape = f"list of {(spec.get('items') or {}).get('type', 'string')}s" if kind == "array" else str(kind)
        lines.append(f"  {name} ({shape}) - {spec.get('description', '')}".rstrip())

    example = {key: [{name: _placeholder(spec) for name, spec in fields.items()}]}
    lines += [
        "",
        "So an answer with one entry looks like:",
        json.dumps(example),
        "",
        f'If there is nothing to report, return {{"{key}": []}}.',
        "Return only that JSON. No explanation, no Markdown fences.",
    ]
    return "\n".join(lines)


def time_fields(schema: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Where the times are in an answer of this shape: the list's key, and the fields to move.

    Read out of the schema rather than written down, so this cannot drift from what the caller asks
    for and validates. Times are the numeric fields named `*_time_s`, which is the convention the
    schema follows; `confidence` is numeric too, and must not be shifted.

    Raises where the schema has no such shape or no times in it, because the alternative -- shifting
    nothing and saying nothing -- is a windowed run whose times are all wrong and whose record looks
    healthy. Callers ask before they spend anything.
    """
    walked = _item_fields(schema)
    if walked is None:
        raise ValueError("schema has no list of objects to find times in")

    key, fields = walked
    found = tuple(name for name, spec in fields.items() if name.endswith(TIME_SUFFIX) and spec.get("type") == "number")
    if not found:
        raise ValueError(f"no {TIME_SUFFIX!r} fields in the schema's {key!r} items")
    return key, found


def shift_times(payload: Any, offset: float, *, key: str, fields: Sequence[str]) -> Any:
    """Return `payload` with every interaction's times moved onto the clip's clock.

    **This never raises, whatever shape it is handed.** A caller records `raw_text` and `usage`
    only after `generate` returns, and its parser is what turns a malformed answer into a recorded
    response error or per-item `malformed` entries. Raising here would turn the model's bad output
    into a *backend* failure and lose the text and usage that were paid for, so
    anything unexpected -- no interactions key, a non-list, items that are not objects, a time that
    is not a number -- passes through untouched for the validator to judge.

    Times are not clamped either. A model that reports 8.2 s on an 8.0 s segment produces a shifted
    time slightly past the window, and the validator's tolerance is what decides whether that is
    acceptable; clamping would hide what the model actually said about its own timekeeping.
    """
    if not isinstance(payload, dict):
        return payload
    items = payload.get(key)
    if not isinstance(items, list):
        return payload

    shifted: list[Any] = []
    for item in items:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(item, dict):
            shifted.append(item)
            continue
        moved: dict[str, Any] = dict(item)  # pyright: ignore[reportUnknownArgumentType]
        for field in fields:
            value = moved.get(field)
            if isinstance(value, int | float) and not isinstance(value, bool):
                moved[field] = value + offset
        shifted.append(moved)
    return {**payload, key: shifted}


def trim_segment(source: Path, window: Window, destination: Path) -> Path:
    """Cut `window` out of `source` into `destination`, frame-accurately.

    The segment's `t=0` has to *be* the window's start, because every corrected timestamp is
    computed from that. So this re-encodes rather than stream-copying: `-c copy` cuts at the nearest
    keyframe and can begin a segment early, which would put a constant, invisible offset into every
    time the model reports.

    `-t` rather than `-to`: after an input seek, ffmpeg versions differ over whether `-to` is
    measured on the seeked timeline or the source's, and a duration says the same thing everywhere.

    The result is probed and rejected if its length is not the window's, so a bad cut fails the call
    instead of quietly shifting a whole run's output.
    """
    start, end = window
    length = end - start
    if length <= 0:
        raise ValueError(f"window {window} has no duration")

    completed = subprocess.run(
        [
            require_binary("ffmpeg"),
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{length:.3f}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg could not cut {window} from {source}: {completed.stderr.strip()}")

    actual = probe_duration(destination)
    if abs(actual - length) > TRIM_TOLERANCE_S:
        raise RuntimeError(
            f"trimming {window} from {source} gave a {actual:.2f}s segment, expected {length:.2f}s; "
            f"every corrected timestamp would be wrong"
        )
    return destination


class MlxRuntime:
    """Qwen3.5 through mlx-vlm, on Apple Silicon.

    `entry_points` exists for tests: left None the library is imported on first use, so importing
    this module never needs mlx-vlm present. Handed a stand-in, the adapter's own behaviour can be
    exercised on a machine that has never installed it, which is the only way the other platform's
    adapter gets covered at all.
    """

    name = "mlx"

    def __init__(self, model_id: str, *, entry_points: Any = None) -> None:
        self.model_id = model_id
        self._entry_points = entry_points
        self._loaded: Any = None

    @property
    def version(self) -> str | None:
        try:
            return version("mlx-vlm")
        except PackageNotFoundError:
            return None

    @property
    def device(self) -> str | None:
        """Metal, always. MLX has no device to choose between."""
        return "metal" if self._loaded is not None else None

    def _library(self) -> Any:
        if self._entry_points is None:
            import mlx.core as mx
            from mlx_vlm.generate.dispatch import generate
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load, load_config

            self._entry_points = MlxEntryPoints(
                load=load,
                generate=generate,
                apply_chat_template=apply_chat_template,
                load_config=load_config,
                seed=mx.random.seed,
            )
        return self._entry_points

    def _model(self) -> Any:
        if self._loaded is None:
            library = self._library()
            model, processor = library.load(self.model_id)
            self._loaded = (model, processor, library.load_config(self.model_id))
        return self._loaded

    @staticmethod
    def _frames(video: Path) -> int:
        """How many frames this runtime will sample from `video`.

        Only reached when the caller did not know -- a whole-clip call -- and only on this path,
        which is the one that needs the number.
        """
        return round(probe_duration(video) * LOCAL_FPS)

    def generate(
        self, video: Path, prompt: str, *, frames: int | None, max_new_tokens: int, seed: int
    ) -> tuple[str, dict[str, float]]:
        """One mlx-vlm call over `video`, decoded greedily.

        `max_pixels` is a budget for the whole video here, so the per-frame figure is multiplied by
        the frame count; handing over the per-frame number directly would shrink every frame. That
        is why this path, and only this path, works the count out when it is not given one -- first,
        before the weights are loaded, so a clip that cannot be probed fails in a second rather than
        after several gigabytes have come off disk.
        """
        sampled = max(1, frames if frames is not None else self._frames(video))
        library = self._library()
        model, processor, config = self._model()
        library.seed(seed)

        formatted = library.apply_chat_template(
            processor,
            config,
            prompt,
            num_images=0,
            video=str(video),
            max_pixels=PIXELS_PER_FRAME * sampled,
            fps=LOCAL_FPS,
            enable_thinking=False,
        )
        result = library.generate(
            model,
            processor,
            formatted,
            video=str(video),
            max_tokens=max_new_tokens,
            temperature=0.0,
            seed=seed,
            enable_thinking=False,
            verbose=False,
        )

        usage = {
            "total_input_tokens": float(result.prompt_tokens),
            "total_output_tokens": float(result.generation_tokens),
        }
        usage["total_tokens"] = sum(usage.values())
        return result.text, usage


@dataclass(frozen=True, slots=True)
class MlxEntryPoints:
    """The mlx-vlm functions `MlxRuntime` calls, gathered so a test can supply its own."""

    load: Any
    generate: Any
    apply_chat_template: Any
    load_config: Any
    seed: Any


class TorchRuntime:
    """Qwen3.5 through transformers, on CUDA or CPU.

    The video is decoded here with OpenCV rather than handed over as a path, because the video
    processor's own fetching hardcodes `torchcodec` and falls back only to torchvision's removed
    decoder -- neither of which this project declares. What reaches the processor is still a *video*,
    with its temporal patching intact; only the decoding is ours.

    `entry_points` is for tests, as on `MlxRuntime`.
    """

    name = "torch"

    def __init__(self, model_id: str, *, entry_points: Any = None) -> None:
        self.model_id = model_id
        self._entry_points = entry_points
        self._loaded: Any = None
        self._device: str | None = None

    @property
    def version(self) -> str | None:
        try:
            return version("transformers")
        except PackageNotFoundError:
            return None

    @property
    def device(self) -> str | None:
        return self._device

    def _library(self) -> Any:
        if self._entry_points is None:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor
            from transformers.video_utils import VideoMetadata, load_video

            self._entry_points = TorchEntryPoints(
                torch=torch,
                load_model=AutoModelForMultimodalLM.from_pretrained,
                load_processor=AutoProcessor.from_pretrained,
                load_video=load_video,
                video_metadata=VideoMetadata,
            )
        return self._entry_points

    def _select_device(self, library: Any) -> str:
        """Whatever accelerator this machine has, and loudly when it has none.

        The fallback order is `select_device`'s; only the complaint is this runtime's, because the
        CPU is slow enough here to look hung.
        """
        device = select_device(library.torch)
        if device == "cpu":
            warnings.warn(
                f"no GPU found; running {self.model_id} on the CPU, which takes minutes per call",
                RuntimeWarning,
                stacklevel=2,
            )
        return device

    def _model(self, library: Any) -> Any:
        if self._loaded is None:
            self._device = self._select_device(library)
            model = library.load_model(self.model_id, dtype=library.torch.bfloat16, device_map=self._device)
            self._loaded = (model, library.load_processor(self.model_id))
        return self._loaded

    def generate(
        self, video: Path, prompt: str, *, frames: int | None, max_new_tokens: int, seed: int
    ) -> tuple[str, dict[str, float]]:
        """One transformers call over `video`, decoded greedily.

        `cap_pixels_per_frame=True` adopts the per-frame cap the reference implementation applies and
        transformers is making its default; without it a long video costs far more tokens than it
        should. `frames` is unused on this path -- the processor does its own sampling from the
        metadata below -- so this path never asks what it would have been.
        """
        library = self._library()
        model, processor = self._model(library)

        library.torch.manual_seed(seed)
        # Best effort: raising instead would cost the call whenever an op lacks a deterministic
        # kernel, which is a poor trade for a run that is greedy anyway.
        library.torch.use_deterministic_algorithms(True, warn_only=True)

        decoded, _ = library.load_video(str(video), fps=LOCAL_FPS, backend="opencv")
        # Metadata describing the *decoded* array, not the source: the processor samples again from
        # whatever it is told, and handing it the source's rate makes it index past the end.
        metadata = library.video_metadata(
            total_num_frames=int(decoded.shape[0]),
            fps=LOCAL_FPS,
            duration=float(decoded.shape[0]) / LOCAL_FPS,
            video_backend="opencv",
        )
        messages = [
            {"role": "user", "content": [{"type": "video", "video": decoded}, {"type": "text", "text": prompt}]}
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
            video_metadata=[metadata],
            cap_pixels_per_frame=True,
        ).to(self._device)

        generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        # Only the completion: `generate` returns prompt and completion concatenated, and the prompt
        # holds the schema, so decoding the whole sequence would let the brace-slice repair pick the
        # schema out of the prompt and hand it back as the model's answer.
        prompt_length = int(inputs["input_ids"].shape[1])
        completion = generated[0][prompt_length:]
        text = processor.decode(completion, skip_special_tokens=True)

        usage = {
            "total_input_tokens": float(prompt_length),
            "total_output_tokens": float(len(completion)),
        }
        usage["total_tokens"] = sum(usage.values())
        return text, usage


@dataclass(frozen=True, slots=True)
class TorchEntryPoints:
    """The transformers/torch callables `TorchRuntime` uses, gathered so a test can supply its own."""

    torch: Any
    load_model: Any
    load_processor: Any
    load_video: Any
    video_metadata: Any


def select_runtime(model_id: str) -> LocalRuntime:
    """The runtime this platform installed: MLX on Apple Silicon, transformers everywhere else.

    Chosen by platform rather than by what imports. `mlx-vlm` depends on `transformers`, so on Apple
    Silicon both libraries are present and an availability probe would be ambiguous exactly where it
    matters. `pyproject.toml` decides what is installed; this decides what is used.
    """
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return MlxRuntime(model_id)
    return TorchRuntime(model_id)


class QwenBackend:
    """Qwen3.5, run locally, shown video.

    The model is loaded lazily on first `generate`, so constructing a backend costs nothing and a
    caller's correctness checks run with no weights on disk -- the same reason `GeminiBackend`
    builds its client lazily.

    **A window becomes a trimmed segment.** Neither local runtime accepts a time range, so the clip
    is cut to a temporary mp4 and the model is shown that, as a clip in its own right beginning at
    zero. The times it reports are moved back onto the clip's clock here, which is why
    `segment_local_prompt` is True: asking the model to add the offset itself would be arithmetic on
    top of perception, and this way there is nothing to comply with.

    Which runtime runs is settled by platform in `select_runtime`, and the model id must match it --
    `Qwen/Qwen3.5-9B` under transformers, `mlx-community/Qwen3.5-27B-4bit` under MLX. Nothing here
    reasons about whether the weights will fit; an oversized id fails at load.
    """

    def __init__(
        self,
        model_id: str,
        *,
        max_new_tokens: int = 8192,
        seed: int = 1,
        runtime: LocalRuntime | None = None,
    ) -> None:
        """`model_id` is required and runtime-specific; see the class docstring.

        There is no default because the right id differs per machine, where `GeminiBackend` can
        default because one id is right everywhere.

        `runtime` is for tests, as `GeminiBackend` takes `client`.
        """
        if not model_id:
            raise ValueError("model_id is required, and must match the runtime this platform uses")
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, not {max_new_tokens!r}")

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self._runtime = runtime if runtime is not None else select_runtime(model_id)

    @property
    def tolerance_s(self) -> float:
        """One temporal patch: two frames at `LOCAL_FPS` are a single moment to this model."""
        return TEMPORAL_PATCH_SIZE / LOCAL_FPS

    @property
    def segment_local_prompt(self) -> bool:
        """True: the model is shown the window as a clip, and the times come back corrected."""
        return True

    @property
    def config(self) -> Mapping[str, Any]:
        """What ran. Never raises and never forces a load -- see the note below."""
        return {
            "backend": "qwen",
            "model_id": self.model_id,
            "runtime": self._runtime.name,
            # None until something has loaded: this is read while recording a run, outside the try
            # that catches a failing generate, so a version lookup that raised -- or a load forced
            # just to answer it -- would turn one recorded window failure into a lost run.
            "device": self._runtime.device,
            "runtime_version": self._runtime.version,
            "fps": LOCAL_FPS,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            # Static properties of this backend, so a saved record says for itself which caveats
            # apply to it rather than depending on anyone remembering them.
            "schema_in_prompt": True,
            "response_repaired": True,
            "times_shifted": True,
        }

    @property
    def slug(self) -> str:
        return f"{self.model_id.split('/')[-1]}__fps{LOCAL_FPS:g}__seed{self.seed}"

    def generate(self, video: Path, prompt: str, *, window: Window | None, schema: dict[str, Any]) -> VlmResponse:
        """Show the model the clip, or a trimmed window of it, and return its answer as JSON.

        The schema goes into the prompt text, since nothing constrains a local model's output; the
        answer is then repaired into something `json.loads` accepts, and its times shifted onto the
        clip's clock.

        Where there is a shift to apply, where the times live is worked out *first* -- before the
        segment is cut and before the model is loaded. The schema does not depend on the video, so
        one that hides its times costs nothing to reject, where finding out afterwards would mean an
        answer that cannot be corrected and a call already spent.
        """
        asked = f"{prompt}\n\n{describe_schema(schema)}"
        offset = window[0] if window is not None else 0.0
        located = time_fields(schema) if offset else None

        with tempfile.TemporaryDirectory() as directory:
            if window is None:
                # Whole clip: nobody here knows how many frames that is, and only a runtime that
                # needs the count pays to find out.
                shown, frames = video, None
            else:
                shown = trim_segment(video, window, Path(directory) / "segment.mp4")
                frames = max(1, round((window[1] - window[0]) * LOCAL_FPS))
            text, usage = self._runtime.generate(
                shown, asked, frames=frames, max_new_tokens=self.max_new_tokens, seed=self.seed
            )

        repaired, payload = repair_json(text)
        if payload is not None and located is not None:
            key, fields = located
            repaired = json.dumps(shift_times(payload, offset, key=key, fields=fields))
        return VlmResponse(text=repaired, usage={**usage, "estimated_cost_usd": 0.0})
