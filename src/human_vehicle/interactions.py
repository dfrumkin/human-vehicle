"""Human-vehicle interactions read out of an annotated clip by a vision-language model.

The input is the mp4 `overlay.render_tracked_video` writes in its standard form: every person and
vehicle the tracker found is marked at the corners of its box and labelled `P3`, `V1`, and so on.
`find_interactions` shows that clip to a model -- whole, or a window at a time -- and returns what
it reports, validated against the clip and never repaired: one list for the clip, earliest start
first, however many calls it took to gather.

An interaction is one person with one vehicle. Besides the description, the span and the confidence,
each one carries the **track labels** the overlay put on that person and that vehicle:

    person_ids=["P3", "P7"], vehicle_ids=["V2"]

An ideal detector-tracker would give each object exactly one label for the whole clip. A real one
gives none, one, or several -- a person can go unlabelled for a stretch and come back under a new
number -- so these are lists, in the order the labels were first seen, and an empty one is a normal
answer rather than a failure. They are what lets a later merge recognise the same interaction seen
by two overlapping windows as one event. Nothing here merges anything: a run's record keeps every
sighting.

**Windowed runs assume the backend reports times on the clip's clock** when it is shown only a
segment of the clip. The prompt says so and the validator holds the answer to it, but neither can
prove a model obeys: only a live check against a clip whose times are known does, and it should
pass for a backend before its windowed output is believed.

Which model runs is `human_vehicle.vlm`'s business, not this module's.
"""

import json
import re
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from human_vehicle.labels import CATEGORY_INITIALS
from human_vehicle.tracking import Category
from human_vehicle.video import probe_duration
from human_vehicle.vlm import VlmBackend, Window, sum_usage

# Bumped whenever the prompt's wording changes, and written into every record: two runs are
# comparable only if this matches. "a1" is the first prompt written for annotated clips.
PROMPT_VERSION = "a1"

# A track label as `labels.label_text` emits it: the category's letter and an unpadded positive
# number. Built from the same mapping the prompt's key is built from, so the two cannot drift.
_LABEL_PATTERNS: Mapping[Category, re.Pattern[str]] = {
    category: re.compile(rf"^{initial}[1-9][0-9]*$") for category, initial in CATEGORY_INITIALS.items()
}

# Which field holds which category's labels.
_ID_FIELDS: Mapping[str, Category] = {"person_ids": Category.PERSON, "vehicle_ids": Category.VEHICLE}


class Interaction(BaseModel):
    """One person, one vehicle, one episode. This is the shape the model is asked for.

    Times are named `*_time_s`: `vlm.time_fields` reads the suffix out of this schema to find what
    to move when a backend answers about a segment, so renaming one is a decision, not a typo.
    """

    person_ids: list[str] = Field(
        description="Every distinct label this person carried, in the order first seen, e.g. ['P1', 'P4']. "
        "Empty if the person was never labelled."
    )
    vehicle_ids: list[str] = Field(
        description="Every distinct label this vehicle carried, in the order first seen, e.g. ['V2']. "
        "Empty if the vehicle was never labelled."
    )
    person_description: str = Field(description="Concise but discriminative, e.g. 'person in a red jacket'.")
    vehicle_description: str = Field(description="Concise but discriminative, e.g. 'white sedan'.")
    interaction: str = Field(description="What the person does to or with the vehicle.")
    start_time_s: float = Field(description="When the interaction begins, in seconds from clip start.")
    evidence_time_s: float = Field(description="The moment the interaction is clearest, in seconds.")
    end_time_s: float = Field(description="When the interaction ends, in seconds from clip start.")
    confidence: float = Field(description="0.0 to 1.0.")


class ClipInteractions(BaseModel):
    """The whole answer to one call. Its JSON schema is what constrains the model's output."""

    interactions: list[Interaction]


class ReportedInteraction(Interaction):
    """An accepted interaction, plus which call produced it.

    A subclass rather than a field on `Interaction`, so provenance the model must not be asked for
    stays out of the schema sent to it.
    """

    window_index: int = Field(description="Index of the window whose call reported this.")


class MalformedInteraction(BaseModel):
    """One record that could not be trusted, kept exactly as it arrived.

    Nothing is repaired and nothing is silently dropped: a confused model has to stay visible, and
    traceable to the call that produced it.
    """

    window_index: int
    position: int
    errors: list[str]
    raw: Any


class WindowRun(BaseModel):
    """One call: what was asked, over which span, and what came back.

    A whole-clip run has exactly one of these, spanning the clip. That keeps one shape for both call
    styles -- `InteractionRun.window_s` is what says which was used -- so anything reading a record
    handles windows and never a second, special case.
    """

    window_index: int
    start_s: float
    end_s: float
    prompt: str
    raw_text: str | None = None
    interactions: list[ReportedInteraction] = []
    malformed: list[MalformedInteraction] = []
    usage: dict[str, float] = {}
    error: str | None = None


class InteractionRun(BaseModel):
    """Everything one run of `find_interactions` produced, ready to write to disk.

    `interactions` is every window's own, gathered into one list in **clip order**, earliest start
    first. It is the clip-level answer; the per-window detail is there for diagnosis.

    Nothing is de-duplicated. Overlapping windows report the same event twice on purpose, and
    collapsing them here would throw away both the corroboration and the id lists a later merge
    needs. Expect adjacent duplicates in this list and merge them downstream.

    `malformed` stays in call order, since a record is often there precisely because its times are
    not usable.
    """

    source: str
    clip_id: str
    duration_s: float
    prompt_version: str
    backend_config: dict[str, Any]
    backend_slug: str
    # When the run started, ISO 8601 in UTC. A configuration does not identify a run: there is no
    # temperature here, `seed` is best effort, and two runs at identical settings do differ -- so a
    # record is a sample, and when it was taken is part of what it is. Empty only on a record
    # written before this field existed.
    started_at: str = ""
    window_s: float | None = None
    stride_s: float | None = None
    windows: list[WindowRun] = []
    interactions: list[ReportedInteraction] = []
    malformed: list[MalformedInteraction] = []
    usage: dict[str, float] = {}
    failed_windows: int = 0
    error: str | None = None
    elapsed_s: float = 0.0

    @property
    def windowed(self) -> bool:
        return self.window_s is not None


def stated_duration(duration_s: float) -> float:
    """The clip length as the prompt states it, to one decimal.

    The prompt cannot show full float precision without inviting nonsense, so it states one decimal.
    Validation must then use this same number: telling a model the clip is 27.3 s and rejecting its
    answer of 27.3 against an unrounded 27.29863... marks a correct "runs to the end of the clip"
    answer as out of range. One value, used in both places.
    """
    return round(duration_s, 1)


def build_prompt(clip_id: str, duration_s: float, window: Window | None = None) -> str:
    """The prompt for one call, stating the contract the answer is judged against.

    The key -- which letter means which category -- is built from `labels.CATEGORY_INITIALS`, the
    same mapping `overlay` draws from, so the prompt cannot describe a different key from the one in
    the pixels.

    For a window, the model is told what it can see and that times are still on the full clip's
    clock. It is *not* told its answer must end inside the segment: an interaction beginning near
    the edge may genuinely continue past it, and a prompt demanding containment would push the model
    to truncate exactly the events windowing splits. The validator accepts the same contract.
    """
    duration_s = stated_duration(duration_s)
    person = CATEGORY_INITIALS[Category.PERSON]
    vehicle = CATEGORY_INITIALS[Category.VEHICLE]

    if window is None:
        setting = f"The clip is {duration_s:.1f} seconds long."
    else:
        start, end = window
        setting = (
            f"The full clip is {duration_s:.1f} seconds long. You are being shown the segment from "
            f"{start:.1f} s to {end:.1f} s of it.\n"
            f"Report times on the FULL clip's clock, so a moment at the start of this segment is "
            f"{start:.1f}, not 0. Describe each interaction as you observe it."
        )

    return f"""You are analyzing a surveillance-style video clip ("{clip_id}").
{setting}

The clip has been annotated automatically. People and vehicles found by a detector and followed by
a tracker are marked at the corners of their boxes and labelled:

- {person}<number> is a person - {person}1, {person}2, and so on.
- {vehicle}<number> is a vehicle - {vehicle}1, {vehicle}2, and so on. Cars, motorcycles, buses and
  trucks are all labelled this way.

The two numberings are independent, so {person}5 and {vehicle}5 have nothing to do with each other.

The detector and tracker are imperfect, and these labels are a best-effort reference rather than
ground truth:

- a person or a vehicle may have been missed altogether and carry no label at any point;
- an object's label may disappear for part of the time it is visible;
- one real person or vehicle may be given several different numbers over time.

Judge what is happening from the video itself, never from the labels. An unlabelled person
interacting with a vehicle is an interaction like any other and must be reported.

Your task is to identify ALL human-vehicle interactions in what you are shown.

An interaction means a person physically acts on, enters, exits, uses, or handles a vehicle.
For example:
- entering or exiting a vehicle
- opening or closing a door, trunk, hatch, hood, or similar vehicle part
- loading or unloading something
- leaning or reaching into a vehicle
- any other clear physical interaction with the vehicle

Merely walking past a vehicle, standing near one, or crossing in front of or behind one is not an
interaction.

Do not omit a candidate because you are unsure. If you cannot tell whether you are seeing a genuine
interaction or mere proximity, report it with a confidence below 0.5. Report a clearly visible,
well-supported interaction with a confidence above 0.8. Several people and vehicles may appear at
once, and interactions may overlap in time. If the same person and vehicle interact in two clearly
separated episodes, report two records.

For each interaction, report the labels the video showed on the two objects involved:
- person_ids: each distinct label that person carried during the interaction, in the order you
  first saw it. A person labelled {person}1, then unlabelled, then {person}4 is
  ["{person}1", "{person}4"], and one that flickers between two labels is those two labels, once
  each.
- vehicle_ids: the same for the vehicle.
Report an empty list when the object carried no label at any point. Never invent a label, and never
report one you did not actually read in the video.

Report times as seconds from the start of the clip, as decimal numbers - for example 4.5, not
"00:04". Fractional seconds are expected and useful; do not round to whole seconds.
Every time you report must satisfy 0 <= start_time_s <= evidence_time_s <= end_time_s <= {duration_s:.1f}.
Set evidence_time_s to the single moment where the interaction is most clearly visible, and to a
moment you can actually see.

If the clip contains no genuine interaction, return an empty list."""


_REQUIRED = tuple(Interaction.model_fields)
_TIMES = ("start_time_s", "evidence_time_s", "end_time_s")
_STRINGS = ("person_description", "vehicle_description", "interaction")


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _id_errors(field: str, value: Any) -> list[str]:
    """Why `value` is not a usable list of track labels for `field`.

    An empty list is fine -- the tracker missing an object entirely is the case this whole design
    exists to tolerate -- and so is a repeated label, even though the prompt asks for each one once:
    a repeat is a model wording its answer differently, not an answer that cannot be trusted, and a
    later merge can collapse it.
    """
    if not isinstance(value, list):
        return [f"{field} is not a list"]
    pattern = _LABEL_PATTERNS[_ID_FIELDS[field]]
    return [
        f"{field}[{index}] {label!r} is not a {pattern.pattern[1]} label"
        for index, label in enumerate(value)
        if not (isinstance(label, str) and pattern.match(label))
    ]


def validate_interaction(
    item: Any,
    duration_s: float,
    window: Window | None = None,
    tolerance: float = 0.0,
) -> list[str]:
    """Every reason this record cannot be trusted. An empty list means it is usable.

    A schema guarantees types, not sense: it cannot know the clip is 27 seconds long, so a detection
    at 900 s satisfies it. Missing and wrong-typed fields are checked too, rather than assumed away,
    because a backend that cannot constrain its output is a supported case.
    """
    if not isinstance(item, dict):
        return ["not a JSON object"]

    errors = [f"missing field {name!r}" for name in _REQUIRED if name not in item]
    for name in _TIMES:
        if name in item and not _is_number(item[name]):
            errors.append(f"{name} is not a number")
    if "confidence" in item and not _is_number(item["confidence"]):
        errors.append("confidence is not a number")
    for name in _STRINGS:
        if name in item and not isinstance(item[name], str):
            errors.append(f"{name} is not a string")
    for name in _ID_FIELDS:
        if name in item:
            errors.extend(_id_errors(name, item[name]))
    if errors:
        return errors

    start, evidence, end = (float(item[name]) for name in _TIMES)
    for name, value in zip(_TIMES, (start, evidence, end), strict=True):
        if not 0.0 <= value <= duration_s:
            errors.append(f"{name} {value} outside clip [0, {duration_s:.1f}] s")
    if start > end:
        errors.append(f"start_time_s {start} > end_time_s {end}")
    elif not start <= evidence <= end:
        errors.append(f"evidence_time_s {evidence} outside [{start}, {end}]")
    confidence = float(item["confidence"])
    if not 0.0 <= confidence <= 1.0:
        errors.append(f"confidence {confidence} outside [0, 1]")

    if window is not None:
        start_s, end_s = window
        low, high = start_s - tolerance, end_s + tolerance
        # The span need only OVERLAP the window: an event beginning near the edge may genuinely
        # continue past it, and the neighbouring window reporting the same event is the
        # corroboration that not merging rests on.
        if end < low or start > high:
            errors.append(f"span {start}-{end} does not overlap window [{start_s:.1f}, {end_s:.1f}] s")
        # Evidence is held to the stricter rule. start/end describe an event that may extend beyond
        # what was shown; evidence names the single clearest moment, which can only be a moment the
        # model was actually given.
        if not low <= evidence <= high:
            errors.append(f"evidence_time_s {evidence} outside window [{start_s:.1f}, {end_s:.1f}] s")
    return errors


def parse_response(
    raw_text: str,
    duration_s: float,
    *,
    window_index: int,
    window: Window | None = None,
    tolerance: float = 0.0,
) -> tuple[list[ReportedInteraction], list[MalformedInteraction]]:
    """Split one response into what can be used and what cannot.

    Raises only when the response as a whole is unusable -- not JSON, or carrying no interactions
    list -- which `find_interactions` records against the call rather than letting it end the run.
    """
    payload = json.loads(raw_text)
    items = payload.get("interactions") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError("response has no 'interactions' list")

    accepted: list[ReportedInteraction] = []
    malformed: list[MalformedInteraction] = []
    for position, item in enumerate(items):
        errors = validate_interaction(item, duration_s, window, tolerance)
        if not errors:
            try:
                # Merged rather than passed alongside: a model that put a `window_index` of its own
                # in the record would otherwise make this a TypeError and cost the call its other,
                # good records. Ours is the one that counts.
                accepted.append(ReportedInteraction.model_validate({**item, "window_index": window_index}))
                continue
            except ValidationError as exc:
                # The validator above covers every field, so this is unreachable in practice. It is
                # here because the alternative -- a raised exception mid-response -- would cost the
                # call its other, good records.
                errors = [f"could not be loaded: {exc}"]
        malformed.append(MalformedInteraction(window_index=window_index, position=position, errors=errors, raw=item))
    return accepted, malformed


def make_windows(duration_s: float, window_s: float, stride_s: float) -> list[Window]:
    """Segments in seconds tiling [0, duration], the last one anchored to the clip end.

    Anchoring rather than leaving a short remainder: a 0.4 s tail at 2 fps holds at most one frame
    and would cost a whole call to look at almost nothing. The extra overlap is cheaper.
    """
    windows: list[Window] = []
    start = 0.0
    while start + window_s < duration_s:
        windows.append((round(start, 1), round(start + window_s, 1)))
        start += stride_s
    final = (round(max(0.0, duration_s - window_s), 1), round(duration_s, 1))
    if not windows or windows[-1] != final:
        windows.append(final)
    return windows


def _check_window_settings(window_s: float | None, stride_s: float | None) -> None:
    """Reject a windowing that would leave gaps, before anything is called.

    Bad arguments raise where a failed call does not: a caller can fix an argument, and finding out
    after paying for seven calls that the run left stretches of the clip unexamined is worse than
    finding out immediately. An uncovered stretch is indistinguishable from a missed interaction in
    the one view that measures recall.
    """
    if (window_s is None) != (stride_s is None):
        raise ValueError("window_s and stride_s must be set together; one alone is a half-configured run")
    if window_s is None or stride_s is None:
        return
    if window_s <= 0 or stride_s <= 0:
        raise ValueError(f"window_s and stride_s must be positive, not {window_s} and {stride_s}")
    if stride_s > window_s:
        raise ValueError(f"stride_s {stride_s} > window_s {window_s} would leave stretches of the clip unexamined")


def find_interactions(
    video: str | Path,
    backend: VlmBackend,
    *,
    window_s: float | None = None,
    stride_s: float | None = None,
    clip_id: str | None = None,
) -> InteractionRun:
    """Ask `backend` for the human-vehicle interactions in an annotated clip.

    `run.interactions` is the answer: every interaction found in the clip, earliest start first,
    whichever call reported it. Two overlapping windows reporting one event leave two records --
    see below.

    `video` is the annotated mp4 from `overlay.render_tracked_video`, rendered without
    `show_confidence`: a debugging render's labels are four times as wide, which is straight out of
    the pixel budget that keeps the footage itself readable.

    With `window_s` and `stride_s` set, the clip is examined a segment at a time -- 8.0 and 4.0
    tiles it in 8-second windows advancing 4 seconds, so neighbours overlap by 4. **Counts are not
    comparable between a windowed run and a whole-clip one**: the overlap means one interaction is
    often reported twice, and nothing here de-duplicates. That is deliberate -- the same event seen
    by two windows is corroboration, and collapsing them is a later merge's job, keyed on the track
    labels reported here.

    A windowed run assumes the backend reports times on the **clip's** clock when shown a segment.
    The prompt says so and every record is validated against it, but a model that quietly reverted
    to segment-local times would shift every windowed timestamp while still looking plausible; a
    live timebase check against a clip whose times are known is what verifies that for a backend.

    **A failed call never raises.** It is recorded against its window, so one bad window does not
    cost the run the others, and only a run where every call failed is itself marked failed. Bad
    arguments -- a missing clip, a windowing that would leave gaps -- raise before anything is
    called: a caller can fix those, and an unattended sweep should come back with a diagnosable
    record rather than a traceback.
    """
    source = Path(video)
    if not source.is_file():
        raise FileNotFoundError(f"no such video: {source}")
    _check_window_settings(window_s, stride_s)

    # The bound the model is given and the bound its answer is judged against must be one number.
    duration = stated_duration(probe_duration(source))
    # A whole-clip run is one span covering the clip, so both call styles share one shape below;
    # `windowed` is what decides which prompt and which bounds each span is given.
    if window_s is not None and stride_s is not None:
        windowed = True
        spans = make_windows(duration, window_s, stride_s)
    else:
        windowed = False
        spans = [(0.0, duration)]
    schema = ClipInteractions.model_json_schema()

    started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat(timespec="milliseconds")
    runs: list[WindowRun] = []
    for index, span in enumerate(spans):
        # A whole-clip call is one span covering the clip, but it is not a *window*: the model is
        # shown everything, so it gets the plain prompt and none of the per-window bounds.
        window = span if windowed else None
        if window is not None and backend.segment_local_prompt:
            # This backend shows the model the window as a clip in its own right, so the prompt has
            # to describe it that way: the plain whole-clip wording, stating the segment's duration.
            # The backend puts the times back on the clip's clock before returning, so everything
            # below -- validation included -- still sees clip-global times.
            prompt = build_prompt(clip_id or source.stem, window[1] - window[0], None)
        else:
            prompt = build_prompt(clip_id or source.stem, duration, window)
        run = WindowRun(window_index=index, start_s=span[0], end_s=span[1], prompt=prompt)
        try:
            response = backend.generate(source, prompt, window=window, schema=schema)
            # Kept even when parsing fails below: they were paid for, and the text is the only
            # evidence of what the model actually said.
            run.raw_text = response.text
            run.usage = dict(response.usage)
            run.interactions, run.malformed = parse_response(
                response.text,
                duration,
                window_index=index,
                window=window,
                tolerance=backend.tolerance_s,
            )
        except Exception as exc:
            run.error = f"{type(exc).__name__}: {exc}"
        runs.append(run)

    failed = sum(1 for run in runs if run.error)
    return InteractionRun(
        source=str(source),
        clip_id=clip_id or source.stem,
        duration_s=duration,
        prompt_version=PROMPT_VERSION,
        backend_config=dict(backend.config),
        backend_slug=backend.slug,
        started_at=started_at,
        window_s=window_s,
        stride_s=stride_s,
        windows=runs,
        # In clip order, not call order: which window reported an event is an artifact of how the
        # calls were made, and a later window can report an earlier moment than its predecessor did.
        # Each record still carries its `window_index`, so nothing is lost by reordering them.
        interactions=sorted(
            (interaction for run in runs for interaction in run.interactions),
            key=lambda interaction: (interaction.start_time_s, interaction.end_time_s),
        ),
        # Left in call order. A record is often malformed *because* its times are nonsense, so
        # ordering these by time would sort them on the field that failed validation.
        malformed=[record for run in runs for record in run.malformed],
        usage=sum_usage([run.usage for run in runs]),
        failed_windows=failed,
        # One bad window out of seven must not cost the other six their figures, so partial failure
        # is carried by failed_windows and only a total loss is an error.
        error=f"all {len(runs)} call(s) failed" if failed == len(runs) else None,
        elapsed_s=time.perf_counter() - started,
    )


def run_slug(run: InteractionRun) -> str:
    """A filename fragment naming what produced `run`, in the style of `tracking.config_slug`.

    The backend's own slug plus the windowing, which is this module's setting rather than the
    backend's, and the prompt version, because two prompts are two different questions.

    This names a *configuration*, so two runs at the same settings share it. To name a run, use
    `run_tag`.
    """
    windowing = f"__w{run.window_s:g}s{run.stride_s:g}" if run.windowed else ""
    return f"{run.backend_slug}{windowing}__{run.prompt_version}"


def run_tag(run: InteractionRun) -> str:
    """A filename fragment naming *this* run: its configuration, then when it started.

    The timestamp is what keeps repeats apart. There is no temperature in this API and `seed` is
    best effort, so two runs at identical settings genuinely differ -- which is the reason to repeat
    one before believing a small difference, and the reason a configuration alone must not decide a
    filename. Naming a record by `run_slug` means the second run quietly destroys the first, and the
    comparison that repeating was for is lost.

    A record written before `started_at` existed falls back to the configuration alone.
    """
    if not run.started_at:
        return run_slug(run)
    # To the millisecond, not the second: two runs that start inside the same second would share a
    # name and the second would overwrite the first, which is the one thing this exists to prevent.
    started = datetime.fromisoformat(run.started_at)
    return f"{run_slug(run)}__{started:%Y%m%d-%H%M%S}-{started.microsecond // 1000:03d}"
