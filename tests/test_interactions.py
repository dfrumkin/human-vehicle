"""Tests for the interactions a vision-language model reports about an annotated clip.

No model is ever called: a stub backend stands in for one, so what is under test is the prompt, the
validation and the assembly of a run rather than anything a provider does.
"""

import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from human_vehicle.interactions import (
    PROMPT_VERSION,
    ClipInteractions,
    InteractionRun,
    ReportedInteraction,
    WindowRun,
    build_prompt,
    find_interactions,
    make_windows,
    parse_response,
    run_slug,
    run_tag,
    validate_interaction,
    verify_labels,
)
from human_vehicle.merge import merge_interactions
from human_vehicle.vlm import VlmResponse, Window
from tests.conftest import MakeVideo

DURATION = 27.3

GOOD: dict[str, Any] = {
    "person_ids": ["P1"],
    "vehicle_ids": ["V2"],
    "person_description": "person in a red jacket",
    "vehicle_description": "white sedan",
    "interaction": "opens the driver-side door",
    "start_time_s": 4.0,
    "evidence_time_s": 6.0,
    "end_time_s": 8.0,
    "confidence": 0.9,
}


def _with(**overrides: Any) -> dict[str, Any]:
    return {**GOOD, **overrides}


def _response(*items: dict[str, Any]) -> str:
    return json.dumps({"interactions": list(items)})


class _StubBackend:
    """A backend that replays canned text, and records what it was asked.

    `responses` is consumed one per call. An entry that is an exception is raised instead of
    returned, which is how a failing call is simulated.
    """

    def __init__(
        self, responses: list[str | Exception], *, tolerance_s: float = 0.5, segment_local_prompt: bool = False
    ) -> None:
        self._responses = list(responses)
        self._tolerance_s = tolerance_s
        self._segment_local_prompt = segment_local_prompt
        self.calls: list[dict[str, Any]] = []

    @property
    def tolerance_s(self) -> float:
        return self._tolerance_s

    @property
    def segment_local_prompt(self) -> bool:
        return self._segment_local_prompt

    @property
    def config(self) -> dict[str, Any]:
        return {"backend": "stub", "model_id": "stub-1"}

    @property
    def slug(self) -> str:
        return "stub-1"

    def generate(self, video: Path, prompt: str, *, window: Window | None, schema: dict[str, Any]) -> VlmResponse:
        self.calls.append({"video": video, "prompt": prompt, "window": window, "schema": schema})
        reply = self._responses.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return VlmResponse(text=reply, usage={"total_input_tokens": 100.0, "estimated_cost_usd": 0.01})


def test_track_labels_are_kept_in_the_order_reported() -> None:
    """The whole point of the id lists: what the tracker called each object, in order.

    A tracker that lost a person and re-found them under a new number produces exactly the
    multi-label case, and one that never found them at all produces the empty one. Both are
    ordinary answers, not failures, and neither is de-duplicated or sorted on the way through.
    """
    raw = _response(
        _with(person_ids=["P3", "P7"], vehicle_ids=["V2"]),
        _with(person_ids=[], vehicle_ids=["V2"]),
        # A repeat is accepted: a tracker really can hand an id back after issuing another.
        _with(person_ids=["P1", "P4", "P1"], vehicle_ids=[]),
    )

    accepted, malformed = parse_response(raw, DURATION, window_index=0)

    assert not malformed
    assert [interaction.person_ids for interaction in accepted] == [["P3", "P7"], [], ["P1", "P4", "P1"]]
    assert [interaction.vehicle_ids for interaction in accepted] == [["V2"], ["V2"], []]


@pytest.mark.parametrize(
    ("label", "item"),
    [
        ("a free-form id", _with(person_ids=["person_1"])),
        ("the wrong category's letter", _with(person_ids=["V2"])),
        ("a letter with no number", _with(person_ids=["P"])),
        # `label_text` numbers from 1 and never pads, so neither of these can come from the overlay,
        # and a later merge keyed on the strings would read "P01" and "P1" as two different people.
        ("a zero id", _with(person_ids=["P0"])),
        ("a padded id", _with(person_ids=["P01"])),
        ("an empty string", _with(vehicle_ids=[""])),
        ("a bare string instead of a list", _with(vehicle_ids="V2")),
        ("a number in the list", _with(vehicle_ids=[2])),
    ],
)
def test_an_unusable_track_label_makes_the_record_malformed(label: str, item: dict[str, Any]) -> None:
    """Rejected whole rather than repaired: a label nothing could have drawn means the model is
    reading something other than the overlay, and quietly dropping it would hide that."""
    errors = validate_interaction(item, DURATION)

    assert errors, f"{label} should have been rejected"


@pytest.mark.parametrize(
    ("label", "item"),
    [
        ("past the end of the clip", _with(end_time_s=900.0)),
        ("negative time", _with(start_time_s=-2.0)),
        ("reversed span", _with(start_time_s=8.0, end_time_s=4.0)),
        ("evidence outside the span", _with(evidence_time_s=20.0)),
        ("confidence above 1", _with(confidence=1.4)),
        ("time as a string", _with(start_time_s="4.0")),
        ("missing field", {name: value for name, value in GOOD.items() if name != "interaction"}),
        ("not an object", ["nope"]),
    ],
)
def test_a_record_that_violates_the_clip_is_kept_as_malformed(label: str, item: Any) -> None:
    """A schema guarantees types, not sense: it cannot know the clip is 27 seconds long."""
    assert validate_interaction(item, DURATION), f"{label} should have been rejected"


def test_a_good_record_validates_and_malformed_ones_are_kept_beside_it() -> None:
    """One bad record must not cost a response its good ones, and must stay inspectable."""
    raw = _response(GOOD, _with(end_time_s=900.0))

    accepted, malformed = parse_response(raw, DURATION, window_index=3)

    assert [interaction.interaction for interaction in accepted] == ["opens the driver-side door"]
    assert len(malformed) == 1
    assert malformed[0].position == 1
    assert malformed[0].window_index == 3
    # The record is kept exactly as it arrived, errors beside it, nothing repaired.
    assert malformed[0].raw["end_time_s"] == 900.0
    assert "outside clip" in malformed[0].errors[0]


@pytest.mark.parametrize(
    ("label", "raw"),
    [
        ("not JSON at all", "the model apologises at length"),
        ("no interactions list", '{"result": []}'),
        ("interactions is not a list", '{"interactions": 3}'),
    ],
)
def test_an_unusable_response_fails_the_whole_call(label: str, raw: str) -> None:
    """`find_interactions` records this against the call rather than losing the run."""
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_response(raw, DURATION, window_index=0)


WINDOW: Window = (16.0, 22.0)
TOLERANCE = 0.5  # one sampling interval at 2 fps


@pytest.mark.parametrize(
    ("label", "times"),
    [
        ("inside the window", {"start_time_s": 17.0, "evidence_time_s": 18.0, "end_time_s": 20.0}),
        # The span may run past the edge: the event continues, and the next window corroborates it.
        ("a span running past the end", {"start_time_s": 20.0, "evidence_time_s": 21.0, "end_time_s": 26.0}),
        ("evidence a tolerance past the edge", {"start_time_s": 17.0, "evidence_time_s": 22.4, "end_time_s": 23.0}),
    ],
)
def test_the_windowed_contract_accepts_clip_global_times(label: str, times: dict[str, float]) -> None:
    assert not validate_interaction(_with(**times), DURATION, WINDOW, TOLERANCE), f"{label} should validate"


@pytest.mark.parametrize(
    ("label", "times"),
    [
        # What a model reverting to segment-local times produces: no overlap with [16, 22] at all.
        ("segment-local times", {"start_time_s": 1.5, "evidence_time_s": 3.0, "end_time_s": 5.0}),
        ("evidence far past the window", {"start_time_s": 17.0, "evidence_time_s": 25.0, "end_time_s": 26.0}),
        ("evidence before the window", {"start_time_s": 2.0, "evidence_time_s": 2.0, "end_time_s": 17.0}),
    ],
)
def test_the_windowed_contract_rejects_times_it_cannot_have_seen(label: str, times: dict[str, float]) -> None:
    assert validate_interaction(_with(**times), DURATION, WINDOW, TOLERANCE), f"{label} should have been rejected"


def test_an_accepted_span_is_never_clipped_to_its_window() -> None:
    """The reported event is kept whole; only coverage figures intersect it with the window."""
    raw = _response(_with(start_time_s=20.0, evidence_time_s=21.0, end_time_s=26.0))

    accepted, malformed = parse_response(raw, DURATION, window_index=4, window=WINDOW, tolerance=TOLERANCE)

    assert not malformed
    assert accepted[0].end_time_s == 26.0
    assert accepted[0].window_index == 4


@pytest.mark.parametrize(
    ("duration", "window_s", "stride_s"),
    [(27.3, 6.0, 4.0), (9.2, 6.0, 4.0), (5.0, 6.0, 4.0), (26.0, 6.0, 4.0), (20.0, 5.0, 5.0)],
)
def test_windows_tile_the_clip(duration: float, window_s: float, stride_s: float) -> None:
    """Full coverage, no gap, no degenerate tail: a gap would read as a missed interaction."""
    windows = make_windows(duration, window_s, stride_s)

    assert windows[0][0] == 0.0
    assert windows[-1][1] == pytest.approx(round(duration, 1))
    assert len(windows) == len(set(windows)), "a window was emitted twice"
    for earlier, later in itertools.pairwise(windows):
        assert later[0] <= earlier[1], f"gap between {earlier} and {later}"
    if duration > window_s:
        assert all(end - start == pytest.approx(window_s) for start, end in windows)
    else:
        assert windows == [(0.0, round(duration, 1))], "a short clip must be one window"


@pytest.mark.parametrize(
    ("label", "settings"),
    [
        ("a window with no stride", {"window_s": 6.0}),
        ("a stride with no window", {"stride_s": 4.0}),
        ("a negative window", {"window_s": -1.0, "stride_s": 1.0}),
        ("a stride longer than the window", {"window_s": 4.0, "stride_s": 6.0}),
    ],
)
def test_a_gap_leaving_windowing_raises_before_anything_is_called(
    label: str, settings: dict[str, Any], make_video: MakeVideo
) -> None:
    """Bad arguments raise where a failed call does not: the caller can fix these, and finding out
    after paying for the calls that stretches went unexamined is worse than finding out now."""
    backend = _StubBackend([_response()])

    with pytest.raises(ValueError):
        find_interactions(make_video(frames=30), backend, **settings)

    assert backend.calls == [], f"{label} must be rejected before any call"


def test_a_missing_clip_raises() -> None:
    with pytest.raises(FileNotFoundError):
        find_interactions(Path("no/such/clip.mp4"), _StubBackend([]))


def test_a_whole_clip_run_makes_one_call_over_the_clip(make_video: MakeVideo) -> None:
    """No window means the model is shown everything, and is given none of the window bounds."""
    video = make_video(frames=30)
    backend = _StubBackend([_response(_with(start_time_s=0.2, evidence_time_s=0.3, end_time_s=0.5))])

    run = find_interactions(video, backend)

    assert len(backend.calls) == 1
    assert backend.calls[0]["window"] is None
    assert not run.windowed
    assert run.error is None
    assert len(run.interactions) == 1
    assert run.prompt_version == PROMPT_VERSION
    assert run.backend_config["model_id"] == "stub-1"
    # Both call styles have the same shape: a whole-clip run is one span covering the clip.
    assert len(run.windows) == 1
    assert (run.windows[0].start_s, run.windows[0].end_s) == (0.0, run.duration_s)


def test_a_failed_call_costs_its_own_window_and_nothing_else(make_video: MakeVideo) -> None:
    """One bad window out of three must not cost the other two their results."""
    video = make_video(frames=90, fps="30/1")  # 3.0 s, so 1 s / 1 s windowing gives three calls
    backend = _StubBackend(
        [
            _response(_with(start_time_s=0.1, evidence_time_s=0.2, end_time_s=0.4)),
            RuntimeError("the API fell over"),
            # On the clip's clock, inside the third window: a segment-local 0.5 would be rejected.
            _response(_with(start_time_s=2.5, evidence_time_s=2.6, end_time_s=2.8)),
        ]
    )

    run = find_interactions(video, backend, window_s=1.0, stride_s=1.0)

    assert len(backend.calls) == 3
    assert run.failed_windows == 1
    assert run.windows[1].error == "RuntimeError: the API fell over"
    assert run.windows[1].raw_text is None
    # A partial failure is not a failed run: the other two windows keep their figures.
    assert run.error is None
    assert [interaction.window_index for interaction in run.interactions] == [0, 2]
    # Usage totals across the calls that produced any, and the failed one contributes nothing.
    assert run.usage["total_input_tokens"] == 200.0
    assert run.usage["estimated_cost_usd"] == pytest.approx(0.02)


def test_the_clip_level_list_is_in_time_order_not_call_order(make_video: MakeVideo) -> None:
    """The clip's answer is ordered by when things happened, not by which call found them.

    Windows overlap, so a later window routinely reports an earlier moment than its predecessor
    did -- here window 1 reports 1.1 s after window 0 has already reported 1.6 s. Call order is an
    artifact of how the run was made; the reader wants the clip's timeline.
    """
    video = make_video(frames=90, fps="30/1")  # 3.0 s, so 2 s / 1 s gives windows (0, 2) and (1, 3)
    backend = _StubBackend(
        [
            _response(
                _with(start_time_s=0.2, evidence_time_s=0.3, end_time_s=0.5),
                _with(start_time_s=1.6, evidence_time_s=1.7, end_time_s=1.9),
            ),
            _response(_with(start_time_s=1.1, evidence_time_s=1.2, end_time_s=1.4)),
        ]
    )

    run = find_interactions(video, backend, window_s=2.0, stride_s=1.0)

    assert [interaction.start_time_s for interaction in run.interactions] == [0.2, 1.1, 1.6]
    # Reordering must not cost a record its provenance: the out-of-order one came from window 1.
    assert [interaction.window_index for interaction in run.interactions] == [0, 1, 0]
    # The windows keep their own records in the order the model gave them.
    assert [interaction.start_time_s for interaction in run.windows[0].interactions] == [0.2, 1.6]


def test_a_run_fails_only_when_every_call_failed(make_video: MakeVideo) -> None:
    video = make_video(frames=90, fps="30/1")
    backend = _StubBackend([RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom")])

    run = find_interactions(video, backend, window_s=1.0, stride_s=1.0)

    assert run.failed_windows == 3
    assert run.error == "all 3 call(s) failed"


def test_an_unparseable_response_keeps_what_it_cost(make_video: MakeVideo) -> None:
    """The text and the usage are kept: they were paid for, and the text is the only evidence of
    what the model actually said."""
    video = make_video(frames=30)
    backend = _StubBackend(["the model apologises at length"])

    run = find_interactions(video, backend)

    assert run.failed_windows == 1
    assert run.windows[0].raw_text == "the model apologises at length"
    assert run.windows[0].usage["total_input_tokens"] == 100.0
    assert run.error is not None


def test_each_window_is_asked_about_its_own_span(make_video: MakeVideo) -> None:
    """A window's prompt has to name that window, or every call is asked the same question."""
    video = make_video(frames=90, fps="30/1")
    backend = _StubBackend([_response(), _response(), _response()])

    run = find_interactions(video, backend, window_s=1.0, stride_s=1.0)

    assert [call["window"] for call in backend.calls] == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    for call in backend.calls:
        start, end = call["window"]
        assert f"segment from {start:.1f} s to {end:.1f} s" in call["prompt"]
        assert "Report times on the FULL clip's clock" in call["prompt"]
    assert run.windowed
    assert (run.window_s, run.stride_s) == (1.0, 1.0)


def test_the_prompt_states_the_contract_the_answer_is_judged_against(make_video: MakeVideo) -> None:
    """The overlay key, the imperfection of the tracker, and the bound times are checked against."""
    prompt = build_prompt("clip", 27.34)

    assert "P<number> is a person" in prompt
    assert "V<number> is a vehicle" in prompt
    # The bound the model is told must be the one validation uses, or an answer of "runs to the end
    # of the clip" is marked out of range.
    assert "<= 27.3" in prompt
    assert "best-effort reference rather than\nground truth" in prompt
    # A model does invent labels, so the prompt asks for a reading rather than a name -- and holds
    # the opposite mistake to be just as costly, or it trades invention for silence.
    assert "Read labels; never assign them" in prompt
    assert "empty list only when\nthe object carried no label at any moment of the interaction" in prompt
    assert "Both mistakes cost the same" in prompt
    assert "One frame is\n  enough" in prompt


def test_a_repeat_run_does_not_overwrite_the_one_before_it(make_video: MakeVideo) -> None:
    """Two runs at identical settings are two samples, and both have to survive on disk.

    There is no temperature in this API and `seed` is best effort, so repeating a configuration is
    how a small difference is told from noise. Naming a record after the configuration alone would
    make the second run destroy the first -- losing exactly the comparison the repeat was for.
    """
    video = make_video(frames=30)
    first = find_interactions(video, _StubBackend([_response()]))
    second = find_interactions(video, _StubBackend([_response()]))

    # Same configuration, so the same slug: that is what the slug is for.
    assert run_slug(first) == run_slug(second)
    # Different runs, so different names -- even back to back, which is where a second-resolution
    # stamp would collide and lose one of them.
    assert run_tag(first) != run_tag(second)
    assert first.started_at and second.started_at
    # A record written before `started_at` existed still gets a usable name.
    assert run_tag(first.model_copy(update={"started_at": ""})) == run_slug(first)


def test_a_record_survives_a_round_trip_through_json(make_video: MakeVideo) -> None:
    """The record is written to disk and read back by the notebook, so it has to survive that."""
    video = make_video(frames=30)
    backend = _StubBackend([_response(_with(start_time_s=0.2, evidence_time_s=0.3, end_time_s=0.5))])

    run = find_interactions(video, backend)
    restored = InteractionRun.model_validate_json(run.model_dump_json())

    assert restored == run
    assert restored.interactions[0].person_ids == ["P1"]


def test_a_window_is_described_on_the_clips_clock_by_default(make_video: MakeVideo) -> None:
    """The ordinary case: the model is told where in a longer clip it is looking."""
    backend = _StubBackend([_response(), _response()])
    find_interactions(make_video(frames=30), backend, window_s=0.6, stride_s=0.4, clip_id="clip")

    assert "You are being shown the segment from" in backend.calls[0]["prompt"]
    assert backend.calls[0]["window"] is not None


def test_a_segment_local_backend_is_told_the_segments_own_duration(make_video: MakeVideo) -> None:
    """A backend that shows the model a trimmed window needs the prompt to match what it shows.

    Asking for clip-global times *and* correcting them afterwards would shift everything twice, so
    the window is described as a clip in its own right. `window` still reaches `generate` -- that is
    how the backend knows what to trim and by how much to correct.
    """
    backend = _StubBackend([_response(), _response()], segment_local_prompt=True)
    find_interactions(make_video(frames=30), backend, window_s=0.6, stride_s=0.4, clip_id="clip")

    prompt = backend.calls[0]["prompt"]
    assert "The clip is 0.6 seconds long." in prompt
    assert "segment from" not in prompt
    assert backend.calls[0]["window"] == (0.0, 0.6)


def test_a_whole_clip_call_is_unaffected_by_the_prompt_clock(make_video: MakeVideo) -> None:
    """With no window there is nothing to describe differently, whichever clock the backend uses."""
    clip = make_video(frames=30)
    plain = _StubBackend([_response()])
    local = _StubBackend([_response()], segment_local_prompt=True)

    find_interactions(clip, plain, clip_id="clip")
    find_interactions(clip, local, clip_id="clip")

    assert plain.calls[0]["prompt"] == local.calls[0]["prompt"]


# --- Holding reported labels against the clip --------------------------------------------------


def _reported(
    person_ids: list[str], vehicle_ids: list[str], *, start: float, end: float, window_index: int = 0
) -> ReportedInteraction:
    return ReportedInteraction.model_validate(
        {
            **GOOD,
            "person_ids": person_ids,
            "vehicle_ids": vehicle_ids,
            "start_time_s": start,
            "evidence_time_s": start,
            "end_time_s": end,
            "window_index": window_index,
        }
    )


def _run_of(*items: ReportedInteraction) -> InteractionRun:
    """A run holding `items`, each also in its own window, as `find_interactions` assembles one."""
    windows = [
        WindowRun(
            window_index=index,
            start_s=0.0,
            end_s=DURATION,
            prompt="p",
            interactions=[item for item in items if item.window_index == index],
        )
        for index in sorted({item.window_index for item in items})
    ]
    return InteractionRun(
        source="annotated.mp4",
        clip_id="clip",
        duration_s=DURATION,
        prompt_version=PROMPT_VERSION,
        backend_config={},
        backend_slug="stub",
        windows=windows,
        interactions=list(items),
    )


def test_a_label_never_drawn_is_moved_aside() -> None:
    """The `P6` case: the interaction was real, the label was not."""
    run = _run_of(_reported(["P6"], ["V4"], start=19.3, end=24.3))

    verified = verify_labels(run, {"V4": [20.0]}, tolerance_s=0.5)

    item = verified.interactions[0]
    assert (item.person_ids, item.unverified_person_ids) == ([], ["P6"])
    assert (item.vehicle_ids, item.unverified_vehicle_ids) == (["V4"], [])
    # Only the ids move. What the model saw is not in question.
    assert (item.start_time_s, item.end_time_s, item.confidence) == (19.3, 24.3, GOOD["confidence"])


def test_a_label_drawn_only_outside_the_span_is_moved_aside() -> None:
    """The `P2` case, and why the rule is temporal rather than a check that the label exists."""
    run = _run_of(_reported(["P2"], [], start=17.5, end=20.0))

    verified = verify_labels(run, {"P2": [14.25, 15.6]}, tolerance_s=0.5)

    assert (verified.interactions[0].person_ids, verified.interactions[0].unverified_person_ids) == ([], ["P2"])


def test_a_label_drawn_inside_the_span_survives() -> None:
    run = _run_of(_reported(["P1"], ["V4"], start=6.0, end=8.5))

    verified = verify_labels(run, {"P1": [7.0], "V4": [7.0]}, tolerance_s=0.5)

    item = verified.interactions[0]
    assert (item.person_ids, item.vehicle_ids) == (["P1"], ["V4"])
    assert (item.unverified_person_ids, item.unverified_vehicle_ids) == ([], [])


def test_a_label_just_outside_the_span_survives_on_tolerance() -> None:
    """Too tight a rule would strip correct labels on rounding, which is the worse failure."""
    run = _run_of(_reported(["P3"], [], start=21.0, end=24.0))

    within = verify_labels(run, {"P3": [20.85]}, tolerance_s=0.5)
    beyond = verify_labels(run, {"P3": [20.85]}, tolerance_s=0.1)

    assert within.interactions[0].person_ids == ["P3"]
    assert beyond.interactions[0].person_ids == []


def test_vehicles_are_held_to_the_same_rule() -> None:
    run = _run_of(_reported([], ["V9"], start=4.0, end=8.0))

    verified = verify_labels(run, {"V1": [5.0]}, tolerance_s=0.5)

    assert (verified.interactions[0].vehicle_ids, verified.interactions[0].unverified_vehicle_ids) == ([], ["V9"])


def test_both_copies_of_a_record_agree() -> None:
    """A record lives in the run's list and in its window's; a file cannot say two things about it."""
    run = _run_of(_reported(["P6"], ["V4"], start=19.3, end=24.3, window_index=0))

    verified = verify_labels(run, {"V4": [20.0]}, tolerance_s=0.5)

    assert verified.windows[0].interactions[0].person_ids == []
    assert verified.windows[0].interactions[0].unverified_person_ids == ["P6"]
    assert verified.interactions[0] == verified.windows[0].interactions[0]


def test_records_sharing_only_an_invented_label_no_longer_merge() -> None:
    """The bug this exists for: an invented id manufactured a second sighting of one event.

    Window 3 reported ["P2", "P5"] and window 4 ["P5"], and they collapsed into one event at
    `sightings: 2` on the strength of a `P5` the tracker never drew.
    """
    run = _run_of(
        _reported(["P2", "P5"], ["V4"], start=17.5, end=20.0, window_index=0),
        _reported(["P5"], ["V4"], start=18.0, end=24.0, window_index=1),
    )
    times = {"V4": [18.0, 19.0, 20.0], "P2": [14.25, 15.6]}

    assert len(merge_interactions(run).interactions) == 1, "the unverified run merges them into one"
    assert len(merge_interactions(verify_labels(run, times, tolerance_s=0.5)).interactions) == 2


def test_the_model_is_never_asked_for_the_verification_fields() -> None:
    """Moving these onto `Interaction` would put provenance into the request sent to the model."""
    properties = ClipInteractions.model_json_schema()["$defs"]["Interaction"]["properties"]

    assert "unverified_person_ids" not in properties
    assert "unverified_vehicle_ids" not in properties
