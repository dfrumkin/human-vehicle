"""Tests for collapsing a run's duplicate sightings into one record per event.

Records are written by hand rather than produced by a model: the merge is a pure function of a run,
and what is under test is which records it joins and which record it then believes.
"""

from typing import Any

import pytest

from human_vehicle.interactions import InteractionRun, MalformedInteraction, ReportedInteraction, WindowRun
from human_vehicle.merge import CONFIDENCE_BAND, merge_interactions

# Fixed window geometry the hand-written records below are positioned against. Chosen for these
# tests, not mirroring any configuration: the merge is a function of the spans it is given, so what
# is under test is where a record sits relative to its window, never the window's own size.
WINDOWS = [(0.0, 6.0), (4.0, 10.0), (8.0, 14.0)]


def _item(
    *,
    window: int,
    start: float,
    end: float,
    evidence: float | None = None,
    people: list[str] | None = None,
    vehicles: list[str] | None = None,
    interaction: str = "opens the driver-side door",
    person: str = "person in a red jacket",
    vehicle: str = "white sedan",
    confidence: float = 0.95,
) -> ReportedInteraction:
    return ReportedInteraction(
        person_ids=["P1"] if people is None else people,
        vehicle_ids=["V1"] if vehicles is None else vehicles,
        person_description=person,
        vehicle_description=vehicle,
        interaction=interaction,
        start_time_s=start,
        evidence_time_s=(start + end) / 2 if evidence is None else evidence,
        end_time_s=end,
        confidence=confidence,
        window_index=window,
    )


def _run(
    items: list[ReportedInteraction],
    *,
    windows: list[tuple[float, float]] | None = None,
    malformed: int = 0,
    failed: int = 0,
) -> InteractionRun:
    spans = WINDOWS if windows is None else windows
    return InteractionRun(
        source="Videos/clip.mp4",
        clip_id="clip",
        duration_s=14.0,
        prompt_version="a1",
        backend_config={},
        backend_slug="stub",
        started_at="2026-09-19T08:00:00+00:00",
        window_s=6.0,
        stride_s=4.0,
        windows=[
            WindowRun(window_index=index, start_s=start, end_s=end, prompt="p")
            for index, (start, end) in enumerate(spans)
        ],
        interactions=items,
        malformed=[
            MalformedInteraction(window_index=0, position=index, errors=["bad"], raw={}) for index in range(malformed)
        ],
        failed_windows=failed,
    )


def test_a_relabelled_object_still_matches() -> None:
    """The worked case: intersection, not equality, is what joins a relabelled person.

    One window saw `P1` become `P2`; the next saw only `P2`. They are one person because `P2` is in
    both, and the merged ids keep the whole relabelling history for a later step to join on.
    """
    run = _run(
        [
            _item(window=0, start=4.5, end=5.5, people=["P1", "P2"], vehicles=["V1", "V2"]),
            _item(window=1, start=5.0, end=6.0, people=["P2"], vehicles=["V2"]),
        ]
    )

    merged = merge_interactions(run)

    assert merged.merged_interaction_count == 1
    assert merged.interactions[0].person_ids == ["P1", "P2"]
    assert merged.interactions[0].vehicle_ids == ["V1", "V2"]
    assert merged.interactions[0].sightings == 2
    assert merged.interactions[0].window_indexes == [0, 1]


@pytest.mark.parametrize(
    ("label", "left", "right"),
    [
        (
            "matching people, disjoint vehicles",
            {"people": ["P1"], "vehicles": ["V1"]},
            {"people": ["P1"], "vehicles": ["V2"]},
        ),
        # Two people at one car in the same seconds: merging would fuse an entry with an exit.
        (
            "matching vehicles, disjoint people",
            {"people": ["P1"], "vehicles": ["V1"]},
            {"people": ["P2"], "vehicles": ["V1"]},
        ),
    ],
)
def test_correspondence_is_required_on_both_sides(label: str, left: dict[str, Any], right: dict[str, Any]) -> None:
    run = _run([_item(window=0, start=4.5, end=5.5, **left), _item(window=1, start=5.0, end=6.0, **right)])

    assert merge_interactions(run).merged_interaction_count == 2, f"{label} must not merge"


@pytest.mark.parametrize(
    ("label", "people", "vehicles"),
    [("the person was never labelled", [], ["V1"]), ("the vehicle was never labelled", ["P1"], [])],
)
def test_a_record_without_ids_never_merges_but_is_still_reported(
    label: str, people: list[str], vehicles: list[str]
) -> None:
    """The conservative key: with nothing to match on, only time and text remain, and a wrong merge
    invents an event where a missed one merely leaves a visible duplicate."""
    run = _run(
        [
            _item(window=0, start=4.5, end=5.5),
            _item(window=1, start=5.0, end=6.0, people=people, vehicles=vehicles),
        ]
    )

    merged = merge_interactions(run)

    assert merged.merged_interaction_count == 2
    # The unmergeable record is still an event in the output, not dropped.
    assert merged.source_interaction_count == 2


def test_time_must_overlap() -> None:
    """The same person and vehicle in two separated episodes stay two events."""
    run = _run([_item(window=0, start=1.0, end=2.0), _item(window=2, start=9.0, end=10.0)])

    assert merge_interactions(run).merged_interaction_count == 2


def test_touching_endpoints_count_as_overlapping() -> None:
    """One event split at a window seam, each half truncated at the boundary."""
    run = _run([_item(window=0, start=4.0, end=6.0), _item(window=1, start=6.0, end=8.0)])

    assert merge_interactions(run).merged_interaction_count == 1


def test_grouping_is_transitive_including_the_bridge_case() -> None:
    """A and C never overlap; both overlap B, so all three are one event.

    Accepted rather than guarded: the alternative is a rule about which pairwise matches "really"
    count. A fusion shows as a span far longer than the interaction it describes.
    """
    run = _run(
        [
            _item(window=0, start=1.0, end=3.0),
            _item(window=0, start=2.0, end=9.0),
            _item(window=2, start=8.0, end=10.0),
        ]
    )

    merged = merge_interactions(run)

    assert merged.merged_interaction_count == 1
    assert merged.interactions[0].sightings == 3


def test_at_equal_confidence_the_contained_record_is_believed() -> None:
    """The case this rule was derived from: a truncated record called an entry an exit at 0.95,
    while the window that saw past the seam called it correctly at the same confidence."""
    run = _run(
        [
            # Ends exactly at window 1's edge, so it never saw the action finish.
            _item(window=1, start=8.5, end=10.0, people=["P9"], interaction="exiting the vehicle", confidence=0.95),
            _item(window=2, start=9.5, end=13.0, people=["P9"], interaction="entering the vehicle", confidence=0.95),
        ]
    )

    merged = merge_interactions(run)

    assert merged.interactions[0].interaction == "entering the vehicle"
    assert (merged.interactions[0].start_time_s, merged.interactions[0].end_time_s) == (9.5, 13.0)


def test_a_record_far_below_the_best_confidence_cannot_supply_fields() -> None:
    """Confidence is a veto on the clearly unsure: having seen the whole event does not repair a
    reading the model could barely make."""
    run = _run(
        [
            _item(window=1, start=5.0, end=9.5, interaction="hedging about what it sees", confidence=0.35),
            _item(window=2, start=8.0, end=12.0, interaction="entering the vehicle", confidence=0.95),
        ]
    )

    merged = merge_interactions(run)

    assert merged.interactions[0].interaction == "entering the vehicle"
    # The vetoed record still contributes its ids and its sighting.
    assert merged.interactions[0].sightings == 2


def test_a_marginal_confidence_gap_does_not_outrank_containment() -> None:
    """0.92 contained beats 0.95 truncated -- the case strict confidence-ranking gets wrong.

    The model uses the scale in coarse steps, so a 0.03 gap carries no information while containment
    carries real information about what was seen.
    """
    run = _run(
        [
            _item(window=1, start=5.0, end=10.0, interaction="truncated account", confidence=0.95),
            _item(window=1, start=5.0, end=9.0, interaction="contained account", confidence=0.92),
        ]
    )

    merged = merge_interactions(run)

    assert merged.interactions[0].interaction == "contained account"
    assert merged.interactions[0].confidence == 0.92


def test_widening_the_band_changes_which_record_is_believed() -> None:
    """`confidence_band` is an argument, and moving it moves the veto."""
    run = _run(
        [
            _item(window=1, start=5.0, end=9.0, interaction="contained but unsure", confidence=0.55),
            _item(window=1, start=5.0, end=10.0, interaction="truncated but sure", confidence=0.95),
        ]
    )

    assert merge_interactions(run).interactions[0].interaction == "truncated but sure"
    assert merge_interactions(run, confidence_band=0.5).interactions[0].interaction == "contained but unsure"


def test_a_negative_band_raises() -> None:
    """It would empty the candidate set, leaving no record able to supply anything."""
    with pytest.raises(ValueError, match="confidence_band"):
        merge_interactions(_run([_item(window=0, start=1.0, end=2.0)]), confidence_band=-0.1)


def test_a_contained_group_keeps_one_account_and_drops_the_rest() -> None:
    """No `"opens the door | opens the door and gets in"`: something saw the whole event, so its
    sentence already covers what the truncated records saw a piece of."""
    run = _run(
        [
            _item(window=0, start=4.0, end=6.0, interaction="opens the door"),
            _item(window=1, start=4.5, end=5.5, interaction="opens the door and gets in"),
        ]
    )

    merged = merge_interactions(run)

    assert merged.interactions[0].interaction == "opens the door and gets in"
    assert (merged.interactions[0].start_time_s, merged.interactions[0].end_time_s) == (4.5, 5.5)


def test_an_event_longer_than_every_window_keeps_each_phase_and_unions_the_span() -> None:
    """Nothing saw it whole, so the records are complementary rather than competing: one saw the
    reaching, a later one the loading."""
    run = _run(
        [
            _item(window=1, start=4.0, end=10.0, evidence=5.0, interaction="reaching into the rear door"),
            _item(window=2, start=8.0, end=14.0, evidence=9.0, interaction="loading through the rear door"),
        ]
    )

    merged = merge_interactions(run)
    event = merged.interactions[0]

    assert event.interaction == "reaching into the rear door | loading through the rear door"
    assert (event.start_time_s, event.end_time_s) == (4.0, 14.0)


def test_identical_accounts_collapse_when_nothing_is_contained() -> None:
    run = _run(
        [
            _item(window=1, start=4.0, end=10.0, evidence=5.0, interaction="reaching in"),
            _item(window=2, start=8.0, end=14.0, evidence=9.0, interaction="reaching in"),
        ]
    )

    assert merge_interactions(run).interactions[0].interaction == "reaching in"


def test_ids_union_from_every_contributor_ordered_and_deduplicated() -> None:
    """Including the record the veto excluded: a weak record may be the only one that saw a label."""
    run = _run(
        [
            _item(window=0, start=4.5, end=5.5, people=["P1", "P4", "P1"], confidence=0.95),
            _item(window=1, start=5.0, end=6.0, people=["P4", "P7"], confidence=0.30),
        ]
    )

    assert merge_interactions(run).interactions[0].person_ids == ["P1", "P4", "P7"]


def test_a_merged_record_never_contradicts_itself() -> None:
    """`start <= evidence <= end`, which taking every non-union field from one record guarantees."""
    run = _run(
        [
            _item(window=1, start=4.0, end=10.0, evidence=9.5, interaction="a", confidence=0.95),
            _item(window=2, start=8.0, end=14.0, evidence=13.0, interaction="b", confidence=0.90),
        ]
    )

    for event in merge_interactions(run).interactions:
        assert event.start_time_s <= event.evidence_time_s <= event.end_time_s


def test_separate_events_pass_through_unchanged() -> None:
    run = _run(
        [
            _item(window=0, start=1.0, end=2.0, people=["P1"], vehicles=["V1"]),
            _item(window=2, start=9.0, end=10.0, people=["P2"], vehicles=["V2"]),
        ]
    )

    merged = merge_interactions(run)

    assert merged.merged_interaction_count == 2
    assert [event.sightings for event in merged.interactions] == [1, 1]


def test_two_records_from_one_call_merge() -> None:
    """The prompt asks for two records only where two episodes are clearly separated, so an
    overlapping pair from one call is the model reporting one event twice."""
    run = _run([_item(window=0, start=4.0, end=5.5), _item(window=0, start=4.5, end=5.0)])

    merged = merge_interactions(run)

    assert merged.interactions[0].sightings == 2
    assert merged.interactions[0].window_indexes == [0]


def test_the_result_is_time_ordered_and_deterministic() -> None:
    """Merged twice, byte for byte the same -- including where every ranking key ties."""
    run = _run(
        [
            _item(window=2, start=9.0, end=10.0, people=["P2"], vehicles=["V2"]),
            _item(window=0, start=1.0, end=2.0, people=["P1"], vehicles=["V1"]),
            # Two records tied on every key: same confidence, both truncated, same window.
            _item(window=1, start=4.0, end=10.0, people=["P3"], vehicles=["V3"], interaction="first"),
            _item(window=1, start=4.0, end=10.0, people=["P3"], vehicles=["V3"], interaction="second"),
        ]
    )

    merged = merge_interactions(run)

    assert [event.start_time_s for event in merged.interactions] == [1.0, 4.0, 9.0]
    assert merged.model_dump_json() == merge_interactions(run).model_dump_json()


def test_the_file_says_whether_it_is_complete() -> None:
    """Merging reads only the accepted records, so what a run lost has to travel with the result."""
    run = _run([_item(window=0, start=4.5, end=5.5), _item(window=1, start=5.0, end=6.0)], malformed=2, failed=1)

    merged = merge_interactions(run)

    assert (merged.clip_id, merged.source_video) == ("clip", "Videos/clip.mp4")
    assert merged.run_tag.startswith("stub__w6s4__a1__")
    assert merged.duration_s == 14.0
    assert (merged.source_interaction_count, merged.merged_interaction_count) == (2, 1)
    assert (merged.failed_windows, merged.malformed_count) == (1, 2)


def test_a_record_naming_a_window_that_does_not_exist_raises() -> None:
    run = _run([_item(window=7, start=1.0, end=2.0)])

    with pytest.raises(ValueError, match="windows that do not exist"):
        merge_interactions(run)


def test_a_run_with_a_repeated_window_index_raises() -> None:
    """The lookup would silently resolve to one of two windows, changing which record is believed."""
    run = _run([_item(window=0, start=1.0, end=2.0)])
    run.windows.append(WindowRun(window_index=0, start_s=0.0, end_s=6.0, prompt="p"))

    with pytest.raises(ValueError, match="more than one window with index"):
        merge_interactions(run)


def test_the_default_band_is_the_documented_one() -> None:
    assert CONFIDENCE_BAND == 0.3
