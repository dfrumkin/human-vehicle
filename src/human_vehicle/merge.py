"""Collapsing a windowed run's duplicate sightings into one record per event.

Overlapping windows report the same event once per window that saw it, on purpose: the second
sighting is corroboration, and `find_interactions` de-duplicates nothing. This is where they are
collapsed, so a clip's answer reads as a list of events rather than a list of sightings.

Two records are the same event when **all three** hold: their spans overlap in time, their
`person_ids` intersect, and their `vehicle_ids` intersect. Intersection rather than equality is what
lets a relabelled object still match -- `["P1", "P2"]` and `["P2"]` are one person.

**A record with no person ids, or no vehicle ids, never merges.** With nothing to match on, only time
and free text are left, and those are not enough: one clip here has an unlabelled person entering a
car's driver seat while a labelled one exits its passenger side, overlapping in time at the same
vehicle. Merging on the vehicle alone would fuse them into a person who did both at once. A missed
merge leaves visible duplicates; a wrong merge invents an event that never happened.

Nothing here calls a model. The merge is a pure function of the run, which is what makes it testable
and what keeps a merged file regenerable from the record beside it.
"""

from collections.abc import Iterable, Sequence

from pydantic import BaseModel, Field

from human_vehicle.interactions import InteractionRun, ReportedInteraction, WindowRun, run_tag

# How far below a group's best confidence a record may be and still supply fields. A record further
# down is the model saying it could barely tell what it was looking at, and having seen the whole
# event does not repair that.
#
# A band rather than a ranking because the model uses the scale in coarse steps -- records sit at
# 0.95, 0.92, 0.90, 0.85 -- so a 0.03 gap carries no information while containment carries real
# information about what was seen. Ranking on confidence first would let that noise decide every
# comparison and containment almost none.
#
# 0.3 is a judgement call, not a calibrated number. On the runs this was written against, every
# threshold between 0.01 and 0.47 gives identical output: no pair of a contained and a truncated
# record falls in between.
CONFIDENCE_BAND = 0.3

# Times are reported to a tenth of a second, so containment is compared with room for float noise.
_EPSILON = 0.01


class MergedInteraction(BaseModel):
    """One event, assembled from every record that reported it.

    Not a drop-in for `Interaction` or `ReportedInteraction`: it carries `window_indexes` and
    `sightings` where a reported record carries a single `window_index`.
    """

    person_ids: list[str] = Field(description="Every label any contributor saw on the person, in order first seen.")
    vehicle_ids: list[str] = Field(description="Every label any contributor saw on the vehicle, in order first seen.")
    person_description: str
    vehicle_description: str
    interaction: str
    start_time_s: float
    evidence_time_s: float
    end_time_s: float
    confidence: float
    window_indexes: list[int] = Field(description="The distinct calls that reported this event, ascending.")
    sightings: int = Field(description="How many records were merged; larger than the windows when a call repeated.")


class MergedInteractions(BaseModel):
    """A run's events, and enough to tie this file to the run it was derived from.

    `failed_windows` and `malformed_count` travel with it because merging reads only the accepted
    records. Without them, a merged list from a run that lost records either way would read as a
    complete account of the clip.
    """

    clip_id: str
    source_video: str
    run_tag: str
    duration_s: float
    interactions: list[MergedInteraction] = []
    source_interaction_count: int = 0
    merged_interaction_count: int = 0
    failed_windows: int = 0
    malformed_count: int = 0


def _windows_by_index(run: InteractionRun) -> dict[int, WindowRun]:
    """The run's windows, keyed by index, rejecting a run that cannot be looked up in.

    Containment and the evidence margin both rest on this lookup. A repeated index would silently
    resolve to one of two windows, and a missing one would surface as a `KeyError` from inside a
    comparison -- either way changing which record is believed, without saying so.
    """
    windows: dict[int, WindowRun] = {}
    for window in run.windows:
        if window.window_index in windows:
            raise ValueError(f"run has more than one window with index {window.window_index}")
        windows[window.window_index] = window
    missing = sorted({item.window_index for item in run.interactions if item.window_index not in windows})
    if missing:
        raise ValueError(f"interactions name windows that do not exist: {missing}")
    return windows


def _overlaps(left: ReportedInteraction, right: ReportedInteraction) -> bool:
    """Whether two spans share any time, touching endpoints included.

    `[4.0, 8.0]` and `[8.0, 11.0]` count. That is what one event split at a window seam looks like,
    each half truncated at the boundary, and it is commoner than two distinct episodes abutting
    exactly -- which the prompt discourages by asking for two records only where two episodes are
    *clearly separated*.
    """
    return left.start_time_s <= right.end_time_s and right.start_time_s <= left.end_time_s


def _same_event(left: ReportedInteraction, right: ReportedInteraction) -> bool:
    """Whether two records describe one event. See the module docstring for why ids are required."""
    if not (left.person_ids and left.vehicle_ids and right.person_ids and right.vehicle_ids):
        return False
    if not _overlaps(left, right):
        return False
    return bool(set(left.person_ids) & set(right.person_ids)) and bool(set(left.vehicle_ids) & set(right.vehicle_ids))


def _group(interactions: Sequence[ReportedInteraction]) -> list[list[int]]:
    """Positions of `interactions`, gathered into events, transitively.

    If A matches B and B matches C, all three are one event whether or not A matches C. That can fuse
    two genuinely separate episodes when one over-broad span bridges them -- accepted, because the
    alternative is a rule about which pairwise matches "really" count, which is the complexity this
    merge exists without. A fusion shows up as a span far longer than the interaction it describes,
    and a `sightings` count above what the overlapping windows could account for.

    Groups come back in order of their earliest member, and each group's positions ascending, so the
    result is a function of the input alone.
    """
    parent = list(range(len(interactions)))

    def find(position: int) -> int:
        while parent[position] != position:
            parent[position] = parent[parent[position]]
            position = parent[position]
        return position

    for left in range(len(interactions)):
        for right in range(left + 1, len(interactions)):
            if _same_event(interactions[left], interactions[right]):
                parent[find(left)] = find(right)

    groups: dict[int, list[int]] = {}
    for position in range(len(interactions)):
        groups.setdefault(find(position), []).append(position)
    return sorted(groups.values(), key=lambda positions: positions[0])


def _contained(item: ReportedInteraction, window: WindowRun) -> bool:
    """Whether the model watched this event start *and* finish.

    Strictly inside, so a span touching either edge counts as truncated -- which is the point of the
    test. An event that genuinely finished while the model was still watching gets an end time inside
    the window; an end time landing on the boundary is the model saying the action was still going
    when its view ran out. The validator's tolerance is deliberately not used: that exists to accept
    an answer near a seam, while this asks whether the whole event was seen.
    """
    return item.start_time_s > window.start_s + _EPSILON and item.end_time_s < window.end_s - _EPSILON


def _evidence_margin(item: ReportedInteraction, window: WindowRun) -> float:
    """How much context surrounded the moment the model called clearest."""
    return min(item.evidence_time_s - window.start_s, window.end_s - item.evidence_time_s)


def _ordered_union(lists: Iterable[Sequence[str]]) -> list[str]:
    """Every label, first occurrence kept, in the order the contributors are visited.

    Ordered because the lists are first-seen order and that order is the relabelling history;
    deduplicated because a label repeated inside one record says nothing extra.
    """
    seen: list[str] = []
    for labels in lists:
        for label in labels:
            if label not in seen:
                seen.append(label)
    return seen


def _representative(
    group: Sequence[int],
    interactions: Sequence[ReportedInteraction],
    windows: dict[int, WindowRun],
    confidence_band: float,
) -> int:
    """The position of the record that supplies every field that is not a union.

    Confidence first, but as a veto rather than a ranking: a record further than `confidence_band`
    below the group's best cannot supply anything. Among the rest, containment leads, then the
    evidence margin, then confidence as an ordinary tiebreak, then two keys that only make the choice
    deterministic.
    """
    best_confidence = max(interactions[position].confidence for position in group)
    credible = [
        position for position in group if interactions[position].confidence >= best_confidence - confidence_band
    ]

    def key(position: int) -> tuple[bool, float, float, int, int]:
        item = interactions[position]
        window = windows[item.window_index]
        return (
            not _contained(item, window),
            -_evidence_margin(item, window),
            -item.confidence,
            item.window_index,
            position,
        )

    return min(credible, key=key)


def _merge_group(
    group: Sequence[int],
    interactions: Sequence[ReportedInteraction],
    windows: dict[int, WindowRun],
    confidence_band: float,
) -> MergedInteraction:
    """One event from the records that reported it."""
    members = [interactions[position] for position in group]
    chosen = interactions[_representative(group, interactions, windows, confidence_band)]
    contained = [item for item in members if _contained(item, windows[item.window_index])]

    if contained:
        # Something watched the whole event, so its account already covers what the truncated records
        # saw a piece of. Joining them would give "opens the door | opens the door and gets in".
        start, end = chosen.start_time_s, chosen.end_time_s
        interaction = chosen.interaction
    else:
        # The event outlasted every window. The records are complementary rather than competing --
        # one saw the reaching, a later one the loading -- so every distinct account is kept, and the
        # union of spans is the only honest extent.
        start = min(item.start_time_s for item in members)
        end = max(item.end_time_s for item in members)
        ordered = sorted(
            range(len(members)),
            key=lambda index: (members[index].start_time_s, members[index].window_index, group[index]),
        )
        interaction = " | ".join(_ordered_union([members[index].interaction] for index in ordered))

    return MergedInteraction(
        # From every contributor, including any the confidence veto excluded: a weak record may be
        # the only one that saw a label, and the union is what a later step joins on.
        person_ids=_ordered_union(item.person_ids for item in members),
        vehicle_ids=_ordered_union(item.vehicle_ids for item in members),
        person_description=chosen.person_description,
        vehicle_description=chosen.vehicle_description,
        interaction=interaction,
        start_time_s=start,
        evidence_time_s=chosen.evidence_time_s,
        end_time_s=end,
        confidence=chosen.confidence,
        window_indexes=sorted({item.window_index for item in members}),
        sightings=len(members),
    )


def merge_interactions(run: InteractionRun, *, confidence_band: float = CONFIDENCE_BAND) -> MergedInteractions:
    """Collapse a run's duplicate sightings into one record per event.

    The result is derived data: it can be regenerated from `run` at any time, and it is never the
    place to look for what the model actually said -- that is the run's own record.

    `confidence_band` is how far below a group's best confidence a record may be and still supply
    fields; see `CONFIDENCE_BAND`. It must not be negative, which would empty the candidate set and
    leave no record able to supply anything, including the group's own best.

    Runs on any run, windowed or not. A whole-clip run is not a special case: it simply tends to have
    nothing to collapse, because one call reports each event once.
    """
    if confidence_band < 0:
        raise ValueError(f"confidence_band must not be negative, got {confidence_band}")

    windows = _windows_by_index(run)
    merged = [_merge_group(group, run.interactions, windows, confidence_band) for group in _group(run.interactions)]
    # Ordered as a run's own list is. The sort is stable and `_group`'s order is a function of the
    # input, so two events sharing a span order the same way every time: this is written to disk and
    # compared across runs.
    merged.sort(key=lambda item: (item.start_time_s, item.end_time_s))

    return MergedInteractions(
        clip_id=run.clip_id,
        source_video=run.source,
        run_tag=run_tag(run),
        duration_s=run.duration_s,
        interactions=merged,
        source_interaction_count=len(run.interactions),
        merged_interaction_count=len(merged),
        failed_windows=run.failed_windows,
        malformed_count=len(run.malformed),
    )


def merge_summary(merged: MergedInteractions) -> str:
    """One line saying whether the merge did anything, and whether the clip was fully examined."""
    line = (
        f"{merged.clip_id}: {merged.source_interaction_count} record(s) -> {merged.merged_interaction_count} event(s)"
    )
    if merged.failed_windows or merged.malformed_count:
        line += (
            f"  [{merged.failed_windows} call(s) failed, {merged.malformed_count} malformed -- clip not fully examined]"
        )
    return line
