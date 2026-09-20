"""The two-class labels drawn on an annotated clip, and the renumbering that produces them.

A label is a category letter and a number: `P3` is the third person, `V3` the third vehicle, and the
two are unrelated. The downstream task cares only about people and vehicles, so the COCO class a box
was detected as (car, bus, truck, motorcycle) does not reach the label, though it stays in the
record.

A tracker issues ids across all classes at once, so a clip's people come out sparsely numbered --
`P3`, `P17`, `P42`. `relabel_tracks` renumbers them, per category and from 1, and hands back the
translation from the original identities so a run can still be traced back to what the tracker did.
"""

from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction

from human_vehicle.tracking import Category, VideoTracks

# The letter that stands for each category in a label. Both categories have one, so no box can fail
# to be labelled.
CATEGORY_INITIALS: Mapping[Category, str] = {
    Category.PERSON: "P",
    Category.VEHICLE: "V",
}

# An object as the tracker identified it: the id alone is not an identity, since a tracker issues
# ids across classes.
Identity = tuple[Category, int]


def label_text(category: Category, track_id: int) -> str:
    """The label for one object, e.g. `"P3"`.

    The single place a label string is formed, so the translation `relabel_tracks` returns and the
    glyphs drawn on the clip cannot disagree.
    """
    return f"{CATEGORY_INITIALS[category]}{track_id}"


def relabel_tracks(tracks: VideoTracks) -> tuple[VideoTracks, dict[Identity, str]]:
    """Renumber `tracks` so each category's ids run from 1, and return the translation.

    The record that comes back is the one to render: its people are numbered `1..n` and its vehicles
    `1..m`, each sequence contiguous and independent of the other, so a label reads as "the fifth
    person" rather than as whatever id the tracker happened to issue.

    The second return value maps each original `(category, track_id)` identity to its new label. It
    exists for debugging -- print it, save it, compare a `V7` in the video against the tracker's own
    output -- and nothing downstream needs it.

    Identities are numbered in order of first appearance, with the original id breaking ties inside
    a frame, so labels ascend as the clip plays and the same record always renumbers the same way.

    Rendering does not require a renumbered record and cannot tell one: a record whose ids already
    run from 1 per category is indistinguishable from this one. Skipping the step costs contiguous
    numbering, not correctness -- the category letter keeps a person and a vehicle apart even where
    they share a tracker id.
    """
    first_seen: dict[Identity, tuple[int, int]] = {}
    for frame in tracks.frames:
        for box in frame.boxes:
            identity = (box.category, box.track_id)
            if identity not in first_seen:
                first_seen[identity] = (frame.index, box.track_id)

    renumbered: dict[Identity, int] = {}
    for category in Category:
        identities = sorted(
            (identity for identity in first_seen if identity[0] is category),
            key=lambda identity: first_seen[identity],
        )
        for number, identity in enumerate(identities, start=1):
            renumbered[identity] = number

    frames = tuple(
        replace(
            frame,
            boxes=tuple(replace(box, track_id=renumbered[(box.category, box.track_id)]) for box in frame.boxes),
        )
        for frame in tracks.frames
    )
    translation = {
        (category, track_id): label_text(category, number) for (category, track_id), number in renumbered.items()
    }
    return replace(tracks, frames=frames), translation


def label_times(tracks: VideoTracks) -> dict[str, list[float]]:
    """When each label was on screen: the times in seconds of the frames it appears in.

    Keyed by the label as it was drawn, so `"P3"` maps to every moment a `P3` was in the picture.
    A label absent from the mapping was never drawn at all.

    Pass the record that was rendered -- the renumbered one. The keys are the glyphs in the pixels,
    so a record that was not renumbered gives a mapping keyed by the tracker's own sparse ids, which
    are not what a model reading the clip can have seen.

    A frame's time is its own `index` over the frame rate, not its position in the list. The two
    agree for any record the renderer would accept, since it refuses one that is not in frame order,
    but the index is what ties a time to the frame the tracker actually saw.

    The times are what `interactions.verify_labels` holds a model's reported labels against, which
    is why this is a flat list per label rather than intervals: a tracker drops a label for a frame
    and picks it up again constantly, and collapsing those gaps would claim the label was on screen
    through a stretch where it was not.
    """
    fps = float(Fraction(tracks.fps))
    times: dict[str, list[float]] = {}
    for frame in tracks.frames:
        for box in frame.boxes:
            times.setdefault(label_text(box.category, box.track_id), []).append(frame.index / fps)
    return times
