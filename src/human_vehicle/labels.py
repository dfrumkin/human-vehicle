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
    glyphs `overlay` draws cannot disagree.
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
