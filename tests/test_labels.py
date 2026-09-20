"""Tests for the two-class labels and the renumbering that produces them."""

from dataclasses import replace
from pathlib import Path

from human_vehicle.labels import label_text, relabel_tracks
from human_vehicle.tracking import Category, FrameTracks, TrackedBox, VideoTracks

# A vehicle's COCO class, and the category it groups into. Two different classes here is the point:
# a car and a truck are one numbering.
CATEGORIES = {"person": Category.PERSON, "car": Category.VEHICLE, "truck": Category.VEHICLE}


def _box(track_id: int, class_name: str, *, left: float = 10.0) -> TrackedBox:
    return TrackedBox(
        track_id=track_id,
        category=CATEGORIES[class_name],
        class_name=class_name,
        confidence=0.9,
        xyxy=(left, 6.0, left + 20.0, 38.0),
    )


def _tracks(frames: list[list[TrackedBox]]) -> VideoTracks:
    """A record holding exactly `frames`, with plausible values for everything else."""
    return VideoTracks(
        source=Path("Videos/clip.mp4"),
        width=64,
        height=48,
        fps="30000/1001",
        frames=tuple(FrameTracks(index=index, boxes=tuple(boxes)) for index, boxes in enumerate(frames)),
        weights="yolo26s.pt",
        tracker="botsort.yaml",
        reid="none",
        imgsz=640,
        buffer_seconds=3.0,
        track_buffer=90,
    )


def _labels(tracks: VideoTracks) -> list[list[str]]:
    """The label each frame's boxes would be drawn with, in the frame's own box order."""
    return [[label_text(box.category, box.track_id) for box in frame.boxes] for frame in tracks.frames]


def test_label_text_is_the_documented_key() -> None:
    """The key the README publishes and a prompt has to be given: `P` for people, `V` for vehicles."""
    assert label_text(Category.PERSON, 1) == "P1"
    assert label_text(Category.VEHICLE, 1) == "V1"


def test_each_category_is_numbered_from_one_and_contiguously() -> None:
    """The requirement: `P5` is the fifth person and `V5` the fifth vehicle, independently.

    The original ids are sparse and shared across the two categories, which is what the tracker
    produces and what makes the raw numbering unusable downstream.
    """
    tracks = _tracks(
        [
            [_box(3, "person"), _box(4, "car"), _box(9, "truck")],
            [_box(3, "person"), _box(17, "person"), _box(4, "car")],
            [_box(17, "person"), _box(42, "person"), _box(20, "car")],
        ]
    )

    relabelled, translation = relabel_tracks(tracks)

    assert _labels(relabelled) == [
        ["P1", "V1", "V2"],
        ["P1", "P2", "V1"],
        ["P2", "P3", "V3"],
    ]
    # Both sequences start at 1 and skip nothing, and the two are independent of each other.
    assert sorted(translation.values()) == ["P1", "P2", "P3", "V1", "V2", "V3"]


def test_an_identity_keeps_one_number_through_the_clip() -> None:
    """Renumbering must not break a track: one identity in, one identity out.

    An identity is `(category, track_id)`, so a person and a vehicle sharing an id are two of them
    and are numbered in separate sequences, while a car and a truck sharing one are a single vehicle
    identity -- the COCO class is not part of an identity and does not split it.
    """
    tracks = _tracks(
        [
            [_box(2, "car"), _box(2, "person")],
            [_box(2, "truck")],
            [_box(2, "car"), _box(2, "person"), _box(2, "truck")],
        ]
    )

    relabelled, translation = relabel_tracks(tracks)

    assert translation == {
        (Category.VEHICLE, 2): "V1",
        (Category.PERSON, 2): "P1",
    }
    # `(vehicle, 2)` is one identity whichever class it was detected as, so the car and the truck
    # both carry V1 -- while the person, sharing only the id, is numbered in its own sequence.
    assert _labels(relabelled) == [["V1", "P1"], ["V1"], ["V1", "P1", "V1"]]


def test_numbering_follows_first_appearance() -> None:
    """Labels ascend as the clip plays, whatever order the tracker issued its ids in.

    Sorting by the original id would number these backwards. Within one frame the original id
    breaks the tie, so the result is fixed by the record alone.
    """
    tracks = _tracks(
        [
            [_box(50, "person")],
            [_box(50, "person"), _box(8, "person")],
            [_box(8, "person"), _box(30, "person"), _box(4, "person")],
        ]
    )

    relabelled, translation = relabel_tracks(tracks)

    assert translation == {
        (Category.PERSON, 50): "P1",
        (Category.PERSON, 8): "P2",
        # Both appear first in frame 2, so the smaller original id takes the lower number.
        (Category.PERSON, 4): "P3",
        (Category.PERSON, 30): "P4",
    }
    assert _labels(relabelled) == [["P1"], ["P1", "P2"], ["P2", "P4", "P3"]]


def test_nothing_but_the_track_ids_changes() -> None:
    """The renumbered record still has to render against the same video and name its own run."""
    tracks = _tracks(
        [
            [_box(7, "person", left=10.0), _box(7, "car", left=30.0)],
            [],
            [_box(9, "truck", left=15.0)],
        ]
    )

    relabelled, _ = relabel_tracks(tracks)

    assert replace(relabelled, frames=tracks.frames) == tracks, "a field outside the frames changed"
    for original, renumbered in zip(tracks.frames, relabelled.frames, strict=True):
        assert renumbered.index == original.index
        assert len(renumbered.boxes) == len(original.boxes)
        for before, after in zip(original.boxes, renumbered.boxes, strict=True):
            assert replace(after, track_id=before.track_id) == before, "a box changed beyond its id"


def test_the_translation_is_what_gets_drawn() -> None:
    """The mapping is for debugging, so it has to name every original identity by its new label."""
    tracks = _tracks(
        [
            [_box(3, "person"), _box(4, "truck")],
            [_box(11, "car"), _box(3, "person")],
        ]
    )

    relabelled, translation = relabel_tracks(tracks)

    original = [[(box.category, box.track_id) for box in frame.boxes] for frame in tracks.frames]
    assert [[translation[identity] for identity in frame] for frame in original] == _labels(relabelled)


def test_an_empty_record_renumbers_to_nothing() -> None:
    """A clip the tracker confirmed nothing in is a record with frames and no boxes, not an error."""
    tracks = _tracks([[], []])

    relabelled, translation = relabel_tracks(tracks)

    assert translation == {}
    assert relabelled == tracks
