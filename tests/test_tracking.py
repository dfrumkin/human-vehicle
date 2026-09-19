"""Tests for the tracking record `track_video` produces.

The detector itself is never run: that would download weights, be slow, and test Ultralytics
rather than this code. A stub stands in for `YOLO` so the conversion is what is under test.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from ultralytics.utils import YAML

from human_vehicle import tracking
from human_vehicle.tracking import (
    COCO_CLASSES,
    Category,
    VideoTracks,
    build_tracker_config,
    config_slug,
    track_video,
)
from tests.conftest import MakeVideo

COCO_NAMES: Mapping[int, str] = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class _StubBoxes:
    """The few attributes `_to_boxes` reads off an Ultralytics `Boxes`."""

    def __init__(
        self,
        ids: Sequence[int] | None,
        classes: Sequence[int],
        confidences: Sequence[float],
        xyxy: Sequence[Sequence[float]],
    ) -> None:
        self.id = None if ids is None else np.array(ids, dtype=float)
        self.cls = np.array(classes, dtype=float)
        self.conf = np.array(confidences, dtype=float)
        self.xyxy = np.array(xyxy, dtype=float).reshape(-1, 4)


class _StubResult:
    def __init__(self, boxes: _StubBoxes | None) -> None:
        self.boxes = boxes


# Frame 0 is tracked, frame 1 has detections the tracker never confirmed (no ids), frame 2 has
# nothing at all. Frames 1 and 2 must still appear, so indices stay aligned with the video.
STUB_RESULTS = [
    _StubResult(
        _StubBoxes(
            ids=[1, 2],
            classes=[0, 2],
            confidences=[0.91, 0.85],
            xyxy=[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        )
    ),
    _StubResult(_StubBoxes(ids=None, classes=[0], confidences=[0.12], xyxy=[[1.0, 1.0, 2.0, 2.0]])),
    _StubResult(None),
]


class _StubYOLO:
    """Records the arguments `track_video` passes, and replays canned results."""

    last_kwargs: ClassVar[dict[str, Any]] = {}
    last_tracker_config: ClassVar[dict[str, Any]] = {}
    names: Mapping[int, str] = COCO_NAMES

    def __init__(self, weights: str) -> None:
        self.weights = weights

    def track(self, source: str, **kwargs: Any) -> Iterator[_StubResult]:
        _StubYOLO.last_kwargs = {"source": source, **kwargs}
        return self._results(kwargs["tracker"])

    @staticmethod
    def _results(tracker: str) -> Iterator[_StubResult]:
        """Read the tracker YAML as iteration begins, exactly as Ultralytics does.

        Reading it here rather than in `track` is the point: `stream=True` makes the real call
        lazy, so the generated config has to outlive `model.track()`. Reading eagerly would pass
        even if the temporary directory were already gone.
        """
        _StubYOLO.last_tracker_config = YAML.load(tracker)
        yield from STUB_RESULTS


def test_coco_classes_map_to_expected_categories() -> None:
    """People and the four vehicle types this project cares about, and nothing else."""
    assert {class_id: category for class_id, (_, category) in COCO_CLASSES.items()} == {
        0: Category.PERSON,
        2: Category.VEHICLE,
        3: Category.VEHICLE,
        5: Category.VEHICLE,
        7: Category.VEHICLE,
    }
    # The names must be the COCO ones, because the ids are what the categories are keyed by.
    assert all(COCO_NAMES[class_id] == name for class_id, (name, _) in COCO_CLASSES.items())
    assert 1 not in COCO_CLASSES  # bicycles are deliberately out of scope


def test_track_video_converts_results(monkeypatch: pytest.MonkeyPatch, make_video: MakeVideo) -> None:
    """Ids are kept, id-less detections are dropped, and every decoded frame gets an entry."""
    monkeypatch.setattr(tracking, "YOLO", _StubYOLO)
    source = make_video(width=64, height=48, fps="30000/1001", frames=3)

    result = track_video(source)

    assert (result.width, result.height, result.fps) == (64, 48, "30000/1001")
    assert [frame.index for frame in result.frames] == [0, 1, 2]

    first, second, third = result.frames
    assert [box.track_id for box in first.boxes] == [1, 2]
    assert [box.category for box in first.boxes] == [Category.PERSON, Category.VEHICLE]
    assert [box.class_name for box in first.boxes] == ["person", "car"]
    assert first.boxes[0].xyxy == (1.0, 2.0, 3.0, 4.0)
    assert second.boxes == ()
    assert third.boxes == ()


def test_track_video_asks_the_tracker_for_the_mapped_classes(
    monkeypatch: pytest.MonkeyPatch, make_video: MakeVideo
) -> None:
    """The class filter must stay in step with the category table, and results must stream."""
    monkeypatch.setattr(tracking, "YOLO", _StubYOLO)

    track_video(make_video(frames=3))

    assert _StubYOLO.last_kwargs["classes"] == sorted(COCO_CLASSES)
    # stream=True keeps one decoded frame alive at a time; without it a 4K clip retains gigabytes.
    assert _StubYOLO.last_kwargs["stream"] is True
    # Ultralytics falls through to the CPU on macOS unless MPS is named explicitly.
    assert _StubYOLO.last_kwargs["device"] == "mps"
    # persist carries tracker state between calls, which would leak ids from a previous clip.
    assert "persist" not in _StubYOLO.last_kwargs


def test_track_video_rejects_non_coco_weights(monkeypatch: pytest.MonkeyPatch, make_video: MakeVideo) -> None:
    """Custom weights would otherwise mislabel every box silently."""

    class _WrongNames(_StubYOLO):
        names: Mapping[int, str] = {0: "cat", 2: "dog"}

    monkeypatch.setattr(tracking, "YOLO", _WrongNames)

    with pytest.raises(ValueError, match="not COCO-classed"):
        track_video(make_video(frames=3))


def test_track_video_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        track_video(tmp_path / "nope.mp4")


@pytest.mark.parametrize("tracker", ["botsort.yaml", "tracktrack.yaml", "deepocsort.yaml"])
def test_reid_none_disables_the_reid_stage(tracker: str) -> None:
    assert build_tracker_config(tracker, "none")["with_reid"] is False


def test_reid_auto_uses_the_detector_features() -> None:
    config = build_tracker_config("tracktrack.yaml", "auto")
    assert config["with_reid"] is True
    assert config["model"] == "auto"


def test_reid_names_a_model() -> None:
    config = build_tracker_config("botsort.yaml", "yolo26s-reid.onnx")
    assert config["with_reid"] is True
    assert config["model"] == "yolo26s-reid.onnx"


@pytest.mark.parametrize("tracker", ["bytetrack.yaml", "ocsort.yaml", "fasttrack.yaml"])
def test_reid_none_leaves_a_non_reid_tracker_alone(tracker: str) -> None:
    """Writing with_reid into a config that never reads it would misstate what is running."""
    assert "with_reid" not in build_tracker_config(tracker, "none")


@pytest.mark.parametrize("tracker", ["bytetrack.yaml", "ocsort.yaml", "fasttrack.yaml"])
def test_reid_on_a_tracker_without_one_is_an_error(tracker: str) -> None:
    """The alternative is a silent no-op: the run looks configured for ReID and simply is not."""
    with pytest.raises(ValueError, match="no ReID stage"):
        build_tracker_config(tracker, "auto")


def test_thresholds_default_to_the_trackers_own() -> None:
    """The shipped defaults differ per tracker, so None must mean "leave it alone"."""
    assert build_tracker_config("tracktrack.yaml", "none")["track_low_thresh"] == 0.25
    assert build_tracker_config("tracktrack.yaml", "none")["new_track_thresh"] == 0.7
    assert build_tracker_config("botsort.yaml", "none")["track_low_thresh"] == 0.1
    assert build_tracker_config("botsort.yaml", "none")["new_track_thresh"] == 0.25


def test_thresholds_can_be_overridden() -> None:
    """These decide what reaches the output: track_low_thresh what an existing track will attach
    to, new_track_thresh what may start a new one."""
    config = build_tracker_config("tracktrack.yaml", "auto", track_low_thresh=0.05, new_track_thresh=0.5)

    assert config["track_low_thresh"] == 0.05
    assert config["new_track_thresh"] == 0.5
    # Overriding one must leave the other where the tracker had it.
    assert build_tracker_config("tracktrack.yaml", "none", track_low_thresh=0.05)["new_track_thresh"] == 0.7


def test_setting_a_threshold_a_tracker_lacks_is_an_error(tmp_path: Path) -> None:
    """Same reasoning as ReID: a key the tracker never reads would be a silent no-op."""
    custom = tmp_path / "custom.yaml"
    YAML.save(str(custom), {"tracker_type": "bytetrack"})

    with pytest.raises(ValueError, match="no track_low_thresh"):
        build_tracker_config(str(custom), "none", track_low_thresh=0.05)


def test_tracktrack_settings_can_be_overridden() -> None:
    """The buffer is a frame count here: turning a duration into one needs the clip's frame rate,
    which `track_video` has and this does not."""
    config = build_tracker_config(
        "tracktrack.yaml", "auto", track_buffer=90, lost_match_thr=0.8, iou_weight=0.4, reid_weight=0.6
    )

    assert config["track_buffer"] == 90
    assert config["lost_match_thr"] == 0.8
    assert (config["iou_weight"], config["reid_weight"]) == (0.4, 0.6)
    # Overriding these must leave everything else where the tracker had it.
    assert (config["track_low_thresh"], config["new_track_thresh"]) == (0.25, 0.7)
    assert config["match_thresh"] == 0.7


def test_tracktrack_settings_default_to_the_trackers_own() -> None:
    """0.0 for lost_match_thr is TrackTrack's way of switching its relaxed rebind pass off."""
    config = build_tracker_config("tracktrack.yaml", "none")

    assert config["track_buffer"] == 30
    assert config["lost_match_thr"] == 0.0
    assert (config["iou_weight"], config["reid_weight"]) == (0.5, 0.5)


def test_setting_a_tracktrack_only_key_on_another_tracker_is_an_error() -> None:
    """BoT-SORT has no relaxed rebind pass, so this would configure nothing at all."""
    with pytest.raises(ValueError, match="no lost_match_thr"):
        build_tracker_config("botsort.yaml", "none", lost_match_thr=0.8)


@pytest.mark.parametrize(
    ("fps", "expected"),
    [
        # 3 s at 29.97 is 89.91 frames, which rounds to 90 rather than truncating to 89.
        ("30000/1001", 90),
        ("25/1", 75),
        # 3 s at 6 fps is 18 frames, which the floor lifts to 20.
        ("6/1", 20),
    ],
)
def test_track_buffer_follows_the_clips_frame_rate(
    monkeypatch: pytest.MonkeyPatch, make_video: MakeVideo, fps: str, expected: int
) -> None:
    """Ultralytics reads track_buffer as frames and never scales it, so the same duration has to
    become a different frame count per clip. A constant here would silently be the old bug."""
    monkeypatch.setattr(tracking, "YOLO", _StubYOLO)

    result = track_video(make_video(fps=fps, frames=3), tracker="tracktrack.yaml")

    assert result.track_buffer == expected
    assert _StubYOLO.last_tracker_config["track_buffer"] == expected
    # The seconds are recorded as asked for, whatever the clip turned them into.
    assert result.buffer_seconds == 3.0


def test_track_buffer_applies_on_the_default_path(monkeypatch: pytest.MonkeyPatch, make_video: MakeVideo) -> None:
    """The conversion is tracker-independent. Every other buffer case here names TrackTrack, so
    without this one an implementation that wired it into a TrackTrack-only branch would pass."""
    monkeypatch.setattr(tracking, "YOLO", _StubYOLO)

    result = track_video(make_video(fps="30000/1001", frames=3))

    assert result.tracker == "botsort.yaml"
    assert _StubYOLO.last_tracker_config["track_buffer"] == 90
    assert "__buf3__" in config_slug(result)


def test_a_non_positive_buffer_is_an_error(monkeypatch: pytest.MonkeyPatch, make_video: MakeVideo) -> None:
    """The floor would otherwise swallow it, leaving a run configured for nothing it asked for."""
    monkeypatch.setattr(tracking, "YOLO", _StubYOLO)

    with pytest.raises(ValueError, match="buffer_seconds must be positive"):
        track_video(make_video(frames=3), buffer_seconds=0)


def test_track_video_records_settings_the_tracker_lacks_as_none(
    monkeypatch: pytest.MonkeyPatch, make_video: MakeVideo
) -> None:
    """BoT-SORT has none of TrackTrack's association settings, and the record says so rather than
    inventing values. This is what keeps them out of a BoT-SORT slug."""
    monkeypatch.setattr(tracking, "YOLO", _StubYOLO)

    result = track_video(make_video(frames=3), tracker="botsort.yaml")

    assert (result.lost_match_thr, result.iou_weight, result.reid_weight) == (None, None, None)
    assert "lost" not in config_slug(result)


def test_track_video_records_the_effective_tracktrack_settings(
    monkeypatch: pytest.MonkeyPatch, make_video: MakeVideo
) -> None:
    """Recorded from the built config, so an overridden run and a defaulted one both say what ran."""
    monkeypatch.setattr(tracking, "YOLO", _StubYOLO)

    tuned = track_video(
        make_video(frames=3),
        tracker="tracktrack.yaml",
        lost_match_thr=0.8,
        iou_weight=0.4,
        reid_weight=0.6,
    )

    assert (tuned.lost_match_thr, tuned.iou_weight, tuned.reid_weight) == (0.8, 0.4, 0.6)
    assert _StubYOLO.last_tracker_config["lost_match_thr"] == 0.8

    defaulted = track_video(make_video(frames=3), tracker="tracktrack.yaml")
    assert (defaulted.lost_match_thr, defaulted.iou_weight, defaulted.reid_weight) == (0.0, 0.5, 0.5)


def test_track_video_records_the_effective_thresholds(monkeypatch: pytest.MonkeyPatch, make_video: MakeVideo) -> None:
    """Recorded from the built config, so a default run still says what the tracker used."""
    monkeypatch.setattr(tracking, "YOLO", _StubYOLO)

    defaulted = track_video(make_video(frames=3), tracker="tracktrack.yaml")
    assert (defaulted.track_low_thresh, defaulted.new_track_thresh) == (0.25, 0.7)

    lowered = track_video(make_video(frames=3), tracker="tracktrack.yaml", track_low_thresh=0.05)
    assert (lowered.track_low_thresh, lowered.new_track_thresh) == (0.05, 0.7)
    assert _StubYOLO.last_tracker_config["track_low_thresh"] == 0.05


def test_unsupported_tracker_type_is_rejected(tmp_path: Path) -> None:
    custom = tmp_path / "custom.yaml"
    YAML.save(str(custom), {"tracker_type": "not-a-tracker"})

    with pytest.raises(ValueError, match="Ultralytics supports"):
        build_tracker_config(str(custom), "none")


def test_config_slug_describes_the_run(make_video: MakeVideo) -> None:
    """The output filename comes from this, so it must reflect the record and stay path-safe."""
    tracks = VideoTracks(
        source=make_video(frames=1),
        width=64,
        height=48,
        fps="30000/1001",
        frames=(),
        weights="yolo26x.pt",
        tracker="tracktrack.yaml",
        reid="auto",
        imgsz=1280,
        conf=0.05,
        buffer_seconds=3.0,
        track_buffer=90,
        track_low_thresh=0.05,
        new_track_thresh=0.7,
        lost_match_thr=0.8,
        iou_weight=0.4,
        reid_weight=0.6,
    )

    assert config_slug(tracks) == (
        "yolo26x__tracktrack__reid-auto__imgsz1280__conf0.05__buf3__low0.05__new0.7__lost0.8__iouw0.4__reidw0.6"
    )

    # A ReID given as a path must not leak separators or extensions into a filename.
    slug = config_slug(replace(tracks, reid="/models/my reid.onnx"))
    assert slug == (
        "yolo26x__tracktrack__reid-my-reid__imgsz1280__conf0.05__buf3__low0.05__new0.7__lost0.8__iouw0.4__reidw0.6"
    )
    assert "/" not in slug

    # Settings the tracker does not have are left out rather than written as a placeholder, which is
    # what keeps a BoT-SORT slug free of TrackTrack's.
    bare = replace(
        tracks,
        track_low_thresh=None,
        new_track_thresh=None,
        lost_match_thr=None,
        iou_weight=None,
        reid_weight=None,
    )
    assert config_slug(bare) == "yolo26x__tracktrack__reid-auto__imgsz1280__conf0.05__buf3"

    # The buffer is slugged as the seconds asked for, never as the frames they became: the frame
    # count varies per clip, so slugging it would rename one configuration on every clip.
    assert config_slug(replace(tracks, track_buffer=20)) == config_slug(tracks)


def test_track_video_records_its_configuration(monkeypatch: pytest.MonkeyPatch, make_video: MakeVideo) -> None:
    """Non-default values, because a field that is recorded but never threaded through from the
    arguments would pass a defaults-only check while mislabelling every output."""
    monkeypatch.setattr(tracking, "YOLO", _StubYOLO)

    result = track_video(
        make_video(frames=3),
        weights="yolo26x.pt",
        tracker="tracktrack.yaml",
        reid="auto",
        imgsz=1280,
        conf=0.05,
    )

    assert (result.weights, result.tracker, result.reid) == ("yolo26x.pt", "tracktrack.yaml", "auto")
    assert (result.imgsz, result.conf) == (1280, 0.05)
    assert _StubYOLO.last_kwargs["imgsz"] == 1280
    assert _StubYOLO.last_kwargs["conf"] == 0.05
    # Nothing but the buffer was passed, so the slug carries TrackTrack's own values for the rest --
    # including the 0.0 that means its relaxed rebind pass is switched off. A slug states what ran,
    # not what was typed, so a defaulted setting is named just as an overridden one is.
    assert config_slug(result) == (
        "yolo26x__tracktrack__reid-auto__imgsz1280__conf0.05__buf3__low0.25__new0.7__lost0__iouw0.5__reidw0.5"
    )


def test_track_video_hands_the_tracker_a_live_config(monkeypatch: pytest.MonkeyPatch, make_video: MakeVideo) -> None:
    """The generated YAML must still be readable when Ultralytics gets to it, which with
    stream=True is during iteration rather than at the call."""
    monkeypatch.setattr(tracking, "YOLO", _StubYOLO)

    track_video(make_video(frames=3), tracker="tracktrack.yaml", reid="auto")

    assert _StubYOLO.last_tracker_config["tracker_type"] == "tracktrack"
    assert _StubYOLO.last_tracker_config["with_reid"] is True
    assert _StubYOLO.last_tracker_config["model"] == "auto"
