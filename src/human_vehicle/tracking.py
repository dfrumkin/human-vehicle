"""Detection and tracking of people and vehicles in a video.

`track_video` runs an Ultralytics detector and tracker over a clip and returns a compact record of
the per-frame boxes and track ids. The record is small enough to keep in memory and to pass around whole.

The detector, tracker and ReID model are all chosen by plain strings. See `track_video`.
"""

import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Any

from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml

from human_vehicle.device import select_device
from human_vehicle.video import probe_stream


class Category(StrEnum):
    """The two kinds of thing this project cares about."""

    PERSON = "person"
    VEHICLE = "vehicle"


# The COCO classes we keep, mapped to the name the detector must report and the category they group
# into. Bicycles (id 1) are excluded: the interest here is people entering and exiting vehicles.
COCO_CLASSES: Mapping[int, tuple[str, Category]] = {
    0: ("person", Category.PERSON),
    2: ("car", Category.VEHICLE),
    3: ("motorcycle", Category.VEHICLE),
    5: ("bus", Category.VEHICLE),
    7: ("truck", Category.VEHICLE),
}


@dataclass(frozen=True, slots=True)
class TrackedBox:
    """One tracked object in one frame, in original-resolution pixel coordinates."""

    track_id: int
    category: Category
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class FrameTracks:
    """Everything tracked in a single frame. `boxes` is empty when nothing was tracked."""

    index: int
    boxes: tuple[TrackedBox, ...]


@dataclass(frozen=True, slots=True)
class VideoTracks:
    """A whole clip's tracking result, with the source metadata needed to re-render it.

    `frames` has one entry per decoded frame, in order, so indices line up with the video.
    `fps` is the exact rational ffprobe reports, e.g. "30000/1001".

    The trailing fields record the configuration that produced this record, so an output file can
    be named after the run rather than after what the caller believes it configured. `reid` is the
    string that was requested; for every `yolo26*` detector that is also what ran, but Ultralytics
    can resolve "auto" to a separate encoder for an end2end detector without that showing here.

    `buffer_seconds` is what was asked for and `track_buffer` the frame count it became for this
    clip's frame rate. Both are kept: the seconds are what identifies a configuration across clips,
    while the frames are what the tracker was actually given, and are the number worth having when
    asking why one clip held its ids and another did not.
    """

    source: Path
    width: int
    height: int
    fps: str
    frames: tuple[FrameTracks, ...]
    weights: str
    tracker: str
    reid: str
    imgsz: int
    buffer_seconds: float
    track_buffer: int


def build_tracker_config(tracker: str, reid: str, *, track_buffer: int | None = None) -> dict[str, Any]:
    """Return the tracker configuration to run, with the requested overrides applied.

    `tracker` names a tracker Ultralytics ships ("botsort.yaml", "tracktrack.yaml", ...) or points
    at a custom YAML. `reid` is "none" for no ReID stage, "auto" to reuse the detector's own
    backbone features, or the name or path of a ReID model.

    `track_buffer` is left at the tracker's own default when None. Setting it on a tracker that does
    not have it raises, since it would otherwise be a silent no-op. It is a frame count, which is
    what the tracker wants — deriving it from a duration needs the clip's frame rate, and that
    belongs to `track_video`, not here.

    Neither of these can be passed to `model.track()` as an argument: Ultralytics reads them out of
    the tracker YAML. So they are applied to a copy of that config here, and `track_video` writes
    the result to a file for Ultralytics to read back.
    """
    # str(): check_yaml is typed as returning a list for list input, which this call cannot pass.
    config: dict[str, Any] = YAML.load(str(check_yaml(tracker)))

    tracker_type = config.get("tracker_type")
    if tracker_type not in TRACKER_MAP:
        raise ValueError(
            f"{tracker} declares tracker_type={tracker_type!r}; Ultralytics supports {sorted(TRACKER_MAP)}"
        )

    if track_buffer is not None:
        if "track_buffer" not in config:
            raise ValueError(f"{tracker} ({tracker_type}) has no track_buffer, so setting it would be silently ignored")
        config["track_buffer"] = track_buffer

    # Whether a tracker has a ReID stage is read from its own config rather than from a list kept
    # here, so a future Ultralytics that adds one needs no change.
    supports_reid = "with_reid" in config

    if reid == "none":
        # Only where the key already exists: writing with_reid into ByteTrack's config would add a
        # field it never reads, making the effective configuration claim something untrue.
        if supports_reid:
            config["with_reid"] = False
        return config

    if not supports_reid:
        # The three names are for the reader, not for the logic above, which reads ReID support out
        # of the config itself. A future Ultralytics could add a fourth and still work.
        raise ValueError(
            f"{tracker} ({tracker_type}) has no ReID stage, so reid={reid!r} would be silently ignored; "
            f"use reid='none', or botsort.yaml, tracktrack.yaml or deepocsort.yaml"
        )

    config["with_reid"] = True
    config["model"] = reid
    return config


def _slug_part(value: str) -> str:
    """One filename-safe component of a configuration slug.

    `stem` drops the directory and extension, and the substitution handles what is left, such as
    spaces. Two models whose names differ only by extension collapse to the same part, which is
    accepted: keeping extensions would clutter every filename to guard a case that choosing among
    the yolo26 assets cannot produce.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "-", Path(value).stem)


def config_slug(tracks: VideoTracks) -> str:
    """A filename fragment naming the configuration that produced `tracks`.

    For example "yolo26x__tracktrack__reid-auto__imgsz1280__buf3".

    The buffer appears as the seconds that were asked for, not the frame count they became. The
    frame count depends on the clip, so slugging it would give one configuration a different name on
    every clip, and could give two different configurations the same name on a clip where the floor
    flattens both.
    """
    return (
        f"{_slug_part(tracks.weights)}__{_slug_part(tracks.tracker)}"
        f"__reid-{_slug_part(tracks.reid)}__imgsz{tracks.imgsz}__buf{tracks.buffer_seconds:g}"
    )


# No fewer frames than this, however few seconds they span. At 6 fps a three-second window is 18
# frames, and a tracker given that little has very few chances to re-associate whatever it lost.
MIN_TRACK_BUFFER = 20


def _track_buffer_frames(buffer_seconds: float, fps: str) -> int:
    """How many frames a lost track survives, for a clip at `fps`.

    Ultralytics reads `track_buffer` as a frame count and never scales it by the frame rate, so a
    single number means wildly different things across clips: the shipped default of 30 is one
    second at 30 fps and five at 6 fps. Occlusion is measured in seconds, so the duration is what
    the caller gives and the frame count is derived here.

    `fps` is the exact rational ffprobe reports, which `Fraction` parses directly; `video.probe_stream`
    has already rejected the degenerate rates, including the "0/0" that `Fraction` raises on rather
    than parses. The floor, not that check, is what keeps a very low rate from yielding a useless
    buffer.
    """
    if buffer_seconds <= 0:
        raise ValueError(f"buffer_seconds must be positive, not {buffer_seconds!r}")
    return max(MIN_TRACK_BUFFER, round(buffer_seconds * float(Fraction(fps))))


def _check_class_names(names: Mapping[int, str]) -> None:
    """Reject weights that do not use the COCO class ids `COCO_CLASSES` is keyed by.

    Without this, non-COCO weights would mislabel every box silently rather than failing.
    """
    mismatched = {
        class_id: names.get(class_id)
        for class_id, (expected, _) in COCO_CLASSES.items()
        if names.get(class_id) != expected
    }
    if mismatched:
        expected = {class_id: name for class_id, (name, _) in COCO_CLASSES.items()}
        raise ValueError(f"weights are not COCO-classed: expected {expected}, but these ids differ: {mismatched}")


def _to_boxes(result: Results) -> tuple[TrackedBox, ...]:
    """Convert one Ultralytics result to our boxes, dropping detections the tracker gave no id.

    Ultralytics feeds the tracker low-confidence detections by design, and leaves the raw
    detections in place for frames where the tracker confirmed nothing. Ids are the point of this
    output, so those frames become empty.
    """
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return ()

    tracked: list[TrackedBox] = []
    for track_id, class_id, confidence, xyxy in zip(
        boxes.id.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True
    ):
        class_name, category = COCO_CLASSES[int(class_id)]
        x1, y1, x2, y2 = xyxy
        tracked.append(
            TrackedBox(
                track_id=int(track_id),
                category=category,
                class_name=class_name,
                confidence=float(confidence),
                xyxy=(float(x1), float(y1), float(x2), float(y2)),
            )
        )
    return tuple(tracked)


def track_video(
    source: str | Path,
    *,
    weights: str = "yolo26s.pt",
    tracker: str = "tracktrack.yaml",
    reid: str = "none",
    imgsz: int = 640,
    buffer_seconds: float = 3.0,
) -> VideoTracks:
    """Detect and track people and vehicles through a clip.

    The model choices are plain strings:

    - `weights`: any Ultralytics detector, e.g. "yolo26n.pt" through "yolo26x.pt". Downloads on
      first use.
    - `tracker`: TrackTrack by default, or any other tracker Ultralytics ships ("botsort.yaml",
      "bytetrack.yaml", "deepocsort.yaml", "ocsort.yaml", "fasttrack.yaml"), or a path to your own
      YAML.
    - `reid`: "none" for no ReID stage, "auto" to reuse the detector's own backbone features, or a
      model such as "yolo26s-reid.onnx". Only BoT-SORT, TrackTrack and Deep OC-SORT have a ReID
      stage; asking the others for one raises rather than being ignored.

    The device is detected rather than chosen: `select_device` prefers CUDA, then MPS, then the
    CPU. Ultralytics has its own automatic selection, but it falls through to the CPU on macOS
    unless MPS is asked for by name. `imgsz` is worth raising to 960 or 1280 on 4K footage, where
    640 misses small, distant people.

    `buffer_seconds` is how long a lost track stays re-findable before its id is retired, which is
    what decides whether an id survives an occlusion. It is given in seconds because that is what
    an occlusion is measured in; the tracker's own `track_buffer` is a frame count it never scales
    by the frame rate, so a fixed one means a five-fold different window across clips shot at 6 fps
    and 30 fps. The derived frame count is floored at `MIN_TRACK_BUFFER`.

    Everything else the tracker reads is left at the value its own YAML ships, including the
    detection thresholds. The detector's `conf` is likewise Ultralytics' own: it uses 0.1 in track
    mode, deliberately below its 0.25 for plain prediction, so the low-score association stage has
    weak detections to work with.
    """
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"no such video: {source_path}")

    # Probed first: the buffer is a frame count derived from this clip's rate.
    width, height, fps = probe_stream(source_path)
    track_buffer = _track_buffer_frames(buffer_seconds, fps)

    tracker_config = build_tracker_config(tracker, reid, track_buffer=track_buffer)

    model = YOLO(weights)
    _check_class_names(model.names)

    # The tracker config has to reach Ultralytics as a file, and stream=True makes model.track()
    # lazy, so the file is read when iteration starts rather than when the call is made. Hence the
    # whole generator is consumed inside this scope.
    with tempfile.TemporaryDirectory() as directory:
        tracker_path = Path(directory) / "tracker.yaml"
        YAML.save(str(tracker_path), tracker_config)
        frames = _tracked_frames(model, source_path, tracker_path, imgsz=imgsz, device=select_device())

    return VideoTracks(
        source=source_path,
        width=width,
        height=height,
        fps=fps,
        frames=frames,
        weights=weights,
        tracker=tracker,
        reid=reid,
        imgsz=imgsz,
        buffer_seconds=buffer_seconds,
        track_buffer=track_buffer,
    )


def _tracked_frames(
    model: YOLO, source: Path, tracker_path: Path, *, imgsz: int, device: str
) -> tuple[FrameTracks, ...]:
    """Run the tracker over the clip and convert every result, one decoded frame at a time."""
    # stream=True keeps one decoded frame alive at a time; the non-streaming call would retain
    # every frame of the clip, which is around 15 GB for a 4K minute.
    #
    # No conf: Ultralytics applies 0.1 in track mode itself, so passing it would only restate the
    # default.
    results = model.track(
        str(source),
        tracker=str(tracker_path),
        stream=True,
        imgsz=imgsz,
        device=device,
        classes=sorted(COCO_CLASSES),
        save=False,
        verbose=False,
    )
    return tuple(FrameTracks(index=index, boxes=_to_boxes(result)) for index, result in enumerate(results))
