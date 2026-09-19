"""Detection and tracking of people and vehicles in a video.

`track_video` runs an Ultralytics detector and tracker over a clip and returns a compact record of
the per-frame boxes and track ids. The record is small enough to keep in memory and to hand to
`human_vehicle.overlay` for rendering.

The detector, tracker and ReID model are all chosen by plain strings. See `track_video`.
"""

import json
import re
import shutil
import subprocess
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
    be named after the run rather than after what the caller believes it configured. The tracker
    settings are the values the tracker actually used, whether they were overridden or left at its
    defaults, and are None where the tracker has no such setting at all. `reid` is the string that
    was requested; for every `yolo26*` detector that is also what ran, but Ultralytics can resolve
    "auto" to a separate encoder for an end2end detector without that showing here.

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
    conf: float
    buffer_seconds: float
    track_buffer: int
    track_low_thresh: float | None
    new_track_thresh: float | None
    lost_match_thr: float | None
    iou_weight: float | None
    reid_weight: float | None


def build_tracker_config(
    tracker: str,
    reid: str,
    *,
    track_buffer: int | None = None,
    track_low_thresh: float | None = None,
    new_track_thresh: float | None = None,
    lost_match_thr: float | None = None,
    iou_weight: float | None = None,
    reid_weight: float | None = None,
) -> dict[str, Any]:
    """Return the tracker configuration to run, with the requested overrides applied.

    `tracker` names a tracker Ultralytics ships ("botsort.yaml", "tracktrack.yaml", ...) or points
    at a custom YAML. `reid` is "none" for no ReID stage, "auto" to reuse the detector's own
    backbone features, or the name or path of a ReID model.

    Every override is left at the tracker's own default when None, because the defaults differ per
    tracker and there is no single value to fall back to. Setting one the tracker does not have
    raises, since it would otherwise be a silent no-op.

    `track_buffer` is a frame count, which is what the tracker wants — deriving it from a duration
    needs the clip's frame rate, and that belongs to `track_video`, not here.

    `lost_match_thr`, `iou_weight` and `reid_weight` exist only on TrackTrack. Values are passed
    through as given: the weights need not sum to anything in particular, and no range is enforced.

    None of this can be passed to `model.track()` as an argument: Ultralytics reads it out of the
    tracker YAML. So the settings are applied to a copy of that config here, and `track_video`
    writes the result to a file for Ultralytics to read back.
    """
    # str(): check_yaml is typed as returning a list for list input, which this call cannot pass.
    config: dict[str, Any] = YAML.load(str(check_yaml(tracker)))

    tracker_type = config.get("tracker_type")
    if tracker_type not in TRACKER_MAP:
        raise ValueError(
            f"{tracker} declares tracker_type={tracker_type!r}; Ultralytics supports {sorted(TRACKER_MAP)}"
        )

    overrides: tuple[tuple[str, int | float | None], ...] = (
        ("track_buffer", track_buffer),
        ("track_low_thresh", track_low_thresh),
        ("new_track_thresh", new_track_thresh),
        ("lost_match_thr", lost_match_thr),
        ("iou_weight", iou_weight),
        ("reid_weight", reid_weight),
    )
    for key, value in overrides:
        if value is None:
            continue
        if key not in config:
            raise ValueError(f"{tracker} ({tracker_type}) has no {key}, so setting it would be silently ignored")
        config[key] = value

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

    For example
    "yolo26x__tracktrack__reid-auto__imgsz1280__conf0.1__buf3__low0.25__new0.7__lost0.8__iouw0.4__reidw0.6".

    Every tracker setting is included because they are what decide the output, so two runs that
    differ only in one of them must not land on the same filename. A setting the tracker does not
    have is left out rather than written as a placeholder, which is why a BoT-SORT slug is shorter
    than a TrackTrack one.

    The buffer appears as the seconds that were asked for, not the frame count they became. The
    frame count depends on the clip, so slugging it would give one configuration a different name on
    every clip, and could give two different configurations the same name on a clip where the floor
    flattens both.
    """
    slug = (
        f"{_slug_part(tracks.weights)}__{_slug_part(tracks.tracker)}"
        f"__reid-{_slug_part(tracks.reid)}__imgsz{tracks.imgsz}__conf{tracks.conf:g}"
        f"__buf{tracks.buffer_seconds:g}"
    )
    optional = (
        ("low", tracks.track_low_thresh),
        ("new", tracks.new_track_thresh),
        ("lost", tracks.lost_match_thr),
        ("iouw", tracks.iou_weight),
        ("reidw", tracks.reid_weight),
    )
    for name, value in optional:
        if value is not None:
            slug += f"__{name}{value:g}"
    return slug


# No fewer frames than this, however few seconds they span. At 6 fps a three-second window is 18
# frames, and a tracker given that little has very few chances to re-associate whatever it lost.
MIN_TRACK_BUFFER = 20


def _track_buffer_frames(buffer_seconds: float, fps: str) -> int:
    """How many frames a lost track survives, for a clip at `fps`.

    Ultralytics reads `track_buffer` as a frame count and never scales it by the frame rate, so a
    single number means wildly different things across clips: the shipped default of 30 is one
    second at 30 fps and five at 6 fps. Occlusion is measured in seconds, so the duration is what
    the caller gives and the frame count is derived here.

    `fps` is the exact rational ffprobe reports; `_probe` has already rejected the rates that would
    make this zero.
    """
    if buffer_seconds <= 0:
        raise ValueError(f"buffer_seconds must be positive, not {buffer_seconds!r}")
    return max(MIN_TRACK_BUFFER, round(buffer_seconds * float(Fraction(fps))))


def require_binary(name: str) -> str:
    """Return the path to an external binary, or raise a clear error naming how to install it.

    Lives here rather than in `overlay` because `track_video` needs ffprobe before any model work
    starts, and `overlay` imports from this module anyway.
    """
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name!r} is not on PATH; install it with 'brew install ffmpeg'")
    return path


def _probe(source: Path) -> tuple[int, int, str]:
    """Return (width, height, fps) for a video's first video stream.

    The frame rate is `avg_frame_rate` (frames over duration) rather than `r_frame_rate` (the base
    rate needed to express every timestamp), because the render writes constant-frame-rate output
    and `avg_frame_rate` is the rate that reproduces the source's duration.
    """
    completed = subprocess.run(
        [
            require_binary("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate",
            "-of",
            "json",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {source}: {completed.stderr.strip()}")

    streams = json.loads(completed.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"{source} has no video stream")

    stream = streams[0]
    fps = str(stream["avg_frame_rate"])
    if fps in {"0/0", "0/1"}:
        raise ValueError(f"{source} reports no usable average frame rate ({fps!r})")
    return int(stream["width"]), int(stream["height"]), fps


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
    tracker: str = "botsort.yaml",
    reid: str = "none",
    imgsz: int = 640,
    conf: float = 0.1,
    buffer_seconds: float = 3.0,
    track_low_thresh: float | None = None,
    new_track_thresh: float | None = None,
    lost_match_thr: float | None = None,
    iou_weight: float | None = None,
    reid_weight: float | None = None,
    device: str = "mps",
) -> VideoTracks:
    """Detect and track people and vehicles through a clip.

    The model choices are plain strings:

    - `weights`: any Ultralytics detector, e.g. "yolo26n.pt" through "yolo26x.pt". Downloads on
      first use.
    - `tracker`: a tracker Ultralytics ships ("botsort.yaml", "tracktrack.yaml", "bytetrack.yaml",
      "deepocsort.yaml", "ocsort.yaml", "fasttrack.yaml") or a path to your own YAML.
    - `reid`: "none" for no ReID stage, "auto" to reuse the detector's own backbone features, or a
      model such as "yolo26s-reid.onnx". Only BoT-SORT, TrackTrack and Deep OC-SORT have a ReID
      stage; asking the others for one raises rather than being ignored.

    `device` defaults to "mps" because Ultralytics' automatic device selection falls through to the
    CPU on macOS unless MPS is asked for by name. `imgsz` is worth raising to 960 or 1280 on 4K
    footage, where 640 misses small, distant people.

    `conf` is the detector's threshold, defaulting to the 0.1 Ultralytics itself uses in track mode
    (deliberately lower than its 0.25 for plain prediction, so the tracker's low-score association
    stage has weak detections to work with). It is usually *not* what decides the output, though:
    the two tracker thresholds below are, and a detection under `track_low_thresh` is discarded
    whatever `conf` let through.

    - `track_low_thresh`: the weakest detection the tracker will attach to a track it is already
      following. Lowering it buys continuity through occlusion and motion blur — objects keep their
      id instead of being dropped and re-acquired — at the risk of a track coasting on a bad box.
    - `new_track_thresh`: how confident a detection must be to *start* a track. This is the one
      that governs whether weak detections can create new objects, so lowering `track_low_thresh`
      alone extends existing tracks without inventing new ones.

    Both default to None, meaning the tracker's own value: the shipped defaults differ per tracker
    (TrackTrack 0.25/0.7, BoT-SORT 0.1/0.25), so there is no single sensible fallback.

    `buffer_seconds` is how long a lost track stays re-findable before its id is retired, and is
    the other half of how long an id survives an occlusion. It is given in seconds because that is
    what an occlusion is measured in; the tracker's own `track_buffer` is a frame count it never
    scales by the frame rate, so a fixed one means a five-fold different window across clips
    shot at 6 fps and 30 fps. The derived frame count is floored at `MIN_TRACK_BUFFER`.

    The last three are TrackTrack's, and asking any other tracker for them raises:

    - `lost_match_thr`: gate for a second, looser association pass that tries already-lost tracks
      against detections nothing else claimed. TrackTrack ships 0.0, which switches the pass off
      entirely. Setting it a little above `match_thresh` gives a lost track a second chance to be
      rebound under its original id instead of a new track being started.
    - `iou_weight` and `reid_weight`: how the association cost is split between where a track was
      predicted to be and what it looked like. Favouring appearance helps across a gap, where the
      predicted box has been coasting and is the less trustworthy of the two, and costs a little
      accuracy in the ordinary frame-to-frame case where the box is excellent.
    """
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"no such video: {source_path}")

    # Probed first: the buffer is a frame count derived from this clip's rate.
    width, height, fps = _probe(source_path)
    track_buffer = _track_buffer_frames(buffer_seconds, fps)

    tracker_config = build_tracker_config(
        tracker,
        reid,
        track_buffer=track_buffer,
        track_low_thresh=track_low_thresh,
        new_track_thresh=new_track_thresh,
        lost_match_thr=lost_match_thr,
        iou_weight=iou_weight,
        reid_weight=reid_weight,
    )

    model = YOLO(weights)
    _check_class_names(model.names)

    # The tracker config has to reach Ultralytics as a file, and stream=True makes model.track()
    # lazy, so the file is read when iteration starts rather than when the call is made. Hence the
    # whole generator is consumed inside this scope.
    with tempfile.TemporaryDirectory() as directory:
        tracker_path = Path(directory) / "tracker.yaml"
        YAML.save(str(tracker_path), tracker_config)
        frames = _tracked_frames(model, source_path, tracker_path, imgsz=imgsz, conf=conf, device=device)

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
        conf=conf,
        buffer_seconds=buffer_seconds,
        track_buffer=track_buffer,
        # The effective values, not the arguments: a record should say what the tracker used. A
        # tracker without one of these settings records None, which keeps it out of the slug.
        track_low_thresh=tracker_config.get("track_low_thresh"),
        new_track_thresh=tracker_config.get("new_track_thresh"),
        lost_match_thr=tracker_config.get("lost_match_thr"),
        iou_weight=tracker_config.get("iou_weight"),
        reid_weight=tracker_config.get("reid_weight"),
    )


def _tracked_frames(
    model: YOLO, source: Path, tracker_path: Path, *, imgsz: int, conf: float, device: str
) -> tuple[FrameTracks, ...]:
    """Run the tracker over the clip and convert every result, one decoded frame at a time."""
    # stream=True keeps one decoded frame alive at a time; the non-streaming call would retain
    # every frame of the clip, which is around 15 GB for a 4K minute.
    results = model.track(
        str(source),
        tracker=str(tracker_path),
        stream=True,
        imgsz=imgsz,
        conf=conf,
        device=device,
        classes=sorted(COCO_CLASSES),
        save=False,
        verbose=False,
    )
    return tuple(FrameTracks(index=index, boxes=_to_boxes(result)) for index, result in enumerate(results))
