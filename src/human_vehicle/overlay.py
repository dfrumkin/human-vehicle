"""Drawing tracked boxes onto frames, and rendering an annotated copy of a clip.

The annotated video is meant to be read by a vision-language model, so the overlay has to make the
track id and the category unambiguous while covering as little of the original image as it can.
Every pixel it paints is a pixel the model cannot see, which is why a label is a category letter and
a number rather than a word on a filled plate. The labels themselves come from `human_vehicle.labels`.
"""

import contextlib
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
from cv2.typing import MatLike

from human_vehicle.labels import label_text
from human_vehicle.tracking import Category, TrackedBox, VideoTracks, require_binary

# Category is carried by color as well as by the label's letter. Azure and orange stay high-contrast
# against typical street scenes and remain distinguishable under the common forms of color
# blindness. Both are light, which is what lets them read as bright glyphs over a black halo.
# Values are BGR, as OpenCV expects.
_COLORS: dict[Category, tuple[int, int, int]] = {
    Category.PERSON: (255, 160, 0),
    Category.VEHICLE: (0, 140, 255),
}
_HALO_COLOR = (0, 0, 0)
_FONT = cv2.FONT_HERSHEY_DUPLEX
# The eight directions the halo is drawn in: the same string offset by one halo width in each of
# them surrounds the glyphs. A thicker stroke would be the obvious way to do this, but OpenCV 5
# ignores `thickness` past 2 for the Hershey fonts, so the halo has to be built from offsets.
_HALO_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
# Glyph stroke weight. Fixed rather than scaled with the frame: what carries a label against the
# background is the halo, not the weight of the stroke, so the thinnest stroke is the right one at
# every size. (It is also the only real choice -- OpenCV 5 ignores Hershey stroke weight above 2.)
_TEXT_THICKNESS = 1
# How much of each side a corner bracket runs along. A quarter from each end leaves the middle half
# of the side open, which is enough to read the box's extent from its corners alone.
_ARM_FRACTION = 0.25


@dataclass(frozen=True, slots=True)
class _Metrics:
    """Drawing sizes for one frame size, in pixels except for `font_scale`."""

    thickness: int
    font_scale: float
    halo: int

    @property
    def clearance(self) -> int:
        """How far a label's ink stays from the box edge it is attached to.

        Half the outline plus a pixel. `cv2.rectangle` straddles the coordinate it is given, so this
        is the smallest gap that leaves the glyphs clear of the outline rather than merged with it.
        """
        return self.thickness // 2 + 1


def _metrics(width: int, height: int) -> _Metrics:
    """Return drawing sizes scaled to the frame.

    The clips in this project run from 352x288 to 3840x2160, so fixed sizes would be either
    invisible at one end or overwhelming at the other. The 2px outline floor keeps box edges from
    becoming hairlines, which 4:2:0 chroma subsampling smears, and the 0.5 font floor is what stops
    the sub-VGA clips from shrinking to nothing.

    `font_scale` is set to the smallest size that survives the downscale into a vision-language
    model -- a 4K frame goes to 640px on its long side -- with a margin. Read at that resolution,
    2.0 is comfortable and 1.8 still legible; by 1.6 neighbouring labels merge into each other. The
    floors mean only the 4K footage is affected by the choice at all.
    """
    short_side = min(width, height)
    return _Metrics(
        thickness=max(2, round(short_side / 540)),
        font_scale=max(0.5, short_side / 1080),
        halo=max(1, round(short_side / 1080)),
    )


@dataclass(frozen=True, slots=True)
class _Rect:
    """A pixel rectangle.

    Whether the far edges count as inside depends on who made it, and the two producers differ.
    Label rectangles are half-open, so their width is `right - left`. Box rectangles from
    `_box_rect` are inclusive, because that is how OpenCV draws: the corner handed to `cv2.rectangle`
    is painted, so a box's width is `right - left + 1`. Anything measuring a box has to use the
    second convention or it loses a pixel at each far edge.
    """

    left: int
    top: int
    right: int
    bottom: int

    def intersects(self, other: "_Rect") -> bool:
        """Whether the two rectangles share a pixel. Touching edges do not count."""
        return (
            self.left < other.right and other.left < self.right and self.top < other.bottom and other.top < self.bottom
        )


@dataclass(frozen=True, slots=True)
class _Label:
    """One placed label: what to draw, where its ink lands, and where the text baseline starts."""

    text: str
    color: tuple[int, int, int]
    rect: _Rect
    origin: tuple[int, int]


def _box_rect(box: TrackedBox, width: int, height: int) -> _Rect:
    """`box` rounded to pixels and clamped inside a `width` x `height` frame, far edges inclusive."""
    left, top, right, bottom = (round(value) for value in box.xyxy)
    return _Rect(
        left=max(0, min(left, width - 1)),
        top=max(0, min(top, height - 1)),
        right=max(0, min(right, width - 1)),
        bottom=max(0, min(bottom, height - 1)),
    )


def _candidate_rects(box: _Rect, size: tuple[int, int], metrics: _Metrics, frame: tuple[int, int]) -> list[_Rect]:
    """Positions a label of `size` may take, best first, each attached to a corner of its own box.

    Every candidate sits one clearance from an edge of `box` and is aligned to one of its corners, so
    wherever a label ends up it still reads as belonging to this box rather than to a neighbour.
    Vertically it prefers above the box, then just inside the top edge, then below the bottom edge;
    within each of those, left-aligned before right-aligned, so a label stays as high on its box as
    it can before moving down.

    A candidate that would fall outside the frame vertically is dropped, because sliding it back in
    would detach it from the box. Horizontally it is clamped instead: that slides the label along the
    edge it is attached to, which keeps the attachment.
    """
    label_width, label_height = size
    frame_width, frame_height = frame
    # Each position is a top edge and the horizontal inset that goes with it. Only the one inside
    # the box needs an inset: there the label runs down alongside the box's own corner arms, and
    # without it the halo would sit on one of them. Above and below the box those arms point the
    # other way, so flush with the corner is already clear of them.
    placements = (
        (box.top - metrics.clearance - label_height, 0),
        (box.top + metrics.clearance, metrics.clearance),
        (box.bottom + metrics.clearance, 0),
    )
    rects: list[_Rect] = []
    for top, inset in placements:
        if top < 0 or top + label_height > frame_height:
            continue
        for aligned_left in (box.left + inset, box.right - inset - label_width):
            left = max(0, min(aligned_left, frame_width - label_width))
            rects.append(_Rect(left, top, left + label_width, top + label_height))
    return rects


def _draw_box(frame: MatLike, box: _Rect, color: tuple[int, int, int], metrics: _Metrics) -> None:
    """Mark `box` at its four corners rather than outlining it.

    A closed rectangle cages the object, and thinning it is not an option: `cv2.rectangle` quantises
    thickness in pairs, so the step below the width used here is thin enough to wash out once a 4K
    frame has been downscaled to the 640px a vision-language model reads. Drawing only the corners
    costs less ink than thinning does and stays crisp at that size, because the saving comes from
    less line rather than fainter line. The corners are where a box's extent is anyway, and they are
    also where its label attaches.

    Each arm runs a quarter of the way along its side, leaving the middle half open. A box too small
    to show that gap is outlined normally: `cv2.line` paints about half its thickness past each
    endpoint, so arms that stop short of each other by no more than the stroke width still meet, and
    the result would be a rectangle drawn as overlapping pieces.
    """
    # Inclusive, as `_Rect` explains: these are the painted extents, not half-open bounds.
    span_x = box.right - box.left + 1
    span_y = box.bottom - box.top + 1
    arm_x = max(metrics.thickness, round(span_x * _ARM_FRACTION))
    arm_y = max(metrics.thickness, round(span_y * _ARM_FRACTION))

    if span_x - 2 * arm_x <= metrics.thickness or span_y - 2 * arm_y <= metrics.thickness:
        cv2.rectangle(frame, (box.left, box.top), (box.right, box.bottom), color, metrics.thickness)
        return

    for x, toward_x in ((box.left, 1), (box.right, -1)):
        for y, toward_y in ((box.top, 1), (box.bottom, -1)):
            cv2.line(frame, (x, y), (x + toward_x * (arm_x - 1), y), color, metrics.thickness)
            cv2.line(frame, (x, y), (x, y + toward_y * (arm_y - 1)), color, metrics.thickness)


def _label_layout(
    boxes: Sequence[TrackedBox],
    width: int,
    height: int,
    metrics: _Metrics,
    *,
    show_confidence: bool = False,
) -> list[_Label]:
    """Decide where every label goes, before anything is drawn.

    Labels are placed greedily, each taking the first candidate that does not intersect one already
    placed. Bare glyphs are small enough to sit beside each other, which is what makes this worth
    doing: where a filled plate hid the label under it outright, two haloed strings that touch merge
    into a plausible but wrong id, and a model reading the frame has no way to tell.

    `show_confidence` makes each label four times as wide, and nothing here is told to expect that:
    the sizes come from `cv2.getTextSize` on whatever string was built, so placement adapts on its
    own. What does change is how often it succeeds -- wider labels intersect more of what is already
    placed, so more of them fall through to the detached fallback below. That is the price of the
    debugging view, and the reason it is not the default.

    The order is by identity -- `(track_id, category)` -- rather than by position or by whatever
    order the tracker emitted. A track's label should move only when the labels it collides with
    change, not whenever two objects cross, because a label that jumps between frames is exactly
    what a model reading successive frames cannot follow. The id alone is not a total order, since
    this project treats an identity as a `(category, track_id)` pair, so the category breaks ties.

    A box always gets a label, even when every candidate is taken or none fits the frame: an
    unlabelled box would be untraceable rather than merely crowded.
    """
    placed: list[_Rect] = []
    labels: list[_Label] = []

    for box in sorted(boxes, key=lambda box: (box.track_id, box.category)):
        text = label_text(box.category, box.track_id)
        if show_confidence:
            # Three decimals, not two: rounding a marginal detection to two hides the distinction
            # the number is being read for.
            text = f"{text} {box.confidence:.3f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, _FONT, metrics.font_scale, _TEXT_THICKNESS)
        size = (text_width + 2 * metrics.halo, text_height + baseline + 2 * metrics.halo)

        box_rect = _box_rect(box, width, height)
        candidates = _candidate_rects(box_rect, size, metrics, (width, height))
        if not candidates:
            # The frame has no room for the label above, inside or below the box, which takes a box
            # nearly as tall as the frame. Clamp the preferred position into view and accept the
            # detachment; a missing label would be worse than a misplaced one.
            left = max(0, min(box_rect.left, width - size[0]))
            top = max(0, min(box_rect.top - metrics.clearance - size[1], height - size[1]))
            candidates = [_Rect(left, top, left + size[0], top + size[1])]

        rect = next((rect for rect in candidates if not any(rect.intersects(other) for other in placed)), candidates[0])
        placed.append(rect)
        labels.append(
            _Label(
                text=text,
                color=_COLORS[box.category],
                rect=rect,
                origin=(rect.left + metrics.halo, rect.top + metrics.halo + text_height),
            )
        )

    return labels


def _draw_label(frame: MatLike, label: _Label, metrics: _Metrics) -> None:
    """Draw `label` as coloured glyphs over a black halo. Nothing is filled.

    What keeps an id readable over an arbitrary background is luminance contrast, and a dark halo
    behind a light glyph gives that while leaving the pixels between the strokes alone. Luma is
    stored at full resolution under 4:2:0, so the contrast survives chroma subsampling. The halo is
    opaque rather than blended: translucency is what would put the background back into the glyph.
    """
    x, y = label.origin
    for offset_x, offset_y in _HALO_OFFSETS:
        origin = (x + offset_x * metrics.halo, y + offset_y * metrics.halo)
        cv2.putText(frame, label.text, origin, _FONT, metrics.font_scale, _HALO_COLOR, _TEXT_THICKNESS, cv2.LINE_AA)
    cv2.putText(frame, label.text, (x, y), _FONT, metrics.font_scale, label.color, _TEXT_THICKNESS, cv2.LINE_AA)


def draw_boxes(frame: MatLike, boxes: Sequence[TrackedBox], *, show_confidence: bool = False) -> MatLike:
    """Return a copy of `frame` with each box cornered and labelled `"<category letter><id>"`.

    `P29` is person 29 and `V4` is vehicle 4. Colour repeats the same split, so the two read apart
    without reading the letter. The numbers are the record's own track ids; run the boxes through
    `labels.relabel_tracks` first for the 1-based numbering per category that a reader expects.

    Nothing is filled: a box is marked at its corners and a label is a few glyphs over a halo, so on
    a crowded 4K frame the overlay covers about 2% of the image.

    `show_confidence` appends each box's detection confidence -- `P29 0.531` -- which is for reading
    a clip while debugging, not for anything a model is meant to see. It is off by default because
    it takes a label from two characters to eight, and the pixel budget above is the requirement
    this overlay exists to satisfy. Expect labels to crowd and detach on busy frames when it is on.

    Boxes are drawn first and every label afterwards, so a label is never carved up by a box drawn
    over it.
    """
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    metrics = _metrics(width, height)

    for box in boxes:
        _draw_box(annotated, _box_rect(box, width, height), _COLORS[box.category], metrics)

    for label in _label_layout(boxes, width, height, metrics, show_confidence=show_confidence):
        _draw_label(annotated, label, metrics)

    return annotated


def render_tracked_video(
    tracks: VideoTracks, output_path: str | Path, *, crf: int = 15, show_confidence: bool = False
) -> Path:
    """Write an annotated copy of `tracks.source` at its original resolution and frame rate.

    CRF 15 is visually near-lossless for x264, and there is no scaling filter, so the only quality
    the output loses relative to the source is one generation of encoding.

    Frames are decoded with `cv2.VideoCapture`, the same decoder Ultralytics uses, which is what
    makes frame *n* of the render provably frame *n* of the tracking pass.

    `show_confidence` is handed to `draw_boxes`; see it for what the flag draws and why a clip meant
    to be read by a model should not have it set.
    """
    if tracks.width % 2 or tracks.height % 2:
        raise ValueError(f"yuv420p needs even dimensions, got {tracks.width}x{tracks.height}")

    ffmpeg = require_binary("ffmpeg")
    output = Path(output_path)
    # The clips this runs on are local input data that exists nowhere else, and the render ends in
    # a rename, so a mistyped destination would otherwise destroy the source.
    if output.resolve() == tracks.source.resolve():
        raise ValueError(f"refusing to overwrite the source video at {tracks.source}")
    output.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(tracks.source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {tracks.source} for decoding")

    # ffmpeg writes beside the destination and the result is moved into place only once every
    # frame has been encoded. A record that turns out not to match this video is detected partway
    # through, and without this the caller would be left holding a plausible-looking truncated mp4.
    partial = output.with_suffix(f".partial{output.suffix}")
    args = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{tracks.width}x{tracks.height}",
        "-framerate",
        tracks.fps,
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(partial),
    ]

    try:
        # ffmpeg's stderr goes to a file rather than a pipe: a pipe drained only after the last
        # frame is written can fill and deadlock the writer on a long clip.
        with tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=stderr_file)
            if process.stdin is None:  # pragma: no cover - guaranteed by stdin=PIPE
                raise RuntimeError("ffmpeg stdin was not piped")
            stdin = process.stdin
            broken_pipe = False
            try:
                for position, frame_tracks in enumerate(tracks.frames):
                    if frame_tracks.index != position:
                        raise ValueError(
                            f"tracking record is not in frame order: entry {position} is indexed {frame_tracks.index}"
                        )
                    decoded, frame = capture.read()
                    if not decoded:
                        raise ValueError(
                            f"{tracks.source} has fewer frames than the tracking record "
                            f"({len(tracks.frames)}); the record does not belong to this video"
                        )
                    if frame.shape[:2] != (tracks.height, tracks.width):
                        raise ValueError(
                            f"{tracks.source} decodes {frame.shape[1]}x{frame.shape[0]} frames, but the "
                            f"tracking record says {tracks.width}x{tracks.height}"
                        )
                    stdin.write(draw_boxes(frame, frame_tracks.boxes, show_confidence=show_confidence).tobytes())

                if capture.read()[0]:
                    raise ValueError(
                        f"{tracks.source} has more frames than the tracking record "
                        f"({len(tracks.frames)}); the record does not belong to this video"
                    )
            except BrokenPipeError:
                # ffmpeg died before it had taken every frame. Its stderr says why, so let the
                # checks below report that rather than a bare BrokenPipeError.
                broken_pipe = True
            finally:
                capture.release()
                with contextlib.suppress(BrokenPipeError):
                    stdin.close()
                returncode = process.wait()

            stderr_file.seek(0)
            stderr = stderr_file.read().decode(errors="replace").strip()

        if returncode != 0:
            raise RuntimeError(f"ffmpeg exited with {returncode}: {stderr}")
        if broken_pipe:
            raise RuntimeError(f"ffmpeg stopped reading before every frame was written: {stderr}")
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    partial.replace(output)
    return output
