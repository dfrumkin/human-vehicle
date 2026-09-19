"""Tests for drawing tracked boxes and rendering an annotated clip."""

# Label placement is deliberately private: it is how `draw_boxes` works, not something a caller
# should reach for. The tests reach for it anyway, because asserting that two labels miss each other
# is exact against their rectangles and guesswork against anti-aliased pixels.
# pyright: reportPrivateUsage=false

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from human_vehicle import overlay
from human_vehicle.labels import label_text
from human_vehicle.overlay import (
    _TEXT_THICKNESS,
    _label_layout,
    _metrics,
    _Rect,
    draw_boxes,
    render_tracked_video,
)
from human_vehicle.tracking import COCO_CLASSES, Category, FrameTracks, TrackedBox, VideoTracks
from tests.conftest import MakeVideo

FPS = "30000/1001"

# The 35 boxes a detector found in one frame of gt1125_06.mp4, as (track_id, left, top, right,
# bottom, class_name). This is the frame the overlay's pixel budget was measured on, so the budget
# test has to use these exact boxes -- and these exact ids, since a label's width is set by its
# digit count.
DENSE_FRAME = (3840, 2160)
DENSE_BOXES = (
    (1, 2079, 603, 2579, 851, "truck"),
    (2, 2220, 1052, 2556, 1218, "car"),
    (3, 2328, 1567, 2656, 1743, "car"),
    (4, 2350, 1984, 2770, 2158, "car"),
    (5, 2216, 1302, 2554, 1457, "car"),
    (6, 2228, 955, 2470, 1080, "car"),
    (7, 1703, 577, 1823, 729, "car"),
    (8, 2114, 235, 2322, 337, "car"),
    (9, 1689, 761, 1819, 938, "car"),
    (10, 3647, 510, 3839, 634, "car"),
    (11, 1973, 1334, 2023, 1419, "person"),
    (12, 2648, 65, 2834, 149, "car"),
    (13, 2162, 475, 2399, 568, "car"),
    (14, 2757, 283, 2992, 391, "car"),
    (15, 2970, 655, 3259, 787, "car"),
    (16, 2359, 1153, 2406, 1255, "person"),
    (17, 1737, 1420, 1784, 1511, "person"),
    (18, 2130, 21, 2278, 105, "car"),
    (19, 2680, 141, 2855, 223, "car"),
    (20, 2601, 0, 2765, 58, "car"),
    (21, 3571, 376, 3838, 471, "car"),
    (22, 3602, 430, 3839, 537, "car"),
    (23, 2180, 544, 2393, 615, "car"),
    (24, 2705, 197, 2914, 276, "car"),
    (25, 3236, 222, 3291, 312, "motorcycle"),
    (26, 1662, 86, 1833, 173, "car"),
    (27, 1534, 961, 1597, 1029, "person"),
    (28, 2730, 244, 2952, 327, "car"),
    (29, 1750, 1037, 1939, 1292, "car"),
    (30, 2100, 0, 2273, 28, "car"),
    (31, 1501, 1311, 1630, 1458, "car"),
    (32, 1421, 1442, 1721, 1741, "truck"),
    (33, 1750, 1037, 1939, 1292, "truck"),
    (34, 1862, 1113, 1905, 1175, "person"),
    (35, 1415, 865, 1600, 1441, "truck"),
)
# Measured at 2.1% for this implementation, against 3.7% for the heavier labels and full outlines
# before it and 11.2% for the filled plates before those. The ceiling sits above the first and below
# the second, with room for anti-aliasing to differ between OpenCV builds.
PIXEL_BUDGET = 0.03


def _probe(path: Path) -> dict[str, str]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return {key: str(value) for key, value in json.loads(completed.stdout)["streams"][0].items()}


def _box(track_id: int, category: Category, class_name: str) -> TrackedBox:
    return TrackedBox(
        track_id=track_id,
        category=category,
        class_name=class_name,
        confidence=0.9,
        xyxy=(8.0, 6.0, 40.0, 38.0),
    )


def _tracks(source: Path, *, width: int = 64, height: int = 48, frames: int = 5) -> VideoTracks:
    boxes = (_box(1, Category.PERSON, "person"), _box(2, Category.VEHICLE, "car"))
    return VideoTracks(
        source=source,
        width=width,
        height=height,
        fps=FPS,
        frames=tuple(FrameTracks(index=index, boxes=boxes) for index in range(frames)),
        weights="yolo26s.pt",
        tracker="botsort.yaml",
        reid="none",
        imgsz=640,
        conf=0.1,
        buffer_seconds=3.0,
        track_buffer=90,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        # BoT-SORT has no such settings, so a record of a BoT-SORT run leaves them unset.
        lost_match_thr=None,
        iou_weight=None,
        reid_weight=None,
    )


def test_render_preserves_source_format(make_video: MakeVideo, tmp_path: Path) -> None:
    """The whole point of the renderer: same resolution, same exact frame rate, same frame count."""
    source = make_video(width=64, height=48, fps=FPS, frames=5)

    output = render_tracked_video(_tracks(source), tmp_path / "annotated.mp4")

    assert output.is_file()
    probed = _probe(output)
    assert probed["width"] == "64"
    assert probed["height"] == "48"
    assert probed["avg_frame_rate"] == FPS
    assert probed["nb_frames"] == "5"


@pytest.mark.parametrize("record_frames", [4, 6])
def test_render_rejects_frame_count_mismatch(make_video: MakeVideo, tmp_path: Path, record_frames: int) -> None:
    """A record that is not this video's must fail loudly, not misattribute boxes."""
    source = make_video(frames=5)
    output = tmp_path / "annotated.mp4"

    with pytest.raises(ValueError, match="frames than the tracking record"):
        render_tracked_video(_tracks(source, frames=record_frames), output)

    # A truncated clip is detected partway through encoding; no half-written mp4 may survive it.
    assert list(tmp_path.glob("annotated*")) == []


def test_render_rejects_dimension_mismatch(make_video: MakeVideo, tmp_path: Path) -> None:
    """Wrongly sized raw bytes would otherwise reach ffmpeg and produce garbage."""
    source = make_video(width=64, height=48)
    output = tmp_path / "annotated.mp4"

    with pytest.raises(ValueError, match="tracking record says"):
        render_tracked_video(_tracks(source, width=128, height=96), output)

    assert list(tmp_path.glob("annotated*")) == []


def test_render_rejects_an_out_of_order_record(make_video: MakeVideo, tmp_path: Path) -> None:
    """Frame entries are positional; a shuffled record would draw boxes on the wrong frames."""
    source = make_video(frames=3)
    shuffled = _tracks(source, frames=3)
    reordered = (shuffled.frames[0], shuffled.frames[2], shuffled.frames[1])
    tracks = replace(shuffled, frames=reordered)

    with pytest.raises(ValueError, match="not in frame order"):
        render_tracked_video(tracks, tmp_path / "annotated.mp4")


def test_render_refuses_to_overwrite_the_source(make_video: MakeVideo) -> None:
    """Input clips are local data that exists nowhere else; the render must not land on one."""
    source = make_video(frames=3)
    before = source.read_bytes()

    with pytest.raises(ValueError, match="refusing to overwrite the source"):
        render_tracked_video(_tracks(source, frames=3), source)

    assert source.read_bytes() == before


def test_render_reports_ffmpeg_failure(make_video: MakeVideo, tmp_path: Path) -> None:
    """ffmpeg's own diagnostics must reach the caller, not a bare BrokenPipeError."""
    source = make_video(frames=3)

    # An extension ffmpeg cannot map to a muxer makes it exit before reading any frame.
    with pytest.raises(RuntimeError, match="Unable to choose an output format"):
        render_tracked_video(_tracks(source, frames=3), tmp_path / "annotated.xyz")

    assert list(tmp_path.glob("annotated*")) == []


def test_render_rejects_odd_dimensions(make_video: MakeVideo, tmp_path: Path) -> None:
    """yuv420p cannot encode odd dimensions; say so rather than failing inside ffmpeg."""
    source = make_video()

    with pytest.raises(ValueError, match="even dimensions"):
        render_tracked_video(_tracks(source, width=65, height=48), tmp_path / "annotated.mp4")


def _categorized(class_name: str) -> Category:
    """The category `tracking` pairs with `class_name`, so a test box cannot contradict itself."""
    return next(category for name, category in COCO_CLASSES.values() if name == class_name)


def _tracked(track_id: int, left: int, top: int, right: int, bottom: int, class_name: str) -> TrackedBox:
    return TrackedBox(
        track_id=track_id,
        category=_categorized(class_name),
        class_name=class_name,
        confidence=0.9,
        xyxy=(float(left), float(top), float(right), float(bottom)),
    )


def _changed_pixels(annotated: np.ndarray, original: np.ndarray) -> int:
    """How many pixels the overlay touched. A pixel differing in any channel counts once."""
    return int(np.any(annotated != original, axis=2).sum())


def test_draw_boxes_is_localized() -> None:
    """Boxes and labels must be visible, and everything else must be left exactly as it was."""
    frame = np.full((200, 200, 3), 127, dtype=np.uint8)
    original = frame.copy()
    box = _tracked(7, 50, 50, 100, 100, "person")

    annotated = draw_boxes(frame, [box])

    # The box is marked at its corners, and the label sits just above the top-left one.
    assert annotated[50, 50].tolist() != original[50, 50].tolist()
    label = _label_layout([box], 200, 200, _metrics(200, 200))[0].rect
    assert label.bottom <= 50
    assert np.any(annotated[label.top : label.bottom, label.left : label.right] != 127)
    # Everything well away from the box is byte-identical: the overlay does not degrade the image.
    assert np.array_equal(annotated[150:200, 150:200], original[150:200, 150:200])
    # The caller's frame is untouched.
    assert np.array_equal(frame, original)


def test_overlay_stays_within_its_pixel_budget() -> None:
    """The requirement the overlay exists to satisfy: leave the video visible underneath it.

    Filled plates put this at 11.2% on this frame, which is what made the annotated clip unreadable
    to a model. The canvas is mid-grey rather than black so the labels' black halo registers as a
    change like everything else.
    """
    width, height = DENSE_FRAME
    frame = np.full((height, width, 3), 127, dtype=np.uint8)
    boxes = [_tracked(*fields) for fields in DENSE_BOXES]

    annotated = draw_boxes(frame, boxes)

    covered = _changed_pixels(annotated, frame) / (width * height)
    assert covered < PIXEL_BUDGET, f"overlay covers {covered:.1%} of the frame, budget is {PIXEL_BUDGET:.0%}"


@pytest.mark.parametrize(
    ("edge", "box"),
    [
        ("top", (200, 0, 280, 60)),
        ("bottom", (200, 419, 280, 479)),
        ("left", (0, 200, 80, 260)),
        ("right", (600, 200, 639, 260)),
    ],
)
def test_a_label_at_a_frame_edge_is_drawn_whole(edge: str, box: tuple[int, int, int, int]) -> None:
    """A label must not be cut in half by the frame, whichever edge its box is against.

    Asserting that the label's drawn pixels land inside the frame would prove nothing, because
    OpenCV clips drawing to the image silently. So the test counts the glyph ink where the label was
    placed and compares it against an identical label in the middle of the frame: one losing glyphs
    to the frame edge comes up short. Counting inside the halo margin rather than the whole
    rectangle keeps the box outline out of the window, so only label ink is compared.

    A three-digit id, as a long clip produces, makes the label wide enough that the right-hand box
    cannot hold it without being slid back into the frame.
    """
    width, height = 640, 480
    metrics = _metrics(width, height)
    frame = np.full((height, width, 3), 127, dtype=np.uint8)

    def ink(left: int, top: int, right: int, bottom: int) -> int:
        tracked = _tracked(123, left, top, right, bottom, "person")
        rect = _label_layout([tracked], width, height, metrics)[0].rect
        assert rect.left >= 0 and rect.right <= width, f"{edge}: label runs off the frame horizontally"
        assert rect.top >= 0 and rect.bottom <= height, f"{edge}: label runs off the frame vertically"
        window = (
            slice(rect.top + metrics.halo, rect.bottom - metrics.halo),
            slice(rect.left + metrics.halo, rect.right - metrics.halo),
        )
        return _changed_pixels(draw_boxes(frame, [tracked])[window], frame[window])

    at_edge, centred = ink(*box), ink(300, 200, 380, 260)
    # Exact equality holds on the OpenCV this is developed against, since the rasteriser translates
    # a glyph unchanged. The tolerance is for other builds, and is far below the quarter of the ink
    # a clipped glyph would cost.
    assert abs(at_edge - centred) <= 0.02 * centred, f"{edge} edge: {at_edge} ink against {centred} centred"


def _is_attached(rect: _Rect, box: TrackedBox, clearance: int) -> bool:
    """Whether `rect` sits a clearance from an edge of `box` and aligned to one of its corners.

    This is what stops a de-conflicted label reading as the neighbour's: it may move, but only to
    another corner of the box it belongs to. A label inside the box is inset horizontally too, since
    there it runs alongside the box's own vertical outlines.
    """
    left, top, right, bottom = (round(value) for value in box.xyxy)
    above = rect.bottom == top - clearance
    below = rect.top == bottom + clearance
    inside = rect.top == top + clearance
    inset = clearance if inside else 0
    return (above or below or inside) and (rect.left == left + inset or rect.right == right - inset)


def test_labels_are_placed_off_each_other() -> None:
    """Two labels that would land on each other must not, or they merge into a plausible wrong id.

    A moved label still has to read as belonging to its own box, so this pins the attachment too:
    each one stays a clearance from an edge of its box and flush with one of its sides.
    """
    width, height = 1920, 1080
    metrics = _metrics(width, height)
    # Same top edge and lefts a few pixels apart, so both prefer the same position above the box.
    boxes = [_tracked(1, 400, 300, 700, 600, "person"), _tracked(2, 405, 300, 705, 600, "car")]

    labels = _label_layout(boxes, width, height, metrics)

    # Placed on their own the two labels would land on top of each other, so this is a real
    # collision rather than two boxes that happened to be far enough apart.
    alone = [_label_layout([box], width, height, metrics)[0].rect for box in boxes]
    assert alone[0].intersects(alone[1])

    assert [label.text for label in labels] == ["P1", "V2"]
    assert not labels[0].rect.intersects(labels[1].rect)
    for label, box in zip(labels, boxes, strict=True):
        assert _is_attached(label.rect, box, metrics.clearance), f"{label.text} floated away from its box"


def test_every_box_keeps_a_label_when_none_can_be_placed_freely() -> None:
    """A crowd must cost legibility, never a label: an unlabelled box is untraceable."""
    width, height = 1920, 1080
    boxes = [_tracked(track_id, 400, 300, 700, 600, "car") for track_id in range(1, 10)]

    labels = _label_layout(boxes, width, height, _metrics(width, height))

    assert [label.text for label in labels] == [f"V{track_id}" for track_id in range(1, 10)]


@pytest.mark.parametrize(
    ("size", "thickness", "font_scale", "halo"),
    [
        ((3840, 2160), 4, 2.0, 2),
        ((1280, 720), 2, 720 / 1080, 1),
        ((640, 360), 2, 0.5, 1),
    ],
)
def test_drawing_sizes_are_the_ones_that_were_measured(
    size: tuple[int, int], thickness: int, font_scale: float, halo: int
) -> None:
    """The sizes are the deliverable, so they are pinned rather than left to the pixel budget.

    The budget test would pass a half-done job — a smaller halo with the old stroke weight would
    still fit under the ceiling — and these are the numbers that were read off the 640px downscale a
    model sees. At 4K they are what changed; at the two smaller sizes the point is the opposite one,
    that the floors hold and those clips keep what they had.
    """
    metrics = _metrics(*size)

    # Stroke weight is pinned too, and it is the one that did *not* move. Corner brackets save ink
    # by drawing less line, not fainter line; thinning it instead would wash the boxes out at the
    # downscale while making the pixel budget look better, so nothing else here would notice.
    assert metrics.thickness == thickness
    assert metrics.font_scale == pytest.approx(font_scale)
    assert metrics.halo == halo


def test_glyph_stroke_is_the_thinnest() -> None:
    """The halo carries a label against the background, so the stroke is as light as it goes."""
    assert _TEXT_THICKNESS == 1


def test_a_box_is_marked_at_its_corners() -> None:
    """Corners, not a cage: the extent still reads, but the object underneath is left alone.

    Run at a frame size that draws with the 4K stroke width, which is where this matters and where
    the pixel budget cannot see it — reduced labels inside full rectangles still fit under the
    ceiling, so only this test stands between the boxes and a silent return to outlines.
    """
    width = height = 1920
    frame = np.full((height, width, 3), 127, dtype=np.uint8)
    left, top, right, bottom = 200, 200, 599, 599

    annotated = draw_boxes(frame, [_tracked(7, left, top, right, bottom, "car")])
    painted = np.any(annotated != frame, axis=2)

    # Every corner is drawn, `right`/`bottom` included: that is the extent a full outline reached,
    # and it is where a one-pixel slip would hide.
    for x in (left, right):
        for y in (top, bottom):
            assert painted[y, x], f"corner ({x}, {y}) is not drawn"

    # Both arms are drawn at each corner. Ten pixels in is well inside an arm here (they run a
    # quarter of a 400px side) and well clear of the other arm's width, so this distinguishes a real
    # bracket from edges that were simply never drawn.
    assert painted[top + 10, left] and painted[top, left + 10], "the top-left corner is missing an arm"
    assert painted[bottom - 10, right] and painted[bottom, right - 10], "the bottom-right corner is missing an arm"

    # ... and the middle of all four sides is left open.
    mid_x, mid_y = (left + right) // 2, (top + bottom) // 2
    assert not painted[top, mid_x], "the top edge is closed"
    assert not painted[bottom, mid_x], "the bottom edge is closed"
    assert not painted[mid_y, left], "the left edge is closed"
    assert not painted[mid_y, right], "the right edge is closed"


def test_corner_marks_reach_the_same_extent_as_an_outline() -> None:
    """The corners must sit where the full outline's did, to the pixel.

    A box rectangle's far edges are inclusive, because that is how OpenCV draws, while the label
    rectangles beside them in this module are half-open. Measuring a box with the wrong one of those
    two conventions moves it by a pixel, which a four-pixel stroke would hide. So the extent is
    compared against the outline it replaced: `cv2.rectangle` given the same corners is what a box
    used to be, and is an oracle the bracket code has no hand in.
    """
    width = height = 1920
    frame = np.full((height, width, 3), 127, dtype=np.uint8)
    left, top, right, bottom = 200, 200, 599, 599

    annotated = draw_boxes(frame, [_tracked(7, left, top, right, bottom, "car")])
    outlined = frame.copy()
    cv2.rectangle(outlined, (left, top), (right, bottom), (0, 140, 255), _metrics(width, height).thickness)

    def extent(image: np.ndarray, window: tuple[slice, slice], axis: int) -> tuple[int, int]:
        """The first and last row or column `image` paints inside `window`."""
        drawn = np.nonzero(np.any(np.any(image[window] != 127, axis=2), axis=axis))[0]
        return int(drawn.min()), int(drawn.max())

    # A band across the top edge catches the two top arms, and the whole top edge of the outline.
    # It starts below the label, which stops a clearance above the box, so only box ink is counted.
    top_band = (slice(top - 2, top + 3), slice(None))
    assert extent(annotated, top_band, 0) == extent(outlined, top_band, 0), "the corners miss the outline's width"

    # The right edge for the other axis, because the label is left-aligned and nowhere near it.
    right_band = (slice(None), slice(right - 2, right + 3))
    assert extent(annotated, right_band, 1) == extent(outlined, right_band, 1), "the corners miss the outline's height"


def test_a_box_too_small_to_show_a_gap_is_outlined() -> None:
    """Below the cutoff the corners would meet, so the box is drawn as the rectangle it becomes.

    `cv2.line` paints about half its thickness past each endpoint, so arms that stop short of each
    other by no more than the stroke width still join up. Drawing such a box as brackets would be
    the same rectangle assembled from overlapping pieces.
    """
    width = height = 1920
    frame = np.full((height, width, 3), 127, dtype=np.uint8)
    left, top, right, bottom = 200, 200, 211, 211

    annotated = draw_boxes(frame, [_tracked(7, left, top, right, bottom, "car")])
    painted = np.any(annotated != frame, axis=2)

    mid_x, mid_y = (left + right) // 2, (top + bottom) // 2
    assert painted[top, mid_x] and painted[bottom, mid_x], "a small box has gaps along its sides"
    assert painted[mid_y, left] and painted[mid_y, right], "a small box has gaps along its sides"


def test_a_label_names_the_category_not_the_class() -> None:
    """The overlay draws exactly what `labels.label_text` says, for every class the project keeps.

    The COCO class a box was detected as no longer reaches the label: a car, a truck, a bus and a
    motorcycle are all `V`. Drawing it from `class_name` again -- `C4`, `T4` -- would put a key in
    the video that neither the README nor the translation knows about.
    """
    width, height = 1920, 1080
    boxes = [_tracked(4, 400, 300, 700, 600, class_name) for class_name, _ in COCO_CLASSES.values()]

    labels = _label_layout(boxes, width, height, _metrics(width, height))

    assert [label.text for label in labels] == [label_text(box.category, box.track_id) for box in boxes]
    assert {label.text for label in labels} == {"P4", "V4"}


def test_confidence_is_drawn_only_when_asked_for() -> None:
    """The debugging view, and the precision it has to have.

    Three decimals is not cosmetic. This overlay gets read against TrackTrack's 0.6 and 0.7 gates,
    and at two decimals 0.596 prints as "0.60" -- a detection that missed `track_high_thresh` and
    carried no ReID embedding would look like one that cleared it. The 0.596 here fails that way if
    the format is ever shortened.
    """
    width, height = 1920, 1080
    box = replace(_tracked(7, 400, 300, 700, 600, "person"), confidence=0.596)

    off = _label_layout([box], width, height, _metrics(width, height))
    on = _label_layout([box], width, height, _metrics(width, height), show_confidence=True)

    assert [label.text for label in off] == ["P7"]
    assert [label.text for label in on] == ["P7 0.596"]


def test_confidence_labels_reach_the_drawn_frame() -> None:
    """`draw_boxes` has to pass the flag down, not just accept it.

    Asserting through the public entry point rather than a stub: a wider label paints more pixels,
    so more coverage is proof the string actually reached the drawing.
    """
    frame = np.full((480, 640, 3), 127, dtype=np.uint8)
    boxes = [_tracked(7, 200, 150, 400, 400, "person")]

    plain = _changed_pixels(draw_boxes(frame, boxes), frame)
    with_confidence = _changed_pixels(draw_boxes(frame, boxes, show_confidence=True), frame)

    assert with_confidence > plain


def test_render_passes_the_confidence_flag_to_the_drawing(
    monkeypatch: pytest.MonkeyPatch, make_video: MakeVideo, tmp_path: Path
) -> None:
    """A flag accepted by the renderer and dropped before the drawing looks exactly like a working
    one from a notebook, and every frame would come out plain."""
    source = make_video(frames=3)
    drawn = overlay.draw_boxes
    seen: list[bool] = []

    def _spy(frame: np.ndarray, boxes: list[TrackedBox], *, show_confidence: bool = False) -> np.ndarray:
        seen.append(show_confidence)
        return drawn(frame, boxes, show_confidence=show_confidence)

    monkeypatch.setattr(overlay, "draw_boxes", _spy)
    render_tracked_video(_tracks(source, frames=3), tmp_path / "annotated.mp4", show_confidence=True)

    assert seen == [True, True, True]
