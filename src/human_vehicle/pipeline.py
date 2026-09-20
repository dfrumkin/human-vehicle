"""The whole run in one command: source clips in, annotated video and interactions out.

Each clip goes through the four stages the notebooks run by hand -- track, renumber, render, then
ask a vision-language model what human-vehicle interactions are in the render -- and leaves an
annotated mp4 and a JSON record on disk. Every stage is a call into the rest of the package; what
is here is the loop, the arguments, and the reporting.

The defaults are the ones the notebooks use, not the library's: `track_video` defaults to
`yolo26s.pt` with no ReID, where the work on these clips has been done with `yolo26x.pt` and a
matched ReID encoder.

A clip that runs through leaves two files side by side, both named after it:

    <output>/<clip>.mp4     the annotated render
    <output>/<clip>.json    the merged interactions

Nothing else is written. The run record -- the raw model text, the per-call usage, the per-window
detail -- is used here and then discarded, and an unsupported label is reported on stderr and
nowhere else. Names carry no configuration, so a second run over a clip replaces the first:
comparing two configurations means two output folders.
"""

import argparse
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

from human_vehicle.interactions import InteractionRun, find_interactions, verify_labels
from human_vehicle.labels import label_times, relabel_tracks
from human_vehicle.merge import merge_interactions, merge_summary
from human_vehicle.overlay import render_tracked_video
from human_vehicle.tracking import Category, track_video
from human_vehicle.video import probe_stream
from human_vehicle.vlm import DEFAULT_MODEL_ID, GeminiBackend, QwenBackend, VlmBackend

# What these clips have been run with, which is not what the library defaults to: the largest
# detector and a ReID encoder matched to it.
DEFAULT_WEIGHTS = "yolo26x.pt"
DEFAULT_TRACKER = "tracktrack.yaml"
DEFAULT_REID = "yolo26x-reid.onnx"
DEFAULT_BUFFER_SECONDS = 3.0

# Eight-second windows advancing four: every second is examined in a short context, and neighbours
# overlap by four, so an event shorter than that is seen whole by at least one window.
DEFAULT_WINDOW_S = 8.0
DEFAULT_STRIDE_S = 4.0

# Folders searched for a clip are searched for this and nothing else. Everything the project has
# been run on is mp4, and both notebooks glob for the same.
VIDEO_PATTERN = "*.mp4"

# Settings that belong to one backend and mean nothing to the other. Passing one to the backend
# that is not running is refused rather than ignored -- see `build_backend`.
GEMINI_ONLY = ("fps", "resolution", "thinking_level", "max_output_tokens")
LOCAL_ONLY = ("max_new_tokens",)

# Named in the error when `--vlm local` arrives without a model id. There is no default because the
# right id differs per machine, and `QwenBackend` refuses to guess for the same reason.
LOCAL_MODEL_EXAMPLES = "mlx-community/Qwen3.5-9B-4bit under MLX, Qwen/Qwen3.5-9B under transformers"


@dataclass(frozen=True, slots=True)
class TrackingSettings:
    """What the detection and tracking stage was asked for.

    `imgsz` is None for the per-clip rule in `imgsz_for` rather than a fixed size, which is how the
    notebook runs it: these clips range from 352x288 to 4K, and one input size cannot suit both.
    """

    weights: str = DEFAULT_WEIGHTS
    tracker: str = DEFAULT_TRACKER
    reid: str = DEFAULT_REID
    imgsz: int | None = None
    buffer_seconds: float = DEFAULT_BUFFER_SECONDS


@dataclass(frozen=True, slots=True)
class ClipResult:
    """What one clip produced, or why it produced nothing.

    A clip that raised carries `error` and nothing else; a clip whose calls all failed carries a
    `run` with its own error, which is a completed clip with an unusable answer. Both count as
    failures, because both leave the clip unanswered.
    """

    source: Path
    annotated: Path | None = None
    record: Path | None = None
    people: int = 0
    vehicles: int = 0
    run: InteractionRun | None = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None or (self.run is not None and self.run.error is not None)


def imgsz_for(height: int) -> int:
    """The detector's input size for a clip this tall.

    Ultralytics scales a frame's longest side to `imgsz`, so 640 loses the small, distant people in
    4K footage. Running the sub-HD clips at their native scale was measured and found about 10% more
    detections for two to four times the runtime, which is why the step is at 1080 rather than lower.
    """
    return 1280 if height >= 1080 else 640


def discover_videos(source: Path, output_dir: Path) -> list[Path]:
    """The clips to run: `source` itself if it is a file, else every mp4 directly inside it.

    A clip whose annotated render would be written over the clip itself is refused. The annotated
    file is named after its source now, so pointing the output at a clip's own folder -- an
    ordinary thing to do -- aims the render at the input. `render_tracked_video` refuses that too,
    so the clip is safe either way; refusing here is what makes it cost nothing, since the render's
    own check fires only after a tracking pass, which is minutes on a 4K clip.
    """
    if source.is_file():
        videos = [source]
    elif source.is_dir():
        videos = sorted(source.glob(VIDEO_PATTERN))
        if not videos:
            raise FileNotFoundError(f"no {VIDEO_PATTERN} files in {source}")
    else:
        raise FileNotFoundError(f"no such file or folder: {source}")

    for video in videos:
        if annotated_path(video, output_dir).resolve() == video.resolve():
            raise ValueError(f"the annotated clip would be written over {video}; use another output folder")
    return videos


def annotated_path(source: Path, output_dir: Path) -> Path:
    """Where `source`'s annotated render goes: beside the output folder's other clips, same name."""
    return output_dir / f"{source.stem}.mp4"


def _given(args: argparse.Namespace, names: Sequence[str]) -> list[str]:
    """Which of `names` the caller actually passed, spelled as flags.

    These all default to None precisely so that "left alone" and "set to the value that happens to
    be the default" can be told apart, which is what makes refusing a flag meant for the other
    backend possible at all.
    """
    return [f"--{name.replace('_', '-')}" for name in names if getattr(args, name) is not None]


def _refuse(flags: Sequence[str], reason: str) -> None:
    """Raise when flags were given that cannot apply, naming every one of them."""
    if flags:
        verb = "is" if len(flags) == 1 else "are"
        raise ValueError(f"{', '.join(flags)} {verb} {reason}")


def resolve_window(args: argparse.Namespace) -> tuple[float, float] | None:
    """The windowing to run with, or None for one call over the whole clip.

    The bounds are checked here rather than left to `find_interactions`, which checks them too: a
    batch would otherwise spend a tracking pass -- minutes, on a 4K clip -- before finding out that
    an argument was wrong.
    """
    if args.whole_clip:
        _refuse(_given(args, ("window_s", "stride_s")), "a windowing setting, and --whole-clip asks for no windows")
        return None

    window_s = DEFAULT_WINDOW_S if args.window_s is None else args.window_s
    stride_s = DEFAULT_STRIDE_S if args.stride_s is None else args.stride_s
    if window_s <= 0 or stride_s <= 0:
        raise ValueError(f"--window-s and --stride-s must be positive, not {window_s:g} and {stride_s:g}")
    if stride_s > window_s:
        raise ValueError(f"--stride-s {stride_s:g} > --window-s {window_s:g} would leave stretches of a clip unseen")
    return window_s, stride_s


def build_backend(args: argparse.Namespace) -> VlmBackend:
    """The vision-language backend the flags ask for.

    Neither backend loads anything here: the Gemini client and the local weights are both resolved
    on the first call, so a wrong argument further down still fails before any of it is paid for.
    """
    settings: dict[str, Any] = {
        name: getattr(args, name)
        for name in (GEMINI_ONLY if args.vlm == "gemini" else LOCAL_ONLY)
        if getattr(args, name) is not None
    }
    if args.seed is not None:
        settings["seed"] = args.seed

    if args.vlm == "gemini":
        _refuse(_given(args, LOCAL_ONLY), "a --vlm local setting, and --vlm gemini is running")
        # usecwd: a bare load_dotenv() searches upward from the *calling module's* file, which for
        # an installed console script is the package directory rather than where the user is.
        load_dotenv(find_dotenv(usecwd=True))
        return GeminiBackend(model_id=args.vlm_model or DEFAULT_MODEL_ID, **settings)

    _refuse(_given(args, GEMINI_ONLY), "a --vlm gemini setting, and --vlm local is running")
    if not args.vlm_model:
        raise ValueError(f"--vlm local needs --vlm-model; the right id differs per machine ({LOCAL_MODEL_EXAMPLES})")
    return QwenBackend(args.vlm_model, **settings)


def _unverified_summary(run: InteractionRun) -> str | None:
    """One line naming the labels the clip did not support, or None when it supported them all.

    Counted over `run.interactions` alone. Every accepted record also sits in its own window's list,
    so counting both would double every number in the line.
    """
    counts = Counter(
        label for item in run.interactions for label in (*item.unverified_person_ids, *item.unverified_vehicle_ids)
    )
    if not counts:
        return None
    records = sum(1 for item in run.interactions if item.unverified_person_ids or item.unverified_vehicle_ids)
    labels = ", ".join(f"{label} x{count}" for label, count in sorted(counts.items()))
    return f"{sum(counts.values())} unsupported label(s) in {records} record(s): {labels}"


def process_clip(
    source: Path,
    output_dir: Path,
    backend: VlmBackend,
    tracking: TrackingSettings,
    window: tuple[float, float] | None,
) -> ClipResult:
    """One clip, all the way through, with everything it wrote reported back.

    Raises if a stage does; `run_pipeline` is what keeps one bad clip from costing a batch the rest.
    """
    _, height, _ = probe_stream(source)
    imgsz = imgsz_for(height) if tracking.imgsz is None else tracking.imgsz

    started = time.perf_counter()
    tracks = track_video(
        source,
        weights=tracking.weights,
        tracker=tracking.tracker,
        reid=tracking.reid,
        imgsz=imgsz,
        buffer_seconds=tracking.buffer_seconds,
    )
    relabelled, translation = relabel_tracks(tracks)
    # One entry per identity the tracker found, so this counts objects rather than detections.
    people = sum(1 for category, _ in translation if category is Category.PERSON)

    annotated = annotated_path(source, output_dir)
    render_tracked_video(relabelled, annotated)
    print(
        f"  tracked  {people} people, {len(translation) - people} vehicles at imgsz={imgsz} "
        f"in {time.perf_counter() - started:.1f}s -> {annotated}"
    )

    window_s, stride_s = window if window is not None else (None, None)
    # The source clip names the run, not the file the model was shown. The two stems are the same
    # string today; saying which one is meant keeps that from being an accident.
    run = find_interactions(annotated, backend, window_s=window_s, stride_s=stride_s, clip_id=source.stem)

    # Against the record that was rendered, so the labels checked are the glyphs the model saw.
    run = verify_labels(run, label_times(relabelled), tolerance_s=backend.tolerance_s)
    unverified = _unverified_summary(run)
    if unverified is not None:
        # The only account of what the model claimed: the merged file carries the surviving ids and
        # has nowhere to put the quarantined ones, so a label invented here is said once, and here.
        print(f"{run.clip_id}: {unverified}", file=sys.stderr)

    note = run.error or (
        f"{len(run.interactions)} interaction(s), {len(run.malformed)} malformed"
        + (f", {run.failed_windows} call(s) FAILED" if run.failed_windows else "")
        + f", ${run.usage.get('estimated_cost_usd', 0.0):.4f}"
    )
    print(f"  asked    {note} in {run.elapsed_s:.1f}s")

    # A whole-clip run is merged too: `merge_interactions` runs on any run, and one call reporting
    # each event once simply leaves it nothing to collapse. One file shape, whichever way it ran.
    merged = merge_interactions(run)
    record = output_dir / f"{source.stem}.json"
    record.write_text(merged.model_dump_json(indent=2), encoding="utf-8")
    print(f"  merged   {merge_summary(merged)} -> {record}")

    return ClipResult(
        source=source,
        annotated=annotated,
        record=record,
        people=people,
        vehicles=len(translation) - people,
        run=run,
    )


def run_pipeline(
    videos: Sequence[Path],
    output_dir: Path,
    backend: VlmBackend,
    tracking: TrackingSettings,
    window: tuple[float, float] | None,
) -> list[ClipResult]:
    """Every clip, in order, one failure never costing the others.

    A clip that raises is reported and recorded, and the batch goes on: an unattended sweep should
    come back with a diagnosable account of every clip rather than a traceback from the third one.
    """
    results: list[ClipResult] = []
    for number, source in enumerate(videos, start=1):
        print(f"[{number}/{len(videos)}] {source}")
        try:
            results.append(process_clip(source, output_dir, backend, tracking, window))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED   {error}", file=sys.stderr)
            results.append(ClipResult(source=source, error=error))
    return results


def _summarize(results: Sequence[ClipResult]) -> None:
    """One line per clip, then what failed. A batch must not hide a bad clip among good ones."""
    header = f"{'clip':<34} {'people':>6} {'vehic':>6} {'inter':>6} {'malf':>5} {'fail':>5} {'usd':>8}  status"
    print(f"\n{header}\n{'-' * len(header)}")
    for result in results:
        run = result.run
        status = result.error or (run.error if run is not None else None) or "ok"
        cost = run.usage.get("estimated_cost_usd", 0.0) if run is not None else 0.0
        print(
            f"{result.source.stem[:34]:<34} {result.people:>6} {result.vehicles:>6} "
            f"{len(run.interactions) if run else 0:>6} {len(run.malformed) if run else 0:>5} "
            f"{run.failed_windows if run else 0:>5} {cost:>8.4f}  {status[:60]}"
        )

    failed = [result for result in results if result.failed]
    print(f"\n{len(results) - len(failed)}/{len(results)} clip(s) answered")
    for result in failed:
        print(f"  FAILED {result.source.stem}", file=sys.stderr)


def _delete_uploads(backend: VlmBackend) -> None:
    """Remove whatever this run uploaded.

    Gemini keeps an upload for 48 hours on its own, and the footage is of people, so an interrupted
    batch should not leave it sitting there. The local backend uploads nothing.
    """
    if not isinstance(backend, GeminiBackend):
        return
    for name in backend.delete_uploads():
        print(f"deleted upload {name}")


def _imgsz(value: str) -> int | None:
    """`--imgsz`: a positive size, or "auto" for the per-clip rule in `imgsz_for`."""
    if value == "auto":
        return None
    try:
        size = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number or 'auto', not {value!r}") from None
    if size <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, not {size}")
    return size


def build_parser() -> argparse.ArgumentParser:
    """The command line.

    The backend-specific settings and the windowing default to None rather than to their real
    defaults, which are stated in the help text: the value that ends up being used is then the
    backend's own -- one place, no copy here to drift from it -- and a flag that was passed can be
    told from one that was not, which is what lets a setting meant for the other backend be refused
    instead of silently ignored.
    """
    parser = argparse.ArgumentParser(
        prog="human-vehicle",
        description="Track people and vehicles through each clip, then read the human-vehicle "
        "interactions out of the annotated render with a vision-language model.",
    )
    parser.add_argument("input", type=Path, help=f"a video file, or a folder searched for {VIDEO_PATTERN}")
    parser.add_argument("output", type=Path, help="folder to write the annotated clips and records into")

    tracking = parser.add_argument_group("detection and tracking")
    tracking.add_argument("--weights", default=DEFAULT_WEIGHTS, help=f"detector (default: {DEFAULT_WEIGHTS})")
    tracking.add_argument("--tracker", default=DEFAULT_TRACKER, help=f"tracker (default: {DEFAULT_TRACKER})")
    tracking.add_argument(
        "--reid", default=DEFAULT_REID, help=f"'none', 'auto', or a ReID model (default: {DEFAULT_REID})"
    )
    tracking.add_argument(
        "--imgsz",
        type=_imgsz,
        default=None,
        metavar="SIZE",
        help="detector input size, or 'auto' for 1280 on frames 1080px tall or more, else 640 (default: auto)",
    )
    tracking.add_argument(
        "--buffer-seconds",
        type=float,
        default=DEFAULT_BUFFER_SECONDS,
        help=f"how long a lost track stays re-findable (default: {DEFAULT_BUFFER_SECONDS:g})",
    )

    model = parser.add_argument_group("vision-language model")
    model.add_argument("--vlm", choices=("gemini", "local"), default="gemini", help="which backend (default: gemini)")
    model.add_argument(
        "--vlm-model",
        metavar="ID",
        help=f"model id (default: {DEFAULT_MODEL_ID} for gemini; required for local, {LOCAL_MODEL_EXAMPLES})",
    )
    model.add_argument("--seed", type=int, default=None, help="passed to whichever backend runs (default: 1)")
    model.add_argument("--fps", type=float, default=None, help="gemini only: video sampling rate (default: 2)")
    model.add_argument(
        "--resolution", choices=("low", "medium", "high"), default=None, help="gemini only (default: low)"
    )
    model.add_argument(
        "--thinking-level", choices=("low", "medium", "high"), default=None, help="gemini only (default: medium)"
    )
    model.add_argument("--max-output-tokens", type=int, default=None, help="gemini only (default: 8192)")
    model.add_argument("--max-new-tokens", type=int, default=None, help="local only (default: 8192)")

    windowing = parser.add_argument_group("windowing")
    windowing.add_argument(
        "--window-s", type=float, default=None, help=f"window length (default: {DEFAULT_WINDOW_S:g})"
    )
    windowing.add_argument(
        "--stride-s", type=float, default=None, help=f"how far each window advances (default: {DEFAULT_STRIDE_S:g})"
    )
    windowing.add_argument(
        "--whole-clip", action="store_true", help="one call over the whole clip instead of windowing"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline. Returns 1 if any clip went unanswered, 0 otherwise."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        videos = discover_videos(args.input, args.output)
        window = resolve_window(args)
        backend = build_backend(args)
    except (ValueError, FileNotFoundError) as exc:
        # Exits 2, the same as any other bad argument, and before anything has been spent.
        parser.error(str(exc))

    tracking = TrackingSettings(
        weights=args.weights,
        tracker=args.tracker,
        reid=args.reid,
        imgsz=args.imgsz,
        buffer_seconds=args.buffer_seconds,
    )
    windowing = f"{window[0]:g}s/{window[1]:g}s windows" if window is not None else "whole clip"
    print(f"{len(videos)} clip(s) -> {args.output}")
    print(f"{tracking.weights} + reid={tracking.reid} | {backend.slug} | {windowing}\n")

    try:
        results = run_pipeline(videos, args.output, backend, tracking, window)
    finally:
        # In a finally: an interrupted batch has already uploaded the clips it got through, and
        # leaving them behind is the one failure here with a consequence off this machine.
        _delete_uploads(backend)

    _summarize(results)
    return 1 if any(result.failed for result in results) else 0


if __name__ == "__main__":  # pragma: no cover - the console script calls main() directly
    raise SystemExit(main())
