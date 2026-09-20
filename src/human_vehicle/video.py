"""What this project asks of ffmpeg and ffprobe: find the binaries, and ask about a file.

Every ffprobe call in the project is here, so the two questions worth asking of a clip -- its
stream's shape and its duration -- are answered in one place and phrased the same way. ffmpeg is
not the same story: `require_binary` lives here, but `overlay` and `vlm` build and run their own
ffmpeg commands, because what they ask it to do is theirs.

Nothing first-party is imported here, so anything in the package may depend on this module.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def require_binary(name: str) -> str:
    """Return the path to an external binary, or raise a clear error naming how to install it."""
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name!r} is not on PATH; install it with 'brew install ffmpeg'")
    return path


def _ffprobe(source: Path, *entries: str) -> Any:
    """Ask ffprobe `entries` about `source` and return the parsed JSON.

    The transport both probes below share: the argv, the run, and the one failure that is ffprobe's
    rather than the file's. What the answer means is the caller's business.

    A `JSONDecodeError` propagates. ffprobe exiting zero and then writing something unparseable is
    not a case either caller handles, and inventing a friendlier error for it here would be a
    behaviour change dressed as plumbing.
    """
    completed = subprocess.run(
        [require_binary("ffprobe"), "-v", "error", *entries, "-of", "json", str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {source}: {completed.stderr.strip()}")
    return json.loads(completed.stdout)


def probe_stream(source: Path) -> tuple[int, int, str]:
    """Return (width, height, fps) for a video's first video stream.

    The frame rate is `avg_frame_rate` (frames over duration) rather than `r_frame_rate` (the base
    rate needed to express every timestamp), because the render writes constant-frame-rate output
    and `avg_frame_rate` is the rate that reproduces the source's duration.
    """
    streams = _ffprobe(source, "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate").get(
        "streams", []
    )
    if not streams:
        raise ValueError(f"{source} has no video stream")

    stream = streams[0]
    fps = str(stream["avg_frame_rate"])
    if fps in {"0/0", "0/1"}:
        raise ValueError(f"{source} reports no usable average frame rate ({fps!r})")
    return int(stream["width"]), int(stream["height"]), fps


def probe_duration(video: Path) -> float:
    """The clip's duration in seconds, from ffprobe."""
    duration = _ffprobe(video, "-show_entries", "format=duration").get("format", {}).get("duration")
    if duration is None:
        raise ValueError(f"{video} reports no duration")
    seconds = float(duration)
    if seconds <= 0:
        raise ValueError(f"{video} reports a duration of {seconds}s")
    return seconds
