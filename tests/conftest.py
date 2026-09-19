"""Shared test fixtures."""

import subprocess
from pathlib import Path
from typing import Protocol

import pytest


class MakeVideo(Protocol):
    """Builds a small synthetic clip on disk and returns its path."""

    def __call__(
        self,
        name: str = ...,
        *,
        width: int = ...,
        height: int = ...,
        fps: str = ...,
        frames: int = ...,
    ) -> Path: ...


@pytest.fixture
def make_video(tmp_path: Path) -> MakeVideo:
    """Generate fixtures with ffmpeg rather than cv2.VideoWriter.

    VideoWriter takes the frame rate as a float, so a clip written that way could not carry an
    exact rate like 30000/1001 and the frame-rate assertions would be testing rounding instead of
    the code. ffmpeg is a documented prerequisite of this project, so these tests fail rather than
    skip when it is missing: a silent skip would hide the renderer's only automated coverage.
    """

    def _make_video(
        name: str = "clip.mp4",
        *,
        width: int = 64,
        height: int = 48,
        fps: str = "30000/1001",
        frames: int = 5,
    ) -> Path:
        path = tmp_path / name
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"testsrc=size={width}x{height}:rate={fps}",
                "-frames:v",
                str(frames),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
        )
        return path

    return _make_video
