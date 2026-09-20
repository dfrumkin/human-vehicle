"""Tests for what this project asks of ffprobe.

The happy paths are covered wherever a real clip is probed -- the tracker's tests, the renderer's,
and every test that runs `find_interactions`. What none of them reach is the failure below, which
is the one branch both probes share.
"""

import re
from pathlib import Path

import pytest

from human_vehicle.video import probe_duration


def test_a_file_ffprobe_cannot_read_fails_with_an_error_naming_it(tmp_path: Path) -> None:
    """The shared transport's one failure, and the only thing it must say.

    A caller holding this error is usually looking at a batch of clips, so which file failed is the
    whole of its value; what ffprobe writes after the colon varies by build and is not asserted.
    """
    not_a_video = tmp_path / "clip.mp4"
    not_a_video.write_text("this is not a video")

    # Escaped: `match` is a regex, and a temporary path is free to contain metacharacters.
    with pytest.raises(RuntimeError, match=rf"ffprobe failed on {re.escape(str(not_a_video))}:"):
        probe_duration(not_a_video)
