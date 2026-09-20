"""Tests for device detection.

A stub stands in for torch: no machine has all three devices, so the real one can only ever
exercise one branch, and which one depends on where the suite runs.
"""

from typing import Any

from human_vehicle.device import select_device


def _torch(*, cuda: bool, mps: bool) -> Any:
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return cuda

    class _Mps:
        @staticmethod
        def is_available() -> bool:
            return mps

    class _Backends:
        mps = _Mps()

    class _Torch:
        cuda = _Cuda()
        backends = _Backends()

    return _Torch()


def test_cuda_wins_over_mps() -> None:
    """The order is the whole point: both available means the faster one is taken."""
    assert select_device(_torch(cuda=True, mps=True)) == "cuda"


def test_mps_when_there_is_no_cuda() -> None:
    assert select_device(_torch(cuda=False, mps=True)) == "mps"


def test_the_cpu_when_there_is_no_accelerator() -> None:
    assert select_device(_torch(cuda=False, mps=False)) == "cpu"
