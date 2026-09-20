"""Which compute device this machine has, decided in one place.

Both callers -- the Ultralytics tracker and the transformers VLM runtime -- take the same device
strings, so one answer serves both.
"""

from typing import Any


def select_device(torch: Any | None = None) -> str:
    """The fastest device available: CUDA, else MPS, else the CPU.

    torch is imported on call rather than at module scope. `human_vehicle.vlm` is imported
    unconditionally by `human_vehicle.interactions` and defers loading torch until something is
    actually run; a module-level import here would undo that.

    Pass `torch` to reuse a module already in hand -- `TorchRuntime` holds one in its entry points,
    which is also how a test reaches this code on a machine with no GPU.

    Landing on the CPU is not warned about here, because how alarming that is depends on the
    caller: minutes per VLM call looks hung, where CPU tracking is merely slow.
    """
    if torch is None:
        import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
