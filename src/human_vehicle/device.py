"""Which compute device this machine has, decided in one place.

Every library this project runs a model through takes the same device strings, so one answer
serves them all.
"""

from typing import Any


def select_device(torch: Any | None = None) -> str:
    """The fastest device available: CUDA, else MPS, else the CPU.

    torch is imported on call rather than at module scope, so importing this module needs no torch
    installed and costs nothing until a device is actually asked for; a module-level import here
    would undo that for everything that depends on this module.

    Pass `torch` to reuse a module already in hand -- `TorchRuntime` holds one in its entry points,
    which is also how a test reaches this code on a machine with no GPU.

    Landing on the CPU is not warned about here, because how alarming that is depends on the
    workload: minutes per call looks hung, where a pass that is merely slower does not.
    """
    if torch is None:
        import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
