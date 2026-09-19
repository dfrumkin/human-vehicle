"""Smoke tests for the project environment."""

import importlib
import importlib.util
import platform
from importlib.metadata import version

import pytest

# The local backend needs `qwen3_5`, which transformers gained in 5.17.
MIN_TRANSFORMERS = (5, 17)

APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"


def test_project_package_is_importable() -> None:
    """The editable src-layout install makes first-party code importable from anywhere."""
    importlib.import_module("human_vehicle")


@pytest.mark.parametrize(
    "module_name", ["cv2", "PIL", "matplotlib", "ultralytics", "lap", "onnx", "onnxruntime", "torch", "transformers"]
)
def test_declared_dependency_is_importable(module_name: str) -> None:
    """Every library the homework depends on everywhere is present in the synced environment."""
    importlib.import_module(module_name)


def test_the_local_runtime_for_this_platform_is_installed() -> None:
    """Exactly one local runtime is installed, and it is the one this platform can use.

    `pyproject.toml` decides this by environment marker, so a broken marker would leave the local
    backend with nothing to run on. Only the *selected* runtime is asserted: `transformers` is
    importable on Apple Silicon too, since `mlx-vlm` depends on it, so its absence is what this can
    detect and its presence proves nothing about the split.
    """
    importlib.import_module("mlx_vlm" if APPLE_SILICON else "accelerate")


def test_transformers_is_new_enough_for_qwen3_5() -> None:
    """The version matters, not just the import, and on Apple Silicon it arrives second-hand.

    There `mlx-vlm` supplies transformers with a floor of only 5.14, and the MLX path needs
    `qwen3_5` as much as the torch one does -- the tokenizer and video processor are transformers
    code either way. A too-old version imports perfectly and fails when the processor is built,
    which is why this asserts the number rather than the import.
    """
    installed = tuple(int(part) for part in version("transformers").split(".")[:2])
    assert installed >= MIN_TRANSFORMERS, f"transformers {version('transformers')} predates qwen3_5 support"

    assert importlib.util.find_spec("transformers.models.qwen3_5") is not None
