"""ATPO: Adaptive Tversky Policy Optimization for multi-label video safety classification."""

from __future__ import annotations

__version__ = "0.1.0.dev0"

__all__ = ["VideoSafetyModel", "__version__"]


def __getattr__(name: str):
    # Imported lazily so that `import atpo` does not require the inference extra.
    if name == "VideoSafetyModel":
        from .inference.model import VideoSafetyModel

        return VideoSafetyModel
    raise AttributeError(f"module 'atpo' has no attribute {name!r}")
