"""opi: shared library for the on-policy-distillation mechanistic study.

Subsystems (data / training / eval / mech) import from here so the couplet
prompt format, rhyme scoring, token-position conventions, and model loading stay
identical across every training regime.
"""
from . import io_utils, positions, prompts, rhyme  # noqa: F401

# ``models`` pulls in torch/transformers; import lazily so the pure-Python parts
# (rhyme, prompts, positions) are usable in environments without torch.
__all__ = ["rhyme", "prompts", "positions", "models", "io_utils"]


def __getattr__(name):  # PEP 562 lazy submodule import
    if name == "models":
        from . import models as _models
        return _models
    raise AttributeError(f"module 'opi' has no attribute {name!r}")
