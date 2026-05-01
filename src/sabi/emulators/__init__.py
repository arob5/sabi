"""Emulators: fittable predictive models for the target map.

Subpackage layout — emulators live under per-backend subdirectories so
that backend-specific dependencies stay scoped:

- `sabi.emulators.tinygp`: lightweight tinygp-backed emulators. Always
  available.
- `sabi.emulators.gpjax`: gpjax-backed emulators (DSP-prior GP, etc.).
  Requires the optional ``gpjax`` extra.

Top-level convenience re-exports are kept for the always-available
backends (`TinyGPEmulator` here is the tinygp-backed one); gpjax-backed
emulators are imported from their full path so that touching this
namespace doesn't trigger a missing-optional-dep error.
"""

from sabi.emulators.base import Emulator
from sabi.emulators.dispatch import (
    EmulatorUpdateMethod,
    emulator_update_registry,
    update_emulator,
)
from sabi.emulators.gp import GPEmulator
from sabi.emulators.tinygp.gp import TinyGPEmulator
from sabi.emulators.updates import (
    AppendRows,
    EmulatorUpdate,
    RescaleOutputs,
    RescaleThenAppend,
)

# Register the GPEmulator-typed cheap-update dispatch handlers.
# Side-effect import: the module's bottom registers the handlers
# with ``emulator_update_registry`` exactly once. No backend
# dependency — the handlers dispatch on ``isinstance(em, GPEmulator)``
# at call time.
from sabi.emulators import _handlers  # noqa: E402, F401

__all__ = [
    "AppendRows",
    "Emulator",
    "EmulatorUpdate",
    "EmulatorUpdateMethod",
    "GPEmulator",
    "TinyGPEmulator",
    "RescaleOutputs",
    "RescaleThenAppend",
    "emulator_update_registry",
    "update_emulator",
]
