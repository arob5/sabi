"""Tempering: transform a `LogDensityForm` at a given tempering state.

The **tempering state** is an opaque PyTree — its type is tempering-strategy-
specific (a scalar β for likelihood tempering, a subset identifier for data
tempering, richer objects for more exotic bridges). Dispatch is keyed on
`(type(tempering), type(form))` so new strategies slot in without modifying
existing forms; the state is unpacked inside each registered implementation.

v0 ships `NoTempering` only. Real strategies (`LikelihoodTempering`,
`DataTempering`) land in later phases (design doc §4.11).
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sabi.problems.forms import LogDensityForm

_TemperImpl = Callable[["Tempering", LogDensityForm, Any], LogDensityForm]

_REGISTRY: dict[tuple[type, type], _TemperImpl] = {}


def register_tempering(
    tempering_cls: type,
    form_cls: type,
) -> Callable[[_TemperImpl], _TemperImpl]:
    """Register a `(Tempering, LogDensityForm)` → tempered form implementation."""

    def deco(fn: _TemperImpl) -> _TemperImpl:
        key = (tempering_cls, form_cls)
        if key in _REGISTRY:
            raise ValueError(f"Tempering already registered for {key}.")
        _REGISTRY[key] = fn
        return fn

    return deco


class Tempering(ABC):
    """Transforms a `LogDensityForm` at a given tempering state.

    Subclasses are typically parameter-free singletons; the state is passed to
    `apply`, not stored on the tempering object.
    """

    def apply(self, form: LogDensityForm, state: Any) -> LogDensityForm:
        key = (type(self), type(form))
        fn = _REGISTRY.get(key)
        if fn is None:
            raise NotImplementedError(
                f"No tempering registered for ({type(self).__name__}, "
                f"{type(form).__name__}). Use `register_tempering` to add one."
            )
        return fn(self, form, state)


@dataclass(frozen=True)
class NoTempering(Tempering):
    """Identity tempering: state is ignored, form is returned unchanged."""

    def apply(self, form: LogDensityForm, state: Any) -> LogDensityForm:
        return form
