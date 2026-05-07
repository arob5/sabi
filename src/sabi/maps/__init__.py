"""sabi.maps — `Map` ABC, concrete maps, and `pushforward(map, dist)`.

Phase 2 primitives from ``docs/link_functions.md``. The `Map`
abstraction is intentionally aligned with what ProbPipe is converging
on; sabi's local op becomes a re-export when ProbPipe lands.
"""

from sabi.maps._affine import Affine, Constant, Identity
from sabi.maps._base import Compose, Map
from sabi.maps._dirac import Dirac
from sabi.maps._elementwise import (
    Exp,
    Log,
    LogSoftplus,
    LogSquare,
    Softplus,
    Square,
)
from sabi.maps._likelihood import GaussianLogLik, LogProb
from sabi.maps._pushforward import pushforward

__all__ = [
    # ABC
    "Map",
    "Compose",
    # Linear / constant
    "Identity",
    "Constant",
    "Affine",
    # Elementwise
    "Exp",
    "Log",
    "Softplus",
    "LogSoftplus",
    "Square",
    "LogSquare",
    # Likelihood / log-prob
    "GaussianLogLik",
    "LogProb",
    # Distribution shim
    "Dirac",
    # Dispatch op
    "pushforward",
]
