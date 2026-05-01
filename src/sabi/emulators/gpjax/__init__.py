"""gpjax-backed `Emulator` implementations.

Wraps the `gpjax` library for use as a sabi `Emulator`. Heavier than the
`tinygp` siblings — pulls `numpyro`, `paramax`, `optimistix`, `lineax`,
`tensorstore` (mac), etc. — but provides full hyperparameter
optimization, ARD kernels, and the dimension-scaled lengthscale priors
that perform well in higher dimensions.

**gpjax is an optional dependency.** Install with one of:

.. code-block:: bash

    pip install 'sabi[gpjax]'
    uv sync --extra gpjax
    uv pip install 'gpjax>=0.14,<0.15'

Importing concrete emulators from this package without `gpjax` installed
raises a helpful `ImportError` with the install command. Imports inside
this package are deferred (lazy) via ``__getattr__`` so that touching
the parent ``sabi.emulators`` namespace doesn't fail when the extra
isn't installed.

Available (when the extra is installed):

- `DSPGPEmulator`: Hvarfner et al. (2024) dimension-scaled-prior GP.
  RBF or Matérn-5/2 ARD kernel; lengthscale prior
  ``LogNormal(√2 + 0.5·log(d), √3)``; outputscale fixed at 1.0; noise
  prior ``LogNormal(-4, 1)`` on **stddev**; floor 2.5e-2 on lengthscale,
  1e-4 on noise. Assumes inputs normalized to ``[0, 1]^d`` and outputs
  standardized to zero-mean unit-variance.
"""

from __future__ import annotations

__all__ = ["DSPGPEmulator"]


def __getattr__(name: str):
    """Lazy-load gpjax-backed emulators on first access.

    Defers the gpjax import (and its dependency stack) until a concrete
    class is referenced. Touching ``sabi.emulators.gpjax`` with no
    attribute access — e.g., during ``import sabi.emulators`` — does
    NOT trigger gpjax loading.
    """
    if name == "DSPGPEmulator":
        from sabi.emulators.gpjax.dsp_gp import DSPGPEmulator as _DSPGPEmulator

        return _DSPGPEmulator
    raise AttributeError(
        f"module 'sabi.emulators.gpjax' has no attribute '{name}'. "
        f"Available: {__all__}"
    )
