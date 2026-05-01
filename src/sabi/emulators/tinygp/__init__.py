"""tinygp-backed `Emulator` implementations.

Lightweight, dependency-minimal GP emulators built on `tinygp`. Suitable
when you want a fast, standalone GP without pulling in the heavier
`numpyro` / `paramax` / `optimistix` stack that gpjax requires.

Available:

- `TinyGPEmulator`: Matérn-5/2 isotropic kernel, data-adaptive lengthscale
  (median nearest-neighbor distance × factor, floored), standardized
  inputs and outputs, fixed small noise + Cholesky jitter. No
  hyperparameter optimization — kernel params are set once at fit time
  and held fixed.

For higher-dimensional problems where the dimension-scaled prior of
Hvarfner et al. (2024) is preferred, see `sabi.emulators.gpjax`. That
sibling package requires the optional ``gpjax`` extra.
"""

from sabi.emulators.tinygp.gp import TinyGPEmulator

__all__ = ["TinyGPEmulator"]
