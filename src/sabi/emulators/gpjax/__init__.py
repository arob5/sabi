"""gpjax-backed `Emulator` implementations.

Wraps the `gpjax` library for use as a sabi `Emulator`. Heavier
than the `tinygp` siblings — pulls `numpyro`, `paramax`, `optimistix`,
`lineax`, `tensorstore` (mac), etc. — but provides full hyperparameter
optimization, ARD kernels, and the dimension-scaled lengthscale priors
that perform well in higher dimensions.

**gpjax is an optional dependency.** Install with:

.. code-block:: bash

    pip install 'sabi[gpjax]'   # via pyproject extras
    # or
    uv sync --extra gpjax

Importing this package without ``gpjax`` installed will raise a clear
`ImportError` with the install command. Imports inside this package are
deferred (lazy) where possible so that touching the parent
`sabi.emulators` namespace doesn't fail when the extra isn't installed.

Available (when the extra is installed):

- `DSPGPEmulator`: Hvarfner et al. (2024) dimension-scaled-prior GP.
  RBF or Matérn-5/2 ARD kernel; lengthscale prior
  ``LogNormal(√2 + 0.5·log(d), √3)``; outputscale fixed at 1.0; noise
  prior ``LogNormal(-4, 1)`` on **stddev**; floor 2.5e-2 on lengthscale,
  1e-4 on noise. Assumes inputs normalized to ``[0, 1]^d`` and outputs
  standardized to zero-mean unit-variance.
"""

# Public exports are populated lazily inside each submodule's import
# guard so that a missing `gpjax` install fails with a clear message
# only when a concrete emulator is referenced — not on package touch.
__all__: list[str] = []
