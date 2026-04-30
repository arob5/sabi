"""Benchmark problems and form abstractions.

Submodules expose their public API directly — there is intentionally
**no** eager package-level re-export here. Two reasons:

1. **Cycle-breaking.** ``sabi.target_distribution`` imports
   ``sabi.problems.forms.LogDensityForm``. Loading the parent package
   runs this ``__init__``; if it eagerly imported ``Problem`` /
   ``banana`` / ``gaussian2d`` (which all transitively depend on
   ``sabi.target_distribution``), Python would see ``target_distribution``
   as partially-loaded and raise ``ImportError`` whenever the import
   chain entered via ``sabi.tempering`` or any other path that hit the
   forms module before ``sabi.problems.base`` had finished loading.
2. **Fast imports.** Tools that only need ``LogDensityForm`` (e.g.,
   the algorithm loop) shouldn't pay to load every benchmark builder.

Use submodule paths for everything:

.. code-block:: python

    from sabi.problems.forms import LogDensityForm, Identity, LogLikPlusPrior
    from sabi.problems.base import Problem
    from sabi.problems.gaussian2d import gaussian2d
    from sabi.problems.banana import banana
    from sabi.problems.neals_funnel import neals_funnel
"""
