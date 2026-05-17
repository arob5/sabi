"""Benchmark problems.

Submodules expose their public API directly — there is intentionally
**no** eager package-level re-export here, both for fast imports
(tools that only need ``Problem`` shouldn't pay to load every
benchmark builder) and for cycle-breaking under the post-#65 layout.

Use submodule paths for everything:

.. code-block:: python

    from sabi.problems.base import Problem, BenchmarkProblem
    from sabi.problems.gaussian import gaussian
    from sabi.problems.banana import banana
    from sabi.problems.neals_funnel import neals_funnel
    from sabi.problems.benchmarks import (
        banana_2d, banana_10d,
        gaussian_2d, gaussian_10d,
        neals_funnel_3d,
    )
"""
