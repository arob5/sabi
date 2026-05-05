"""Sphinx configuration for the sabi documentation site.

Build locally with:

    uv run sphinx-build -W -b html docs docs/_build/html
"""

from __future__ import annotations

import datetime as _dt

project = "sabi"
author = "Andrew Roberts"
copyright = f"{_dt.datetime.now().year}, {author}"
release = "0.0.0"

extensions = [
    "myst_nb",  # supersedes myst_parser; renders notebooks without pandoc
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "autodoc2",
]

source_suffix = {
    ".md": "myst-nb",
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}
master_doc = "index"
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
]
myst_heading_anchors = 3

autodoc2_packages = [
    {
        "path": "../src/sabi",
        "auto_mode": True,
    },
]
autodoc2_render_plugin = "myst"
autodoc2_output_dir = "api"
autodoc2_hidden_objects = ["private", "dunder", "inherited"]
autodoc2_docstring_parser_regexes = [
    (r".*", "rst"),
]

# Notebooks ship with their cell outputs already populated; CI renders the
# cached outputs without re-executing. Reason: sabi tracks in-flight ProbPipe
# APIs that are not always present on TARPS-group/prob-pipe@main, so a fresh
# CI clone of prob-pipe cannot import sabi reliably yet. Switch back to
# "force" once the ProbPipe overhaul lands and sabi pins to a stable
# release. Tracked in https://github.com/arob5/sabi/issues/49.
#
# Authors who edit notebook code or who change sabi behavior visible in a
# notebook MUST re-execute the notebook locally before committing — see
# docs/contributing.md ("Notebook outputs are cached").
nb_execution_mode = "off"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}

html_theme = "furo"
html_title = "sabi"
html_static_path = ["_static"]
