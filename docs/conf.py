"""Sphinx configuration for KempnerPulse documentation."""

from __future__ import annotations

import importlib.metadata

project = "KempnerPulse"
author = "Kempner Institute for the Study of Natural and Artificial Intelligence at Harvard University"

try:
    release = importlib.metadata.version("kempnerpulse")
except importlib.metadata.PackageNotFoundError:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

copyright = f"2026, Kempner Institute for the Study of Natural and Artificial Intelligence at Harvard University · v{release}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Custom autosummary template: skip a package member that is also a submodule
# (e.g. compute re-exports the `classify` function alongside a `classify`
# submodule) so it is documented once, via its module, not twice.
templates_path = ["_templates"]

autosummary_generate = True
# Runtime dependencies are the standard library plus rich; both are importable in
# the docs build environment, so no autodoc mocks are needed.
autodoc_mock_imports = []
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_class_signature = "separated"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "linkify",
    "substitution",
]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

html_theme = "furo"
html_title = f"KempnerPulse v{version}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    # No logo image yet, so keep the project name visible in the sidebar.
    "sidebar_hide_name": False,
    "source_repository": "https://github.com/KempnerInstitute/kempnerpulse",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/KempnerInstitute/kempnerpulse",
            "class": "",
        },
    ],
}

nitpicky = False

# Autosummary can generate stubs for both a package re-export and its defining
# submodule, so a short-name docstring reference resolves to two targets; silence
# those ambiguities.
suppress_warnings = ["ref.python"]
