# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import os
import sys
from pathlib import Path

# Monorepo: packages live under core/src and mndm/src (see README PYTHONPATH).
_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "core" / "src"))
sys.path.insert(0, str(_repo_root / "mndm" / "src"))

project = "NeuralManifoldDynamics"
copyright = "NeuralManifoldDynamics contributors"
author = "NeuralManifoldDynamics contributors"

version = ""
release = ""

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Napoleon: Google-style docstrings ----------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# -- Autodoc -----------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Heavy optional deps: keep import light for doc builds if something is missing.
autodoc_mock_imports: list[str] = []
