"""MNDM: MNPS projection, Jacobian/tensor estimation, and robustness utilities.

The CLI entry point is :mod:`mndm.cli`; core algorithms live in
:mod:`mndm.projection`, :mod:`mndm.jacobian`, :mod:`mndm.robustness`, and
:mod:`mndm.schema`.
"""

from .__about__ import __version__  # noqa: F401

__all__ = ["__version__", "projection", "robustness", "transients", "jacobian", "schema"]


