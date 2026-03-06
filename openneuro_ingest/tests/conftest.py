"""pytest configuration for openneuro tests.

This conftest ensures that real optional dependencies (h5py, scipy, mne)
are used when available, overriding any stubs from the root conftest.py.
"""

import importlib
import sys
from pathlib import Path

import pytest


def _is_stub(module_name: str) -> bool:
    """Check if a module is a stub (SimpleNamespace or lacks __file__)."""
    if module_name not in sys.modules:
        return False
    mod = sys.modules[module_name]
    return not hasattr(mod, "__file__") or mod.__file__ is None


def _reload_real_module(module_name: str) -> bool:
    """Remove stub and reimport real module. Returns True if real module loaded."""
    if not _is_stub(module_name):
        return module_name in sys.modules
    
    # Remove stub and submodules
    del sys.modules[module_name]
    to_remove = [k for k in sys.modules if k.startswith(f"{module_name}.")]
    for k in to_remove:
        del sys.modules[k]
    
    # Try to import real module
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


# Ensure core/src is available for shared imports
repo_root = Path(__file__).resolve().parents[2]
core_src = repo_root / "core" / "src"
if str(core_src) not in sys.path:
    sys.path.append(str(core_src))

# Attempt to use real packages if installed
for pkg in ("h5py", "scipy", "mne"):
    _reload_real_module(pkg)


@pytest.fixture
def require_real_h5py():
    """Skip test if h5py is stubbed or unavailable."""
    if _is_stub("h5py"):
        pytest.skip("h5py is stubbed, not real")
    try:
        import h5py
        _ = h5py.File
    except (ImportError, AttributeError):
        pytest.skip("Real h5py not available")

