from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_sleep_eap_contract_import_does_not_load_optional_eeg_stack() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    core_source_root = Path(__file__).resolve().parents[2] / "core" / "src"
    check = (
        "import sys; "
        f"sys.path.insert(0, {str(source_root)!r}); "
        f"sys.path.insert(0, {str(core_source_root)!r}); "
        "import mndm.features.sleep_eap as sleep_eap; "
        "assert sleep_eap.PHASE_CONTINUOUS_V1_CONTRACT == 'phase_continuous_v1'; "
        "from mndm import orchestrate; "
        "assert 'mndm.features.eeg' not in sys.modules; "
        "assert 'mndm.features.sleep_eap.spindle_strength' not in sys.modules; "
        "assert 'antropy' not in sys.modules"
    )

    subprocess.run(
        [sys.executable, "-c", check],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
