"""Documentation-only M4A reference-semantics gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families.nmd_qc_float_m4a import (  # noqa: E402
    audit_m4a,
)


def run_gate(
    output_path: Path,
    *,
    fetch_remote: bool = True,
) -> dict:
    result = audit_m4a(fetch_remote=fetch_remote)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--skip-remote", action="store_true")
    args = parser.parse_args()
    result = run_gate(
        args.output_json,
        fetch_remote=not args.skip_remote,
    )
    print(
        json.dumps(
            {
                "gate": result["gate_id"],
                "global_status": result["global_status"],
                "selected_candidate": result["selected_candidate"],
                "signal_samples_opened": result["signal_samples_opened"],
                "m4b_authorized": result["m4b_authorized"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
