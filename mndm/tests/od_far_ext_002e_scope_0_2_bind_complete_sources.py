"""Dry-run / bind complete E1/E2 XML for FAR-EXT-002E scope 0.2.

Never freezes the scope and never writes the 0.2 result archive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MNDM_SRC = _REPO_ROOT / "mndm" / "src"
if str(_MNDM_SRC) not in sys.path:
    sys.path.insert(0, str(_MNDM_SRC))

from mndm.dynamical_families.far_ext_002e_ds003670_post_ramp_interpretation import (  # noqa: E402
    bind_complete_e1_e2_xml,
)

FAR_ROOT = (
    _REPO_ROOT
    / "project"
    / "orthagonal_axis"
    / "orthagonal_dynamics"
    / "finite-amplitude_resilience"
)
SNAPSHOT_ROOT = FAR_ROOT / "source_snapshots" / "far_ext_002e_scope_0.2"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assess or bind complete E1/E2 XML without a 0.2 live run"
    )
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument(
        "--e1-xml",
        type=Path,
        default=SNAPSHOT_ROOT / "E1_f30_followup.xml",
    )
    parser.add_argument(
        "--e2-xml",
        type=Path,
        default=SNAPSHOT_ROOT / "E2_f30_response.xml",
    )
    parser.add_argument(
        "--source-object-class",
        choices=("PUBLISHER_VOR", "COMPLETE_OA_SOURCE_OBJECT"),
        default="COMPLETE_OA_SOURCE_OBJECT",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write snapshots/hashes only if both candidates PASS; never freeze",
    )
    args = parser.parse_args(argv)
    payload = bind_complete_e1_e2_xml(
        repo_root=args.repo_root,
        e1_xml=args.e1_xml,
        e2_xml=args.e2_xml,
        source_object_class=args.source_object_class,
        apply=bool(args.apply),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["current_candidate_status"] == "E1_E2_COMPLETE_PENDING_FREEZE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
