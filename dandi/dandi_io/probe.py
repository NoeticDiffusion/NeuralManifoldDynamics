from __future__ import annotations

from pathlib import Path
from typing import Sequence

from dandi_io.contracts import AssetRecord, ProbeSummary


def probe_assets(records: Sequence[AssetRecord], *, raw_root: Path) -> list[ProbeSummary]:
    return [probe_local_asset(record, raw_root=raw_root) for record in records]


def probe_local_asset(record: AssetRecord, *, raw_root: Path) -> ProbeSummary:
    local_path = record.local_path(raw_root)
    if not local_path.exists():
        return ProbeSummary(
            path=record.path,
            local_path=local_path,
            exists=False,
            subject_id=record.subject_id,
            session_id=record.session_id,
            error="file_not_found",
        )

    top_level_groups: tuple[str, ...] = ()
    metadata: dict[str, object] = {}
    file_size = local_path.stat().st_size

    try:
        import h5py

        with h5py.File(local_path, "r") as handle:
            top_level_groups = tuple(sorted(str(key) for key in handle.keys()))
    except Exception as exc:  # pragma: no cover - probe should degrade gracefully
        metadata["hdf5_probe_error"] = str(exc)

    acquisitions: tuple[str, ...] = ()
    processing_modules: tuple[str, ...] = ()
    intervals: tuple[str, ...] = ()
    imaging_planes: tuple[str, ...] = ()
    devices: tuple[str, ...] = ()
    lab_meta_data: tuple[str, ...] = ()
    subject_id = record.subject_id
    session_id = record.session_id
    error: str | None = None

    try:
        from pynwb import NWBHDF5IO

        with NWBHDF5IO(str(local_path), "r", load_namespaces=True) as io:
            nwbfile = io.read()
            subject = getattr(nwbfile, "subject", None)
            subject_id = subject_id or getattr(subject, "subject_id", None)
            session_id = session_id or getattr(nwbfile, "session_id", None)
            acquisitions = tuple(sorted(getattr(nwbfile, "acquisition", {}).keys()))
            processing_modules = tuple(sorted(getattr(nwbfile, "processing", {}).keys()))
            intervals = tuple(sorted(getattr(nwbfile, "intervals", {}).keys()))
            imaging_planes = tuple(sorted(getattr(nwbfile, "imaging_planes", {}).keys()))
            devices = tuple(sorted(getattr(nwbfile, "devices", {}).keys()))
            lab_meta_data = tuple(sorted(getattr(nwbfile, "lab_meta_data", {}).keys()))
    except ImportError:
        error = "pynwb_not_installed"
    except Exception as exc:  # pragma: no cover - depends on local files
        error = str(exc)

    return ProbeSummary(
        path=record.path,
        local_path=local_path,
        exists=True,
        file_size=file_size,
        subject_id=subject_id,
        session_id=session_id,
        top_level_groups=top_level_groups,
        acquisitions=acquisitions,
        processing_modules=processing_modules,
        intervals=intervals,
        imaging_planes=imaging_planes,
        devices=devices,
        lab_meta_data=lab_meta_data,
        modality_hints=_infer_modality_hints(
            top_level_groups=top_level_groups,
            acquisitions=acquisitions,
            processing_modules=processing_modules,
            imaging_planes=imaging_planes,
        ),
        error=error,
        metadata=metadata,
    )


def _infer_modality_hints(
    *,
    top_level_groups: Sequence[str],
    acquisitions: Sequence[str],
    processing_modules: Sequence[str],
    imaging_planes: Sequence[str],
) -> tuple[str, ...]:
    hints: list[str] = []
    search_space = " ".join(
        (
            " ".join(top_level_groups),
            " ".join(acquisitions),
            " ".join(processing_modules),
            " ".join(imaging_planes),
        )
    ).lower()
    if "ophys" in search_space or imaging_planes:
        hints.append("ophys")
    if "ecephys" in search_space:
        hints.append("ecephys")
    if "behavior" in search_space:
        hints.append("behavior")
    if "image" in search_space:
        hints.append("imaging")
    if "eeg" in search_space:
        hints.append("eeg")
    if "emg" in search_space:
        hints.append("emg")
    return tuple(dict.fromkeys(hints))
