# Bug: `mndm --subject N` zero-pads to `sub-00N`, breaking non-padded BIDS datasets

**Status**: Open  
**Priority**: Low (workaround available)  
**Discovered**: 2026-05-08, multi-subject event-locked batch run  
**Component**: `mndm/src/mndm/cli.py` → subject filter logic

---

## Summary

When running `mndm features` or `mndm summarize` with `--subject N`, the CLI zero-pads the subject label to `sub-00N`. For example, `--subject 2` is matched as `sub-002`.

This is incompatible with BIDS datasets that use non-zero-padded subject IDs such as `sub-1`, `sub-2`, etc. (ds005555 is an example: subjects are `sub-1` through `sub-10`).

## Symptom

```
mndm summarize --dataset ds005555 --subject 2
WARNING: No epochs for subject sub-002 in ds005555
# return code 0, no H5 produced
```

Intermediate features for `sub-2` exist on disk (`intermediate/sub-2_task-Sleep_acq-psg_eeg.json`) but are not matched because the filter uses `sub-002`.

## Workaround

Omit `--subject` entirely. Running without a subject filter processes all subjects correctly:

```bash
mndm summarize --config config_ingest_ds005555_sleep_spindles.yaml --dataset ds005555
# Processes sub-1, sub-2, sub-3, sub-4, sub-5 correctly
```

## Suggested fix

In the CLI subject-filter resolution, try `sub-N` (unpadded) first, then fall back to `sub-00N` (padded). Or accept an explicit full label: `--subject sub-2`.

```python
# Current (broken for unpadded IDs):
subject_label = f"sub-{int(args.subject):03d}"

# Proposed:
def resolve_subject(s: str) -> str:
    if s.startswith("sub-"):
        return s  # explicit full label
    n = int(s)
    return f"sub-{n}"  # try unpadded first
```

If the unpadded form is not found in the index, the code can fall back to the padded form.

## Affected datasets

Any BIDS dataset with single-digit unpadded IDs: `sub-1`, `sub-2`, etc.  
Known affected: `ds005555`.

## Impact

- Science path: not blocking (workaround: drop `--subject` flag)
- Usability: footgun — CLI returns exit code 0 with silent "No epochs" warning
- Risk of confusion when resuming single-subject runs or debugging one subject at a time
