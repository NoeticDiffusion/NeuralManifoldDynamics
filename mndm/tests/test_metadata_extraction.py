"""Tests for enhanced metadata extraction (group/condition/task)."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from mndm.pipeline.extractors import build_dataset_label, extract_mapped_metadata


class TestExtractMappedMetadata:
    """Test extract_mapped_metadata with various config scenarios."""

    def test_session_map_condition(self):
        """Session map directly assigns condition from session ID."""
        config = {
            "metadata_extraction": {
                "datasets": {
                    "ds003171": {
                        "condition": {
                            "session_map": {
                                "ses-01": "awake",
                                "ses-02": "light",
                                "ses-03": "deep",
                                "ses-04": "recovery",
                            }
                        }
                    }
                }
            }
        }
        result = extract_mapped_metadata({}, config, "ds003171", "ses-02")
        assert result["condition"] == "light"

    def test_session_map_unknown_session_fallback(self):
        """Unknown session falls through to candidates or default."""
        config = {
            "metadata_extraction": {
                "datasets": {
                    "ds003171": {
                        "condition": {
                            "session_map": {"ses-01": "awake"},
                            "default": "unknown",
                        }
                    }
                }
            }
        }
        result = extract_mapped_metadata({}, config, "ds003171", "ses-99")
        assert result["condition"] == "unknown"

    def test_group_default(self):
        """Default group is used when no candidate matches."""
        config = {
            "metadata_extraction": {
                "datasets": {
                    "ds003171": {
                        "group": {
                            "candidates": ["Group"],
                            "default": "Healthy",
                        }
                    }
                }
            }
        }
        result = extract_mapped_metadata({}, config, "ds003171", None)
        assert result["group"] == "Healthy"

    def test_group_from_meta_overrides_default(self):
        """Candidate from participants.tsv takes priority over default."""
        config = {
            "metadata_extraction": {
                "datasets": {
                    "ds003171": {
                        "group": {
                            "candidates": ["Group"],
                            "default": "Healthy",
                        }
                    }
                }
            }
        }
        meta = {"Group": "Patient"}
        result = extract_mapped_metadata(meta, config, "ds003171", None)
        assert result["group"] == "Patient"

    def test_task_from_filename(self):
        """Parse task from BIDS filename when enabled."""
        config = {
            "metadata_extraction": {
                "datasets": {
                    "ds003171": {
                        "task": {"from_filename": True}
                    }
                }
            }
        }
        filename = "sub-02CB_ses-01_task-rest_bold.nii.gz"
        result = extract_mapped_metadata({}, config, "ds003171", "ses-01", filename=filename)
        assert result["task"] == "rest"

    def test_task_from_filename_audio(self):
        """Parse audio task from BIDS filename."""
        config = {
            "metadata_extraction": {
                "datasets": {
                    "ds003171": {
                        "task": {"from_filename": True}
                    }
                }
            }
        }
        filename = "sub-02CB_ses-03_task-audio_bold.nii.gz"
        result = extract_mapped_metadata({}, config, "ds003171", "ses-03", filename=filename)
        assert result["task"] == "audio"

    def test_task_fallback_to_candidates(self):
        """Fall back to candidates if from_filename is off or filename missing."""
        config = {
            "metadata_extraction": {
                "datasets": {
                    "ds003171": {
                        "task": {
                            "from_filename": False,
                            "candidates": ["task"],
                        }
                    }
                }
            }
        }
        meta = {"task": "motor"}
        result = extract_mapped_metadata(meta, config, "ds003171", None)
        assert result["task"] == "motor"

    def test_task_default_when_no_match(self):
        """Default task is used when nothing else matches."""
        config = {
            "metadata_extraction": {
                "datasets": {
                    "ds003171": {
                        "task": {"default": "rest"}
                    }
                }
            }
        }
        result = extract_mapped_metadata({}, config, "ds003171", None)
        assert result["task"] == "rest"

    def test_full_ds003171_scenario(self):
        """Full ds003171 scenario: awake + rest."""
        config = {
            "metadata_extraction": {
                "datasets": {
                    "ds003171": {
                        "group": {"default": "Healthy"},
                        "condition": {
                            "session_map": {
                                "ses-01": "awake",
                                "ses-02": "light",
                                "ses-03": "deep",
                                "ses-04": "recovery",
                            }
                        },
                        "task": {"from_filename": True},
                    }
                }
            }
        }
        filename = "sub-02CB_ses-01_task-rest_bold.nii.gz"
        result = extract_mapped_metadata({}, config, "ds003171", "ses-01", filename=filename)

        assert result["group"] == "Healthy"
        assert result["condition"] == "awake"
        assert result["task"] == "rest"

    def test_session_from_filename_when_session_arg_none(self):
        """Session parsed from filename when session argument is None."""
        config = {
            "metadata_extraction": {
                "datasets": {
                    "ds003171": {
                        "condition": {
                            "session_map": {
                                "ses-awake": "awake",
                                "ses-deep": "deep",
                            }
                        }
                    }
                }
            }
        }
        filename = "sub-02CB_ses-awake_task-rest_bold.nii.gz"
        # session argument is None, but filename contains ses-awake
        result = extract_mapped_metadata({}, config, "ds003171", None, filename=filename)
        assert result["condition"] == "awake"

    def test_session_arg_takes_priority_over_filename(self):
        """Explicit session argument takes priority over filename parsing."""
        config = {
            "metadata_extraction": {
                "datasets": {
                    "ds003171": {
                        "condition": {
                            "session_map": {
                                "ses-01": "awake",
                                "ses-awake": "awake_alt",
                            }
                        }
                    }
                }
            }
        }
        filename = "sub-02CB_ses-awake_task-rest_bold.nii.gz"
        # session argument is ses-01, even though filename says ses-awake
        result = extract_mapped_metadata({}, config, "ds003171", "ses-01", filename=filename)
        assert result["condition"] == "awake"  # From ses-01, not ses-awake

    def test_compound_task_audioawake(self):
        """Compound task 'audioawake' splits into task='audio', condition='awake'."""
        config = {
            "metadata_extraction": {
                "datasets": {
                    "ds003171": {
                        "task": {
                            "from_filename": True,
                            "compound_conditions": ["recovery", "awake", "light", "deep"],
                        }
                    }
                }
            }
        }
        filename = "sub-30AQ_task-audioawake_run-01_bold.nii.gz"
        result = extract_mapped_metadata({}, config, "ds003171", None, filename=filename)
        assert result["task"] == "audio"
        assert result["condition"] == "awake"

    def test_compound_task_restdeep(self):
        """Compound task 'restdeep' splits into task='rest', condition='deep'."""
        config = {
            "metadata_extraction": {
                "datasets": {
                    "ds003171": {
                        "task": {
                            "from_filename": True,
                            "compound_conditions": ["recovery", "awake", "light", "deep"],
                        }
                    }
                }
            }
        }
        filename = "sub-30AQ_task-restdeep_run-01_bold.nii.gz"
        result = extract_mapped_metadata({}, config, "ds003171", None, filename=filename)
        assert result["task"] == "rest"
        assert result["condition"] == "deep"

    def test_compound_task_audiorecovery(self):
        """Compound task 'audiorecovery' splits into task='audio', condition='recovery'."""
        config = {
            "metadata_extraction": {
                "datasets": {
                    "ds003171": {
                        "task": {
                            "from_filename": True,
                            "compound_conditions": ["recovery", "awake", "light", "deep"],
                        }
                    }
                }
            }
        }
        filename = "sub-30AQ_task-audiorecovery_run-01_bold.nii.gz"
        result = extract_mapped_metadata({}, config, "ds003171", None, filename=filename)
        assert result["task"] == "audio"
        assert result["condition"] == "recovery"

    def test_compound_task_restlight(self):
        """Compound task 'restlight' splits into task='rest', condition='light'."""
        config = {
            "metadata_extraction": {
                "datasets": {
                    "ds003171": {
                        "task": {
                            "from_filename": True,
                            "compound_conditions": ["recovery", "awake", "light", "deep"],
                        }
                    }
                }
            }
        }
        filename = "sub-30AQ_task-restlight_run-01_bold.nii.gz"
        result = extract_mapped_metadata({}, config, "ds003171", None, filename=filename)
        assert result["task"] == "rest"
        assert result["condition"] == "light"

    def test_compound_task_with_normalization(self):
        """Condition from compound task is normalized."""
        config = {
            "metadata_extraction": {
                "datasets": {
                    "ds003171": {
                        "condition": {
                            "normalize": {"awake": "Awake", "deep": "Deep"}
                        },
                        "task": {
                            "from_filename": True,
                            "compound_conditions": ["awake", "deep"],
                        }
                    }
                }
            }
        }
        filename = "sub-30AQ_task-audioawake_run-01_bold.nii.gz"
        result = extract_mapped_metadata({}, config, "ds003171", None, filename=filename)
        assert result["task"] == "audio"
        assert result["condition"] == "Awake"  # Normalized

    def test_simple_task_no_compound(self):
        """Simple task without compound_conditions stays as-is."""
        config = {
            "metadata_extraction": {
                "datasets": {
                    "ds005114": {
                        "task": {"from_filename": True}
                    }
                }
            }
        }
        filename = "sub-001_task-rest_bold.nii.gz"
        result = extract_mapped_metadata({}, config, "ds005114", None, filename=filename)
        assert result["task"] == "rest"
        assert result["condition"] is None

    def test_normalization_applied(self):
        """Normalization mapping is applied to extracted values."""
        config = {
            "metadata_extraction": {
                "default": {
                    "group": {
                        "candidates": ["Group"],
                        "normalize": {"hc": "Healthy", "patient": "Patient"},
                    }
                }
            }
        }
        meta = {"Group": "HC"}
        result = extract_mapped_metadata(meta, config, "ds000001", None)
        assert result["group"] == "Healthy"


class TestBuildDatasetLabel:
    """Test build_dataset_label for various input combinations."""

    def test_basic_no_condition_no_task(self):
        """Basic label with just dataset and subject."""
        label = build_dataset_label("ds005114", "sub-001", None, None, None)
        assert label == "ds005114:sub-001"

    def test_with_session_no_condition(self):
        """Session ID used when no condition provided."""
        label = build_dataset_label("ds005114", "sub-001", "ses-01", None, None)
        assert label == "ds005114:sub-001:ses-01"

    def test_with_condition_no_task(self):
        """Condition replaces session in label."""
        label = build_dataset_label("ds003171", "sub-02CB", "ses-01", "awake", None)
        assert label == "ds003171:sub-02CB:awake"

    def test_with_condition_and_task(self):
        """Full label with condition and task."""
        label = build_dataset_label("ds003171", "sub-02CB", "ses-01", "awake", "rest")
        assert label == "ds003171:sub-02CB:awake_rest"

    def test_deep_audio(self):
        """Label for deep sedation with audio task."""
        label = build_dataset_label("ds003171", "sub-02CB", "ses-03", "deep", "audio")
        assert label == "ds003171:sub-02CB:deep_audio"

    def test_task_without_condition(self):
        """Task appended even without condition (uses session)."""
        label = build_dataset_label("ds005114", "sub-001", "ses-01", None, "movie")
        assert label == "ds005114:sub-001:ses-01_movie"

