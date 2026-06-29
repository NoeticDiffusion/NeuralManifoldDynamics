Major concern 1: EEG artifact handling is still too weak for a primary EEG-manifold paper

This is now the central data-quality issue.

S3 says the pipeline used band-pass filtering and epoch-level z-score rejection, with no RANSAC/correlation-based bad-channel detection, no interpolation, and no ICA. It also says all subjects had zero bad channels and no artifact methods applied.

That is transparent, but a reviewer will read it as:

The ECG was carefully repaired, but the EEG layer — the source of the primary claim — did not receive comparable artifact handling.

For a working paper, this is acceptable if framed as a limitation. For a serious journal submission, I would strongly recommend at least one of these:

Rerun with standard EEG cleaning
RANSAC or correlation-based bad-channel detection, interpolation, ICA/SSP/EOG regression if EOG channels exist, then recompute MNPS.
Sensitivity on high-artifact subjects/windows
Use amplitude, variance, high-frequency power, channel kurtosis, or muscle-band proxies to exclude the worst 10–20% of epochs and rerun primary m/d.
Show condition-balanced artifact metrics
Report whether artifact proxies differ by condition. This is crucial because the main contrast is presentation mode, and visual sequential conditions could differ in blink/saccade/muscle artifacts.

The biggest danger is not random noise. The danger is condition-structured artifact.

Fast/FastDelay might differ in eye movements, visual transients, blinks, or micro-saccades compared with Simultaneous/Slow. If those influence MNPS m/d, the result may partly reflect sensory/ocular dynamics rather than cognitive item-updating.

Required before journal submission: add an EEG artifact-balance table by condition.