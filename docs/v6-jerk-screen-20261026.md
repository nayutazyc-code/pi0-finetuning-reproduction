# Original-pi0 v6 jerk-reference screen — 2026-10-26

## Scope and evidence status

Stage 2 investigates an optional local arm-reference backend after the completed [cadence experiment](v6-cadence-optimization-20261026.md) retained cadence15 as its next control. It does not rewrite that Stage 1 source-freeze audit or historical outcomes. The four-run GUI screen is complete. Results are mixed: the optional backend reduced the worst observed physical spike but increased typical physical jerk. No candidate is promoted and the screen did not expand to validation.

The [plan](../outputs/v6_jerk_20261026/plan.json) compares unchanged [v6 cadence15 bridge20](../configs/banana_v6_bridge20_pi0.toml) with [jerk50](../configs/banana_v6_jerk50_pi0.toml) in order A-B-B-A. It predeclares no automatic validation: further model evaluation requires support from the GUI screen, and a candidate with no completions in two attempts or worse reference tracking is not promoted. Model noise and network timing are unpaired. The [harness](../outputs/v6_jerk_20261026/run_comparison.py), [ledger](../outputs/v6_jerk_20261026/comparison.json), and [log](../outputs/v6_jerk_20261026/experiment.log) preserve all attempts and failures. No training, checkpoint, normalization, data, or cloud-serving change is part of this screen.

## Mechanism and state boundary

The optional [helper](../src/pi0_repro/jerk_reference.py) uses project-local Ruckig 0.19.4. With `max_joint_jerk` omitted, the existing limiter remains selected. Enabling it requires finite positive velocity/acceleration limits. Jerk50 retains cadence15, the 20-step quintic bridge, 3 Hz filter, 40 rad/s² acceleration limit, 1.1 rad/s velocity limit, and existing gripper rate limits.

For each 20 ms step, the 12-arm reference advances through a stateless local `calculate` operation toward post-filter target position and clipped filter velocity, zero terminal acceleration, and independent-axis `Synchronization.No`. It uses no intermediate/cloud waypoints. The input filter runs independently rather than receiving the jerk-limited output through the legacy anti-windup path. Grippers do not enter this solver. Only numeric solver state is retained, allowing RTC to deep-copy the same position, velocity, acceleration, input-filter state, and limits before replaying projection-aware command forecasts.

Hard actuator projection can override the derivative constraints. Projected axes reset reference velocity and acceleration to zero and increment `jerk_reference_projection_resets`. That reset breaks any smooth-derivative guarantee across the hard stop. Measured physical joints, especially during contact, are not bounded by the commanded reference's nominal jerk setting. Recording provenance includes the helper source hash and optional package version.

## Retained local diagnosis

The original [frozen-reference diagnostic](../outputs/v6_jerk_20261026/frozen_reference_diagnostic.json), [independent-filter diagnostic](../outputs/v6_jerk_20261026/independent_filter_diagnostic.json), and [velocity-reference diagnostic](../outputs/v6_jerk_20261026/velocity_reference_diagnostic.json) retain earlier variants, including poor tracking with zero terminal velocity and filter/output feedback. They are not successful model episodes or evidence of improved physical motion. The [jerk300 configuration](../configs/banana_v6_jerk300_pi0.toml) remains an offline preparation; jerk50 is the active screen candidate selected to exercise the peak constraint.

Nine unit checks passed before model screening: five jerk-reference checks and four existing Temporal Ensembling checks. The jerk checks include step/reversal derivative bounds, exact float32 copied-state forecasting, hard-projection resets, and runtime integration boundaries. These establish implementation properties under their inputs, not model task quality or subjective improvement.

## Actual-model screen and assistant inspection

The completed [summary](../outputs/v6_jerk_20261026/summary.json) records four GUI attempts in the declared A-B-B-A order, with two successes per profile. Controls are fresh for this screen; prior experiment counts are not pooled.

| Profile | Observed completions | Simulated durations, s | Median of per-run physical joint-jerk p95, rad/s³ | Worst physical joint jerk, rad/s³ |
| --- | --- | --- | --- | --- |
| bridge20 | 2/2 | 24.60, 26.18 | 16.36 | 2535.34 |
| jerk50 | 2/2 | 23.70, 22.42 | 41.70 | 696.44 |

All four attempts had zero running inference underruns and zero maximum command-prefix forecast error. Typical physical jerk was worse with jerk50 despite its lower worst spike and shorter observed completion durations. These are different unpaired trajectories and do not establish a general tradeoff or guarantee that the configured reference limit bounds contact response. No headless validation expansion followed this mixed screen.

The user reported not seeing the GUI and explicitly asked the assistant to inspect videos/logs. The assistant inspected [peak-motion frames](../outputs/v6_jerk_20261026/peak_motion_frames.jpg) alongside the logs: the baseline peak was near step 1192 (23.84 s), during above-box release, whereas the candidate peak was near step 442 (8.84 s), during grasping. They occur in different phases. Small differences in frames separated by 40 ms cannot establish a blanket human-perceived smoothness improvement or a causal contact explanation.

## Selection and later-stage limits

The completed screen retains both successful task outcomes and worse typical physical jerk. Keep the unchanged cadence15 bridge20 control; do not promote jerk50 from the lower isolated maximum or unit checks. Lower reference jerk or a low-motion failed trajectory is not a control-quality result. Retain all failures and hard-reset evidence; do not promote from unit checks alone.

Any later recovery-data stage must use preassigned training layouts, preserve provenance and failed records, and remain distinct from validation/test layouts. No recovery collection or new training result is claimed by this screen.
