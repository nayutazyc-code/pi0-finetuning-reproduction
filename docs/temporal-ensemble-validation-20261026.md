# Original-pi0 v6 Temporal Ensembling comparison — 2026-10-26

## Scope and evidence status

This dated experiment compares a fresh [v6 bridge20 control](../configs/banana_v6_bridge20_pi0.toml) with the separate [temporal3 candidate](../configs/banana_v6_temporal3_pi0.toml). It does not pool the earlier [bridge-length experiment](v6-motion-validation-20261026.md) into its control outcomes. The predeclared ten-trial comparison is complete. Its outcomes are reported separately for six GUI trials and four headless validation attempts below.

Both profiles use original pi0 v6 checkpoint `banana_v6_long100_accum8000/7999`, matching normalization, horizon 100, nominal 15-step request age, the same 3 Hz arm filter, RTC, target limits, scene physics, and task-success criteria. Temporal3 replaces the 20-step quintic crossfade with no chunk bridge and an arm-only ensemble of at most three accepted plans, using coefficient 0.01. No server, checkpoint, training, or gripper-decision change is made. The existing bridge20 configuration remains unchanged.

## Predeclared method

The [plan](../outputs/v6_temporal_20261026/plan.json) fixes six native GUI trials at default-center seed 7 in order A-B-B-A-A-B, where A is bridge20 and B is temporal3. Four subsequent headless attempts use preassigned validation layouts: `episode_020` in A-B order and `episode_026` in B-A order, once per profile/layout. No test-partition layout is used. The [harness](../outputs/v6_temporal_20261026/run_comparison.py) checks validation membership, records configuration/script digests and every attempt, refuses to overwrite an existing ledger, and preserves infrastructure errors rather than silently retrying them as new successes.

This is a sequential comparison with fixed scene/reset conditions, not paired policy sampling or matched network histories. The [full ledger](../outputs/v6_temporal_20261026/comparison.json), [experiment log](../outputs/v6_temporal_20261026/experiment.log), and [summary](../outputs/v6_temporal_20261026/summary.json) retain evidence. GUI and validation results must be reported separately. Parameters are fixed for this round rather than retuned after observing outcomes. The predeclared selection rule keeps bridge20 unchanged and considers TE only if motion improves without observed task regression; finite counts cannot establish a general success rate.

## Reference strategy and implementation consistency

This is sparse, ACT-inspired, timestamp-aligned Temporal Ensembling with RTC, not a reproduction of dense ACT inference at every control step. Requests retain the nominal 15-step cadence. Each accepted reply contributes a plan, and the newest three still-valid accepted plans are retained. For any current or forecast timestamp, each eligible plan supplies its prediction for that same timestamp. Only the 12 arm joints are averaged; grippers come from the newest valid raw plan. Oldest-first rank weights are proportional to `exp(-0.01 * rank)`, slightly favoring older predictions, with expiry and renormalization at each queried timestamp. See [Mac tuning](mac-motion-tuning.md) for operation and bounds.

The pure reference is shared by actual execution and RTC forecasting through the copied target limiter and actuator projection, including expiry during a forecast. Averaging does not extend expired plans or create new physical-state predictions. Ensemble mode is mutually exclusive with nonzero client chunk blending. The existing `blended_action` artifact now denotes the selected ensemble reference for this candidate; `temporal_ensemble_counts` records contributors, and `blended_steps` includes steps with more than one contributor.

Four structural checks cover zero-valued actions, expiry and grippers; bounded retention and stale replies; RTC forecasting through the limiter/projection across expiry; and configuration rejection. A frozen legacy bridge20 arm-command reconstruction has maximum error zero. These checks support implementation consistency, not improved model behavior.

The recording summarizer independently reconstructs the weighted reference, verifies contributor counts, compares grippers with the newest raw plan, and checks raw prediction/timestamp alignment. Floating-point reconstruction error near `5.96e-8` is compatible with float32 rounding rather than exact bitwise identity. Across the five actual-model TE trials, maximum contributors were three, all contributor counts matched the independent reconstruction, newest-plan gripper error was zero, and maximum reference-reconstruction error was approximately `5.96e-8`. All ten trials had zero raw-prediction reconstruction error, zero action-timestamp error, zero running inference underruns, and zero maximum RTC forecast-versus-executed prefix error. These establish recording/reference consistency without demonstrating a physical-motion advantage.

## Actual-model outcomes

| Phase / profile | Observed completions | Successful simulated durations, s | Median of trial physical joint-jerk p95, rad/s³ | Worst physical joint jerk, rad/s³ |
| --- | --- | --- | --- | --- |
| GUI / fresh bridge20 | 2/3 | 25.16, 22.74 | 14.10 | 868.50 |
| GUI / temporal3 | 2/3 | 24.12, 22.92 | 17.74 | 1201.83 |
| Validation / fresh bridge20 | 0/2 | None | 13.68 | 1018.15 |
| Validation / temporal3 | 1/2 | 25.36 | 20.74 | 957.92 |

The following trial table follows execution order. Failure descriptions identify recorded events without inventing a physical cause. “Inside box” is an intermediate event, not the complete release/support/stability success rule.

| Trial | Phase / layout | Profile | Outcome / simulated duration | Recorded progress if failed | Physical joint-jerk p95 / maximum, rad/s³ |
| --- | --- | --- | --- | --- | --- |
| 1 | GUI / default-center | bridge20 | Failed / 40.00 s | No confirmed lifted pinch | 12.20 / 174.82 |
| 2 | GUI / default-center | temporal3 | Failed / 40.00 s | No confirmed lifted pinch | 9.55 / 289.13 |
| 3 | GUI / default-center | temporal3 | Success / 24.12 s | — | 17.74 / 288.23 |
| 4 | GUI / default-center | bridge20 | Success / 25.16 s | — | 14.86 / 868.50 |
| 5 | GUI / default-center | bridge20 | Success / 22.74 s | — | 14.10 / 555.52 |
| 6 | GUI / default-center | temporal3 | Success / 22.92 s | — | 19.66 / 1201.83 |
| 7 | Validation / episode_020 | bridge20 | Failed / 40.00 s | No confirmed lifted pinch | 11.45 / 203.18 |
| 8 | Validation / episode_020 | temporal3 | Success / 25.36 s | — | 21.47 / 318.73 |
| 9 | Validation / episode_026 | temporal3 | Failed / 40.00 s | Lifted pinch at 9.44 s; first inside box at 29.18 s; full success not achieved | 20.01 / 957.92 |
| 10 | Validation / episode_026 | bridge20 | Failed / 40.00 s | Lifted pinch at 10.86 s; no inside-box event | 15.92 / 1018.15 |

Both GUI groups completed two of three attempts, while TE's typical GUI jerk was higher and its worst spike was larger. TE completed one additional validation attempt, but its validation median jerk p95 was also higher; validation worst spikes favored TE while GUI worst spikes favored bridge20. This is mixed evidence, not a stable motion advantage or a reliability improvement. No infrastructure error interrupted the ten recorded attempts. Zero inference underruns did not exclude other timing variation: the worst application interval was 124.20 ms in the fresh bridge20 validation block.

The fresh bridge20 control outcomes differ from the preceding dated report, including both failed validation attempts here. Those earlier successful observations remain valid historical records; pooling them into this experiment would obscure the actual control variation. The parameter settings were not retuned after these outcomes.

## Final audit

The completed [consistency audit](../outputs/v6_temporal_20261026/consistency_audit.json) covers all ten trials: application source remained unchanged during execution, model metadata matched the v6 checkpoint/normalization/horizon-100 pins, and raw reconstruction, action-timestamp, and RTC prefix errors were zero. All five TE trials had at most three contributors, matching counts, zero newest-gripper error, and reference reconstruction error below `1e-6`. The [failure diagnostics](../outputs/v6_temporal_20261026/failure_diagnostics.json) retain pinch, inside-box, and final flags for every attempt; intermediate events do not substitute for task success.

The [summary cache audit](../outputs/v6_temporal_20261026/summary_cache_audit.json) recomputed all ten trial summaries, including five TE trials, and exactly matched the original results. Caching lazy NPZ arrays avoids repeated decompression without changing analysis formulas or recorded outcomes; the running harness used the equivalent originally imported summarizer. This analysis optimization does not change application behavior or constitute another model experiment.

## Outcome interpretation and selection boundary

Measured physical joint velocity, acceleration, and jerk use finite differences of the 12 observed arm-joint positions at 0.02 s simulation intervals. Group p95 summaries first compute a p95 within each run and then take the median of those run statistics; they do not pool all control samples. Whole-episode summaries depend on trajectory and duration. A failed or stalled episode with less movement can have a lower jerk p95, so lower movement alone does not demonstrate better control. Task completion, missing lifted-pinches, completion duration, worst physical spikes, gripper reversals, applied-reference changes, timing, and inference underruns must be assessed together.

This round does not establish a stable TE smoothness or task-reliability advantage. Keep bridge20 as the current comparison control and temporal3 as a separate candidate, without default promotion or post-result retuning. Preserve all failures, gripper/progress evidence, and worst physical spikes. Implementation consistency and a single additional validation completion do not resolve the remaining motion problem or establish a population success rate. Earlier historical reports remain unchanged.
