# Original-pi0 v6 bridge-length validation — 2026-10-26

## Scope and evidence status

This dated round investigates the user's report of obvious jumps in actual v6 motion. It compares the existing [v6 bridge8 profile](../configs/banana_v6_long100_pi0.toml) with the separate [bridge20 candidate](../configs/banana_v6_bridge20_pi0.toml). The only runtime configuration difference is `chunk_blend_steps`, from 8 to 20: a nominal 0.16 s arm-reference crossfade becomes 0.40 s at 50 Hz. Both use original pi0 checkpoint `banana_v6_long100_accum8000/7999`, matching normalization, horizon 100, request age 15, 3 Hz arm filtering, RTC, and the same target limits. No retraining or new transition algorithm is part of this round, and the existing profiles remain unchanged.

This round is complete with six default-center GUI recordings and four separate headless attempts on preassigned validation layouts `episode_020` and `episode_026`. No test-partition layouts are used. No new profile is promoted.

## Diagnostic method and boundaries

The existing `ActionReference` crossfades old/new arm references at the same simulation timestamp, retaining nested unfinished transitions; grippers are excluded. RTC forecasting uses the same reference through a copied limiter and actuator projection. A transition shortens to valid old-plan overlap rather than extending expired commands. See [Mac motion tuning](mac-motion-tuning.md) for operation and limitations.

The [frozen-command sweep](../outputs/v6_motion_20261026/frozen_command_sweep.json) reconstructs the three earlier v6 recordings from frozen predictions and acceptance times without physics or fresh model feedback. It compares bridge length and several additional limiter/filter settings; only bridge20 was selected for the actual-model comparison here. Baseline arm-command reconstruction error is zero in all three source runs. Gripper projection is omitted from this command-only diagnostic. Lower reconstructed command acceleration/jerk motivates the candidate but does not establish physical smoothness, task success, or how pi0 responds to changed observations.

The [aborted local diagnostic](../outputs/v6_motion_20261026/aborted_offline_attempt.json) was terminated with exit code 143 after graphical initialization stalled with macOS connection errors. It produced no motion evidence and is not counted as either a physical replay result or a model trial. Its replacement was the pure command-only reconstruction above.

## Actual-model GUI observations

The [GUI comparison](../outputs/v6_motion_20261026/gui_comparison.json) retains full diagnostics and run directories; the [compact comparison](../outputs/v6_motion_20261026/comparison.json) records group summaries and configuration hashes. All six runs use the default-center scene with reset seed 7. The three bridge8 recordings comprise one assistant run and two earlier user runs; the three bridge20 GUI runs followed afterward. Model noise and network histories are unpaired, and this is a small sequential comparison rather than a controlled estimate of success rates or causality.

| GUI group | Observed task completions | Successful simulated duration, s | Median of per-run physical joint-jerk p95, rad/s³ | Worst physical joint jerk, rad/s³ | Running inference underruns |
| --- | --- | --- | --- | --- | --- |
| v6 bridge8 | 2/3 | 23.40–24.80 | 34.89 | 647.26 | 0 |
| v6 bridge20 | 3/3 | 22.90–32.74 | 15.57 | 1208.60 | 0 |

The bridge8 failure did not achieve the required grasp; it remains in all group observations. The joint-motion statistics are absolute finite differences of the 12 measured arm-joint positions at the 0.02 s simulation interval. Each run's p95 is computed first, then the three p95 values are summarized by their median. The approximately 55% lower median in the bridge20 recordings is descriptive of these trajectories, not a paired or causal effect estimate. Episode lengths, contacts, and trajectories differ. The longer candidate completion range is consistent with a possible responsiveness/lag tradeoff, not proof of faster task progression.

The maximum individual physical-jerk spike increased rather than decreased. The largest bridge20 spike was identified at step 455 (9.10 simulated seconds), action/state index 11 on the right arm, while applied commands were smooth and grasp/table-contact activity was underway. This temporal context does not prove that contact caused the spike. The candidate therefore cannot be described as uniformly smoother or free of isolated jumps.

Across all six GUI runs, maximum forecast-versus-executed RTC prefix error was zero and there were no running inference underruns. The worst recorded action-application interval was 126.11 ms for bridge8 and 40.63 ms for bridge20; zero inference underruns does not exclude other long frames. The first bridge8 run also had a 55.63 s startup wait, which must remain separate from running-control timing. The user directly reported that jumps were reduced after viewing the three bridge20 windows. This supports a limited subjective improvement report, not guaranteed smoothness or task reliability across layouts.

## Limited validation and selection boundary

The [validation comparison](../outputs/v6_motion_20261026/validation_comparison.json) records one attempt per profile/layout with display disabled. The original profile's two-layout evaluation ran first, followed by the candidate's two-layout evaluation. Sampling noise and network histories are unpaired. The [bridge8 evaluation](../outputs/banana_evaluations/20261026-120943-81f90e/evaluation.json) and [bridge20 evaluation](../outputs/banana_evaluations/20261026-121056-3ffba2/evaluation.json) retain original results and recording paths.

| Validation layout | bridge8 outcome / simulated duration | bridge20 outcome / simulated duration | bridge8 physical joint-jerk p95, rad/s³ | bridge20 physical joint-jerk p95, rad/s³ |
| --- | --- | --- | --- | --- |
| episode_020 | Failed at 40.00 s; no confirmed lifted pinch | Success at 24.04 s | 20.41 | 18.20 |
| episode_026 | Success at 29.84 s | Success at 29.64 s | 52.32 | 15.22 |

All four had zero running inference underruns, zero action-timestamp and raw-prediction reconstruction error, and zero maximum RTC prefix-forecast error. The bridge8 `episode_020` near-to-pinch interval is undefined, not zero. These observations give one completion in two validation attempts for bridge8 and two in two for bridge20. They remain a tiny, sequential, single-attempt comparison per layout, separate from the GUI block's two in three versus three in three observations. They are not estimates of general reliability or proof that bridge length caused the outcome difference. Headless outcomes do not establish GUI responsiveness, and validation layouts must not be relabeled as held-out tests.

The bridge20 configuration is retained as a candidate for further user observation, without default promotion. Preserve the failed grasps, the worse isolated physical spike, the 32.74 s candidate GUI completion, possible added lag, aborted infrastructure attempt, and original profile for comparison. The user's reduced-jump feedback and lower typical joint-jerk measurements support continuing to observe this candidate. Neither the frozen-command sweep nor these finite GUI/validation observations establish a broadly improved success rate, eliminate all jumps, or guarantee smoother physical motion. No algorithm implementation, training, or checkpoint change occurred in this comparison.
