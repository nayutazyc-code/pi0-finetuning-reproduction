# Original-pi0 v6 RTC guidance-strength comparison — 2026-10-26

## Scope and current status

This separate runtime comparison follows the [v7 recovery result](v7-recovery-optimization-20261026.md), which did not establish a stable training gain and retained v6. The candidate [RTC 0.1 configuration](../configs/banana_v6_rtc01_pi0.toml) changes only `rtc_guidance_max` from 0.5 to 0.1 relative to [v6 bridge20](../configs/banana_v6_bridge20_pi0.toml). It preserves checkpoint/normalization, horizon 100, cadence15, bridge20, 3 Hz filter, limits, and other execution settings. No new training or sampler algorithm is introduced. The six candidate attempts are complete. Guidance 0.1 did not improve the observed completion count and regressed grasp progress on one layout; retain v6 bridge20 with guidance 0.5.

## Fixed method and hypothesis

The [plan](../outputs/v6_rtc_strength_20261026/plan.json) fixes the existing validation source IDs in order `episode_004`, `episode_012`, `episode_019`, `episode_020`, `episode_026`, `episode_029`, once each with display disabled. The [ledger](../outputs/v6_rtc_strength_20261026/comparison.json) preserves every attempt. No test layout is used and no further strength tuning is planned after observing these results.

The comparison control is the six preserved v6 validation attempts in the [same-round model comparison](../outputs/v6_recovery_20261026/model_comparison.json), not pooled observations from earlier cadence, jerk, or Temporal Ensembling screens. Those controls completed five of six layouts, with no confirmed lifted pinch on `episode_020`. Default confirmations are excluded from this validation comparison. Candidate and control noise/network histories are unpaired and were recorded sequentially.

Strong prefix guidance could preserve a preceding action intention longer than useful and contribute to repeated phase postponement. This is a hypothesis, not a proven cause of grasp or placement hesitation. Lowering its cap could also weaken continuity. Metadata in each RTC response can verify the actual guidance setting but cannot prove that it improves behavior.

Assess completion and phase progress first, then physical jerk p95 and worst spikes, completion duration, gripper intent/execution, application timing, and inference underruns together. Preserve failed or stalled trajectories: low movement can reduce whole-episode jerk without improving control. The fixed rule retains 0.5 unless 0.1 improves completion and phase progress without material motion regression. Six single attempts cannot establish general reliability, a causal mechanism, or absence of visible pauses.

## Completed observations

The [completed strength comparison](../outputs/v6_rtc_strength_20261026/strength_comparison.json) records one attempt per cap/layout. “Near-to-pinch” uses the existing diagnostic region and confirmed lifted-pinch event; undefined values are not zero-duration improvements.

| Validation layout | RTC 0.5 outcome / simulated duration | RTC 0.1 outcome / simulated duration | RTC 0.5 near-to-pinch, s | RTC 0.1 near-to-pinch, s |
| --- | --- | --- | --- | --- |
| episode_004 | Success / 21.66 s | Success / 20.86 s | 3.08 | 2.64 |
| episode_012 | Success / 21.38 s | Success / 21.66 s | 2.50 | 2.76 |
| episode_019 | Success / 24.52 s | Success / 33.44 s | 2.90 | 14.78 |
| episode_020 | Failure / 40.00 s; no confirmed lifted pinch | Failure / 40.00 s; no confirmed lifted pinch | Undefined | Undefined |
| episode_026 | Success / 30.48 s | Success / 31.88 s | 2.40 | 2.88 |
| episode_029 | Success / 21.00 s | Success / 22.76 s | 2.42 | 2.94 |

| Guidance cap | Observed completions | Median successful duration, s | Median of trial physical joint-jerk p95, rad/s³ | Worst physical joint jerk, rad/s³ |
| --- | --- | --- | --- | --- |
| 0.5 | 5/6 | 21.66 | 14.87 | 1416.23 |
| 0.1 | 5/6 | 22.76 | 15.16 | 2091.03 |

Physical jerk uses finite differences of observed arm positions on the 0.02 s simulation interval. Group medians summarize each trial's p95; they do not pool all samples. The completion count did not improve, successful durations were generally longer, and `episode_019` showed substantially longer near-to-pinch delay. Typical jerk and the worst spike also did not favor 0.1. These observations do not establish that prefix guidance caused the remaining hesitation or that some other strength would resolve it.

## Final consistency audit

The [final audit](../outputs/v6_rtc_strength_20261026/final_consistency_audit.json) covers the 16 v6/v7 comparison trials plus these six RTC attempts. Application source hashes were identical across all 22 trials and matched the source at audit time. Raw-action reconstruction, action timestamps, and RTC command-prefix errors were zero, and all trials had zero running inference underruns. Request/response guidance caps and checkpoint/normalization pins matched; all declared original pi0 horizon 100 without jerk-reference or Temporal Ensembling. The six matched RTC scenes and reset seeds were identical. The RTC configuration comparison differed only in guidance cap, and the v6/v7 client comparison only in checkpoint suffix.

These checks establish consistency, not hard real-time behavior, matched stochastic sampling, or task reliability. The audit's source equality is scoped to this round and its audit time; it does not freeze future implementation changes.

## Selection boundary

Retain the v6 bridge20 configuration with RTC guidance 0.5. The 0.1 profile remains an unpromoted research comparison, with its failed grasp, long `episode_019` delay, and larger worst spike preserved. No further strength retuning or retraining was performed in this round. The finite unpaired evidence does not solve all pauses, guarantee smoothness, or establish a population success rate.

V6 restoration was verified earlier in the comparison. The [final endpoint snapshot](../outputs/v6_rtc_strength_20261026/final_endpoint_status.json) records a local forwarding listener but a timed-out metadata handshake. The task's ControlMaster socket was absent or expired and unattended SSH access was denied, leaving the current remote service state unverified. The empty task temporary directory was removed; no service restart or new training followed. Earlier successful restoration and the completed model comparisons remain valid recorded evidence, but do not establish continuing endpoint availability.

Assistant inspection of the [`episode_019` contact sheet](../outputs/v6_rtc_strength_20261026/episode019_contact_sheet.jpg) found the 0.1 trajectory still near the banana around 20 s while the control was already above the box, consistent with the delayed-pinch metric. This does not establish general visual smoothness or the delay's unique cause.
