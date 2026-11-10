# Original-pi0 v6 request-cadence optimization — 2026-10-26

## Scope and status

Stage 1 compares three request ages using the same v6 bridge20 execution strategy: [cadence15](../configs/banana_v6_bridge20_pi0.toml), [cadence30](../configs/banana_v6_cadence30_pi0.toml), and [cadence50](../configs/banana_v6_cadence50_pi0.toml). Only `request_interval_steps` changes. Checkpoint, normalization, horizon 100, RTC, 3 Hz arm filter, quintic bridge length 20, target limits, scene physics, and rendering remain fixed. No training or data change is part of this experiment. Stage 1 is complete with 12 actual-model attempts: six GUI trials and six separate headless validation attempts. The existing cadence15 remains the next control; this comparison does not promote cadence30 or cadence50.

The initial metadata connection was refused. The user restored forwarding, after which the original-pi0 v6 horizon-100 endpoint was verified. Connection recovery is a prerequisite record, not model-performance evidence or proof that a prior grasp failure was caused by transport.

## Predeclared protocol

The [plan](../outputs/v6_cadence_20261026/plan.json), [harness](../outputs/v6_cadence_20261026/run_comparison.py), and [live ledger](../outputs/v6_cadence_20261026/comparison.json) retain the fixed experiment. Six native GUI runs use order 15, 30, 50, 50, 30, 15, giving two fresh runs per profile. Six headless validation attempts use preassigned `episode_020` in order 15, 30, 50, followed by `episode_026` in order 50, 30, 15, once per profile/layout. No test layout is used. Scene/reset conditions are repeated, but model noise and network timing are unpaired. GUI and validation outcomes remain separate; earlier bridge20 and TE records are not pooled into these fresh controls.

At 50 Hz, request ages 15, 30, and 50 correspond to nominal 0.30, 0.60, and 1.00 s from the active observation origin. Their old-plan overlaps at request are 85, 70, and 50 steps. Requests do not lengthen the two-second prediction horizon, append every complete chunk, or guarantee a switch at the nominal request age. One request remains in flight; actual replacements depend on latency and accepted-plan validity.

## Closure-intent and placement diagnosis

Inspection of earlier failed recordings found predicted gripper closure 55–89 steps ahead while executed gripper targets remained open. Repeated replanning might postpone an intended phase transition, but this is a hypothesis rather than a uniquely established cause. Frozen future predictions do not prove the policy would have closed successfully if left to execute, and gripper projection/rate limits, arm alignment, contact, and new observations can affect the realized transition.

Evaluate raw gripper intent and its predicted time against actual submitted/executed gripper targets, confirmed lifted pinch, first inside-box event, release, and full task completion. Missing pinch or release events remain missing rather than zero-duration improvements. Separate time near the banana from time above/inside the box and preserve failed trajectories. Lower whole-episode jerk can reflect stalled or reduced motion; it is not sufficient evidence of better control. Interpretation must include physical jerk p95 and worst spikes, completion duration, repeated gripper reversals, target/reference changes, inference underruns, and application timing.

## Recorded outcomes

The completed [summary](../outputs/v6_cadence_20261026/summary.json) retains per-trial motion, task progress, timing, and reconstruction fields. Group jerk values below are the median of each run's p95 across the 12 measured arm joints, computed by finite differences on the 0.02 s simulation interval; maxima retain isolated spikes. Different movement and episode lengths affect these distributions.

| Phase / request age | Observed completions | Successful simulated durations, s | Median of trial physical jerk p95, rad/s³ | Worst physical jerk, rad/s³ |
| --- | --- | --- | --- | --- |
| GUI / 15 | 2/2 | 28.92, 22.80 | 15.54 | 2201.90 |
| GUI / 30 | 2/2 | 29.88, 22.24 | 16.33 | 1042.61 |
| GUI / 50 | 2/2 | 22.88, 25.10 | 15.69 | 1265.64 |
| Validation / 15 | 2/2 | 26.52, 32.48 | 16.86 | 1233.11 |
| Validation / 30 | 1/2 | 26.56 | 17.90 | 2081.73 |
| Validation / 50 | 1/2 | 23.06 | 16.98 | 1820.49 |

| Trial | Phase / layout | Request age | Outcome / simulated duration | Near-to-pinch interval, s |
| --- | --- | --- | --- | --- |
| 1 | GUI / default-center | 15 | Success / 28.92 s | 2.42 |
| 2 | GUI / default-center | 30 | Success / 29.88 s | 4.08 |
| 3 | GUI / default-center | 50 | Success / 22.88 s | 3.30 |
| 4 | GUI / default-center | 50 | Success / 25.10 s | 4.34 |
| 5 | GUI / default-center | 30 | Success / 22.24 s | 2.74 |
| 6 | GUI / default-center | 15 | Success / 22.80 s | 2.84 |
| 7 | Validation / episode_020 | 15 | Success / 26.52 s | 4.76 |
| 8 | Validation / episode_020 | 30 | Failure / 40.00 s; no confirmed lifted pinch | Undefined |
| 9 | Validation / episode_020 | 50 | Success / 23.06 s | 3.82 |
| 10 | Validation / episode_026 | 50 | Failure / 40.00 s; lifted pinch at 13.04 s, full task not achieved | 6.16 |
| 11 | Validation / episode_026 | 30 | Success / 26.56 s | 2.32 |
| 12 | Validation / episode_026 | 15 | Success / 32.48 s | 2.88 |

All 12 attempts had zero running inference underruns and zero maximum RTC prefix error. The [stage-specific consistency audit](../outputs/v6_cadence_20261026/consistency_audit.json) confirms raw reconstruction and action-timestamp errors were also zero and application sources stayed stable through Stage 1. That audit was recorded before later integration work; it is not a claim that the current source will remain identical after subsequent changes.

## Thresholded phase observations

The [phase diagnostics](../outputs/v6_cadence_20261026/phase_diagnostics.json) compare future closure intents with the span during which a plan actually supplied actions, using post-hoc thresholds: predicted gripper below 0.6, currently open above 0.8, near-box XY radius 0.10 m, high banana Z above 0.10 m, and still-speed below 0.01 m/s. These are analysis thresholds, not policy inputs, action rules, or independent evidence of human-visible pauses.

Counts of future closures outside the executed plan span were 2 and 4 for cadence15's two successful GUI runs and 6 and 4 for its two successful validation runs. Thus deferrals occur in successful cadence15 trajectories too and cannot by themselves identify a unique failure cause. Cadence30's failed `episode_020` attempt had two such intents; cadence50's failed `episode_026` attempt had one despite achieving a lifted pinch. Postponement remains a useful diagnostic hypothesis, not a proven explanation resolved by less frequent replanning.

The longest thresholded high/near-box still interval ranged from 0 to 0.40 simulated seconds across this round. Both the successful cadence15 and failed cadence50 `episode_026` runs reached 0.40 s under this definition. These intervals do not identify the user's exact perceived pause and do not establish that placement hesitation was eliminated.

## Later-stage boundary

Jerk-limited reference work is a separate investigation, not an operational setting in this cadence experiment. A standalone local prototype and structural checks do not establish integration into the runtime limiter or an actual-model result. Any later integration needs its own implementation/forecast consistency and task evaluation. If targeted recovery data becomes necessary, collection must use preassigned training layouts with retained provenance; validation/test observations must not silently become training corrections. No later-stage implementation, data collection, or training outcome is claimed here.

## Selection boundary

These 12 attempts are a small sequential comparison with unpaired sampling, not a population success-rate estimate or proof of causal superiority. All three profiles completed their two GUI trials, while longer request intervals each failed one of two validation attempts. There is no consistent observed grasp-progress or typical-jerk benefit that justifies replacing cadence15. Keep cadence15 as the next control and retain cadence30/50 as unpromoted comparisons. The closure/placement issue is not declared solved; all failures and isolated spikes remain in the record. Later-stage work requires its own evidence rather than rewriting these Stage 1 outcomes.
