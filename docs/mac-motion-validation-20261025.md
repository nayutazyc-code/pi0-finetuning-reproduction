# Original-pi0 Mac motion validation — 2026-10-25

## Scope and evidence status

This dated report records actual model closed-loop validation of the Mac request-cadence and arm-reference-filter candidates. It is separate from the earlier [offline verification](../outputs/mac_motion_20261025/verification.json). No training, data collection, data conversion, controller-gain change, or task-success-rule change is part of this experiment. The selected [recommended profile](../configs/banana_recommended_pi0.toml) remains unchanged.

This validation round is complete with 27 actual-model episodes: 15 fixed-layout headless trials (four candidates times three repeats, plus three contemporary recommended-profile controls), 10 headless trials across five additional validation layouts, and two separate GUI checks. No application code, weights, or configuration files were changed. These results do not promote a candidate or establish generalization.

## Protocol and compatibility

The [protocol record](../outputs/mac_motion_validation_20261025/protocol.json) binds the experiment to source/configuration hashes and the metadata read before evaluation. The endpoint declared original `pi0`, `pi0_banana_lora`, checkpoint suffix `banana_v3_roi2000/1999`, normalization SHA-256 `2946707a8b5fc28659e96ca0ee5a017559cc6694a5b74f0091b08531764e3f0c`, and pinned upstream revision `215abfb217dbac7d5f1273282331b9b1866c0479`. It reported 14 native action dimensions, a 50-step prediction horizon, 10 internal flow-integration steps, and RTC schema `pi0-rtc-v1` with adapter SHA-256 `94d7c70f6d52accef0d0f7210c7153ee6be7172c6b9ecc2c5f79055b3e33835a`. These are recorded compatibility checks, not a promise of continuing service availability.

The initial comparison uses preassigned validation layout `episode_026` from `20261024-185715-merged-ebdbe3`, layout ID `layout-a716be94fd68`, reset seed `20261026`. Each candidate has three trials. The first cadence25 trial was used to inspect latency and counts among its three trials. All trials have display disabled and retain 25 fps video. Model sampling noise and network timing are not paired. The [trial ledger](../outputs/mac_motion_validation_20261025/trials.json) retains every trial, including failures, its command/configuration, recording path, and diagnostic report. No held-out test layout is used.

The candidate settings and scheduling semantics are described in [Mac motion tuning](mac-motion-tuning.md). The three unfiltered candidates differ in nominal next-request age: 15, 25, or 35 control steps. Filtered25 uses the same cadence as cadence25 and adds a critically damped 3 Hz arm-target filter. The grippers are excluded from this filter. All candidates retain the same weights, RTC guidance settings, target limits, physical control period, and task criteria. A nominal request age is distinct from actual chunk replacement time.

## Initial fixed-layout results

| Candidate | Task success | Confirmed lifted pinch | Near-to-pinch seconds, when defined | Median of trial state-jerk p95, rad/s³ | Running underruns |
| --- | --- | --- | --- | --- | --- |
| cadence15 | 1/3 | 2/3 | 1.66, 1.82 | 1078.4 | 0 |
| cadence25 | 3/3 | 3/3 | 2.08, 2.12, 1.86 | 955.4 | 0 |
| cadence35 | 3/3 | 3/3 | 1.92, 2.80, 2.08 | 883.8 | 0 |
| filtered25 | 3/3 | 3/3 | 2.36, 2.40, 2.12 | 42.7 | 1 |
| recommended, contemporary control | 3/3 | 3/3 | 2.06, 1.86, 1.98 | 1150.2 | 0 |

The jerk entry first computes each trial's p95, then takes the median of the three trial statistics; it does not pool all control samples. [The summarizer](../scripts/summarize_banana_execution.py) takes absolute finite differences of the 12 measured arm-joint positions using the 0.02 s simulation period, pooling joint/time entries within a trial. This is physical simulated joint motion, distinct from requested or submitted target smoothness, and is not a wall-clock jerk measurement. Differences in trajectory and episode length also affect whole-episode distributions. The filtered candidate shows substantially lower measured simulation-time jerk on this layout, while its near-to-pinch intervals are not shorter than unfiltered25's observed intervals.

Cadence15's first trial never achieved a confirmed lifted pinch and reached the 2,000-step limit. Its near-to-pinch duration is undefined, not zero; it had 12 right-gripper command reversals before episode end, including 11 after first entering the near-grasp region. This latter count covers the continuous interval from first region entry until the first lifted pinch or episode end; it does not require every reversal to occur inside the 0.04 m region. Cadence15's third trial achieved a lifted pinch but still failed the full task. A pinch therefore cannot be counted as task success. All failures remain in the denominators and original recordings.

Filtered25's first trial succeeded physically but had one late replacement rejected as stale and one running inference underrun totaling 0.3600 s. Its maximum round trip was 580.10 ms and maximum recorded control interval was 373.96 ms; the maximum recorded server-inference duration was 138.13 ms. The extra delay cannot be assigned to model computation alone, and these records do not independently locate the transport/scheduling cause. The first cadence25 trial also had a 5.798 s startup wait, reported separately from running underruns. Lower simulation-time jerk does not remove these wall-clock interruptions.

The three contemporary recommended-profile controls use the same fixed validation layout and reset seed, retaining the selected configuration. Their median near-to-pinch interval was 1.98 s, compared with 2.36 s for filtered25. Median successful episode duration was 22.02 simulated seconds for recommended, 19.92 s for cadence25, and 29.54 s for filtered25. The filter's motion benefit is therefore accompanied by longer observed completion times on this fixed layout; it has not demonstrated faster grasp progression. The selected profile also differs in rendering and RTC delay floor, so comparisons with it cannot isolate filtering or cadence alone.

## Other five validation layouts

The [extension plan](../outputs/mac_motion_validation_20261025/extension_plan.json) and [extension ledger](../outputs/mac_motion_validation_20261025/extension_trials.json) retain the contemporary controls and additional-layout runs. Filtered25 and unfiltered cadence25 were each evaluated once on every other preassigned validation layout. These are five different layouts, distinct from the three repeats on `episode_026`.

| Validation source episode | cadence25 task outcome | filtered25 task outcome | cadence25 near-to-pinch, s | filtered25 near-to-pinch, s |
| --- | --- | --- | --- | --- |
| episode_004 | Failed after lifted pinch | Success | 3.94 | 4.56 |
| episode_012 | Success | Success | 2.20 | 2.46 |
| episode_019 | Failed after lifted pinch | Success | 2.02 | 4.48 |
| episode_020 | Success | Success | 9.46 | 7.60 |
| episode_029 | Success | Success | 2.24 | 3.04 |

Across this separate five-layout block, filtered25 succeeded 5/5 and cadence25 succeeded 3/5. All 10 runs achieved a confirmed lifted pinch. The median of trial state-jerk p95 was 48.2 rad/s³ for filtered25 and 953.3 rad/s³ for cadence25; median defined near-to-pinch duration was 4.48 s versus 2.24 s. Neither group had a running underrun or stale chunk in this block. Successful completion and faster grasp progression are separate outcomes: filtering retained task success in these observations while four of the five layouts had longer near-to-pinch intervals than their unfiltered counterpart.

Filtered25 thus completed eight headless runs across six validation layouts, including three repeated runs on one layout. These are eight observed successes, not eight independent layouts, and the comparison is small and unpaired. It does not establish a reliable population success rate or justify silently replacing the selected profile.

## Separate native GUI checks

The [GUI protocol](../outputs/mac_motion_validation_20261025/gui_protocol.json) and [GUI ledger](../outputs/mac_motion_validation_20261025/gui_trials.json) record one native GLFW run per profile on the default-center scene with seed 7. These two runs are separate from the validation-layout blocks, with unpaired policy noise.

| Profile | Task outcome | Simulated duration, s | Control wall duration, s | Near-to-pinch, s | State-jerk p95, rad/s³ | Control-interval p95, ms |
| --- | --- | --- | --- | --- | --- | --- |
| recommended | Success | 15.24 | 15.649 | 1.68 | 1040.8 | 26.45 |
| filtered25 | Success | 23.50 | 23.929 | 2.62 | 41.4 | 31.50 |

The [recommended recording](../outputs/20261025-200226-d615c0/summary.json) and [video](../outputs/20261025-200226-d615c0/episode_000.mp4), and the [filtered recording](../outputs/20261025-200243-0e22c5/summary.json) and [video](../outputs/20261025-200243-0e22c5/episode_000.mp4), retain the source evidence. Both runs had zero running underruns, zero stale chunks, and zero control intervals above 100 ms. Filtered25 again had much lower simulation-time joint jerk, but its task and grasp progression took longer. Its control-interval p95 was higher, so these results do not support a claim that filtering improves overall wall-clock speed or timing stability. The profiles also differ in render/display scheduling and RTC delay floor, which confound comparisons with the recommended configuration. One run per profile supports only a limited native-display check, not a general responsiveness guarantee.

## Recording integrity

The [comparison](../outputs/mac_motion_validation_20261025/comparison.json) preserves aggregate and per-trial metrics. The [integrity audit](../outputs/mac_motion_validation_20261025/integrity.json) checked all 27 episodes: metadata matched, states and commands were finite, timestamp alignment error and raw-prediction reconstruction error were zero, and PNG pixel hashes were verified. Initial robot-state differences within each layout were zero. Across 1,586 forecast-prefix comparisons, maximum forecast-versus-executed command error was zero. Source hashes matched the preceding offline verification record and the pinned upstream worktree remained clean. These checks support implementation and recording consistency, not task generalization or hard real-time guarantees.

## Interpretation boundaries and continuation

Near-grasp uses the existing 0.04 m finger-midpoint-to-banana-center diagnostic. It does not change actions or success criteria. Missing pinch events remain missing. Target continuity, commanded movement, measured movement, gripper reversals, task completion, and wall-clock timing must be considered together. Inclusive stage timers are not an exclusive CPU-time decomposition.

Three repeats on one layout and one run per additional layout cannot establish a reliable success-rate advantage or a globally optimal request cadence. The historical successful GUI recording is a different layout/run and cannot serve as a paired contemporary baseline. Headless trials do not establish GUI responsiveness, and video playback follows simulation time. Filtered25 is the preferred candidate for further Mac smoothing evaluation because measured joint jerk was consistently lower in these observations. Grasp hesitation remains unresolved: most additional layouts and the GUI comparison took longer to reach a lifted pinch, and the recorded running interruption must remain visible. The recommended profile remains selected; no default promotion occurred. No held-out tests, retraining, or model-family changes were performed.
