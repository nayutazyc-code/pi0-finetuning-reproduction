# Final candidate acceptance — 2026-10-30

## Status and scope

**All 24 attempts and the numerical audit are complete. The decision is `retain_v6_reference`: two typical-jerk gates failed.** The candidate completed 12/12 tasks versus v6's 10/12, with favorable physical-peak and observed timing results, but its typical-jerk reduction was 16.4%, below the frozen 20% requirement. This separate acceptance protocol compares original-pi0 v6 with the relabeled-data candidate plus doubled wrist damping. It does not pool the [earlier four-profile comparisons](release-retreat-optimization-20261029.md), OOD evaluation or individual demonstration videos.

The [attempt02 plan](../outputs/final_candidate_acceptance_20261030_attempt02/plan.json), [complete ledger](../outputs/final_candidate_acceptance_20261030_attempt02/comparison.json), [aggregate](../outputs/final_candidate_acceptance_20261030_attempt02/aggregate.json), [658-check audit](../outputs/final_candidate_acceptance_20261030_attempt02/audit.json) and [completion receipt](../outputs/final_candidate_acceptance_20261030_attempt02/completion.json) preserve the protocol and every attempt. The initial zero-case preparation failure is retained separately below. Primary sampled-image review, post-run checkpoint/source verification and v6 restoration are complete. No training, dataset change or recommended-profile change followed this acceptance run; successful review does not reverse the two failed numerical gates.

## Frozen profiles and design

| Property | v6 reference | Combined candidate |
| --- | --- | --- |
| Checkpoint suffix | `banana_v6_long100_accum8000/7999` | `banana_v6_ik_candidate1000/999` |
| Simulated wrist damping scale | 1 | 2 |
| Planned attempts | 12 | 12 |
| Unique validation layouts | The same six existing layouts | The same six existing layouts |
| Repeats per layout | 2 | 2 |

The six layouts are `episode_004`, `episode_012`, `episode_019`, `episode_020`, `episode_026` and `episode_029`. Their scenes, reset seeds and validation mapping are frozen in the plan. Four six-layout blocks use **v6 → candidate → candidate → v6** order, with predeclared rotated/reversed layout orders. Each block selects its policy and performs its own readiness requests, including the two consecutive candidate blocks.

Both profiles use H100, denoise10, generation-time RTC20, a separate 20-step client bridge, 3 Hz arm filtering, original target limits, headless 25 fps top-camera recording and a 40-second simulation-time maximum. The policy input and normalization are shared. The required normalization SHA256 is `6c6e7a92643c76996bf15ba9fc3d3e346dd6bc3d8697712118d9fdb49dedf269`.

The intervention combines different learned weights and different simulated damping. An observed gain cannot be attributed to weights alone. These are six previously used validation layouts, not twelve independent layouts per group or a fresh held-out generalization test. The plan explicitly sets `noise_paired=false`; equal resets do not pair policy noise, latency histories or physical contact trajectories. ABBA ordering reduces one temporal-order risk but does not eliminate drift.

## Source, checkpoint and runtime binding

The plan binds source/configuration SHA256 values and copies their versions into [frozen_source](../outputs/final_candidate_acceptance_20261030_attempt02/frozen_source/). It also binds the validation mapping and [remote preflight receipt](../outputs/final_candidate_preflight_20261030/remote_receipt.json). Preflight recorded complete manifests for v6 (37 files, 5,591,368,295 bytes) and candidate (38 files, 5,591,753,609 bytes), with the expected normalization digest for both. Those are dated checks, not a continuing guarantee of remote availability.

The new [runner](../scripts/evaluate_final_candidate.py) checks the frozen inputs during execution. No source/configuration edits are allowed within the active cohort. Per-trial metadata, action-prefix reconstruction, prediction/action timestamps, scene binding and video provenance must remain auditable. Missing metrics, incomplete cases or audit failures do not count as passing acceptance.

The protocol permits the current unchanged route, including a TUN route if present, without changing VPN settings. **The actual route observed for attempt02 is `en0`.** Route checks occur at block/trial boundaries; an interface change aborts and preserves the partial cohort. This is a new network-scoped protocol, not a reuse of the older no-TUN four-profile acceptance gates. Results must remain scoped to the route and host workload actually recorded.

Each block performs six inference readiness requests without physical environment stepping. The last four must have median RTT strictly below 800 ms and maximum RTT strictly below 1,500 ms. Reset/render and warmup requests are readiness work, not scored policy episodes. An operational failure is retained rather than repaired through outcome-selected reruns.

## Frozen acceptance requirements

The following requirements are fixed before scored outcomes. Ratios and offsets use the new v6 cohort, not a historical favorable baseline. All checks are conjunctive and use unrounded values.

| Measure | Candidate requirement |
| --- | --- |
| Complete/scorable attempts | 12 per group, with all 12 containing scorable active motion; complete 24-attempt cohort |
| Complete task successes | At least 10/12 and no fewer than v6 |
| Transport-covered attempts | At least 10/12 and no fewer than v6 |
| All-attempt median active physical-jerk p99 | At most 0.8 × v6 |
| Transport-covered median active physical-jerk p99 | At most 0.8 × v6 |
| Worst active-motion and all-frame physical-jerk peaks | Each at most v6 |
| Worst physical joint increment | At most 1.05 × v6 |
| Longest thresholded low-motion pause | At most v6 + 0.1 s |
| Median total thresholded low-motion pause | At most v6 + 0.1 s |
| Covered median simulation duration | At most 1.1 × v6 |
| Covered median control-wall duration | At most 1.1 × v6 |
| Median of trial application-interval p99 values | At most 1.1 × v6 |
| Worst application interval across trials | At most 1.1 × v6 |
| Total running inference underruns | No more than v6 |
| Total running inference-wait duration | At most v6 + 0.1 s |

The 10/12 success floor is an operational threshold for this run, not a statistical guarantee of reliability or a formal non-inferiority result. Failure of any numerical requirement, integrity audit or required review prevents recommendation. Thresholds are not relaxed after inspecting outcomes.

Physical jerk is measured from recorded joint motion on simulation time. It must not be confused with requested-target derivatives or wall-clock stutter. All-attempt and transport-covered summaries remain separate: conditioning on achieved progress can change cohort composition, so covered smoothness does not erase failures. Missing phases are unavailable, not zero-duration or zero-jerk wins. A zero thresholded low-motion pause does not exclude slow, moving hesitation.

Application intervals and inference-wait measurements address real control timing separately from video playback and simulation duration. The application-p99 requirement uses the median of per-trial p99 values, not a pooled frame percentile. Small-cohort maxima are sensitive to individual events and do not establish a population tail distribution.

## Retained preparation failure and unchanged retry scope

The [original directory](../outputs/final_candidate_acceptance_20261030/) preserves its [plan](../outputs/final_candidate_acceptance_20261030/plan.json) and [incomplete ledger](../outputs/final_candidate_acceptance_20261030/comparison.json). It stopped at local generated-scene validation with `AssertionError: Generated scene changed`: equivalent list/tuple representations were compared directly. There were **zero formal policy cases and zero warmup model predictions**, so this is a tooling/preparation failure rather than a failed task attempt.

The original runner dispatched v6 restoration, but the same local validator prevented its normal restoration verification. The separate [metadata readback](../outputs/final_candidate_acceptance_20261030/restoration_readback.json) subsequently confirmed v6/7999 and the expected normalization digest. The original ledger's failed verification remains preserved; the independent readback is supplementary evidence, not a rewrite of that ledger.

The corrected runner normalized the scene-validation representation and added coverage for actual configured layouts. Comparing the two plans shows that only `source_sha256` changed, and within it only `scripts/evaluate_final_candidate.py`. Cases, profiles, thresholds, checkpoint bindings and runtime scope are identical. Attempt02 uses a new output directory and does not overwrite the failed preparation attempt. This repair is not an outcome-based model/threshold change.

## Checks and restoration boundary

Eight runner tests passed after the repair, including the twelve real model/layout configuration combinations. Four existing launcher checks and six damping checks also passed. These are bounded local checks; they are not training, policy quality or proof of every failure mode. The runner's `finally` restoration path was inspected in source, but a deliberate live failure-injection test was not performed and is not claimed.

On completion or failure, the runner attempts v6 selection and verifies metadata separately, then cleans up its owned tunnel. Actual restoration must be read from the receipt; abrupt termination or network failure can prevent it. A successful policy-selection dispatch alone is not verification. The acceptance workflow restores v6 first; only a complete numerical/integrity pass plus primary review can support a later candidate recommendation. No automatic promotion is performed.

## Complete results and unchanged decision

All 24 formal attempts completed, and all 658 integrity checks passed. The frozen gate evaluation passed 15 of 17 checks. Only `typical_jerk` and `covered_typical_jerk` failed; no threshold was changed after observing the results.

| Measure | v6 / damping1 | Candidate999 / damping2 |
| --- | --- | --- |
| Complete task successes | 10/12 | 12/12 |
| Transport-covered attempts | 10/12 | 12/12 |
| All-attempt median active jerk p99 (rad/s³) | 94.435259 | 78.946594 |
| Covered median active jerk p99 (rad/s³) | 94.435259 | 78.946594 |
| Worst active / all-frame physical jerk (rad/s³) | 1052.774489 | 815.988460 |
| Worst physical joint increment (rad) | 0.022060931 | 0.022025049 |
| Longest thresholded low-motion pause (s) | 0.90 | 0 |
| Median total thresholded low-motion pause (s) | 0 | 0 |
| Covered median simulation duration (s) | 22.00 | 21.62 |
| Covered median control-wall duration (s) | 22.415698 | 21.975659 |
| Median trial application-interval p99 (ms) | 20.911051 | 20.883390 |
| Worst application interval (ms) | 684.796542 | 65.854250 |
| Total running inference underruns | 1 | 0 |
| Total running inference wait (s) | 0.682524 | 0 |

The combined candidate's typical physical jerk was **16.4014% lower** and its worst peak **22.4916% lower** in this cohort. Both typical-jerk requirements needed a candidate value at most `94.43525897950168 × 0.8 = 75.54820718360135` rad/s³; the observed `78.94659414887423` did not meet that bound. All-attempt and covered medians happen to coincide numerically here, but their denominators remain different. Favorable completion, peak and timing observations do not override the two failed requirements.

The two v6 failures were case5 (`episode_029`, repeat1) and case22 (`episode_026`, repeat2). Both had `ever_grasped=false`, `ever_lifted=false` and `final_inside_box=false`; neither confirmed a lifted pinch or box entry. They remain in the full 12-attempt denominator, video table and all-attempt metrics. All twelve candidate tasks completed. This finite result on reused validation layouts does not establish a population success rate or statistical superiority.

V6's single running underrun occurred in case1 (`episode_012`, repeat1), with 0.682524 s of waiting and a 684.796542 ms maximum application interval. Candidate's largest observed interval was 65.854250 ms with no running underruns. These are valid observations under this cohort's recorded `en0` route, but one baseline interruption cannot establish general latency improvement, identify its unique cause or prove that future candidate execution will be stall-free. Video playback alone does not measure this timing difference.

The recorded restoration has `verified=true`, with v6/7999, the expected normalization and RTC metadata. This is a completed end-of-run check, not a continuing availability guarantee. The candidate remains an optional experiment; the reference/recommended configuration was not promoted, and no retraining or parameter tuning followed the result.

## Media and post-run verification boundary

All 24 original videos are retained below, including both v6 failures. The [media manifest](../outputs/final_candidate_acceptance_20261030_attempt02/media/manifest.json) binds six start/first-lift-or-midpoint/final contact sheets. The [primary review receipt](../outputs/final_candidate_acceptance_20261030_attempt02/primary_review.json) records actual inspection of all six sheets, covering all 24 attempts. Final sampled images show the banana inside the box for the declared successful cases; task logs establish the exact release, support and stability conditions. This is sampled static review, not complete continuous playback of all videos, and it does not establish wall-clock continuity.

The separate [remote post-run receipt](../outputs/final_candidate_preflight_20261030/remote_post_receipt.json) verifies that both complete checkpoint-tree hashes match their pre-run manifests, all four bound serving/selector source files are unchanged, and v6 metadata has been restored. The primary agent also rehashed all 346 pre-existing local source/configuration files listed in the [local pre-run manifest](../outputs/final_candidate_preflight_20261030/local_sources_before.json), finding no changes. Only the acceptance runner/test and associated documentation were added for this task. No training or promotion occurred. These completed checks preserve provenance; they do not turn a failed acceptance threshold into a pass or guarantee future service availability.

## All original attempts

Case numbers retain the frozen ledger's zero-based order. `candidate_damping2` means the combined candidate checkpoint and simulated damping intervention.

| Case | Profile | Layout / repeat | Outcome | Original video |
| --- | --- | --- | --- | --- |
| 0 | v6 | episode_004 / 1 | Success | [Video](../outputs/20261030-115617-8ddab1/episode_000.mp4) |
| 1 | v6 | episode_012 / 1 | Success | [Video](../outputs/20261030-115639-8cb8b0/episode_000.mp4) |
| 2 | v6 | episode_019 / 1 | Success | [Video](../outputs/20261030-115701-1735b3/episode_000.mp4) |
| 3 | v6 | episode_020 / 1 | Success | [Video](../outputs/20261030-115725-c9a2de/episode_000.mp4) |
| 4 | v6 | episode_026 / 1 | Success | [Video](../outputs/20261030-115748-489635/episode_000.mp4) |
| 5 | v6 | episode_029 / 1 | Failure | [Video](../outputs/20261030-115824-59aa9d/episode_000.mp4) |
| 6 | candidate_damping2 | episode_012 / 1 | Success | [Video](../outputs/20261030-115930-fe9b3d/episode_000.mp4) |
| 7 | candidate_damping2 | episode_019 / 1 | Success | [Video](../outputs/20261030-115951-c97b31/episode_000.mp4) |
| 8 | candidate_damping2 | episode_020 / 1 | Success | [Video](../outputs/20261030-120013-2f2dde/episode_000.mp4) |
| 9 | candidate_damping2 | episode_026 / 1 | Success | [Video](../outputs/20261030-120036-7183a7/episode_000.mp4) |
| 10 | candidate_damping2 | episode_029 / 1 | Success | [Video](../outputs/20261030-120100-f48d1b/episode_000.mp4) |
| 11 | candidate_damping2 | episode_004 / 1 | Success | [Video](../outputs/20261030-120122-841de0/episode_000.mp4) |
| 12 | candidate_damping2 | episode_020 / 2 | Success | [Video](../outputs/20261030-120208-af9fc6/episode_000.mp4) |
| 13 | candidate_damping2 | episode_019 / 2 | Success | [Video](../outputs/20261030-120231-e05019/episode_000.mp4) |
| 14 | candidate_damping2 | episode_012 / 2 | Success | [Video](../outputs/20261030-120256-4441b7/episode_000.mp4) |
| 15 | candidate_damping2 | episode_004 / 2 | Success | [Video](../outputs/20261030-120317-cc0af6/episode_000.mp4) |
| 16 | candidate_damping2 | episode_029 / 2 | Success | [Video](../outputs/20261030-120337-09c52c/episode_000.mp4) |
| 17 | candidate_damping2 | episode_026 / 2 | Success | [Video](../outputs/20261030-120400-1e1ecf/episode_000.mp4) |
| 18 | v6 | episode_019 / 2 | Success | [Video](../outputs/20261030-120449-eb5f38/episode_000.mp4) |
| 19 | v6 | episode_012 / 2 | Success | [Video](../outputs/20261030-120511-e16e64/episode_000.mp4) |
| 20 | v6 | episode_004 / 2 | Success | [Video](../outputs/20261030-120533-eb02fe/episode_000.mp4) |
| 21 | v6 | episode_029 / 2 | Success | [Video](../outputs/20261030-120555-39263d/episode_000.mp4) |
| 22 | v6 | episode_026 / 2 | Failure | [Video](../outputs/20261030-120618-7338b2/episode_000.mp4) |
| 23 | v6 | episode_020 / 2 | Success | [Video](../outputs/20261030-120659-2906a6/episode_000.mp4) |
