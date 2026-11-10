# v6 progress-state recovery and timing diagnosis — 2026-10-28

## Evidence boundary

This record separates completed v6 training-layout diagnostics from reviewed recovery data, completed continuation, an interrupted evaluation and a separate complete retest. The early training-layout diagnostics are not held-out evaluation; the separate six-layout validation pilot is reported below. Test data were not used. Both continuation arms finished training. The separate complete validation retest did not satisfy the recovery candidate adoption gates; the original-pi0 v6 RTC20 default remains unchanged.

The frozen [plan](../outputs/v6_progress_20261028/plan.json) and [12-attempt ledger](../outputs/v6_progress_20261028/comparison.json) cover six geometrically spaced training parents, each run headless and with worker display. Selection used geometry rather than learner outcomes. These parents come from the 20 full demonstrations within v6's actual 22-segment training set; they do not add independent layouts. Policy noise was not paired. Workload bundles are not isolated causal treatments. Only headless recordings are possible recovery sources; worker runs diagnose runtime costs.

## Fresh wall/thread-CPU diagnosis

The [tail diagnosis](../outputs/v6_progress_20261028/tail_diagnosis.json) retains both running inference underruns:

| Apply gap (ms) | Current inference wait (ms) | Previous environment wall / thread CPU (ms) |
| --- | --- | --- |
| 1,986.781 | 1,984.688 | 1.826 / 1.797 |
| 881.403 | 878.005 | 3.152 / 3.149 |

The largest environment wall span was 16.796 ms with 2.854 ms of calling-thread CPU. The historical approximately 144 ms environment span did not recur here. These fresh measurements cannot retrospectively identify its cause. Wall-minus-thread-CPU includes waits, scheduling and other-thread work; it is not a unique diagnosis of networking or MuJoCo computation. Backend identity remained unchanged across this diagnostic cohort.

The slowest recorded request had 2,187.007 ms client RTT and 159.528 ms server inference time. The difference does not uniquely locate a queue or transport delay. In particular, upstream `server_timing.prev_total_ms` includes the preceding server loop's receive wait and is not current-request pure inference time.

## Transport instrumentation and completed probe

The [PNG sidecar](../server/image_transport_proxy.py) adds `pi0-proxy-timing-v1` timing envelopes without changing decoded pixels, state, RTC fields or returned policy actions. The [operational guide](smooth-execution.md#transport-timing-and-echo-diagnostics) defines nested timing and bounded echo semantics. Two proxy unit checks passed. Only the sidecar was restarted, with backup/rollback paths; the v6 backend was not stopped.

The initial temporary-forward probe was rejected by the SSH server's forwarding policy; the failed attempt remains preserved under [transport_probe](../outputs/v6_progress_20261028/transport_probe/). No key or SSH permissions were changed to bypass it. The subsequent [40-pair probe via the sidecar](../outputs/v6_progress_20261028/transport_probe_via_proxy/probe.json) completed:

| Timing envelope (ms) | Median | Maximum |
| --- | --- | --- |
| Synthetic echo | 77.127 | 478.266 |
| Model client RTT | 266.423 | 428.129 |
| Proxy handler | 178.162 | 202.532 |
| Proxy upstream round trip | 158.918 | 180.677 |
| Outside encoding and proxy | 70.405 | 221.795 |

Echo p95 was 175.407 ms. Echo does not call the model. Nested envelopes cannot be summed, and these later probes do not establish the cause of the earlier 2.187-second request.

## Reviewed recovery data

The [selection](../outputs/v6_progress_20261028/recovery_selection.json) limits each headless parent to at most one near-grasp and one held-near-box state, distinguishing delay triggers from reference coverage. The [initial eleven candidates](../outputs/v6_progress_20261028/recovery_candidates.json) yielded ten automatic successes and one failure (index 4, near-grasp delay, descend target not reached). The primary inspected all four review sheets: [explicit decisions](../outputs/v6_progress_20261028/recovery_review/decisions.json) accepted ten and rejected that failure. Automatic success alone was not acceptance.

A separately declared [reapproach amendment](../outputs/v6_progress_20261028/reapproach_amendment.json) used the existing full-reopen mode for one successful attempt, followed by primary visual review and collection-local `progress_review.json` acceptance. The rejected original remains preserved; the amendment does not replace its history. Initial takeover checks included zero replay-state error and approximately 0.001045/0.000204 rad command seams for the first two candidates.

The [data preparation record](../outputs/v6_progress_20261028/data_preparation.json) binds eleven accepted new dependent segments to six existing parents: zero new independent layouts. The merged export contains 33 training segments / 37,144 frames versus the base 22 / 26,497. Local validation (6 / 7,423) and test (6 / 7,483) exports passed pixel checks but were not used for policy training. Expert takeover supplies offline labels, never a runtime policy fallback.

## Guarded train-only deployment — training completed

The original [v1 preparation snapshot](../outputs/v6_progress_20261028/training_plan.json) is retained. Its first deployment stopped at preflight because cloud held-out lineage clear lists were missing. A proposed transfer of raw held-out JSON was rejected by automatic approval review and did not execute. The active [train-only v2 plan](../outputs/v6_progress_20261028/training_plan_train_only.json) is the safe alternative: only training repositories are uploaded. Original held-out lists and images remain on the Mac; local tuple-lineage comparison sends only digests/counts, not an independently inspected cloud held-out dataset. This is a validation boundary, not a claim of hard filesystem isolation. Earlier sources remain archived under [pre_train_only_sources](../outputs/v6_progress_20261028/pre_train_only_sources/), and failed deployment records remain preserved. A subsequent approval-quota rejection did not launch training.

The [successful dispatch](../outputs/v6_progress_20261028/training_dispatch.json) and [status snapshot](../outputs/v6_progress_20261028/cloud_status_latest.json) establish `both_trained_pending_validation`: both arms completed 1,000 optimizer updates and committed final checkpoint `999`. The [training completion receipt](../outputs/v6_progress_20261028/training_complete_receipt.json) records 1,895.274 seconds of supervisor elapsed time. Real-reader H100 boundary checks passed for 66 control and 99 recovery samples. Both normalization compatibility audits passed with the fixed normalization digest, normalized action/state absolute maxima approximately 3.6983 / 3.2011, and zero fractions above ten. Capacity guards passed on both checkpoint filesystems, each requiring 13.330 GB peak reserve. No weights were deleted by this preparation/deployment.

Both real GPU probes passed with shape `[16, 100, 32]`; peak usage was 16,877 MiB for control and 16,881 MiB for recovery, out of 24,564 MiB. Both full runs used microbatch 8 × accumulation 2 = effective batch 16, the same weighted flow-matching objective, no added temporal loss, and matched ROI, prompt, normalization, seed 54 and learning rate. The intervention bundles new recovery data with a 20% parent-balanced sampler; it cannot isolate their individual effects.

An early status-reader response unintentionally included a `DataConfig`/`NormStats` log fragment. This known disclosure boundary is retained without reproducing vectors. The status reader now positively selects scalar progress/errors and compatibility aggregates, clears the old log tail, and passed its redaction test. The normalization asset itself remains fixed and unchanged; this record does not claim that no normalization vector ever left the cloud.

The [job supervisor](../server/progress_train_job.py) validates immutable plan/source/data/normalization bindings, local held-out lineage digest attestations and storage before GPU work. Two training-guard unit checks passed, including invalid normalization/source/disk and unbound-parquet cases. The [policy selector](../server/progress_policy_select.py) admits only registered committed checkpoints, rejects active training or unexpected policy ownership, and records dispatch separately from metadata verification. Neither helper establishes policy quality or current availability.

## Interrupted evaluation — no model ranking

The frozen [evaluator](../outputs/v6_progress_20261028/evaluation/runner_source.py) started the planned 18 attempts: six validation layouts × three models, one attempt each. Severe transport delays led to an intentional SIGINT, preserving the [incomplete ledger](../outputs/v6_progress_20261028/evaluation/comparison.json) with `complete=false`. The first six v6 attempts finished (five successes, one failure); index 6, the first control attempt, was interrupted. This is insufficient for a model comparison or improvement claim. The [interruption receipt](../outputs/v6_progress_20261028/evaluation_transport_interruption.json) binds the retained ledger and confirms subsequent restoration metadata for v6/7999 with the fixed normalization asset.

The observed route interface was `utun4`. In the last completed v6 case, 22.18 simulation seconds took 116.306 control-wall seconds, including 88.493 seconds of inference-underrun waiting. One request had 4,568.205 ms RTT, 150.189 ms server inference and 174.536 ms proxy-handler time. The outside residual cannot precisely identify a network component or establish that the virtual interface alone caused the delays. The user subsequently confirmed TUN was disabled; the separate retest below preserves this interrupted cohort without pooling it.

## Separate network-qualified retest — complete

The [network preflight](../scripts/preflight_v6_progress_network.py) requires a non-tunnel route and 20 no-environment-step echo/model pairs, with model RTT p95 ≤1,000 ms / maximum ≤1,500 ms and echo p95 ≤500 ms / maximum ≤1,000 ms. The [gate receipt](../outputs/v6_progress_20261028/network_retest_preflight/gate.json) passed on unchanged `en0` routing: echo median/p95/maximum 111.176/208.683/223.806 ms and model RTT 268.519/543.453/711.235 ms. Passing qualifies a run; it does not guarantee future latency or identify the interrupted cohort's precise delay cause.

The independent [retest plan](../outputs/v6_progress_20261028/evaluation_network_retest/plan.json) and [complete ledger](../outputs/v6_progress_20261028/evaluation_network_retest/comparison.json) cover all 18 attempts, with one attempt per model per six validation layouts. Policy noise was not paired. The [audit](../outputs/v6_progress_20261028/evaluation_network_retest/audit.json) passed all 404 checks. The ledger confirms restoration to v6/7999 with the fixed normalization asset. The original incomplete comparison and frozen runner remain preserved, not pooled or outcome-selectively retried.

The [aggregate](../outputs/v6_progress_20261028/evaluation_network_retest/aggregate.json) separates progress coverage from physical motion. Active jerk is the per-frame maximum across arm axes; p99 values below are medians of trial p99s, not a pooled derivative distribution.

| Measure | v6 | Continued control | Recovery bundle |
| --- | --- | --- | --- |
| Successes / attempts (secondary) | 6/6 | 5/6 | 5/6 |
| Transport-stage coverage / attempts | 6/6 | 5/6 | 5/6 |
| Median active jerk p99 (rad/s³), all attempts | 139.0625 | 96.9559 | 96.8490 |
| Median active jerk p99, covered attempts | 139.0625 | 102.6989 | 100.8877 |
| Worst physical jerk peak (rad/s³) | 2,674.2467 | 724.4647 | 1,110.8102 |
| Worst physical joint increment (rad) | 0.0220738 | 0.0220690 | 0.0220242 |
| Longest thresholded low-motion dwell ≥0.5 s (s) | 0 | 0.64 | 0 |
| Worst apply interval (ms) | 72.059 | 32.801 | 34.273 |

All eighteen attempts had zero running inference underruns and zero inference-wait time. Zero thresholded low-motion dwell does not establish prompt task progress: moving slowly or repeatedly without completing a phase need not satisfy the stillness threshold.

## Decision and reviewed phase evidence

The aggregate decision is `retain_v6_reference`. Recovery's typical active jerk was approximately 30.4% below raw v6, but almost identical to continued control. It lost coverage relative to v6; relative to control it failed the required ≥10% typical-jerk improvement and its worst peak increased approximately 53.3%. These small, unpaired samples do not support a recovery-data benefit or a reliability claim. Neither continuation is promoted, and the v6 RTC20 default remains unchanged.

The primary inspected all six all-outcome event pages and all six stall pages, recorded in [primary visual review](../outputs/v6_progress_20261028/evaluation_network_retest/primary_visual_review.json). The [media manifest](../outputs/v6_progress_20261028/evaluation_network_retest/media/manifest.json) and [stall review](../outputs/v6_progress_20261028/evaluation_network_retest/stall_review/review.json) now link that receipt with review pending false; hash-verified pre-review manifests remain preserved. This is sampled static review, not complete continuous video viewing.

The [three-model episode004 video](../outputs/v6_progress_20261028/evaluation_network_retest/media/episode_004_three_models.mp4) uses the first preassigned layout rather than selecting successes. It plays at 25 fps, 1× simulation time, with terminal-frame freezing explicitly labeled. Control episode004 had 33.04 seconds of no-lift exposure, censored at termination. Recovery episode026 lifted at 12.8 seconds but had not entered the box at 40 seconds: its 27.2-second post-lift exposure includes movement and must not be called entirely stationary. Recovery episode020 succeeded, yet lift-to-entry took 23.06 seconds versus 10.06 for this cohort's v6. Thus lower typical jerk and zero low-motion dwell do not resolve grasp/placement delay.

## Next steps — proposal only

Repeat validation before any adoption decision, and inspect training-only transport trajectories and expert labels for the remaining phase delays. Held-out failure states must not become training corrections. Do not automatically add training updates or combine new loss/batch/update changes without a separately reviewed plan. No additional collection or training is implied by this proposal. Test data were not used for policy training or this evaluation; historical reports, failures and the transport-interrupted cohort remain intact.
