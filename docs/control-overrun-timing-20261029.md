# Control-overrun timing experiment — 2026-10-29

## Scope and decision

This twelve-attempt selected-layout comparison held continued-control weights, H100, RTC20, kp200 and the original live 25 fps video path fixed. Both arms used the same tracing; only `recover_control_overruns` differed. Policy noise was unpaired. The decision retains the default false: one of eight live adoption gates failed, despite a smaller observed worst wall gap. No training, original-v6 quality improvement or general GUI timing claim follows.

The [operation guide](mac-motion-tuning.md#optional-post-apply-overrun-recovery-and-timing-traces) and [pure deadline helper](../src/pi0_repro/pacing.py) separate optional post-apply deadline recovery from normal clock behavior. No upstream, model, RTC, gains or video default changed.

## Native screen and declared amendment

The [native/frozen probe](../outputs/control_overrun_probe_20261029/summary.json) preserved all 1,108 full integration-state comparisons and six sampled RGB checks exactly; four frozen replays had recorded 14D state error zero. Injected overruns were 35/50/80 ms. Median observed saving was 9.47098 ms, failing the original ≥15 ms target. Next-loop preparation/rendering can consume part of nominal extra-idle removal.

The [conditional live-screen amendment](../outputs/control_wakeup_diagnosis_20261029/native_screen_amendment.json) was declared before live outcomes. It does not relabel the native screen as passed. The original [runner preflight failure](../outputs/control_wakeup_diagnosis_20261029/runner_preflight_attempt01.json) remains preserved: a copied guard-field mismatch stopped it locally before SSH or actions, not a model-task failure.

## Complete live results

The [plan](../outputs/control_overrun_ab_20261029/plan.json), [ledger](../outputs/control_overrun_ab_20261029/comparison.json), [aggregate](../outputs/control_overrun_ab_20261029/aggregate.json) and [audit](../outputs/control_overrun_ab_20261029/audit.json) retain six attempts per arm. All 407 checks passed and final metadata verified v6 restoration. Both arms completed four of six attempts; failed coverage remains separate from successful transport.

| Measure | Baseline | Overrun recovery |
| --- | --- | --- |
| Success / coverage | 4/6 | 4/6 |
| Median trial apply p99 (ms) | 20.9403 | 20.9008 |
| Worst apply gap (ms) | 87.5622 | 38.2930 |
| Gaps above 50 ms | 1 | 0 |
| Covered median active jerk p99 (rad/s³) | 80.6643 | 91.6057 |
| All-attempt median active jerk p99 (rad/s³) | 93.116 | 89.517 |
| Worst physical jerk peak (rad/s³) | 1,217.365 | 1,035.766 |
| Covered median duration (simulation s) | 22.33 | 22.08 |

All attempts had zero running inference underruns. Covered typical jerk increased 13.56%, exceeding the 1.1 ratio limit: this was the only failed live gate. All-attempt and covered statistics cannot be interchanged to avoid that failure. These small, unpaired samples do not justify adoption or establish consistently smoother physical motion.

## What the recorded envelopes establish

The [phase diagnosis](../outputs/control_overrun_ab_20261029/phase_diagnosis.json) retains all stage and GC envelopes. Baseline's 84.432 ms camera wall span (4.1205 ms calling-thread CPU, no same-step GC) preceded application 718 and contributed to the worst interval 717→718, 87.562 ms. Candidate control row 188 contains sequential spans on opposite sides of application: 35.998 ms camera time before apply188 contributed to the worst interval 187→188 (38.293 ms); approximately 34.788 ms environment work after apply188 contributed to the next interval 188→189 (34.9524 ms). Its forward phase measured 23.055 ms wall, 0.248 ms thread CPU and 1.119 ms process CPU, without overlapping GC. The before/after spans must not be summed into the 38.293 ms worst interval and do not prove a unique OS, GPU, GIL or physics-solver cause.

Median per-trial oversleep p99 was approximately 1.0396/1.0368 ms. Trace collection counts 119/113 include startup and must not be labeled exclusively running GC. The maximum GC span, 13.0109 ms at baseline round 6 step 634, was distinct from the worst camera event at step 718.

The [targeted recovery receipt](../outputs/control_overrun_ab_20261029/targeted_recovery_receipt.json) shows zero baseline deadline overruns and one candidate overrun. After candidate step 188's 34.83096 ms post-apply span, the new policy added zero idle; the next apply gap was 34.9524 ms. Its local deadline counterfactual removes 20 ms from the legacy deadline under that recorded history. It is not a second policy trajectory or a causal explanation of the groups' different maxima, which involved different camera events. Normal anchored behavior remains unchanged, so the option does not guarantee every interval is at least 20 ms.

## Tests and media boundary

The [test receipt](../outputs/control_overrun_ab_20261029/tests.json) passed nine checks: four pacing, one exceptional GC-cleanup and four multiview integration checks. Hook cleanup and arithmetic correctness do not establish model quality or wall deadlines.

The [primary review receipt](../outputs/control_overrun_ab_20261029/primary_visual_review.json) binds the exact [runtime plot](../outputs/control_overrun_ab_20261029/plots/runtime_comparison.png), both all-six outcome sheets and the combined first frame showing all twelve labels. This is sampled static review, not complete continuous playback. Four current manifests mark pending false while preserving pre-review snapshots.

The [all-twelve video](../outputs/control_overrun_ab_20261029/media/all12_default_vs_recovery.mp4) was verified at 960×1200, 25 fps, 1,001 frames and 40.04 seconds. All outcomes remain visible with labeled terminal freezes at 1× simulation-time playback, not wall-clock timing. Separate [baseline grid](../outputs/control_overrun_ab_20261029/media/baseline/all6_same_layout.mp4) and [recovery grid](../outputs/control_overrun_ab_20261029/media/recover/all6_same_layout.mp4) remain available. Final verification confirmed 407 audit checks, nine tests, unchanged source hashes, v6 checkpoint/normalization restoration and closure of the owned local endpoint. The failed native threshold and failed live physical-motion gate both remain recorded; defaults and historical results are unchanged.
