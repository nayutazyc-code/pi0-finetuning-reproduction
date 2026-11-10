# Deferred top-video timing comparison — 2026-10-29

## Scope and decision

The optional recorder removed most running-loop top-video rendering, but the twelve-attempt closed-loop comparison failed five of seven predeclared gates. The decision is `retain_existing_default`: deferred rendering remains off by default. This is a recording/runtime experiment with continued-control weights, not evidence of improved original-v6 quality, GUI smoothness or general task reliability.

The [operation guide](mac-motion-tuning.md#optional-deferred-top-video-rendering) and [recorder](../src/pi0_repro/deferred_video.py) describe bounded integration snapshots, fresh live policy pixels and post-control clone rendering. Model weights, H100, RTC20 and kp200 were fixed; the headless fixed-layout AB comparison had six fresh attempts per arm with unpaired policy noise. Earlier camera-reuse bundles and wrist/controller cohorts are separate evidence and are not pooled.

## Camera-stage diagnosis

The completed [frozen camera-stage replay](../outputs/camera_stage_diagnosis_20261029/summary.json) used four legacy/persistent/persistent/legacy passes, each with 588 renders and 23 sampled RGB checks. All recorded 14-dimensional states replayed exactly; sampled pixels matched and rendering preserved physics state/control/time. These checks do not establish equality of unrecorded full physical state or closed-loop outcomes. Typical calls were approximately 8 ms legacy and 6.6 ms persistent. One approximately 31 ms call overlapped about 19 ms of garbage collection; that association does not uniquely explain every long call. The earlier approximately 173 ms camera span did not recur. GL submission time is not total GPU time, and this replay cannot retrospectively explain the earlier outlier.

## Snapshot and video fidelity

The [snapshot audit](../outputs/deferred_video_ab_20261029/snapshot_audit.json) passed for six candidate records: captured pre-action 14D state errors were zero and 72 sampled raw RGB comparisons matched. A separate [full replay](../outputs/deferred_video_full_replay_20261029/summary.json) matched all 555 video frames in raw RGB and recorded 14D state with zero error. Raw-pixel equivalence does not make encoded MP4 lossless or guarantee identical closed-loop stochastic trajectories.

Captured snapshots are persisted before encoding. Running capture cost and post-control rendering/encoding remain separate measurements, and total task/job cost includes both. Candidate allocation was 2,154,152 bytes, with median post-control rendering/encoding 4.293 seconds. This transfers work rather than eliminating it.

## Complete closed-loop results

The [plan](../outputs/deferred_video_ab_20261029/plan.json), [ledger](../outputs/deferred_video_ab_20261029/comparison.json), [aggregate](../outputs/deferred_video_ab_20261029/aggregate.json) and [audit](../outputs/deferred_video_ab_20261029/audit.json) retain all twelve trials; 349 checks passed with zero audit failures. Final metadata verified restoration to v6. Startup attempts 01/02 stopped with zero actions because the sidecar endpoint was missing; their records and the attempt02 proxy recovery remain preserved. They are infrastructure attempts, not model-task failures.

| Measure | Baseline live top video | Deferred top video |
| --- | --- | --- |
| Success / coverage | 6/6 | 4/6 |
| Median trial apply p99 (ms) | 20.9149 | 22.5753 |
| Worst apply interval (ms) | 43.5519 | 53.5832 |
| Intervals over 50 ms | 0 | 1 |
| Covered median physical active jerk p99 (rad/s³) | 87.5664 | 200.9438 |
| Median running camera time per trial (s) | 4.1162 | 0.7576 |
| Median running camera render count | 621 | 78.5 |
| Covered median duration (simulation s) | 23.28 | 22.50 |

All trials had zero running inference underruns and zero inference waiting. Reduced camera work did not produce better measured tail timing or covered physical jerk in this sample. Coverage differs, so incomplete attempts must not be treated as smoother successful transport. The failed gates were median apply p99, worst apply interval, over-50-ms rate, covered jerk and coverage; covered duration and zero-underrun gates passed. The gates are not revised after observing results. Neither all physical spikes nor local timing stalls are resolved by moving recording work.

## Remaining wall-gap envelopes

The [wall-gap diagnosis](../outputs/deferred_video_ab_20261029/wall_gap_diagnosis.json) locates baseline's 43.5519 ms maximum at round 6, step 1051: current camera and inference-blocking time were zero, pacing waited 42.0185 ms with 24.5718 ms lateness, and previous environment wall/thread CPU were 1.4268/1.4247 ms. Deferred's 53.5832 ms maximum, round 6 step 222, likewise had zero current camera/inference blocking; previous environment wall/thread CPU were 32.518/1.7792 ms and current pacing waited 20.944 ms. The remaining maxima fall in wake/pacing and environment-wall envelopes rather than current-frame camera or inference blocking. This does not uniquely identify OS/kernel/CPU causes; live garbage collection was not captured. Typical request-receipt offsets were mostly 12 control steps for baseline and 13 for deferred, an unpaired observation rather than proof of a latency effect.

## Tests, retained issues and media boundary

The [initial test receipt](../outputs/deferred_video_ab_20261029/unit_test_attempt01.json) retains a test-only `MemoryError` caused by `list(imageio_reader)` using an effectively unbounded length hint. The test iteration was fixed; the [final test receipt](../outputs/deferred_video_ab_20261029/unit_test_final.json) passed. This is not a runtime encoder or model failure. The final checks passed four deferred native/config tests and four multiview integration tests, including worker/deferred isolation. Installed-imageio resource warnings remain in the test receipt; they are not a runtime feature failure. These checks remain distinct from actual-model outcome evidence.

The [primary review receipt](../outputs/deferred_video_ab_20261029/primary_visual_review.json) binds the exact corrected [runtime plot](../outputs/deferred_video_ab_20261029/plots/runtime_comparison.png), both all-six outcome sheets covering all twelve attempts, and the combined first frame for labels. This is sampled static review, not complete continuous playback or evidence of user-perceived improvement. Four current manifests mark review pending false and preserve their pre-review snapshots.

The [all-twelve comparison video](../outputs/deferred_video_ab_20261029/media/all12_live_vs_deferred.mp4) was verified at 960×1200, 25 fps, 1,001 frames and 40.04 seconds, retaining all outcomes with labeled terminal freezes and 1× simulation-time playback. Separate [baseline six-attempt grid](../outputs/deferred_video_ab_20261029/media/baseline/all6_same_layout.mp4) and [deferred six-attempt grid](../outputs/deferred_video_ab_20261029/media/deferred/all6_same_layout.mp4) are bound by their manifests. Terminal-freeze onset follows sampled-frame time; encoded duration includes its final 25 fps frame and is not control-wall duration. Defaults, weights, success rules and old records remain unchanged.
