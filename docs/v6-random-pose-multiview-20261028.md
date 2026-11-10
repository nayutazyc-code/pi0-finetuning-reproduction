# Original-pi0 v6 bounded random pose and multi-view evaluation — 2026-10-28

## Outcome and scope

V6 RTC20 completed 11 of 12 predeclared fresh banana XY/yaw combinations in this bounded simulation suite. One attempt never achieved a confirmed lifted pinch. This is descriptive capability evidence, not a 91.7% population guarantee, arbitrary-pose generalization, or improvement over the earlier fixed-layout 13/18 comparison. No checkpoint, normalization, controller, training, or model-camera change was made.

The [combined plan](../outputs/v6_random_multiview_complete_20261028/plan.json), [ledger](../outputs/v6_random_multiview_complete_20261028/comparison.json), [aggregate](../outputs/v6_random_multiview_complete_20261028/aggregate.json), and [446-check audit](../outputs/v6_random_multiview_complete_20261028/audit.json) preserve every sample, actual settled pose, outcome, warning, and recording. All audit checks passed. The [operational guide](random-pose-multiview.md) describes reproducible resets, display-only cameras, and separate recordings.

## Range and two-phase protocol

Each fixed independent seed samples banana X uniformly in 0.16–0.20 m, Y in 0.46–0.50 m, and yaw in 0–20°. Initial Z is 0.045 m before free settling; box position stays `(0.13, 0.72, 0)` m. These are fresh combinations inside the original acquisition envelope, not whole-table placements or arbitrary height/roll/pitch. Realized sampled and settled states are retained separately. Each seed had one attempt with no outcome-based resampling. Policy noise and network histories are unpaired.

The [initial phase](../outputs/v6_random_multiview_20261028/comparison.json) completed four physically successful attempts. Its fourth attempt incurred one running inference underrun and approximately 1.121 s wait; the original strict timing wrapper then aborted. That task outcome and infrastructure warning remain retained rather than being relabeled or rerun. A separately declared [continuation](../outputs/v6_random_multiview_continuation_20261028/comparison.json) ran only the remaining eight disjoint seeds, retaining transport warnings, with six additional no-step preflight requests recorded separately from task counts. Thus the combined suite is two-phase descriptive capability evidence, not an uninterrupted timing benchmark.

## All outcomes

| Case | Seed | Task outcome | Simulated duration, s | Running inference underruns |
| --- | --- | --- | --- | --- |
| 1 | 2026102801 | Success | 21.56 | 0 |
| 2 | 2026102802 | Success | 34.36 | 0 |
| 3 | 2026102803 | Success | 21.02 | 0 |
| 4 | 2026102804 | Success; original timing wrapper aborted afterward | 22.94 | 1 |
| 5 | 2026102805 | Failure; no confirmed lifted pinch | 40.00 | 0 |
| 6 | 2026102806 | Success | 21.92 | 0 |
| 7 | 2026102807 | Success | 23.82 | 0 |
| 8 | 2026102808 | Success | 23.02 | 0 |
| 9 | 2026102809 | Success | 21.84 | 0 |
| 10 | 2026102810 | Success | 22.52 | 0 |
| 11 | 2026102811 | Success | 31.98 | 0 |
| 12 | 2026102812 | Success | 21.76 | 0 |

The failure is zero-based ledger index 4. Its near-to-pinch interval is undefined, not zero; the episode had 44 right-gripper command reversals under the existing excursion-threshold diagnostic. This describes repeated commanded opening/closure changes without proving why the grasp failed. Successful episodes can still hesitate: case 2 took 15.68 simulated seconds from near-region entry to confirmed lifted pinch, so completion does not establish prompt grasp progression or uniformly smooth movement.

## Display isolation, timing, and limits

The original policy input remains raw top-camera RGB 640×480 mapped to `cam_high` with unchanged cloud ROI. Front/side and the physically attached right-wrist camera render only in the read-only spawned display clone; no display pixels enter pi0. Canonical top videos remain separate from multi-view videos and source-step receipts. The clone disables its own shadows without changing main policy lighting or physics.

Across the 12 recordings, pooled action-application intervals had median 20.001 ms, p95 21.017 ms, and maximum 1123.186 ms. The single inference wait is retained. Even among the 11 attempts without an inference underrun, the largest local interval was 404.051 ms, with a recorded long step including approximately 142 ms environment/physics, 42 ms camera work, and 171 ms video submission containing approximately 85 ms exchange. These stage timings are inclusive/nested, not exclusive slices or proof of one cause. Zero inference underruns would not imply zero local pauses or guaranteed 50 Hz wall-clock control.

Display source age had median 2, p95 4, and maximum 8 control steps: 40/80/160 ms in simulated-step equivalents. Composed frames may repeat older full snapshots. `frame_age_wall_seconds` measures time since publication, not complete capture-to-display latency. The 25 fps interface/video rate is a sampling target, not an achieved zero-delay presentation guarantee. Video playback follows simulation time and does not reproduce wall-clock waits faithfully.

## Media inspection and helper smoke test

The [media manifest](../outputs/v6_random_multiview_complete_20261028/media/manifest.json) retains source bindings and both all-outcome sheets covering all 12 endings. The primary assistant inspected both overviews, the [first lifted view](../outputs/v6_random_multiview_complete_20261028/media/pose_01_lifted.png), and the [fifth case's final view](../outputs/v6_random_multiview_complete_20261028/media/pose_05_final.png). The wrist thumbnail shows the actual gripper-mounted camera view with banana and fingers visible; it is not a fabricated model observation. This sampled inspection does not mean every full video was continuously watched or establish a physical failure cause.

The [visual-review receipt](../outputs/v6_random_multiview_complete_20261028/media/visual_review.json) records the subsequent sampled inspection; the manifest's `manual_review_pending` field describes its generation-time status rather than overriding that later receipt.

The [first random-pose multi-view video](../outputs/v6_random_multiview_complete_20261028/media/first_random_pose_multiview.mp4) is a copy of the first predeclared case, not a success-selected replacement. An extra [helper smoke run](../outputs/20261028-142300-6b145f/summary.json) using seed 2026102801 succeeded in 20.92 simulated seconds and exited normally. It verifies the actual Mac/cloud helper entry and owned-tunnel cleanup path; it is outside the 12-case cohort and is not added to its outcome denominator.

The [helper-smoke receipt](../outputs/v6_random_multiview_complete_20261028/helper_smoke.json) confirms the owned local endpoint refused connections after exit and all 12 recorded source-file SHA values matched the source at verification time.

The bounded-range observations support trying this opt-in presentation/reset workflow while retaining all failures and timing warnings. They do not establish a superior policy, general success probability, arbitrary 3D generalization, physical-robot transfer, or absence of the user's perceived jumps/pauses.
