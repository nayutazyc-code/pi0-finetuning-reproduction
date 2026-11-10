# Latest-candidate OOD evaluation — 2026-10-29

## Frozen scope — complete

The [frozen plan](../outputs/ood50_20261029/plan.json) specifies fifty unique seed/layout cases for the latest original-pi0 candidate checkpoint `999`, H100, denoise10, RTC20/client bridge20 and wrist damping2. Each case receives one attempt with a 40-second simulation-time cap; cases are randomly interleaved. All fifty attempts completed: 31 successes (62%) and 19 failures within this deliberately balanced suite. This is a descriptive suite fraction, not a population OOD success-rate estimate.

OOD is defined relative to the local task fine-tuning distribution, not guaranteed novelty relative to pretrained model data. Policy top camera and ROI remain unchanged. The box stays fixed at (0.13, 0.72, 0) m. No training, success-threshold changes, prior held-out-layout reuse or pooling with historical in-distribution outcomes is part of this evaluation.

| Group | Cases | Frozen shift |
| --- | --- | --- |
| Position | 20 | Sample outside one side of the original x=[0.16,0.20], y=[0.46,0.50] m rectangle: x=[0.12,0.15] or [0.21,0.24], or y=[0.42,0.45] or [0.51,0.54]; other dimensions remain in distribution |
| Yaw | 20 | Yaw in [−60,−20]° or [40,80]°, with XY inside the original rectangle |
| Appearance | 10 | New blue/red palettes combined with lighting scale 0.65 or 1.4 |

Appearance combines palette and lighting changes; this group cannot isolate their individual causal effects. Position/yaw shifts are bounded task perturbations, not arbitrary 3D-pose or real-robot generalization tests. All unsuccessful cases, timeouts, transport waits and operational interruptions must remain in the ledger; task outcomes and wall-clock timing are separate evidence.

## Complete outcomes and audit

The [complete ledger](../outputs/ood50_20261029/comparison.json), [aggregate](../outputs/ood50_20261029/aggregate.json) and [audit](../outputs/ood50_20261029/audit.json) preserve every attempt. All 1,849 checks passed, including actual settled OOD/reset conditions, model/source pins, prefixes and timestamps. End-of-run metadata verified restoration to original v6.

| Group | Successes / attempts | Observed fraction | Failures |
| --- | --- | --- | --- |
| Position | 10/20 | 50% | 10 |
| Yaw | 12/20 | 60% | 8 |
| Appearance | 9/10 | 90% | 1 |
| Total | 31/50 | 62% | 19 |

All fifty attempts had zero running inference underruns and zero inference-wait time. This does not establish that local wall-clock stalls were eliminated or motion was uniformly smooth. Among nineteen failures, eighteen never confirmed lift; one had lifted, and none ended inside the box. These per-attempt stage flags describe observed progress, not why the policy failed. Failed cases remain in both counts and source-video provenance.

## Interpretation and media boundary

These are one-attempt stochastic observations on fifty predeclared scenes, with no historical in-distribution pooling or direct OOD control-model arm. The combined candidate includes damping2, so results cannot be attributed purely to model weights. The 90% appearance subset is especially small and combines palette/lighting factors. Outcomes do not establish universal OOD reliability, remove [earlier adoption-gate failures](release-retreat-optimization-20261029.md), or promote the candidate/default.

The [media manifest](../outputs/ood50_20261029/media/manifest.json) binds all fifty original-video hashes and five start/lift/final contact sheets. The [primary review receipt](../outputs/ood50_20261029/primary_review.json) records inspection of all five sheets. This is sampled static review, not frame-by-frame or continuous viewing of every video. Images and stage flags do not establish reachability or failure causality. The one failed yaw case with confirmed lift is case47; eighteen other failures never confirmed lift.

Original v6 remains the default. No training or threshold tuning followed these outcomes, and historical records remain unchanged.


## All fifty original videos

The table preserves ledger order, including every success and failure; case numbers are the zero-based ledger indices.

| Case | Group | Outcome | Original video |
| --- | --- | --- | --- |
| 0 | position | Failure; no confirmed lift | [Video](../outputs/20261029-183944-357cad/episode_000.mp4) |
| 1 | position | Success | [Video](../outputs/20261029-184027-b28ccf/episode_000.mp4) |
| 2 | position | Failure; no confirmed lift | [Video](../outputs/20261029-184051-d31bac/episode_000.mp4) |
| 3 | appearance | Failure; no confirmed lift | [Video](../outputs/20261029-184133-ea35b1/episode_000.mp4) |
| 4 | yaw | Success | [Video](../outputs/20261029-184215-e441d8/episode_000.mp4) |
| 5 | yaw | Success | [Video](../outputs/20261029-184235-819939/episode_000.mp4) |
| 6 | position | Success | [Video](../outputs/20261029-184258-99bf14/episode_000.mp4) |
| 7 | yaw | Success | [Video](../outputs/20261029-184322-a2f19a/episode_000.mp4) |
| 8 | position | Failure; no confirmed lift | [Video](../outputs/20261029-184346-13e014/episode_000.mp4) |
| 9 | yaw | Failure; no confirmed lift | [Video](../outputs/20261029-184428-08a70a/episode_000.mp4) |
| 10 | yaw | Failure; no confirmed lift | [Video](../outputs/20261029-184510-30989c/episode_000.mp4) |
| 11 | yaw | Success | [Video](../outputs/20261029-184553-487794/episode_000.mp4) |
| 12 | position | Failure; no confirmed lift | [Video](../outputs/20261029-184614-507688/episode_000.mp4) |
| 13 | position | Failure; no confirmed lift | [Video](../outputs/20261029-184657-fce52f/episode_000.mp4) |
| 14 | appearance | Success | [Video](../outputs/20261029-184738-36651f/episode_000.mp4) |
| 15 | yaw | Success | [Video](../outputs/20261029-184804-92ef0c/episode_000.mp4) |
| 16 | appearance | Success | [Video](../outputs/20261029-184825-caa766/episode_000.mp4) |
| 17 | position | Success | [Video](../outputs/20261029-184849-7c93d1/episode_000.mp4) |
| 18 | position | Success | [Video](../outputs/20261029-184911-92767d/episode_000.mp4) |
| 19 | position | Failure; no confirmed lift | [Video](../outputs/20261029-184932-b017c2/episode_000.mp4) |
| 20 | appearance | Success | [Video](../outputs/20261029-185013-d26e81/episode_000.mp4) |
| 21 | position | Success | [Video](../outputs/20261029-185039-b004bc/episode_000.mp4) |
| 22 | position | Failure; no confirmed lift | [Video](../outputs/20261029-185100-426a6b/episode_000.mp4) |
| 23 | yaw | Success | [Video](../outputs/20261029-185141-b30d47/episode_000.mp4) |
| 24 | yaw | Success | [Video](../outputs/20261029-185202-876e9d/episode_000.mp4) |
| 25 | position | Failure; no confirmed lift | [Video](../outputs/20261029-185223-2a8882/episode_000.mp4) |
| 26 | position | Failure; no confirmed lift | [Video](../outputs/20261029-185304-9a10f4/episode_000.mp4) |
| 27 | yaw | Success | [Video](../outputs/20261029-185345-f8b8cc/episode_000.mp4) |
| 28 | position | Success | [Video](../outputs/20261029-185409-8a13d8/episode_000.mp4) |
| 29 | appearance | Success | [Video](../outputs/20261029-185430-3006ea/episode_000.mp4) |
| 30 | yaw | Failure; no confirmed lift | [Video](../outputs/20261029-185452-42188b/episode_000.mp4) |
| 31 | position | Success | [Video](../outputs/20261029-185533-1e0327/episode_000.mp4) |
| 32 | position | Failure; no confirmed lift | [Video](../outputs/20261029-185557-6e46c9/episode_000.mp4) |
| 33 | appearance | Success | [Video](../outputs/20261029-185638-1e7ff7/episode_000.mp4) |
| 34 | position | Success | [Video](../outputs/20261029-185703-03cae5/episode_000.mp4) |
| 35 | appearance | Success | [Video](../outputs/20261029-185725-0ce160/episode_000.mp4) |
| 36 | yaw | Success | [Video](../outputs/20261029-185751-ec4ec1/episode_000.mp4) |
| 37 | appearance | Success | [Video](../outputs/20261029-185820-414459/episode_000.mp4) |
| 38 | appearance | Success | [Video](../outputs/20261029-185846-fec7e7/episode_000.mp4) |
| 39 | yaw | Failure; no confirmed lift | [Video](../outputs/20261029-185911-f7c941/episode_000.mp4) |
| 40 | yaw | Failure; no confirmed lift | [Video](../outputs/20261029-185952-ac68c5/episode_000.mp4) |
| 41 | yaw | Failure; no confirmed lift | [Video](../outputs/20261029-190033-037200/episode_000.mp4) |
| 42 | position | Success | [Video](../outputs/20261029-190114-5cb536/episode_000.mp4) |
| 43 | appearance | Success | [Video](../outputs/20261029-190135-96191a/episode_000.mp4) |
| 44 | position | Success | [Video](../outputs/20261029-190201-d635af/episode_000.mp4) |
| 45 | yaw | Success | [Video](../outputs/20261029-190230-02c1d6/episode_000.mp4) |
| 46 | yaw | Failure; no confirmed lift | [Video](../outputs/20261029-190255-108ded/episode_000.mp4) |
| 47 | yaw | Failure; lifted | [Video](../outputs/20261029-190337-5c494d/episode_000.mp4) |
| 48 | yaw | Success | [Video](../outputs/20261029-190418-78e47b/episode_000.mp4) |
| 49 | yaw | Success | [Video](../outputs/20261029-190446-9b9d1c/episode_000.mp4) |
