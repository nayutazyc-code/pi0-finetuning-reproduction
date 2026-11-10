# v6 versus continued control: fixed-layout trajectory jumps — 2026-10-28

## Scope and provenance

This sequential comparison adds ten original-v6 attempts to the [existing ten control attempts](v6-control-fixed-layout-repeats-20261028.md), preserving both cohorts. All use validation `episode_004`, reset seed `20261027`, the same scene, source and RTC20/client bridge20 execution settings. Control ran first, v6 later: the blocks were not counterbalanced and stochastic policy noise was unpaired. Time-of-day/runtime differences prevent a causal interpretation of wall-timing differences. This deliberately selected difficult layout is not an overall reliability estimate.

The new [v6 plan](../outputs/v6_baseline_repeats_20261028/plan.json), [ledger](../outputs/v6_baseline_repeats_20261028/comparison.json), [aggregate](../outputs/v6_baseline_repeats_20261028/aggregate.json) and [audit](../outputs/v6_baseline_repeats_20261028/audit.json) retain all ten attempts, with 252 audit checks passed. The joint [comparison](../outputs/v6_fixed_layout_jump_comparison_20261028/comparison.json) and [71-check audit](../outputs/v6_fixed_layout_jump_comparison_20261028/audit.json) verify scene/seed/configuration/source compatibility. No old record was rewritten, no model retrained and no default changed. Test data were not used; validation states do not become training data.

Implementation entry points are the [v6 runner](../scripts/repeat_v6_baseline_layout.py), [v6 analyzer](../scripts/analyze_v6_baseline_repeats.py), [offline jump metrics](../scripts/trajectory_jump_metrics.py) and [comparison builder](../scripts/compare_fixed_layout_trajectories.py). Two jump-metric unit checks passed; the diagnostics do not change execution.

## Physical motion and completion

v6 completed nine of ten attempts; round 5 had no confirmed lift by timeout. Control completed seven of ten, with no-lift rounds 3, 4 and 6 retained. These observed counts cannot establish one model's general success probability or superiority. All twenty attempts had zero running inference underruns and zero inference-wait time.

| Measure | v6 | Control | Control relative to v6 |
| --- | --- | --- | --- |
| Covered attempts / all attempts | 9/10 | 7/10 | Different motion coverage |
| Median active jerk p99, covered attempts (rad/s³) | 128.3117 | 106.2741 | −17.18% |
| Median active jerk p99, all attempts (rad/s³) | 117.712 | 81.694 | −30.60% |
| Worst physical jerk peak (rad/s³) | 1,531.586 | 2,710.678 | +76.99% |
| Median physical increment p99, covered attempts (rad) | 0.0189187 | 0.0194899 | +3.02% |
| Worst physical increment (rad) | 0.0220267 | 0.0217044 | −1.46% |
| Worst wall apply gap (ms) | 55.872 | 96.268 | Sequential timing, not isolated weight effect |

Typical p99 values are medians of per-trial p99s. Active jerk uses the frozen per-frame maximum across arm axes; physical increments use 20 ms pre-step joint samples. Incomplete no-lift trajectories do not cover equivalent transport motion, so all-attempt and covered-only metrics must not be mixed. The covered physical increment p99 did not improve, despite lower typical jerk. The largest jerk peaks were both on right `wrist_angle` (joint 11) during near grasp: v6 round 9, step 397, and control round 8, step 419. Their nearest accepted chunk switches were 3 and 13 steps away respectively. Temporal proximity does not establish a chunk-switch cause.

## Offline chunk-boundary diagnosis

| Covered-attempt median p99 (rad) | v6 | Control | Relative change |
| --- | --- | --- | --- |
| Same-timestamp raw prediction disagreement | 0.0751358 | 0.0848289 | +12.90% |
| Executed command increment at switches | 0.0187489 | 0.0176663 | −5.77% |

Raw disagreement compares old/new arm predictions at the same valid timestamp and excludes expired old-plan overlap. Executed switch increments compare adjacent timestamps and include normal intended motion; they are not the same quantity as same-time prediction disagreement or physical state jumps. Grippers are excluded. The lower executed switch increment does not show that raw predictions became more continuous: raw disagreement increased. Neither statistic identifies the cause of physical wrist peaks.

## Media review boundary

The [distribution figure](../outputs/v6_fixed_layout_jump_comparison_20261028/plots/jump_distributions.png) retains all twenty trial points; the [plot manifest](../outputs/v6_fixed_layout_jump_comparison_20261028/plots/manifest.json) binds those data and independent peak selections. The [worst-wrist trajectories](../outputs/v6_fixed_layout_jump_comparison_20261028/plots/worst_wrist_trajectory.png) show each cohort's separately selected extreme event with coordinates and units, not a paired comparison. Within their local ±0.5-second windows, maximum right-wrist 20 ms increments were 0.449867° for v6 round 9 and 0.687602° for control round 8. These illustrative local amplitudes are not whole-cohort metrics and do not identify contact, control or chunk-switch causality.

The [all-twenty video](../outputs/v6_fixed_layout_jump_comparison_20261028/media/all20_v6_vs_continued.mp4) and [media manifest](../outputs/v6_fixed_layout_jump_comparison_20261028/media/manifest.json) retain every outcome: v6's ten attempts above control's ten, each block arranged rounds 1–5 above 6–10. Decoding verified 1600×1200, 25 fps, 1,001 frames and 40.04 seconds. Playback is 1× simulation time rather than wall time; shorter attempts have labeled terminal-frame freezing. Terminal-freeze onset follows the sampled frame timeline, and encoded duration includes the final 25 fps frame.

The [primary review receipt](../outputs/v6_fixed_layout_jump_comparison_20261028/primary_visual_review.json) records two new v6 outcome pages, both plots and the grid's first frame for label readability. Control's ten outcomes link the [earlier review receipt](../outputs/v6_control_repeats_20261028/primary_visual_review.json). This is sampled static review, not complete continuous playback. Current plot/media/baseline-media manifests mark review pending false; pre-review snapshots remain preserved. Final v6 checkpoint and normalization metadata were reverified.

Original v6 videos: [Round 1](../outputs/20261028-213602-baf100/episode_000.mp4), [Round 2](../outputs/20261028-213625-062909/episode_000.mp4), [Round 3](../outputs/20261028-213648-105398/episode_000.mp4), [Round 4](../outputs/20261028-213712-ab4ddb/episode_000.mp4), [Round 5](../outputs/20261028-213734-4d8f86/episode_000.mp4), [Round 6](../outputs/20261028-213815-ff5a1c/episode_000.mp4), [Round 7](../outputs/20261028-213838-9a2c6d/episode_000.mp4), [Round 8](../outputs/20261028-213859-61d9ee/episode_000.mp4), [Round 9](../outputs/20261028-213922-b9dac9/episode_000.mp4), [Round 10](../outputs/20261028-213944-4a670a/episode_000.mp4).

## Decision

Typical covered jerk was lower for control, but its worst peak was higher and its typical physical displacement increment did not improve. The screen's typical-jerk gate passed while its worst-peak/increment gate failed. These observations do not demonstrate stable suppression of jumps or universally smaller jump amplitude. Control may remain a separate experimental candidate for user observation; there is no formal promotion and v6 RTC20 remains the reference. Earlier six-layout cohorts are not pooled with these selected-layout sequential blocks.
