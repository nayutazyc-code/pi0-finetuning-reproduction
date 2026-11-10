# Original-pi0 v6 RTC taper: same-session comparison — 2026-10-28

## Decision and scope

Retain [v6 RTC20](../configs/banana_v6_bridge20_pi0.toml) as the motion baseline. The separate [RTC10 configuration](../configs/banana_v6_rtc10_pi0.toml) completed more tasks, had better observed coverage and a lower all-attempt worst dwell, but covered-transport worst dwell did not improve. Its typical active-motion jerk was worse in the new same-session comparison. It did not pass the predeclared motion-first selection gate and is not promoted. No training, checkpoint, normalization, plant, or execution-algorithm change occurred.

RTC10 changes only `rtc_blend_steps` from 20 to 10. This is the generation-time prefix-guidance taper, not request cadence, prediction horizon, or the client's unchanged 20-step quintic bridge. Both use original pi0 v6 `banana_v6_long100_accum8000/7999`, matching normalization, horizon 100, nominal request age 15, 3 Hz arm filtering, and the same limits/scene.

## Separate diagnostic review of the older 18 RTC10 recordings

The [old-tail numerical audit](../outputs/v6_rtc10_tail_review_20261028/summary.json) recomputed the original 18 recordings without changing their frozen metrics. The [visual review](../outputs/v6_rtc10_tail_review_20261028/visual_review.json) binds four selected full-frame/ROI contact sheets and aligned short clips. A worker and the primary assistant inspected all four sheets; clips were encoded/alignment-checked, without a claim of continuous playback review. These old recordings are diagnostic material, not additional trials pooled into the new comparison below.

The two largest physical peaks were 2043.63 and 1466.21 rad/s³ on the right forearm-roll axis, at 9.72 and 8.18 simulated seconds. Each occurred 0.02 s after recorded banana/robot contact ended, while the nearest future chunk switch was 0.10 and 0.04 s later. Their aligned preceding applied-target jerk was approximately -1.12 and -2.25 rad/s³; four-frame physical ranges were 1.524° and 1.030°, with maximum single-step changes 0.820° and 0.567°. The sheets showed local banana/gripper changes, without a large whole-arm posture flip. These observations distinguish physical response, command derivatives, and displacement amplitude; they do not establish the contact-release mechanism or exclude earlier blends.

A separate 1110.55 rad/s³ box-area peak occurred while banana/robot contact remained true, with its nearest contact change 0.50 s later. The worst-p99 recording had a different maximum, 628.94 rad/s³, after release inside the box; its 11 own-p99 exceedances were classified from task logs as six near grasp and five inside the box. Only its maximum-event sheet was visually reviewed. Its requested post-peak context was censored by the recording end. These cases do not support one common contact or new-chunk cause.

All selected old runs had zero prefix error, local inference blocking, and running underruns. Raw requested targets still varied in surrounding windows; smooth applied targets at a physical peak do not prove every policy output smooth. Force, torque, solver impulses, and complete non-banana contact logs were absent. Compressed 50 fps static views cannot establish subjective continuity or refute the user's observed pauses. All peaks, percentile scores, amplitudes, failures, and phase coverage remain retained. Review encoding ended before the new timed comparison began.

## New 36-attempt protocol and integrity

The [frozen plan](../outputs/v6_rtc_same_session_20261028/plan.json), [complete ledger](../outputs/v6_rtc_same_session_20261028/comparison.json), [aggregate](../outputs/v6_rtc_same_session_20261028/aggregate.json), and [audit](../outputs/v6_rtc_same_session_20261028/audit.json) record six preassigned validation layouts, three repeats per profile, and 36 attempts. Adjacent profile order alternates by layout/repeat. Reset scenes/seeds match, repeat warmup counts are 2/3/4, and `noise_paired = false`. No test layout is used. The same v6 server process identity was verified across both arms and at the end; nine execution-source files match the old RTC10 recordings, and the motion metric is separately hash-bound. Private connection details and process identifiers are unnecessary to interpret the comparison.

All 992 audit checks passed, with zero failed checks, running inference underruns, raw-prediction reconstruction errors, action-timestamp errors, and recomputed prefix errors. No infrastructure-failure trial occurred in the new 36. Same-session serving reduces one drift risk but does not pair model noise, latency histories, or physical contact trajectories. Matched resets are not paired stochastic experiments or proof of causal superiority.

Before scored trials, six no-environment-step preflight requests checked readiness; the final four round trips were approximately 234–244 ms after two graph warmups. Preflight requests did not contribute episode scores or task counts.

The motion-first measure takes per-frame maxima across 12 physical arm joints and reports active-motion jerk percentiles on simulation time. Group values below are medians of per-trial statistics. All-attempt and transport-covered results remain separate; no-motion distributions are unavailable rather than perfect scores, and missing stages never become zero-delay wins. The original dwell threshold of at least 0.5 s and supplementary 0.16 s dwell diagnostics are reported separately without changing the frozen metric. Required stable success holds are excluded. Low movement can still include deliberate holds or stalled failures.

## Recorded results

| Measure | RTC20 | RTC10 |
| --- | --- | --- |
| All-attempt median active jerk p99, rad/s³ | 106.772 | 117.251 |
| Transport-covered median active jerk p99, rad/s³ | 109.504 | 119.462 |
| Median per-trial active jerk peak, rad/s³ | 396.017 | 697.963 |
| Worst active jerk peak, rad/s³ | 1571.284 | 1508.416 |
| Worst physical joint increment, rad | 0.02204716 | 0.02205384 |
| Longest observed dwell of at least 0.5 s, s | 1.34 | 1.10 |
| Transport-covered longest dwell of at least 0.5 s, s | 0.64 | 0.74 |
| Transport-covered attempts | 14/18 | 17/18 |
| Covered median episode duration, simulated s | 24.62 | 22.76 |
| Task completions, secondary | 13/18 | 17/18 |

RTC10's all-attempt typical p99 was 9.81% higher; its transport-covered typical p99 was 9.09% higher. A slightly smaller single worst peak does not offset the higher median per-trial peak, and the worst increment was essentially unchanged. The lower all-attempt worst dwell includes different failure/coverage composition; covered-transport worst dwell instead increased from 0.64 to 0.74 s. Better observed task progress therefore does not establish generally shorter pauses or a population success advantage. Neither the whole-group comparison nor failed low-motion trajectories alone can establish improved completed transport.

| Validation layout | RTC20 completions | RTC10 completions |
| --- | --- | --- |
| episode_004 | 2/3 | 2/3 |
| episode_012 | 1/3 | 3/3 |
| episode_019 | 2/3 | 3/3 |
| episode_020 | 3/3 | 3/3 |
| episode_026 | 2/3 | 3/3 |
| episode_029 | 3/3 | 3/3 |

Among the 13 matched-reset groups where both profiles reached transport coverage, RTC10 minus RTC20 had median active-p99 difference +3.271 rad/s³ and median peak difference +239.039 rad/s³. Typical p99 was lower/higher in 6/7 such comparisons; peaks were lower/higher in 5/8. These are descriptive within-reset differences with unpaired noise, not deterministic paired treatment effects. All failures remain in the ledger; the covered subset supplements rather than replaces the full denominator.

## Sampled media review

The [media manifest](../outputs/v6_rtc_same_session_20261028/media/manifest.json) binds all outcomes, event-aligned frames, source videos, and comparison assets. The [visual-review receipt](../outputs/v6_rtc_same_session_20261028/media/visual_review.json) records the sampled inspection. The primary assistant inspected six overview sheets covering the milestones/final outcomes of all 36 attempts, four new extreme-event sheets, and the fixed `episode_019` repeat-1 alignment QA. This was static sampled-frame inspection, not continuous viewing of every full video.

The [fixed episode_019 repeat-1 side-by-side video](../outputs/v6_rtc_same_session_20261028/media/episode_019_repeat1_all_outcomes.mp4) uses a layout chosen before outcomes rather than selecting successful examples. It is 50 fps at original simulation-time speed. RTC10 ends at 20.84 s and its terminal frame is then frozen until RTC20 ends at 38.20 s; that padding is not an extra policy hold or wall-clock timing measurement. One sample cannot establish a general smoothness or timing advantage.

The overview evidence includes a long near-grasp interval followed by success for RTC20 on `episode_019` repeat 1, a long near-grasp interval for RTC10 on `episode_020` repeat 3, and RTC20 on `episode_026` repeat 3 entering the box but ultimately failing the complete success rule. Not every failure is a no-grasp failure. All four new extreme sheets are in near-grasp contexts; without force/torque evidence, these associations do not establish a unique physical cause or an invisible/absent motion problem.

## Selection boundary

Two of nine numerical checks failed: both all-attempt and covered typical-p99 checks required at least a 10% reduction, whereas RTC10 increased both. The remaining peak, increment, dwell, coverage, and duration checks passed. Task success was explicitly secondary and not a gate that could override worse typical motion. The numerical decision is `retain_rtc20`, with no automatic configuration change. This report makes no claim that all new recordings were continuously watched or that the motion problem is solved.

Keep RTC10 as an experimental task-progress/dwell tradeoff, not a default smoothness upgrade. Preserve the old 18-recording tail review and the new 36-attempt comparison as distinct evidence. Small samples, one policy family, unpaired noise, differing phase coverage, and unstable extremes limit generalization; neither a favorable success count nor one improved extreme justifies causal or reliability claims.
