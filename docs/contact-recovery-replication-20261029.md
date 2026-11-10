# Contact-recovery teacher replication — 2026-10-29

## Scope and decision

The frozen 8 mm clearance floor plus ten-step stationary unloading teacher was tested on eight other existing v6 training parents, disjoint from the previous six-parent acquisition. The [earlier four-state screen](contact-target-recovery-20261029.md) remains separate and is not pooled. These are existing training layouts, not new independent layouts or held-out performance data. The decision is `do_not_accept_export_or_train`: the typical first-second wrist-jerk gate again failed. Runtime, weights and v6 RTC20 defaults remain unchanged.

The [plan](../outputs/contact_recovery_replication_20261029/plan.json), [takeover selection](../outputs/contact_recovery_replication_20261029/takeovers.json), [teacher ledger](../outputs/contact_recovery_replication_20261029/teacher_candidates.json), [aggregate](../outputs/contact_recovery_replication_20261029/aggregate.json) and [audit](../outputs/contact_recovery_replication_20261029/audit.json) preserve the complete experiment. All 399 audit checks passed. Entry points are the [learner acquisition runner](../scripts/replicate_contact_train.py), [deterministic selector](../scripts/select_replication_takeovers.py), [paired teacher collector](../scripts/collect_replication_experts.py) and [audit](../scripts/audit_contact_replication.py).

## Acquisition and same-state teacher comparison

Eight v6 learner acquisitions completed with five successes and zero inference underruns. This is train-only state acquisition, not a validation success estimate. The frozen selector used six sustained-effort states and two reference states; all eight were retained. The aggregate's subgroup diagnostics preserve those roles rather than silently treating references as effort-triggered cases.

Each selected state generated clearance-only and stationary-unload expert trajectories, sixteen automatic successes in eight pairs. Paired teachers share the same reconstructed full-integration start-state hash; exact replay errors for recorded 14D pre/post state were zero. This establishes equality between reconstructed teacher starts, not equality of every unrecorded physical variable to the original learner source. Same-state deterministic teachers do not imply paired stochastic learner sampling. Prior learner controls provide a continuity boundary; future learner actions are not ground-truth labels.

## Frozen quality gates

| Metric | Clearance reference | Stationary unload | Observed change |
| --- | --- | --- | --- |
| First-second median wrist jerk peak (rad/s³) | 125.605613 | 114.891678 | −8.53%; failed required −20% |
| First-second worst wrist jerk (rad/s³) | 570.654869 | 207.193196 | −63.69% |
| First-two-second median wrist jerk peak (rad/s³) | 264.681876 | 136.222690 | −48.53% |
| Whole-episode worst arm jerk (rad/s³) | 570.654869 | 370.783731 | −35.02% |
| Median active jerk p99 (rad/s³) | 234.877169 | 232.618451 | −0.96% |
| Median duration (simulation s) | 23.68 | 23.88 | +0.20 s |
| Automatic expert success | 8/8 | 8/8 | All retained |

The other eight gates passed, but lower worst peaks and successful expert completion do not override the failed typical first-second reduction. No new tuning was performed after seeing outcomes. The [training decision](../outputs/contact_recovery_replication_20261029/training_decision.json) retains no acceptance, export or training; no upload or runtime intervention is implied. This is label-quality evidence, not a VLA-model gain.

## Box-stage diagnosis is separate

Learner and teacher box geometry/dwell records are diagnostics, never policy inputs, label acceptance shortcuts or proof of intent. Five learner trajectories had available box coverage with exposure 3.14–4.20 seconds; the other three report unavailable values as null. All five available trajectories had zero low-motion dwell meeting the ≥0.5-second threshold. That does not establish absence of subjective hesitation or a moving-but-unproductive policy stall. Missing phases must not be scored as zero delay or smooth completed transport.

## Post-hoc delayed-peak diagnosis

The later [three-second diagnostic](../outputs/contact_recovery_replication_20261029/delayed_peak_diagnostic/diagnostic.json) and [expanded-window plot](../outputs/contact_recovery_replication_20261029/delayed_peak_diagnostic/first_three_seconds.png) qualify the apparent first-two-second improvement. First-three-second median wrist peak was 264.681875706 versus 264.439731836 rad/s³ (−0.091485%, virtually unchanged); worst was 570.65486908 versus 273.10103178 (−52.1425%). Lift peaks for parents 1, 2, 5 and 6 shifted from 1.88 to 2.08 seconds, and parent 3 from 1.90 to 2.10 seconds. Thus the two-second cutoff excludes an almost unchanged-amplitude candidate lift peak; its −48.53% median must not be interpreted as robust improvement or absence of delayed peaks. This is explicitly post-hoc, not a new gate or changed decision.

The [near-peak target samples](../outputs/contact_recovery_replication_20261029/delayed_peak_diagnostic/peak_target_samples.json) show parent 1 increments of approximately 0.000873/0/0.001116/0/0.001382 rad, repeated ten steps later by the candidate. [Expert IK](../src/pi0_repro/banana_expert.py) seeds from `last_q` and can return early within tolerance. Reference quantization is a possible hypothesis needing a controlled experiment, not an established cause. Unload's initial post-step effort also retains roughly 1–2 Nm while clearance decays immediately; effort was not uniformly lower.

## Media and final verification

The [primary review receipt](../outputs/contact_recovery_replication_20261029/primary_review.json) binds eleven inspected images: six outcome/takeover sheets, two original plots, the three-second plot and two grid first frames. This is sampled static review, not complete continuous playback or explicit data acceptance. The [media manifest](../outputs/contact_recovery_replication_20261029/media/manifest.json) records review pending false with scope and preserves its pre-review snapshot.

Two all-outcome videos retain all sixteen teacher attempts: [parents 0–3](../outputs/contact_recovery_replication_20261029/media/all_outcomes_parents_0_3.mp4) and [parents 4–7](../outputs/contact_recovery_replication_20261029/media/all_outcomes_parents_4_7.mp4). Both are 960×1600 at 25 fps with labeled terminal-frame freezing and simulation-time playback. The first contains 606 frames (24.24 s); the second 619 frames (24.76 s). All original sixteen source videos remain at 50 fps. These grids sample frames, not a new policy trajectory.

The [final verification](../outputs/contact_recovery_replication_20261029/final_verification.json) passed 81 checks, binding unchanged source/default pins, eight unchanged training parents, script snapshots/syntax and all sixteen candidates still pending/ineligible for training. No data were accepted, exported, uploaded or trained.

## Next step — proposal only

Investigate velocity/acceleration continuity at offline recovery entry and close-to-lift reference/IK-tolerance behavior in a separately predeclared experiment. This is not implemented functionality or a new training authorization implied by the results. Retain the frozen v6 RTC20 reference, the failed gate and all original teacher/learner records. Data acceptance, merge/export and any data-only matched continuation require a subsequent quality pass and explicit review; these candidates remain unaccepted.
