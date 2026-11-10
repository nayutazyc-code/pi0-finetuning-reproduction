# Release/retreat diagnosis and bounded optimization — 2026-10-29

## Scope

This record investigates the new candidate's release/retreat spikes while preserving original pi0 v6 RTC20 as reference. Exact replay and single-variable screens are offline physics responses, not closed-loop policy improvement. Application source and defaults are unchanged; the existing scene damping option is the only proposed plant intervention. Prior teacher/training successes do not override the [failed learned-motion gates](ik-expert-optimization-20261029.md).

## Full-cohort exact replay and contact evidence

The [diagnosis plan](../outputs/release_retreat_20261029/diagnosis_plan.json) includes all twelve candidate and twelve v6 sources, including failures, rather than selecting only successful videos. For available inside-box/released/supported phases it targets the maximum-jerk window ±60 control steps and records right-arm six-axis forces/contact at 2 ms resolution. The [complete diagnosis](../outputs/release_retreat_20261029/diagnosis.json) contains 48 plain/instrumented full replays; each reproduced recorded 14D pre-action states with maximum absolute error zero. This does not establish equality of every unrecorded physical variable or a new policy trajectory.

In case18, step1085 substeps 0→1 kept the same wrist target, approximately 0.9883803 rad, while right wrist-angle contact torque changed from −1.890829 to 0 Nm and acceleration from 65.608 to 245.445 rad/s². The primary-inspected [three-event plot](../outputs/release_retreat_20261029/contact_plots/three_release_events.png) and [manifest](../outputs/release_retreat_20261029/contact_plots/manifest.json) select the largest 2 ms acceleration change on each dominant 50 Hz peak axis. Case19 forearm roll likewise loses contact at unchanged target; case20 wrist angle instead recontacts at unchanged target. A nearby earlier wrist-rotation release is a different sub-event. These are multiple contact switches, not uniformly contact release. These aligned local observations help distinguish contact response from target updates at those instants; they do not prove a universal cause of all spikes or user-visible hesitation. No controller was modified for diagnosis.

## Failed gripper-opening velocity screen

The [opening screen plan](../outputs/release_retreat_20261029/opening_velocity_screen/plan.json) and [complete results](../outputs/release_retreat_20261029/opening_velocity_screen/results.json) contain 24 replays: twelve candidate sources × baseline/limited opening. Both arms append the same 100-step terminal hold; that added tail is not represented as the original recording. Only positive gripper opening speed was limited to 1.0 normalized units/s; arm commands, closure and plant settings stayed fixed.

All twelve task completions were preserved, but original-prefix typical jerk p99 changed only from 120.54856 to 113.86120 rad/s³ (approximately −5.5%) and worst peak from 1,748.912 to 1,624.011 (−7.1%). The frozen gate failed. This candidate is not applied, and no multi-value search of the same parameter is implied. Preserved completion alone does not establish an adequate motion improvement.

## Existing damping option — frozen screen passed

The [damping plan](../outputs/release_retreat_20261029/damping_screen/plan.json) and [complete results](../outputs/release_retreat_20261029/damping_screen/results.json) cover 48 frozen replays: v6/candidate twelve sources × existing `scene.wrist_damping_scale` 1/2. All sixteen gates passed. Original commands and the common 100-step terminal hold are fixed; baseline original-prefix recorded-state error was zero. Targets, weights and all other compiled coefficients were asserted unchanged. This is a plant-response screen, not learned-policy improvement.

| Original-prefix measure | Scale1 | Scale2 |
| --- | --- | --- |
| Candidate median active jerk p99 (rad/s³) | 120.54856 | 74.49086 |
| Candidate worst jerk (rad/s³) | 1,748.912 | 759.61649 |
| Candidate completion / coverage | 12/12 | 12/12 |
| v6 all-attempt median active jerk p99 (rad/s³) | 88.03748 | 62.50627 |
| v6 covered median active jerk p99 (rad/s³) | 108.11914 | 67.25568 |
| v6 worst jerk (rad/s³) | 1,161.836 | 629.22097 |
| v6 completion / coverage | 10/12 | 10/12 |

Candidate prefix typical jerk fell 38.21% and worst peak 56.57%, preserving completion/coverage. Failures are retained and all-attempt/covered v6 statistics remain separate. These responses use original actions, not fresh observations fed to the policy.

## Fresh four-profile comparison — complete, gates not all passed

The [closed-loop plan](../outputs/release_retreat_20261029/closed_loop/plan.json), [complete ledger](../outputs/release_retreat_20261029/closed_loop/comparison.json), [aggregate](../outputs/release_retreat_20261029/closed_loop/aggregate.json) and [audit](../outputs/release_retreat_20261029/closed_loop/audit.json) bind 48 fresh attempts: four weight/damping profiles, eight reversed blocks, six layouts twice per profile. Policy noise was unpaired. All 1,451 checks passed, including only six damping XML attributes changing, prefix/timestamp/reset consistency and v6 restoration.

| Measure | v6 / damping1 | New weights / damping1 | v6 / damping2 | New weights / damping2 |
| --- | --- | --- | --- | --- |
| Success / attempts | 10/12 | 11/12 | 9/12 | 12/12 |
| All-attempt median active jerk p99 (rad/s³) | 96.97814 | 117.07590 | 79.53959 | 77.68378 |
| Covered median active jerk p99 (rad/s³) | 104.18721 | 103.63013 | 98.82152 | 77.68378 |
| Worst physical jerk (rad/s³) | 1,146.30079 | 1,561.31387 | 987.76445 | 867.66481 |
| Longest thresholded low-motion pause (s) | 0.98 | 0 | 1.42 | 0 |
| Running inference underruns | 0 | 0 | 1 | 0 |
| Covered median simulation duration (s) | 21.85 | 21.84 | — | 22.08 |
| Covered median wall duration (s) | 22.59698 | 23.00383 | — | 22.66030 |
| Worst apply gap (ms) | 231.85379 | 116.13771 | — | 152.20679 |

The primary new-weights+damping2 candidate reduced covered typical jerk by 25.44% and worst physical jerk by 24.31% versus original v6. Its all-attempt typical reduction was 19.896%, narrowly below the frozen ≥20% requirement. Its 152.20679 ms worst wall gap also exceeded the same-weight damping1 reference's 116.13771 ms ×1.1 bound. These failed gates remain failed: the decision is `retain_v6_reference`, not automatic promotion or a claim that every objective improved. Physical improvement is observed, but task/coverage and wall constraints remain separately evaluated. v6+damping2 lost coverage and had one 0.34685-second inference wait, so the plant intervention cannot be generalized as uniformly beneficial across weights.

## Separate direct-route replication — complete

The original [24-case replication plan](../outputs/release_retreat_20261029/closed_loop_replication/plan.json) binds the pilot plan/aggregate and unchanged parameters/gates. That cohort started but stopped after two attempts when route validation detected a virtual interface. Its [interruption receipt](../outputs/release_retreat_20261029/closed_loop_replication/interruption.json) preserves the incomplete 2/24 record; it is not a complete replication and is not pooled with later results.

After public routing recovered on `en0`, the same unchanged script started a separate [direct-route plan](../outputs/release_retreat_20261029/closed_loop_replication_direct/plan.json) and [ledger](../outputs/release_retreat_20261029/closed_loop_replication_direct/comparison.json). Four reversed blocks cover six layouts once per four profiles. This independent complete-budget cohort completed all 24 attempts. The [aggregate](../outputs/release_retreat_20261029/closed_loop_replication_direct/aggregate.json) and [743-check audit](../outputs/release_retreat_20261029/closed_loop_replication_direct/audit.json) passed integrity checks and verified v6 restoration. No outcome-selected reruns, threshold revision, new training or default change is implied.

The [pilot review receipt](../outputs/release_retreat_20261029/closed_loop/primary_review.json) records six event pages, both full-trajectory plots, the three-contact-event plot and one grid frame. This is sampled static review, not continuous video playback. Earlier failed screens and original recordings remain intact.

### Replication outcomes and retained decision

| Measure | v6 / damping1 | New weights / damping1 | v6 / damping2 | New weights / damping2 |
| --- | --- | --- | --- | --- |
| Success / attempts | 5/6 | 6/6 | 4/6 | 5/6 |
| Transport coverage / attempts | 5/6 | 6/6 | 4/6 | 6/6 |
| All-attempt median active jerk p99 (rad/s³) | 104.34767 | 117.96975 | 78.26976 | 82.80542 |
| Covered median active jerk p99 (rad/s³) | 110.70765 | 117.96975 | 83.69456 | 82.80542 |
| Worst physical jerk (rad/s³) | 1,576.14797 | 2,018.88382 | 799.44357 | 928.95166 |
| Longest thresholded physical pause (s) | 0.84 | 0 | 0.90 | 0 |
| Worst apply gap (ms) | 49.09812 | 222.24629 | 29.45467 | 123.28212 |

All four groups had zero running inference underruns/waiting. The combined candidate's physical jerk benefit replicated, with six transport-covered trajectories but only five successful completions. Coverage and task success are different conditions. The frozen secondary success floor was six, so its five successes failed that requirement, including comparison with the same-weight damping1 reference. Its 123.28212 ms worst wall gap also exceeded v6's 49.09812 ms ×1.1 limit. The decision remains `retain_v6_reference`; physical gains do not establish that every gate passed or that new weights alone caused improvement.

This 24-case cohort is reported independently of the 48-case pilot and interrupted two-case attempt, without pooling. No default or training change follows. The [direct-cohort media manifest](../outputs/release_retreat_20261029/closed_loop_replication_direct/media/manifest.json) binds all 24 attempts in six sheets and six comparison grids; the [plot manifest](../outputs/release_retreat_20261029/closed_loop_replication_direct/plots/manifest.json) binds both full-trajectory plots. The [primary review](../outputs/release_retreat_20261029/closed_loop_replication_direct/primary_review.json) records inspection of all six sheets, both plots and one grid frame, not complete continuous playback.

The [worst candidate wall-gap diagnosis](../outputs/release_retreat_20261029/closed_loop_replication_direct/worst_candidate_wall_gap.json) locates the 123.282 ms gap at case1 step794: the previous environment-step span was 102.908 ms wall, 5.148 ms thread CPU and 10.889 ms process CPU, with zero inference blocking. This locates an observed envelope without proving its cause or showing that wall-clock stalls are fixed. Physical benefit repeated; timing remains unresolved.

## Optional experimental candidate entry — headless smoke verified

The [candidate configuration](../configs/banana_ik_continuous_damping2_pi0.toml) and its paired scene select the new candidate `999`, RTC20/client bridge20 and damping2 on fixed validation layout004/reset seed20261027, headless with 25 fps video. Relative to the corresponding audited run, only the scene-file path and owned connection port differ. The separate [launcher](../scripts/run_release_candidate.py) passed local `--check-only` and the separate headless smoke below. These dated checks do not guarantee future service availability or establish recommendation.

From the application directory:

```sh
.venv/bin/python scripts/run_release_candidate.py --check-only
.venv/bin/python scripts/run_release_candidate.py
```

The [delivery preparation](../outputs/release_retreat_20261029/candidate_delivery_preparation.json) passed four mocked lifecycle checks and records `live_headless_launcher_verified_experimental_only`. The [predeclared smoke plan](../outputs/release_retreat_20261029/launcher_smoke_plan.json) and [actual smoke receipt](../outputs/release_retreat_20261029/launcher_smoke.json) bind one successful headless run: candidate `999` metadata, RTC20/damping2 and zero prefix/timestamp error were verified, followed by verified v6 metadata on exit. The [smoke video](../outputs/20261029-180344-9e1d15/episode_000.mp4) and [four-frame review sample](../outputs/release_retreat_20261029/launcher_smoke_review.png) remain separate from formal cohorts. This single predeclared smoke was not retried or included in the 48/24 comparisons.

The launcher attempts v6 restoration on exit, which cannot be guaranteed after abrupt termination or network loss. Optional `--display` has not been verified and adds native-viewer workload outside the headless evidence. This explicit experimental candidate does not change global defaults, promote the new weights or override failed timing/success gates. Original v6 remains the selected reference.
