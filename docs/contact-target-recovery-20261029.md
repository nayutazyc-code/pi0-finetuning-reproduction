# Contact-target tracking and recovery preparation — 2026-10-29

## Boundary and current decision

This record separates retrospective tracking audits, failed offline effort-limit prototypes and completed train-only recovery teacher screens awaiting acceptance. No runtime integration, configuration field, cloud trial, weight change or default promotion occurred. Existing execution/plant settings remain untouched. Frozen-action response is not closed-loop inference evidence.

## Expert versus learner tracking audit

The [audit plan](../outputs/contact_target_audit_20261029/plan.json) and [summary](../outputs/contact_target_audit_20261029/summary.json) cover all twenty full expert trajectories from v6's actual training lineage and two learner diagnostic recordings. Recorded 14D joint/gripper states replayed exactly, with expert final-object positions preserved. This does not establish equality of every unrecorded physical variable. Learner diagnostic validation states are not labels.

At 50 Hz post-step sampling, the largest expert actuator effort was approximately 0.55992 Nm, versus learner diagnostic peaks of approximately 3.05 and 4.28 Nm. A 20 ms sample interval aliases shorter impulses and cannot represent 2 ms physics-step peaks. It therefore cannot directly justify a low physical torque bound or classify all expert contact as gentle.

## Failed static effort screen

The [static screen](../outputs/wrist_effort_screen_20261029/summary.json) calibrated expert effort at 2 ms resolution: the expert peak was 4.62725 Nm. A single static cap of 6 Nm left the learner curves unchanged and failed its gate. The apparently low expert maximum at 50 Hz did not survive finer sampling as an equivalent bound. Plans and source snapshots remain separate for each native experiment.

## Failed duration-based prototype

The [offline-only prototype](../src/pi0_repro/wrist_effort_guard.py) observes previous-command unbounded tracking demand: three high samples above 1 Nm activate a ±1 Nm cap; three samples below 0.5 Nm release it. It does not consume task-phase or contact labels and is not wired into the environment, configuration or simulation runner. Availability of this module is not an operational feature.

The [sustained-guard screen](../outputs/sustained_wrist_guard_screen_20261029/summary.json) replayed all twenty expert trajectories exactly with zero guard activation. On frozen learner commands, the continued-model physical jerk peak fell from approximately 2,710.68 to 1,318.80 rad/s³ (−51%) and v6's from 1,531.59 to 1,090.29 (−29%). However, both frozen tasks ended unsuccessful and v6 typical jerk p99 increased approximately 14.6%. The gate failed and remains failed. Lower selected peaks do not establish usable task behavior, global controller improvement or closed-loop policy benefit. Four offline-guard unit checks passed; this does not change the failed task/jerk screen or integrate the prototype into runtime.

## Preselected train-only recovery states

The [selection](../outputs/contact_recovery_20261029/selection.json) examined six existing headless v6 training parents using exact replay and strict canonical scene/split checks. Four earliest persistent-effort entries were preselected: parent 0 step 387, parent 2 step 441, parent 4 step 344 and parent 5 step 353. Two validation diagnostic recordings never contribute labels. These are dependent states within existing training layouts, not new independent layouts.

Collection uses the existing recovery collector and motion-expert profile. The first four attempts completed with two automatic successes and two failures, all retained in the [candidate ledger](../outputs/contact_recovery_20261029/candidates.json). Review, acceptance, merge, export, upload and training are not established here. Selector preflight attempts 01/02 remain preserved: schema/default-tuple JSON-canonical guards stopped them before physics stepping. They are not policy failures. Review must retain failed candidates and examine takeover continuity, subsequent contact/transport and source provenance before any training proposal.

## Separate offline clearance amendment

The failed labels exposed an expert descend target near 0.001974 m, actual height around 0.0037 m and blocked XY motion after the banana rolled lower. This is a concrete label-generation issue, not proof of a cloud-policy failure. The prior expert source and its hash are preserved as `expert_before_clearance.py` in the experiment directory.

The [expert implementation](../src/pi0_repro/banana_expert.py) adds optional `recovery_grasp_min_height`, default `None`, finite within 0–0.05 m. It clamps grasp height only in recovery-mode table-resting preparation before descent; normal/default experts and the already-held recovery branch are unaffected. Three clearance unit checks passed. The output-local [clearance configuration](../outputs/contact_recovery_20261029/clearance_expert.toml) sets 0.008 m; the maintained motion-expert configuration is unchanged.

A [second candidate batch](../outputs/contact_recovery_20261029/clearance_candidates.json) uses the same four states with this single clearance change. These are additional attempts, not replacements or new parents. All four clearance attempts succeeded automatically, but takeover quality is not ready for acceptance; automatic success is not explicit acceptance. A read-only visual/metric review receipt will not itself merge, export or upload labels. No cloud policy inference or training was performed for these label screens, and the effort guard remains offline only.

## Failed moving bridge and separately gated stationary unloading

A third four-state batch adds output-local `recovery_target_bridge_steps = 10`: over ten steps it crossfades the previous executed target into each newly generated moving expert target. Default zero preserves existing behavior; the option accepts integers 0–50. All four attempts succeeded, but the predeclared [bridge comparison](../outputs/contact_recovery_20261029/bridge_comparison.json) failed: median first-second wrist jerk increased from 330.99 to 410.56 rad/s³ (approximately +24%), and worst fell from 1,103.54 to 892.79 rad/s³, only 19.1% rather than the required 20%. The bridge is not accepted merely because tasks succeeded. Original, clearance and bridge batches each retain all four attempts and source snapshots.

A distinct [unload plan](../outputs/contact_recovery_20261029/unload_plan.json) was frozen before acquisition. `recovery_unload_steps`, default zero and integer 0–50, is mutually exclusive with the moving bridge. Only recovery-mode table-resting/not-grasped entry performs a stationary ten-step (0.2 s) quintic transition from the prior arm control toward the original measured arm q, preserving gripper commands; it then refreshes actual q and begins normal expert motion. Ordinary, held and disabled behavior are unchanged. The output-local `unload_expert.toml` combines the 8 mm floor with ten unload steps and no moving bridge. Prior learner control is a continuity boundary; no future learner actions are copied as ground truth.

The [unload candidates](../outputs/contact_recovery_20261029/unload_candidates.json) completed four automatic successes with structural checks passed. The [unload audit](../outputs/contact_recovery_20261029/unload_audit.json) passed 90 checks: eight clearance-reference/unload replays reproduced recorded 14D state with zero error. This does not establish every unrecorded physical variable. The [predeclared comparison](../outputs/contact_recovery_20261029/unload_comparison.json) retained all eight gates:

| Metric | Clearance reference | Stationary unload | Gate outcome |
| --- | --- | --- | --- |
| First-second median wrist jerk (rad/s³) | 330.992 | 318.360 | −3.82%; failed required ≥20% reduction |
| First-second worst wrist jerk (rad/s³) | 1,103.543 | 412.062 | −62.66%; passed |
| First-two-second median / worst (rad/s³) | 330.992 / 1,103.543 | 318.360 / 412.062 | No delayed early peak; passed |
| Whole-episode worst arm jerk (rad/s³) | 1,103.543 | 710.482 | −35.62%; passed |
| Whole-active median p99 (rad/s³) | 241.870 | 248.968 | +2.93%; within 10% bound |
| Median duration (simulation s) | 23.54 | 23.74 | +0.20 s; passed |
| Automatic task success | 4/4 | 4/4 | Passed |

Only the typical first-second reduction gate failed. Its failure remains decisive: reduced worst takeover/whole-episode peaks do not justify rewriting the median target or accepting the labels. The moving-bridge median regression remains a separate failed screen.

Across original, clearance, moving-bridge and stationary-unload stages there are sixteen attempts at the same four training states, with automatic success counts 2/4, 4/4, 4/4 and 4/4. They are not sixteen independent layouts. The three replay audits contain 82, 82 and 90 checks and repeat clearance references; their checked records must not be summed as independent trials. Candidate metadata remains ineligible for training with human review pending. There has been no acceptance, merge, export, upload or training.

The final `tests_after_unload` result passed twenty checks: four prototype, seven new expert-option and nine earlier expert checks. Disabled-path fixture history and frozen `expert_before_*` sources remain preserved. Only offline expert options changed; default runtime configuration, simulation, scene and environment paths were untouched. No cloud-policy inference or VLA-model quality gain is established. The [final verification](../outputs/contact_recovery_20261029/final_verification.json) confirms unchanged runtime source fingerprints, unchanged metadata/trajectories for all six original training parents, all sixteen candidates still ineligible with human review pending, twenty tests and the 82/82/90-check audits. The [next-stage record](../outputs/contact_recovery_20261029/next_stage.json) explicitly retains `candidate_only_failed_typical_motion_gate`: no acceptance, export or training.

## Final media and review boundary

The [primary review receipt](../outputs/contact_recovery_20261029/primary_review.json) binds the exact [four-stage plot](../outputs/contact_recovery_20261029/final_plots/expert_contact_and_takeover.png), original/clearance/bridge/unload outcome sheets covering all sixteen unique attempts, and the final comparison's first frame. This is sampled static review, not complete continuous playback or explicit data acceptance. Current final-plot and unload-media manifests mark pending false and preserve pre-review snapshots; earlier intermediate manifests may remain pending and are not represented as universally reviewed.

The [eight-outcome expert comparison](../outputs/contact_recovery_20261029/unload_media/all8_expert_unload_comparison.mp4) was verified at 960×1600, 25 fps, 602 frames and 24.08 seconds. Clearance reference appears left and stationary unloading right, comparing the same four training states. It retains all eight expert outcomes with labeled terminal freezes at 1× simulation time. Original 50 fps videos remain preserved; final 25 fps grids sample frames without task-frame interpolation. This expert-label comparison does not demonstrate a VLA inference improvement.

## Next steps — proposal

The completed acquisition and sampled visual/structural review did not satisfy the typical-motion gate; keep these segments unaccepted. Any revised label-generation proposal needs a new predeclared comparison before reconsideration. Any later data or continuation experiment needs its own bound plan and preflight. Do not increase batch size or training updates by default, use validation failure states as training labels, or infer that failed effort-cap prototypes should become production fallbacks. Historical reports and original recordings remain unchanged.
