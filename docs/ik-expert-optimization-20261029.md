# Offline expert IK and continuous-transport optimization — 2026-10-29

## Scope

These two rounds optimize offline teacher trajectories at the same eight preselected training states, not learned-policy execution. Original-pi0 runtime weights, RTC20, scene/controller defaults and held-out data remain unchanged. The first failed quality decision is preserved; the second round passed its frozen teacher gates. Twenty regenerated full-training labels are now assistant-accepted; the versioned export and both matched training runs are complete, and the complete policy evaluation failed its motion-adoption gates. The original v6 reference remains selected.

## Round 1: tighter IK, motion gain with idle regression

The [precision plan](../outputs/ik_precision_20261029/plan.json), [aggregate](../outputs/ik_precision_20261029/aggregate.json) and [288-check audit](../outputs/ik_precision_20261029/audit.json) retain eight fresh baseline/tight-IK pairs, all sixteen automatically successful. Only output-local position tolerance changed from 0.0002 to 0.00002 m, keeping the 8 mm floor and ten-step stationary unload. Full-lift and whole-trajectory measurements avoid the earlier two-second window-shift artifact.

Median full-lift wrist jerk fell from 274.48684 to 19.57268 rad/s³ (−92.87%); whole-active p99 median fell from 232.61845 to 29.41081 (−87.36%), and worst arm jerk from 370.78 to 145.52 (−60.75%). Median duration remained 23.88 seconds. Nevertheless, low-motion duration median increased from 1.02 to 2.09 seconds and worst span from 0.58 to 0.94 seconds. Both idle gates failed. Those pauses occurred at lift→transport, transport→lower and lower→release, rather than being excluded gripper holds. Reduced jerk did not establish acceptable overall teacher behavior. The first conditional relabel plan was prepared but never run.

## Round 2: separately declared continuous transport

The [second plan](../outputs/ik_continuous_20261029/plan.json), [aggregate](../outputs/ik_continuous_20261029/aggregate.json) and [164-check audit](../outputs/ik_continuous_20261029/audit.json) compare the bound original eight baselines with eight new candidates. The tight-only first-round reference is diagnostic, not a replacement baseline. Two additional default-off parent0 compatibility generations exactly reproduced the first-round baseline/tight arrays.

The output-local candidate keeps tight IK and enables `continuous_transport_radius = 0.01` m. The [pure path helper](../src/pi0_repro/expert_path.py) rounds Cartesian corners in virtual distance with C2 transitions and smooth 0.5-second velocity ramps, bounded by the minimum existing precision/transit Cartesian speed. [Expert execution](../src/pi0_repro/banana_expert.py) uses it after closure for lift/transport/lower. Mathematical path regularity does not guarantee physical joint/servo acceleration, jerk or contact forces.

| Metric | Bound original baseline | Continuous candidate |
| --- | --- | --- |
| Whole-active median jerk p99 (rad/s³) | 232.61845 | 28.65946 |
| Full-lift median wrist jerk (rad/s³) | 274.48684 | 28.34573 |
| Worst arm jerk (rad/s³) | 370.78373 | 145.51729 |
| Median duration (simulation s) | 23.88 | 20.88 |
| Median low-motion duration / worst span (s) | 1.02 / 0.58 | 0 / 0 |
| Box-stage duration (s) | 5.73 | 3.36 |
| Candidate automatic success | — | 8/8 |

All twelve predeclared gates passed. Whole p99, full-lift median and worst arm jerk decreased approximately 87.68%, 89.67% and 60.75%. These are offline teacher outcomes at selected training states, not VLA policy gains or general success-rate evidence. Twenty unit checks passed: thirteen expert/path checks and seven clearance checks. Old failed screens and source snapshots remain unchanged.

## Full-training-label regeneration — reviewed and exported

The [relabel plan](../outputs/ik_continuous_20261029/relabel_plan.json) and [450-check audit](../outputs/ik_continuous_20261029/relabel_audit.json) cover all twenty original full-training layouts. Original sources contain sixteen base and four quality-speed-scaled configurations; both groups' first-episode compatibility arrays matched exactly before candidate collection. All twenty candidates succeeded, passed structural checks and all seven full-label quality gates.

This is a combined teacher change from original 1.5 mm IK tolerance/cubic transport to 0.02 mm tolerance plus 0.01 m continuous transport, not an isolated tolerance experiment. Median active jerk p99 fell from 2,565.0303066 to 83.7180161 rad/s³ (−96.736%), worst arm jerk from 2,793.550491 to 1,668.587327 (−40.27%), and median duration from 23.85 to 20.47 seconds. Thresholded low-motion dwell remained zero. Residual peaks in the unchanged cubic approach remain; neither these teacher gains nor successful tasks imply a learned-policy gain.

The [full-label review receipt](../outputs/ik_continuous_20261029/fulltrain_primary_review.json) records primary inspection of five all-twenty outcome sheets, two plots and two round-two candidate sheets, approving twenty labels under user-delegated assistant review. Their new metadata are assistant-accepted; original records remain unchanged. The eight recovery candidates remain pending and are not automatically accepted alongside them.

The [data preparation record](../outputs/ik_continuous_20261029/data_preparation.json) is `exported_pending_training_preflight`: the new export has 22 training episodes / 23,859 frames versus the original 22 / 26,497. Only twenty full-training segments were replaced. Both legacy recovery trajectories and the six validation/six test mapping/trajectory digests are unchanged. Original per-layout/per-phase sampler probability mass is preserved exactly, with old within-phase weights constant; shorter episodes are not silently assigned a different group/phase mass. Native source-immutability checks passed.

Export used the existing dedicated `.venv-data` and pinned-vendor environment without installs. The initial wrong-environment attempt failed at import before export writes; resumption used the same owned version rather than reaccepting or recopying data.

The [train-only preparation plan](../outputs/ik_continuous_20261029/training_plan_train_only.json) binds two fresh control/candidate runs to the same v6 initializer, fixed normalization asset, seed 54, microbatch 8 × accumulation 2, 1,000 optimizer updates and weighted flow matching. Both use final-only save interval 1,000, producing `999` under the pinned upstream schedule. Three source-guard checks passed. Capacity must cover both finals plus a 20% margin and 2 GiB; no old weights are deleted by this preparation.

Public network connectivity was restored and the [upload verification](../outputs/ik_continuous_20261029/upload_verification.json) confirms all uploaded hashes and unchanged base data. No held-out contents were uploaded and no old files were deleted by this deployment. An initial preflight mistakenly required a not-yet-uploaded new helper and stopped before upload or service changes; its retained log distinguishes that guard issue from a training failure. Handling for new helpers was corrected. The [dispatch receipt](../outputs/ik_continuous_20261029/training_dispatch.json) established the first supervisor launch. Capacity checks passed: initializer size 5,591,368,295 bytes, free space 10,726,109,184 bytes for control and 11,887,321,088 for candidate, against 8,857,125,602 bytes required per arm. Control's GPU probe passed at 16,881/24,564 MiB peak and approximately 0.6534 s/update. Candidate's probe failed on PyArrow footer magic before any gradient update; neither arm entered formal training.

The [late-upload diagnosis](../outputs/ik_continuous_20261029/late_upload_diagnosis.json) found all original-data hashes intact but candidate episode15 changed to 3,599,644 bytes, with mtime later than upload verification/reader checks. Delayed writing by an old interrupted receiver is a hypothesis, not a proven cause; no rsync or training process remained at inspection. The [repair receipt](../outputs/ik_continuous_20261029/upload_repair_receipt.json) preserves the bad file's hash/size and the absence of formal training. The failed remote job was retained as `cloud_training_attempt01`, and the damaged file under `upload_repair`, without deletion. Deployment now puts rsync partials in an output-local `--partial-dir`; missing files were resumed before retrying the unchanged frozen plan. Training sources, configuration, labels and update budget are unchanged. The [post-repair stability audit](../outputs/ik_continuous_20261029/cloud_data_stability.json) passed 79 bound-source/data checks across both arms, with no delayed rewrite observed. Both retried probes passed with actual shape `[16, 100, 32]`: control peak 16,879 MiB and 0.6545675 s/update; candidate peak 16,881 MiB and 0.6537909 s/update, out of 24,564 MiB. The [training audit](../outputs/ik_continuous_20261029/cloud_training_audit.json) passed 85 checks, verifying frozen source/configuration/sampler/data/normalization bindings. Status is `both_trained_pending_validation`: both arms completed 1,000 optimizer updates and committed final `999` checkpoints (control 5,591,956,710 bytes; candidate 5,591,753,609 bytes). The original probe/data-file failure remains retained; successful retry does not erase it. Training completion alone does not establish a learned-policy improvement.

## Complete policy evaluation — no motion promotion

The [predeclared plan](../outputs/ik_continuous_20261029/evaluation/plan.json), [complete ledger](../outputs/ik_continuous_20261029/evaluation/comparison.json), [aggregate](../outputs/ik_continuous_20261029/evaluation/aggregate.json) and [completion receipt](../outputs/ik_continuous_20261029/completion.json) bind all 36 cases: six validation layouts × two attempts × three models with reversed model blocks. Policy noise was unpaired. All [899 audit checks](../outputs/ik_continuous_20261029/evaluation/audit.json) passed, and restoration metadata verified v6. Test data were not used. Earlier cohorts are not pooled.

| Measure | v6 | Fresh control | Relabeled-data candidate |
| --- | --- | --- | --- |
| Success / attempts | 10/12 | 9/12 | 12/12 |
| All-attempt median active jerk p99 (rad/s³) | 88.0375 | 91.4884 | 120.5486 |
| Transport-covered median active jerk p99 (rad/s³) | 108.1191 | 91.4884 | 120.5486 |
| Worst physical jerk peak (rad/s³) | 1,161.8361 | 2,131.8942 | 1,748.9120 |
| Worst physical joint increment (rad) | 0.02202055 | 0.02200186 | 0.02204618 |
| Longest thresholded low-motion pause (s) | 0.68 | 0.92 | 0 |
| Covered median duration (simulation s) | 22.56 | 23.30 | 21.13 |
| Covered median control-wall duration (s) | 23.1488 | 23.6657 | 21.8530 |
| Worst apply interval (ms) | 156.48 | 141.10 | 977.98 |

Median all-attempt pause duration was zero in every group; zero thresholded pause does not establish absence of moving hesitation. Coverage differs, so all-attempt and covered jerk remain separate. Candidate completion and duration were favorable in this sample, but versus v6 covered typical jerk increased approximately 11.5%, all-attempt typical jerk 36.9% and worst peak 50.5%. The primary jump-reduction objective failed; the decision is `retain_v6_reference`. Teacher smoothness did not transfer into improved policy motion under this continuation. These small unpaired results do not establish general reliability or justify automatic promotion, more training or a pass-triggered replication stage.

The candidate had one inference underrun with 0.975961 seconds of waiting and a 977.98 ms worst apply gap. Thus the successful task count must not be reported as absence of wall-clock stalls.

## Post-hoc peak localization

Only [peak-localization v2](../outputs/ik_continuous_20261029/evaluation/peak_localization.json) is valid for interpretation: accepted chunk changes use `action_prediction_index`. Its three largest candidate peaks occurred with the banana inside the box, released and supported. Those phase flags do not prove contact or chunk-switch causality. An earlier draft misused `chunk_transition_steps` as a timestamp and remains preserved as `draft_invalid_transition_field`; it is not supporting evidence.

## Media boundary

The [round-one static review](../outputs/ik_precision_20261029/primary_static_review.json) and [round-two static review](../outputs/ik_continuous_20261029/primary_static_review.json) record primary inspection of all six pages per round (twelve pages total, with two baseline pages reused), plus both round-two plots. This is static key-frame review, not continuous video playback. It does not change the eight diagnostic recovery candidates' unaccepted status or include them in training. The separately reviewed twenty full-training labels remain accepted and exported; numerical/source audits and sampled images do not establish learned-policy improvement.


## Evaluation media and next step

The [evaluation media manifest](../outputs/ik_continuous_20261029/evaluation/media/manifest.json) binds all 36 outcomes across six sheets/grids. The [plot manifest](../outputs/ik_continuous_20261029/evaluation/plots/manifest.json) binds both complete-trajectory plots: [all-run jerk](../outputs/ik_continuous_20261029/evaluation/plots/jerk_all36.png) and [wrist-angle trajectories](../outputs/ik_continuous_20261029/evaluation/plots/wrist_angle_all36.png). The [primary review receipt](../outputs/ik_continuous_20261029/evaluation/primary_review.json) records actual inspection of all six sheets, both plots and an episode020 grid screenshot at 21 seconds. This is sampled static review, not complete continuous playback.

| Validation layout | All-outcome comparison video |
| --- | --- |
| episode_004 | [Video](../outputs/ik_continuous_20261029/evaluation/media/episode_004_all_outcomes.mp4) |
| episode_012 | [Video](../outputs/ik_continuous_20261029/evaluation/media/episode_012_all_outcomes.mp4) |
| episode_019 | [Video](../outputs/ik_continuous_20261029/evaluation/media/episode_019_all_outcomes.mp4) |
| episode_020 | [Video](../outputs/ik_continuous_20261029/evaluation/media/episode_020_all_outcomes.mp4) |
| episode_026 | [Video](../outputs/ik_continuous_20261029/evaluation/media/episode_026_all_outcomes.mp4) |
| episode_029 | [Video](../outputs/ik_continuous_20261029/evaluation/media/episode_029_all_outcomes.mp4) |

Next steps are proposals only: replay the candidate's complete release/retreat trajectory, distinguish target handoff from contact response, then predeclare one targeted intervention. Do not blindly increase batch size or update count. No new training or default change is initiated by this report; the failed first teacher round and first cloud probe attempt remain preserved.
