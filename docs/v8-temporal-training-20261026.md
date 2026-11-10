# Original-pi0 v8 expert-displacement training — 2026-10-26

## Scope and current status

This round first audits existing expert action continuity, then prepares two matched original-pi0 continuations from v6: [control](../configs/banana_finetune_v8_control.toml) and [temporal](../configs/banana_finetune_v8_temporal.toml). The [original frozen plan](../outputs/v8_temporal_20261026/plan.json) binds dataset, sampler, normalization, source, and initial configurations; a later storage-retention amendment is disclosed separately below. Deployment, CPU/reader checks, and both real-data GPU probes passed. Both 3,000-update continuations completed, and both final `2999` checkpoints are committed. All twelve fixed validation attempts are complete. Control completed six of six and temporal four of six; the temporal objective is not promoted. The existing v6 service was restored and independently checked after evaluation.

The dataset remains `20261026-142111-merged-e54349`, with all 26 reviewed training episodes/segments and 29,647 frames retained. No labels were changed after audit, no independent layouts were added, and validation/test membership remains separate. Both arms initialize from v6 `banana_v6_long100_accum8000/7999`, not the unpromoted v7 continuation, with its exact normalization digest `6c6e7a92643c76996bf15ba9fc3d3e346dd6bc3d8697712118d9fdb49dedf269`.

## Expert-action audit and provenance boundary

The [expert audit](../outputs/v8_temporal_20261026/expert_audit.json) checks requested/executed targets, observed joint motion, phase boundaries, initial target/state differences, and recorded provenance across the training set. No adjacent training target exceeded 0.05 rad on an arm joint; the maximum was approximately 0.0209978 rad. The largest phase-boundary target change was approximately 0.0116929 rad. Six recovery command seams were audited separately, with maximum approximately 0.0493503 rad; takeover seams are not ordinary adjacent training frames and are not silently merged into that continuity statistic.

The assistant inspected [representative review frames](../outputs/v8_temporal_20261026/expert_review_frames.jpg), including peaks, a phase boundary, and a recovery seam. This inspection and the finite-difference audit did not show a gross target jump under the stated diagnostic threshold. They do not establish complete smoothness, absence of physical contact spikes, or a full historical reconstruction of every IK branch. Current expert source documents bounded IK/phase generation, while recordings provide only the provenance and state/action history actually saved. High physical jerk and a discontinuous IK target are distinct findings.

Retained extremes include commanded acceleration 29.2323 rad/s² in `episode_036`, observed physical jerk 4277.654 rad/s³ in `episode_034`, and an initial target/state offset of 0.0491894 rad in `episode_034`. The separate recovery-command seam maximum of 0.0493503 rad is not the internally adjacent target maximum. The 0.05 rad threshold is an audit lens, not a newly imposed hard quality-acceptance rule or evidence that every demonstration is physically smooth. No high-jerk records were silently removed or relabeled as poor IK merely from these statistics.

## Matched training method

Both arms retain horizon 100, weighted original-pi0 flow matching, LoRA plus the configured state/action/time projections, ROI, unchanged data/sampler, and fresh optimizer state. Each requests microbatch 8 with accumulation 4 (effective batch 32), 3,000 optimizer updates, seed 48, 100 warmup updates, peak learning rate `5e-6`, decay target `1e-6`, and checkpoints every 500 updates. Final `2999` is the predeclared eventual evaluation checkpoint, not a checkpoint selected after looking at validation outcomes.

The only intended objective difference is auxiliary weight: 0 for control and 0.1 for temporal. The [native-arm objective](../server/banana_temporal.py) recovers the clean estimate as `x_t - t * predicted_flow`, then compares adjacent predicted and expert displacements in native radians. It does not divide by the control period or penalize all movement toward zero. Grippers and padded model axes are excluded. A normalization-scale conversion restores native displacement units; action means cancel and squared errors are invariant to ALOHA sign flips. Existing FM targets and dimension weights remain unchanged.

The auxiliary term is invariant to a constant joint offset applied across one chunk, as covered by a numerical check. Flow matching continues to anchor action positions. Expert-difference supervision alone therefore does not explicitly align independently replanned chunks or solve grasp/placement stalls.

The [data-mask adapter](../server/banana_temporal_data.py) supplies an explicit action-validity mask from per-episode frame counts. Both endpoints must be valid for an adjacent pair to contribute, excluding repeated episode-tail padding. Equal or zero actions are not treated as padding, so static expert poses remain valid. Both arms use the same masked data pipeline and auxiliary metric reporting. See [training operations](long-chunk-training.md) for accumulation/update semantics and the objective boundary.

## Verified preparation and reader checks

The [original preparation verification](../outputs/v8_temporal_20261026/preparation_verification.json) records source/config checksum checks before the storage amendment, when auxiliary weight was the only configuration difference. Current full TOMLs additionally differ in retention; the only objective/training-computation difference remains auxiliary weight. All five temporal-loss tests passed on the cloud, including JAX gradients. The local broader suite had 13 passes and one cloud-only skip. These numerical/structural tests are not trained-policy evidence.

The cloud CPU reader audit checked 78 real samples: start, penultimate, and final anchors for all 26 episodes. Explicit masks matched LeRobot `action_is_pad` exactly, normalized action values were unchanged with shape `[100, 32]`, and batched `TemporalObservation` construction/JIT passed. The first prelaunch reader tool attempt failed its unbatched observation type check; batching the audit input fixed it. The original prelaunch log was retained. This was an audit-tool issue, not a dataset failure, and inference had not been stopped during that repair.

## Disclosed storage-retention amendment

A read-only capacity check found 59,193,851,904 free bytes (about 55.1 GiB), with the initializer checkpoint occupying 5,591,368,295 bytes. Keeping all six scheduled checkpoints per arm could exceed that space. The [amendment record](../outputs/v8_temporal_20261026/storage_amendment.json) preserves the original configuration/hash and estimate rather than rewriting the initial plan.

Only the not-yet-started temporal arm's `train.keep_period` changed from 500 to 2,000; `save_interval = 500` remained unchanged. The [application verification](../outputs/v8_temporal_20261026/storage_amendment_verification.json) confirms the atomic change occurred before temporal full training, while control was active and no temporal training log existed. Control configuration and its 500-step retention remain unchanged. With Orbax `max_to_keep = 1` and temporal keep period 2,000, temporal retention keeps the rolling latest checkpoint, the 2,000-period checkpoint, and final `2999`, rather than preserving all intermediate 500-step saves. No preexisting weights were deleted.

This is an operational retention difference between the full TOMLs, not an objective, model, data, batch, optimizer, RNG, or loss change. Both completed GPU probes still apply to training computation; future storage behavior remains separately relevant. Using initializer size gives an estimated peak of nine new checkpoints, approximately 50.3 GB versus 59.2 GB free. Checkpoint-size variation, temporary files, and other disk growth mean this estimate is not a capacity guarantee. The original plan and preparation-verification artifacts remain historical records, and the amendment/verification supply the later configuration lineage.

## GPU execution and outcome boundaries

Two sequential runs on the 24 GB-class GPU require inference to pause. The [supervisor](../server/temporal_train_job.py), [preflight](../server/temporal_preflight.py), and [read-only status helper](../server/temporal_train_status.py) separate consistency/measurement checks, real training, and service-state verification. The [deployment helper](../scripts/deploy_v8_temporal.sh) was executed using a user-supplied existing SSH alias and the previous helper's authentication-only option. It validates frozen plan inputs, uploads only the bound sources/configurations, sampler, and evidence, and dispatches the guarded supervisor without saving passwords or creating new authentication keys. Connection identifiers are intentionally absent from this report.

The [status snapshot](../outputs/v8_temporal_20261026/training_status.json) verifies both probes passed three finite updates with action shape `[32, 100, 32]`. Each sampled peak was 16,949 MiB out of 24,564 MiB, leaving 7,615 MiB of observed headroom. Post-compilation update estimates were 1.299401 s for control and 1.303131 s for temporal. Roughly 65 minutes of update computation per 3,000-update arm is a short-probe projection, excluding compilation, saves, and throughput variation; it is not a promised completion time.

The [completion verification](../outputs/v8_temporal_20261026/checkpoint_verification.json) confirms `both_trained_pending_validation`, committed final `2999` parameters for both arms, effective batch 32, configured 3,000 updates, and the intended auxiliary coefficients. Each final normalization asset matches the initializer in values and SHA-256. The last periodically logged index is 2975, not the final update count or a distinct final-checkpoint loss. No nonfinite metric lines were found. Probe updates remain separate disposable measurements. The supervisor recorded approximately 9,078.5 seconds (2 h 31 min) from start through both completed runs, including preflight/probes and overhead. V6 restoration was subsequently verified by metadata before switching to validation candidates; this does not establish their task quality.

The [aggregate probe metrics](../outputs/v8_temporal_20261026/probe_metrics.json) verify total loss minus flow loss minus weighted auxiliary is below `1e-7` in absolute value. Temporal weighted auxiliary values in the three probe updates were approximately `1.16e-6`, `1.02e-6`, and `1.47e-6`, contributing about 0.008–0.038% of the FM term. This native-radian objective with weight 0.1 is deliberately conservative in observed magnitude; it does not promise a large training effect, and loss-value ratios alone do not quantify gradient influence. No coefficient tuning after closed-loop results is planned. Frozen source hashes still matched at the aggregate audit.

The independent [CPU accumulation regression](../outputs/v8_temporal_20261026/accumulation_diagnostic.log) passed with maximum parameter difference approximately `1.164e-10`, zero loss difference, and gradient-norm difference approximately `1.192e-7`. It checks the deterministic accumulation calculation, not trained-policy quality. A dispatched deployment or short probe does not establish completed continuation. A planned v6 restart after all runs or a failure must be verified independently; private connection details are not part of this report.

Completed optimizer updates, committed final checkpoints, finite metrics, and eventual closed-loop validation must be recorded before interpreting the temporal objective's effect. Compare control/temporal completion, grasp and placement progress, physical/reference motion, worst spikes, and timing under frozen execution settings. Preserve every failed sample and previous negative cadence, jerk, v7 recovery, and RTC-strength result. This preparation does not establish smoother motion, solved pauses, or a training winner.

## Post-training connection and evaluation boundary

A later status request failed after the task SSH socket expired. Shell redirection briefly replaced the local snapshot with an empty file; the earlier successful output was recovered with explicit historical provenance before a newly authenticated remote observation replaced it. Empty failed-poll files remain as evidence. The [status-check wrapper](../scripts/check_v8_training.sh) now validates successful JSON before atomically replacing a snapshot, and failed requests preserve the last good observation. A [local isolated failure-path check](../outputs/v8_temporal_20261026/status_poll_failure_check.json) uses an explicit SSH stub; it is a wrapper test, not remote training evidence.

The final retained checkpoints are control `500`, `1000`, `1500`, `2000`, `2500`, `2999`, and temporal `2000`, `2999`, consistent with the disclosed retention difference. Completion verification measured about 14.46 GB free afterward. Existing weights were not deleted to make space.

The [evaluation policy selector](../server/select_v8_policy.py) switches only among committed v6/control/temporal checkpoints after checking the normalization digest and absence of a live training process. A dispatched selector still needs a metadata handshake. The [control client profile](../configs/banana_v8_control32_pi0.toml) and [temporal profile](../configs/banana_v8_temporal32_pi0.toml) preserve the same v6 bridge20 execution settings with their own checkpoint pins; neither introduces new filtering or execution changes.

The [control plan](../outputs/v8_control_validation_20261026/plan.json) and [temporal plan](../outputs/v8_temporal_validation_20261026/plan.json) fix the six existing validation layouts, once per model: `004`, `012`, `019`, `020`, `026`, `029`. Model blocks run sequentially with unpaired inference noise. Reset seeds and scenes come from the preserved split mapping; no test partition, checkpoint search, loss-coefficient tuning, or execution retuning is part of this comparison. Two reset-observation RTC warmups per model are excluded from task outcomes. The completed ledgers, timing and physical/reference derivatives are compared below; warmup requests are not task attempts.

## Completed fixed-layout comparison

The [control ledger](../outputs/v8_control_validation_20261026/comparison.json) and [temporal ledger](../outputs/v8_temporal_validation_20261026/comparison.json) preserve all twelve attempts. The [combined comparison](../outputs/v8_temporal_20261026/validation_comparison.json) independently reconstructs motion from saved arrays. No attempts were removed, rerun to replace failures, or extended beyond the fixed 40-second cap.

| Layout | Control outcome / simulated duration | Temporal outcome / simulated duration |
| --- | --- | --- |
| episode_004 | Success / 22.54 s | Failure / 40.00 s; no confirmed lifted pinch |
| episode_012 | Success / 23.16 s | Success / 24.78 s |
| episode_019 | Success / 22.24 s | Success / 23.92 s |
| episode_020 | Success / 24.92 s | Success / 24.70 s |
| episode_026 | Success / 37.14 s | Success / 34.06 s |
| episode_029 | Success / 23.42 s | Failure / 40.00 s; delayed release/stabilization after entry |

| Recorded measure | Control | Temporal |
| --- | --- | --- |
| Observed completions | 6/6 | 4/6 |
| Successful-duration median | 23.29 s (six cases) | 24.74 s (four cases) |
| Median of trial physical arm-jerk p95 | 15.04 rad/s³ | 14.62 rad/s³ |
| Worst physical arm jerk | 1540.45 rad/s³ | 1167.28 rad/s³ |
| Median of active-motion trial jerk p95 | 15.07 rad/s³ | 14.67 rad/s³ |
| Running inference underruns | 0 | 0 |

Successful-duration medians use different eligible cases and are not a paired speed comparison. In the conditional four-layout subset where both models succeeded (`012`, `019`, `020`, `026`), mean completion time was 26.865 s for both; temporal was slower on two and faster on two. This post-hoc conditional view does not replace the full six-layout failure counts. Motion derivatives use the 0.02-second simulation period. Active-motion statistics additionally select samples with any observed arm speed above 0.01 rad/s; that diagnostic threshold is not a policy rule or a general smoothness guarantee.

The [episode_004 frames](../outputs/v8_temporal_20261026/validation_episode004_frames.jpg), inspected by the assistant, show control progressing to the box while temporal remains near the banana through the failed trajectory. The temporal log records 25 near-grasp gripper reversals and no confirmed lifted pinch. Its low whole-episode jerk is therefore not evidence of successful smooth task completion.

For `episode_029`, the [release diagnostic](../outputs/v8_temporal_20261026/temporal_release_failure_diagnostic.json) records first box entry at 19.20 s, a longest interval of 17.42 s inside while still in robot contact, first released observation after entry at 36.60 s, and first gripper command above 0.8 at 39.30 s. The episode did not satisfy the unchanged stable-placement rule by 40 s. Transient release and geometric entry are not task success; these observations document a remaining phase delay, not its unique cause. Neither failure was caused by a recorded inference underrun.

## Final consistency and service decision

The [twelve-trial audit](../outputs/v8_temporal_20261026/validation_consistency_audit.json) verifies identical recorded client sources across both blocks and at audit time, matched scenes/reset seeds, and runtime configuration differences limited to checkpoint pins. Every trial had zero raw-action reconstruction, action-timestamp, and RTC command-prefix error. Metadata matched each final checkpoint, native normalization, original pi0 horizon 100, auxiliary coefficient and RTC cap 0.5; lossless pixel hashes were verified. These checks establish implementation consistency rather than reliable task behavior, physical tracking, hard real-time operation, or population success rates.

The modest jerk decrease did not accompany improved completion or consistent phase progress. Do not promote the temporal checkpoint from this comparison. Retain control as an unpromoted training candidate: it still has the 37.14-second trajectory and large physical spikes, and no fresh v6 block was run in this comparison. Older v6 observations remain historical context, not a matched estimate of a larger-batch/longer-training gain.

Both new final checkpoints and all intermediate/failed evidence remain available. The [final service verification](../outputs/v8_temporal_20261026/final_service_verification.json) independently confirms restored v6 metadata after candidate evaluation, with the existing bridge20/3 Hz/RTC 0.5 client profile. The supervisor's earlier unverified restoration flag is distinct from this later handshake. No new checkpoint or runtime profile was promoted, no coefficient was retuned after validation, and no test-partition evaluation was performed.
