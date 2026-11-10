# Project state recovery — 2026-10-30

## Scope and verification boundary

This is a dated recovery snapshot for continuing development after the previous conversation lost its usable context. It combines a new read-only inspection of local guidance, source, Git state, configuration, manifests and recorded results with explicitly identified historical cloud receipts. It is not a live service-status page or a replacement for experiment protocols.

The recovery pass did not modify application code or configuration, start training, run simulation, install dependencies, or access the remote host. Only this snapshot and its documentation-index entry were added. Historical training, evaluation and restoration outcomes below were checked against existing artifacts; they were not rerun. Complete video playback and a comprehensive code audit were outside this pass.

The working application is `code/pi0复现` within the multi-project VLA study workspace. Mac simulation/data preparation, Linux GPU training/inference, the root learning environment, and independent upstream trees have separate boundaries. Read [project guidance](../AGENTS.md), [the workspace index](../../../docs/README.md) and [architecture](architecture.md) before implementation. Some selection/status statements in older documents lag the recorded results; the conflicts are listed below.

## Current conclusion

Original pi0 reproduction, supervised task adaptation, simulation execution, data preparation and evaluation are implemented. The latest completed original-pi0 experiments retain **v6 with RTC20 as the working reference and rollback target**. The newer relabeled-data candidate plus doubled wrist damping has a replicated physical-jerk benefit in specified cohorts, but failed the combined adoption requirements. It was not promoted. ACT is a separate diagnostic baseline with a completed negative task result, followed by release/contact diagnosis.

“Working reference” is not the global CLI default: [the CLI](../src/pi0_repro/cli.py) still defaults its general model workflows to `configs/aloha_sim.toml`. Explicit banana configuration selection is required. A historical restoration receipt does not establish that v6, its proxy or its tunnel is running now.

## Repository and provenance

| Item | Local inspection result |
| --- | --- |
| Workspace root | Not a Git repository |
| Application repository | Branch `main`, with no commits; no application commit history or recent commits to review |
| Existing index | `.gitmodules` and the `repos/openpi` gitlink were already staged; application files/directories were otherwise untracked |
| Pinned upstream | `repos/openpi` was clean at `215abfb217dbac7d5f1273282331b9b1866c0479` |
| Source lock | All four locked upstream-file SHA256 values matched [sources.lock.json](../sources.lock.json) |
| Local artifact footprint | Approximately 26 GB under `data/` and 3 GB under `outputs/`; these are inspection-time sizes |
| Local checkpoint files | The checkpoint directory existed but contained no files; inspected output/asset locations did not contain local `.pt`, `.pth` or `.safetensors` weights |

No Git initialization, staging, commit, reset or cleanup was performed. Application provenance therefore depends on source snapshots, run metadata, plans, receipts and their recorded digests rather than an application commit graph. The clean upstream checkout does not make the application checkout versioned or prove every experiment's recorded source equals today's source.

## Implemented modules and data flow

| Responsibility | Main entry points |
| --- | --- |
| CLI/configuration and environment launching | [cli.py](../src/pi0_repro/cli.py), [config.py](../src/pi0_repro/config.py), [run_mac.sh](../scripts/run_mac.sh) |
| MuJoCo banana scene, task conditions and simulation loop | [scene.py](../src/pi0_repro/scene.py), [banana_env.py](../src/pi0_repro/banana_env.py), [simulation.py](../src/pi0_repro/simulation.py) |
| Timestamped chunks, asynchronous request scheduling, RTC prefixes, target filtering/limits | [execution.py](../src/pi0_repro/execution.py), [pacing.py](../src/pi0_repro/pacing.py) |
| Observation/metadata validation and transport | [observations.py](../src/pi0_repro/observations.py), [remote.py](../src/pi0_repro/remote.py), [image_transport_proxy.py](../server/image_transport_proxy.py) |
| Original-pi0 serving, ROI and generation-time guidance | [serve_pi0.py](../server/serve_pi0.py), [banana_roi.py](../server/banana_roi.py), [rtc_policy.py](../server/rtc_policy.py) |
| Scripted/IK expert, optional continuous transport, training-state corrections | [banana_expert.py](../src/pi0_repro/banana_expert.py), [expert_path.py](../src/pi0_repro/expert_path.py), [recovery.py](../src/pi0_repro/recovery.py) |
| Collection, explicit review, split-preserving merge and LeRobot export | [demonstrations.py](../src/pi0_repro/demonstrations.py), [demo_review.py](../src/pi0_repro/demo_review.py), [dataset_merge.py](../src/pi0_repro/dataset_merge.py), [lerobot_export.py](../src/pi0_repro/lerobot_export.py) |
| Pi0 LoRA/projection adaptation, accumulation, weighted objectives and sampling | [banana_config.py](../server/banana_config.py), [banana_finetune.py](../server/banana_finetune.py), [banana_accumulation.py](../server/banana_accumulation.py), [banana_temporal.py](../server/banana_temporal.py), [banana_sampling.py](../server/banana_sampling.py) |
| Guarded training jobs and selection | [motion_train_job.py](../server/motion_train_job.py), [progress_train_job.py](../server/progress_train_job.py), [progress_policy_select.py](../server/progress_policy_select.py) |
| Logging, physical-motion/timing diagnosis and display | [recording.py](../src/pi0_repro/recording.py), [motion_quality.py](../src/pi0_repro/motion_quality.py), [diagnostics.py](../src/pi0_repro/diagnostics.py), [control_timing.py](../src/pi0_repro/control_timing.py), [async_multiview.py](../src/pi0_repro/async_multiview.py) |
| Separate ACT training, serving and evaluation | [act_baseline.py](../server/act_baseline.py), [serve_act_baseline.py](../server/serve_act_baseline.py), [evaluate_act_baseline.py](../scripts/evaluate_act_baseline.py) |

The policy path is top RGB camera plus 14-dimensional joint/gripper state and text instruction → transport/ROI/normalization → remote action chunk → timestamped local execution → MuJoCo step → next observation and recorded evidence. The control period is 20 ms in simulation. Configured 50 Hz pacing is not a wall-clock guarantee. Policy targets, measured physical motion, contact response and wall-clock delays are distinct quantities.

The training path is scripted expert or training-state takeover → structural checks and explicit acceptance → versioned export/sampler → supervised original-pi0 adaptation → committed checkpoint and paired normalization assets → separate closed-loop evaluation. Recovery experts are offline label generators, not runtime fallbacks. This project is simulation work, not physical-robot deployment or online reinforcement learning.

## Reference and experimental configurations

The latest retained reference is [banana_v6_bridge20_pi0.toml](../configs/banana_v6_bridge20_pi0.toml):

- Checkpoint suffix `banana_v6_long100_accum8000/7999`; original pi0, action horizon 100.
- Required normalization SHA256 `6c6e7a92643c76996bf15ba9fc3d3e346dd6bc3d8697712118d9fdb49dedf269`.
- Nominal request interval 15 steps, asynchronous PNG transport, before-step pacing, 3 Hz arm-target filtering.
- Client quintic bridge 20 steps; generation-time RTC delay 10, taper 20 and guidance cap 0.5. The RTC taper and client bridge are different settings.
- Arm target velocity/acceleration limits 1.1 rad/s and 40 rad/s²; gripper velocity limit 1.5 normalized units/s.

The [experimental candidate](../configs/banana_ik_continuous_damping2_pi0.toml) uses the relabeled-data checkpoint `999` with RTC20 and wrist damping scale 2. [Its launcher](../scripts/run_release_candidate.py) attempts restoration to v6 on exit. A dated single headless smoke passed; display mode was not verified, and abrupt termination/network failure can prevent restoration. The candidate remains experimental despite that smoke and the OOD evaluation.

## Data, training and checkpoint evidence

Several versioned datasets coexist. Their counts describe different experiments and must not be merged into one current training count.

| Dataset lineage | Recorded contents and status |
| --- | --- |
| Original v6 task data | 20 full demonstrations plus 2 recovery segments; 22 training segments / 26,497 frames |
| Motion-first v9 export | `20261026-213307-merged-ffaed3`: 77 train segments / 84,701 frames, including 12 new independent training layouts and 39 dependent corrections; full export complete |
| Progress-recovery export | 33 train segments / 37,144 frames after adding 11 accepted dependent corrections from 6 existing parents |
| Latest IK-relabeled export | `20261029-135803-ik-replaced-b8ed3a`: 22 train segments / 23,859 frames; 20 full demonstrations replaced, 2 legacy corrections retained; validation 6 / 7,423 frames and test 6 / 7,483 frames |

The latest export manifest is complete. All 22 current training Parquet hashes matched the ACT training receipt during recovery. This verifies those file identities, not that every aspect of the data is sufficient for learning. Dependent correction segments are not independent layouts. Some separate contact-recovery diagnostic candidates remain unaccepted and ineligible for training.

| Training branch | Recorded completed work and evidence boundary |
| --- | --- |
| v6 H100 | Final checkpoint suffix `banana_v6_long100_accum8000/7999` is supported by later serving metadata receipts. The actual selected configuration/log used microbatch 8 × accumulation 2, not the generic template's 2 × 8; the local long-chunk log contains startup only, not a complete 8,000-update terminal log |
| v8 control/temporal | Matched 3,000-update continuations, effective batch 32; control and temporal final `2999`; the temporal branch was not promoted |
| v9 motion pilot and A/B/C | Pilot 1,500 updates/final `1499` completed; A disabled near-term weighting, B used 3,000 updates, C removed event boost; completed evaluations survive locally despite stale preparation prose |
| Progress control/recovery | Both 1,000-update continuations committed final `999`, followed by a completed separate 18-attempt network-qualified evaluation |
| IK control/candidate | Both 1,000-update continuations from v6 committed final `999`; 85 training-audit checks passed, followed by a complete 36-attempt policy evaluation |
| ACT | 20,000 updates completed; 201 logged training points; final total loss 0.037853, L1 0.027918 and KL 0.000994; four remote checkpoint receipts each record 206,834,507 bytes |

The latest IK training uses original pi0 LoRA and trainable state/action mappings, H100, microbatch 8 × accumulation 2, seed 54, learning rate 5e-6 → 1e-6, and ROI `[275,110,540,345]`. Both runs use matched initializer/normalization/budget. Their sampled logs contain 40 points at steps 0…975: candidate total loss 0.0119 → 0.0102 and control 0.0036 → 0.0069. These are first/last sampled values; completion of 1,000 updates is established by checkpoint/status receipts, not by inventing a step-999 loss. Different data and stochastic samples limit comparisons between those endpoint losses.

ACT is a distinct official LeRobot architecture: ImageNet-pretrained ResNet18, randomly initialized Transformer/CVAE, batch 8, learning rate 1e-5, KL weight 10, seed 54. It predicts 100 actions and executes 25 before replanning, without RTC or temporal ensembling. Its synchronous execution and different initialization/objective prevent treating its runtime as a fair pi0 speed comparison.

**Checkpoint restoration is not locally complete.** No model weights were available in the inspected local checkpoint/output/asset locations; the cloud paths and sizes above are historical receipts, not freshly verified remote files. Local `assets_ik_continuous_norm` and `assets_v8_temporal_norm` were absent. The local `assets_quality_norm` normalization file's byte SHA256 was `34acb8c348b43f5e64cb79a542dec2b07178f0826e3b228899b451e7b01c7350`, which differs from the reference metadata pin. This is an already recorded serialization difference: [the v5 receipt](../outputs/rtc_improvement_20261024/banana_v5_quality_norm1000-result.json) reports `normalization_values_match=true`, the source byte hash above, and checkpoint hash `6c6e7a92…f269`; [the RTC quality report](rtc-quality-experiment.md) explains the checkpoint serialization. This is not a newly established numerical mismatch. Serving hashes the file bytes, so the source file cannot directly satisfy the checkpoint byte pin. Restore the checkpoint's own normalization asset rather than relaxing that pin.

The [dated cloud cleanup](cloud-storage-cleanup-20261029.md) removed five superseded intermediate checkpoints and regenerable reader cache, preserving every v6 file and the newer committed siblings. It verified 206 protected files and approximately 35.93 GB reclaimed at that time. It grants no standing deletion permission and says nothing about current free space.

## Completed experimental conclusions

| Experiment | Outcome | Retained interpretation |
| --- | --- | --- |
| [V9 pilot and ablations](../outputs/motion_first_20261026/ablations_20261027/REPORT.md) | v9 14/18, control 16/18; A 14/18, B 12/18, C 12/18. Pilot median p99 87.93 versus control 89.91, but worst jerk 1672.87 versus 1199.60 | No v9 promotion; one-factor results do not prove the data mixture is the sole cause |
| [RTC20 versus RTC10](v6-rtc-same-session-20261028.md) | Same-session 36 attempts: RTC20 13/18, RTC10 17/18; RTC10 typical jerk worsened about 9–10% | RTC20 retained under the frozen motion-first criterion |
| [Progress recovery](v6-progress-recovery-20261028.md) | Network-qualified retest: v6 6/6, matched control 5/6, recovery 5/6; recovery's typical jerk nearly matched control | Recovery data benefit not established; interrupted earlier cohort remains separate |
| [Contact teacher replication](contact-recovery-replication-20261029.md) | Eight pairs succeeded automatically, but typical first-second wrist-jerk improvement was 8.53%, short of 20% | No acceptance/export/training; delayed peaks invalidate a broad gain inferred from the shorter two-second window |
| [IK/continuous expert optimization](ik-expert-optimization-20261029.md) | First tight-IK round failed idle gates; continuous-transport teacher passed and 20 full labels were accepted. Subsequent policy evaluation: v6 10/12, control 9/12, candidate 12/12 | Teacher smoothness did not translate to reduced policy jerk; candidate wall gap reached about 978 ms; v6 retained |
| [Release/retreat factorial and replication](release-retreat-optimization-20261029.md) | 48-trial pilot and independent 24-trial replication complete; interrupted 2/24 cohort retained separately | Physical gains replicated but combined adoption gates failed |
| [OOD50](ood50-evaluation-20261029.md) | Candidate999 + RTC20 + damping2: 31/50; position 10/20, yaw 12/20, appearance 9/10 | Descriptive bounded suite; no v6 OOD control, no pure-weight attribution or default promotion |
| [ACT evaluation](act-baseline-validation-20261029.md) | Train layouts 0/4, validation 0/12; 9 attempts lifted/entered the box but failed release, 7 never lifted | Negative task result, not proof of data quality or universal ACT failure |
| [ACT release diagnosis](act-release-diagnosis-20261029.md) | 24 learner queries: 0 predict opening ≥0.8 in first25, 22 do in later75. Full100 suffixes released at least once in 6/6, but complete task success remained 0/6 in the two-second window | Execution horizon and contact constraints warrant a complete controlled test; no execution-setting change was deployed |

In the independent release/retreat replication, new weights plus damping2 reduced all-attempt typical physical jerk from 104.34767 to 82.80542 rad/s³ (20.6%) and worst peak from 1576.14797 to 928.95166 (41.1%). It completed 5/6 tasks, below the frozen floor of six, and its worst application gap was 123.282 ms versus v6's 49.098 ms, exceeding the 1.1× timing bound. These numbers describe the combined weight/plant intervention in that cohort. They do not establish a weight-only gain, eliminated pauses or full adoption readiness.

Among OOD's 19 failures, 18 never confirmed lift and one lifted without completing placement. OOD here means shifts outside the local fine-tuning task distribution, not guaranteed novelty relative to pretraining. Appearance combines color and lighting. None of these small-group fractions establishes population reliability.

Other completed runtime screens—[overhead](v6-runtime-overhead-20261028.md), [deferred video](deferred-video-timing-20261029.md), [overrun pacing](control-overrun-timing-20261029.md), and [wrist/contact adjustment](continued-wrist-contact-optimization-20261028.md)—retain mixed or failed adoption outcomes. Lower camera work, better median timing or smaller isolated force peaks did not establish a complete solution to motion and wall-clock stalls.

## Evidence that must remain excluded or qualified

- ACT's original `diagnosis.json` terminal intervention branches were invalid because cached dynamics were not correctly restored after `mj_setState`. Only `corrected_branches.json` supports those branch comparisons. Six corrected hold/open pairs both had 0/6 complete successes. The separate sixteen full-prefix replays and prediction queries remain valid.
- Corrected ACT contact instrumentation found finger/box-bottom contact in all six cases. Projected opening targets near 0.975 produced actual openings only about 0.546–0.657; command opening and physical release differ. This does not identify a unique cause of every failure.
- IK peak-localization v2 uses `action_prediction_index`; the preserved earlier draft used the wrong transition field and is invalid for interpretation.
- Required successful stable holds are excluded from low-motion pause diagnostics, but zero thresholded pause is not proof of prompt progress. Moving unproductively may never satisfy a stillness threshold.
- Physical jerk uses simulation-time derivatives. Video playback follows simulation time. Neither establishes wall-clock continuity, and task coverage must accompany smoothness comparisons.
- Recorded static image reviews are not complete continuous video reviews. Repeated layout resets do not pair model noise, network history or contact trajectories.

## Open tasks and proposed order

1. **Restore runnable asset certainty before any future execution.** Verify the intended remote checkpoint, exact normalization artifact, server metadata, proxy/tunnel and GPU/storage status. This recovery did not make those live checks or download weights. Obtain the checkpoint-serialized normalization asset required by the byte pin; the known source/checkpoint serialization difference is not evidence of local corruption.
2. **Test ACT execution horizon/replanning in full closed loop with fixed weights and scenes.** The 100-step suffix diagnostic motivates this test but does not complete it. Coordinate release with arm retreat/contact behavior; opening-only branches were insufficient. Freeze metrics and retain failures before evaluation.
3. **Resolve motion versus wall-clock tradeoffs.** No candidate has yet met the selected physical-motion, task-success and timing requirements together. Distinguish network/inference waits, local scheduling, environment-step envelopes, target updates and physical contact rather than assigning all delay to one cause.
4. **Investigate OOD grasp failures.** Position/yaw cases identify a coverage gap; targeted collection and subsequent evaluation have not yet been completed. A matched v6 OOD arm is also absent. Held-out failures must not silently become training corrections.
5. **Keep unaccepted teacher candidates out of training.** Automatic completion is not explicit acceptance. The contact-recovery candidates and eight diagnostic IK recovery candidates retain their pending/rejected boundaries; the accepted 20 full relabels are a different cohort.
6. **Review historical v9 follow-ups only if still relevant.** Data-side D (old data only) and E (without 39 corrections) remain proposals; no corresponding result was found in this inspection. The historical change to `motion_train_job.py` from literal update/near-term guards to plan-bound guards is present in current `validate()`: near-term fields and update count use plan values with legacy defaults, and `final_step` must equal updates minus one. Independent review of that implemented change was flagged historically; no separate review-completion record was found.
7. **Reconcile documentation and application versioning in a separately authorized pass.** The stale claims below can misroute future work. There is also no application commit baseline. This recovery did not stage or commit existing work.

These are remaining work items, not permission to train, collect data, modify runtime behavior or delete files. The user's present instruction remains read-only recovery with no training.

## Documentation conflicts and how to resume safely

| Older source | Stale or ambiguous statement | Evidence to prefer for this snapshot |
| --- | --- | --- |
| [README](../README.md) and parts of [architecture](architecture.md) | v3 selected operation profile | Later 2026-10-28/29 experiments retain v6 RTC20; explicit CLI profile selection still applies |
| [Motion-first record](motion-first-optimization-20261026.md), [long-chunk guide](long-chunk-training.md), older index entries | v8 control as working reference; v9 prepared/not deployed/no result | Completed v9/ablation ledgers, later retained v6 decisions; the original preparation plan is a historical immutable artifact |
| [2026-10-27 afternoon handoff](../HANDOFF_2026-10-27.md) | v9 training in progress in its first sections | [Evening handoff](../HANDOFF_2026-10-27-eod.md) and complete local evaluation ledger establish later completion |
| Historical handoffs and cloud receipts | Live-looking PIDs, ports, free space and “restored” wording | Treat as dated observations only; no remote availability check occurred during this recovery |
| Intermediate status values such as `both_trained_pending_validation` | Can appear to imply evaluation is still pending | Later complete evaluation/aggregate/completion artifacts determine the completed experiment outcome |

Do not erase historical failures or rewrite preparation snapshots to make them appear contemporaneously correct. A future maintenance pass should add precise supersession pointers and update current navigation/selection language while preserving experimental history. This snapshot adds a recovery entry without silently changing those older conclusions.

## Primary evidence to carry into the next development conversation

New local static checks parsed 257 Python files under `src/`, `server/`, `scripts/` and `tests/` with `ast.parse`, and 89 configuration TOML files with `tomllib`, without syntax errors. The four upstream-source and 22 training-Parquet hash checks also passed. Syntax/hash checks are not unit-test execution, training validation or a simulation result. The primary agent performs the workspace documentation checker and final source consistency review before delivery.

- [Pinned sources](../sources.lock.json), [reference config](../configs/banana_v6_bridge20_pi0.toml), and [experimental candidate config](../configs/banana_ik_continuous_damping2_pi0.toml).
- [V9 complete evaluation ledger](../outputs/motion_first_20261026/v9_evaluation_20261027/comparison.json) and [ablation aggregate](../outputs/motion_first_20261026/ablations_20261027/aggregate.json).
- [IK training audit](../outputs/ik_continuous_20261029/cloud_training_audit.json), [evaluation aggregate](../outputs/ik_continuous_20261029/evaluation/aggregate.json), and [completion](../outputs/ik_continuous_20261029/completion.json).
- [Independent release/retreat replication](../outputs/release_retreat_20261029/closed_loop_replication_direct/aggregate.json) and [audit](../outputs/release_retreat_20261029/closed_loop_replication_direct/audit.json).
- [OOD aggregate](../outputs/ood50_20261029/aggregate.json), [ACT training readback](../outputs/act_baseline_20261029/training_readback.json), [ACT completion](../outputs/act_baseline_20261029/completion.json), and [ACT diagnostic completion](../outputs/act_release_diagnosis_20261029/completion.json).

The next conversation should start from this snapshot and these specific artifacts, then inspect actual live dependencies only when execution is requested. The existing results establish an implemented and extensively diagnosed experimental pipeline; they do not establish a fully restored remote runtime or a universally improved policy.
