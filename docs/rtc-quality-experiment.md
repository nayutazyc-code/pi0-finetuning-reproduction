# RTC and quality-data experiment — 2026-10-24–25

## Evidence scope

This report records development experiments on the original-pi0 banana simulation. It is separate from the [current architecture](architecture.md) and [operational guide](rtc-and-recovery.md). Recorded experiments span 2026-10-24–25 in Asia/Shanghai. It does not establish real-robot performance, arbitrary-scene generalization, a causal improvement from one component, or current remote-service availability.

The user's pasted pi0.5 fine-tuning post supplied hypotheses: better demonstrations, broader scenes, corrections from visited states, and action-prefix guidance. Its charts were not available here, and its real-robot claims are not evidence for this original-pi0 project. RTC's conceptual references are the [original paper](https://arxiv.org/html/2506.07339v2) and [Physical Intelligence explanation](https://www.pi.website/research/real_time_chunking); the local adapter and its measurements remain a separate implementation.

The reference model for pre-continuation experiments is `banana_v3_roi2000/1999`, original pi0 with the fixed ROI and existing normalization. The physical task success rule remains grasp, containment, controlled release/support, and stability. Background inference, prefix guidance, data preparation, and continued training must be evaluated as distinct changes.

## Implemented experiment boundaries

- The optional JAX adapter guides clean-action estimates during generation; it does not edit upstream source or change checkpoint parameters.
- Native 14-dimensional prefixes pass through the same serving transforms into normalized/padded model coordinates. Padding is not guided.
- The executor uses timestamped prefixes and rejects replacements outside the old plan's valid overlap.
- Precision Cartesian phases are slower; table appearance and lighting vary within the camera-visible scene.
- Learner states provide correction starting points; only subsequent expert actions supply new labels.
- Existing baseline configurations, failed attempts, review history, and source data are retained.

This report distinguishes exploratory code from the later corrected command projection. A [25-file implementation digest record](../outputs/rtc_improvement_20261024/implemented-source-sha256.json) and [implementation snapshot](../outputs/rtc_improvement_20261024/implemented_source_snapshot/) retain the captured implemented source separately from the baseline copies. Some earlier exploratory artifacts retain source hashes but not a complete source snapshot; a later snapshot or the current file contents alone cannot reproduce every intermediate implementation exactly.

## Exploratory RTC validation before final gripper projection

These development-validation runs reused the four existing validation layouts plus the default-center scene. They were used to choose a candidate and are not an untouched final test.

| Setting | Recorded successes | Failure locations | Running-control inference waits |
| --- | --- | --- | --- |
| Guidance maximum 2.0 | 3/5 | Existing episodes 004 and 020 | 0 |
| Guidance maximum 0.5 | 4/5 | Existing episode 004 | 5 waits totaling about 1.183 seconds |

Sources: [guidance-2 suite](../outputs/banana_evaluations/20261024-185350-f42721/evaluation.json) and [guidance-0.5 suite](../outputs/banana_evaluations/20261024-185736-5e123d/evaluation.json).

The softer suite's waits occurred in default-center, episode 004, and episode 019; it was not an all-zero-underrun suite. Both suites preceded the final physical-finger projection fix and therefore do not validate the final code. The earlier time-alignment-only candidate also recorded 4/5 on a small validation set, with a different failure location; see [prior execution evidence](smooth-execution-evidence.md). Policy noise was not paired between these experiments, and each layout has too few repetitions to establish a statistically reliable advantage.

The maintained RTC candidate uses guidance maximum 0.5 with prefetch 35, delay floor 15, and blend 20. This is an exploratory setting, not an optimum or proof of universal improvement.

## Prefix fidelity: discovered mismatch and correction

The first explicit prefix audit recorded a maximum command-prefix discrepancy of about 0.02356 in a normalized gripper target. The forecast applied nominal gripper bounds, but simulator finger-actuator projection imposed an additional bound. The audit and its limitation are preserved in [the earlier prefix audit](../outputs/rtc_improvement_20261024/rtc_prefix_audit.json), which is marked superseded.

The corrected executor calls the environment's pure action projection when forecasting commands, including the physical finger-actuator limits. In the [subsequent default-scene run](../outputs/20261024-190412-6a0b3f/summary.json):

| Metric | Recorded value |
| --- | --- |
| Task outcome | Success |
| Control steps | 1,206 |
| Simulated duration | 24.12 seconds |
| Recorded control wall duration | About 24.92 seconds |
| Inference underruns | 1 |
| Running-control underrun wait | About 0.558 seconds |
| Saved guided prefixes | 76, each 50 by 14 |
| Maximum forecast/executed command-prefix discrepancy | 0 |

Source: [final prefix audit](../outputs/rtc_improvement_20261024/rtc_prefix_audit_final.json).

This audit verifies the sent prefix against executed command targets over the inference interval. It does not show zero physical tracking error, prove disappearance of mechanical jitter, or establish a success rate. The run was headless; it does not validate GUI refresh performance.

Two early connection attempts failed during startup, and another network-interrupted attempt reached 800 steps before connection loss. The latter had no completed task outcome and is retained separately in [its connection/run log](../outputs/rtc_improvement_20261024/rtc-default-retry2.log). It is not counted as a completed policy trial in the suites above. The reported TUN/network problem was an operational condition, not evidence that the policy could never complete the task.

## New demonstrations and explicit review

The new [appearance/phase-aware collection](../data/banana_demos/20261024-184556-67ee29/summary.json) recorded eight automatically successful controlled demonstrations across four preassigned training, two validation, and two test layouts.

Assistant review retained the [phase overview](../data/banana_demos/20261024-184556-67ee29/phase_review.png), image-decode and physical-event evidence, and explicit per-episode decisions. All eight were accepted under the user's review delegation. Their `human_review` fields remain `pending`; assistant acceptance is not human review.

Two accepted correction segments come from one existing training layout and one actual learner run. They are dependent segments, not two additional independent layouts:

| Takeover | Recorded expert segment | Outcome and disposition |
| --- | --- | --- |
| Learner step 180 | 1,313 steps | Accepted after controlled completion |
| Learner step 500, first attempt | 3 steps | Premature contact loss; rejected and retained |
| Learner step 500, corrected closure handling | 703 steps | Accepted after controlled completion |

Sources: [step-180 metadata](../data/banana_demos/20261024-185103-99ca0a/episode_000/episode.json), [failed step-500 metadata](../data/banana_demos/20261024-185241-3a0f7c/episode_000/episode.json), and [corrected step-500 metadata](../data/banana_demos/20261024-185434-0055e0/episode_000/episode.json).

Both accepted segments reconstructed the recorded learner observed state with maximum error 0 before expert takeover. Full resulting physical snapshots and learner trajectory hashes were saved. The failed held-object attempt exposed a distinction between measured finger position and commanded closure; using the displaced measured fingers as a new target could loosen an existing grasp. The corrected expert retains the commanded closure for that supported takeover state.

This is a bounded DAgger-style correction collection step, not arbitrary failure recovery or an automatic online learning loop. No validation/test layout was relabeled as a training correction.

## Aggregation, export, and upload integrity

The new aggregate combines the original reviewed batch, eight new appearance/phase demonstrations, and two accepted correction segments. The failed correction was excluded from training while its original files remained available.

The [complete LeRobot manifest](../data/lerobot/20261024-185715-merged-ebdbe3/dataset_manifest.json) records:

| Split | Episodes/segments | Frames |
| --- | --- | --- |
| Train | 22 | 26,497 |
| Validation | 6 | 7,423 |
| Test | 6 | 7,483 |
| Total | 34 | 41,403 |

The training partition comprises 20 base layouts plus two dependent correction segments. The merge preserves source review lineage, source split assignments, and independent original files. Conversion retains 50 Hz alignment, 14-dimensional absolute action targets, and pixel-preserving PNG storage in the pinned LeRobot v2.1 format.

The training upload retained a [28-file SHA-256 integrity manifest](../outputs/rtc_improvement_20261024/upload-integrity.json). This records checked file content, not remote checkpoint authenticity or a permanent service state.

## Pre-continuation appearance validation

Before quality continuation, the v3 model was evaluated on the two newly preassigned appearance-validation layouts using synchronous PNG transport and 50-step execution. It completed 1/2: the failed layout grasped and lifted the banana but did not finish the placement.

Source: [v3 appearance-validation evaluation](../outputs/banana_evaluations/20261024-190822-d3e7e9/evaluation.json).

These two results form a small pre-continuation reference. They are not directly comparable to the asynchronous RTC suites above because the execution protocol and layouts differ.

## Fixed-normalization pilot: retained failure evidence

The controlled `banana_v4_quality1000` pilot was launched at about 19:10 with 1,000 planned updates, batch size 4, peak learning rate `2e-5`, original-pi0 LoRA plus the established projection training, and initialization from v3 step 1999. It retained the existing v3 action/state normalization and used the train-only group/episode/phase sampler.

The [downloaded training log](../outputs/rtc_improvement_20261024/banana_v4_quality1000.log) records losses in the hundreds; for example, step 800 reports about 336.38 and step 900 about 520.01. It reaches a logged step 975 and ends with an interruption during asynchronous checkpoint saving. This pilot was stopped and not promoted. The [interruption record](../outputs/rtc_improvement_20261024/v4-interruption.json) lists checkpoint directory `500` and an incomplete `999.orbax-checkpoint-tmp-4`; restoration of checkpoint 500 was not attempted. Logged progress alone does not establish a completely committed final checkpoint.

A [train-only normalization audit](../outputs/rtc_improvement_20261024/quality-normalization-audit.json), performed after the ordinary ALOHA action transform, identified nearly constant old action axes. Recovery motion in zero-based dimension 10 encountered an old standard deviation of about `2.735e-5`; its maximum absolute normalized target was about 280.17, while applying an action-standard-deviation floor of 0.01 reduced that maximum to about 0.7942. This provides a concrete coordinate-scaling explanation for part of the large loss. It is not proof that normalization accounts for every source of training error.

The original normalization is therefore not automatically compatible with all new recovery trajectories. The earlier recommendation to retain checkpoint statistics is conditional on that compatibility. The original pilot configuration remains in [banana_finetune_quality_v4.toml](../configs/banana_finetune_quality_v4.toml).

## Versioned normalization and replacement continuation

The [normalization report](../assets_quality_norm/pi0_banana_lora/local/pi0_banana_train/normalization_report.json) records the explicit response: floor transformed action standard deviations at 0.01 for dimensions 0 through 6 and 10. State statistics, action means, and quantiles remain unchanged. The source normalization, training dataset manifest, and new assets have recorded digests.

This changes the action coordinate system; initialization is not exactly function-preserving. Original pi0 uses z-score normalization here, so leaving quantiles unchanged does not cancel that scaling change. The new assets must accompany the newly trained checkpoint during evaluation and serving and must not be substituted into the old v3 deployment.

The replacement `banana_v5_quality_norm1000` candidate starts again from original v3 step 1999, not from the stopped v4 pilot. It retains the planned 1,000 updates, batch size 4, peak learning rate `2e-5`, seed 45, and the same grouped sampler, while selecting the versioned `assets_quality_norm` assets in [the current quality configuration](../configs/banana_finetune_quality.toml).

The replacement completed the 1,000-update budget with [exit code 0](../outputs/rtc_improvement_20261024/banana_v5_quality_norm1000.exit-code) and committed numeric checkpoints 500 and 999. The [result record](../outputs/rtc_improvement_20261024/banana_v5_quality_norm1000-result.json) retains source/configuration digests, environment versions, the clean pinned upstream status, and confirmation that checkpoint normalization values equal the intended versioned assets. Source and checkpoint normalization JSON byte hashes differ because serialization differs; the checked values match.

The final checkpoint is `banana_v5_quality_norm1000/999`. Training losses near the end were about 0.0094–0.0116; see the [training log](../outputs/rtc_improvement_20261024/banana_v5_quality_norm1000.log) and [launch settings](../outputs/rtc_improvement_20261024/banana_v5_quality_norm1000-launch.json). A successful training exit and lower loss do not establish improved closed-loop behavior, especially across changed normalization coordinates.

## Completed v5 step-999 validation

The fixed final checkpoint was evaluated using synchronous PNG/50-step execution and the optional RTC configuration on default-center plus all six preassigned validation layouts:

| Scene | Synchronous outcome | RTC outcome |
| --- | --- | --- |
| Default-center | Failed within 2,000 steps | Success |
| Original validation episode 004 | Success | Failed within 2,000 steps |
| Original validation episode 012 | Failed within 2,000 steps | Success |
| Original validation episode 019 | Success | Failed within 2,000 steps |
| Original validation episode 020 | Failed within 2,000 steps | Success |
| New appearance-validation episode 026 | Failed within 2,000 steps | Success |
| New appearance-validation episode 029 | Failed within 2,000 steps | Failed within 2,000 steps |

Sources: [completed v5 synchronous validation](../outputs/banana_evaluations/20261024-221520-bb1cb6/evaluation.json), 2/7 successful episodes, and [completed v5 RTC validation](../outputs/banana_evaluations/20261024-221810-907ec8/evaluation.json), 4/7 successful episodes. All seven RTC episodes recorded zero running-control inference underruns and zero associated wait seconds; this excludes startup acquisition.

The candidate is not promoted. On the two new appearance-validation layouts, the earlier v3 reference recorded 1/2 while v5 recorded 0/2 under synchronous execution and 1/2 with RTC. These are single stochastic rollouts, not a repeated paired-noise study. The differing failure locations and lack of repeated trials do not isolate a causal RTC improvement.

The already saved v5 step-500 checkpoint was also evaluated with RTC on the same seven-scene development-validation set. It completed 5/7: default-center and new appearance episode 026 failed; original validation episodes 004, 012, 019, 020 and new appearance episode 029 succeeded. All seven recorded zero running-control inference underruns. Source: [completed step-500 RTC validation](../outputs/banana_evaluations/20261024-222401-ea854c/evaluation.json).

Validation has been used for development and checkpoint comparison; it is not an untouched final test. Do not pool different execution protocols into one success percentage or present low training loss as improved task performance.

## Matched v3 comparison and frozen selection

The established v3 checkpoint was rechecked on the same seven-scene validation set:

| Execution | Completed successes | Recorded running-control blocking |
| --- | --- | --- |
| Synchronous PNG, 50-step chunks | 5/7 | 180 scheduled chunk waits, about 32.350 seconds |
| RTC, combined completed cases | 5/7 | 6 buffer underruns, about 19.189 seconds |

Sources: [v3 synchronous suite](../outputs/banana_evaluations/20261024-222856-0584f1/evaluation.json) and [combined v3 RTC cases](../outputs/rtc_improvement_20261024/v3-final-rtc-combined.json). Startup waits are excluded. Synchronous blocking is an ordinary consequence of that protocol; asynchronous underruns are buffer exhaustion. The measured totals are not a latency-controlled causal comparison or a real-time guarantee.

The synchronous successes were default-center, original episodes 004, 012, 019 and new appearance episode 026. Its failures were original episode 020 and new episode 029. RTC succeeded in default-center, original episodes 012 and 020, and both new appearance episodes 026/029; original episodes 004 and 019 failed.

The first [v3 RTC suite artifact](../outputs/banana_evaluations/20261024-223120-d4eedc/evaluation.json) contains only five completed cases. A sixth episode suffered connection loss and is retained as an [interrupted recording](../outputs/20261024-223445-674be9/summary.json), not a completed task failure. After transport restoration, the [two remaining preassigned layouts](../outputs/banana_evaluations/20261025-000141-5482b1/evaluation.json) completed successfully with zero underruns. The combined record includes exactly seven completed cases, preserves the interrupted attempt separately, and retains all six earlier underruns; it is not a zero-underrun suite.

The [selection record](../outputs/rtc_improvement_20261024/deployment-selection.json) was frozen at 00:04 on October 25 before this round's reserved-test run. It selects established `banana_v3_roi2000/1999` with generation-time RTC: it tied v3 synchronous and v5 step-500 validation counts, retained the successful default task, and recorded fewer running-control waits than synchronous execution. Small stochastic samples and different network conditions do not establish superiority.

[banana_recommended_pi0.toml](../configs/banana_recommended_pi0.toml) pins that checkpoint suffix and normalization digest. The v4 pilot and both v5 checkpoints remain unpromoted; the newer weights did not demonstrate a sufficient validation advantage. The continuation code and reviewed data remain useful artifacts, not evidence that the newly trained model improved.

This round's frozen evaluation uses six preassigned test layouts and reports default-center separately. Four older test layouts were already evaluated in the earlier v3 study; the two appearance-test layouts are new to this round. They are not all never-before-observed scenes.

## Frozen reserved-test outcome

The selected v3 RTC checkpoint and parameters were held fixed after the selection record. The [completed evaluation](../outputs/banana_evaluations/20261025-000416-eca6f4/evaluation.json) and [cohort summary](../outputs/rtc_improvement_20261024/frozen-test-summary.json) record:

| Cohort | Successes | Interpretation |
| --- | --- | --- |
| Six preassigned test layouts | 5/6 | This round's test result |
| Four historically evaluated test layouts | 4/4 | Part of the six; previously seen in earlier v3 evaluation |
| Two newly reserved appearance-test layouts | 1/2 | Part of the six; limited fresh appearance evidence |
| Default-center | 1/1 | Separate confirmation, not a held-out test layout |

The raw runner reports 6/7 across test plus default. Calling that a 6/7 held-out test success rate would mix the cohorts incorrectly.

The failed test case is merged source `episode_031`, layout `layout-dc5b7ded1b0e`. It did not confirm a grasped-and-lifted pinch within 2,000 steps; see [the retained failed rollout](../outputs/20261025-000628-cbbfa2/summary.json). The selected profile therefore still has a concrete appearance/layout failure and does not solve all grasp situations.

Across the six test layouts plus default, five running-control inference underruns totaled about 2.201 seconds. The run group is not zero-wait and does not establish a real-time guarantee. These counters exclude startup waits and are distinct from measured physical jitter.

The summary records `unchanged_after_test = true`: no parameters or checkpoint were tuned from these outcomes. Each layout received one stochastic completed rollout, so the rates are descriptive results with limited uncertainty evidence.

## Final native-GUI confirmation

The selected pinned v3 RTC profile completed the default banana task in an actual GLFW window with video recording enabled. Sources: [run summary](../outputs/20261025-000808-63bba3/summary.json), [timing and motion analysis](../outputs/rtc_improvement_20261024/final-gui-summary.json), and [recorded video](../outputs/20261025-000808-63bba3/episode_000.mp4).

| Metric | Recorded value |
| --- | --- |
| Outcome | Successful grasp and placement |
| Steps / simulation duration | 899 / 17.98 seconds |
| Recorded control wall duration | About 20.01 seconds |
| Native displayed frames | 861 |
| Cumulative native drawing cost | About 6.33 seconds |
| Startup inference wait | About 0.354 seconds |
| Running-control inference underruns | 0 |
| Stale chunks discarded | 0 |
| Maximum observed command-prefix error | 0 |
| Inference round trip: median / P95 / maximum | About 218 / 294 / 597 milliseconds |
| Control interval: median / P95 / maximum | About 20.29 / 31.24 / 43.60 milliseconds |
| Steps with deadline lateness above 2 milliseconds | 314 |
| Maximum deadline lateness | About 37.80 milliseconds |

The episode-wide display average is about 43 frames per wall-clock second, including startup. A requested 50 FPS and nominal 50 Hz simulation controller therefore did not establish constant achieved 50 FPS or uninterrupted real-time execution. This particular GUI run avoided inference buffer waits after startup; other recorded validation/test runs did not.

State-derived joint acceleration had P95 about 11.91 radians per second squared; commanded-target acceleration P95 was about 17.24. These are finite-difference estimates from the recorded joint samples, not independent accelerometer measurements. They describe this run and do not prove a reduction against unmatched earlier trajectories. Nonzero acceleration/jerk and failure cases remain; complete physical-jitter elimination is not established.

The completed artifacts support retaining the pinned established v3 RTC profile for use while preserving synchronous rollback and the experimental training candidates. The newer v5 weights remain unpromoted. No checkpoint or scheduling parameter was changed in response to the frozen-test outcomes. Further improvement requires targeted failure analysis and repeated matched motion/success measurements.

A [28-file final source digest record](../outputs/rtc_improvement_20261024/final-source-sha256.json) and [final source snapshot](../outputs/rtc_improvement_20261024/final_source_snapshot/) retain the final implementation/configuration separately from the earlier 25-file snapshot and exploratory evidence. They identify captured source content, not permanent remote-service availability.

## Interpretation limits

The recorded prefix correction addresses a concrete command mismatch. It does not isolate all causes of arm oscillation. The old and new models, synchronous and asynchronous execution, GUI and headless performance, and different layout sets must not be pooled into one success percentage.

Use validation for development and reserve test data for a frozen evaluation protocol. Preserve attempts with failures or connection errors and report their disposition. A more convincing smoothness claim requires repeated comparable runs and measured physical motion, alongside task success and latency.
