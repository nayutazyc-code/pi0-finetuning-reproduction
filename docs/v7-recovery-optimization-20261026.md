# Original-pi0 v7 recovery-data optimization — 2026-10-26

## Scope and status

Stage 3 prepares targeted supervised corrections after cadence and jerk-reference screens did not establish a general solution to grasp/placement hesitation. This is train-only offline expert labeling, not online reinforcement learning or a policy-evaluation fallback. The [correction ledger](../outputs/v6_recovery_20261026/corrections.json) records four reviewed corrections merged into a new raw batch. LeRobot export and deployment are complete, and the separately guarded v7 continuation was verified in its training stage after a successful real-data probe. The 1,000-update continuation completed and final checkpoint `999` is committed. All 16 model-comparison attempts are complete. The recovery continuation did not establish a stable gain; v7 is not promoted, and v6 serving was restored and its metadata verified. The v7 checkpoint remains available for research.

## Learner states and split boundary

Two actual-model learner rollouts came from preassigned training sources `episode_000` and `episode_001` in the existing merged dataset. Both learner episodes succeeded; they are not failed rollouts. Their near-grasp and held-near-box states supplied useful correction takeovers despite eventual success. No validation/test state was collected for correction labels.

| Training source | Learner recording | Correction phase | Takeover step | New collection |
| --- | --- | --- | --- | --- |
| episode_000 | 20261026-140933-48ec6b | Grasp | 370 | 20261026-141541-50ddab |
| episode_000 | 20261026-140933-48ec6b | Place | 736 | 20261026-141619-e6ec1b |
| episode_001 | 20261026-140958-395218 | Grasp | 365 | 20261026-141647-dfd707 |
| episode_001 | 20261026-140958-395218 | Place | 794 | 20261026-141726-9276b9 |

Replay reconstructed the source takeover robot state with maximum error zero in all four segments. Subsequent expert actions supply the supervised labels; preceding learner commands reconstruct the state and do not become expert labels. The four segments add 3,150 frames and remain dependent corrections from two existing training layouts. They do not increase the dataset's 20 independent base training layouts.

## Recovery-specific expert and review

The opt-in [recovery expert configuration](../configs/banana_v6_recovery_expert.toml) sets `direct_recovery_grasp = true`. The ordinary expert default remains false, and the separate recovery-mode flag is supplied only during collection with recovery metadata. For a resting banana and sufficiently close/aligned active site (distance below 0.05 m, rotation error below 0.35 rad), the recovery expert skips the initial opening/hover and proceeds through descent, closure, and lift. If the replayed banana is physically held, it retains closure while placing. This does not add a runtime phase rule, grasp trigger, or fallback to pi0 evaluation.

All four correction episodes reached the unchanged physical task-success rule and passed structural checks. Under the user's explicit delegation, the assistant inspected [review frames](../outputs/v6_recovery_20261026/review_frames.jpg), phase views and terminal orientation evidence, then explicitly accepted each segment through the review API. Automated success alone was not treated as review. Assistant acceptance is not relabeled as human review.

## Completed merge and export

The new raw batch is `20261026-142111-merged-e54349`, recorded as `reviewed_and_merged` in the ledger. Original data, failed histories, split membership, source mappings, and dependent-segment lineage remain preserved. The [export log](../outputs/v6_recovery_20261026/export.log) records the separate data-environment conversion. The complete export at `data/lerobot/20261026-142111-merged-e54349` contains 26 training episodes/segments and 29,647 frames: the previous 26,497 plus 3,150 correction frames. Validation and test retain six episodes each. The preparation script checks that their original trajectory/layout lineage hashes are unchanged.

## Guarded continuation preparation and launch

The [preparation script](../scripts/prepare_v7_recovery.py) writes the dedicated [v7 training configuration](../configs/banana_finetune_v7_recovery.toml) and sampler. The [preparation record](../outputs/v6_recovery_20261026/training_preparation.json) is an artifact of configuration generation; its `prepared_not_launched` status must not be interpreted as the subsequent remote job's live state.

The candidate initializes from committed v6 `banana_v6_long100_accum8000/7999`, with fresh optimizer state, horizon 100, microbatch 8 and accumulation 2 (effective batch 16), and 1,000 optimizer updates. Peak learning rate is `5e-6`, warmup is 100 updates, decay target `1e-6`, and seed 47. The sampler preserves the prior sampler's relative weights within 80% probability mass, reserving 20% for the four new segments equally. Within new segments, descend/close/lower/release frames receive relative weight 2. These are stochastic target probabilities, not guaranteed realized counts.

Normalization is copied byte-for-byte on the cloud from the committed v6 asset with SHA-256 `6c6e7a92643c76996bf15ba9fc3d3e346dd6bc3d8697712118d9fdb49dedf269`. An older local normalization file with a different byte digest was not substituted. This round does not intentionally change the action coordinate system again.

The deployment used the user's existing authenticated SSH session. The [deployment helper](../scripts/deploy_v7_recovery.sh) and [remote supervisor](../server/recovery_train_job.py) preserve a separate experiment and guard initializer commitment, normalization bytes, train frame/split counts, and sampler/manifest binding before stopping old serving. The supervisor then requires a finite three-update real-data probe, action shape `[16, 100, 32]`, and at least 2,048 MiB memory headroom before starting continuation. Failure paths attempt v6 service restoration; an attempted restore is not proof that serving recovered.

The actual probe completed three finite updates, with the expected tensor shape and sampled peak GPU use 16,879 MiB out of 24,564 MiB. The [training status](../outputs/v6_recovery_20261026/training_status.json) subsequently verified `trained_pending_validation`, at least 1,000 completed optimizer updates, and a committed final checkpoint. Total elapsed time recorded for probe, compilation, and training was approximately 17 min 52 s. The last logged update index is 975, not a distinct final-checkpoint loss, and no nonfinite metric lines were found. The short probe itself remains separate from the full continuation.

The [checkpoint verification](../outputs/v6_recovery_20261026/checkpoint_verification.json) confirms committed parameters and normalization values matching initialization, with the expected normalization byte digest. A fresh metadata handshake declared original pi0, horizon 100, v7 step `999`, the matching norm, and RTC adapter digest `94d7c70f6d52accef0d0f7210c7153ee6be7172c6b9ecc2c5f79055b3e33835a`. These are checkpoint/serving compatibility checks, not policy-quality outcomes.

The [real-reader audit](../outputs/v6_recovery_20261026/reader_h100_audit.json) checked all four new segments through the pinned LeRobot v2.1 reader: each supplies action shape `[100, 14]`, exact initial actions/states, zero initial-image pixel error, and final-window padding that repeats the episode's last action without crossing into another episode. Segment lengths were 965, 599, 978, and 608 frames.

The [deployment-wrapper audit](../outputs/v6_recovery_20261026/deployment_wrapper_audit.json) retains a genuine infrastructure issue: the local script was edited while its earlier SSH launch was waiting, and resumed shell parsing failed with exit code 2 after the independent remote training had succeeded. That wrapper failure is not reported as a failed training run or silently discarded. The repaired future launch uses a detached Python subprocess, script syntax was checked, and the final checkpoint was verified independently. The [read-only status helper](../server/recovery_train_status.py) exposes aggregate job evidence without requiring normalization-content export.

## Separate reference evaluation and outcome boundary

A [fresh v6 reference comparison](../outputs/v7_recovery_baseline_20261026/comparison.json) records four headless cases in order: default-center success, `episode_020` failure without a confirmed lifted pinch, default-center success, and `episode_026` success. These observations are a contemporary baseline, not a general reliability estimate.

Only the final v7 `999` checkpoint was evaluated in the identical original case order and the same cadence15/bridge20/3 Hz/RTC execution profile, without the jerk backend. Model noise remains unpaired. Candidate performance must be compared with task completion, dwell/progress, physical motion, and timing; the baseline's failure remains in the record. Preserve v6 selection if results are mixed or regress. The [candidate ledger](../outputs/v7_recovery_candidate_20261026/comparison.json) records all four completed original attempts. The original baseline/candidate comparison is retained below, with default-center confirmations separate from validation layouts.

| Original case order | v6 outcome / simulated duration | v7 outcome / simulated duration |
| --- | --- | --- |
| Default-center confirmation 1 | Success / 27.40 s | Success / 25.54 s |
| Validation episode_020 | Failure / 40.00 s; no confirmed lifted pinch | Success / 37.16 s |
| Default-center confirmation 2 | Success / 27.82 s | Success / 24.66 s |
| Validation episode_026 | Success / 30.48 s | Success / 33.12 s |

The original comparison observed three completions in four v6 cases and four in four v7 cases, but the 37.16 s placement trajectory remained slow and v7's `episode_026` completion was longer. Completion does not establish resolved hesitation. These small, unpaired observations do not establish a reliable advantage.

## Disclosed validation extension

The unresolved placement lag justified extending evaluation to the remaining four preassigned validation source IDs, `episode_004`, `episode_012`, `episode_019`, and `episode_029`, selected from the split mapping before those extra attempts. This expansion followed inspection of the original four cases and is disclosed rather than presented as part of a wholly predeclared eight-case experiment. Execution parameters and final checkpoint were not retuned, no test data were used, and earlier cadence/jerk outcomes are not pooled into this comparison.

The [v7 extension ledger](../outputs/v7_recovery_extra_validation_20261026/comparison.json) records:

| Additional validation layout | v7 outcome / simulated duration |
| --- | --- |
| episode_004 | Failure / 40.00 s; no confirmed lifted pinch |
| episode_012 | Success / 25.76 s |
| episode_019 | Success / 22.28 s |
| episode_029 | Success / 24.90 s |

V7 therefore completed five of six validation-layout attempts, with its two default confirmations separately both successful. These are six layouts with one sample each, not a population success estimate. The matching v6 extension also completed all four layouts successfully. The chronology is v6 original four, v7 original four, v7 additional four, then v6 additional four; model noise and network histories remain unpaired. No checkpoints were selected after inspecting these outcomes; only the predeclared final `999` is evaluated. Training completion and successful metadata checks establish checkpoint availability, not improved policy behavior or default promotion.

## Completed comparison and service decision

The [combined model comparison](../outputs/v6_recovery_20261026/model_comparison.json) preserves all 16 attempts, including the [v6 extension](../outputs/v6_recovery_extra_validation_20261026/comparison.json). Both models completed their two default-center confirmations and five of six distinct validation-layout attempts. These blocks remain separate, and all observations use unpaired policy noise and network histories.

| Validation layout | v6 outcome / simulated duration | v7 outcome / simulated duration |
| --- | --- | --- |
| episode_004 | Success / 21.66 s | Failure / 40.00 s; no confirmed lifted pinch |
| episode_012 | Success / 21.38 s | Success / 25.76 s |
| episode_019 | Success / 24.52 s | Success / 22.28 s |
| episode_020 | Failure / 40.00 s; no confirmed lifted pinch | Success / 37.16 s |
| episode_026 | Success / 30.48 s | Success / 33.12 s |
| episode_029 | Success / 21.00 s | Success / 24.90 s |

| Phase / model | Observed completions | Median of trial physical joint-jerk p95, rad/s³ | Worst physical joint jerk, rad/s³ |
| --- | --- | --- | --- |
| Default confirmation / v6 | 2/2 | 15.98 | 755.49 |
| Default confirmation / v7 | 2/2 | 15.14 | 1727.22 |
| Validation / v6 | 5/6 | 14.87 | 1416.23 |
| Validation / v7 | 5/6 | 15.31 | 1077.40 |

All groups had zero running inference underruns. Physical jerk is measured from finite differences at the simulation period, with group medians taken over each trial's p95. Worst spikes favor different models in the default and validation blocks. Among the four validation layouts where both models succeeded, v7 took longer on three; only `episode_019` was faster. The failed layout changed from `episode_020` to `episode_004`, without increasing the observed validation completion count. The successful 37.16 s v7 trajectory remains evidence of unresolved delay.

These small, unpaired results do not support a stable recovery-training advantage, solved hesitation, or improved reliability. V7 is not promoted. V6 serving was restored and metadata verified; the v7 checkpoint and all preparation, training, failed episodes, and deployment-wrapper evidence are retained for further research. No test-partition evaluation was performed.

The [final cross-experiment audit](../outputs/v6_rtc_strength_20261026/final_consistency_audit.json) includes these 16 model-comparison trials and the later six RTC-cap attempts. It verifies stable application source, checkpoint/normalization pins, zero raw/timestamp/command-prefix errors, and no running inference underruns. This supports comparison integrity without changing the negative v7 promotion decision.
