# ACT baseline validation — 2026-10-29

## Scope — completed negative result

The [training plan](../outputs/act_baseline_20261029/plan.json) and [evaluation plan](../outputs/act_baseline_20261029/evaluation_plan.json) freeze an actual ACT baseline, separate from original pi0 and the earlier sparse inference-time Temporal Ensembling candidate. It uses unmodified official LeRobot ACTPolicy at the pinned commit recorded in the plan. ResNet18 starts from ImageNet visual weights; Transformer/CVAE parameters initialize randomly. This is not pi0 fine-tuning, and evaluation completed with zero task successes; this does not validate data quality or establish universal ACT ineffectiveness.

## Data and model interface

The baseline uses the existing 22 training episodes / 23,859 frames. All 22 Parquet hashes match the frozen relabeled candidate dataset. Actions are the original 14D absolute executed targets without openpi coordinate conversion. The original top-camera ROI is cropped and PIL-letterboxed to 224 pixels; visual normalization uses ImageNet statistics, while state/action means and standard deviations use training data only. Episode-end padding and target index zero must retain the intended action alignment.

[Training integration](../server/act_baseline.py), the separate [ACT service](../server/serve_act_baseline.py) and [evaluator](../scripts/evaluate_act_baseline.py) define this baseline's boundaries. Two unit checks passed for target-zero alignment/episode-boundary padding and equality of PNG-wire versus training preprocessing. These interface checks do not establish that the learned policy succeeds or that every dataset issue is absent.

## Frozen optimization and probe

Standard CVAE L1 reconstruction plus KL weight 10 is trained with batch 8, learning rate 1e-5, weight decay 1e-4, gradient clip 10, seed 54 and 20,000 updates. A real three-update GPU probe completed with approximately 1,047 MiB peak allocation and 0.0458 s median update time. The formal run restarted from the fresh seed and completed 20,000 updates in 1,387.49 seconds, separately from the probe. Final total loss was 0.0378531292, L1 0.02791755646 and KL 0.00099355727, with 1,047 MiB peak usage. The [training readback](../outputs/act_baseline_20261029/training_readback.json) and [eight-check binding audit](../outputs/act_baseline_20261029/training_audit.json) verify official ACT/configuration/normalization source hashes against the pinned local vendor and all 22 original Parquet hashes. Low training loss does not establish closed-loop task success.

ACT has no RTC or Temporal Ensembling here. It predicts chunk100 and requests a new chunk every 25 steps. The pi0 backend remains retained separately. ACT endpoint metadata has now been verified for the temporary evaluation route described below; this dated check does not guarantee future availability.

## Predeclared evaluation and retained transport attempts

Only final update20,000 will be evaluated: four first training layouts once each for memorization diagnosis, followed by six validation layouts twice each, sixteen attempts total. Training-layout results are not held-out performance. There is no checkpoint selection after validation, outcome-hidden rerun or concealed OOM/failure retry.

Evaluation retains the original damping1 plant and actuator limits only, without pi0 target filtering. Simulation dt is 50 Hz, but execution uses synchronous inference waiting and is not wall-clock real-time paced. Consequently this is not a fair pi0 speed comparison or a controlled attribution of algorithm superiority: initialization, objective, execution and compute budget differ. All failures and waits must remain recorded. Neither training dispatch nor a small successful probe establishes ACT task validation, learned-policy generalization or proof that the data are good.


## Transport adjustment and retained failures

The initial separate ACT endpoint was rejected by SSH PermitOpen. The first evaluator failed its first-case handshake before any observation or policy action; its `evaluation_transport_attempt00` ledger/error/reset are retained. A second launch was blocked by the existing-ledger guard. That guard overwrote the original log file; the original traceback remains in tool records and the ledger error, so the first log file must not be represented as wholly unchanged.

Without changing SSH permissions, evaluation temporarily reused the already permitted sidecar route for ACT. The pi0 model backend remained retained, and the prior image-proxy command was recorded in cloud `port_transition.json` for restoration afterward. Endpoint/runtime-port amendments are infrastructure changes, not model-protocol, training, scene or action-setting changes. The sixteen attempts completed without additional tuning.

## Final results and restoration

The [completion](../outputs/act_baseline_20261029/completion.json), [ledger](../outputs/act_baseline_20261029/evaluation.json) and [167-check audit](../outputs/act_baseline_20261029/evaluation_audit.json) bind training-layout success 0/4 and validation success 0/12. Validation comprises six unique layouts with two deterministic repeats, not twelve independent layouts. The [failure-stage audit](../outputs/act_baseline_20261029/failure_stages.json) records nine grasp/lift attempts: all entered the box and ended inside but unreleased (seven supported, two unsupported). Seven others never lifted. These flags locate progress, not root cause. This 20k-update baseline does not demonstrate that the data are high quality, that the model learned nothing or that ACT generally fails.

The independent [post-hoc source manifest](../outputs/act_baseline_20261029/posthoc_source_manifest.json) binds prediction diagnostics. [Final cloud receipts](../outputs/act_baseline_20261029/final_cloud_receipts.json) record 66 fixed training-teacher states: public select_action versus actual serving formula error zero, normalization round-trip maximum 1.192e-7 and prior normalized L1 mean 0.042883. This differs from posterior training loss and is neither a validation metric nor additional policy attempts.

The receipts verify ACT endpoint shutdown, exact original image-proxy command restoration and v6 metadata. [Tunnel cleanup](../outputs/act_baseline_20261029/tunnel_cleanup.json) confirms the owned ACT tunnel exited and endpoint closed. These dated checks do not guarantee future availability. Default pi0 remains unchanged.

The [media manifest](../outputs/act_baseline_20261029/media/manifest.json) binds all sixteen videos and two start/lift/final sheets. [Primary review](../outputs/act_baseline_20261029/primary_review.json) covers both sheets and the [training-loss plot](../outputs/act_baseline_20261029/training_loss.png), not complete continuous playback.

## All original attempts

| Case | Split | Layout / repeat | Outcome | Video |
| --- | --- | --- | --- | --- |
| 0 | train | episode_000 / 1 | Failure | [Video](../outputs/act_baseline_20261029/evaluation_00/episode.mp4) |
| 1 | train | episode_001 / 1 | Failure | [Video](../outputs/act_baseline_20261029/evaluation_01/episode.mp4) |
| 2 | train | episode_002 / 1 | Failure | [Video](../outputs/act_baseline_20261029/evaluation_02/episode.mp4) |
| 3 | train | episode_003 / 1 | Failure | [Video](../outputs/act_baseline_20261029/evaluation_03/episode.mp4) |
| 4 | validation | episode_004 / 1 | Failure | [Video](../outputs/act_baseline_20261029/evaluation_04/episode.mp4) |
| 5 | validation | episode_012 / 1 | Failure | [Video](../outputs/act_baseline_20261029/evaluation_05/episode.mp4) |
| 6 | validation | episode_019 / 1 | Failure | [Video](../outputs/act_baseline_20261029/evaluation_06/episode.mp4) |
| 7 | validation | episode_020 / 1 | Failure | [Video](../outputs/act_baseline_20261029/evaluation_07/episode.mp4) |
| 8 | validation | episode_026 / 1 | Failure | [Video](../outputs/act_baseline_20261029/evaluation_08/episode.mp4) |
| 9 | validation | episode_029 / 1 | Failure | [Video](../outputs/act_baseline_20261029/evaluation_09/episode.mp4) |
| 10 | validation | episode_004 / 2 | Failure | [Video](../outputs/act_baseline_20261029/evaluation_10/episode.mp4) |
| 11 | validation | episode_012 / 2 | Failure | [Video](../outputs/act_baseline_20261029/evaluation_11/episode.mp4) |
| 12 | validation | episode_019 / 2 | Failure | [Video](../outputs/act_baseline_20261029/evaluation_12/episode.mp4) |
| 13 | validation | episode_020 / 2 | Failure | [Video](../outputs/act_baseline_20261029/evaluation_13/episode.mp4) |
| 14 | validation | episode_026 / 2 | Failure | [Video](../outputs/act_baseline_20261029/evaluation_14/episode.mp4) |
| 15 | validation | episode_029 / 2 | Failure | [Video](../outputs/act_baseline_20261029/evaluation_15/episode.mp4) |
