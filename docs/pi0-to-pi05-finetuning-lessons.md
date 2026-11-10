# Lessons from pi0 for the next pi0.5 fine-tuning project

Date: 2026-11-02. This is a retrospective and proposed preparation workflow, not an implemented pi0.5 port or authorization to train. The accompanying [Chinese research memo](../outputs/pi05_preparation_20261102/pi0_to_pi05_lessons_zh.md) presents the same practical lessons for project use.

## What this project established

The project connected Mac MuJoCo/ALOHA simulation and data preparation to Linux GPU training and inference. The latest relabeled training set has 20 complete demonstrations and two correction segments, totaling 23,859 frames. Frames, segments, independent layouts and repeated attempts are different counting units.

The final two-profile acceptance compared candidate999 plus simulated wrist damping2 against v6 plus damping1 on six existing validation layouts, twice per profile. Candidate completed 12/12 versus 10/12; typical physical jerk fell 16.4% and the worst peak 22.5%. The frozen 20% typical-jerk requirement failed. The user later selected the combination for operation. Preserve that distinction between an operational choice and a complete numerical acceptance pass. These are finite, repeatedly used validation scenes, not population reliability estimates. See [acceptance](final-candidate-acceptance-20261030.md) and [selected runtime](final-runtime.md).

The separate OOD suite achieved 31/50: position 10/20, yaw 12/20 and appearance 9/10. Eighteen of nineteen failures never lifted the object. There was no matched v6 OOD arm. The experiment identifies an early-grasp weakness but does not establish improved generalization over v6. See [OOD evidence](ood50-evaluation-20261029.md).

## Transfer the investigation method, not the final parameter values

| Lesson | Evidence from pi0 | Practice for the next project |
| --- | --- | --- |
| Validate the data contract before training | The pipeline required consistent images, pre-action states, timestamps and actually executed targets | Check camera order/encoding, action units/order, gripper convention, absolute versus delta targets, sampling period and episode boundaries through export, reader and serving transforms |
| Count independent coverage | Correction segments reuse training parents; the latest 22 segments are not 22 new independent layouts | Split by parent episode/layout before collection or correction; measure pose, orientation, approach and release coverage separately from frame count |
| Smooth teachers do not imply smooth policies | Continuous-transport relabels passed teacher gates; the next policy cohort completed more tasks but failed its motion objective | Review label quality, then perform an independent learned-policy comparison; never promote solely on expert plots or loss |
| Normalization is part of checkpoint identity | Nearly constant axes caused extreme normalized targets; a versioned action-standard-deviation floor was investigated | Audit transformed train-only distributions and normalization round trips; bind statistics to the exact model/data recipe and checkpoint |
| Compute feasibility must be measured | The selected pi0 run with a 100-step action horizon used microbatch8 × accumulation2 after real probes | Probe the chosen pi0.5 implementation, horizon, token length and trainable set; distinguish compilation, steady updates, checkpoint costs and memory headroom |
| Execution can prevent predicted actions from happening | ACT predict-100/execute-25 often placed release beyond the executed prefix | Compare prediction horizon, request cadence and actual executed suffix; test full-task outcomes after any execution change |
| Contact is distinct from command intent | Opening targets did not produce the requested jaw opening in box-contact branches | Log requested/submitted/executed targets, measured joints and contact state; separate policy, controller and geometry interventions |
| Physical smoothness and real-time continuity differ | A simulation-time video can omit inference waits; actual application intervals exposed stalls | Report physical derivatives, task progress, inference waits and actual action-application intervals separately |
| Improvements need controlled attribution | Candidate weights and doubled damping changed together | First compare models under one common controller; evaluate damping separately, or use a factorial design if attributing both effects |
| Preserve a usable experiment record | Early upload corruption, invalid replay branches and old documentation caused misleading status | Freeze inputs, retain negative results and version repairs; keep live state separate from dated receipts |

### Normalization details worth carrying forward

The pi0 quality-data audit found an old action-axis standard deviation near 2.735e-5 and normalized magnitude near280. A versioned 0.01 standard-deviation floor changed those coordinates; it was not a universally safe tuning constant. New data and pi0.5 transforms must be audited afresh. Retain existing statistics for a continuation only after checking their compatibility; a new-base adaptation may need dataset-specific statistics or compatible pretrained assets. Never copy the pi0 file blindly.

Also distinguish equality of JSON values from equality of serialized bytes. The pi0 source and checkpoint statistics had different byte hashes but a recorded numerical match. A serving pin based on bytes still requires the checkpoint-serialized asset. See [normalization history](rtc-quality-experiment.md).

### ACT was a diagnostic, not a data-quality certificate

ACT training completed 20,000 updates but full tasks remained 0/4 on training layouts and 0/12 on validation attempts. Of 24 learner-state queries, none predicted sufficient opening in the first 25 actions; 22 predicted it in the later 75. Full 100-action suffixes reached release in 6/6 selected states, yet complete success remained 0/6 in that short diagnostic window. The next project should test execution hypotheses in complete closed loops, not turn suffix release into task success. Use [corrected branches](act-release-diagnosis-20261029.md); the initial cached-dynamics branches were invalid.

## Pi0.5-specific migration checks

The local upstream source is pinned at `215abfb217dbac7d5f1273282331b9b1866c0479`. The official source pages below were checked on 2026-11-02; future work must pin its own chosen revision rather than assume that mutable `main` is identical.

In the inspected [model configuration](../repos/openpi/src/openpi/models/pi0_config.py), pi0.5 defaults to discrete state input and a longer prompt-token budget, and its action expert uses adaRMSNorm time conditioning. These are defaults, not universal task settings: the inspected `pi05_libero` configuration explicitly disables discrete state input, while the ALOHA example uses the pi0.5 default. Follow the chosen robot/checkpoint's concrete transforms. Official references: [model configuration](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0_config.py), [training configurations](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py), and [base-model workflow](https://github.com/Physical-Intelligence/openpi#fine-tuning-base-models-on-your-own-data).

| Current project component | Why it cannot simply be renamed |
| --- | --- |
| [banana_config.py](../server/banana_config.py) | Requires family `pi0`/`pi0_banana_lora` and constructs `pi05=False`; its projection filter names include `state_proj` and `action_time_mlp_*` |
| [Upstream model](../repos/openpi/src/openpi/models/pi0.py) | Pi0.5 uses `time_mlp_in/out` instead of the pi0 state/time-projection branch; inspect the actual parameter tree before choosing trainable modules |
| [serve_pi0.py](../server/serve_pi0.py) and [client](../src/pi0_repro/remote.py) | Explicit original-pi0 model-family checks reject a pi0.5 service; introduce a genuine separate model contract rather than mislabel metadata |
| [rtc_policy.py](../server/rtc_policy.py) | Explicitly rejects pi0.5 and PyTorch; prefix guidance needs a separately validated implementation, not removal of that check |
| [banana_model.py](../server/banana_model.py), [ROI adapter](../server/banana_roi.py), temporal loss and sampling extensions | Their local testing applies to this pi0 path; review masking, conditioning, camera treatment and gradient behavior before porting |
| A 100-step action horizon, ROI coordinates, 8×2 batch, 2× damping, RTC20 and 3Hz filtering | These are task/hardware/executor choices from pi0, not pi0.5 defaults or validated optima |

Use a compatible pi0.5 checkpoint/configuration pair, such as the official base workflow where appropriate; do not initialize pi0.5 by relabeling candidate999. Choose JAX or PyTorch explicitly, with the corresponding weight format and training path. Enumerate actual trainable parameter names/counts and check finite, nonzero gradients in the intended modules before a formal run. LoRA and full fine-tuning are separate resource/quality choices; neither success nor feasibility follows from this pi0 experiment.

## Proposed sequence for the next fine-tuning run

1. **Freeze the task and baseline.** Specify robot/controller, cameras, action semantics, success/timeout rules and data split. Preserve the selected pi0 workflow as a reference. Record model-only and system-level comparisons as different questions.
2. **Pin pi0.5 and prove minimal inference.** Use its intended checkpoint, tokenizer/state transform, normalization and framework. Validate one real observation's action shape, finite values and action interpretation before custom losses or RTC.
3. **Audit the exported data.** Sample whole episodes and boundary/release frames; compare reader output to source; verify image/state/action synchronization, padding masks and parent-level split separation. Use train-only statistics and preserve held-out identities.
4. **Run a disposable feasibility probe.** Measure actual batch/horizon/token shapes, trainable modules, finite losses/gradients, memory and steady update time. Confirm space for checkpoint retention plus a margin, and use separate temporary experiment names.
5. **Establish one simple fine-tuned baseline.** Use the selected upstream-compatible objective first. Choose an initial budget after the probe; inspect sparse early checkpoints with a predeclared rule rather than assuming more updates help. A tiny-set overfit can debug fitting, but does not prove closed-loop competence.
6. **Evaluate complete tasks.** Separate a training-layout diagnostic, validation selection and a held-out final test. Keep every scored failure. Freeze a common controller and executor for a model comparison; keep the existing pi0 production combination as a separate system reference if its controller differs.
7. **Change one subsystem at a time.** Let failure stages determine the next action: data coverage, model fit, scheduling, physical release or networking. Add weighting, horizon changes, RTC or damping only after a baseline and a specific hypothesis.
8. **Decide and package.** Apply predeclared practical thresholds, report sample limitations and any unmet condition, and record user-selected deployment separately from numerical acceptance. Provide one launcher, checkpoint/statistics pins, recording and an independent fallback path.

Suggested first milestones are interface correctness, feasible finite updates, and successful complete tasks on a few training layouts. They are debugging stages, not evidence of broad generalization. Do not promise that pi0.5, a particular episode count or another fixed 1,000 updates will improve this task.

## Required evidence bundle and stop conditions

Keep source revision/hash, environment/backend, dataset manifest and parent split, normalization artifact, trainable-parameter list, optimizer-update/microbatch accounting, probe results, immutable configuration, checkpoint completion metadata and matched evaluation logs/videos. Record command/reference versus actual state, RTC/prediction timestamps where relevant, contact diagnostics and wall-clock waits. Reserve a genuinely untouched test set for later claims; the six layouts reused throughout this pi0 work cannot become a fresh pi0.5 test set by renaming them.

Stop before formal training on model/checkpoint mismatch, non-finite or incorrect-shape data, normalization incompatibility, cross-episode labels/split leakage, an unexpected frozen/trainable parameter set, insufficient memory/storage, or an unverifiable checkpoint write. Diagnose data-transfer failures before interpreting them as model-training failures; the [IK training record](ik-expert-optimization-20261029.md) retains a corrupted-file probe failure and its verified repair.

Do not start additional training solely because offline loss decreased or a smooth teacher passed its own gates. Failure to lift should trigger pose/approach/coverage investigation; a learned late release should trigger execution/contact diagnosis; long actual control intervals require timing diagnosis, not automatic trajectory relabeling. These are investigation routes, not proven unique causes.

## Evidence retention

The [2026-10-31 cleanup](local-output-cleanup-20261031.md) removed unreferenced older raw runs. Some historical statements concern results whose raw arrays are no longer available; compact summaries cannot replace replay data. The complete latest 24-attempt acceptance was retained. For the next project, define retention before starting: keep the selected and fallback checkpoints with matching assets, all scored acceptance attempts, failures needed for diagnosis, and a manifest of any intentionally pruned artifacts. Do not accumulate hundreds of anonymous output folders as the only experiment index.
