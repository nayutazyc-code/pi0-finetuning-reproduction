# Original-pi0 long-chunk training

## Scope and selection boundary

This workflow continues original pi0 with longer action chunks and sequential microbatch gradient accumulation on one Linux CUDA GPU. It uses project-local adapters without changing the pinned upstream checkout. Mac simulation remains a separate environment. The selected v3 deployment remains the comparison baseline; initializing a new candidate from v5 does not promote v5 or the new weights. Dated measurements and launch outcomes belong to the [training record](long-chunk-training-20261025.md).

The [long100 configuration](../configs/banana_finetune_long100.toml) requests an action horizon of 100, 8,000 optimizer updates, and effective batch 16. At the existing 50 Hz control period, a 100-action prediction covers two simulated seconds. Prediction length does not determine request cadence, actual executed suffix length, internal flow-integration count, or number of training updates. A longer chunk can still hesitate, become stale, or fail the task.

The candidate initializes model parameters from v5 step `999` and retains its matching `assets_quality_norm` normalization rather than changing action coordinates again. It starts fresh optimizer state unless explicitly resuming the same experiment. Original-pi0 LoRA and the configured state/action/time projections remain trainable under the weighted flow-matching objective. This is supervised continuation, not reinforcement learning or strictly adapter-only training.

## Batch and update semantics

In the training TOML, `batch_size` is the microbatch size and `gradient_accumulation_steps` is the number of sequential microbatches per optimizer update. [Configuration construction](../server/banana_config.py) passes their product as the loader's effective batch size. For example, microbatch 2 with accumulation 8 forms effective batch 16. These provisional values must be replaced by the actual GPU selection, not treated as a measured optimum.

The [accumulation adapter](../server/banana_accumulation.py) slices the effective loader batch into equal microbatches and evaluates them sequentially with JAX `lax.fori_loop`. Each microbatch sees the same model weights and receives its own step/index-derived random key. It accumulates gradients in float32, averages gradients and loss, then calls the optimizer once. Clipping and AdamW operate on the averaged gradient; optional EMA updates once afterward. It does not clip or update weights independently after each microbatch. The accumulated path supports exactly one GPU/process. Accumulation 1 retains the upstream train step.

`TrainState.step`, learning-rate warmup/decay, log intervals, save intervals, and the configured training budget count optimizer updates. Thus 8,000 updates at effective batch 16 expose 128,000 sampled training anchors and 64,000 microbatches if microbatch 2/accumulation 8 is selected. With 26,497 train frames, 128,000 anchors are approximately 4.83 frame-count equivalents, not 4.83 deterministic epochs: the phase/group sampler samples with replacement, windows overlap, and episode membership is not uniform.

## Data and normalization provenance

The candidate uses the existing merged dataset `20261024-185715-merged-ebdbe3`, restricted to `local/pi0_banana_train`. Its 22 train episodes/segments contain 26,497 frames; two correction segments are dependent on an existing training layout, so these are not 22 independent layouts. Preserve raw data, review provenance, failures, and preassigned validation/test membership. The launcher checks completion, train-only source mapping, normalization availability, and initialization parameters before starting optimization.

The loader requests the model's configured action horizon from the existing per-frame absolute action data; longer targets do not require relabeling old dataset metadata as if historical runs already used horizon 100. Preserve the manifest's recorded `action_horizon_at_training = 50`. Inspect actual probe action shape and episode-boundary behavior before accepting the long-horizon data path. Do not recompute normalization on validation/test data or interchange v3 and v5 assets.

## Inspect and measure before the full run

Use the existing Linux openpi interpreter. Set `PI0_ROOT` to the existing deployment root without writing credentials or private connection settings into documents. Inspect the candidate:

```sh
"$PI0_ROOT/openpi/.venv/bin/python" "$PI0_ROOT/server/banana_finetune.py" describe \
  --root "$PI0_ROOT" --config "$PI0_ROOT/configs/banana_finetune_long100.toml" \
  --exp-name banana_long100_candidate
```

The memory sweep uses independent processes for microbatch 1/2/4/8, with accumulation 16/8/4/2 respectively to keep effective batch 16 fixed. Give each its own versioned probe config, experiment name, JSON, and captured process exit/log. Pause inference serving deliberately if needed to free GPU memory, then verify actual memory availability; historical GPU capacity and usage are not current evidence.

For each probe config:

```sh
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
"$PI0_ROOT/openpi/.venv/bin/python" "$PI0_ROOT/server/banana_finetune.py" probe \
  --root "$PI0_ROOT" --config "$PI0_PROBE_CONFIG" \
  --exp-name "$PI0_PROBE_NAME" --probe-updates 3 --probe-output "$PI0_PROBE_OUTPUT"
```

The [probe](../server/banana_probe.py) loads real data and runs real gradient/optimizer updates, checking finite metrics and the actual action tensor shape. These disposable updates are not saved as a trained checkpoint. It polls whole-device GPU memory every 0.2 s and reports sampled peak, total memory, update durations, and JAX allocator statistics. Sampled peaks can miss brief excursions and include other processes; allocator statistics are a separate measurement. The first update includes compilation; the reported steady estimate is the median of later updates, with a three-update probe providing only two post-warmup measurements. Its projected full-run hours exclude later variability and checkpoint overhead.

Select a feasible candidate using measured memory, finite updates, and post-warmup throughput, retaining at least 2,048 MiB below reported total GPU capacity. Preserve failed/OOM attempts and missing measurements; do not turn a killed process into a successful result or assume the largest microbatch is fastest. A short pass is not a guarantee that an 8,000-update run will remain within budget.

## Checkpoint storage capacity

GPU headroom and checkpoint-disk headroom are separate prerequisites. Before sequential training arms, inspect free space on the actual checkpoint volume and estimate peak retained checkpoints plus temporary save files using measured checkpoint sizes. Declare per-arm `save_interval` and `keep_period`; equal save intervals do not imply equal retention. Keep original plans/configuration hashes and explicitly record any operational retention amendment before the affected run starts, without silently deleting preexisting weights. A retention-only change does not alter the gradient calculation, but the full configuration files are no longer identical except for the objective. Estimates are not guarantees against later disk growth or checkpoint-size changes.

## Upload readiness and interrupted deployment

For the motion-first pilot, [deployment](../scripts/deploy_motion_training.sh), [read-only status](../server/motion_train_status.py), and [task reconnection](../scripts/reconnect_motion_cloud.sh) are separate operations. A reusable SSH master can authenticate while real remote commands or upload throughput remain unusable. The reconnect helper validates task ownership and requires a successful remote round trip before recording a replacement socket; it does not upload data, launch training, delete files, or alter VSCode forwarding.

After an interrupted upload, retain partial data and check the remote supervisor/status before resuming the same digest-bound plan. Do not create a duplicate job or count copied base Parquet files as completion of a larger new export. Distinguish authentication, upload start/completion, supervisor dispatch, GPU probe, optimizer updates, checkpoint commitment, and evaluation in evidence. An upload exit before dispatch is not a model/training failure. Current free-space receipts remain resource snapshots; the actual job must still pass GPU and checkpoint-volume guards. See [the dated motion-first receipt](motion-first-optimization-20261026.md) for the interrupted attempt and prepared evaluation boundary.

The optional [dedicated-key setup](../scripts/setup_motion_ssh_key.sh) requires the user's explicit `--install-key` choice because it adds persistent remote public-key authorization. It creates a dedicated passphrase-free local key with private-file mode 0600, preserves/backups existing authorized entries, validates algorithm/blob identity independently of comment text, and restricts forwarding to the intended loopback policy/sidecar destinations while retaining remote commands. It does not modify SSH configuration/VSCode, upload a dataset, or start training. The reconnect helper may consume an identity-file path without reading or logging private-key contents. Verify installation and a real remote command separately; key authentication does not guarantee stable bulk transfer. Never infer training start from successful key setup or a small in-memory network probe.

### Separate early checks from the canonical training gate

A small in-memory upload verifies authentication and bounded transfer only, not sustained dataset upload. The optional [key transport wrapper](../scripts/run_motion_deployment_with_key.py) retries only declared network exits, with a fresh remote status guard before every attempt; unknown status fails rather than being treated as a safe new job. It leaves the bound training sources/configuration unchanged.

Action/state-only Parquet projections may be checked for exact column equality and aggregate normalization compatibility before image transfer finishes. Keep them isolated from the canonical dataset: a successful projection audit is not a full reader/image check. Likewise, a disposable shape probe using an existing dataset but the new horizon/batch/loss settings is a computation-path diagnostic, not a passed new-data probe or saved training. The actual supervisor must still validate canonical upload completion, reader masks/images, normalization, sampler binding, and its real-data GPU probe before full optimization. Report which data each check used and keep those milestones separate.

## Train and resume

After recording the measured batch selection in a dedicated candidate configuration, launch under the existing persistent Linux session:

```sh
bash "$PI0_ROOT/server/run_banana_train.sh" banana_long100_candidate \
  --config "$PI0_ROOT/configs/banana_finetune_long100.toml"
```

Keep the experiment distinct from v3/v5 and other probes. The launcher records dataset/normalization digests, microbatch, accumulation, effective batch, and optimizer-update semantics. Preserve the launch JSON, full log, exit code, and committed checkpoints. A step logged every 25 updates is not a distinct final-checkpoint loss, and a planned budget is not completed training.

To resume an interrupted experiment, use the same experiment/configuration and `--resume`, relying on its committed checkpoint and restored training state. Do not substitute fresh initialization for resume, change horizon/assets/batch silently, or overwrite the earlier checkpoint lineage. Confirm the restored update count before reporting progress. The candidate's original initialization checkpoint is a model-parameter source, not restored optimizer history.

## Optional expert-displacement objective and matched training

The project-local [temporal objective](../server/banana_temporal.py) supervises adjacent displacements of the predicted clean action sequence against expert displacements. This is training-time expert-difference supervision, separate from inference-time Temporal Ensembling and not a penalty that pushes all velocities toward zero. Original-pi0 flow matching retains its existing target and weighting; the optional term is added with a configured coefficient.

For the existing interpolation $x_t=t\epsilon+(1-t)a$ and predicted flow $v_\theta$, the clean estimate is $\hat a=x_t-tv_\theta$. The auxiliary loss is the mean squared residual between $\hat a_{k+1}-\hat a_k$ and $a_{k+1}-a_k$ on the 12 native arm axes. Normalization standard deviations plus the serving/training epsilon restore native radians; action means cancel. ALOHA sign changes do not alter the squared residual. There is no division by the control period: this measures displacement error in rad² rather than velocity or jerk error. Grippers and padded model axes are excluded, while a genuinely static expert pose remains a valid target.

A constant per-chunk joint offset cancels from adjacent differences, so the auxiliary term alone does not anchor absolute positions or explicitly align independently replanned chunks. Flow matching still supplies position-related supervision. This auxiliary objective is not by itself a solution to grasp/placement stalls or chunk-boundary continuity.

Interpret the auxiliary coefficient together with native-unit scales and measured flow/auxiliary components. A coefficient of 0.1 does not mean a 10% loss contribution or guarantee a large model effect. Report the weighted term and its ratio separately; scalar loss ratios alone do not measure relative gradient influence. Keep the declared objective settings fixed for the matched evaluation rather than selecting them after closed-loop outcomes.

[Training-only observation/data wrappers](../server/banana_temporal_data.py) attach an explicit per-episode action-validity mask without changing upstream source. An adjacent pair contributes only when both samples are valid; repeated terminal padding does not become a supervised zero-displacement segment. Validity is not inferred from zero or equal-valued actions. Each example averages over its valid pairs and arm axes; no valid pair yields zero auxiliary contribution. Both control and temporal arms use this same masked loader, loss component reporting, and training pipeline, with auxiliary weights 0 and 0.1 respectively. Disabled auxiliary weighting is the control, not a switch to different training data or a different FM target.

The [v8 matched experiment](v8-temporal-training-20261026.md) fixes microbatch 8 times accumulation 4, effective batch 32, and 3,000 optimizer updates per arm from the same v6 initializer and exact normalization. This is 96,000 sampled anchors per arm; microbatch iterations do not increment the optimizer schedule independently. The two runs are sequential on the 24 GB-class GPU, so inference must pause while they occupy the device. GPU preflight/probe completion, finite training metrics, committed final checkpoints, and serving restoration must each be verified separately. A planned restart after completion or failure is not proof of restored service. Only the predeclared final checkpoint `2999` from each arm is intended for later matched-execution validation; lower combined training loss does not select a winner.

For this matched workflow, [deployment](../scripts/deploy_v8_temporal.sh) accepts the user's existing SSH alias as an argument and reuses the authentication-only helper without storing passwords or creating keys. The supervisor requires reader/config/source checks and successful probes for both arms, including `[32, 100, 32]` action shape and memory headroom, before launching either full run. Use the [read-only status helper](../server/temporal_train_status.py) to distinguish probing, training, checkpoint commitment, and attempted/verified v6 restoration. The local [status wrapper](../scripts/check_v8_training.sh) accepts the authenticated SSH alias, validates returned JSON, and atomically updates the local snapshot only on success; failed connections preserve the last successful observation. A saved observation is not continuous proof of availability. Supply actual connection details privately at invocation time rather than embedding them in maintained documentation.

## Optional near-term flow weighting

The [near-term helper](../server/banana_temporal.py) can reweight the existing per-step FM loss toward upcoming valid actions without changing its target or prediction horizon. `near_term_steps = 0` and `near_term_weight = 1` retain legacy weighting; a prospective 30-step/factor-2 candidate is separately described in [motion-first preparation](motion-first-optimization-20261026.md). Explicit episode-tail masks zero padded actions, and per-example weights are normalized to mean one across the horizon, preserving nominal rather than automatically doubling overall loss scale. Static real actions are not padding. This is training-loss weighting, not a phase controller, temporal ensemble, or zero-velocity penalty. Local implementation checks do not establish GPU compatibility, a deployed dataset, or improved trained behavior.

### Prepared v9 motion pilot

**Prepared, not deployed: no pilot training has run and no result exists.** A second prepared candidate bundles the motion-first data, sampler, and loss weighting. [banana_finetune_v9_motion.toml](../configs/banana_finetune_v9_motion.toml) is bound by digest in [training_plan.json](../outputs/motion_first_20261026/training_plan.json), whose plan records `status: prepared_not_deployed`, experiment `banana_v9_motion1500`, and final step `1499`.

It retains the 100-action horizon, sets `num_train_steps = 1500` at microbatch 8 times accumulation 4 (effective batch 32), `near_term_steps = 30` with `near_term_weight = 2.0`, `temporal_loss_weight = 0.0`, `save_interval = 500`, and `keep_period = 1000000`. It initializes from the v8 control step `2999` (`checkpoints/pi0_banana_lora/banana_v8_control32_3000/2999`) with `restore_model = control` and the recorded normalization digest, and directs checkpoints to a separate volume so existing weights are not deleted.

The [sampler asset](../assets_motion_v9/banana_sampling.json) is produced by [the preparer](../scripts/prepare_motion_training.py) from [event windows](../server/banana_event_sampling.py). It targets 0.5/0.25/0.25 probability mass across old/new-full/correction groups over the 26/12/39 recorded segments, balancing new-full mass across 12 independent parent layouts and correction mass across 11 parent layouts before dividing by segment, and applies union `before = 15`/`after = 25`-exclusive factor-2 windows. Weights are bound to the exported train manifest digest and are stochastic probabilities, not realized counts. Because data, expert shape, and loss change together, a bundled pilot cannot attribute an outcome to any one of them.

## Serve and evaluate separately

Before model evaluation, verify serving metadata declares original pi0, the trained action horizon, checkpoint suffix, matching normalization digest, upstream revision, and compatible RTC adapter. The [RTC client](../src/pi0_repro/remote.py) now rejects a mismatch between server-declared `action_horizon` and configured `replan_steps` before sending an observation. [Configuration validation](../src/pi0_repro/config.py) bounds RTC taper by the configured horizon and keeps the delay floor shorter than prefetch overlap; empty chunk-recording arrays also use the configured horizon. This removes a hardcoded client-side horizon-50 constraint but does not make new trained weights or a service available. A horizon-100 checkpoint still requires matched serving and client settings; existing horizon-50 selected profiles remain controls and cannot silently consume it. Horizon extension must not be simulated by repeating a 50-action output or changing metadata alone.

Use preassigned validation layouts to compare candidate checkpoints and inspect success, lifted-pinch/placement/release failures, dwell near the banana and box, gripper reversals, reference/physical motion, inference latency, and underruns together. Keep new execution settings fixed when attributing a checkpoint difference. Do not select from lower training loss alone or compare losses across incompatible action normalization as proof of improvement. Freeze any chosen checkpoint/settings before separate held-out testing. A completed run or metadata handshake does not establish reduced hesitation or smoother model behavior.

### Separate v6 horizon-100 Mac candidate

The [v6 Mac configuration](../configs/banana_v6_long100_pi0.toml) pins checkpoint suffix `banana_v6_long100_accum8000/7999`, normalization SHA-256 `6c6e7a92643c76996bf15ba9fc3d3e346dd6bc3d8697712118d9fdb49dedf269`, and `replan_steps = 100`. It retains continuous15's 15-step request age, before-step pacing, 3 Hz arm filter, eight-step quintic transition, and RTC settings. At 50 Hz, the nominal request age remains 0.30 s and the derived prefetch window is 85 steps. A 100-action prediction spans two simulated seconds; it does not require executing two seconds before requesting another prediction. Replacement timing still depends on latency and valid overlap.

Before connecting, provide the matching original-pi0 horizon-100 RTC service, the lossless PNG sidecar on remote loopback port 8001, and the correct SSH forward to the new instance. A policy listener on port 8000 alone does not provide the configured PNG endpoint. From the Mac application directory, inspect metadata before running:

```sh
bash scripts/run_mac.sh connect --config configs/banana_v6_long100_pi0.toml
bash scripts/run_mac.sh run --config configs/banana_v6_long100_pi0.toml --display
```

This is a separate evaluation candidate; the v3 baseline remains unchanged. Reported server startup establishes startup only. Local compatibility checks, a successful connection, or a running display do not establish task success or improved smoothness, and this guide asserts no such v6 result.
