# Generation-time RTC and recovery-data workflow

## Status and boundaries

This guide covers optional original-pi0 improvements around the existing banana task. The [RTC configuration](../configs/banana_rtc_pi0.toml) is experimental. The [selected operation profile](../configs/banana_recommended_pi0.toml) pins established v3 weights and their normalization while using the same RTC schedule; the newer quality-continuation checkpoints are not promoted. Selection evidence and its limits belong to the [dated report](rtc-quality-experiment.md). Source support does not establish lower jitter, a new success rate, a running server, or a completed fine-tuning experiment. Keep the [synchronous baseline](../configs/banana_finetuned_pi0.toml) and the earlier [smooth-execution candidate](smooth-execution.md) available for controlled comparison.

The project remains a Mac simulator/data client and a separate Linux JAX model service. It does not switch to pi0.5, change the pinned upstream checkout, or turn simulator rewards into reinforcement-learning updates. See [architecture](architecture.md) for component boundaries.

## What generation-time guidance changes

The earlier smooth candidate requests a new chunk in the background, skips expired actions, and limits commanded targets. Its optional `chunk_blend_steps` blends already generated arm targets. It does not constrain how the model generates the replacement.

The [RTC adapter](../server/rtc_policy.py) guides the clean-action estimate inside every flow integration step. It differentiates the estimate with respect to the current noisy action using a vector-Jacobian product, then uses the weighted discrepancy from the previous plan to correct the flow update. The integration follows the pinned original-pi0 convention: time 1 is noise and time 0 is clean actions.

For each subsequent request, the [executor](../src/pi0_repro/execution.py) forecasts the still-valid old plan through a copied target limiter and the environment's pure action projection, including physical finger-actuator bounds beyond nominal gripper limits. It sends that projected command prefix in native ALOHA coordinates, together with its valid length, estimated inference delay, and guidance settings. These are command targets, not a prediction that physical joints will exactly reach those positions.

The server passes the prefix through the same input transformations as ordinary serving: ALOHA conventions, checkpoint normalization, and padding from 14 to 32 dimensions. The temporal guidance weights apply to the native 14 dimensions, including grippers; padded dimensions are unweighted. This differs from client-side blending and acceleration limiting, which exclude grippers. Outputs retain the existing absolute 14-dimensional joint/gripper target interface.

The prefix has full weight before the estimated delay and decreasing weight over the following blend window, clipped by the old plan's remaining valid steps. The delay estimate takes the configured floor and recent measured delays into account. A replacement received after its old-plan overlap has expired is discarded. Guidance is soft: it does not guarantee exact position/velocity continuity, physical stability, or deadline compliance.

The first request has zero temporal guidance weights and compiles the guided graph during startup. A later request with no valid old-plan overlap uses the same zero-weight RTC path, reusing the compiled graph instead of first compiling a separate plain sampler during a stall. These requests do not use invented successful actions as a prefix. Startup compilation and the extra differentiation work can increase latency. An exhausted action horizon still blocks while a fresh prediction is obtained.

## Enable the optional server path

Use the existing Linux openpi environment and matching ROI training configuration/checkpoint. The launcher accepts `PI0_RTC=1`; the underlying [serving entry](../server/serve_pi0.py) accepts `--rtc`. Substitute the deployment's existing project and checkpoint paths:

```sh
PI0_RTC=1 bash "$PI0_ROOT/server/run_banana_serve.sh" \
  "$PI0_ROOT/configs/banana_finetune_roi.toml" \
  "$PI0_CHECKPOINT" 10
```

The adapter requires the pinned original-pi0 JAX banana model. It rejects pi0.5 and the PyTorch model path. A request without an `rtc` object continues through ordinary policy inference. The server advertises `rtc.schema = "pi0-rtc-v1"` only when enabled; the Mac RTC configuration requires that capability before sending observations. Current serving source also records the RTC adapter's SHA-256 digest in metadata. That field is emitted when a process starts with the updated source; editing a file does not update an already running service's metadata.

The candidate also selects the lossless PNG sidecar on forwarded local port 8001. Start the sidecar and SSH forwarding as described in [smooth execution](smooth-execution.md#required-services-and-environments), after the compatible model service is available. Starting the sidecar alone does not enable RTC or load a model.

From the Mac project directory:

```sh
bash scripts/run_mac.sh connect --config configs/banana_recommended_pi0.toml
bash scripts/run_mac.sh run --config configs/banana_recommended_pi0.toml
```

These commands are entry points, not evidence that the service is currently deployed.

Client model settings may also pin `expected_checkpoint_suffix` and `expected_norm_stats_sha256`. The suffix must be relative and exclude parent traversal; the digest is 64 lowercase hexadecimal characters. If configured, `RemotePi0` rejects mismatching checkpoint or normalization metadata before sending observations. Pin the actual checkpoint's normalization-byte digest reported by the intended server: equivalent values serialized into a different JSON file can have a different digest. Unpinned legacy configurations retain their previous behavior. These checks catch mismatched deployment settings; server-declared metadata is not independent authentication of checkpoint or asset authenticity.

## Candidate settings and interpretation

The [TOML](../configs/banana_rtc_pi0.toml) is authoritative. Current candidate values are an experimental starting point, not an established optimum.

| Setting | Current value | Meaning |
| --- | --- | --- |
| `replan_steps` | `50` | Maximum action validity horizon |
| `prefetch_steps` | `35` | Start the next request when 35 valid old actions remain |
| `rtc_delay_steps` | `15` | Minimum forecast delay; recent measured delays can increase it |
| `rtc_blend_steps` | `20` | Soft guidance taper after the delay, limited to valid overlap |
| `rtc_guidance_max` | `0.5` | Upper bound on the guidance gain; experimental, not an established optimum |
| `chunk_blend_steps` | `0` | Disable separate client-side target interpolation |

At the configured 50 Hz control frequency, 15 steps cover 0.30 seconds and 20 steps cover 0.40 seconds of simulation time. This does not make the model run at 50 requests per second. The client retains one request in flight and requires `0 < prefetch_steps < replan_steps`; the post's `reinfer=50` convention is not a directly interchangeable project setting.

The RTC client requires an original-pi0 banana policy, asynchronous inference, wall-clock pacing, and agreement between server-declared `action_horizon` and client `replan_steps`. The selected profiles and the candidate values above remain horizon 50. Longer horizons require separately trained/matched serving and client settings; current client compatibility does not establish an available longer-horizon checkpoint or service. `rtc_blend_steps` must remain within the configured horizon, and the positive delay floor must be shorter than the prefetch window. See [long-chunk training](long-chunk-training.md) for the separate horizon-extension workflow.

The selected profiles retain `chunk_blend_steps = 0`. The opt-in [continuous15 profile](../configs/banana_mac_continuous15.toml) is a separate exception: nonzero blending is accepted only with the forecast-aware `chunk_blend_shape = "quintic"` and at least two configured transition steps. Nonzero legacy linear blending remains rejected with RTC. The executor shares its timestamped arm-reference calculation between actual action selection and copied-limiter/projected-prefix forecasting, including any unfinished preceding transition; grippers are not crossfaded. The bridge is bounded by remaining old-plan validity and disabled when fewer than two valid old steps remain. It does not extend expired actions or guarantee physical continuity. See [Mac tuning](mac-motion-tuning.md) for the candidate and diagnostics. Preserve checkpoint, normalization, ROI, scene physics, and success rules when evaluating scheduling changes.

## Evidence to inspect

With the optional jerk-reference backend, copied RTC limiter state must include the same numeric arm position/velocity/acceleration and independent input-filter state used by execution. The pure local Ruckig calculation is then repeated with the same actuator projection; hard-projected axes reset their derivative state in both paths. A native solver handle is not shared across forecasts. Exact command-prefix reconstruction checks this copied-state boundary, not physical tracking or contact jerk. See [Mac tuning](mac-motion-tuning.md) for the optional backend and its limits.

The separate [temporal3 candidate](../configs/banana_v6_temporal3_pi0.toml) uses timestamp-aligned arm-only Temporal Ensembling instead of nonzero client chunk blending. Its pure reference is shared between actual execution and RTC forecasting through the copied limiter and actuator projection. Forecasts drop expired contributors and renormalize the same oldest-first age weights used during execution; grippers remain from the newest valid raw plan. Ensemble mode requires asynchronous inference and cannot be combined with nonzero `chunk_blend_steps`. See [Mac tuning](mac-motion-tuning.md) for weighting, bounded plan retention, and recording checks. Sharing this path supports command-forecast consistency but does not guarantee physical tracking or improved task outcomes.

Latency JSONL records include the request's non-array RTC settings, prefix SHA-256 digest, server guidance metadata, and `planned_vs_executed_prefix_max_error`. The latter compares the sent command forecast with actual executed command targets over the observed inference interval; it is not a physical-state tracking error.

Episode NPZ files retain `rtc_prefixes`, `rtc_request_steps`, and `rtc_valid_steps` so subsequent guided requests can be reconstructed and checked against their original simulation time. New records also retain `chunk_transition_steps` per accepted chunk for optional client transitions. Existing prediction origins, per-action prediction indices/offsets, raw targets, submitted targets, executed controls, and startup/underrun measurements remain available. These fields describe the new recording contract; older artifacts may not contain them. Retain failed and stale-chunk runs rather than selecting only visually smooth successes.

Compare success, completion time, block-boundary target changes, within-block oscillation, measured joint motion, inference latency, and underruns. A lower target jump does not alone establish lower physical jitter. Report display mode and wall-clock timing; MP4 timing follows simulation time.

No improvement measurement is asserted by this guide. Inference-time guidance can be evaluated using the existing checkpoint before committing to another training run.

## Phase-aware demonstrations and visible diversity

### Offline v6 recovery-specific expert entry

The optional [expert field](../src/pi0_repro/banana_expert.py) `recovery_grasp_min_height` defaults to `None`; an explicit value must be finite within 0–0.05 m. Only recovery-mode table-resting grasp preparation raises the descend target to at least this height. Ordinary collection/default targets are unchanged, and already-held recoveries bypass this branch. This is an offline label-generation clearance option, never a runtime policy fallback; any alternate labels still need separate physical checks and explicit review. See the [contact-target recovery record](contact-target-recovery-20261029.md) for the retained original candidates and separately declared clearance comparison.

Two other offline expert options default to zero and accept integer steps from 0–50: `recovery_target_bridge_steps` crossfades the previous executed target into the moving newly generated expert target; `recovery_unload_steps` instead applies a stationary quintic transition toward the initial measured arm configuration before refreshing actual q and starting normal expert motion. They are mutually exclusive. Unloading applies only to recovery-mode table-resting/not-grasped entry and preserves gripper commands; ordinary, already-held and disabled paths retain existing behavior. The previous learner control is a continuity boundary, never copied future learner actions as ground-truth labels. These teacher interventions remain experimental opt-in label-generation candidates requiring separate physical-quality gates and review. The [contact-target screen](contact-target-recovery-20261029.md) failed its typical-motion gate; task success alone does not establish acceptable takeover labels or performance promotion.


The separate [motion expert configuration](../configs/banana_motion_expert.toml) opts into quintic interpolation, tighter IK tolerances, shortest-path SO(3) orientation interpolation, and interpolation-peak-aware segment timing. Defaults preserve the earlier expert behavior. These bounds concern the initial joint-space curve and Cartesian position profile; they do not bound every joint velocity produced by Cartesian IK or measured contact jerk. Structural checks alone do not establish physical improvement or accepted new labels. This remains an offline label-generation option, never an inference fallback. See [motion-first optimization](motion-first-optimization-20261026.md) for the separate protocol and evidence boundary.

That profile also selects quintic gripper interpolation with 0.9 s opening and 1.0 s closure; the default gripper interpolation remains linear. Its recovery-only `direct_recovery_release` branch handles a supported, stable, inside-box banana that is not yet released by retaining arm actuator targets while opening, then retreating/settling, without changing recorded grasp/lift history. `direct_recovery_place` may skip returning to hover for an already held object when TCP XY is within 0.03 m of the offset-adjusted box target. Both flags default false, require recovery mode, and retain ordinary/already-released guards. These offline entry paths require physical screening and explicit review before labels are accepted; they are not runtime ground-truth fallbacks or guarantees on physical contact jerk.

The dated motion-first report records a successful six-run legacy/new expert screen, with lower active jerk but longer new-profile durations. That finite screen does not establish a model benefit or automatically approve subsequently collected candidates. Expert event-window weights can be prepared through [the training-only helper](../server/banana_event_sampling.py), using a union of close/release-entry windows; sampler assets and any deployment remain separate verified steps.

For subsequent motion-first acquisition, a fixed method selects available training-learner states at near-grasp offsets, held-near-box states, and stable supported unreleased states. Missing roles remain missing; corrections do not become independent layouts. Explicit episode review is required after collection. The [motion training job](../server/motion_train_job.py), [status helper](../server/motion_train_status.py), and [deployment helper](../scripts/deploy_motion_training.sh) provide separate source/data/normalization, effective-batch, GPU-probe, and checkpoint-capacity guards. Their implementation does not by itself establish a deployed pilot, accepted correction data, or trained-policy improvement. Accepted correction data and one prepared pilot plan now exist independently of these helpers; consult the dated report for actual stage evidence.

The [v6 recovery expert configuration](../configs/banana_v6_recovery_expert.toml) opts into `direct_recovery_grasp`. The default is false. [Demonstration collection](../src/pi0_repro/demonstrations.py) enables the expert's separate recovery mode only when recovery metadata exists. For a replayed table-resting recovery state, the opt-in branch can skip opening and the initial hover when the active site is within 0.05 m of the grasp target and rotation error is below 0.35 rad, then descend, close, and lift. The physically held-object branch preserves closure before placement. Normal expert collection retains its existing entry behavior.

This is offline supervised-label generation after a checked train-only takeover, never a runtime policy fallback, automatic intervention during evaluation, or altered success rule. All segments still require structural/physical and explicit delegated review. See [the dated v7 recovery optimization](v7-recovery-optimization-20261026.md) for source rollouts, dependent-segment lineage, and export/training boundaries.

The separate [v7 preparer](../scripts/prepare_v7_recovery.py) binds the reviewed export and sampler to the preserved validation/test lineage. [Deployment](../scripts/deploy_v7_recovery.sh) accepts an existing SSH alias supplied by the user; its optional `--auth-only` mode prepares authentication without upload or training. It reuses a task-specific ControlMaster socket in a directory with mode 0700 and a one-hour idle persistence setting, without saving passwords or creating new credentials. Do not copy private connection details into reports.

The [remote job](../server/recovery_train_job.py) checks committed v6 initialization, exact checkpoint normalization, train-only counts, and sampler digest before stopping serving. It probes three real-data optimizer updates and requires shape `[16, 100, 32]` plus at least 2,048 MiB GPU headroom before the separate 1,000-update continuation. It attempts to restore v6 serving on failure, but recovery must be checked independently. Job dispatch, probe success, and training status are distinct from a committed checkpoint and validated policy improvement.

The [quality expert configuration](../configs/banana_quality_expert.toml) slows Cartesian motion in `descend` and `lower_into_box` with `precision_speed_scale = 0.6`. Other non-initial Cartesian moves use `transit_speed_scale = 1.1`. These factors change executed simulator trajectories; they do not duplicate frames or alter the 50 Hz observation/action alignment. Initial joint-space approach and gripper opening/closing durations retain their separate controls.

Collect a preassigned dataset from the Mac project directory:

```sh
bash scripts/run_mac.sh collect-dataset \
  --config configs/banana_quality_dataset.toml
```

The linked [collection configuration](../configs/banana_quality_dataset.toml) plans eight layouts: four train, two validation, and two test. Placement strata and split membership are fixed before collection. It also varies table RGBA colors and light scale within the camera-visible scene. These changes are intended to reach the existing ROI; modifying only pixels outside that crop would not diversify the model's input. The collection remains a limited distribution, not arbitrary-scene coverage. Automatic success leaves each episode pending explicit review.

The [quality audit](../src/pi0_repro/quality.py) checks finite 14-dimensional arrays, synchronized 50 Hz timestamps, state/next-state continuity, image count, and complete controlled grasp/place metadata. It reports per-phase commanded target speeds and large target changes as advisory measurements. These are neither measured physical joint speeds nor automatic grounds for deleting hard examples.

```sh
bash scripts/run_mac.sh audit-demo \
  --episode data/banana_demos/<collection>/episode_NNN
```

The command writes `quality.json`. A structural failure blocks acceptance; advisory motion metrics nominate a trajectory for replay. Neither an audit nor simulator success marks it accepted. Review the saved video, phase images, outcomes, and the audit before recording a decision:

```sh
bash scripts/run_mac.sh review-demo \
  --episode data/banana_demos/<collection>/episode_NNN \
  --accept --reason "Reviewed controlled grasp, transport and release."
```

The default reviewer is human. User-delegated assistant review uses the existing explicit reviewer/evidence protocol in [demonstration collection](仿真专家演示采集.md); never represent assistant review as human approval. Acceptance reruns structural gates and retains the decision history.

## Expert corrections from learner-visited states

The [recovery collector](../src/pi0_repro/recovery.py) implements a DAgger-style data step: replay a recorded learner trajectory, then label subsequent actions with the scripted expert. It is not an automatic multi-round DAgger trainer and does not train on the learner's failed actions.

Select a learner run performed in exactly the scene of an episode whose layout was assigned to the training split before collection. Validation/test source episodes are rejected:

```sh
bash scripts/run_mac.sh collect-recovery \
  --config configs/banana_quality_expert.toml \
  --rollout outputs/<learner-run> \
  --source-episode data/banana_demos/<training-collection>/episode_NNN \
  --step 400
```

The step is an example, not a recommended takeover point. It must be an interior recorded step at which the supported expert can take over.

The collector checks original-pi0 learner metadata and canonical equality of the learner scene, source episode scene, and recovery scene, including controller and appearance. It resets with the source seed and replays the learner's actually executed actions through the simulator. Before takeover, the reconstructed 14-dimensional observed state must agree with the recorded state at that step within `1e-5`; otherwise collection stops. This check validates the observed state, not an independent comparison against a previously recorded full physical snapshot.

It saves `recovery_start.npz` with full simulator `qpos`, `qvel`, `ctrl`, time and grasp/lift history. Physical state and velocities remain intact. Only segment-local counters restart, and the new expert segment records timestamps from zero. `recovery.json` retains learner trajectory and snapshot digests, source run/episode/step, checkpoint, source split, and label provenance.

The supported takeover states are deliberately bounded:

- A banana still resting on the table, allowing the expert to approach and grasp it.
- A banana currently pinched by the two fingers, allowing the expert to continue controlled placement while preserving commanded closure.

Unsupported dropped or airborne states fail explicitly. Keeping the previous commanded closure matters because contact can displace measured finger positions; reissuing those measured positions as targets can loosen the grasp. This is not a claim of arbitrary failure recovery.

New correction segments use new training-only IDs while preserving their parent layout lineage; they are not independent layouts. They retain the same explicit review requirement, structural gates, image/action alignment, and export semantics as ordinary demonstrations. Preserve failed correction attempts with their error metadata.

## Merge, export, and prepare controlled continuation

The [merge API](../src/pi0_repro/dataset_merge.py) accepts reviewed collection directories and creates a new versioned batch. It copies independent episode files, keeps source batches unchanged, preserves each layout's train/validation/test assignment, rejects repeated layout IDs, and carries review evidence plus source metadata digests forward. This is a Python API, not a separate CLI subcommand.

From the project directory, substitute actual reviewed batch directories:

```sh
PYTHONPATH="$PWD/src" .venv/bin/python - <<'PY'
from pi0_repro.dataset_merge import merge_batches
print(merge_batches([
    "data/banana_demos/<original-reviewed>",
    "data/banana_demos/<diversity-reviewed>",
    "data/banana_demos/<recovery-reviewed>",
]))
PY
```

Only episodes marked eligible are copied. The exporter remains responsible for checking explicit review/provenance before conversion. Use the separate existing data environment:

```sh
bash scripts/run_data.sh --source data/banana_demos/<new-merged-batch>
```

Conversion preserves separate split datasets and writes a complete `dataset_manifest.json`. See [LeRobot export](LeRobot数据转换.md) for the pinned format, readback checks, and refusal to overwrite existing output.

### Motion-first merged dataset

The motion-first stage merged the reviewed base batch, the 12 new full-expert episodes, and the accepted dependent corrections. Its export is complete: `data/lerobot/20261026-213307-merged-ffaed3` has a `dataset_manifest.json` with `status: complete` and 77 train episodes/84,701 frames, 6 validation episodes/7,423 frames, and 6 test episodes/7,483 frames. Conversion runs through the same data environment:

```sh
bash scripts/run_data.sh --source data/banana_demos/20261026-213307-merged-ffaed3
```

The export guide records refusal to overwrite existing output, so version a later export rather than reuse this directory. The manifest also records that LeRobot raw statistics are not openpi normalization assets and must be computed from train data after the ALOHA transforms.

The [quality-training preparer](../scripts/prepare_quality_training.py) consumes that complete exported dataset:

```sh
.venv/bin/python scripts/prepare_quality_training.py \
  --dataset data/lerobot/<new-merged-batch> --action-std-floor 0.01
```

It writes `configs/banana_finetune_quality.toml`, `assets_quality/banana_sampling.json`, and explicitly versioned normalization under `assets_quality_norm`. The preparer refuses existing candidate config/sampler files, and the normalization helper refuses an existing target asset; retain them as that candidate's preparation and deliberately version a later preparation. Generation does not start training.

The continuation candidate retains original-pi0 LoRA plus the existing action/state projection training, ROI, and weighted flow-matching objective. It initializes from the existing v3 step-1999 checkpoint and uses 1,000 updates with batch size 4 and peak learning rate `2e-5`. The current candidate explicitly changes nearly constant action-axis normalization; the earlier fixed-normalization pilot is preserved as `banana_finetune_quality_v4.toml` and is not the source checkpoint for this candidate.

The [normalization helper](../src/pi0_repro/normalization_assets.py) applies a minimum action standard deviation of 0.01 in the post-ALOHA-transformation action units. State statistics, action means, and quantiles retain their prior values. Original pi0 uses z-score action normalization here, so unchanged quantiles do not preserve the old action scaling. The versioned asset report records changed dimensions, old/new scales, source/target hashes, and the train-dataset manifest hash.

This is an explicit coordinate-system change, not an exactly function-preserving initialization from v3. Keeping checkpoint statistics is appropriate only while the new data remains compatible with them. New recovery motions can expose nearly constant axes whose old standard deviation amplifies small native changes; the fixed-statistics pilot demonstrated why that compatibility must be audited. Train and evaluate the adapted candidate, and serve it with normalization assets copied into its own checkpoint. Do not use the new action scales with the old v3 checkpoint as if they were interchangeable. See the [dated experiment report](rtc-quality-experiment.md) for the actual pilot outcome and limits.

The sampler allocates nominal probability mass across original/diversity/recovery groups as 0.55/0.30/0.15, renormalized across groups actually present. Episodes share mass equally within their group; precision-phase windows receive 1.5 relative weight inside an episode. Weighted sampling is stochastic, so those shares are not exact counts in a short run. A manifest SHA-256 guard binds weights to the intended exported dataset, and the loader checks train membership and frame count.

After transferring the new dataset, configuration, weights, candidate-matched versioned normalization assets, and code to the matching Linux environment, inspect the candidate before starting a separately named experiment:

```sh
"$PI0_ROOT/openpi/.venv/bin/python" \
  "$PI0_ROOT/server/banana_finetune.py" describe \
  --root "$PI0_ROOT" --config "$PI0_ROOT/configs/banana_finetune_quality.toml" \
  --exp-name banana_quality_candidate
```

For an intentional training run in a persistent Linux session:

```sh
bash "$PI0_ROOT/server/run_banana_train.sh" banana_quality_candidate \
  --config "$PI0_ROOT/configs/banana_finetune_quality.toml"
```

The [evaluation runner](../scripts/evaluate_banana_policy.py) accepts `--dataset` to select an exported preassigned split mapping and `--layouts` to select explicit layout IDs or source episode names within the selected suite. It omits optional null scene fields when reconstructing scene TOML. For example, after preparing a compatible headless client configuration:

```sh
.venv/bin/python scripts/evaluate_banana_policy.py \
  --config configs/<headless-candidate>.toml \
  --suite validation --experiment quality_candidate_validation \
  --dataset data/lerobot/<new-merged-batch> \
  --layouts episode_026 episode_029
```

The episode names above are an example; read the selected dataset's split mapping instead of assuming every aggregate assigns the same names. The runner disables display, so its results do not evaluate GUI behavior.

Use training layouts for corrections and parameter selection on validation layouts. Leave test layouts out of sampler construction, normalization decisions, and recovery collection. If a formerly held-out layout is used for training, disclose that reuse and define a new independent test set.

The candidate config is a controlled starting point, not a full-fine-tuning recommendation or a completed result. Evaluate fixed checkpoint/settings with success, failure types, completion time, commanded and physical motion, and latency. Change execution and training factors separately when attributing effects. No new checkpoint outcome is established by creating these preparations.

## Rollback

Use the synchronous baseline for model/control diagnosis:

```sh
bash scripts/run_mac.sh run --config configs/banana_finetuned_pi0.toml
```

Use the earlier smooth configuration to compare time alignment without generation-time prefix guidance:

```sh
bash scripts/run_mac.sh run --config configs/banana_smooth_pi0.toml
```

Both require their matching transport endpoint. Preserve the optional configuration and its recordings separately; do not overwrite historical baseline evidence.

### Optional continuous expert transport

[Expert configuration](../src/pi0_repro/banana_expert.py) `continuous_transport_radius` defaults to 0 and accepts finite values from 0–0.03 m. A positive radius selects the [rounded Cartesian path helper](../src/pi0_repro/expert_path.py) after closure for ordinary or table-resting-recovery lift/transport/lower motion; already-held/direct-release/default paths remain unchanged. Virtual-distance C2 corners and smooth 0.5-second velocity ramps bound Cartesian speed by existing precision/transit limits, not physical servo derivatives or contact forces. This is offline label generation, never a runtime policy fallback. The [dated IK experiment](ik-expert-optimization-20261029.md) separates teacher quality gates from data acceptance and eventual policy evaluation.
