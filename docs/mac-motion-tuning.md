# Mac motion-tuning candidates

## Scope and status

These Mac-only candidates expose request cadence, optional arm-target filtering, render scheduling, and diagnostics for the original-pi0 banana task. They do not retrain the policy, change the cloud checkpoint, modify the pinned upstream source, or establish a new success rate. Keep the selected [v3 RTC operation profile](../configs/banana_recommended_pi0.toml) and [synchronous baseline](../configs/banana_finetuned_pi0.toml) as controls. Neither has been replaced by these candidates.

The execution-reference candidates preserve the physical controller and task-success rule. Separately labeled scene candidates below can change plant parameters. No task-phase controller, proximity-triggered grasp, expert fallback, automatic gripper closure, or speed retiming is added. A smoother target sequence can still grasp poorly or stall. Evaluate completion and physical motion together.

## Candidate configurations

The four candidates in this table preserve the 50-step prediction horizon and 50 Hz physical control rate, use the same pinned v3 checkpoint and normalization digest, retain existing joint/gripper speed and arm acceleration limits, and require the compatible RTC-enabled PNG endpoint. Connection and serving prerequisites are in [RTC operation](rtc-and-recovery.md) and [PNG transport](smooth-execution.md#required-services-and-environments). Configuration files do not establish that those services are running.

| Configuration | Request interval | Old-plan overlap at request | Arm reference filter |
| --- | --- | --- | --- |
| [cadence 15](../configs/banana_mac_cadence15.toml) | 15 steps / 0.30 s | 35 steps | Disabled |
| [cadence 25](../configs/banana_mac_cadence25.toml) | 25 steps / 0.50 s | 25 steps | Disabled |
| [cadence 35](../configs/banana_mac_cadence35.toml) | 35 steps / 0.70 s | 15 steps | Disabled |
| [cadence 25, filtered](../configs/banana_mac_cadence25_filtered.toml) | 25 steps / 0.50 s | 25 steps | Critically damped, 3 Hz |

`request_interval_steps` is measured from the active chunk's observation origin. It is the nominal earliest request age, not a fixed count of commands executed from every chunk or a guaranteed interval between switches. After a request starts, old valid actions continue until the replacement is received. The executor skips the new chunk's elapsed prefix and permits only one in-flight request. For example, a request at age 25 that returns 10 control steps later can switch around old-plan age 35; it does not append the complete new 50-step chunk.

The loader resolves `prefetch_steps = replan_steps - request_interval_steps`. If both keys are explicitly supplied, they must agree. Existing configurations that omit `request_interval_steps` retain their prefetch convention. These candidates share `rtc_delay_steps = 10`, with recent measured delays still able to increase the forecast; the delay floor is not a latency guarantee. The 35-step candidate leaves only 0.30 seconds of simulated old-plan overlap. A late reply may be rejected after that overlap or produce a running-control underrun. It is a comparison option, not the safest default or an established optimum.

The new cadence-15 profile also changes rendering and the RTC delay floor relative to the selected historical operation profile. Compare the new 15/25/35 candidates to isolate cadence, and compare unfiltered/filtered cadence 25 to isolate the reference filter. Do not attribute a comparison against the historical profile to cadence alone.

## Run one candidate

Use the project environment from this project directory. With the existing compatible RTC service, PNG sidecar, and forwarding already available, inspect metadata before running one candidate:

```sh
bash scripts/run_mac.sh connect --config configs/banana_mac_cadence25.toml
bash scripts/run_mac.sh run --config configs/banana_mac_cadence25.toml
```

The candidate is a starting point for comparison, not a promoted replacement. A successful metadata handshake checks declared compatibility, not physical task success. Close the native window or press Escape to abort through the normal recording path.

## Target filtering and its limits

The optional `joint_target_filter_hz` adds a critically damped reference filter before the existing target limits in [JointTargetLimiter](../src/pi0_repro/execution.py). It filters the 12 arm-joint targets; the two gripper targets keep their existing independent velocity limits and are not averaged. Filtering requires the configured arm acceleration limit, accepts a finite positive frequency up to 20 Hz, and is disabled when omitted. Lower filter frequencies increase smoothing and tracking lag; 3 Hz is a candidate value, not a verified optimum.

In the default limiter backend, a constrained filter axis is reanchored to the executed command after actuator projection so filter state does not accumulate beyond the submitted target. The optional jerk backend below instead keeps its input-filter state independent from the output reference. RTC prefix forecasting deep-copies the selected filter/limiter state and applies the same action projection. The forecast remains a command forecast, not a guarantee that physical joints reach those targets.

Reference filtering is distinct from changing the MuJoCo position-controller gains or retiming the demonstration. It can suppress rapid target variation while delaying arrival or changing contact timing. It does not resolve a policy that repeatedly opens its gripper or fails to transition from alignment to grasping. Keep those failures visible in evaluation rather than hiding them with task-specific closure rules.

## Render and video scheduling

`render_on_request = true` is supported for `BananaBox-v0`. The environment steps physics and returns current robot state without automatically rendering a camera frame on every control step. The simulator obtains camera pixels on the main thread only when inference, display, or video needs them. A same-step cache shares one render among those consumers; it does not reuse an older step's frame for a fresh inference observation. MuJoCo operations and display-event handling remain on the main thread.

The candidates request 25 Hz display and `video_fps = 25`. Video frequency must be a positive integer divisor of 50; it controls frame sampling and encoding metadata, not the physical control rate or model horizon. Display has its own wall-clock refresh schedule. Lower display/video rates can reduce work, but rendering still costs time, and the inference worker still needs transport encoding and cloud inference. These changes do not provide hard real-time execution. Video remains sampled on simulation time, so smooth playback does not prove smooth wall-clock control.

## Optional action-clock alignment candidate

The separate [paced25 profile](../configs/banana_mac_paced25.toml) retains filtered25's v3 weights, 25-step request age, RTC guidance, target limits, and 3 Hz arm-reference filter. It opts into `control_pacing = "before_step"` and changes display/video sampling to 50 fps. The existing recommended and filtered25 profiles are unchanged; configurations that omit `control_pacing` retain the default `"after_step"` behavior.

With `before_step`, the main thread prepares rendering, any observation/request, and the limited target, then waits for the scheduled application deadline immediately before `env.step`. This mode requires `real_time = true`. At display rates of at least 50 Hz it attempts one draw per control step, avoiding a skipped draw when the loop starts just before the next independent display deadline. It does not create extra physical steps or guarantee 50 achieved frames per wall-clock second. MuJoCo and all OpenGL/display work remain on the main thread; no physics thread or frame interpolation is added. In `before_step`, an application miss above 2 ms shifts the next slot to the actual application time plus 20 ms; if loop work has already passed that slot, it reanchors after the work. The legacy mode retains its whole-period reset. Neither executes a burst through expired deadlines. Rendering or inference overruns can still delay action application.

After checking compatible metadata, run this candidate separately:

```sh
bash scripts/run_mac.sh connect --config configs/banana_mac_paced25.toml
bash scripts/run_mac.sh run --config configs/banana_mac_paced25.toml
```

The additional [paced15 profile](../configs/banana_mac_paced15.toml) changes only the nominal request age from 25 to 15 steps relative to paced25. Both retain the 50-step prediction horizon, 50 fps display/video, before-step clock, and 3 Hz filter. Requesting at age 15 leaves 35 old-plan steps, or 0.70 simulated seconds, compared with paced25's 25 steps or 0.50 s. This provides more overlap for late replies but changes how often the policy replans; it is not a larger prediction horizon or a network-latency guarantee. It differs from the earlier unfiltered cadence15 candidate, whose results must not be assigned to this filtered, clock-aligned profile.

```sh
bash scripts/run_mac.sh run --config configs/banana_mac_paced15.toml
```

The [responsive15 profile](../configs/banana_mac_responsive15.toml) changes only the arm-reference-filter frequency from 3 to 5 Hz relative to paced15. It is a response-lag versus fine-motion comparison: a higher cutoff may track target changes sooner while passing more rapid variation. It does not alter gripper decisions or prove that perceived pauses are filter lag. Keep it separate from both 3 Hz profiles and inspect physical smoothness and task progress together. The dated pacing round retained one success and one failure in its two GUI checks and stopped this direction without promotion.

The [continuous15 profile](../configs/banana_mac_continuous15.toml) instead retains paced15's 3 Hz filter and adds `chunk_blend_steps = 8` with `chunk_blend_shape = "quintic"`. It crossfades only arm references from valid old/new plans at the same simulation timestamp. Grippers use the new raw plan without this crossfade. The old reference can include an unfinished earlier transition. Forecasting and execution use the same pure reference before the copied limiter and actuator projection, allowing this opt-in shape with RTC. Nonzero legacy linear blending remains incompatible with RTC; configurations retain the default linear shape and selected RTC profiles retain zero blend steps.

The bridge is shortened to the old reference's remaining validity; fewer than two valid old steps means no bridge. It never extends an expired plan. Quintic weights have zero first and second endpoint derivatives, but time-varying discrete action references, limiter effects, and physical motion do not thereby become mathematically C2 continuous. The transition length actually assigned to each accepted chunk is saved as `chunk_transition_steps` in the episode NPZ. This is a candidate response to reported stepwise motion and jumps near chunk changes, not proof that those changes cause every perceived pause.

New control rows record `action_apply_wall_seconds` immediately before the environment step, `scheduled_deadline_wall_seconds`, `control_pacing`, `pacing_wait_seconds`, and `deadline_phase`. Under `before_step`, deadline lateness refers to the action-application start; under legacy `after_step`, it refers to the loop-end measurement. Do not compare those lateness fields as if they measured the same phase. `action_apply_interval_ms` in the [summarizer](../scripts/summarize_banana_execution.py) compares application starts directly; `control_interval_ms` retains the existing end-of-step timestamps. Older recordings without application timestamps report unavailable application intervals.

Each new `episode_000_display.jsonl` records completed draws and native context, texture-upload, title, and swap durations. Native title text updates at most four times per second, independently of camera-frame draws; `native_title_updates` is the episode-local update count. Per-control-step camera, display, and video-submit durations help locate long frames; these nested stage timers are not exclusive CPU-time slices. The summarizer excludes frames before the first action from `display_interval_ms_after_startup` and counts subsequent display intervals above 50 ms. These are application-side draw timestamps, not monitor presentation measurements. A 25 fps MP4 contains one sample per two control steps; a 50 fps MP4 contains one per step. Neither encodes wall-clock pauses faithfully because frame timestamps follow simulation time. Compare recorded application intervals, display intervals, inference underruns, policy progress, and video separately. Dated diagnosis and actual-model results are in [Mac pacing validation](mac-pacing-validation-20261025.md).

## Original-pi0 v6 longer-bridge candidate

The separate [v6 bridge20 profile](../configs/banana_v6_bridge20_pi0.toml) changes only `chunk_blend_steps` from 8 to 20 relative to the [v6 horizon-100 profile](../configs/banana_v6_long100_pi0.toml): a nominal 0.16 s bridge becomes 0.40 s at 50 Hz. Both retain the same v6 checkpoint and normalization pins, 100-step horizon, 15-step request age, 3 Hz arm filter, RTC settings, and target limits. Existing v3 and v6 profiles remain available unchanged.

This uses the existing pure `ActionReference` crossfade of old/new arm references at the same timestamps; grippers remain unchanged by the crossfade. RTC forecasting uses that same reference through a copied limiter and actuator projection, including nested unfinished transitions. The actual bridge shortens to valid old-plan overlap and never extends expired actions. A longer bridge may reduce reference changes while increasing lag; it does not guarantee smoother physical motion or task success. Frozen-command comparisons before physics are diagnostic evidence only, not actual-model closed-loop validation.

The dated [v6 bridge-length validation](v6-motion-validation-20261026.md) separates frozen-command diagnosis, actual-model GUI observations, and limited validation evidence, including retained failures and isolated physical spikes.

From the application directory, with a matching original-pi0 v6 horizon-100 RTC/PNG endpoint and forwarding available:

```sh
bash scripts/run_mac.sh connect --config configs/banana_v6_bridge20_pi0.toml
bash scripts/run_mac.sh run --config configs/banana_v6_bridge20_pi0.toml --display
```

## Separate v6 acceleration, RTC-taper, and wrist-damping candidates

These independent candidates retain the [v6 bridge20 profile](../configs/banana_v6_bridge20_pi0.toml) as their unchanged control. They do not retrain the model or establish improved motion. Each changes one factor relative to that profile:

| Candidate | Change | Interpretation |
| --- | --- | --- |
| [acc20](../configs/banana_v6_acc20_pi0.toml) | `max_joint_acceleration`: 40 → 20 rad/s² | Limits commanded arm-target acceleration; it does not bound physical joint acceleration during tracking/contact |
| [rtc30](../configs/banana_v6_rtc30_pi0.toml) | `rtc_blend_steps`: 20 → 30 | Extends the generation-time RTC guidance taper; the separate client quintic bridge remains 20 steps |
| [wrist125](../configs/banana_v6_wrist125_pi0.toml) | [Scene](../configs/banana_demo_scene_wrist125.toml) sets `wrist_damping_scale = 1.25` | Changes simulated wrist-joint damping, and therefore dynamics, rather than policy output or scheduling |

The optional scene parameter defaults to 1 and must be finite and positive. A nondefault value requires `controller_profile = "position_pd"`. It scales only both arms' `forearm_roll`, `wrist_angle`, and `wrist_rotate` damping: scale 1.25 changes their respective damping from 2, 2, and 1.5 to 2.5, 2.5, and 1.875. Position gains, finger settings, and contact parameters remain unchanged. The default generated XML is preserved byte-for-byte; the opt-in dynamics change must still be evaluated for tracking, contact, progress, and physical motion.

With matching original-pi0 v6 horizon-100 RTC/PNG serving and forwarding verified, run candidates separately from the application directory:

```sh
bash scripts/run_mac.sh run --config configs/banana_v6_acc20_pi0.toml --display
bash scripts/run_mac.sh run --config configs/banana_v6_rtc30_pi0.toml --display
bash scripts/run_mac.sh run --config configs/banana_v6_wrist125_pi0.toml --display
```

These are unpromoted comparisons. Keep the v6 bridge20 control available; do not combine factors or infer physical/subjective improvement from configuration validation alone.

The separate [RTC10 profile](../configs/banana_v6_rtc10_pi0.toml) shortens only the generation-time RTC taper from 20 to 10. The [2026-10-28 same-session comparison](v6-rtc-same-session-20261028.md) retained RTC20: RTC10 had better observed task coverage but worse typical active jerk, so it remains an experimental tradeoff rather than a default smoothness upgrade.

## Separate v6 horizon-100 cadence comparison

The [cadence30](../configs/banana_v6_cadence30_pi0.toml) and [cadence50](../configs/banana_v6_cadence50_pi0.toml) profiles change only `request_interval_steps` relative to bridge20's cadence15. All three retain the same v6 checkpoint/normalization, horizon 100, 3 Hz arm filter, 20-step quintic bridge, RTC, limits, and rendering settings. Cadence30 requests at nominal age 30 steps (0.60 s), leaving 70 old-plan steps; cadence50 requests at age 50 (1.00 s), leaving 50. The horizon remains two simulated seconds. Nominal request age is measured from the active observation origin, not a guarantee of switch timing or the number of actions executed from every accepted chunk.

```sh
bash scripts/run_mac.sh run --config configs/banana_v6_cadence30_pi0.toml --display
bash scripts/run_mac.sh run --config configs/banana_v6_cadence50_pi0.toml --display
```

Use the matching v6 horizon-100 RTC/PNG endpoint and verify metadata before either run. A longer request interval may preserve a plan's phase progression but also reduces feedback frequency; neither faster grasping nor better physical control follows from the setting alone. The [dated cadence experiment](v6-cadence-optimization-20261026.md) separates fixed protocol, closure-intent diagnosis, and actual-model outcomes.

## Separate v6 RTC-strength candidate

The [RTC 0.1 profile](../configs/banana_v6_rtc01_pi0.toml) changes only `rtc_guidance_max` from bridge20's 0.5 to 0.1. Checkpoint, normalization, horizon 100, request age 15, 20-step quintic bridge, 3 Hz filter, target limits, and all other runtime settings remain fixed. This weakens the existing generation-time prefix guidance; it does not change the sampler algorithm or train new weights.

```sh
bash scripts/run_mac.sh run --config configs/banana_v6_rtc01_pi0.toml --display
```

Use a verified matching v6 RTC/PNG endpoint. Response metadata can verify the actual per-request guidance cap. The [dated RTC-strength comparison](rtc-guidance-validation-20261026.md) tested the prefix-overcommitment hypothesis against preserved same-round v6 controls. Both caps completed five of six layouts; 0.1 had longer grasp delay on `episode_019` and a larger worst physical spike. Retain 0.5; the 0.1 configuration remains an unpromoted comparison, without a claim that all pauses are solved.

## Separate v6 Temporal Ensembling comparison

The [temporal3 profile](../configs/banana_v6_temporal3_pi0.toml) replaces bridge20's crossfade with `chunk_blend_steps = 0`, `temporal_ensemble = true`, `temporal_ensemble_max_chunks = 3`, and `temporal_ensemble_coeff = 0.01`. Other runtime settings match [bridge20](../configs/banana_v6_bridge20_pi0.toml), including checkpoint, normalization, horizon, request cadence, filter, RTC, and target limits. This is a separate arm-reference strategy, not a server/model change or an additional crossfade layered on bridge20.

On each accepted reply, `TemporalEnsembleReference` retains at most the newest three accepted, still-valid plans. At each current or forecast timestamp, it selects each retained plan's action for that same timestamp and averages only the 12 arm joints. Eligible plans are ordered oldest first: rank zero receives weight one, and later ranks receive `exp(-coefficient * rank)`, normalized over eligible plans. This ACT-style age weighting slightly favors older predictions; it does not favor the newest prediction. Expired contributors drop out and weights renormalize, while grippers come from the newest valid raw plan. Zero-valued actions remain valid contributors. A reference with no valid plan raises an error; normal executor horizon exhaustion waits for a new chunk rather than extending expired plans.

The pure reference is shared by execution and RTC forecasting through the copied limiter and actuator projection, including contributor expiry during a forecast. Ensemble mode requires asynchronous inference and excludes nonzero chunk blending. Configuration limits maximum contributors to 2–8 and a finite coefficient to 0–1. Averaging predictions for one timestamp is distinct from filtering actions across successive timestamps, and it can still introduce lag or conflicting intent.

```sh
bash scripts/run_mac.sh connect --config configs/banana_v6_temporal3_pi0.toml
bash scripts/run_mac.sh run --config configs/banana_v6_temporal3_pi0.toml --display
```

New NPZ records include `temporal_ensemble_counts`. The existing `blended_action` field contains the selected reference, including the ensemble reference in this mode; `blended_steps` includes steps with multiple ensemble contributors. The summarizer independently reconstructs the weighted reference and checks contributor counts, newest-plan gripper values, and raw action timestamps. These are implementation-consistency checks, not evidence of better physical motion or task completion. Keep the same layouts and preserve every failure when comparing against a fresh bridge20 baseline; policy noise and network histories remain unpaired.

The dated [Temporal Ensembling comparison](temporal-ensemble-validation-20261026.md) records the predeclared trial order and separates implementation audits, GUI observations, and headless validation outcomes.

## Optional jerk-limited arm-reference screen

`max_joint_jerk` opts into the local [Ruckig reference backend](../src/pi0_repro/jerk_reference.py); omission preserves the existing limiter. It requires finite positive arm velocity and acceleration limits and the optional `motion` dependency pinned to `ruckig==0.19.4` in [pyproject.toml](../pyproject.toml). This dependency belongs to the project Mac environment, not the parent learning environment or cloud model runtime.

The [jerk50 candidate](../configs/banana_v6_jerk50_pi0.toml) screens `max_joint_jerk = 50` rad/s³ against unchanged cadence15 bridge20, retaining the 3 Hz filter, 40 rad/s² acceleration limit, 1.1 rad/s velocity limit, and existing gripper limits. The separately retained [jerk300 configuration](../configs/banana_v6_jerk300_pi0.toml) is an offline preparation, not an established model result.

On each 20 ms step, stateless Ruckig `calculate` advances the 12-arm reference toward the post-filter target position and clipped filter velocity, with zero terminal acceleration and `Synchronization.No`. It uses no intermediate/cloud waypoints. In this mode, input-filter state is independent: output constraints are not fed back through the legacy filter anti-windup path. Grippers retain their existing rate limits. Copied RTC forecasts must preserve the same numeric reference position, velocity, acceleration, filter state, and actuator projection; no mutable native solver handle is required in the copied state.

Hard actuator projection can override the kinematic derivative bounds. Affected reference axes restart from rest, incrementing `jerk_reference_projection_resets`; physical contact response is also outside the commanded-reference guarantee. Consequently, a configured jerk bound does not bound measured physical joint jerk or prove smoother task execution. The [dated jerk screen](v6-jerk-screen-20261026.md) separates retained offline prototypes, implementation checks, and actual-model screening.

## Diagnose commands without cloud inference

The [replay tool](../scripts/replay_banana_actions.py) runs recorded targets in local physics without a model connection or live window. It creates a new `outputs/` directory containing source hashes, the scene, trajectories, and a diagnostic summary. Optional `--video` writes a 25 fps video.

Replay a normal expert episode directly to check whether its original commands already produce unstable motion:

```sh
.venv/bin/python scripts/replay_banana_actions.py \
  --demo data/banana_demos/20261023-213415-07fb84/episode_000 \
  --video
```

For a policy recording, `--rollout outputs/<run>` selects `episode_000`. The default `--action-source executed` replays its executed commands. `--action-source requested` replays its raw requested targets; add `--config configs/banana_mac_cadence25_filtered.toml` to pass them through that candidate's limiter/filter. Applying the limiter to already executed commands is a different experiment and may add smoothing twice; state the selected source explicitly.

Only the limiter/filter settings matter in replay; request cadence does not launch inference. Replay checks the reset observed state against the first recorded state and refuses a mismatch over `1e-5`. It does not restore recovery-segment physical snapshots; those segments are unsupported by this entry point. It also does not pace execution to wall clock.

This is frozen, open-loop action replay. Changed motion does not generate new observations for the original policy, so success cannot establish a closed-loop model improvement or supply an automatic replacement training demonstration. A replay can help separate poor commands from controller/contact response, but it does not by itself prove their cause.

## Compare recorded progress and smoothness

### Motion-first diagnostics

The separate [motion-first protocol](motion-first-optimization-20261026.md) prioritizes reduced jumps and observed low-motion dwell, with completion reported secondarily rather than folded into one score. [Offline motion quality](../src/pi0_repro/motion_quality.py) takes the maximum absolute derivative across the 12 arm joints per frame before reporting active-motion jerk p99/p95 and peak physical joint increment. Active motion requires maximum joint speed above 0.01 rad/s. Stationary recordings with no active samples report an unavailable active-motion distribution, not a perfect smoothness result.

Low-motion spans require at least 0.5 s with maximum arm speed below 0.015 rad/s and maximum commanded gripper speed below 0.05 normalized units/s. Frames already inside, released, supported by the box, and stable are excluded from this dwell measure. Deliberate holds can still qualify; these labels are observed low movement, not inferred intention or a proven policy stall. Pinch/box/success coverage and active-motion duration remain separate, so an incomplete stationary trajectory cannot outrank comparable transport simply through low jerk. Derivatives use simulation time, independently of wall-clock control/latency diagnostics.

The [recording summarizer](../scripts/summarize_banana_execution.py) reads existing artifacts without rerunning physics:

```sh
.venv/bin/python scripts/summarize_banana_execution.py \
  outputs/<baseline-run> outputs/<candidate-run> \
  --output /tmp/pi0-motion-comparison.json
```

Inspect the following together:

| Evidence | Meaning and limitation |
| --- | --- |
| Task outcome | Unchanged physical grasp, containment, release, support, and stability rule |
| `task_progress` | First near-grasp, confirmed lifted pinch, inside-box and success times; near-to-pinch duration and time spent near before pinch |
| Gripper reversals | Direction changes after excursions of 0.05 normalized command units before the first confirmed lifted pinch; not every small numerical change |
| `chunk_continuity` | Absolute per-joint target changes at prediction switches versus within a chunk; separate raw, submitted, and executed commands |
| Motion derivatives | Commanded and measured joint velocity, acceleration, and jerk computed on 0.02-second simulation intervals; not proof of a wall-clock servo frequency |
| Timing | Control intervals, deadline lateness, startup waits, running underruns, and inference round trips |

The proximity diagnostic uses the minimum distance between either finger-pair midpoint and the banana center, with a 0.04 m radius. It is simulator-ground-truth analysis only and is not fed into policy action selection or the success rule. Logs from before this field existed report unavailable proximity timings as `null`, not zero. Missing confirmed grasp/success events are not successful zero-duration results.

New control logs include inference-dispatch and environment-step wall durations. Episode `stage_totals` separate camera, observation preparation, dispatch, physics/environment stepping, display, and video submission. These are nested/inclusive timers, not independent slices to add together: observation preparation can include camera work, dispatch can include observation preparation and waiting, and the old default environment-step path includes rendering. Compare like-for-like profiles and use them to locate cost, not to claim an exact exclusive CPU-time breakdown.

## Repeated validation and rollback

Use the same dataset split and layouts when comparing candidates. The [evaluation runner](../scripts/evaluate_banana_policy.py) supports `--repeats` and turns off display. One explicit example for a single preassigned validation layout is:

```sh
.venv/bin/python scripts/evaluate_banana_policy.py \
  --config configs/banana_mac_cadence25.toml \
  --suite validation --experiment mac_cadence25_validation \
  --dataset data/lerobot/20261024-185715-merged-ebdbe3 \
  --layouts episode_026 --repeats 3
```

Read the selected dataset's source-episode mapping before choosing names for another dataset. Repeats preserve the layout reset seed, but model sampling noise and network timing are not paired across candidates. Preserve every attempt and report failures, underruns, and near-grasp dwell alongside success. Compare GUI runs separately; headless validation does not establish GUI responsiveness. Freeze the chosen configuration before inspecting held-out test results. Do not tune on the test partition.

No new closed-loop success or smoothness gain is established by adding these tools/configurations. Return to the unchanged selected profile for comparison:

```sh
bash scripts/run_mac.sh run --config configs/banana_recommended_pi0.toml
```

Later correction collection and supervised fine-tuning remain separate work, described in [RTC and recovery](rtc-and-recovery.md). This Mac change alone does not remove the model's grasp-transition hesitation.

## Optional right-wrist position gain

The [scene parameter](../src/pi0_repro/scene.py) `right_wrist_angle_kp_scale` defaults to 1; nondefault values must be finite, positive and use `controller_profile = "position_pd"`. Setting 0.75 changes only the right `wrist_angle` position gain from kp200 to kp150, leaving damping, contact settings, fingers and other gains unchanged. This changes simulated plant dynamics rather than predictions or command smoothing. It is an experimental opt-in, not a recommended default; the default XML remains unchanged. The [dated wrist/contact report](continued-wrist-contact-optimization-20261028.md) separates exact replay, the failed two-trace screen and the subsequent continued-model closed-loop comparison.

## Optional deferred top-video rendering

`simulation.defer_video_render = true` moves top-video rendering/encoding after running control. It defaults to false and requires BananaBox-v0, `render_on_request = true` and `save_video = true`. Run headless or use the existing multiview/GLFW `display_in_worker` path; main-process display is rejected, including a conflicting CLI display request. Policy images still render freshly from live main-process physics when inference needs them. This option does not change policy pixels, cameras, weights, RTC settings or action timing rules.

The [deferred recorder](../src/pi0_repro/deferred_video.py) captures a bounded, preallocated matrix of MuJoCo `mjSTATE_INTEGRATION` snapshots at the original video stride, plus the normal terminal capture. After control and worker cleanup, a separate physics clone restores/forwards each snapshot without stepping and renders the original 640×480 top camera for encoding. Composed multiview recording retains its separate existing path. State capture still costs time and memory; this option shifts work rather than eliminating total rendering/encoding cost or guaranteeing deadlines.

Each episode saves `*_video_states.npz` and `*_video_receipt.json`, including frame control steps, XML/state/pixel hashes, completion/error status, capture cost and separate `post_control_seconds`. Captured states are persisted before clone rendering; encoder failure retains the states and failure receipt rather than constituting successful video output. Post-control cost must remain separate from running-control timing, while total job duration must include it. Recorded video remains simulation-time playback, not a wall-clock stall trace. Compare camera wall time with the new `camera_thread_cpu_seconds` field; their difference does not uniquely identify GPU work, scheduling, garbage collection or a render algorithm. The [completed twelve-attempt pilot](deferred-video-timing-20261029.md) failed its adoption gates despite reduced live camera work. The option remains explicitly opt-in and off by default; it is not a recommended default.

## Optional post-apply overrun recovery and timing traces

`simulation.recover_control_overruns = true` is an opt-in deadline policy, default false, requiring `real_time = true` and `control_pacing = "before_step"`. The pure [deadline helper](../src/pi0_repro/pacing.py) preserves default arithmetic exactly. On a post-apply overrun only, recovery replaces the default finish-plus-20-ms extra idle with `max(finish, last_apply + 20 ms)`. Normal anchored-clock behavior and the existing lateness threshold remain unchanged. This does not promise that every application interval is at least 20 ms, guarantee wakeup deadlines or alter simulated time, targets, weights, RTC, controller gains or video defaults. Work performed by the next loop can consume some of the nominal idle saving.

`simulation.trace_control_timing = true` separately enables BananaBox-only diagnostic instrumentation, default false. [Environment stepping](../src/pi0_repro/banana_env.py) records physics step, forward, task-info and observation wall/thread/process CPU spans. [Simulation](../src/pi0_repro/simulation.py) records requested sleep, actual wait, oversleep, CPU spans and exact deadline-update/next-deadline fields. GC callback records include collecting-thread provenance, and hooks are removed during cleanup. Timings locate measured envelopes; process CPU can include other threads, wall-minus-CPU is not a unique scheduling diagnosis, and temporal overlap with GC does not establish causality. Instrumented comparisons must disclose the trace settings and keep them matched across arms.

Native/frozen-action diagnostics are not live policy-quality results. The [completed twelve-attempt pilot](control-overrun-timing-20261029.md) failed its covered-jerk adoption gate despite a smaller observed worst apply gap. The option remains off by default; availability does not establish adoption or uniformly improved timing/motion.
