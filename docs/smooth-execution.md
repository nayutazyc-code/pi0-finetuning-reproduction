# Smooth execution candidate

## Status and scope

The optional [banana_smooth_pi0.toml](../configs/banana_smooth_pi0.toml) configures execution changes around the existing original-pi0 banana policy. It is a candidate pending broader confirmation. It does not retrain the model or establish a new task success rate, and this guide is not a live service check. Keep [banana_finetuned_pi0.toml](../configs/banana_finetuned_pi0.toml) as the synchronous baseline.

The candidate combines lossless PNG camera transport, time-aligned background inference, arm-joint target acceleration limiting, wall-clock pacing, a native GLFW display with its own refresh clock, and background video encoding. The executor also supports optional arm-joint target blending; the current candidate disables it. See [architecture](architecture.md) for component boundaries and the [transport decision](decisions/0001-lossless-image-transport.md) for rationale.

## Required services and environments

Use the project's Mac environment through [run_mac.sh](../scripts/run_mac.sh). On Linux, retain the matching original-pi0 policy, checkpoint, normalization assets, and ROI configuration described in [server deployment](../server/README.md). Model compatibility checks and the 14-dimensional absolute joint/gripper action interface remain unchanged.

The policy service normally listens on remote loopback port 8000. The optional [image transport sidecar](../server/image_transport_proxy.py) listens separately on remote loopback port 8001 and forwards restored observations to port 8000. Both processes must be running for PNG requests. Starting the sidecar does not start or load the model.

The following commands assume that `PI0_ROOT` already identifies the deployed Linux project and that its existing openpi environment is at `$PI0_ROOT/openpi/.venv`. Adapt the interpreter path if the deployment uses another layout; do not install server dependencies into the Mac or workspace learning environment.

In a persistent Linux terminal, after the policy service is available:

```sh
"$PI0_ROOT/openpi/.venv/bin/python" \
  "$PI0_ROOT/server/image_transport_proxy.py" \
  --host 127.0.0.1 --port 8001 --upstream-port 8000
```

On the Mac, from this project's directory, forward the sidecar port in a separate terminal. Replace the generic SSH destination and SSH port with the existing connection settings:

```sh
bash scripts/ssh_tunnel.sh user@server 22 8001 8001
```

The helper enables SSH stream compression. If the existing connection requires a specific key, set `PI0_SSH_IDENTITY` to its local path before invoking the helper; do not copy key contents into configuration or documentation.

Keep that terminal open. This is separate from an existing port-8000 forward used by the baseline. The sidecar is a WebSocket transport endpoint, not a browser application. Do not expose its unauthenticated listener publicly.

Then inspect metadata and run the candidate:

```sh
bash scripts/run_mac.sh connect --config configs/banana_smooth_pi0.toml
bash scripts/run_mac.sh run --config configs/banana_smooth_pi0.toml
```

A successful metadata connection proves protocol compatibility only, not task completion or real-time performance.

## Configuration meaning

The linked TOML is authoritative. Parameters remain subject to candidate evaluation; the table explains their current role rather than prescribing one universally optimal combination.

| Setting | Candidate value | Meaning and limit |
| --- | --- | --- |
| `connection.port` | `8001` | Connect to the PNG-capable sidecar, not directly to the baseline service |
| `connection.image_transport` | `"png"` | Encode full-resolution camera pixels without lossy image compression |
| `model.send_full_resolution` | `true` | Preserve the 640 by 480 input needed for server-side ROI preprocessing |
| `simulation.replan_steps` | `50` | Prediction validity horizon; asynchronous replacement can execute only a suffix of a chunk |
| `simulation.async_inference` | `true` | Allow one inference request in a worker while valid actions continue |
| `simulation.prefetch_steps` | Read the linked config | Request when this many valid actions remain; each step provides 0.02 seconds of simulation-time lead |
| `simulation.real_time` | `true` | Pace toward 0.02-second intervals; retain the clock through small sleep jitter and reset after a whole missed period; no deadline guarantee |
| `simulation.max_joint_velocity` | `1.1` | Maximum commanded arm-joint target rate in radians per second |
| `simulation.max_gripper_velocity` | `1.5` | Separate rate limit on normalized gripper targets |
| `simulation.max_joint_acceleration` | `40.0` | Commanded arm-joint target acceleration limit in radians per second squared, subject to actuator bounds |
| `simulation.chunk_blend_steps` | `0` | Disabled in the current candidate; a positive value blends aligned arm-joint targets, excluding grippers |
| `simulation.display_backend` | `"glfw"` | Use the native RGB window; the application default remains `"matplotlib"` |
| `simulation.display_fps` | `50.0` | Requested maximum refresh frequency on a separately anchored draw clock; actual rate must be measured |
| `simulation.async_video` | `true` | Encode recorded frames on a bounded worker queue; a full queue can block |

For an observation captured at step 30, prediction offset 0 belongs to step 30. If its result becomes available at step 43, the executor skips offsets 0 through 12 and uses offset 13. It does not append the entire returned chunk after the old queue. If the prediction's configured horizon has already elapsed, it discards that chunk and obtains a fresh one.

These rules avoid knowingly replaying expired prefixes. They do not condition the model on actions executed during its inference delay, and prediction replacement can still change the target trajectory. The separate [generation-time RTC option](rtc-and-recovery.md) addresses that boundary with a server sampling adapter and its own explicit capability/configuration; this guide's smooth candidate remains the time-alignment-only comparison. Startup requests and horizon underruns can still wait for the server. If a request takes longer than the configured lead time, the buffer may run out; PNG compression and asynchronous execution cannot compensate for arbitrary network delay.

## Display controls

The candidate selects the [native GLFW viewer](../src/pi0_repro/viewer.py), using the project's GLFW/PyOpenGL stack. Texture uploads and event processing remain on the main thread. The viewer restores the prior OpenGL context after drawing so it can coexist with the simulator renderer, and it does not terminate GLFW globally when closing its own window. The current viewer expects the full 640 by 480 RGB frame supplied by this candidate.

Press Escape or close the native window to abort the episode. Cleanup closes inference/encoding workers and records the partial trajectory with `aborted_by_window_close`; an aborted run is not task completion. Set `simulation.display_backend = "matplotlib"` in a separate configuration to use the legacy viewer, or `simulation.display = false` for a headless run. The `--display` CLI flag can still enable display. GLFW initialization errors are explicit; change the configuration rather than assuming an automatic backend fallback.

The selected refresh value is a target, not proof of 50 FPS. Episode summaries record `display_enabled`, `display_backend`, `native_display_frames`, and `native_display_draw_seconds`. Native frame count divided by the recorded wall duration gives an episode-wide achieved average including startup waits; cumulative draw time measures drawing cost, not the whole control loop. Native counters remain zero for Matplotlib and must not be interpreted as its frame-rate measurement. Compare runs with the same backend and display setting when assessing responsiveness.

## Pixel transport and troubleshooting

The current sidecar accepts one `cam_high` RGB image, restored to the original 640 by 480 pixels. The client sends PNG bytes and a SHA-256 digest of the original CHW array; the sidecar verifies the decoded array before forwarding it. State and prompt retain their ordinary representation. The existing policy service then performs the same ROI preprocessing used by the baseline. This check detects pixel corruption or an incompatible transformation; it does not authenticate the remote endpoint.

If metadata says PNG is unsupported, check that the candidate points to the sidecar's forwarded port. Direct connections to the original policy service correctly reject PNG mode. A wrong image shape or a digest mismatch stops the request rather than silently changing the policy input. Do not substitute JPEG, a client-side crop, or a pre-resized 224 by 224 image without a corresponding interface and training-preprocessing decision.

If live motion still pauses, inspect measured round-trip latency, horizon underruns, rendering costs, and encoder queue waits before increasing training steps. Throttling display alone does not remove network waits. Lowering `replan_steps` changes policy execution behavior and is not a general solution to jitter.

## Recordings and evidence

Each run retains its configuration and server metadata. `episode_000.npz` distinguishes `requested_action` from `blended_action`, `submitted_action`, and the executed `action`. Use `action_prediction_index` and `action_prediction_offset` with `predicted_chunks` to trace raw actions; a fixed `step // 50` assumption is invalid for asynchronous execution. `prediction_steps` records the observation origin of accepted chunks.

The latency JSONL records observation and receipt steps, skipped prefix length, accepted/stale status, transport measurements, and round-trip time. The control JSONL records per-step inference blocking and deadline lateness. Episode summaries separate startup waits, running-control underruns, stale chunks, skipped prefix steps, blended steps, and video queue waits.

Video is written at simulation-time frame rate. Lossless PNG transport does not imply lossless MP4 encoding, and a smooth MP4 does not establish smooth wall-clock execution. Compare wall-clock control duration and timing logs as well as the recorded motion.

For a meaningful evaluation, keep checkpoint, scene physics, normalization, task success rule, and layout membership fixed. Record every attempted run, including failures. Compare baseline and candidate success, completion time, inference blocking, deadline lateness, and commanded versus measured motion. Report whether display was enabled: headless runs do not validate GUI responsiveness. Prior successful synchronous runs do not validate this candidate, and a single successful candidate episode does not establish a broader success rate or a causal benefit from any one combined change.

## Baseline rollback

The synchronous client configuration remains available and does not require the PNG sidecar. With its compatible policy service and port-8000 tunnel available:

```sh
bash scripts/run_mac.sh connect --config configs/banana_finetuned_pi0.toml
bash scripts/run_mac.sh run --config configs/banana_finetuned_pi0.toml
```

Retain candidate artifacts rather than overwriting the baseline's historical results. Return to the baseline when diagnosing whether a failure comes from model behavior, transport, or execution changes.

## Transport timing and echo diagnostics

The [sidecar](../server/image_transport_proxy.py) advertises `transport_proxy.timing_schema = "pi0-proxy-timing-v1"`. PNG replies retain `decode_ms` and add `request_unpack_ms`, `proxy_upstream_round_trip_ms`, `proxy_handler_ms_before_reply_pack`, and request byte count. The upstream round trip includes upstream send/receive; the handler envelope includes unpacking, decoding, packing and that round trip before reply packing. These nested intervals must not be added together. Client RTT also includes work outside these proxy envelopes. Upstream `server_timing.prev_total_ms` includes the previous loop's receive wait, so it is not a measure of current pure inference.

The same loopback sidecar offers WebSocket path `/transport-probe`: it advertises `pi0-transport-echo-v1`, accepts binary messages from 8 KiB through 1 MiB, and replies with their first 8 KiB. This bounded synthetic echo opens no upstream model connection and does not advance policy state. It diagnoses transport separately from model inference; neither a successful echo nor a small RTT establishes closed-loop timing or motion quality. Keep the existing authenticated tunnel boundary; forwarding rejection is not a reason to relax SSH permissions.
