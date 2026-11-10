# Optional banana pose sampling and multi-view display

## Scope

The [random multi-view profile](../configs/banana_v6_random_multiview_pi0.toml) retains original-pi0 v6 RTC20 inference and adds independently optional scene randomization and display-only cameras. Neither feature retrains the model or establishes success on arbitrary poses. The policy still receives the original top-camera `cam_high` RGB image at 640×480, the same joint state/prompt, and existing ROI preprocessing. Front, side, and moving wrist views are for user inspection; they do not add model inputs or replace masked policy wrist views.

## Sampled scene and reproducibility

Scene `randomize_banana_pose` defaults to false. The opt-in [random scene](../configs/banana_random_scene.toml) samples banana X in 0.16–0.20 m, Y in 0.46–0.50 m, and yaw in 0–20° at each episode reset. The box stays fixed at `(0.13, 0.72, 0)` m. Banana initial Z stays 0.045 m, then physics settles freely. This is bounded horizontal placement/yaw sampling, not arbitrary 3D pose, height, roll, or pitch randomization; the settled pose can differ from the initial sample.

The environment keeps the requested base scene as `env.scene`, the realized fixed sample as `episode_scene`, and seed/sample/settled-state provenance in `last_reset`. The realized scene disables randomization for reproducible replay. Each run separately saves `episode_N_scene.json` with actual reset provenance. A known seed repeats reset sampling, not stochastic policy outputs or network history.

From the application directory, with matching original-pi0 v6 horizon-100 RTC serving, PNG sidecar, and existing local forwarding available:

```sh
bash scripts/run_mac.sh connect --config configs/banana_v6_random_multiview_pi0.toml
bash scripts/run_mac.sh run --config configs/banana_v6_random_multiview_pi0.toml \
  --seed 20261028 --episodes 3 --display
```

For a fresh seed and native display:

```sh
bash scripts/run_random_multiview.sh --episodes 3
```

The helper selects the profile and native display, choosing a fresh seed by default. Passing `--seed` disables that fresh-seed default for a repeatable sequence; `--episodes` overrides episode count. It can read a saved runtime SSH alias from `outputs/random_multiview_connection/ssh_target.txt`, or accept `--ssh-target SSH_ALIAS`, and use the existing dedicated key to create its own loopback tunnel on an available local port. It cleans up that owned tunnel when the application exits. No actual private alias or key contents belong in this guide.

Use `--port` only to reuse an existing forwarded endpoint; it cannot be combined with `--ssh-target`. Without an alias or explicit port, the helper uses the existing local endpoint from the profile. The direct launcher examples above also require existing forwarding. Follow the [transport/forwarding guide](smooth-execution.md#required-services-and-environments) for service prerequisites. A successful connection or helper smoke test establishes entry/compatibility, not random-pose reliability.

## Display composition and rendering boundary

Optional cameras are world-fixed front/side views plus a wrist view rigidly attached to the right gripper link. The [composer](../src/pi0_repro/multiview.py) builds a 1280×720 canvas: a 960×720 front view and three 320×240 sidebar cells for side view, wrist view, and status. Front input is 640×480, enlarged with nearest-neighbor sampling; repeated pixels can look blocky. Small views retain their input pixels. Occlusion can hide contacts, and status flags are task diagnostics rather than policy observations.

The [asynchronous renderer](../src/pi0_repro/async_multiview.py) is a spawned process with a read-only physics clone. The main process submits small `physics.get_state` snapshots/task flags through a bounded latest-state queue. The clone sets state and calls forward/render, never steps physics or selects actions. It publishes composed RGB through shared memory, polled without waiting for a running render. Shadows are disabled only in this display clone; main policy-camera lighting and physics remain unchanged.

The first display frame and final terminal frame use bounded waits outside running-control pacing. During control, slower rendering can produce older or repeated display snapshots. This never substitutes those snapshots for fresh top-camera inference. Consult source/target steps and frame age rather than assuming a composed view represents the current control step. The profile targets 25 fps display/video and nominal 50 Hz physical control; achieved timing depends on workload and is not a hard real-time guarantee. A short local hold-plan timing probe is not a cloud-model or random-task result.

## Optional camera lifecycle and worker window

`reuse_policy_camera = true` is a separate BananaBox-only option; it defaults to false. The environment retains the policy camera object rather than recreating it for every render, while preserving top-camera 640×480 pixels, lighting, and physics. Each call returns an independent image copy so later rendering cannot overwrite an observation already in use. Native pixel/state checks and a small actual-model comparison are recorded in [runtime overhead diagnosis](v6-runtime-overhead-20261028.md); a faster camera call is not proof that overall pauses disappear or that the default should change.

`display_in_worker = true` is another opt-in option, default false, requiring multi-view layout and the GLFW backend. It places the native window and OpenGL drawing on the renderer subprocess's main thread. Control submits snapshots and checks close/error signals instead of drawing through the main-process `GlfwViewer`. Window closure propagates through the normal `WindowClosed` path, preserving partial episode recordings and cleanup. Display frame age and bounded startup/final waits still apply; moving the window does not promise zero latency or hard real-time control. A final preferred launcher/profile must be chosen from its own verified comparison, not implied by these option descriptions.

The separate [combined runtime profile](../configs/banana_v6_random_multiview_runtime_pi0.toml) enables both options. Select it explicitly with `bash scripts/run_random_multiview.sh --runtime-optimized`; seed, episode, and connection options still apply. The helper's original default is unchanged. The dated runtime report separates initial screens from the independent 18-attempt validation: typical intervals improved, but a worse maximum interval failed the adoption gate. The bundle remains optional and is not a default upgrade or established physical-motion improvement.

## Recordings and interpretation

The original `episode_N.mp4` remains top-camera video. Multi-view recording adds `episode_N_multiview.mp4` and `episode_N_multiview.jsonl`, including source/target steps and frame age. Video sampling follows simulation time; repeated snapshots and wall-clock pauses require log interpretation. Composed frames never overwrite policy images or the canonical top recording.

Preserve every sampled/settled scene, seed, attempt, failure, and connection interruption, with the exact range and repeats. Keep old fixed validation layouts separate from new sampled layouts. Success within this bounded rectangle/yaw range would not establish arbitrary 3D generalization or physical-robot transfer. Extra display viewpoints improve inspection coverage, not model perception or proven task performance.
