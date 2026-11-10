# Selected final runtime

## Selection and evidence boundary

On 2026-10-30, the user explicitly selected **candidate999 + wrist damping2 + RTC20** as the current final banana-task workflow. Use [run_final_pi0.py](../scripts/run_final_pi0.py) with [banana_final_pi0.toml](../configs/banana_final_pi0.toml). This is a usage decision, not a claim that every previous acceptance requirement passed.

The [frozen final acceptance](final-candidate-acceptance-20261030.md) remains unchanged: candidate completed 12/12 tasks versus v6's 10/12, while typical physical jerk improved 16.4%, short of the required 20%. Its numerical decision was `retain_v6_reference`. The later user choice adopts the combination for operation despite that limitation. V6 remains the reference and explicit fallback. Earlier experiment reports, plans and metrics retain their original decisions.

## Fixed profile

| Setting | Selected value |
| --- | --- |
| Original-pi0 checkpoint suffix | `banana_v6_ik_candidate1000/999` |
| Normalization SHA256 | `6c6e7a92643c76996bf15ba9fc3d3e346dd6bc3d8697712118d9fdb49dedf269` |
| Plant | `position_pd`, wrist damping scale 2 |
| Scene | Fixed validation layout004 from [the paired scene](../configs/banana_ik_continuous_damping2_scene.toml) |
| Default reset seed | `20261027` |
| Prediction / nominal request interval | 100 actions / 15 control steps |
| RTC | Delay 10, generation-time taper 20, guidance cap 0.5 |
| Client arm transition / filter | Quintic bridge 20 steps / 3 Hz |
| Transport | Full-resolution top-camera PNG, with server-side ROI |
| Default run | One headless episode, at most 2,000 steps / 40 simulation seconds |
| Recording | Top-camera video at 25 fps, plus actions, task and timing logs |

The new configuration has the same numerical settings as the evaluated candidate profile. The launcher checks their parsed equality locally. This fixed-scene workflow does not sample random positions or constitute an OOD evaluation. Changing only a seed does not turn the fixed scene into a randomized distribution. Additional GUI workload is outside the headless acceptance evidence.

## Prerequisites

Use the application-local `.venv`, the pinned `repos/openpi` client and the existing authenticated cloud setup. The saved SSH-target record, configured dedicated key, compatible remote policy selector, both checkpoint/normalization assets, and PNG sidecar source must already be available. Connection identifiers and credentials belong in the existing private setup, not this guide. [Server operations](../server/README.md) describe the separate Linux environment.

The launcher can start the known PNG sidecar when its port is unused. It preserves an existing listener and lets metadata validation determine compatibility; it does not kill an unfamiliar process. It does not provision a cloud machine, train a model or reconstruct missing checkpoint assets. It records the current route, including TUN if present, without changing VPN settings. A dynamic local forwarding port avoids relying on the older fixed forwarding ports.

## Commands

Run from the application directory. First validate the local profile without SSH, model calls, simulation or receipt writes:

```sh
.venv/bin/python scripts/run_final_pi0.py --check-only
```

Run the selected workflow and save one video:

```sh
.venv/bin/python scripts/run_final_pi0.py
```

Select and verify the candidate service without stepping simulation:

```sh
.venv/bin/python scripts/run_final_pi0.py --select-only
```

Explicitly return to v6 and verify its metadata without stepping simulation:

```sh
.venv/bin/python scripts/run_final_pi0.py --fallback-v6
```

`--display` enables the native viewer, but GUI performance has not been accepted by the headless experiment. `--episodes N` and `--seed VALUE` override run count and reset seed; they do not change the fixed scene or establish a new evaluation protocol. The three mode flags are mutually exclusive.

## Lifecycle and receipts

The launcher validates local configuration, records a dated runtime receipt directory, checks or starts the known sidecar, opens its own tunnel, selects the candidate and verifies the expected model/checkpoint/normalization/RTC metadata and denoise count. It then runs simulation unless a selection-only or fallback mode was requested.

**After normal workflow completion, the selected remote service remains running.** The launcher closes its owned local tunnel; the next invocation opens another tunnel. A completed episode that fails the task-success criterion still counts as normal workflow completion and does not automatically switch the policy. `completion.json` workflow `success` and its per-episode task `success` are distinct fields.

If an exception or interruption occurs after policy-selection dispatch, the launcher attempts v6 selection and metadata verification before cleanup. It records fallback failure explicitly and propagates the original error. Window closure is treated as an aborted simulation and enters this recovery path. Network loss, abrupt process termination or host shutdown can prevent recovery; read the actual receipt instead of assuming fallback succeeded. Errors before a policy switch do not require changing the remote model.

The terminal prints `Runtime receipts:` and `Saved run:`. Runtime receipts are stored under `outputs/final_runtime_<timestamp>-<id>/`; the simulation has its own output directory. The runtime directory records configuration/scene, route, proxy handling, selection dispatch and metadata, logs, and `completion.json`. For failures, inspect `fallback_verified`, `fallback_error` and any cleanup error. The simulation directory contains `summary.json`, the episode video and action/task/timing artifacts. Dispatch alone is not a verified service, and a service handshake is not task success.

The recording is simulation-time playback. Use the control logs to assess actual application intervals and inference waiting. Neither a smooth video nor a configured 50 Hz control period guarantees wall-clock continuity.

## Earlier entry points

- [run_release_candidate.py](../scripts/run_release_candidate.py) is the older experimental launcher; it restores v6 on every exit and retains its direct-route requirement. It is not the selected final entry.
- Existing random-pose helpers retain their v6 behavior. Use the new final launcher explicitly for candidate operation; this change does not silently redirect those helpers.
- `configs/banana_recommended_pi0.toml` remains the historical v3 profile. Its filename is not the current selection authority.
- The generic `pi0_repro` CLI still defaults to `configs/aloha_sim.toml`. The final workflow is an explicit launcher, not a global CLI-default change.

## Verification scope

Fourteen [launcher tests](../tests/test_final_pi0_launcher.py) and the local `--check-only` invocation passed. These include isolated mocked operating-system behavior; they do not establish live cloud recovery under every fault. The fallback configuration path is independent of the candidate configuration, so a candidate-local profile failure does not itself prevent loading the v6 fallback profile.

The **2026-10-30 19:07 Asia/Shanghai** default-entry smoke is recorded in [runtime completion](../outputs/final_runtime_20261030-190708-a3adc8/completion.json). One headless episode completed successfully in 980 steps / 19.6 simulation seconds, with grasp, lift and box-entry progress. The [recording](../outputs/20261030-190733-79e15d/episode_000.mp4) and [simulation summary](../outputs/20261030-190733-79e15d/summary.json) remain separate from the formal 24-attempt acceptance cohort.

The launcher reported `selected_model=candidate`, `selected_metadata_verified=true` and `kept_selected_service=true`. A separate [post-run metadata readback](../outputs/final_runtime_20261030-190708-a3adc8/post_run_metadata.json) independently verified that candidate999 and the expected normalization were still served after launcher completion. This validates the new entry's normal keep-selected-service integration in that run, not continuing service availability or population reliability.

The smoke used `utun4`, whereas the earlier acceptance cohort used `en0`; the outcomes and timing are not pooled. Its [proxy receipt](../outputs/final_runtime_20261030-190708-a3adc8/proxy.json) records `existing=true`, `started=false`. Therefore this live smoke reused the proxy and did not validate live proxy cold-start creation. GUI mode was not tested. Successful sampled operation does not reverse the historical two failed acceptance gates, and neither fallback under hard termination nor stall-free future execution is guaranteed.
