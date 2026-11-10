# Original pi0 simulation reproduction

This project runs ALOHA-style simulation, observation adaptation, demonstration preparation, and result recording on a Mac. Original pi0 inference and supervised task fine-tuning run in a separate Linux NVIDIA GPU environment using pinned upstream openpi and project-local adapters. This is simulation tooling, not a physical robot deployment or online reinforcement-learning system.

Start with the [documentation index](docs/README.md) and [current architecture](docs/architecture.md). Workspace-wide ownership and review rules are in the [documentation workflow](../../docs/workflow.md).

## Environments and model boundary

Use this project's own `.venv` through [scripts/run_mac.sh](scripts/run_mac.sh). It is separate from the workspace-root Conda environment. The pinned source is [repos/openpi](repos/openpi/README.md), recorded in [sources.lock.json](sources.lock.json); the sibling `../openpi` checkout is not this project's pinned source. Mac setup instructions are in the existing [environment guide](docs/环境与版本管理.md).

The server entry checks original `Pi0Config(pi05=False)` and the expected upstream revision. The client checks declared model metadata and action dimensions. These checks prevent configuration mistakes; they do not independently authenticate checkpoint provenance.

## Run entry points

Run these commands from this project directory after preparing its environment. Local scene preview does not require a model server:

```sh
bash scripts/run_mac.sh scene --config configs/banana_box.toml
```

For the banana task, the user-selected final workflow is **candidate999 + wrist damping2 + RTC20**, configured by [banana_final_pi0.toml](configs/banana_final_pi0.toml). The [final runtime guide](docs/final-runtime.md) explains prerequisites, recording, selection-only operation and explicit v6 fallback. The default is one headless episode on fixed validation layout004 with 25 fps video:

```sh
.venv/bin/python scripts/run_final_pi0.py --check-only
.venv/bin/python scripts/run_final_pi0.py
```

The launcher selects and verifies the remote candidate, records the run, and retains that service after normal completion while closing its own tunnel. On an execution error or interruption after selection, it attempts a verified v6 fallback. `--select-only` selects the candidate without simulation; `--fallback-v6` explicitly restores v6. Remote availability and successful fallback must be established by actual receipts.

This selection follows the user's 2026-10-30 usage decision. The [preceding acceptance report](docs/final-candidate-acceptance-20261030.md) remains unchanged: 12/12 candidate successes versus v6's 10/12, but a 16.4% typical-jerk reduction missed its frozen 20% requirement. Choosing the workflow does not turn that result into a full numerical pass. V6 remains the reference and fallback; the generic CLI default remains ALOHA simulation.

Earlier [synchronous banana](configs/banana_finetuned_pi0.toml), [smooth-execution](docs/smooth-execution.md), [RTC/recovery](docs/rtc-and-recovery.md), and [historical v3 recommended](configs/banana_recommended_pi0.toml) profiles remain available for their documented purposes. Existing random-pose helpers still use v6 and are not redirected by the new final launcher. Training and server setup remain separate; see [server deployment](server/README.md) and [training operations](docs/香蕉LoRA训练.md).

Experimental [Mac motion-tuning tools](docs/mac-motion-tuning.md) add explicit request-cadence candidates, optional arm-target filtering, demand-driven rendering, frozen-action replay, and task-progress diagnostics. Their individual settings and results remain separate from the selected final profile; adding tools does not establish improved grasping or eliminate jitter.

For the upstream ALOHA transfer-cube baseline, use [aloha_sim.toml](configs/aloha_sim.toml) with the matching server profile. The Mac simulation command supports ALOHA transfer-cube and the custom banana scene. LIBERO has an observation-file interface here; full LIBERO simulation belongs to a separate matching Linux setup.

Local forwarding exposes the remote WebSocket policy service at `127.0.0.1:8000`. It is not an HTML application. A successful metadata connection is not evidence of task success. Configurations alone do not establish current remote-service availability; recorded execution evidence is linked separately.

## Evidence and further reading

The existing [banana result report](docs/香蕉任务成功复现.md) records a particular checkpoint and limited-layout simulation evaluation. Treat its metrics and runtime statements as that report's evidence, not as a new test or a broad generalization claim. The [documentation index](docs/README.md) lists scene, data, training, evaluation, and historical materials by purpose.

Run artifacts belong in `outputs/`; demonstration and exported dataset artifacts belong in `data/`. They are not substitutes for maintained architecture. The previous overview is preserved as a labeled [historical snapshot](docs/archive/README.previous.md).
