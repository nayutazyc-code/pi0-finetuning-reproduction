# Mac action-clock and display validation — 2026-10-25

## Scope and evidence status

This dated round investigates the user's continuing reports of fine jitter and visible pauses in both the native window and MP4 after the earlier [motion-candidate validation](mac-motion-validation-20261025.md). It adds an opt-in Mac action-application clock and a separate 50 fps profile. Original-pi0 v3 weights, training, datasets, physical controller, and success criteria are outside this change. The existing recommended and filtered25 profiles remain unchanged.

This round is complete with controlled local diagnosis and 18 actual-model trials. The [model comparison](../outputs/mac_pacing_20261025/model_comparison.json) preserves per-run evidence and stage identity. The counts below describe separate unpaired experiments and must not be pooled into a candidate success rate. No single root cause for all reported pauses is established.

| Actual-model stage | Runs | Observed outcomes |
| --- | --- | --- |
| Prototype GUI | 3 | paced50 2/2; legacy25 1/1 |
| Final-clock GUI | 3 | paced50 2/2; legacy25 1/1, including a paced50 running interruption |
| Earlier-request GUI | 2 | paced15 2/2 |
| Initial limited validation | 4 | paced25 1/2; paced15 2/2 |
| 5 Hz response GUI | 2 | responsive15 1/2; direction stopped |
| Quintic-transition GUI | 2 | continuous15 2/2 |
| Quintic-transition limited validation | 2 | continuous15 1/2 |

Local hold-plan/reference checks are excluded from these 18 actual-model runs. Weights and the original filtered25 configuration remain unchanged; the selected recommended profile was not replaced. No training, held-out testing, or system-proxy changes were performed.

## Diagnostic basis and implementation boundary

The [protocol](../outputs/mac_pacing_20261025/protocol.json) records source snapshots before the change and the preceding filtered25 GUI recording. End-of-step intervals in that recording had alternating even/odd medians of 28.47 and 11.74 ms. In the prior loop, rendering/display/video preparation preceded the action and sleep followed physics; variable preparation cost could therefore shift action-application times even when average control throughput stayed near 50 Hz. This observation motivated controlled local timing diagnosis rather than attributing every pause to network latency or policy indecision.

The new [paced25 configuration](../configs/banana_mac_paced25.toml) preserves filtered25's 25-step request age, 3 Hz arm-target filter, RTC settings, and v3 metadata pins. It selects `control_pacing = "before_step"`, 50 Hz display, and 50 fps video. [Configuration validation](../src/pi0_repro/config.py) keeps `after_step` as the default and requires real-time pacing for the opt-in mode. [The simulation loop](../src/pi0_repro/simulation.py) prepares the current state's pixels/observation and target on the main thread, waits immediately before applying the target, then steps physics. At display rates of at least 50 Hz in this mode, it attempts one draw per control step. In the final implementation, an application miss above 2 ms shifts the next slot to the actual application start plus 20 ms, with a further reanchor if the loop has already passed that slot. This avoids compensating a long frame with a short action interval. The native viewer limits status-title updates to at most 4 Hz and records context, texture-upload, title, and swap durations. Per-step camera/display/video durations are retained. No physics thread, frame interpolation, controller change, or automatic grasp rule is introduced.

Fresh inference pixels and state still describe the same current simulation step. Optional waiting before physics can add wall-clock observation-to-application delay without substituting an older simulation state. Same-step rendering remains shared between inference/display/video where applicable. These scheduling changes cannot guarantee network deadlines, eliminate task-phase hesitation, or establish hard real-time execution.

## Prototype evidence and final implementation boundary

The initial local timings and [prototype GUI ledger](../outputs/mac_pacing_20261025/gui_trials.json) remain separate records. The prototype GUI comparison had two successful paced50 trials with application-interval p95 of 20.82 and 21.58 ms, versus 29.50 ms in the legacy25 trial. Display-interval medians were 19.63 and 19.54 ms versus 40.10 ms. Early long frames above 100 ms remained, so these prototype observations did not establish complete removal of visible pauses. The subsequent title-update limit, stage instrumentation, and late-slot handling define the final implementation; prototype and final results must not be pooled.

## Prototype local hold-plan timing diagnosis

The [local timing record](../outputs/mac_pacing_20261025/local_timing.json) compares three 300-step runs using a hold-plan source and real local MuJoCo. This source is not pi0 and does not evaluate model quality or task success.

| Local condition | Action-application interval p95, ms | End-of-step interval p95, ms | Running underruns |
| --- | --- | --- | --- |
| Legacy after-step, 25 fps | 30.30 | 30.50 | 0 |
| Before-step, 25 fps | 20.63 | 20.82 | 0 |
| Before-step, 50 fps | 20.41 | 20.44 | 0 |

The [command comparison](../outputs/mac_pacing_20261025/local_command_comparison.json) records zero maximum executed-command difference among the three runs. This controlled check supports the action-clock change as a way to reduce timing variation from the local loop on this machine without changing these hold commands. It does not prove equivalent stochastic closed-loop model trajectories or that every observed visual pause has the same cause.

## Final local hold-plan timing diagnosis

The [final local timings](../outputs/mac_pacing_20261025/final_local/local_timing.json) repeat the same three 300-step conditions after the final implementation changes. Action-application interval p95 was 29.26 ms for legacy25, 20.65 ms for paced25, and 20.58 ms for paced50. End-of-step interval p95 was 29.60, 20.73, and 20.56 ms respectively. All three had zero running underruns. The [final command comparison](../outputs/mac_pacing_20261025/final_local/local_command_comparison.json) again records zero maximum executed-command difference. These are controlled local hold-plan results, separate from actual-model GUI and validation outcomes.

## Final-implementation GUI checks

The [final GUI ledger](../outputs/mac_pacing_20261025/final_gui/gui_trials.json) records two paced25-profile runs at 50 fps and one legacy filtered25 run at 25 fps. All three completed the physical task. These use the final title-update and deadline handling implementation; they are separate from the prototype ledger above.

| Final GUI condition | Action-application interval p95, ms | Maximum application interval, ms | Display-interval median, ms | Running underruns |
| --- | --- | --- | --- | --- |
| paced50, repeat 1 | 20.61 | 878.20 | 19.84 | 1 |
| legacy25, contemporary control | 22.04 | 40.59 | 40.06 | 0 |
| paced50, repeat 2 | 20.80 | 81.55 | 20.03 | 0 |

The final contemporary legacy control has a 22.04 ms application-interval p95; the prototype's 29.50 ms must not substitute for it. The 50 fps candidate's typical display interval was approximately 20 ms, compared with approximately 40 ms for the 25 fps control. Its first run nevertheless had a 1,143.14 ms round trip, one stale replacement, and 0.86668 s of running inference wait. The maximum reported server-inference time in that run was 128.26 ms. Action and display intervals reached 878.20 and 886.08 ms respectively, so the clock/display change did not eliminate long pauses. This evidence separates a typical cadence improvement from remaining latency tails without locating every source of the excess delay. Read-only routing checks did not establish that system proxy settings caused these tails.

## Earlier-request candidate

The separate [paced15 configuration](../configs/banana_mac_paced15.toml) retains paced25's final implementation, 3 Hz arm filter, and 50 fps display/video, changing only request age from 25 to 15 steps. This leaves 35 old-plan steps (0.70 simulated seconds) instead of 25 (0.50 s). More overlap may absorb some reply delays, but does not eliminate long-tail latency or guarantee grasp behavior. Its filtered, clock-aligned semantics differ from the preceding unfiltered cadence15 experiments.

The [early-request GUI ledger](../outputs/mac_pacing_20261025/early_gui/gui_trials.json) records two successful paced15 runs with zero running underruns. Action-application interval p95 was 20.54 and 20.65 ms, with maxima of 30.92 and 74.72 ms; median display interval was 19.97 and 19.82 ms. These are a separate pair of unpaired model trials, not repeat measurements of the earlier paced25 trajectories. The user still reported obvious pauses after the 50 fps/earlier-request changes. These timestamp distributions do not invalidate that feedback or establish that the remaining pauses are caused by policy indecision.

## Limited validation regression checks

The [validation ledger](../outputs/mac_pacing_20261025/validation_trials.json) retains one run per profile on each of two preassigned validation layouts. They are a limited regression check, not a new broad success estimate.

| Layout | paced25, 3 Hz | paced15, 3 Hz |
| --- | --- | --- |
| episode_026 | Success | Success |
| episode_020 | Failure | Success |

All four had zero running underruns. The paced25 failure on `episode_020` remains a task failure despite normal clock/availability diagnostics. Clock regularity and physical task reliability are separate requirements.

## Higher-cutoff response candidate

The [responsive15 configuration](../configs/banana_mac_responsive15.toml) changes only the filter cutoff from 3 to 5 Hz relative to paced15. It tests whether reduced arm-target response lag is useful while preserving early requests and the final action clock. Passing more rapid target variation may increase fine motion; it is not an automatic fix for visible pauses. Its two GUI runs produced one success and one failure; this direction was stopped without further 5 Hz validation. Exploratory forward-kinematics motion thresholds do not establish the absence of human-visible pauses and are not used to resolve the user's continuing report.

## Chunk-transition candidate and continuing user feedback

The user further described a pause while holding the banana above the box, segmented movement, and random displacement near segment endings. This narrows the visible symptom but does not independently prove a chunk-switch cause. The remaining concern is not resolved by the typical application/display timing improvements above.

The separate [continuous15 configuration](../configs/banana_mac_continuous15.toml) retains paced15's 3 Hz filter and selects an eight-step quintic arm-reference transition. The executor's pure `ActionReference` samples valid old and new plans at the same simulation timestamp; an unfinished previous crossfade remains part of the old reference. Grippers come directly from the new raw plan. Actual execution and RTC forecasting share `_target_for_step`, followed by the same limiter state and actuator projection, so forecasted overlap can include a transition. The assigned bridge is limited by the remaining valid old-plan steps and disabled below two; expired actions are never extended.

The quintic weight's first and second derivatives vanish at its endpoints. This property does not guarantee C2 continuity of the time-varying discrete reference or physical robot motion. The baseline remains zero client-blend steps, legacy blend shape defaults to linear, and RTC still rejects nonzero linear blending. Actual per-chunk bridge lengths are recorded in `chunk_transition_steps`.

The [reference checks](../outputs/mac_pacing_20261025/transition/reference_checks.json) preserve zero raw/reference error over 1,032 legacy unblended steps, zero nested-transition first-arm and endpoint errors, zero gripper/raw-reference error, and disabled bridging when one old step remains. A separate [frozen-chunk local pipeline](../outputs/mac_pacing_20261025/transition/local_pipeline.json) ran 250 steps in real MuJoCo: 16 prefix comparisons had maximum error zero, four overlapping transitions were observed, and 106 steps used a blended reference. These are local reference/forecast consistency checks, not model task-success evidence.

The [transition GUI ledger](../outputs/mac_pacing_20261025/transition/gui/gui_trials.json) records two actual-model successes, with zero running underruns. Simulated completion times were 19.44 and 17.96 s, near-to-pinch intervals 2.02 and 2.38 s, measured joint-jerk p95 51.96 and 54.24 rad/s³, and application-interval p95 20.46 and 20.53 ms. Maximum application intervals were 49.21 and 58.01 ms.

| Transition GUI trial | Raw switch jump p95, rad | Blended-reference switch jump p95, rad | Executed-command switch jump p95, rad |
| --- | --- | --- | --- |
| continuous15, repeat 1 | 0.10039 | 0.01679 | 0.01620 |
| continuous15, repeat 2 | 0.13085 | 0.01765 | 0.01636 |

For this table, each switch's jump is the maximum absolute difference among the 12 arm joints across adjacent control steps, followed by the p95 over switches. This differs from pooling all joint/switch entries in the ordinary continuity summary. The raw-to-reference reduction describes reference seams within these runs, not an equivalent reduction in physical jerk. Earlier unbridged paced15 GUI runs had raw/reference switch-jump p95 of 0.06642 and 0.07155 rad, executed-command values of 0.01444 and 0.01060 rad, and measured joint-jerk p95 of 55.13 and 60.16 rad/s³. The transition runs followed different unpaired model trajectories: larger raw jumps do not mean equally large executed jumps, and the observed physical-jerk change is modest relative to the reference-seam reduction. The user had not confirmed that the new recording resolved the above-box pause when this round ended.

The [transition validation ledger](../outputs/mac_pacing_20261025/transition/validation_trials.json) records success on `episode_026` with near-to-pinch 2.14 s, but failure on `episode_020` without any confirmed lifted pinch. Its near-to-pinch duration is undefined. Both had zero running underruns and application-interval p95 approximately 20.01 ms. Thus the candidate had two GUI successes and one success/one failure in the limited validation block. These are three successes in four observations across distinct conditions, not evidence sufficient for default promotion; regular timing and lower reference jumps did not ensure successful grasping.

## Timestamp interpretation

New control logs record application-start time, scheduled deadline, pacing wait, pacing mode, and deadline phase. Legacy `after_step` lateness measures the loop-end timestamp; `before_step` lateness measures action-application start. Those lateness values are not directly comparable measurements of one event. The [summarizer](../scripts/summarize_banana_execution.py) separately reports `action_apply_interval_ms` and the existing end-of-step `control_interval_ms`.

Completed draws are recorded in `episode_000_display.jsonl`. Display interval diagnostics exclude draws before the first action, including startup waits. Completed-draw timing does not measure actual monitor presentation. A 25 fps recording samples every other control step; 50 fps samples each step. MP4 presentation follows simulation time, so wall-clock inference pauses are not reproduced as equivalent video-frame durations. Measured physical jerk, action timing, display timing, inference underruns, and policy progress remain distinct evidence.

## Final recording integrity

The [integrity audit](../outputs/mac_pacing_20261025/integrity.json) checked all 18 model runs: model metadata matched, states and commands were finite, raw-prediction reconstruction and action timestamp errors were zero, and PNG pixels were verified. Across 1,302 model-run forecast-prefix comparisons, maximum forecast-versus-executed command error was zero. Final source hashes matched their recorded values, the original filtered25 configuration was unchanged, and the pinned upstream worktree remained clean. This supports command/recording consistency without establishing task reliability, monitor presentation timing, or a subjective solution.

## Limits and selection

Typical action-application timing improved in controlled local checks, and the 50 fps candidates typically drew approximately every 20 ms. A recorded 0.86668 s running interruption and other long frames remain in the evidence. Quintic transitions reduced the measured reference seam within their GUI runs, while physical-jerk differences from the earlier unbridged runs were modest and unpaired. One transition validation trial failed before any confirmed lifted pinch. Neither smooth timing nor a small reference seam establishes a solved task or removes policy-level hesitation.

The user's reported segmented motion and above-box pause remain unresolved by confirmed subjective feedback. No new profile is promoted; the prior historical reports and selected recommended profile remain unchanged. This round ended without retraining, held-out testing, or proxy changes. A possible next investigation is to identify the user's exact pause interval and compare raw policy/gripper intent, accepted-chunk boundaries, and actual execution there; controlled generation-sampling or RTC-parameter comparisons would be proposals requiring their own evidence, not work completed here.
