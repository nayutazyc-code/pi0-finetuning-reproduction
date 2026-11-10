# Original-pi0 v6 runtime overhead and phase diagnosis — 2026-10-28

## Scope and status

This round separates runtime workload from grasp/placement progress while fixing original-pi0 v6 `7999`, horizon 100, RTC20 guidance taper, client bridge20, filter, scene/controller, and normalization. It does not retrain the policy, change the task rule, or infer a solved motion problem from a faster local loop. The initial nine-attempt workload comparison, native camera checks, and six-attempt camera-reuse comparison are complete. The first six-attempt worker-window screen is also complete, with mixed timing evidence. The separate predeclared 18-attempt combined-option validation is complete. It improved typical wall-clock intervals but failed the worst-interval gate; retain the existing default and leave the runtime bundle explicitly optional.

## Initial workload protocol

The [frozen plan](../outputs/v6_runtime_cost_20261028/plan.json), [complete ledger](../outputs/v6_runtime_cost_20261028/comparison.json), and [runner snapshot](../outputs/v6_runtime_cost_20261028/runner_source.py) preserve three diagnostic seeds, each tried once under three workloads. Seeds 2026102801, 2026102802, and 2026102805 were selected because earlier recordings showed ordinary, slow-grasp, and no-lift behavior; this is diagnostic selection, not population sampling. Policy noise and latency histories are unpaired.

| Workload | Included work |
| --- | --- |
| Control only | Fresh original top-camera observations and actual cloud inference, without GUI or saved video |
| Top video | Control workload plus 25 fps top-camera recording, without GUI |
| Multi-view | Top-video workload plus native GUI, spawned read-only display renderer, and a second 25 fps multi-view recording |

The three seeds use cyclic order control/video/multi-view, video/multi-view/control, and multi-view/control/video. The multi-view condition is a workload bundle, not an independent estimate of GUI, rendering, memory exchange, or encoding effects. All outcomes, inference warnings, and operational exceptions remain recorded; source/metadata/timestamp/prefix corruption would abort rather than silently rerun. No automatic policy promotion follows the screen.

Running action-application intervals exclude initial acquisition and post-loop teardown. Control-only still renders fresh policy-camera pixels and sends real model requests; it is not a camera-free or inference-free benchmark. Per-stage timers are nested/inclusive, so summing camera, observation, dispatch, video submission, and exchange as exclusive slices would double-count work. Different model trajectories/contact states and network histories can also change run costs. A worst interval from three runs cannot identify a unique expensive component or establish a general throughput guarantee.

## Initial workload observations

The [aggregate](../outputs/v6_runtime_cost_20261028/aggregate.json) and [241-check audit](../outputs/v6_runtime_cost_20261028/audit.json) retain all nine attempts. All audit checks passed and no running inference underrun occurred.

| Workload | Observed completions | Median trial application-interval p99, ms | Worst application interval, ms | Gaps above 50 ms |
| --- | --- | --- | --- | --- |
| Control only | 2/3 | 22.47 | 73.36 | 1 |
| Top video | 3/3 | 20.89 | 45.87 | 0 |
| Multi-view bundle | 1/3 | 22.66 | 152.47 | 3 |

The top-video sample did not have the largest observed intervals, despite adding work. This illustrates why these small stochastic, diagnostically selected samples do not isolate component costs or prove video improves timing. Pooled percentiles weight longer episodes; per-trial summaries remain separate. Zero inference underruns did not eliminate the local long intervals or establish task progression.

## Native camera lifecycle and actual-model comparison

Three native probe attempts are retained. The first two had nearby test/media activity and are not used for the final timing conclusion. The [final independent frozen-replay probe](../outputs/policy_camera_probe_final_20261028/probe.json) checked 105 legacy/persistent render pairs: all pixels and physics states matched exactly and replay-state error was zero. Legacy median render time was 8.1099 ms versus persistent 6.5705 ms; the median of paired per-render savings was 1.9305 ms. This is same-state native camera evidence, not model task quality.

The [six-attempt camera comparison](../outputs/v6_camera_reuse_ablation_20261028/aggregate.json) and [175-check audit](../outputs/v6_camera_reuse_ablation_20261028/audit.json) compare legacy and reusable camera paths on the selected seeds with actual model inference. All checks passed and both profiles completed three of three attempts with no inference underruns. Median per-render camera cost fell from 7.6077 to 5.5494 ms, but median trial application-interval p99 increased from 29.874 to 30.833 ms, and gaps above 50 ms increased from three to six. The largest single interval fell from 168.13 to 96.75 ms. These mixed observations support a camera-call efficiency finding without proving that reuse removes interruptions or justifying automatic replacement of the legacy default.

## Separate phase-progress review

The [old random-suite review](../outputs/v6_stall_review_20261028/review.json) diagnoses all 12 earlier random-pose recordings and selects six extreme grasp/box events. It remains separate from the nine new runtime attempts. Recorded gripper intent, applied targets, confirmed pinch, box entry, release/support/stability, and inference waits must be distinguished before labeling a pause. Slow or repeated movement can delay phase completion without satisfying a low-motion threshold, and a required stable hold is not an unintended stall.

The [primary visual-review receipt](../outputs/v6_stall_review_20261028/visual_review.json) records inspection of all six selected sheets. Near-grasp sheets show the gripper around the banana for several sampled simulated seconds without confirmed lift in those frames; post-lift sheets show prolonged slow approach/descent near the box while holding the banana. Static sampled frames do not reproduce wall-clock waits or identify a unique policy, contact, display, or encoding cause. Missing force/torque/contact detail and limited camera sampling constrain interpretation.

## Read-only continuation-data progress audit

The [training-progress audit](../outputs/training_progress_audit_20261028/audit.json) passed all 585 checks over a later continuation dataset: 77 training segments/episodes, 84,701 frames, 32 complete demonstrations and 45 recovery segments, from 32 distinct parent layouts. These are not 77 independent layouts, and this dataset must not be presented as v6's actual training dataset. Parent multiplicity, phase durations, and motion diagnostics identify a read-only review queue rather than automatic label rejection.

Twenty-two older segments in this continuation dataset lack near-grasp geometry, so their corresponding waiting measures are unavailable, not zero. Low-motion intervals may be deliberate closure/release/settling. This audit did not delete, trim, retime, resample, export, relabel, or train anything; it does not convert validation diagnostics into training labels or establish a new checkpoint result.

The separate [actual-v6 dataset audit](../outputs/v6_training_progress_audit_20261028/audit.json) passed 157 checks over 22 segments and 26,497 frames: 20 complete demonstrations plus two recovery segments from 20 parent layouts. All 22 lack near-grasp geometry and must report unavailable proximity waits rather than zero. V6's [selected training configuration](../outputs/long_chunk_20261025/selected_training.toml) used microbatch 8 times accumulation 2, effective batch 16, and 8,000 updates, initializing from v5. This lineage is distinct from the later 77-segment continuation dataset. The latter's [parent-lineage cross-check](../outputs/training_progress_audit_20261028/lineage_check.json) confirms all 13 recovery-parent IDs belong to its 32 complete training-layout IDs; dependent corrections still do not add independent layouts. Both audits are read-only and did not trim data or launch training.

## Worker-window screen and independent combined validation

Both `reuse_policy_camera` and `display_in_worker` are implemented opt-in flags with legacy defaults false. The latter requires multi-view plus GLFW and puts the native window in the renderer subprocess's main thread. Control then checks close/error signals and submits snapshots without executing native draw calls. A close signal follows the normal partial-save/cleanup path. The final relevant checks passed in their respective post-change runs: 15 multi-view, two native-camera, eight random/camera, and three selector checks. Three integration checks also passed after the thread-CPU diagnostic field was added. These establish structural boundaries, not an actual-model timing or subjective improvement result for the worker window. The separate [worker-display ledger](../outputs/v6_worker_display_ablation_20261028/comparison.json), [aggregate](../outputs/v6_worker_display_ablation_20261028/aggregate.json), and [175-check audit](../outputs/v6_worker_display_ablation_20261028/audit.json) now cover six completed actual-model attempts. All checks passed and each profile completed 3/3, with zero running underruns. Reused-camera/parent-window versus reused-camera/worker-window median trial application p99 fell from 30.936 to 21.855 ms and the worst interval from 312.093 to 130.833 ms. However, gaps above 50 ms increased from 11 to 12. These mixed results must not be reduced to the favorable percentile or peak alone.

The independent [combined validation plan](../outputs/v6_runtime_optimized_validation_20261028/plan.json) fixes three diagnostic seeds times three repeats times two profiles: legacy camera plus parent GUI versus reused camera plus worker GUI. Noise and latency remain unpaired. Its predeclared runtime adoption gate requires at least 10% lower median trial p99, no worse worst interval or above-50-ms gap rate, zero underruns, no lost coverage, at most 10% longer covered duration, and at most 10% higher covered physical p99. The completed 18-attempt [ledger](../outputs/v6_runtime_optimized_validation_20261028/comparison.json) is separate from the six-attempt screen and the initial workload/camera comparisons; those counts are not pooled into this validation.

The random multi-view helper selects the [combined runtime profile](../configs/banana_v6_random_multiview_runtime_pi0.toml) through explicit `--runtime-optimized` selection. The original default remains available and unchanged. No training or model change is introduced. The combined validation failed one adoption gate, so availability is explicitly optional rather than a new default.

The completed workload/camera comparisons, worker-window experiment, and old phase review remain distinct. Preserve failed/unfinished stages and the user's observed pauses; neither zero inference underruns nor a low median control interval proves absence of visible interruptions. The original default is retained; the combined runtime profile remains available only by explicit selection.

## Completed combined-option validation and decision

The [final aggregate](../outputs/v6_runtime_optimized_validation_20261028/aggregate.json), [475-check audit](../outputs/v6_runtime_optimized_validation_20261028/audit.json), and [selection receipt](../outputs/v6_runtime_optimized_validation_20261028/selection.json) cover all 18 retained attempts. Every audit check passed. The predeclared gate returned `retain_existing_default`; only the worst application-interval gate failed.

| Measure | Legacy camera + parent window | Reused camera + worker window |
| --- | --- | --- |
| Attempts | 9 | 9 |
| Median trial application-interval p99, ms | 27.882 | 21.245 |
| Pooled application-interval p99, ms | 33.537 | 22.672 |
| Gaps above 50 ms | 30 | 6 |
| Gaps above 50 ms per 1,000 intervals | 2.476 | 0.551 |
| Worst application interval, ms | 129.150 | 165.563 |
| Gaps above 100 ms | 2 | 2 |
| Running inference underruns | 0 | 0 |
| Transport-covered attempts | 7 | 8 |
| Covered median simulated duration, s | 22.68 | 21.95 |
| Covered physical active-jerk p99 median, rad/s³ | 154.857 | 156.451 |
| Task completions, secondary | 7/9 | 8/9 |

Typical wall-clock intervals and the normalized above-50-ms rate improved, but the worst gap grew and physical active jerk did not improve. Neither more completions in this small unpaired sample nor better median timing overrides the failed tail gate. This is not evidence that physical jumps or all visible pauses were eliminated.

The worst worker-window event was trial index 5, run `20261028-152642-9747af`, application step 559. Its 165.56 ms interval includes approximately 144.45 ms environment-step wall time from preceding step 558: post-application time about 144.53 ms plus current pre-application time about 21.03 ms. Inference blocking and main-process display time were zero there. This locates a long environment-execution envelope without proving that MuJoCo calculation, a solver/contact event, or CPU computation caused it; the 18 attempts had no thread-CPU measurement.

After the validation ended, `environment_step_thread_cpu_seconds` was added using `time.thread_time` as a diagnostic field only. It changes no action/model/physics behavior and was not present in these 18 recordings. The extra helper smoke/CPU evidence below remains separate rather than being retroactively attributed to this validation. The runtime bundle is retained as an explicit option, with the original default unchanged and no training or policy promotion.

## Extra helper smoke with retained timing warning

The [helper receipt](../outputs/v6_runtime_optimized_validation_20261028/helper_smoke.json) records an extra run, [20261028-153744-a8024d](../outputs/20261028-153744-a8024d/summary.json), outside the 18-attempt cohort. It exited 0, completed the physical task in 21.36 simulated seconds, and drew 488 native frames in the worker. The owned local endpoint refused connections after exit, and all 12 recorded source-file hashes matched the source at verification time. The helper matched the validation candidate configuration except for its local connection port.

This was an entry/cleanup and new-diagnostic smoke test, not a timing pass. It had one running inference underrun with 0.924918 s wait, worst application interval 933.148 ms, and maximum round trip 1293.466 ms versus 156.247 ms reported server inference for that request. These measurements do not independently locate all excess time or prove a proxy/network cause. Public routing was observed, without a private endpoint being needed in this report. Do not replace the validated cohort's zero-underrun count with an “all runs zero” statement or hide this additional warning.

The smoke's largest measured environment wall interval was 15.168 ms with 3.618 ms thread CPU. Their difference does not uniquely distinguish scheduling, other-thread work, or I/O. It cannot retrospectively explain the earlier 144 ms environment envelope, which had no thread-CPU field. The explicit `--runtime-optimized` name remains an experimental option label, not proof of comprehensive optimization.

## Reconstructed progress from v6's full expert demonstrations

The [offline replay record](../outputs/v6_expert_progress_replay_20261028/replay.json) and [aggregate](../outputs/v6_expert_progress_replay_20261028/aggregate.json) reconstruct all 20 full demonstrations from v6's actual training lineage. Recorded joint states and targets matched exactly and final banana-position error was zero. The two recovery segments require restoration snapshots and were explicitly skipped rather than counted as failed or equivalent full demonstrations.

Reconstructed near-region entry to confirmed lift had median 2.66 and maximum 3.16 simulated seconds; lift to first containment had median 9.62 and maximum 10.58 s. Right-gripper reversals before lift had median/maximum one, not zero. That event alone does not label an expert action as bad or uniquely explain learner delay. These are new offline diagnostic flags, not replacement historical raw fields: old missing near-grasp geometry remains unavailable in the original records. Exact recorded joint/target/final-object-position replay is not proof that every historical physical state and camera pixel was reproduced. No labels, raw data, independent layouts, exports, or training were created by the reconstruction.

## Prepared next-stage boundary

The [next-stage plan](../outputs/v6_next_stage_20261028/plan.json) is `prepared_not_collected_or_trained`. It retains original pi0 v6 RTC20/client bridge20 and the existing default runtime. Remaining local-tail work should use newly captured wall/thread-CPU and inference timing rather than reattribute the unmeasured historical event. Subsequent recovery acquisition is proposed only on v6's 20 eligible full-training parent layouts; v8-derived corrections are not automatically added to v6.

Inspection triggers of 4.16 s near-to-lift and 12.58 s lift-to-containment derive from expert maxima plus margins. They select records for inspection, never control policy actions or establish intent. Validation states are not recovery labels. Review/preflight must precede any final training batch/update choice, and loss, batch, and update count are not to be changed together as an undocumented intervention. No new recovery collection or training is claimed by this proposal. Typical timing improved in the completed runtime comparison, but worst delays and physical/stage hesitation remain unresolved.
