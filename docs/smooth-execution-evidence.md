# Smooth execution: recorded evidence

Recorded on 2026-10-24. This report covers finite local simulation runs of the existing original-pi0 banana policy. The smooth configuration remains a candidate. These observations do not establish reliable unseen-layout generalization, a real-robot result, a hard real-time guarantee, or an improvement in every motion metric.

## Scope and source artifacts

The changes concern image transport, action execution, and display. No model retraining was performed for this optimization. The model remained the existing `banana_v3_roi2000/1999` checkpoint with the existing ROI preprocessing, normalization assets, absolute 14-dimensional ALOHA action interface, scene controller, and physical grasp/containment/release/stability success rule.

Primary records:

- [Recorded comparison](../outputs/execution_optimization_20261024/recorded_comparison.json): per-run configurations, episode results, latency, action provenance, and motion summaries, including preliminary failures.
- [Final GUI run](../outputs/20261024-173817-1e1ab5/summary.json), [configuration/source record](../outputs/20261024-173817-1e1ab5/run.json), [video](../outputs/20261024-173817-1e1ab5/episode_000.mp4), and [replay analysis](../outputs/20261024-173817-1e1ab5/analysis/metrics.json).
- [Synchronous GUI control](../outputs/20261024-173328-4f72ab/summary.json) and its [configuration](../outputs/20261024-173328-4f72ab/run.json).
- [Final fixed-code layout suite](../outputs/banana_evaluations/20261024-174028-bddf0a/evaluation.json).

Run folders, rather than this document, retain exact parameters and source hashes. The [operations guide](smooth-execution.md) describes the current candidate and baseline rollback; the [architecture](architecture.md) describes responsibilities without adopting the results as guarantees.

## Selected candidate and comparison conditions

The final GUI candidate used PNG transport, a 50-step prediction validity horizon, asynchronous prefetch with 20 steps of lead, commanded arm-joint acceleration limit 40 radians per second squared, no chunk blending, wall-clock control pacing, background video encoding, and the GLFW viewer requested at 50 FPS. Arm-joint and gripper target velocity limits remained 1.1 radians per second and 1.5 normalized units per second respectively.

The synchronous GUI control also used PNG transport, wall-clock pacing, and background video encoding. It used synchronous requests, no joint-target acceleration limit, and the earlier Matplotlib display at 20 FPS. It was therefore a comparison control, not an unchanged run of the original raw-transport baseline. Its source snapshot predates the native viewer. Both runs completed the default scene, but different action histories, model sampling, network jitter, and display backends confound single-factor attribution.

| Recorded measure | Synchronous GUI control | Final asynchronous GLFW candidate |
| --- | ---: | ---: |
| Default-scene task success | Yes | Yes |
| Simulation steps | 986 | 1165 |
| Simulated duration | 19.72 s | 23.30 s |
| Recorded control wall duration, including startup | 23.124 s | 24.487 s |
| Running-control inference waits | 19 | 3 |
| Total running-control inference blocking | 3.039002 s | 0.346703 s |
| Startup wait | See run summary | 0.252067 s |
| Control-interval 95th percentile | 44.778 ms | 30.397 ms |
| Maximum recorded control interval | 206.865 ms | 207.838 ms |
| Control intervals above 100 ms | 19 | 3 |
| Request round-trip median | 158.278 ms | 218.959 ms |
| Request round-trip 95th percentile | 170.792 ms | 446.670 ms |
| Median MessagePack observation payload | 39,232 bytes | 40,581 bytes |
| Commanded joint acceleration maximum | 63.214898 rad/s² | 40.000081 rad/s² |
| Commanded joint acceleration 95th percentile | 14.045679 rad/s² | 16.288925 rad/s² |
| Commanded joint path, summed absolute change | 12.950035 rad | 17.508773 rad |
| Measured-state joint acceleration maximum | 55.516064 rad/s² | 57.557821 rad/s² |

This pair shows fewer and shorter explicit inference stalls, a lower control-interval 95th percentile, and enforcement of the configured target acceleration ceiling within floating-point precision. It does not show faster overall task completion, a shorter path, lower acceleration across the distribution, or lower physical joint acceleration. The maximum control interval remained about 0.208 seconds. The candidate also encountered higher network latency in this pair, so the stall comparison is not a controlled measurement of transport speed.

For the final candidate, the native viewer recorded 1,145 draws and 9.277783 seconds of cumulative drawing time. Dividing draws by the 24.487-second recorded control duration gives approximately 46.8 draws per second, including startup and inference waits. This is an episode-wide measured average, not proof of uninterrupted 50 FPS or a 50 Hz deadline guarantee. The earlier native-window run [20261024-173708-9732b2](../outputs/20261024-173708-9732b2/summary.json) completed the task but exposed an approximately 25 FPS draw-clock issue; the reported final run followed its correction.

The older raw synchronous run [20261024-141702-febed5](../outputs/20261024-141702-febed5/summary.json) had a commanded-joint acceleration peak of approximately 75.606 rad/s². That is historical context, not the synchronous comparison value in the table and not a paired estimate of the candidate's benefit.

## Metric interpretation and integrity

Joint-motion statistics exclude the two gripper channels and use the 0.02-second simulation interval. For commanded joint targets, finite differences are defined as:

\[
\dot q_t^{\mathrm{cmd}}=
\frac{q_t^{\mathrm{cmd}}-q_{t-1}^{\mathrm{cmd}}}{0.02},
\qquad
\ddot q_t^{\mathrm{cmd}}=
\frac{q_t^{\mathrm{cmd}}-2q_{t-1}^{\mathrm{cmd}}+q_{t-2}^{\mathrm{cmd}}}{0.02^2}.
\]

These are target-trajectory measures in simulation time. They are not wall-clock finite differences or guarantees about actuator acceleration. The path statistic sums absolute consecutive changes across arm-joint channels; it is not Cartesian end-effector path length. State-derived statistics describe the recorded physical state separately.

For the final GUI run, the recorded raw action-to-prediction mapping maximum error was zero and the action timestamp-offset error was zero. The sidecar reported verified full-resolution RGB pixel digests. Exact pixel restoration precedes the existing ROI preprocessing; this is a transport-integrity observation, not endpoint authentication. The replay analysis reported zero maximum pre-action state error and zero maximum banana-position error. Replay checks the recorded trajectory with its original submitted actions; it does not rerun policy inference or establish success on another layout.

## Preliminary outcomes and failures retained

The comparison manifest contains 25 completed recorded episodes across historical, comparison, and changing candidate configurations: 20 successes and five task failures. This pooled count is provenance, not a success-rate estimate for one fixed policy/executor configuration. The following records remain part of the evidence. Later successes do not erase them.

Two interrupted attempts are retained separately and excluded from those completed-episode counts: the [raw-transport attempt](../outputs/20261024-143423-bceae4/summary.json) ended with `KeyboardInterrupt` after 686 steps, and the [PNG connection attempt](../outputs/20261024-171751-9327dc/summary.json) ended with a WebSocket `ConnectionClosedError` after 110 steps during network-route changes. Their `success=false` fields do not represent completed task-limit failures. The comparison manifest notes that subsequent comparison runs used the direct network route instead of TUN; route changes are another timing confound.

| Attempt or suite | Observed outcome | Scope and limitation |
| --- | --- | --- |
| [Initial PNG smoke suite](../outputs/banana_evaluations/20261024-171207-0d774a/evaluation.json), acceleration 20, blend 3, prefetch 20 | Default and one training layout both completed | The two runs spent about 72.65 and 74.27 seconds waiting after startup; successful completion did not mean smooth execution under that network route |
| [First direct-route validation suite](../outputs/banana_evaluations/20261024-171932-a8a5e3/evaluation.json), acceleration 20, blend 3, prefetch 20 | 4/5 overall; 3/4 validation layouts | Validation episode 019 failed after grasp/lift; the banana did not satisfy complete containment after release |
| [Prefetch-10 retry on validation episode 019](../outputs/20261024-172337-7c8547/summary.json), acceleration 20, blend 3 | Failed at the 2,000-step limit | Reducing lead time did not solve that attempt; do not label it an improvement |
| [Targeted episode-019 run](../outputs/20261024-172457-cfc7a9/summary.json), acceleration 40, blend 0, prefetch 20 | Completed in 969 steps | One successful retry was insufficient to establish layout reliability |
| [Second direct-route validation suite](../outputs/banana_evaluations/20261024-172632-03f704/evaluation.json), acceleration 40, blend 0, prefetch 20 | 4/5 overall; 3/4 validation layouts | Episode 019 failed again despite the earlier successful retry |
| [Synchronous episode-019 comparison](../outputs/20261024-173241-92d648/summary.json), without acceleration limiting | Failed at the 2,000-step limit | The layout failure was not observed only with asynchronous execution; this does not identify its cause |
| [Earlier paced Matplotlib default-scene candidate](../outputs/20261024-172944-0391fe/summary.json) | Completed in 858 steps | A single default layout and earlier display implementation; not the final GLFW evidence |

The two preliminary validation suites each passed the default layout, episodes 004 and 012, and episode 020, while episode 019 failed. These are finite recorded attempts with reused development layouts. They must not be presented as 5/5 suites or independent estimates from a fresh test set.

## Final fixed-code layout suite

The [final suite record](../outputs/banana_evaluations/20261024-174028-bddf0a/evaluation.json) uses the final clock logic and the selected acceleration-40, blend-0, prefetch-20 execution settings. It is headless: the configured backend is GLFW, but `display_enabled` is false and native frame counters are zero. It checks layout behavior, not GUI responsiveness.

| Layout | Recorded run | Outcome | Steps |
| --- | --- | --- | ---: |
| Default center | [174028-f7d291](../outputs/20261024-174028-f7d291/summary.json) | Success | 980 |
| Validation episode 004 | [174049-fcf9a3](../outputs/20261024-174049-fcf9a3/summary.json) | Success | 1055 |
| Validation episode 012 | [174110-0837a5](../outputs/20261024-174110-0837a5/summary.json) | Success | 796 |
| Validation episode 019 | [174127-869c4c](../outputs/20261024-174127-869c4c/summary.json) | Success | 950 |
| Validation episode 020 | [174147-a8f4c0](../outputs/20261024-174147-a8f4c0/summary.json) | Failed; no confirmed grasp/lift | 2000 |

Overall: 4/5 completed successfully, including 3/4 validation layouts. The last layout reached the configured step limit. This suite changed which validation layout failed; it did not establish that all known layouts were reliable. Previous failure of episode 019 and later success on that same layout must both remain visible.

## Interpretation and remaining work

The recorded default-scene GUI run demonstrates that the candidate can complete the task while reducing explicit inference stalls in this comparison. The evidence supports keeping it available as an optional local simulation configuration. The synchronous baseline remains available because task behavior, network timing, and target smoothing changed together, and validation still contains failures.

Useful next evaluation would repeat fixed configurations across all development layouts and network conditions, report every attempt, and compare task success alongside physical motion, path length, and timing. Controller tuning and model sampling can affect contact and release behavior; a lower commanded acceleration peak alone is insufficient to judge grasp quality. The existing held-out test results were not rerun or updated in this optimization. No new held-out test evaluation or retraining is claimed here. The reused validation layouts are development evidence, not an untouched generalization benchmark.
