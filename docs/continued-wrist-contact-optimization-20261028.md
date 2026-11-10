# Continued-model wrist/contact optimization — 2026-10-28

## Scope

This experiment distinguishes exact open-loop diagnosis from a completed closed-loop plant-parameter comparison. It preserves the original v6 and continued-model recordings, weights and defaults. The candidate changes simulated right-wrist position gain, not the model or task-success rule. The closed-loop observations are mixed and did not pass every predeclared adoption gate.

## Exact replay and force alignment

The [diagnostic script](../scripts/diagnose_wrist_contacts.py), [plan](../outputs/wrist_contact_diagnosis_20261028/plan.json) and [summary](../outputs/wrist_contact_diagnosis_20261028/summary.json) bind the prior continued-model round 8 and v6 round 9 peaks. Plain and instrumented full-action replay for both recordings reproduced the recorded 14-dimensional joint/gripper states with maximum absolute error zero (four replays). This does not establish equality of every unrecorded physical-state variable. Instrumentation retains target/actual q, qvel, actuator/constraint/contact force provenance. Solved forces and Jacobians align to saved pre-integration state at each 2 ms physics step; post-state is the next 2 ms sample. The original 20 ms control traces are not relabeled as 2 ms measurements.

The [analysis](../outputs/wrist_contact_diagnosis_20261028/analysis.json) and primary-inspected [contact-release figure](../outputs/wrist_contact_diagnosis_20261028/contact_release.png) show two specific mechanisms: finger/table constraint torque approximately +3 Nm opposed approximately −3 Nm actuator torque, with roughly 15–20 N contact normal force before release. After contact disappears, wrist qvel moves rapidly negative. At continued-model step 417.3 (8.346 s), wrist acceleration was approximately −500.326 rad/s², actuator torque −2.714 Nm and contact torque zero; the preceding step 417.0 retained finger/table contact. The v6 event likewise reached approximately −386.337 rad/s² following removal of finger/table constraint. These aligned events support a local controller/contact-release explanation, not a universal cause of wrist spikes or all user-visible pauses.

The [first diagnostic attempt failure](../outputs/wrist_contact_diagnosis_20261028_attempt01/failure.json) is retained: an XML trailing-newline guard mismatch stopped it before physics stepping. Successful exact replays are separate evidence, not an erasure of that failure.

## Failed two-trace screen and explicit amendment

The [frozen-action screen](../outputs/wrist_kp_replay_screen_20261028/summary.json) changed only right `wrist_angle` kp from 200 to 150. Continued-model trace peak/p99 decreased approximately 45.4%/23.2%; v6 trace peak decreased approximately 12.0%, but p99 increased approximately 18.4%. The original two-trace gate failed and remains failed. Frozen commands do not generate new policy observations, so these are open-loop plant responses, not closed-loop model outcomes.

The [continued-only amendment](../outputs/wrist_kp_replay_screen_20261028/continued_only_amendment.json) explicitly narrows the research question after that mixed screen. It does not reinterpret the original gate as passed or establish benefit on v6 weights.

## Optional plant boundary

[Scene configuration](../src/pi0_repro/scene.py) adds `right_wrist_angle_kp_scale`, default 1. A nondefault finite positive value requires `position_pd`. Scale 0.75 changes only the right wrist-angle position actuator kp from 200 to 150 (the corresponding compiled gain/bias coefficients); damping, contact, fingers and other gains remain unchanged. Four unit checks passed, including byte-identical default XML, compiled coefficient scope, invalid/profile checks and unchanged other fields. This opt-in candidate is not a recommended default. See [Mac operation](mac-motion-tuning.md#optional-right-wrist-position-gain).

## Closed-loop comparison — complete, original gate retained

The [twenty-attempt plan](../outputs/continued_wrist_kp_ab_20261028/plan.json), [ledger](../outputs/continued_wrist_kp_ab_20261028/comparison.json), [aggregate](../outputs/continued_wrist_kp_ab_20261028/aggregate.json) and [audit](../outputs/continued_wrist_kp_ab_20261028/audit.json) preserve ten fresh attempts per arm, with alternating AB/BA order and identical continued-model checkpoint, requested scene/reset and source hashes. Policy noise was unpaired. All 445 audit checks passed; final restoration metadata verified v6. Historical open-loop traces and earlier cohorts are not pooled.

| Measure | Baseline kp200 | Candidate kp150 |
| --- | --- | --- |
| Success / transport coverage | 8/10 | 10/10 |
| Median active jerk p99, all attempts (rad/s³) | 92.6572 | 97.3037 |
| Median active jerk p99, covered attempts (rad/s³) | 119.8871 | 97.3037 |
| Worst physical jerk peak (rad/s³) | 1,371.1601 | 1,165.2444 |
| Worst physical joint increment (rad) | 0.0220801 | 0.0220112 |
| Longest thresholded low-motion dwell ≥0.5 s (s) | 0.98 | 0 |
| Covered median duration (simulation s) | 22.34 | 21.93 |
| Worst apply gap (ms) | 113.382 | 183.490 |

Baseline rounds 2 and 6 failed without confirmed lift; all outcomes remain retained. Covered typical jerk decreased 18.84% and worst peak decreased 15.02%, while worst physical increment changed only −0.31%, essentially unchanged. However, all-attempt typical jerk increased 5.01%. Different coverage (eight versus ten) requires these statistics to remain separate; failed trajectories cannot be silently removed or low movement treated as successful smooth transport. All twenty runs had zero running inference underruns and zero inference-wait time.

The predeclared all-attempt p99 gate failed. The decision remains `retain_unchanged_controller`; favorable covered statistics do not justify changing the gate after seeing results or promoting a default. The candidate merits further controlled replication, not a claim of improved training weights, general task reliability or uniformly smaller jumps. Four right-wrist-gain, six wrist-damping and four motion-quality unit checks passed (14 total); the default field and controller remain unchanged.

## Wall-gap diagnosis

The [wall-gap receipt](../outputs/continued_wrist_kp_ab_20261028/wall_gap_diagnosis.json) records candidate profile round 5, step 466: a 183.490 ms gap followed a 1.928 ms environment wall / 1.926 ms thread-CPU span. Current preparation took 181.512 ms, including 173.447 ms camera time and 7.780 ms inference dispatch, with zero inference blocking. Baseline's 113.382 ms gap included previous environment wall/CPU 33.039/6.355 ms and camera time 79.258 ms. These inclusive wall envelopes identify where delay was observed, not a unique render-algorithm or controller cause. Zero inference underruns does not mean zero local timing stalls.

## Media review boundary

The [primary review receipt](../outputs/continued_wrist_kp_ab_20261028/primary_visual_review.json) records inspection of the exact force plot, both AB plots, four all-outcome event pages covering all twenty attempts, and the combined first frame for label readability. This is sampled static review, not complete continuous video playback. Contact-analysis, plot, per-arm media and comparison manifests preserve their pre-review snapshots; current manifests mark pending false and link the receipt.

The [distribution plot](../outputs/continued_wrist_kp_ab_20261028/plots/jump_distributions.png), [extreme trajectories](../outputs/continued_wrist_kp_ab_20261028/plots/worst_wrist_trajectory.png) and [plot manifest](../outputs/continued_wrist_kp_ab_20261028/plots/manifest.json) preserve independent extreme selections. Baseline's global extreme is joint 11 (right wrist angle), whereas the candidate's is joint 10 (right forearm roll); these are neither the same joint nor paired events. The exploratory [all-twelve-axis supplement](../outputs/continued_wrist_kp_ab_20261028/per_axis_motion.json) shows target joint 11's worst peak falling from 1,371.1601 to 660.1289 rad/s³ (−51.86%), while an adjacent axis still spikes. This supplement does not replace the global adoption gate or demonstrate that all motion problems are solved.

The [twenty-attempt comparison video](../outputs/continued_wrist_kp_ab_20261028/media/all20_kp200_vs_kp150.mp4) and [comparison manifest](../outputs/continued_wrist_kp_ab_20261028/media/comparison_manifest.json) retain all outcomes, baseline kp200 above candidate kp150, with rounds 1–5 above 6–10 inside each block. Decoding verified 1600×1200, 25 fps, 1,001 frames and 40.04 seconds. Playback follows 1× simulation time with labeled terminal-frame freezing, not wall-clock execution. Encoder/source hashes remain bound by the manifest. Final restoration checkpoint and normalization metadata were reverified for v6.


Baseline kp200 original videos: [Round 1](../outputs/20261028-221026-bddf6d/episode_000.mp4), [Round 2](../outputs/20261028-221136-35a18b/episode_000.mp4), [Round 3](../outputs/20261028-221218-7add88/episode_000.mp4), [Round 4](../outputs/20261028-221328-7293da/episode_000.mp4), [Round 5](../outputs/20261028-221352-701446/episode_000.mp4), [Round 6](../outputs/20261028-221518-7666c8/episode_000.mp4), [Round 7](../outputs/20261028-221600-2244c1/episode_000.mp4), [Round 8](../outputs/20261028-221709-a9acad/episode_000.mp4), [Round 9](../outputs/20261028-221732-bbebec/episode_000.mp4), [Round 10](../outputs/20261028-221844-d5e011/episode_000.mp4).

Candidate kp150 original videos: [Round 1](../outputs/20261028-221049-0acaa6/episode_000.mp4), [Round 2](../outputs/20261028-221113-5e49b6/episode_000.mp4), [Round 3](../outputs/20261028-221242-b34692/episode_000.mp4), [Round 4](../outputs/20261028-221305-9d8a8b/episode_000.mp4), [Round 5](../outputs/20261028-221430-1d7fbf/episode_000.mp4), [Round 6](../outputs/20261028-221453-6352a3/episode_000.mp4), [Round 7](../outputs/20261028-221623-211033/episode_000.mp4), [Round 8](../outputs/20261028-221646-48ef2d/episode_000.mp4), [Round 9](../outputs/20261028-221757-b584e0/episode_000.mp4), [Round 10](../outputs/20261028-221820-44d165/episode_000.mp4).
