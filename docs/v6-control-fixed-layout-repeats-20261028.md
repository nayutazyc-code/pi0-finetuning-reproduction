# Control fixed-layout repeats — 2026-10-28

## Scope and frozen method

Ten fresh inference attempts repeat the continued-control model on validation `episode_004`, deliberately selected because that model previously failed to lift there. This is a difficult-layout sampling observation, not an overall 70% reliability estimate or a cross-model comparison. The earlier single attempt and other cohorts are not pooled.

The [plan](../outputs/v6_control_repeats_20261028/plan.json), [complete ledger](../outputs/v6_control_repeats_20261028/comparison.json), [aggregate](../outputs/v6_control_repeats_20261028/aggregate.json) and [audit](../outputs/v6_control_repeats_20261028/audit.json) preserve all ten attempts. Scene and reset seed `20261027` are identical; policy sampling noise is unpaired. Relative to the prior control configuration, only the owned connection port and equivalent scene-file path changed. Fixed source hashes bind execution with H100, denoise10, RTC taper20 and client bridge20, headless with top-camera video at 25 fps. These repeats do not retrain the model or change defaults. Validation states do not become training data; test data were not used.

Implementation entry points are the [runner](../scripts/repeat_v6_control_layout.py), [analyzer](../scripts/analyze_v6_control_repeats.py) and separate [offline media builder](../scripts/review_v6_control_repeats.py). Identical reset does not pair stochastic inference samples.

## Outcomes and motion

The audit passed 252 checks, route interface remained `en0`, and the ledger verifies final restoration to v6 with its pinned normalization assets. All ten attempts had zero running inference underruns and zero inference-wait time. Seven succeeded in 21.76–23.42 simulation seconds (median 22.06); rounds 3, 4 and 6 reached 40 seconds without confirmed lift.

| Round | Outcome | Simulation seconds | Near-to-lift seconds | Active jerk p99 (rad/s³) | Physical jerk peak (rad/s³) |
| --- | --- | --- | --- | --- | --- |
| [1](../outputs/20261028-210507-1b345f/episode_000.mp4) | Success | 22.04 | 2.26 | 106.27 | 343.18 |
| [2](../outputs/20261028-210530-a36588/episode_000.mp4) | Success | 21.76 | 2.72 | 70.68 | 168.04 |
| [3](../outputs/20261028-210553-67abca/episode_000.mp4) | No lift (timeout) | 40.00 | — | 76.83 | 556.07 |
| [4](../outputs/20261028-210635-674c70/episode_000.mp4) | No lift (timeout) | 40.00 | — | 86.56 | 895.69 |
| [5](../outputs/20261028-210717-835d36/episode_000.mp4) | Success | 23.02 | 3.18 | 150.44 | 677.63 |
| [6](../outputs/20261028-210741-c9ee4f/episode_000.mp4) | No lift (timeout) | 40.00 | — | 65.28 | 395.68 |
| [7](../outputs/20261028-210823-7adcdd/episode_000.mp4) | Success | 22.28 | 2.42 | 64.35 | 321.15 |
| [8](../outputs/20261028-210846-ed71d1/episode_000.mp4) | Success | 23.42 | 2.90 | 145.48 | 2710.68 |
| [9](../outputs/20261028-210911-c44808/episode_000.mp4) | Success | 22.06 | 2.30 | 208.71 | 1590.16 |
| [10](../outputs/20261028-210935-e09710/episode_000.mp4) | Success | 21.84 | 2.64 | 65.28 | 141.77 |

Failed attempts had censored near-grasp-stage exposures of 32.82, 33.54 and 33.68 seconds respectively, not measured successful grasp times. Successful near-to-lift intervals ranged from 2.26 to 3.18 seconds. Censored exposure from first near-grasp entry is not a claim of continuously remaining within the geometric near region or being entirely stationary.

Median per-trial active frame-max jerk p99 was 81.6939 rad/s³ across all attempts versus 106.2741 among the seven transport-covered attempts. Lower motion statistics on incomplete trajectories must not be interpreted as better successful transport. The worst physical jerk peak, 2,710.6777 rad/s³, occurred in successful round 8; maximum physical joint increment was 0.0217044 rad. The longest thresholded low-motion dwell of at least 0.5 seconds was 1.06 seconds. Worst wall-clock apply interval was 96.2679 ms. Physical derivatives, displacement amplitude, task progress and wall timing describe different properties; none alone establishes perceived smoothness.

## Media review boundary

Each round label above links its original individual video; hashes remain in the aggregate. The [media manifest](../outputs/v6_control_repeats_20261028/media/manifest.json) binds the [all-ten comparison video](../outputs/v6_control_repeats_20261028/media/all10_same_layout.mp4): rounds 1–5 above 6–10, 1600×560, 25 fps, 1,001 frames and 40.04 seconds. Playback follows simulation time at 1×, with explicitly labeled terminal-frame freezing for shorter attempts. Terminal-freeze onset labels follow the sampled frame timeline and may be up to 0.02 s later than an odd terminal control step; encoded duration also includes the final 25 fps frame. This is not control-wall timing. The builder decoded and verified geometry, frame rate, frame count and hash.

The [primary visual-review receipt](../outputs/v6_control_repeats_20261028/primary_visual_review.json) records inspection of two all-outcome pages covering all ten rounds and all six stall pages, plus the grid's first frame for label readability. This is eight pages of sampled static outcome/stall review, not a claim of complete continuous playback. Media/stall pre-review snapshots remain preserved, while current manifests mark review pending false and link the receipt. The model was not promoted; final v6 restoration remains verified.

## Interpretation

This selected layout produced both successful transport and prolonged no-lift outcomes under the same control settings. It demonstrates sampling variation, not why any individual failure occurred. Historical six-layout v6/control counts are separate experiments and cannot establish superiority by comparison with these ten selected-layout repeats. The decision is observation only, with no automatic promotion, new training or default change.
