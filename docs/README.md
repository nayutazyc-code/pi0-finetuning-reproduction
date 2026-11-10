# pi0 reproduction documentation

This index separates maintained architecture from existing operational guides and experiment evidence. Start with [architecture.md](architecture.md) for responsibilities and interfaces, or the [project overview](../README.md) for run entry points. The [workspace workflow](../../../docs/workflow.md) defines ownership and review.

[Project state recovery — 2026-10-30](project-state-recovery-20261030.md) is a dated read-only recovery snapshot covering local Git/source/data checks, completed training and evaluation evidence, missing local weights/assets, stale documentation, and remaining work; remote availability was not rechecked.

For current operation, start with the [selected final runtime](final-runtime.md): the user chose candidate999 + damping2 + RTC20 on 2026-10-30, with v6 retained as reference/fallback. This usage choice does not change the failed typical-jerk gates in [final acceptance](final-candidate-acceptance-20261030.md). The [motion-first record](motion-first-optimization-20261026.md) and the recovery snapshot describe the project at their respective dates.

## Maintained architecture and directory roles

| Location | Purpose |
| --- | --- |
| [Architecture](architecture.md) | Mac/server boundary, observation/action flow, demonstration path, and constraints |
| [Project overview](../README.md) | Concise scope, environments, and supported run entry points |
| [Source](../src/pi0_repro/) | Client, simulation, scene, demonstration, review, export, and recording implementation |
| [Server adapters](../server/) | Linux-side original-pi0 training and serving integration |
| [Configurations](../configs/) | Client, scene, collection, and training settings |
| [Decisions](decisions/README.md) | Accepted architectural choices, alternatives, and consequences |
| [Archive](archive/README.md) | Labeled previous overview; not current operating instructions |

## Maintained operational guides

| Guide | Purpose |
| --- | --- |
| [Pi0 lessons for pi0.5 fine-tuning](pi0-to-pi05-finetuning-lessons.md) | Evidence-grounded retrospective, model/adapter migration checks, and a proposed staged workflow; includes a Chinese research memo, no pi0.5 implementation or training |
| [Selected final runtime](final-runtime.md) | User-selected candidate999+damping2+RTC20 launcher, fixed-layout recording, keep-selected-service lifecycle, explicit v6 fallback and evidence limits |
| [Smooth execution candidate](smooth-execution.md) | Optional PNG sidecar, time-aligned execution, target limits, recordings, and baseline rollback; broader evaluation remains necessary |
| [Generation-time RTC and recovery workflow](rtc-and-recovery.md) | Optional sampling guidance, train-only expert corrections, phase/appearance data collection, review/merge/export, and controlled continuation; no new improvement claim |
| [Mac motion-tuning candidates](mac-motion-tuning.md) | Request cadence, arm-reference filtering, opt-in action-clock alignment and forecast-aware transitions, render/video scheduling, and diagnostics; separate candidates preserve the selected baseline |
| [Random pose and multi-view display](random-pose-multiview.md) | Optional seeded banana XY/yaw sampling and asynchronous front/side/wrist presentation; unchanged policy top camera and separate frame-age/recording evidence |
| [Long-chunk training](long-chunk-training.md) | Original-pi0 horizon extension, sequential microbatch accumulation, GPU memory/throughput probes, optimizer-update counting, and separate checkpoint evaluation |

## Existing operational guides

These guides remain in Chinese and retain their original context. Verify paths, configuration/checkpoint pairing, environment state, and remote availability before use. Their runtime statements have not been freshly verified by this documentation change.

| Guide | Purpose |
| --- | --- |
| [Environment and versions](环境与版本管理.md) | Mac environment, source pinning, and dependency setup |
| [Server deployment](../server/README.md) | Linux environment and serving profiles; contains prior deployment context |
| [Data interface](数据接口.md) | Observation/action formats and remote interface |
| [Banana scene](香蕉入盒场景搭建.md) | Scene preview and placement configuration |
| [Banana model execution](香蕉夹取与π0运行.md) | Earlier banana connection and execution instructions |
| [Expert demonstration collection](仿真专家演示采集.md) | Scripted local demonstrations and review |
| [Multiple-layout dataset](多布局演示数据集.md) | Layout generation, splits, and collection evidence |
| [LeRobot export](LeRobot数据转换.md) | Review records and dataset conversion |
| [Banana LoRA training](香蕉LoRA训练.md) | Task-specific training and deployment procedures |
| [Closed-loop evaluation](香蕉Pi0闭环评估.md) | Evaluation procedures and recorded findings |

## Current experiment evidence

| Report | Evidence scope |
| --- | --- |
| [Final candidate acceptance](final-candidate-acceptance-20261030.md) | Dated independent 24-attempt v6 versus candidate+damping2 acceptance: 10/12 versus 12/12 successes, but 16.4% typical-jerk reduction misses the frozen 20% gate; v6 retained, all videos and zero-case preparation failure preserved |
| [Smooth-execution experiments](smooth-execution-evidence.md) | Finite GUI and headless simulation comparisons, retained preliminary failures, and candidate limitations; no broad generalization or real-time guarantee |
| [RTC and quality-data experiments](rtc-quality-experiment.md) | Dated 2026-10-24–25 prefix audits, reviewed data lineage, training outcomes, frozen selection/test, and GUI measurements; results remain scoped to recorded code and layouts |
| [Mac motion validation](mac-motion-validation-20261025.md) | Dated 2026-10-25 actual-model cadence/filter comparisons, retained failures and timing interruptions, and limits of validation evidence; no promoted configuration |
| [Mac pacing validation](mac-pacing-validation-20261025.md) | Dated 2026-10-25 local timing/reference diagnosis and 18 model trials across pacing, earlier requests, filter cutoff, and chunk transitions; retained failures and unresolved user-visible pauses |
| [v6 bridge-length validation](v6-motion-validation-20261026.md) | Dated 2026-10-26 frozen-command diagnosis and separate v6 bridge8/bridge20 model observations; retained failures, physical spikes, and small-sample limits |
| [v6 RTC taper same-session comparison](v6-rtc-same-session-20261028.md) | Dated 2026-10-28 separate old RTC10 tail review and new 36-attempt RTC20/10 comparison; typical jerk regressed with RTC10, RTC20 retained |
| [v6 random-pose multi-view evaluation](v6-random-pose-multiview-20261028.md) | Dated 2026-10-28 two-phase bounded XY/yaw suite, all 12 outcomes, retained inference/local waits, display frame ages, and separate helper smoke test |
| [v6 runtime overhead and phase diagnosis](v6-runtime-overhead-20261028.md) | Dated 2026-10-28 workload/camera/window comparisons and 18-attempt validation; typical timing improved but the worst-gap gate failed, original default retained |
| [v6 progress-state recovery and timing diagnosis](v6-progress-recovery-20261028.md) | Dated 2026-10-28 train-parent diagnosis, reviewed recovery data, two continuations and separate complete 18-attempt validation; recovery gates failed, v6 retained |
| [Control fixed-layout repeats](v6-control-fixed-layout-repeats-20261028.md) | Dated 2026-10-28 ten fresh control attempts on one deliberately selected validation layout; retained no-lift failures and sampling limits, no retraining |
| [v6 versus control fixed-layout jumps](v6-vs-control-fixed-layout-jumps-20261028.md) | Dated 2026-10-28 sequential ten-versus-ten comparison on one fixed validation layout; typical jerk, physical increments and boundary disagreement show mixed results |
| [Continued-model wrist/contact optimization](continued-wrist-contact-optimization-20261028.md) | Dated 2026-10-28 exact force-aligned replay, retained failed two-trace gain screen and mixed 20-attempt plant comparison; original gate failed, unchanged controller retained |
| [Deferred top-video timing](deferred-video-timing-20261029.md) | Dated 2026-10-29 exact snapshot/pixel checks and twelve-attempt recording comparison; five gates failed despite lower live camera work, default retained |
| [Control-overrun timing](control-overrun-timing-20261029.md) | Dated 2026-10-29 native wakeup/phase diagnosis and twelve live attempts; covered jerk gate failed despite smaller observed wall tail, default retained |
| [Contact-target recovery preparation](contact-target-recovery-20261029.md) | Dated 2026-10-29 expert/learner tracking audit, retained failed offline effort screens and preselected train-only recovery acquisition; no runtime integration or training outcome |
| [Contact-recovery teacher replication](contact-recovery-replication-20261029.md) | Dated 2026-10-29 eight disjoint existing training-parent same-state expert pairs; typical takeover gate failed, no data acceptance/export/training |
| [Offline expert IK optimization](ik-expert-optimization-20261029.md) | Dated 2026-10-29 preserved tight-IK idle regression and separately gated continuous-transport teacher improvement; twenty regenerated labels accepted/exported with matched sampling, both continuations and 36 policy trials completed; candidate completion improved but jump gates failed, v6 retained |
| [Release/retreat optimization](release-retreat-optimization-20261029.md) | Dated 2026-10-29 full-cohort contact-aligned replay and failed opening-velocity screen; frozen screen passed; 48-trial factorial shows physical gains but misses motion/wall gates, independent 24-trial replication confirms physical gains but fails wall/success gates; headless experimental launcher verified, v6 default retained |
| [Latest-candidate OOD50 evaluation](ood50-evaluation-20261029.md) | Dated frozen fifty-case position/yaw/appearance evaluation relative to local fine-tuning; complete 31/50 descriptive successes, all failures retained; not a population rate or default promotion |
| [ACT baseline validation](act-baseline-validation-20261029.md) | Dated actual ACT training/probe and frozen 16-case evaluation protocol; 20k training and sixteen evaluations complete: train 0/4, validation 0/12; retained negative result and restoration, no data-quality validation claim |
| [ACT release diagnosis](act-release-diagnosis-20261029.md) | Dated label/prediction/execution alignment and corrected fixed-state branches; completed contact/prediction diagnosis, command versus physical release distinguished, original 0/16 unchanged |
| [Temporal Ensembling comparison](temporal-ensemble-validation-20261026.md) | Dated 2026-10-26 predeclared comparison of fresh v6 bridge20 controls and sparse arm-only Temporal Ensembling; separate consistency and model-outcome evidence |
| [v6 cadence optimization](v6-cadence-optimization-20261026.md) | Dated 2026-10-26 fixed request-age comparison and closure/placement diagnosis; fresh controls, separate GUI/validation evidence, and later-stage boundaries |
| [v6 jerk-reference screen](v6-jerk-screen-20261026.md) | Dated 2026-10-26 optional local jerk-reference implementation, retained offline prototypes, and separate model-screening evidence |
| [v7 recovery optimization](v7-recovery-optimization-20261026.md) | Dated 2026-10-26 train-only corrections, reviewed export, guarded v6-initialized continuation, and separate reference/evaluation boundaries |
| [RTC guidance-strength comparison](rtc-guidance-validation-20261026.md) | Dated 2026-10-26 completed six-layout guidance-cap comparison and 22-trial audit; no gain from 0.1, retained 0.5, no retraining/test-layout tuning |
| [v8 expert-displacement training](v8-temporal-training-20261026.md) | Dated 2026-10-26 expert-action audit and matched v6-initialized control/temporal-loss runs; completed paired continuations and six-layout validation (control 6/6, temporal 4/6); no temporal promotion |
| [Motion-first optimization](motion-first-optimization-20261026.md) | Dated 2026-10-26 fresh 36-attempt motion-first comparison, selected v8-control working reference, slower expert-label screens, 39 accepted dependent corrections, a complete merged/exported dataset, and a prepared-but-undeployed v9 pilot |
| [Prepared v9 motion pilot plan](../outputs/motion_first_20261026/training_plan.json) | `prepared_not_deployed` plan binding the exported merged dataset, the parent-balanced event sampler, and a 1,500-update configuration; no training run and no result |
| [Long-chunk training record](long-chunk-training-20261025.md) | Dated 2026-10-25 preparation and measurement boundary for the newly authorized horizon-100 continuation; preparation is not a completed training result |

## Maintenance evidence

| Record | Scope |
| --- | --- |
| [Local output cleanup](local-output-cleanup-20261031.md) | Dated removal of 652 older unreferenced raw-run directories after metadata archival; recent runs, named experiments, dataset dependencies, and all 638 existing documentation targets retained; deleted raw payloads cannot be replayed from summaries |
| [Cloud data-volume cleanup](cloud-storage-cleanup-20261029.md) | Dated authorized removal of five superseded intermediate checkpoints and regenerable reader cache; all v6 checkpoint files hash-verified unchanged, service checked afterward |

## Existing reports and historical planning

| Document | Evidence scope |
| --- | --- |
| [Banana success report](香蕉任务成功复现.md) | Recorded final checkpoint and limited-layout simulation results; no claim of physical-robot or arbitrary-scene generalization |
| [First training results](首轮香蕉LoRA训练结果.md) | Earlier training and evaluation outcome |
| [Training improvement record](香蕉训练改进记录.md) | Experiment-specific evolution; not the current architecture specification |
| [Fine-tuning route](香蕉场景微调路线.md) | Earlier planning context with a pointer to later results |
| [Previous overview](archive/README.previous.md) | Historical snapshot combining setup, results, and next steps |

This index is navigation; runtime evidence belongs to the dated reports and is not continuously refreshed. Preserve dated reports as evidence; update maintained architecture when the implemented system changes.
