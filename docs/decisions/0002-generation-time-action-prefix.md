# Optional generation-time action-prefix guidance

Status: accepted as an optional original-pi0 JAX serving interface. The RTC configuration remains experimental; interface acceptance does not establish a smoothness or task-success improvement.

## Context

The [background executor](../../src/pi0_repro/execution.py) aligns a replacement prediction to its observation time. During inference, the simulator continues following the old plan. Skipping an expired prediction prefix aligns time indices, but the new plan may assume a different intervening trajectory. Post-generation interpolation and target limits cannot retroactively condition that prediction.

The pinned original-pi0 model generates action chunks by integrating a learned flow. A local adapter can guide that integration without changing learned parameters or editing the upstream checkout.

## Decision

Add an explicit `pi0-rtc-v1` request/capability contract and a [server adapter](../../server/rtc_policy.py) for clean-estimate prefix guidance. The server is enabled deliberately with `--rtc`, checks the original-pi0 JAX model boundary, and advertises the capability in metadata. The client refuses an RTC configuration against a service without that declared schema.

The Mac forecasts still-valid old command targets using a copied limiter and the environment's pure actuator projection, including physical finger bounds, and sends native absolute ALOHA targets. The policy's own input transformations convert them to checkpoint-normalized model coordinates. Temporal weights constrain the 14 real action dimensions, including grippers; padding is unweighted. The server differentiates the clean estimate through the flow prediction and applies a bounded correction during generation.

The executor still owns timestamps, stale-prefix removal, one-request concurrency, and rejection after the old plan's valid overlap expires. It retains local target limits. The optional interface does not give a remote model ownership of simulator stepping or display.

For example, a prediction requested while 35 old actions remain can be guided by projected targets in that overlap and then extend the plan. A hypothetical replacement arriving after all 35 old steps have elapsed is rejected even if its own 50-step horizon is not yet exhausted: the guided overlap no longer provides a valid handoff.

## Alternatives

| Alternative | Tradeoff |
| --- | --- |
| Synchronous complete-chunk execution | Preserves the recorded baseline behavior, but blocks the simulator while requesting the next chunk |
| Background inference with time alignment only | Overlaps work and skips expired actions; does not constrain the generated replacement to the intervening old plan |
| Post-generation target blending | Cheap local smoothing, but changes commands after the model has generated them and may introduce tracking lag |
| Generation-time prefix guidance | Conditions the replacement through its flow integration and retains the existing checkpoint; adds differentiation cost and interface/timing complexity |
| Retraining for a different execution protocol | Can learn additional conditioning, but requires a separate model/data decision and is not performed by this interface |

## Consequences and limits

The adapter intentionally uses private policy/model interfaces supported only by the pinned upstream revision. Upstream changes require source review. Native prefix validation, metadata checks, padded-dimension weighting, and use of serving transforms reduce interface mistakes; they do not authenticate endpoints or weights.

Guidance is soft and its usefulness depends on latency, remaining overlap, prefix validity, and hyperparameters. The configured delay is an estimate, not measured future latency. The forecast is a command sequence, not a physical-state rollout. Hardware saturation or adverse network conditions can still cause underruns.

The first request uses zero guidance weights so the guided graph compiles during startup. Later requests add vector-Jacobian product work. It is possible for extra compute to negate a scheduling benefit; a controlled experiment must report timing as well as success and motion quality.

The baseline and earlier smooth candidate remain available. This decision changes neither training data nor checkpoint parameters. See the [operational guide](../rtc-and-recovery.md) for deployment, candidate settings, artifacts, and rollback.
