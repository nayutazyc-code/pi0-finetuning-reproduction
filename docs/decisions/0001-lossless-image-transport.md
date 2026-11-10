# Optional lossless image transport

Status: accepted as an optional transport architecture. The smooth-execution configuration remains a candidate pending broader evaluation; accepting this interface does not establish a performance gain or promote it over the raw baseline.

## Context

The banana ROI policy consumes the original 640 by 480 RGB observation before applying its trained server-side crop. The Mac sends a top camera image, joint state, and prompt to a Linux policy service. Network transfer is part of end-to-end inference latency. Changing input pixels or moving preprocessing can also change policy behavior, making a transport comparison difficult to interpret.

The existing [policy entry](../../server/serve_pi0.py) loads the pinned original-pi0 model. Its interface and upstream source should not need a codec-specific change merely to support an optional link optimization.

## Decision

Add a separate loopback [PNG sidecar](../../server/image_transport_proxy.py). The [Mac client](../../src/pi0_repro/remote.py) selects PNG explicitly and requires capability metadata before sending compressed observations. The sidecar restores the RGB array, verifies its SHA-256 digest, and forwards the ordinary raw observation to the unchanged policy service. Raw transport continues to work directly and remains the synchronous baseline.

The supported compressed interface is deliberately narrow: one `cam_high` image with RGB uint8 pixels at 640 by 480 resolution, reconstructed in CHW layout. State and prompt are forwarded normally. The ROI policy receives the same pixels and retains responsibility for its preprocessing. Operational details are in the [smooth-execution guide](../smooth-execution.md).

For example, a full-resolution banana observation can cross the network as PNG and still reach the ROI adapter as the exact original array. A hypothetical client that crops it to 224 by 224 before PNG encoding would violate this interface and the server's full-resolution requirement; that would be a preprocessing change, not the accepted transport choice.

## Alternatives

| Alternative | Tradeoff |
| --- | --- |
| Direct raw MessagePack | Simplest existing path, no codec or extra process; remains available as the baseline, but sends the complete raw image payload |
| JPEG transport | Can reduce payload size with lossy compression, but changes camera pixels and introduces a separate policy-input evaluation question |
| Client-side ROI cropping | Can reduce transmitted pixels, but moves preprocessing responsibility and requires an explicit matching interface and crop contract |
| Codec handling inside the policy server or upstream source | Removes the proxy hop but couples a transport option to model serving; the sidecar keeps that responsibility separate |
| Lossless PNG sidecar | Preserves pixels and the existing policy interface, at the cost of encoding/decoding work, an extra process, and another forwarded port |

## Consequences and limits

PNG size depends on image content; compression alone does not guarantee a lower round-trip time. Encoding, decoding, the proxy hop, the network path, and model execution all contribute to latency. The client and sidecar report transport measurements so an experiment can distinguish payload savings from actual timing changes.

Pixel digests check integrity, not endpoint authenticity. Keep both listeners on loopback and use the existing SSH transport rather than publicly exposing these unauthenticated WebSocket endpoints. A direct policy connection without PNG capability correctly fails the compressed-mode handshake. Unsupported shapes or digest mismatches fail explicitly rather than silently modifying the observation.

This decision does not change model weights, action semantics, normalization assets, success criteria, or the original ROI preprocessing. It also does not resolve stale-action handling or controller smoothness; those remain executor responsibilities. An operational success claim requires separately recorded runs with their actual model, scene, transport, and execution configuration.
