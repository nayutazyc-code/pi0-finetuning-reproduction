# Original-pi0 long-chunk training record — 2026-10-25

## Status and scope

This is a dated preparation and measurement record for the newly authorized original-pi0 training round. It is not a completed training or model-quality result. The microbatch sweep and actual training launch are not yet finalized in this record. The earlier Mac-only rounds remain historical evidence with their original no-training scope; this new request explicitly authorizes training.

The planned [candidate](../configs/banana_finetune_long100.toml) increases the prediction horizon from 50 to 100 actions and requests 8,000 optimizer updates at effective batch 16. Microbatch 2 with accumulation 8 is provisional until measured on the actual 24 GB-class GPU. The selected v3 deployment remains the baseline. The new candidate initializes from v5 `banana_v5_quality_norm1000/999` with its matching versioned normalization; initialization is not deployment promotion.

The unchanged merged dataset has 22 training episodes/segments and 26,497 train frames, with validation and test membership preserved. No new data collection, data conversion, or normalization coordinate change is part of this preparation. Existing failed trials and v3/v4/v5 lineage are retained. See [the operational guide](long-chunk-training.md) for batch/update semantics and the measurement procedure.

## Required measurement record

The intended independent-process sweep compares microbatch 1, 2, 4, and 8 at effective batch 16, using three disposable real-data optimizer updates per candidate. Record sampled GPU peak/total memory, actual action tensor shape, finite metrics, post-compilation timings, exit status, and all failures. Selection requires at least 2,048 MiB of observed memory headroom; no measured microbatch is selected by this preparation record.

The [accumulation diagnostic](../outputs/long_chunk_20261025/accumulation-diagnostic.json) compares microbatch 2 times accumulation 2 against a batch of 4 on a deterministic tiny NNX model. It performed one optimizer update, with maximum parameter difference `1.1641532182693481e-10`, zero mean-loss difference, and zero gradient-norm difference. This verifies the small controlled calculation; it is not a pi0 quality or GPU-memory result and does not establish bitwise equivalence under stochastic model sampling.

The [first GPU probe](../outputs/long_chunk_20261025/probe_micro1.json) completed three finite optimizer updates with microbatch 1 and accumulation 16 on an RTX 4090 D reporting 24,564 MiB total memory. Actual action shape was `[16, 100, 32]`. Sampled whole-device peak was 8,686 MiB; first update including compilation took 172.908 s, while the median of the two later updates was 1.11361 s. The projected 8,000-update compute duration of about 2.475 hours excludes compilation/checkpoint overhead and is only a short-probe estimate. These results establish one feasible short probe, not the final microbatch selection or completion of the full training run.

## Outcome boundary

Final batch choice, completed update count, committed checkpoints, actual training losses, and validation outcomes await their corresponding artifacts. Do not infer them from the candidate TOML or an existing historical checkpoint. No improvement in banana grasping, box release, smoothness, or generalization is established here.
