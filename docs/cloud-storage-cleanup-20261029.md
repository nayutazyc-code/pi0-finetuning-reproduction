# Cloud data-volume cleanup — 2026-10-29

This is a completed, user-authorized maintenance record, not an experiment or a standing deletion policy. The user required complete preservation of every v6 checkpoint file. The [frozen cleanup plan](../outputs/cloud_cleanup_20261029/plan.json) and [execution receipt](../outputs/cloud_cleanup_20261029/receipt.json) bind the actual scope; historical experiment records were not rewritten.

## Removed and retained items

| Removed intermediate checkpoint | Retained newer committed sibling |
| --- | --- |
| `banana_pilot_v1/1000` | `1999` |
| `banana_v3_roi2000/1000` | `1999` |
| `banana_v5_quality_norm1000/500` | `999` |
| `banana_v7_recovery1000/500` | `999` |
| `banana_v8_temporal32_3000/2000` | `2999` |

The Hugging Face datasets reader cache was also removed; it can be reconstructed from retained original Parquet data. Model/tokenizer caches, raw data, environments, source, logs and reports were retained. Current IK control/candidate finals, v8 control `2999`, and every other group's existing latest checkpoint were retained. No additional removal is authorized by this record.

The five v6-named checkpoint trees across the data/root volumes contained 206 files whose individual SHA256 digests were unchanged before and after cleanup. The original `banana_v6_long100_accum8000` tree retained all 37 files, including complete parameter/train-state shards, assets, metadata and hidden files. Protecting only its selected `7999` path would not have met the user's full-tree requirement; the recorded verification covers every protected file.

## Space and service verification

Allocated removed bytes totaled 35,925,487,616. Actual data-volume free space increased by 35,925,483,520 bytes, from 4,748,509,184 to 40,673,992,704 bytes. Allocated size and measured free-space gain are separate filesystem quantities.

The subsequent [read-only service check](../outputs/cloud_cleanup_20261029/service_verification.json) verified original pi0 v6/7999 metadata, H100, fixed normalization digest and RTC schema, with the same serving process still present. It also confirmed original data and the latest candidate remained available. This is a dated successful check, not a guarantee of future service availability. No training, model evaluation or policy/runtime change was performed as part of cleanup. The release/retreat diagnosis remains a separate task.
