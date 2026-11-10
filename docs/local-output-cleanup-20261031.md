# Local output cleanup — 2026-10-31

This is a completed, dated maintenance record for the authorized cleanup of local `outputs/`. It does not revise experiment outcomes, checkpoint selection, or the [selected final runtime](final-runtime.md).

## Scope and retention decision

The user requested retaining the latest few versions of results. The cleanup removed only timestamped raw-run directories older than 2026-10-30 that were not explicitly referenced by the audited documentation, dataset/assets, or application source/configuration paths. The [frozen plan](../outputs/cleanup_20261031/plan.json) records the selection and protected paths; the [file manifest](../outputs/cleanup_20261031/files.json) records each removed file's path, size, modification time, and SHA-256.

The retention policy preserved:

- All named experiment cohorts, including their existing reports, receipts, audits, source snapshots, and hash records.
- All 2026-10-30 timestamped raw runs and the latest three final-runtime receipt directories with their associated raw runs, including the incomplete receipt and associated run.
- The complete [24-attempt final acceptance](final-candidate-acceptance-20261030.md), its original zero-case preparation failure, preflight evidence, videos, and NPZ trajectories.
- Every existing local output target found in the 64-document audit, dataset/source dependencies, and the connection record required by the final launcher.

Deleting all older outputs would have removed directly linked experiment media and data-lineage dependencies. Retaining all raw runs would not have met the storage-reduction request. The chosen boundary keeps those dependencies and recent evaluation evidence while removing older unreferenced raw payloads. For example, an older video directly linked by an experiment report remains; an unreferenced older timestamped run can lose its video and trajectory while its compact metadata survives in the archive.

## Archive and deletion procedure

Before deleting the selected directories, compact metadata was collected into [old_run_metadata.tar.gz](../outputs/cleanup_20261031/old_run_metadata.tar.gz). The retained suffixes are `.json`, `.toml`, `.xml`, `.txt`, `.log`, `.md`, `.py`, `.sh`, and `.exit-code`. The cleanup verified every archived member against its source hash before deletion and retained the [archive verification receipt](../outputs/cleanup_20261031/archive_verification.json).

Videos, per-step JSONL logs, NPZ trajectories, and image frames in the selected directories were deliberately excluded from the archive. **Archived summaries are not replay evidence.** They cannot reconstruct the deleted control trajectory, recover a recording, or support a new frame-by-frame audit. Older manifests and dated statements about raw-run availability retain their historical meaning; the cleanup does not rewrite them into claims of current availability. The deletion manifest identifies which original payloads are now absent.

## Completed result

The [completion receipt](../outputs/cleanup_20261031/completion.json) records:

| Measure | Completed value |
| --- | ---: |
| Removed raw-run directories | 652 |
| Removed original files | 7,503 |
| Removed original file bytes | 2,289,595,599 |
| Archived compact metadata files | 3,552 |
| Compressed archive bytes | 3,964,742 |
| Net file bytes reclaimed after archive | 2,285,630,857 |

The net figure is approximately 2.29 GB / 2.13 GiB and subtracts the metadata archive from the removed file bytes. It is not a filesystem block-accounting measurement. The rounded directory-size observation changed from approximately 3.1 GiB to 1.0 GiB.

## Validation and limits

The [post-cleanup validation](../outputs/cleanup_20261031/validation.json) records that all 638 previously existing documentation output targets remain present, the latest acceptance raw runs remain intact, `run_final_pi0.py --check-only` passed, the runtime connection record remains present, and archive hashes were verified. The latest acceptance check included all 24 videos and NPZ trajectories.

The local launcher check validates configuration and local prerequisites; it is not a new inference run or evidence of a currently reachable remote service. No remote operation or training was performed. Data, assets, model weights, application source, and runtime configurations were outside the cleanup scope and were unchanged. No historical experiment report or original hash record was rewritten.

This record describes the local storage state at completion on 2026-10-31. It does not establish continuing remote availability or make the deleted raw payloads recoverable from compact summaries.
