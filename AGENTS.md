# pi0 reproduction project guidance

## Entry points

- Start with docs/README.md and docs/architecture.md.
- This application separates Mac simulation/data preparation from Linux GPU inference and fine-tuning. Do not imply real-robot control, online reinforcement learning, or live remote service availability without new evidence.
- Read source and configuration to establish current behavior; prior run reports establish only their recorded experiment outcomes.
- Use this project's own runtime and scripts/run_mac.sh for application commands. Do not install its dependencies into the parent study environment.
- Treat repos/openpi as pinned upstream source. Do not edit it for documentation housekeeping.

## Documentation ownership

- Assign a documentation subagent at the start of development, before implementation; it may read immediately but writes only after the primary agent approves the affected topics and paths.
- Implementation subagents edit assigned code/tests only. They must not edit Markdown documentation, README files, docs/, AGENTS.md, or .codex/agents. Report documentation suggestions and code evidence to the primary agent.
- The primary agent reviews suggestions and delegates approved document changes to documentation_owner. It owns governance configuration and final validation.
- The documentation owner may edit approved documentation only, and must not edit code, tests, configurations, dependencies, generated data, or results.
- Use project .codex/agents definitions when available; otherwise assign these restrictions explicitly in the subagent prompt. If delegation is unavailable, report that and separate the two passes under primary-agent review.

## Documentation contract

- New and substantially rewritten maintained docs are English. Preserve existing Chinese operational guides and experiment records unless explicitly migrating them.
- docs/architecture.md describes current components, boundaries, interfaces, data flow, constraints, and rationale; link to source instead of repeating implementation.
- README is an entry point, not a running status log. Keep commands in operational guides and results in clearly scoped experiment reports.
- Historical snapshots under docs/archive are not current architecture or live deployment status. Do not append new status updates to them.
- Clearly label proposals and superseded decisions. Use one or two real examples or explicitly hypothetical counterexamples to explain important architectural choices. Do not invent past decisions.
- Keep docs/README.md updated with short descriptions of each guide and documentation directory. Add an index for new documentation directories.
- Never put credentials or private connection details in documents.

## Verification

- From the parent workspace, run `.venv/bin/python scripts/check_docs.py`. If working only in this standalone checkout, verify changed Markdown links, TOML role files, and documentation/code consistency directly; the parent checker is optional outside the workspace.
- Do not initialize repositories, stage pre-existing work, start remote services, or change runtime behavior during documentation-only maintenance.
- Role instructions are not hard filesystem isolation. Report the actual boundary honestly.
