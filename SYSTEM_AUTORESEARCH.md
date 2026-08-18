# System Autoresearch Protocol

This repository is the proving ground for a reusable system-improvement loop:

> propose one bounded change → run a fixed benchmark → measure → keep or revert → preserve evidence

The MLX training loop is the first target. It is isolated from Hermes, AGON, Distro, and trading runtimes. No experiment receives production, publishing, capital, credential, or configuration authority.

## Promotion ladder

1. **Design-only:** benchmark, metric, protected files, and rollback are defined.
2. **Shadow:** trials run on copied inputs or historical fixtures. Outputs cannot reach production.
3. **Supervised:** a human reviews every accepted candidate and its evidence manifest.
4. **Bounded autonomous:** a fixed trial/time budget may run unattended. Promotion still requires review.
5. **Production candidate:** a separate change-management process may consume the winning artifact.

No target may skip a stage.

## Experiment contract

Every target must specify:

- exactly one mutable artifact or closed file allowlist;
- immutable benchmark inputs with hashes or durable source IDs;
- one primary metric and explicit direction (`minimize` or `maximize`);
- tie/noise threshold and repeat/seed policy;
- timeout, memory, disk, concurrency, and retry limits;
- deterministic preflight checks;
- revert procedure;
- evidence manifest containing source hashes, command, exit state, metrics, and logs;
- promotion authority distinct from the proposing agent.

A fluent explanation is not evidence. Missing metrics, changed benchmark inputs, dependency drift, timeout, or resource breach means reject.

## Current MLX target

Run a gated trial:

```bash
uv run python trial_gate.py \
  --description "baseline seed 42" \
  --seed 42
```

The runner enforces:

- one concurrent trial through an exclusive file lock;
- clean protected files (`prepare.py`, `pyproject.toml`, `uv.lock`);
- at least 1 GB estimated available memory and 2 GB free disk before launch;
- isolated subprocess group and hard timeout;
- 12 GB reported peak-memory ceiling;
- explicit seed, training budget, and evaluation token budget;
- source and log SHA-256 hashes in a JSON evidence manifest;
- rejection on crash, timeout, missing metrics, or memory breach.

Runtime artifacts are written under `.autoresearch/runs/` and remain local by default. Accepted summaries belong in `results.tsv`; selected baseline manifests may be copied into `evidence/` for GitHub review.

## Two-model coordinator

`research_team.py` (stdlib only) runs one bounded proposal through two models and
stops at the evidence. It never merges, promotes, commits, pushes, installs, or
edits `prepare.py`, `pyproject.toml`, `uv.lock`, or `trial_gate.py`.

```bash
uv run python research_team.py                     # dry run: both models, zero writes
uv run python research_team.py --apply             # + isolated worktree and one-line patch
uv run python research_team.py --apply --run-trial # + one gated trial in that worktree
```

**Stage 1 — proposer.** A local Qwen server on an OpenAI-compatible loopback
endpoint (`http://127.0.0.1:8088/v1/chat/completions`; non-loopback URLs are
refused) returns exactly one object: `{"parameter", "value", "rationale"}`.
Unknown or missing keys are rejected outright. If the local server requires a
bearer token, set `LOCAL_QWEN_API_KEY` in the environment (or point
`--qwen-api-key-env` at another variable) — the secret is never a command-line
argument, so it stays out of the process table and the JSON report.

**Stage 2 — reviewer.** The coordinator invokes `/opt/homebrew/bin/claude -p`
with `--tools ""`, `--strict-mcp-config --mcp-config '{"mcpServers":{}}'`, a
pinned `--json-schema`, and a scratch working directory. The reviewer sees only
the proposal text — no repo, no tools, no shell. It may only approve, reject, or
substitute one other allowlisted value, which is re-validated from scratch.

The parameter must be one of fourteen allowlisted `train.py` constants, and
every proposal is checked against types, exact bounds, and the relationships
`train.py` relies on: `TOTAL_BATCH_SIZE` divisible by `DEVICE_BATCH_SIZE ×
seq_len`, gradient accumulation ≤ 64 steps, `WARMUP_RATIO + WARMDOWN_RATIO ≤
1.0`, even `HEAD_DIM` with a positive `model_dim` multiple, and a coarse
peak-memory estimate under the coordinator's 8 GB ceiling (below the gate's 12 GB
hard limit). A no-op restatement of the current value is rejected — it is not an
experiment.

Escalation is explicit and one-way:

| Flag | Effect |
| --- | --- |
| *(none)* | Calls both models, writes nothing, creates nothing. |
| `--apply` | Requires a clean repo, creates `autoresearch/<run-id>` in a git worktree outside the repo, rewrites exactly one constant line, verifies only `train.py` changed as a 1-line replacement, and runs `py_compile`. |
| `--apply --run-trial` | Additionally runs the worktree's own `trial_gate.py` through the parent venv, then copies the manifest and log into `.autoresearch/coordinator/<run-id>/` as read-only evidence. |

`--run-trial` without `--apply` is an error. An exclusive lock allows one
coordinator at a time, and the trial stage also takes the parent trial lock so a
worktree run cannot race a direct `trial_gate.py` invocation. Every model call
and subprocess is bounded by a timeout.

Results are a candidate and nothing more. Promotion stays with a human under the
ladder above.

## Baseline and acceptance

A first full-budget result is **provisional**. Before autonomous batches, establish at least three fixed-seed runs and report median plus spread. A candidate must:

```bash
uv run python baseline_report.py --output evidence/m4-mini-baseline.json
```

1. pass every resource and integrity gate;
2. beat the baseline by more than the measured noise floor;
3. reproduce on at least two seeds;
4. add no dependency or protected-harness change;
5. remain simpler or justify added complexity;
6. receive independent review before promotion.

## Broader system targets

`benchmarks/registry.json` lists candidate targets. They are disabled until their fixtures and deterministic evaluators exist.

- **Hermes:** offline tool-task success, latency, tool-call count, and correction rate.
- **AGON:** blind judge calibration, citation validity, adversarial robustness, and latency.
- **Distro:** citation coverage, factual-error rate, human edit distance, and approval rate. Never publish from a trial.
- **SolanaTrader:** historical/shadow missed-runner recall and false-positive rate. Never score, size, sign, or trade from a trial.

## Efficiency doctrine

- Optimize the bottleneck that matters, not vanity metrics.
- Prefer small models for proposal generation and stronger models for review; a
  local proposer plus a tool-less reviewer keeps that split cheap and auditable.
- Keep one active local decode or training job on a 16 GB machine.
- Cache immutable fixtures and hash them instead of repeatedly ingesting broad context.
- Stop after a fixed budget. Do not retry an unchanged failed experiment.
- Promote reusable winners into normal code, skills, prompts, or configs only through their owning project's review path.
