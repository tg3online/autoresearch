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
- Prefer small models for proposal generation and stronger models for review.
- Keep one active local decode or training job on a 16 GB machine.
- Cache immutable fixtures and hash them instead of repeatedly ingesting broad context.
- Stop after a fixed budget. Do not retry an unchanged failed experiment.
- Promote reusable winners into normal code, skills, prompts, or configs only through their owning project's review path.
